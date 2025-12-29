import os
import numpy as np
import neat
import pickle
import math
import sys
import time

# --- XỬ LÝ IMPORT THƯ VIỆN ---
try:
    import gymnasium as gym
    print("Sử dụng Gymnasium.")
except ImportError:
    import gym
    print("Sử dụng Gym.")

from marlenv.marlenv.wrappers import make_snake, RenderGUI

# Đăng ký môi trường marlenv
try:
    import importlib
    importlib.import_module('marlenv.marlenv.envs')
except Exception as e:
    print(f"Warning: could not import marlenv.marlenv.envs: {e}")

# --- CẤU HÌNH TOÀN CỤC ---
CONFIG_PATH = "config-neat-snake.ini"
WINNER_FILE = "winner_snake_genome.pkl"

# Cấu hình môi trường & Training
MAX_SNAKES_PER_ENV = 4      # Số lượng rắn trong 1 phòng (Training & Test)
EPISODES_PER_EVAL = 3       # QUAN TRỌNG: Chạy 5 ván lấy trung bình để giảm nhiễu
MAX_STEPS_PER_EPISODE = 128 # Tăng bước tối đa để rắn kịp ăn hết mồi
GENERATIONS = 50            # Số thế hệ training (tăng lên để kịp hội tụ)
INPUT_SIZE = 24             # 8 hướng x 3 loại vật thể

ENV_KWARGS = dict(
    num_envs=1,
    num_snakes=MAX_SNAKES_PER_ENV, 
    height=20,
    width=20,
    snake_length=5,
    vision_range=None,
    # REWARD SHAPING (Đã tối ưu)
    reward_dict= {
    'fruit': +5.0,     # Rất cao: Khuyến khích ăn mồi tối đa
    'kill': +0.0,      # Không thưởng: Không khuyến khích đánh nhau
    'lose': -10.0,     # Phạt nặng khi chết
    'win': +10.0,
    'time': -0.01,     # Phạt nhẹ: Để nó thong thả tìm mồi
    },
)

# Nội dung file config NEAT
DEFAULT_NEAT_CONFIG = f"""\
[NEAT]
fitness_criterion      = max
fitness_threshold      = 1e9
no_fitness_termination = True
pop_size               = 300
reset_on_extinction    = False

[DefaultGenome]
num_inputs             = {INPUT_SIZE}
num_outputs            = 0 
num_hidden             = 2
num_layers             = 1
initial_connection     = full_direct

feed_forward           = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
single_structural_mutation = false
structural_mutation_surer = default

activation_default     = tanh
activation_mutate_rate = 0.05
activation_options     = tanh relu sigmoid

aggregation_default    = sum
aggregation_mutate_rate = 0.0
aggregation_options    = sum

bias_init_type = gaussian
bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_max_value         = 3.0
bias_min_value         = -3.0
bias_mutate_power      = 0.5
bias_mutate_rate       = 0.7
bias_replace_rate      = 0.1

response_init_type = gaussian
response_init_mean     = 1.0
response_init_stdev    = 0.0
response_max_value     = 3.0
response_min_value     = -3.0
response_mutate_power  = 0.0
response_mutate_rate   = 0.0
response_replace_rate  = 0.0

weight_init_type = gaussian
weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_max_value       = 3.0
weight_min_value       = -3.0
weight_mutate_power    = 0.5
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.1

enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
enabled_default        = True
enabled_mutate_rate    = 0.01

# === Mutation probabilities (BẮT BUỘC) ===
conn_add_prob           = 0.4
conn_delete_prob        = 0.2
node_add_prob           = 0.2
node_delete_prob        = 0.05

[DefaultSpeciesSet]
# Tăng threshold vì input vector lớn (24) tạo ra khoảng cách gen lớn
compatibility_threshold = 3.0 

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
min_species_size = 1
elitism              = 2
survival_threshold   = 0.2
"""

# --- CÁC HÀM HỖ TRỢ ---

def ensure_neat_config(path: str = CONFIG_PATH):
    """Tạo hoặc cập nhật file config NEAT."""
    # Lấy số actions thực tế từ môi trường giả lập
    try:
        kwargs = dict(ENV_KWARGS)
        kwargs["num_snakes"] = 1
        env_temp, _, _, _ = make_snake(**kwargs)
        n_actions = env_temp.action_space.n
        env_temp.close()
    except:
        n_actions = 3 # Mặc định nếu lỗi

    # Luôn ghi đè để đảm bảo config mới nhất được áp dụng
    content = DEFAULT_NEAT_CONFIG.replace("num_outputs            = 0", f"num_outputs            = {n_actions}")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"--> Đã cập nhật file config: Inputs={INPUT_SIZE}, Outputs={n_actions}")

def normalize_dist(dist):
    """Hàm chuẩn hóa khoảng cách: Gần = 1, Xa = 0"""
    if dist == 0:
        return 0.0
    return 1.0 / dist

def obs_to_input_vector(obs_snake: np.ndarray):
    """
    Trích xuất 24 tính năng dựa trên 8 channels:
    0:Wall, 1:Fruit, 2:OtherHead, 3:OtherBody, 4:OtherTail, 5:MyHead, 6:MyBody, 7:MyTail
    """
    H, W, C = obs_snake.shape
    
    # --- TÌM VỊ TRÍ ĐẦU CỦA MÌNH (Channel 5) ---
    head_pos = np.where(obs_snake[:, :, 5] == 1)
    if len(head_pos[0]) > 0:
        center_y, center_x = head_pos[0][0], head_pos[1][0]
    else:
        # Nếu không thấy đầu mình (đã chết), trả về vector 0
        return np.zeros(24, dtype=np.float32)

    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),   # Up, Up-Right, Right, Down-Right
        (1, 0), (1, -1), (0, -1), (-1, -1)  # Down, Down-Left, Left, Up-Left
    ]
    
    input_vector = []
    max_scan_range = max(H, W)

    for dy, dx in DIRECTIONS:
        wall_dist = 0
        food_dist = 0
        obstacle_dist = 0
        
        for k in range(1, max_scan_range):
            y, x = center_y + dy * k, center_x + dx * k

            # 1. Check TƯỜNG (Channel 0 hoặc ra ngoài biên)
            if not (0 <= y < H and 0 <= x < W) or obs_snake[y, x, 0] == 1.0:
                if wall_dist == 0: wall_dist = k
                # Nếu gặp tường thì dừng tia này luôn vì không nhìn xuyên tường được
                break 

            # 2. Check FRUIT (Channel 1)
            if obs_snake[y, x, 1] == 1.0 and food_dist == 0:
                food_dist = k

            # 3. Check OBSTACLE (Thân mình, đuôi mình, và tất cả bộ phận của địch)
            # Channels: 2, 3, 4 (Other) và 6, 7 (My Body/Tail)
            is_obs = (obs_snake[y, x, 2] == 1.0 or obs_snake[y, x, 3] == 1.0 or 
                      obs_snake[y, x, 4] == 1.0 or obs_snake[y, x, 6] == 1.0 or 
                      obs_snake[y, x, 7] == 1.0)
            
            if is_obs and obstacle_dist == 0:
                obstacle_dist = k

        # Chuẩn hóa (1.0 là ngay sát cạnh, 0.0 là không thấy)
        input_vector.append(normalize_dist(wall_dist))
        input_vector.append(normalize_dist(food_dist))
        input_vector.append(normalize_dist(obstacle_dist))

    return np.array(input_vector, dtype=np.float32)

def eval_group(genome_group, config):
    """Đánh giá 1 nhóm rắn."""
    num_snakes = len(genome_group)
    
    # Tạo môi trường với số rắn đúng bằng số genome trong nhóm
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = num_snakes
    env, _, _, _ = make_snake(**kwargs)

    nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genome_group]
    fitness_acc = np.zeros(num_snakes, dtype=np.float32)

    # Chạy nhiều Episode để lấy trung bình (Tránh may mắn)
    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()
        # print("obs = ", obs.shape)
        if isinstance(obs, tuple): obs = obs[0]
            
        dones = [False] * num_snakes
        steps = 0
        epi_rewards = np.zeros(num_snakes, dtype=np.float32)

        while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            # print("steps = ", steps)
            actions = []

            for i in range(num_snakes):
                if dones[i]:
                    actions.append(0) # Action gì cũng được nếu đã chết
                    continue

                inp = obs_to_input_vector(obs[i])
                out = nets[i].activate(inp)
                
                # Chọn action
                act = int(np.argmax(out))
                if not 0 <= act < env.action_space.n:
                    act = 0 # Fallback an toàn
                
                actions.append(act)

            obs, rewards, new_dones, _ = env.step(actions)
            # print("new_dones = ", new_dones)
            
            # Xử lý format dones/rewards
            if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()
            
            # Cập nhật trạng thái chết
            for i in range(num_snakes):
                if dones[i]: continue # Đã chết từ trước
                dones[i] = new_dones[i]
                epi_rewards[i] += rewards[i]

        fitness_acc += epi_rewards

    env.close()

    # Tính Fitness trung bình
    avg_fitness = fitness_acc / float(EPISODES_PER_EVAL)

    for (gid, genome), fit in zip(genome_group, avg_fitness):
        genome.fitness = float(fit)

def eval_genomes(genomes, config):
    """Chia population thành các nhóm nhỏ để thi đấu."""
    idx = 0
    N = len(genomes)
    
    while idx < N:
        # Lấy nhóm tiếp theo
        group = genomes[idx : idx + MAX_SNAKES_PER_ENV]
        
        # Nếu nhóm cuối cùng quá ít người (ví dụ < 2), có thể skip hoặc chạy solo
        # Ở đây marlenv hỗ trợ chạy 1 rắn, nên ta vẫn eval bình thường.
        if len(group) > 0:
            eval_group(group, config)
            
        idx += MAX_SNAKES_PER_ENV

def run_training():
    ensure_neat_config(CONFIG_PATH)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    print(f"Population: {config.pop_size}")
    print(f"Generations: {GENERATIONS}")
    print(f"Snakes per Env: {MAX_SNAKES_PER_ENV}")
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))

    # Train
    winner = pop.run(eval_genomes, n=GENERATIONS)

    print("\n===== HUẤN LUYỆN HOÀN TẤT =====")
    print(f"Best fitness: {winner.fitness}")
    
    with open(WINNER_FILE, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Đã lưu winner tại: {WINNER_FILE}")

    # Test ngay sau khi train
    test_winner_gui(winner, config)

def test_winner_gui(genome, config):
    """Xem winner thi đấu với các bản sao của chính nó."""
    print("\n[GUI] Đang khởi động giả lập Replay...")
    
    kwargs = dict(ENV_KWARGS)
    # Test với đúng số lượng rắn như khi train
    kwargs["num_snakes"] = MAX_SNAKES_PER_ENV 
    
    env, _, _, props = make_snake(**kwargs)
    env = RenderGUI(env, window_name="Snake NEAT Champion Replay")
    
    # Tạo mạng cho winner
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Loop vô tận để xem
    try:
        while True:
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            dones = [False] * props["num_snakes"]
            total_rewards = np.zeros(props["num_snakes"])
            
            while not all(dones):
                env.render()
                time.sleep(0.05) # Làm chậm để dễ nhìn
                
                actions = []
                for i in range(props["num_snakes"]):
                    if dones[i]:
                        actions.append(0)
                        continue
                    
                    inp = obs_to_input_vector(obs[i])
                    out = net.activate(inp)
                    act = int(np.argmax(out))
                    actions.append(act)
                
                obs, rewards, new_dones, _ = env.step(actions)
                if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()
                
                for i in range(props["num_snakes"]):
                    if not dones[i]:
                        dones[i] = new_dones[i]
                        total_rewards[i] += rewards[i]
            
            print(f"Game Over. Rewards: {total_rewards}")
            time.sleep(1) # Nghỉ 1s trước ván mới
            
    except KeyboardInterrupt:
        print("\nĐã tắt GUI.")
    finally:
        env.close()

def load_and_play():
    if not os.path.exists(WINNER_FILE):
        print("Chưa có file winner. Hãy chạy train trước!")
        return
        
    ensure_neat_config(CONFIG_PATH)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
                         
    with open(WINNER_FILE, 'rb') as f:
        winner = pickle.load(f)
        
    test_winner_gui(winner, config)

if __name__ == "__main__":
    # Bỏ comment dòng dưới nếu muốn chỉ load file cũ
    # load_and_play() 
    
    # Mặc định là Train mới
    run_training()
