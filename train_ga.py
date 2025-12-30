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

# Xử lý lỗi import marlenv trên Kaggle/Colab
try:
    from marlenv.marlenv.wrappers import make_snake, RenderGUI
    import importlib
    importlib.import_module('marlenv.marlenv.envs')
except Exception as e:
    print(f"Warning: could not import marlenv correctly: {e}")
    # Fallback cho trường hợp chạy local nếu cần
    pass

# --- CẤU HÌNH TOÀN CỤC ---
CONFIG_PATH = "config-neat-snake.ini"
WINNER_FILE = "winner_snake_genome.pkl"

# Cấu hình môi trường & Training
MAX_SNAKES_PER_ENV = 4      
EPISODES_PER_EVAL = 3       
MAX_STEPS_PER_EPISODE = 512 # Tăng bước tối đa lên chút
GENERATIONS = 50           # Cần train lâu hơn vì bài toán khó hơn
INPUT_SIZE = 26             # UPDATE: 24 cảm biến + 2 hướng táo (dx, dy)
STARVATION_LIMIT = 35       # UPDATE: Số bước tối đa cho phép nhịn đói

ENV_KWARGS = dict(
    num_envs=1,
    num_snakes=MAX_SNAKES_PER_ENV, 
    height=20,
    width=20,
    snake_length=5,
    vision_range=None,
    # REWARD SHAPING (Đã tối ưu để trị bệnh đi vòng tròn)
    reward_dict= {
        'fruit': +10.0,    # Thưởng lớn để kích thích ăn
        'kill': +2.0,      # Thưởng nhỏ nếu giết được đối thủ (tùy chọn)
        'lose': -5.0,      # Phạt vừa phải (đừng phạt quá nặng khiến nó sợ đi)
        'win': +20.0,
        'time': -0.1,      # Phạt thời gian nặng hơn để ép đi nhanh
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

# === Mutation probabilities ===
conn_add_prob           = 0.5
conn_delete_prob        = 0.2
node_add_prob           = 0.3
node_delete_prob        = 0.1

[DefaultSpeciesSet]
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
    try:
        kwargs = dict(ENV_KWARGS)
        kwargs["num_snakes"] = 1
        env_temp, _, _, _ = make_snake(**kwargs)
        n_actions = env_temp.action_space.n
        env_temp.close()
    except:
        n_actions = 3

    content = DEFAULT_NEAT_CONFIG.replace("num_outputs            = 0", f"num_outputs            = {n_actions}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"--> Đã cập nhật file config: Inputs={INPUT_SIZE}, Outputs={n_actions}")

def normalize_dist(dist):
    if dist == 0: return 0.0
    return 1.0 / dist

def obs_to_input_vector(obs_snake: np.ndarray):
    """
    Input: 26 features.
    - 24 features từ Ray-casting (8 hướng x (Tường, Mồi, Vật cản))
    - 2 features hướng tổng quát tới quả táo (dx, dy)
    """
    H, W, C = obs_snake.shape
    
    # 1. Tìm đầu rắn
    head_pos = np.where(obs_snake[:, :, 5] == 1)
    if len(head_pos[0]) > 0:
        center_y, center_x = head_pos[0][0], head_pos[1][0]
    else:
        return np.zeros(INPUT_SIZE, dtype=np.float32)

    input_vector = []

    # 2. Ray-casting (Quét xung quanh)
    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),   
        (1, 0), (1, -1), (0, -1), (-1, -1)  
    ]
    max_scan_range = max(H, W)

    for dy, dx in DIRECTIONS:
        wall_dist = 0
        food_dist = 0
        obstacle_dist = 0
        
        for k in range(1, max_scan_range):
            y, x = center_y + dy * k, center_x + dx * k

            # Check TƯỜNG
            if not (0 <= y < H and 0 <= x < W) or obs_snake[y, x, 0] == 1.0:
                if wall_dist == 0: wall_dist = k
                break 

            # Check FRUIT
            if obs_snake[y, x, 1] == 1.0 and food_dist == 0:
                food_dist = k

            # Check OBSTACLE (Thân mình + Địch)
            is_obs = (obs_snake[y, x, 2] == 1.0 or obs_snake[y, x, 3] == 1.0 or 
                      obs_snake[y, x, 4] == 1.0 or obs_snake[y, x, 6] == 1.0 or 
                      obs_snake[y, x, 7] == 1.0)
            if is_obs and obstacle_dist == 0:
                obstacle_dist = k

        input_vector.append(normalize_dist(wall_dist))
        input_vector.append(normalize_dist(food_dist))
        input_vector.append(normalize_dist(obstacle_dist))

    # 3. UPDATE: Thêm hướng tổng quát tới quả táo (Global Vision)
    # Giúp rắn biết táo ở đâu dù tia quét không trúng
    fruit_pos = np.where(obs_snake[:, :, 1] == 1)
    fruit_dx = 0.0
    fruit_dy = 0.0
    
    if len(fruit_pos[0]) > 0:
        fy, fx = fruit_pos[0][0], fruit_pos[1][0]
        # Chuẩn hóa về [-1, 1]
        fruit_dy = (fy - center_y) / float(H)
        fruit_dx = (fx - center_x) / float(W)
        
    input_vector.append(fruit_dy)
    input_vector.append(fruit_dx)

    return np.array(input_vector, dtype=np.float32)

# --- CẬP NHẬT HÀM HỖ TRỢ ĐỂ LẤY TỌA ĐỘ NHANH ---
def get_positions(obs_snake):
    """Trả về (y, x) của đầu rắn và quả táo."""
    head_pos = np.where(obs_snake[:, :, 5] == 1)
    fruit_pos = np.where(obs_snake[:, :, 1] == 1)
    
    head = None
    fruit = None
    
    if len(head_pos[0]) > 0:
        head = (head_pos[0][0], head_pos[1][0])
    if len(fruit_pos[0]) > 0:
        fruit = (fruit_pos[0][0], fruit_pos[1][0])
        
    return head, fruit

def calculate_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) # Manhattan distance tốt cho grid

# --- HÀM EVAL GROUP ĐÃ ĐƯỢC NÂNG CẤP ---
def eval_group(genome_group, config):
    num_snakes = len(genome_group)
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = num_snakes
    
    env, _, _, _ = make_snake(**kwargs)
    nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genome_group]
    fitness_acc = np.zeros(num_snakes, dtype=np.float32)

    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
            
        dones = [False] * num_snakes
        steps = 0
        epi_rewards = np.zeros(num_snakes, dtype=np.float32)
        starvation = np.zeros(num_snakes, dtype=int)
        
        # [NEW] Lưu lịch sử vị trí để phạt đi vòng tròn
        # visited_positions[i] là một set chứa các tọa độ (y, x) gần đây
        position_history = [[] for _ in range(num_snakes)] 

        # [NEW] Tính khoảng cách ban đầu
        prev_distances = [0] * num_snakes
        for i in range(num_snakes):
            head, fruit = get_positions(obs[i])
            if head and fruit:
                prev_distances[i] = calculate_distance(head, fruit)

        while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            actions = []

            # 1. Dự đoán hành động
            for i in range(num_snakes):
                if dones[i]:
                    actions.append(0)
                    continue

                inp = obs_to_input_vector(obs[i])
                out = nets[i].activate(inp)
                act = int(np.argmax(out))
                if not 0 <= act < env.action_space.n: act = 0
                actions.append(act)

            # 2. Bước đi
            obs, rewards, new_dones, _ = env.step(actions)
            if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()

            # 3. Xử lý logic Reward Shaping
            for i in range(num_snakes):
                if dones[i]: continue 
                
                # --- LOGIC MỚI: DISTANCE REWARD ---
                head, fruit = get_positions(obs[i])
                if head:
                    # Check loop: Nếu vị trí này đã đi qua trong 10 bước gần nhất -> Phạt
                    if head in position_history[i]:
                        rewards[i] -= 2.0 # Phạt nặng vì đi lại đường cũ (quay vòng)
                    
                    # Cập nhật lịch sử (chỉ giữ 15 bước gần nhất)
                    position_history[i].append(head)
                    if len(position_history[i]) > 15:
                        position_history[i].pop(0)

                    if fruit:
                        current_dist = calculate_distance(head, fruit)
                        # Nếu khoảng cách giảm (lại gần hơn) -> Thưởng nhẹ
                        if current_dist < prev_distances[i]:
                            rewards[i] += 1.0 
                        # Nếu khoảng cách tăng (đi xa ra) -> Phạt nhẹ
                        else:
                            rewards[i] -= 1.5 
                        
                        prev_distances[i] = current_dist
                # ----------------------------------

                # Logic Starvation cũ
                if rewards[i] > 5.0: # (Ngưỡng 5.0 vì ăn táo dc +10, cộng dồn distance reward)
                    starvation[i] = 0
                    position_history[i] = [] # Reset lịch sử khi ăn được (để nó ko sợ đi lại chỗ cũ)
                else:
                    starvation[i] += 1
                
                if starvation[i] > STARVATION_LIMIT:
                    new_dones[i] = True     
                    rewards[i] -= 10.0 # Phạt chết đói nặng hơn
                
                epi_rewards[i] += rewards[i]
                
                if new_dones[i]:
                    dones[i] = True

        fitness_acc += epi_rewards

    env.close()

    avg_fitness = fitness_acc / float(EPISODES_PER_EVAL)
    for (gid, genome), fit in zip(genome_group, avg_fitness):
        genome.fitness = float(fit)
        
def eval_genomes(genomes, config):
    idx = 0
    N = len(genomes)
    while idx < N:
        group = genomes[idx : idx + MAX_SNAKES_PER_ENV]
        if len(group) > 0:
            eval_group(group, config)
        idx += MAX_SNAKES_PER_ENV

def run_training():
    ensure_neat_config(CONFIG_PATH)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH
    )
    
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN (Fix Loop) ---")
    print(f"Population: {config.pop_size}")
    print(f"Inputs: {INPUT_SIZE} (Added Vision)")
    print(f"Starvation Limit: {STARVATION_LIMIT} steps")
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    
    # Train
    winner = pop.run(eval_genomes, n=GENERATIONS)

    print("\n===== HUẤN LUYỆN HOÀN TẤT =====")
    print(f"Best fitness: {winner.fitness}")
    
    with open(WINNER_FILE, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Đã lưu winner tại: {WINNER_FILE}")

    test_winner_gui(winner, config)

def test_winner_gui(genome, config):
    print("\n[GUI] Replay Winner (Cửa sổ GUI sẽ hiện lên)...")
    
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = MAX_SNAKES_PER_ENV 
    
    env, _, _, props = make_snake(**kwargs)
    try:
        env = RenderGUI(env, window_name="Snake AI Replay")
    except:
        print("Không thể bật GUI (có thể do môi trường headless).")
        env.close()
        return
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    try:
        while True:
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            dones = [False] * props["num_snakes"]
            starvation = np.zeros(props["num_snakes"], dtype=int)
            
            while not all(dones):
                env.render()
                time.sleep(0.05) 
                
                actions = []
                for i in range(props["num_snakes"]):
                    if dones[i]:
                        actions.append(0)
                        continue
                    
                    inp = obs_to_input_vector(obs[i])
                    out = net.activate(inp)
                    actions.append(int(np.argmax(out)))
                
                obs, rewards, new_dones, _ = env.step(actions)
                if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()
                
                for i in range(props["num_snakes"]):
                    if not dones[i]:
                        # Logic hiển thị starvation khi test
                        if rewards[i] > 1.0: starvation[i] = 0
                        else: starvation[i] += 1
                        
                        if starvation[i] > STARVATION_LIMIT:
                            new_dones[i] = True
                            print(f"Snake {i} chết đói!")

                        dones[i] = new_dones[i]
            
            print("Ván đấu kết thúc. Restarting...")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nĐã tắt GUI.")
    finally:
        env.close()

if __name__ == "__main__":
    run_training()
