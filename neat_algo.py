import os
import numpy as np
import neat
import pickle
import math 
import sys # Thêm sys để xử lý lỗi stdout

# Cố gắng nhập Gymnasium trước, nếu không có thì dùng Gym cũ (để giải quyết cảnh báo)
try:
    import gymnasium as gym
    print("Sử dụng Gymnasium.")
except ImportError:
    import gym
    print("Sử dụng Gym.")
    
# Thư viện của bạn
from marlenv.marlenv.wrappers import make_snake, RenderGUI
# Ensure custom envs are registered with gym/gymnasium (fix NameNotFound: Snake)
try:
    import importlib
    importlib.import_module('marlenv.marlenv.envs')
except Exception as e:
    print(f"Warning: could not import marlenv.marlenv.envs: {e}")

CONFIG_PATH = "config-neat-snake.ini"
WINNER_FILE = "winner_snake_genome.pkl"

# Các tham số env được tối ưu hóa reward
ENV_KWARGS = dict(
    num_envs=1,
    num_snakes=3, 
    height=20,
    width=20,
    snake_length=5,
    vision_range=5,
    # REWARD SHAPING: Tối ưu hóa reward để khuyến khích sinh tồn và ăn mồi
    reward_dict={
        "fruit": 1.0,
        "time": 0.01, # Reward nhỏ cho mỗi bước sống sót
        "lose": -1.0,
        "kill": -1.0,
        "win": 2.0,
    },
)

EPISODES_PER_EVAL = 1
MAX_STEPS_PER_EPISODE = 500
MAX_SNAKES_PER_ENV = 2 # Nhóm 2 genome/env
INPUT_SIZE = 24 # 8 hướng x 3 loại vật thể




DEFAULT_NEAT_CONFIG = f"""\
[NEAT]
fitness_criterion      = max
fitness_threshold      = 100.0
pop_size               = 100 # Tăng Pop Size để đa dạng hóa
reset_on_extinction    = False
no_fitness_termination = False

[DefaultGenome]
num_inputs             = {INPUT_SIZE} # Sẽ được ghi đè
num_outputs            = 0 
num_hidden             = 0
num_layers             = 1
initial_connection     = full

activation_default     = tanh
activation_mutate_rate = 0.0
activation_options     = tanh

aggregation_default    = sum
aggregation_mutate_rate = 0.0
aggregation_options    = sum

bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_max_value         = 30.0
bias_min_value         = -30.0
bias_mutate_power      = 0.5
bias_mutate_rate       = 0.7
bias_replace_rate      = 0.1

response_init_mean     = 1.0
response_init_stdev    = 0.0
response_max_value     = 30.0
response_min_value     = -30.0
response_mutate_power  = 0.0
response_mutate_rate   = 0.0
response_replace_rate  = 0.0

weight_init_mean       = 0.0
weight_init_stdev      = 1.0
weight_max_value       = 30.0
weight_min_value       = -30.0
weight_mutate_power    = 0.3 # Giảm để ổn định hơn
weight_mutate_rate     = 0.8
weight_replace_rate    = 0.1

enabled_default        = True
enabled_mutate_rate    = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 1

[DefaultReproduction]
elitism              = 2
survival_threshold   = 0.2
"""

def ensure_neat_config(path: str = CONFIG_PATH):
    if not os.path.exists(path):
        # Tạo file mới với nội dung đã được định dạng (INPUT_SIZE đã có trong DEFAULT_NEAT_CONFIG)
        # Khởi tạo env để lấy n_actions chính xác
        kwargs = dict(ENV_KWARGS)
        kwargs["num_snakes"] = 1
        env1, _, action_shape, props = make_snake(**kwargs)
        n_actions = env1.action_space.n 
        env1.close()
        
        content = DEFAULT_NEAT_CONFIG.replace("num_outputs = 0", f"num_outputs = {n_actions}")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Đã tạo file config NEAT mặc định: {path} với Inputs={INPUT_SIZE}, Outputs={n_actions}")
    else:
        # THÊM LOGIC ĐỂ GHI ĐÈ INPUT/OUTPUT KHI PHÁT HIỆN FILE CÓ SẴN
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Khởi tạo env để lấy n_actions chính xác (fallback nếu không tạo được env)
            kwargs = dict(ENV_KWARGS)
            kwargs["num_snakes"] = 1
            try:
                env1, _, action_shape, props = make_snake(**kwargs)
                n_actions = env1.action_space.n
                env1.close()
            except Exception as e:
                print(f"Cảnh báo khi tạo env để xác định num_outputs: {e}. Sử dụng num_outputs mặc định=3")
                n_actions = 3

            # Sử dụng regex đơn giản để thay thế các giá trị cũ
            import re
            content = re.sub(r'num_inputs\s*=\s*\d+', f'num_inputs = {INPUT_SIZE}', content)
            content = re.sub(r'num_outputs\s*=\s*\d+', f'num_outputs = {n_actions}', content)
            content = re.sub(r'pop_size\s*=\s*\d+', f'pop_size = 100', content) # Cập nhật pop_size

            # Đảm bảo trường no_fitness_termination tồn tại (một số file cũ thiếu)
            if 'no_fitness_termination' not in content:
                content = re.sub(r'(\[NEAT\]\s*)', r'\1no_fitness_termination = False\n', content, count=1)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Đã cập nhật file config NEAT: Inputs={INPUT_SIZE}, Outputs={n_actions}, Pop_Size=100")
        except Exception as e:
             print(f"Cảnh báo: Không thể cập nhật file config NEAT, sử dụng giá trị cũ. Lỗi: {e}")
        # END THÊM LOGIC CẬP NHẬT

def obs_to_input_vector(obs_snake: np.ndarray):
    """
    Trích xuất 24 tính năng từ ma trận quan sát (vision_range x vision_range x C).
    Input: (11, 11, 8)
    Output: (24,)
    """
    H, W, C = obs_snake.shape
    center = H // 2 # 5
    
    # 8 Hướng: (dY, dX)
    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1), # Up, Up-Right, Right, Down-Right
        (1, 0), (1, -1), (0, -1), (-1, -1) # Down, Down-Left, Left, Up-Left
    ]
    
    input_vector = []
    max_dist = center + 1 # Khoảng cách tối đa (6 ô)

    for dy, dx in DIRECTIONS:
        wall_dist = 0
        food_dist = 0
        obstacle_dist = 0
        
        for k in range(1, max_dist):
            y, x = center + dy * k, center + dx * k

            # 1. Kiểm tra Tường (Wall) - Nếu ra khỏi tầm nhìn (dùng vision_range)
            # Dù marlenv tự pad bằng 0/wall, ta vẫn check bound cho chắc
            if not (0 <= y < H and 0 <= x < W) or obs_snake[y, x, 0] == 1.0:
                 if wall_dist == 0:
                     wall_dist = k
                 # Quan trọng: Dừng tìm kiếm theo hướng này khi gặp vật cản cứng
                 break 
            
            # 2. Kiểm tra Food (Kênh 3)
            if obs_snake[y, x, 3] == 1.0 and food_dist == 0:
                food_dist = k

            # 3. Kiểm tra Obstacle (Kênh 2: Body, 4: Other Head, 5: Other Body)
            is_obstacle = (
                obs_snake[y, x, 2] == 1.0 or 
                obs_snake[y, x, 4] == 1.0 or 
                obs_snake[y, x, 5] == 1.0    
            )
            if is_obstacle and obstacle_dist == 0:
                obstacle_dist = k

        # Chuẩn hóa khoảng cách (1/dist)
        def normalize_dist(dist):
            if dist == 0:
                return 0.0 # Nếu không thấy, giá trị là 0
            # Ngược lại, 1.0/dist (gần hơn -> giá trị lớn hơn)
            return 1.0 / dist 

        # 3 tính năng cho mỗi hướng -> 24 inputs
        input_vector.append(normalize_dist(wall_dist))
        input_vector.append(normalize_dist(food_dist))
        input_vector.append(normalize_dist(obstacle_dist))

    return np.array(input_vector, dtype=np.float32)




def make_single_env():
    """Tạo env 1 rắn để suy ra kích thước action."""
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = 1
    env, _, action_shape, props = make_snake(**kwargs) 
    return env, props



def eval_group(genome_group, config):
    num_snakes = len(genome_group)

    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = num_snakes
    env, obs_shape, action_shape, props = make_snake(**kwargs)

    nets = [
        neat.nn.FeedForwardNetwork.create(genome, config)
        for _, genome in genome_group
    ]

    fitness_acc = np.zeros(num_snakes, dtype=np.float32)

    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()
        if isinstance(obs, tuple): # Xử lý nếu env trả về (obs, info)
            obs = obs[0] 
            
        dones = [False] * num_snakes
        steps = 0
        epi_rewards = np.zeros(num_snakes, dtype=np.float32)

        while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            actions = []

            for i in range(num_snakes):
                if dones[i]:
                    actions.append(0)
                    continue

                inp = obs_to_input_vector(obs[i]) # Dùng feature engineering
                out = nets[i].activate(inp)

                n_actions = env.action_space.n
                act = int(np.argmax(out))
                
                # Đảm bảo action hợp lệ
                if not 0 <= act < n_actions:
                    act = np.random.randint(n_actions) 

                actions.append(act)

            obs, rewards, dones, infos = env.step(actions)
            if isinstance(dones, np.ndarray):
                dones = dones.tolist()

            epi_rewards += np.array(rewards, dtype=np.float32)

        fitness_acc += epi_rewards

    env.close()

    avg_fitness = fitness_acc / float(EPISODES_PER_EVAL)

    for (gid, genome), fit in zip(genome_group, avg_fitness):
        # Thiết lập fitness dựa trên tổng reward
        genome.fitness = float(fit)


def eval_genomes(genomes, config):
    """Chia population thành các nhóm để đánh giá."""
    idx = 0
    N = len(genomes)
    while idx < N:
        group = genomes[idx: idx + MAX_SNAKES_PER_ENV]
        eval_group(group, config)
        idx += MAX_SNAKES_PER_ENV



def run():
    # 1. CẬP NHẬT VÀ ĐẢM BẢO FILE CONFIG ĐÚNG TRƯỚC KHI TẠO neat.Config
    # Hàm ensure_neat_config đã được chỉnh sửa để tự động làm điều này
    ensure_neat_config(CONFIG_PATH)

    # 2. TẠO CONFIG TỪ FILE ĐÃ ĐƯỢC CẬP NHẬT
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    
    # KHÔNG CẦN CẬP NHẬT LẠI TRỰC TIẾP config.genome_config.num_inputs/outputs nữa,
    # vì nó đã được cập nhật trong file config-neat-snake.ini.

    # 3. KHỞI TẠO POPULATION
    print(f"\n Đã cấu hình NEAT: Inputs={config.genome_config.num_inputs}, Outputs={config.genome_config.num_outputs}")
    
    pop = neat.Population(config)
    
    pop.add_reporter(neat.StdOutReporter(True)) 
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    pop.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoint-')) 

    # 5. BẮT ĐẦU HUẤN LUYỆN
    print("Bắt đầu huấn luyện...")
    winner = pop.run(eval_genomes, n=10) 

    print("\n===== KẾT THÚC HUẤN LUYỆN =====")
    print("BEST GENOME:")
    print(winner)

    # LƯU WINNER VÀO FILE
    try:
        with open(WINNER_FILE, 'wb') as f:
            pickle.dump(winner, f)
        print(f"\nĐã lưu winner genome tại: {WINNER_FILE}")
    except Exception as e:
        print(f"\ Lỗi khi lưu winner: {e}")

    # Xem thử best genome với GUI
    test_winner_gui(winner, config, num_snakes=ENV_KWARGS["num_snakes"])



def test_winner_gui(genome, config, num_snakes=2): # Cập nhật num_snakes mặc định
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = num_snakes
    env, obs_shape, action_shape, props = make_snake(**kwargs)

    env = RenderGUI(env, window_name="Snake NEAT Best")

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
        
    done = [False] * props["num_snakes"]

    import time
    print("\nKhởi chạy GUI để xem winner...")
    
    while not all(done):
        env.render()

        actions = []
        for i in range(props["num_snakes"]):
            if done[i]:
                actions.append(0)
                continue
            
            # Dùng feature engineering
            inp = obs_to_input_vector(obs[i])
            out = net.activate(inp)
            
            n_actions = env.action_space.n
            act = int(np.argmax(out))
            
            # Đảm bảo action hợp lệ
            if not 0 <= act < n_actions:
                act = np.random.randint(n_actions)

            actions.append(act)

        obs, rewards, done, infos = env.step(actions)
        if isinstance(done, np.ndarray):
            done = done.tolist()
            
        print("rewards = ", rewards, "done = ", done)
        time.sleep(0.1)

    env.close()


def load_and_test_winner(config_path=CONFIG_PATH, winner_file=WINNER_FILE):
    if not os.path.exists(winner_file):
        print(f"Không tìm thấy file winner đã lưu: {winner_file}. Hãy huấn luyện trước.")
        return

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        
        with open(winner_file, 'rb') as f:
            winner_genome = pickle.load(f)
            
        print(f"Đã tải thành công winner genome từ {winner_file}")
        test_winner_gui(winner_genome, config, num_snakes=ENV_KWARGS["num_snakes"])
        
    except Exception as e:
        print(f"Lỗi khi tải hoặc chạy winner: {e}")


if __name__ == "__main__":
    run()
    load_and_test_winner()