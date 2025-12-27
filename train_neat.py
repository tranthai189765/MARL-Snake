"""
NEAT Snake Training Script
File này dùng để HUẤN LUYỆN model
Chạy: python train.py
"""
import os
import numpy as np
import neat
import pickle
import re

# Import config
from config import (
    ENV_CONFIG, 
    TRAINING_CONFIG, 
    NETWORK_CONFIG, 
    PATHS, 
    NEAT_CONFIG_TEMPLATE
)

# Cố gắng nhập Gymnasium trước, nếu không có thì dùng Gym cũ
try:
    import gymnasium as gym
    print("Sử dụng Gymnasium.")
except ImportError:
    import gym
    print("Sử dụng Gym.")

# Thư viện của bạn
from marlenv.marlenv.wrappers import make_snake

# Ensure custom envs are registered
try:
    import importlib
    importlib.import_module('marlenv.marlenv.envs')
except Exception as e:
    print(f"Warning: could not import marlenv.marlenv.envs: {e}")


def obs_to_input_vector(obs_snake: np.ndarray):
    """
    Trích xuất 24 tính năng từ ma trận quan sát (vision_range x vision_range x C).
    Input: (11, 11, 8)
    Output: (24,)
    """
    H, W, C = obs_snake.shape
    center = H // 2

    # 8 Hướng: (dY, dX)
    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),      # Up, Up-Right, Right, Down-Right
        (1, 0), (1, -1), (0, -1), (-1, -1)     # Down, Down-Left, Left, Up-Left
    ]

    input_vector = []
    max_dist = center + 1  # Khoảng cách tối đa

    for dy, dx in DIRECTIONS:
        wall_dist = 0
        food_dist = 0
        obstacle_dist = 0

        for k in range(1, max_dist):
            y, x = center + dy * k, center + dx * k

            # 1. Kiểm tra Tường (Wall)
            if not (0 <= y < H and 0 <= x < W) or obs_snake[y, x, 0] == 1.0:
                if wall_dist == 0:
                    wall_dist = k
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
                return 0.0
            return 1.0 / dist

        # 3 tính năng cho mỗi hướng -> 24 inputs
        input_vector.append(normalize_dist(wall_dist))
        input_vector.append(normalize_dist(food_dist))
        input_vector.append(normalize_dist(obstacle_dist))

    return np.array(input_vector, dtype=np.float32)


def make_single_env():
    """Tạo env 1 rắn để suy ra kích thước action."""
    kwargs = dict(ENV_CONFIG)
    kwargs["num_snakes"] = 1
    env, _, action_shape, props = make_snake(**kwargs)
    return env, props


def ensure_neat_config():
    """Tạo hoặc cập nhật file config NEAT"""
    config_path = PATHS["config_neat"]
    
    # Lấy số lượng actions
    kwargs = dict(ENV_CONFIG)
    kwargs["num_snakes"] = 1
    try:
        env1, _, action_shape, props = make_snake(**kwargs)
        n_actions = env1.action_space.n
        env1.close()
    except Exception as e:
        print(f"Cảnh báo khi tạo env: {e}. Sử dụng num_outputs mặc định=3")
        n_actions = 3

    # Tạo nội dung config
    content = NEAT_CONFIG_TEMPLATE.format(
        input_size=NETWORK_CONFIG["input_size"],
        output_size=n_actions
    )

    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Đã tạo file config NEAT: {config_path}")
        print(f"Inputs={NETWORK_CONFIG['input_size']}, Outputs={n_actions}")
    else:
        # Cập nhật file cũ
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
            
            # Sử dụng regex để thay thế
            content = re.sub(r'num_inputs\s*=\s*\d+', 
                           f'num_inputs = {NETWORK_CONFIG["input_size"]}', 
                           old_content)
            content = re.sub(r'num_outputs\s*=\s*\d+', 
                           f'num_outputs = {n_actions}', 
                           content)
            content = re.sub(r'pop_size\s*=\s*\d+', 
                           'pop_size = 100', 
                           content)
            
            # Đảm bảo trường no_fitness_termination tồn tại
            if 'no_fitness_termination' not in content:
                content = re.sub(r'(\[NEAT\]\s*)', 
                               r'\1no_fitness_termination = False\n', 
                               content, count=1)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Đã cập nhật file config NEAT")
            print(f"Inputs={NETWORK_CONFIG['input_size']}, Outputs={n_actions}")
        except Exception as e:
            print(f"Cảnh báo: Không thể cập nhật file config. Lỗi: {e}")


def eval_group(genome_group, config):
    """Đánh giá một nhóm genome"""
    num_snakes = len(genome_group)
    kwargs = dict(ENV_CONFIG)
    kwargs["num_snakes"] = num_snakes
    
    env, obs_shape, action_shape, props = make_snake(**kwargs)
    
    nets = [
        neat.nn.FeedForwardNetwork.create(genome, config)
        for _, genome in genome_group
    ]
    
    fitness_acc = np.zeros(num_snakes, dtype=np.float32)
    
    for _ in range(TRAINING_CONFIG["episodes_per_eval"]):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        dones = [False] * num_snakes
        steps = 0
        epi_rewards = np.zeros(num_snakes, dtype=np.float32)
        
        while not all(dones) and steps < TRAINING_CONFIG["max_steps_per_episode"]:
            steps += 1
            actions = []
            
            for i in range(num_snakes):
                if dones[i]:
                    actions.append(0)
                    continue
                
                inp = obs_to_input_vector(obs[i])
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
    
    avg_fitness = fitness_acc / float(TRAINING_CONFIG["episodes_per_eval"])
    for (gid, genome), fit in zip(genome_group, avg_fitness):
        genome.fitness = float(fit)


def eval_genomes(genomes, config):
    """Chia population thành các nhóm để đánh giá."""
    idx = 0
    N = len(genomes)
    max_snakes = TRAINING_CONFIG["max_snakes_per_env"]
    
    while idx < N:
        group = genomes[idx: idx + max_snakes]
        eval_group(group, config)
        idx += max_snakes


def train():
    """Hàm chính để huấn luyện NEAT"""
    print("=" * 70)
    print("BẮT ĐẦU HUẤN LUYỆN NEAT SNAKE")
    print("=" * 70)
    
    # 1. Đảm bảo file config đúng
    ensure_neat_config()
    
    # 2. Tạo config từ file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        PATHS["config_neat"],
    )
    
    print(f"\nCấu hình NEAT:")
    print(f"  - Inputs: {config.genome_config.num_inputs}")
    print(f"  - Outputs: {config.genome_config.num_outputs}")
    print(f"  - Population: {config.pop_size}")
    print(f"  - Generations: {TRAINING_CONFIG['num_generations']}")
    
    # 3. Khởi tạo population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        5, 
        filename_prefix=PATHS["checkpoint_prefix"]
    ))
    
    # 4. Bắt đầu huấn luyện
    print("\n" + "=" * 70)
    print("BẮT ĐẦU QUÁ TRÌNH TIẾN HÓA...")
    print("=" * 70 + "\n")
    
    winner = pop.run(eval_genomes, n=TRAINING_CONFIG["num_generations"])
    
    # 5. Lưu winner
    print("\n" + "=" * 70)
    print("KẾT THÚC HUẤN LUYỆN")
    print("=" * 70)
    print("\nBEST GENOME:")
    print(winner)
    
    try:
        with open(PATHS["winner_file"], 'wb') as f:
            pickle.dump(winner, f)
        print(f"\n✓ Đã lưu winner genome tại: {PATHS['winner_file']}")
    except Exception as e:
        print(f"\n✗ Lỗi khi lưu winner: {e}")
    
    return winner, config


if __name__ == "__main__":
    train()