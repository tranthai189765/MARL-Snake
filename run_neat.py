"""
NEAT Snake Testing Script
File này dùng để TEST và VISUALIZE model đã train
Chạy: python test.py
"""
import os
import numpy as np
import neat
import pickle
import time

# Import config
from config import (
    ENV_CONFIG, 
    PATHS, 
    RENDER_CONFIG
)

# Cố gắng nhập Gymnasium trước, nếu không có thì dùng Gym cũ
try:
    import gymnasium as gym
    print("Sử dụng Gymnasium.")
except ImportError:
    import gym
    print("Sử dụng Gym.")

# Thư viện của bạn
from marlenv.marlenv.wrappers import make_snake, RenderGUI

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


def test_winner_gui(genome, config, num_snakes=None):
    """
    Chạy và visualize genome đã train với GUI
    
    Args:
        genome: NEAT genome đã train
        config: NEAT config
        num_snakes: Số lượng rắn (mặc định lấy từ ENV_CONFIG)
    """
    if num_snakes is None:
        num_snakes = ENV_CONFIG["num_snakes"]
    
    # Tạo environment
    kwargs = dict(ENV_CONFIG)
    kwargs["num_snakes"] = num_snakes
    env, obs_shape, action_shape, props = make_snake(**kwargs)
    env = RenderGUI(env, window_name=RENDER_CONFIG["window_name"])
    
    # Tạo neural network từ genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    done = [False] * props["num_snakes"]
    total_rewards = [0.0] * props["num_snakes"]
    steps = 0
    
    print("\n" + "=" * 70)
    print("KHỞI CHẠY GUI - XEM AGENT ĐÃ TRAIN")
    print("=" * 70)
    print(f"Số rắn: {props['num_snakes']}")
    print(f"Kích thước: {ENV_CONFIG['height']}x{ENV_CONFIG['width']}")
    print("Nhấn Ctrl+C để dừng...\n")
    
    try:
        while not all(done):
            env.render()
            actions = []
            
            for i in range(props["num_snakes"]):
                if done[i]:
                    actions.append(0)
                    continue
                
                # Chuyển đổi observation thành input vector
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
            
            # Cập nhật rewards
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            
            steps += 1
            
            # In thông tin
            print(f"Step {steps:4d} | Rewards: {[f'{r:6.2f}' for r in rewards]} | "
                  f"Done: {done}", end='\r')
            
            time.sleep(RENDER_CONFIG["delay"])
    
    except KeyboardInterrupt:
        print("\n\nDừng bởi người dùng (Ctrl+C)")
    
    finally:
        env.close()
        print("\n" + "=" * 70)
        print("KẾT THÚC GAME")
        print("=" * 70)
        print(f"Tổng bước: {steps}")
        print(f"Tổng rewards:")
        for i, total_r in enumerate(total_rewards):
            print(f"  Snake {i+1}: {total_r:.2f}")
        print()


def load_and_test_winner(num_snakes=None):
    """
    Load genome đã lưu và test với GUI
    
    Args:
        num_snakes: Số lượng rắn (mặc định lấy từ ENV_CONFIG)
    """
    config_path = PATHS["config_neat"]
    winner_file = PATHS["winner_file"]
    
    # Kiểm tra file tồn tại
    if not os.path.exists(winner_file):
        print(f"\n✗ Không tìm thấy file winner: {winner_file}")
        print(f"  Hãy chạy train.py trước để huấn luyện model!\n")
        return
    
    if not os.path.exists(config_path):
        print(f"\n✗ Không tìm thấy file config: {config_path}")
        print(f"  File config cần thiết để load model!\n")
        return
    
    try:
        # Load NEAT config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        
        # Load winner genome
        with open(winner_file, 'rb') as f:
            winner_genome = pickle.load(f)
        
        print(f"\n✓ Đã tải thành công winner genome từ {winner_file}")
        print(f"  Genome ID: {winner_genome.key}")
        print(f"  Fitness: {winner_genome.fitness if winner_genome.fitness else 'N/A'}")
        
        # Test với GUI
        test_winner_gui(winner_genome, config, num_snakes)
        
    except Exception as e:
        print(f"\n✗ Lỗi khi tải hoặc chạy winner: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Hàm main - load và test winner"""
    print("=" * 70)
    print("NEAT SNAKE - TEST MODE")
    print("=" * 70)
    
    # Bạn có thể thay đổi số lượng rắn ở đây
    # None = sử dụng giá trị mặc định từ ENV_CONFIG
    load_and_test_winner(num_snakes=None)


if __name__ == "__main__":
    main()