
import os
import numpy as np
import neat as neat_algo

from wrappers import make_snake  # import đúng file wrappers.py của bạn


# ---------- THAM SỐ CƠ BẢN ----------

CONFIG_PATH = "config-neat-snake.ini"

# ID env đã đăng ký với gym, wrappers.make_snake mặc định dùng "Snake-v1"
# Nếu bạn đăng ký GraphSnake:
#   - GraphSnakeEnv  -> env_id="GraphSnake-v1"
ENV_ID = "Snake-v1"

# Số trận dùng để ước lượng fitness mỗi group genome
EPISODES_PER_EVAL = 1

# Giới hạn bước tối đa mỗi episode
MAX_STEPS_PER_EPISODE = 500

# Số rắn tối đa trong 1 env (tức là số genome trong 1 group cạnh tranh)
MAX_SNAKES_PER_ENV = 4


# ---------- HÀM TIỆN ÍCH ----------

def obs_to_input_vector(obs_snake):
    """
    obs_snake: (H, W, C) của 1 con rắn (đã qua wrapper).
    Trả về vector 1D float32 cho mạng NEAT.
    """
    return obs_snake.astype(np.float32).flatten()


def make_single_env():
    """
    Tạo env với 1 rắn để suy ra kích thước observation cho 1 agent.
    """
    env, obs_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=1,
        env_id=ENV_ID,
    )
    return env, obs_shape, action_shape, props


# ---------- ĐÁNH GIÁ 1 GROUP GENOME (CẠNH TRANH) ----------

def eval_group(genome_group, config):
    """
    Đánh giá 1 nhóm genome cùng lúc trong 1 env multi-snake.

    - len(genome_group) = num_snakes trong env.
    - Mỗi genome điều khiển 1 con rắn.
    - Fitness = tổng reward riêng của con rắn đó qua EPISODES_PER_EVAL episode.
    """
    num_snakes = len(genome_group)

    # Tạo env với num_snakes rắn, single-process (num_envs=1)
    env, obs_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=num_snakes,
        env_id=ENV_ID,
    )

    # Tạo mạng nơ-ron NEAT cho từng genome
    nets = [
        neat_algo.nn.FeedForwardNetwork.create(genome, config)
        for _, genome in genome_group
    ]

    # Fitness tích lũy cho mỗi rắn
    fitness_acc = np.zeros(num_snakes, dtype=np.float32)

    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()  # shape: (num_snakes, H, W, C) hoặc (num_snakes, ..., C)
        dones = [False] * num_snakes
        steps = 0
        epi_rewards = np.zeros(num_snakes, dtype=np.float32)

        while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            actions = []

            for i in range(num_snakes):
                if dones[i]:
                    # Rắn đã chết: luôn noop
                    actions.append(0)
                    continue

                inp = obs_to_input_vector(obs[i])
                out = nets[i].activate(inp)

                # Số action của env (thường là 3 với observer='snake')
                n_actions = env.action_space.n
                act = int(np.argmax(out))
                if act < 0:
                    act = 0
                if act >= n_actions:
                    act = n_actions - 1

                actions.append(act)

            obs, rewards, dones, info = env.step(actions)
            epi_rewards += np.array(rewards, dtype=np.float32)

        fitness_acc += epi_rewards

    env.close()

    # Fitness trung bình trên EPISODES_PER_EVAL
    avg_fitness = fitness_acc / float(EPISODES_PER_EVAL)

    for (genome_id, genome), fit in zip(genome_group, avg_fitness):
        genome.fitness = float(fit)


# ---------- HÀM EVAL CHO CẢ POPULATION ----------

def eval_genomes(genomes, config):
    """
    Hàm callback cho NEAT.

    - Chia population thành nhiều group, mỗi group tối đa MAX_SNAKES_PER_ENV genome.
    - Mỗi group chạy 1 env multi-snake, cạnh tranh với nhau.
    """
    idx = 0
    N = len(genomes)
    while idx < N:
        group = genomes[idx: idx + MAX_SNAKES_PER_ENV]
        eval_group(group, config)
        idx += MAX_SNAKES_PER_ENV


# ---------- HÀM CHÍNH: TẠO POPULATION VÀ CHẠY NEAT ----------

def run():
    # Load config NEAT
    config = neat_algo.Config(
        neat_algo.DefaultGenome,
        neat_algo.DefaultReproduction,
        neat_algo.DefaultSpeciesSet,
        neat_algo.DefaultStagnation,
        CONFIG_PATH,
    )

    # Dùng env 1-rắn để suy ra kích thước obs 1 agent
    env1, obs_shape, action_shape, props = make_single_env()
    in_size = int(np.prod(obs_shape))      # H*W*C (hoặc 5*C nếu GraphSnake)
    n_actions = env1.action_space.n        # số action cho 1 rắn
    env1.close()

    # Cấu hình lại genome để khớp với env
    config.genome_config.num_inputs = in_size
    config.genome_config.num_outputs = n_actions
    config.genome_config.add_bias = True

    # Tạo population
    pop = neat_algo.Population(config)

    # Reporter: in log ra console
    pop.add_reporter(neat_algo.StdOutReporter(True))
    stats = neat_algo.StatisticsReporter()
    pop.add_reporter(stats)

    # Chạy NEAT trong 50 thế hệ (có thể chỉnh)
    winner = pop.run(eval_genomes, n=50)

    print("\n===== BEST GENOME =====")
    print(winner)

    # Test genome thắng trong môi trường multi-snake (self-play)
    test_winner_multi(winner, config, num_snakes=min(MAX_SNAKES_PER_ENV, 4))


# ---------- TEST GENOME THẮNG ----------

def test_winner_multi(genome, config, num_snakes=4, render_ascii=False):
    """
    Test genome thắng trong 1 env multi-snake.

    Đơn giản: dùng cùng 1 genome điều khiển tất cả rắn (self-play).
    Nếu muốn test nhiều genome khác nhau, bạn sửa lại cho phù hợp.
    """
    env, obs_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=num_snakes,
        env_id=ENV_ID,
    )

    net = neat_algo.nn.FeedForwardNetwork.create(genome, config)

    obs = env.reset()
    dones = [False] * num_snakes
    steps = 0
    total_rewards = np.zeros(num_snakes, dtype=np.float32)

    while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
        steps += 1
        actions = []
        for i in range(num_snakes):
            if dones[i]:
                actions.append(0)
                continue
            inp = obs_to_input_vector(obs[i])
            out = net.activate(inp)
            n_actions = env.action_space.n
            act = int(np.argmax(out))
            if act < 0:
                act = 0
            if act >= n_actions:
                act = n_actions - 1
            actions.append(act)

        obs, rewards, dones, info = env.step(actions)
        total_rewards += np.array(rewards, dtype=np.float32)

        if render_ascii:
            env.render(mode="ascii")
            print("Step:", steps, "Rewards:", rewards)
            print("-" * 20)

    print("Total rewards (per snake):", total_rewards)
    env.close()


if __name__ == "__main__":
    run()
