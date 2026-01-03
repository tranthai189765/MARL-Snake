import os
import numpy as np
import neat
import pickle
import math
import sys
import time
import random

# --- XỬ LÝ IMPORT ---
try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    from marlenv.marlenv.wrappers import make_snake, RenderGUI
except Exception:
    pass

# --- CẤU HÌNH ---
CONFIG_PATH = "config-neat-snake.ini"
WINNER_FILE = "winner_snake_genome.pkl"
MAX_SNAKES_PER_ENV = 2     # Train 1 con cho dễ hội tụ trước
EPISODES_PER_EVAL = 5      
MAX_STEPS_PER_EPISODE = 512
GENERATIONS = 50
# UPDATE: 24 (Ray) + 2 (Táo) + 4 (Hướng đầu) = 30 Inputs
INPUT_SIZE = 29           
STARVATION_LIMIT = 200     # Cho phép đói lâu hơn chút để đi tìm đường

ENV_KWARGS = dict(
    num_envs=1,
    num_snakes=MAX_SNAKES_PER_ENV, 
    height=20, # Map nhỏ lại chút để dễ học lúc đầu
    width=20,
    snake_length=5,
    vision_range=None,
    reward_dict= {
        'fruit': 10.0,
        'kill': 0.0,
        'lose': -30.0, # Phạt nhẹ thôi
        'win': 0.0,
        'time': -0.03,  # Bỏ phạt thời gian để tránh nó tự sát
    },
)

# --- FIX LỖI CONFIG NEAT ---
DEFAULT_NEAT_CONFIG = f"""\
[NEAT]
fitness_criterion      = max
fitness_threshold      = 1e9
no_fitness_termination = True
pop_size               = 350 
reset_on_extinction    = False

[DefaultGenome]
num_inputs             = {INPUT_SIZE}
num_outputs            = 3  
# (0: Up, 1: Down, 2: Left, 3: Right) - KHÔNG ĐƯỢC ĐỂ = 0
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

def ensure_neat_config():
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(DEFAULT_NEAT_CONFIG)
    print(f"--> Config updated: Inputs={INPUT_SIZE}, Outputs=3")

def get_head_and_fruit(obs_snake):
    head_pos = np.where(obs_snake[:, :, 5] == 1) 
    fruit_pos = np.where(obs_snake[:, :, 1] == 1) 
    head = (head_pos[0][0], head_pos[1][0]) if len(head_pos[0]) > 0 else None
    fruit = (fruit_pos[0][0], fruit_pos[1][0]) if len(fruit_pos[0]) > 0 else None
    return head, fruit

def obs_to_input_vector(obs_snake, last_action):
    H, W, C = obs_snake.shape
    head, fruit = get_head_and_fruit(obs_snake)
    if head is None: return np.zeros(INPUT_SIZE)
    
    center_y, center_x = head
    input_vector = []

    # 1. RAY CASTING (24 features)
    DIRECTIONS = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    for dy, dx in DIRECTIONS:
        wall_dist, food_dist, body_dist = 0, 0, 0
        dist = 0
        while True:
            dist += 1
            y, x = center_y + dy * dist, center_x + dx * dist
            if not (0 <= y < H and 0 <= x < W) or obs_snake[y, x, 0] == 1:
                if wall_dist == 0: wall_dist = dist
                break 
            if obs_snake[y, x, 1] == 1 and food_dist == 0: food_dist = dist
            is_body = (obs_snake[y, x, 6] == 1 or obs_snake[y, x, 2] == 1)
            if is_body and body_dist == 0: body_dist = dist

        input_vector.append(1.0 / wall_dist if wall_dist > 0 else 0)
        input_vector.append(1.0 / food_dist if food_dist > 0 else 0)
        input_vector.append(1.0 / body_dist if body_dist > 0 else 0)

    # 2. HƯỚNG TÁO (2 features)
    f_dy = (fruit[0] - center_y) / H if fruit else 0
    f_dx = (fruit[1] - center_x) / W if fruit else 0
    input_vector.extend([f_dy, f_dx])

    # 3. LAST ACTION (3 features - Sửa từ 4 về 3 để khớp 29 inputs)
    action_one_hot = [0.0] * 3
    if 0 <= last_action < 3:
        action_one_hot[last_action] = 1.0
    input_vector.extend(action_one_hot)

    return np.array(input_vector)

# 

def eval_group(genome_group, config):
    num_snakes = len(genome_group)
    kwargs = dict(ENV_KWARGS)
    kwargs["num_snakes"] = num_snakes
    
    env, _, _, _ = make_snake(**kwargs)
    nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genome_group]
    
    last_actions = [1] * num_snakes # 1 = Straight
    fitnesses = np.zeros(num_snakes)
    
    for _ in range(EPISODES_PER_EVAL):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        dones = [False] * num_snakes
        steps = 0
        visited_pos = [[] for _ in range(num_snakes)]
        steps_since_eat = [0] * num_snakes
        
        while not all(dones) and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            actions = []
            for i in range(num_snakes):
                if dones[i]:
                    actions.append(1)
                    continue
                
                inp = obs_to_input_vector(obs[i], last_actions[i])
                out = nets[i].activate(inp)
                suggested_action = int(np.argmax(out)) # Luôn là 0, 1, hoặc 2
                
                actions.append(suggested_action)
                last_actions[i] = suggested_action

            obs, rewards, new_dones, _ = env.step(actions)
            if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()
            
            for i in range(num_snakes):
                if dones[i]: continue
                fitnesses[i] += rewards[i]
                if rewards[i] > 1.0:
                    steps_since_eat[i] = 0
                    visited_pos[i] = []
                    fitnesses[i] += 5.0
                else:
                    steps_since_eat[i] += 1
                
                if steps_since_eat[i] > STARVATION_LIMIT:
                    new_dones[i] = True
                    fitnesses[i] -= 2.0

                head, fruit = get_head_and_fruit(obs[i])
                if head:
                    if head in visited_pos[i]: fitnesses[i] -= 0.2
                    else:
                        visited_pos[i].append(head)
                        if len(visited_pos[i]) > 15: visited_pos[i].pop(0)
                
                if new_dones[i]: dones[i] = True
                    
    env.close()
    for i, (gid, genome) in enumerate(genome_group):
        genome.fitness = fitnesses[i] / EPISODES_PER_EVAL

def run_training():
    ensure_neat_config()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.run(eval_genomes, n=GENERATIONS)

def eval_genomes(genomes, config):
    idx = 0
    while idx < len(genomes):
        group = genomes[idx : idx + MAX_SNAKES_PER_ENV]
        eval_group(group, config)
        idx += MAX_SNAKES_PER_ENV

if __name__ == "__main__":
    run_training()
