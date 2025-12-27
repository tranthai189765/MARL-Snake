"""
Cấu hình cho NEAT Snake Training
"""

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
ENV_CONFIG = {
    "num_envs": 1,
    "num_snakes": 3,
    "height": 20,
    "width": 20,
    "snake_length": 5,
    "vision_range": 5,
    "reward_dict": {
        "fruit": 1.0,
        "time": 0.01,      # Reward nhỏ cho mỗi bước sống sót
        "lose": -1.0,
        "kill": -1.0,
        "win": 2.0,
    },
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    "episodes_per_eval": 1,
    "max_steps_per_episode": 500,
    "max_snakes_per_env": 2,  # Nhóm 2 genome/env
    "num_generations": 10,     # Số thế hệ training
}

# ============================================================================
# NEURAL NETWORK CONFIGURATION
# ============================================================================
NETWORK_CONFIG = {
    "input_size": 24,  # 8 hướng x 3 loại vật thể
}

# ============================================================================
# FILE PATHS
# ============================================================================
PATHS = {
    "config_neat": "config-neat-snake.ini",
    "winner_file": "winner_snake_genome.pkl",
    "checkpoint_prefix": "neat-checkpoint-",
}

# ============================================================================
# NEAT CONFIGURATION TEMPLATE
# ============================================================================
NEAT_CONFIG_TEMPLATE = """[NEAT]
fitness_criterion = max
fitness_threshold = 100.0
pop_size = 100
reset_on_extinction = False
no_fitness_termination = False

[DefaultGenome]
num_inputs = {input_size}
num_outputs = {output_size}
num_hidden = 0
num_layers = 1
initial_connection = full
activation_default = tanh
activation_mutate_rate = 0.0
activation_options = tanh

aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum

bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0

weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.3
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

enabled_default = True
enabled_mutate_rate = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 1

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
"""

# ============================================================================
# RENDER CONFIGURATION (for testing)
# ============================================================================
RENDER_CONFIG = {
    "window_name": "Snake NEAT Best",
    "delay": 0.1,  # Delay giữa các frame (seconds)
}