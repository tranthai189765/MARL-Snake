"""
Hybrid NEAT-DQN Training for MARL-Snake
======================================
Strategy:
1. Load pre-trained DQN checkpoint.
2. Freeze DQN weights (Conv + FC layers) to use as Feature Extractor.
3. Use NEAT to evolve the decision head (Input: 128 features from DQN -> Output: 3 Actions).
4. Save "Best Agent" as a combined checkpoint (DQN State + NEAT Genome).

IMPORTANT LOGIC CHANGE (as requested):
- The DQN fc3 head is converted into an equivalent NEAT genome and is treated as the INITIAL WINNER.
- This winner is saved immediately to RESULT_FILENAME.
- During NEAT evolution, if a genome achieves higher fitness, it overwrites the saved file.
- NO baseline_fitness field is stored.
- Rendering / inference code is moved to a SEPARATE script (not included here).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import neat
import pickle
from marlenv.marlenv.wrappers import make_snake

# ================= CONFIGURATION =================
CHECKPOINT_DQN_PATH = "checkpoints/shared_model_15500.pth"
NEAT_CONFIG_FILENAME = "config-neat-hybrid.ini"
RESULT_FILENAME = "hybrid_neat_best.pkl"

NUM_GENERATIONS = 50
NUM_SNAKES = 4
HEIGHT = 20
WIDTH = 20
SNAKE_LENGTH = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REWARD_DICT = {
    'fruit': 1.0,     # Rất cao: Khuyến khích ăn mồi tối đa
    'kill': 0.0,      # Không thưởng: Không khuyến khích đánh nhau
    'lose': 0.0,     # Phạt nặng khi chết
    'win': 0.0,
    'time': 0.0,     # Phạt nhẹ: Để nó thong thả tìm mồi
}
def save_checkpoint_safe(data, filename):
    """Lưu vào file tạm trước, sau đó rename để tránh lỗi corrupt khi tắt ngang."""
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, 'wb') as f:
            pickle.dump(data, f)
        # Lệnh này là nguyên tử (atomic) trên POSIX, và an toàn trên Windows (Python 3.3+)
        if os.path.exists(filename):
            os.remove(filename) # Cần thiết trên một số bản Windows cũ
        os.replace(temp_filename, filename)
        print(f"Saved checkpoint safe to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

# ================= DQN MODEL =================
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        num_envs, h, w, c = input_shape
       
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
       
        conv_out_size = h * w * 64
       
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
       
    def forward_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float()
        x = x / 255.0 if x.max() > 1.0 else x
       
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
       
        x = x.reshape(x.size(0), -1)
       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


# ================= FEATURE EXTRACTOR =================
class FeatureExtractor:
    def __init__(self, ckpt_path, obs_shape):
        self.model = DQN(obs_shape, 3).to(DEVICE)
        ckpt = torch.load(
            ckpt_path,
            map_location=DEVICE,
            weights_only=False   # <<< FIX PYTORCH 2.6+
        )
        state = ckpt['policy_net'] if 'policy_net' in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def embed(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(DEVICE)
            return self.model.forward_features(obs).cpu().numpy()


# ================= NEAT CONFIG =================
def create_neat_config():
    with open(NEAT_CONFIG_FILENAME, 'w') as f:
        f.write("""
[NEAT]
fitness_criterion = max
fitness_threshold = 1e9
pop_size = 100
reset_on_extinction = False
no_fitness_termination = False
[DefaultGenome]
# Node activation options
activation_default = relu
activation_mutate_rate = 0.1
activation_options = relu sigmoid tanh
# Node aggregation options
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
# Node bias options
bias_init_type = gaussian
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 3.0
bias_min_value = -3.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
single_structural_mutation = false
structural_mutation_surer = default
# Connection add/remove rates
conn_add_prob = 0.5
conn_delete_prob = 0.2
# Connection enable options
enabled_default = True
enabled_mutate_rate = 0.01
# Feedforward network options
feed_forward = True
# Node add/remove rates
node_add_prob = 0.2
node_delete_prob = 0.2
# Network parameters
num_inputs             = 128
num_outputs            = 3
num_hidden             = 0
num_layers             = 1
initial_connection     = full_direct

# Node response options
response_init_type = gaussian
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 3.0
response_min_value = -3.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
# Connection weight options
weight_init_type = gaussian
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 3
weight_min_value = -3
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
[DefaultSpeciesSet]
compatibility_threshold = 2.0
[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 1
[DefaultReproduction]
min_species_size = 3
elitism = 1
survival_threshold = 0.2
    """)


# ================= DQN → NEAT GENOME =================
def fc3_to_genome(fc3, config):
    """Convert DQN fc3 layer into a NEAT genome (linear head)."""
    genome = neat.DefaultGenome(0)
    genome.configure_new(config.genome_config)

    W = fc3.weight.detach().cpu().numpy()  # (3,128)
    b = fc3.bias.detach().cpu().numpy()    # (3,)

    for o in range(3):
        out_node = config.genome_config.output_keys[o]
        genome.nodes[out_node].bias = float(b[o])
        for i in range(128):
            in_node = config.genome_config.input_keys[i]
            key = (in_node, out_node)
            genome.connections[key].weight = float(W[o, i])

    return genome


# ================= FITNESS =================
extractor = None
env = None
best_fitness = -1e9


def eval_genomes(genomes, config):
    global best_fitness
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs = env.reset()
        done = [False] * NUM_SNAKES
        R = np.zeros(NUM_SNAKES)

        for _ in range(512):
            emb = extractor.embed(obs)
            acts = []
            for i in range(NUM_SNAKES):
                acts.append(0 if done[i] else int(np.argmax(net.activate(emb[i]))))
            obs, rew, done, _ = env.step(acts)
            R += rew
            if all(done): break

        genome.fitness = float(R.mean())

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            # with open(RESULT_FILENAME, 'wb') as f:
            #     pickle.dump({
            #         'dqn_state_dict': extractor.model.state_dict(),
            #         'neat_genome': genome,
            #         'neat_config': config
            #     }, f)
            # --- CODE MỚI (THÊM VÀO) ---
            save_data = {
                'dqn_state_dict': extractor.model.state_dict(),
                'neat_genome': genome,
                'neat_config': config
            }
            save_checkpoint_safe(save_data, RESULT_FILENAME)

# ================= TRAIN =================
def run_neat():
    global extractor, env, best_fitness

    env, obs_shape, _, _ = make_snake(
        num_envs=1, num_snakes=NUM_SNAKES,
        height=HEIGHT, width=WIDTH,
        snake_length=SNAKE_LENGTH,
        reward_dict={
            'fruit': +10.0, # Rất cao: Khuyến khích ăn mồi tối đa
            'kill': +0.0, # Không thưởng: Không khuyến khích đánh nhau
            'lose': -20.0, # Phạt nặng khi chết
            'win': +0.0,
            'time': -0.03, # Phạt nhẹ: Để nó thong thả tìm mồi
        }
    )
    obs_shape = env.observation_space.shape
    extractor = FeatureExtractor(CHECKPOINT_DQN_PATH, obs_shape)
    create_neat_config()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_FILENAME
    )

    pop = neat.Population(config)   # 🔥 QUAN TRỌNG

    # BÂY GIỜ innovation_tracker ĐÃ CÓ
    init_genome = fc3_to_genome(extractor.model.fc3, config)
    best_fitness = -1e9

    # with open(RESULT_FILENAME, 'wb') as f:
    #     pickle.dump({
    #         'dqn_state_dict': extractor.model.state_dict(),
    #         'neat_genome': init_genome,
    #         'neat_config': config
    #     }, f)

    save_data = {
        'dqn_state_dict': extractor.model.state_dict(),
        'neat_genome': init_genome,
        'neat_config': config
    }
    save_checkpoint_safe(save_data, RESULT_FILENAME)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.run(eval_genomes, NUM_GENERATIONS)

def render_winner(dqn_checkpoint, winner_pickle, neat_cfg_path=None, episodes=1, render=True, num_snakes=NUM_SNAKES, max_steps=256, sleep=0.03):
    """
    Load a saved hybrid checkpoint and render episodes.
    Handles both Single Snake (H,W,C) and Multi Snake (N,H,W,C) dimensions correctly.
    """
    import time
    from marlenv.marlenv.wrappers import RenderGUI

    if not os.path.exists(winner_pickle):
        raise FileNotFoundError("Winner pickle not found: " + str(winner_pickle))

    # --- 1. Load Data ---
    with open(winner_pickle, 'rb') as f:
        data = pickle.load(f)

    dqn_state = data.get('dqn_state_dict', None)
    neat_genome = data.get('neat_genome', None)
    neat_config_saved = data.get('neat_config', None)

    # --- 2. Setup Environment & Detect Shape ---
    # Tạo môi trường thực tế sẽ dùng để render
    arr = ["NEAT"] * 4
    env, _, _, _ = make_snake(num_envs=1, num_snakes=num_snakes, height=HEIGHT, width=WIDTH, snake_length=SNAKE_LENGTH, vision_range=None, reward_dict=REWARD_DICT)
    
    # Lấy mẫu observation đầu tiên để xác định dimension
    # Reset trả về (obs, info) hoặc obs tùy version, ta xử lý cả 2
    temp_obs = env.reset()
    if isinstance(temp_obs, (tuple, list)):
        temp_obs = temp_obs[0]
    
    # LOGIC QUAN TRỌNG: Chuẩn hóa shape cho DQN init
    # DQN class yêu cầu input_shape phải unpack được 4 giá trị (num_envs, h, w, c)
    if temp_obs.ndim == 3:
        # Trường hợp 1 snake: (H, W, C) -> Thêm batch dimension giả: (1, H, W, C)
        h, w, c = temp_obs.shape
        dqn_input_shape = (1, h, w, c)
        print(f"Detected Single Snake Env. Obs shape: {temp_obs.shape} -> DQN Input: {dqn_input_shape}")
    elif temp_obs.ndim == 4:
        # Trường hợp Multi snake: (N, H, W, C) -> Giữ nguyên
        dqn_input_shape = temp_obs.shape
        print(f"Detected Multi Snake Env. Obs/DQN Input: {dqn_input_shape}")
    else:
        raise ValueError(f"Unexpected observation shape: {temp_obs.shape}")

    # --- 3. Build Feature Extractor ---
    if dqn_state is None and dqn_checkpoint is None:
        raise ValueError('No DQN state or checkpoint provided to build feature extractor')

    # Khởi tạo model với shape đã chuẩn hóa (4 chiều)
    fe_model = DQN(dqn_input_shape, 3).to(DEVICE)

    # Load weights
    if dqn_checkpoint and os.path.exists(dqn_checkpoint):
        # Nếu load từ file gốc
        ckpt = torch.load(dqn_checkpoint, map_location=DEVICE, weights_only=False)
        state = ckpt['policy_net'] if 'policy_net' in ckpt else ckpt
        fe_model.load_state_dict(state)
    else:
        # Nếu load từ file pickle hybrid
        try:
            fe_model.load_state_dict(dqn_state)
        except Exception:
            # Fallback nếu key có prefix 'module.'
            new_state = {k.replace('module.', ''): v for k, v in dqn_state.items()}
            fe_model.load_state_dict(new_state)
    
    fe_model.eval()
    for p in fe_model.parameters():
        p.requires_grad = False

    # Wrapper xử lý dimension khi embed
    class _FE_wrapper:
        def __init__(self, model):
            self.model = model
        
        def embed(self, obs):
            # obs đầu vào có thể là (H,W,C) hoặc (N,H,W,C)
            # Chuyển sang numpy array nếu chưa phải
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            # Nếu là (H, W, C), thêm batch dim -> (1, H, W, C)
            if obs.ndim == 3:
                obs = np.expand_dims(obs, axis=0)
            
            with torch.no_grad():
                t = torch.from_numpy(obs).to(DEVICE).float()
                # Gọi forward_features (trả về 128) thay vì forward (trả về 3)
                return self.model.forward_features(t).cpu().numpy()

    feature_extractor = _FE_wrapper(fe_model)

    # --- 4. Prepare NEAT Network ---
    net = None
    if neat_genome is not None:
        cfg = neat_config_saved
        if cfg is None and neat_cfg_path and os.path.exists(neat_cfg_path):
            cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_cfg_path)
        
        if cfg:
            net = neat.nn.FeedForwardNetwork.create(neat_genome, cfg)
        else:
            raise ValueError("NEAT config missing.")

    # --- 5. Render Loop ---
    if render:
        try:
            env = RenderGUI(env,
            save_video=True,
            video_path="neat.mp4",
            fps=10)
        except Exception as e:
            print(f"Warning: Could not wrap with RenderGUI: {e}")

    # Khởi tạo list để lưu kết quả của tất cả episode
    all_episodes_mean_rewards = []
    all_episodes_mean_timelife = []

    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        dones = [False] * num_snakes
        ep_rews = [0.0] * num_snakes
        # Theo dõi số bước sống sót thực tế của từng con rắn
        snake_timelifes = [0] * num_snakes 
        step = 0

        while not all(dones) and step < max_steps:
            step += 1
            
            embeddings = feature_extractor.embed(obs)
            actions = []
            
            for i in range(num_snakes):
                if i >= len(dones) or dones[i]:
                    actions.append(0) # Hành động giả cho rắn đã chết
                    continue
                
                # Rắn còn sống thì tăng timelife
                snake_timelifes[i] += 1
                
                # NEAT Inference
                emb_input = embeddings[i] 
                out = net.activate(emb_input)
                actions.append(int(np.argmax(out)))

            # Render
            if render and hasattr(env, 'render'):
                try: env.render(agent_names=arr) 
                except: pass

            next_obs, rewards, new_dones, _ = env.step(actions)
            
            # Chuẩn hóa format trả về
            if isinstance(new_dones, np.ndarray): new_dones = new_dones.tolist()
            if isinstance(rewards, (int, float, np.float32)): rewards = [rewards]
            if isinstance(new_dones, bool): new_dones = [new_dones]

            for i in range(min(num_snakes, len(rewards))):
                if not dones[i]: # Chỉ cộng reward nếu rắn chưa chết ở step trước
                    ep_rews[i] += float(rewards[i])
                    if new_dones[i]:
                        dones[i] = True

            obs = next_obs
            time.sleep(sleep)

        # --- Tính toán log cho Episode hiện tại ---
        mean_reward_ep = np.mean(ep_rews)
        mean_timelife_ep = np.mean(snake_timelifes)
        
        all_episodes_mean_rewards.append(mean_reward_ep)
        all_episodes_mean_timelife.append(mean_timelife_ep)

        print(f"[Eval] Ep {ep+1}/{episodes} | "
              f"Mean Reward: {mean_reward_ep:.2f} | "
              f"Mean Timelife: {mean_timelife_ep:.1f} steps")

    # --- Tổng kết sau khi chạy xong N episodes ---
    if episodes > 0:
        final_mean_reward = np.mean(all_episodes_mean_rewards)
        final_mean_timelife = np.mean(all_episodes_mean_timelife)
        
        print("\n" + "="*50)
        print(f"FINAL EVALUATION OVER {episodes} EPISODES:")
        print(f"Overall Mean Reward: {final_mean_reward:.3f}")
        print(f"Overall Mean Timelife: {final_mean_timelife:.2f} steps")
        print("="*50)

    try:
        env.close()
    except:
        pass


if __name__ == '__main__':
    # run_neat()
    # after training, call `render_winner(CHECKPOINT_DQN_PATH, RESULT_FILENAME, NEAT_CONFIG_FILENAME, episodes=3, render=True)`
    render_winner(CHECKPOINT_DQN_PATH, RESULT_FILENAME, NEAT_CONFIG_FILENAME, episodes=1, render=True)
