"""
Deep Q-Learning (Parameter Sharing) cho MARL-Snake
==================================================
Shared Model: Tất cả agent dùng chung một mạng DQN và chung Replay Buffer.
Logging: TensorBoard (runs_dqn)
Observation: image grid (H, W, C)
Action: 0 (thẳng), 1 (trái), 2 (phải)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import random
import os
import time
import shutil
from datetime import datetime
import neat
from marlenv.marlenv.wrappers import make_snake, RenderGUI
import pickle

# ========================== CONFIG ==========================
class Config:
    # Environment
    NUM_SNAKES = 4
    HEIGHT = 20
    WIDTH = 20
    SNAKE_LENGTH = 5
    VISION_RANGE = None
   
    # Training
    NUM_EPISODES = 50000
    MAX_STEPS_PER_EPISODE = 256
    BATCH_SIZE = 512
    GAMMA = 0.99
    LR = 5e-4
   
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995
   
    # Replay Buffer
    BUFFER_SIZE = 10000            
    MIN_BUFFER_SIZE = BATCH_SIZE * 3
   
    # Target Network
    TARGET_UPDATE_FREQ = 100
   
    # Reward shaping
    REWARD_PRESET = "default"
    EARLY_DEATH_THRESHOLD = 10
    EARLY_DEATH_PENALTY = -1.0
   
    REWARD_DICT = {
    'fruit': 1.0,     # Rất cao: Khuyến khích ăn mồi tối đa
    'kill': 0.0,      # Không thưởng: Không khuyến khích đánh nhau
    'lose': 0.0,     # Phạt nặng khi chết
    'win': 0.0,
    'time': 0.0,     # Phạt nhẹ: Để nó thong thả tìm mồi
}
   
    REWARD_DICT_LATE = {
     'fruit': 1.0,     # Rất cao: Khuyến khích ăn mồi tối đa
     'kill': 0.0,      # Không thưởng: Không khuyến khích đánh nhau
     'lose': 0.0,     # Phạt nặng khi chết
     'win': 0.0,
     'time': 0.0,     # Phạt nhẹ: Để nó thong thả tìm mồi
 }
   
    # Checkpoints & Logs
    SAVE_FREQ = 500        
    SAVE_BEST_ONLY = True    
    KEEP_LAST_N = 3        
    SAVE_DIR = "checkpoints"
    LOG_DIR = "runs_dqn"  # Thư mục cho Tensorboard
    RESUME_FROM = None      
   
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# ========================== REPLAY BUFFER ==========================
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
   
    def push(self, *args):
        self.buffer.append(Transition(*args))
   
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
   
    def __len__(self):
        return len(self.buffer)
 
 
# ========================== DQN NETWORK ==========================
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
       
    def forward(self, x):
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
        x = self.fc3(x)
        return x
 
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
# ========================== SHARED AGENT LOGIC ==========================
class SharedAgent:
    """
    Agent không sở hữu mạng riêng, chỉ tham chiếu đến mạng chung.
    """
    def __init__(self, agent_id, num_actions):
        self.agent_id = agent_id
        self.num_actions = num_actions
        # Các thông số thống kê riêng
        self.episode_reward = 0
       
    def select_action(self, state, policy_net, epsilon, device, training=True):
        if training and random.random() < epsilon:
            return random.randrange(self.num_actions)
       
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            # Add batch dimension inside DQN forward if needed, or here
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()
 
 
# ========================== TRAINER (SHARED MODEL) ==========================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.start_episode = 1
       
        # Reward setup
        reward_dict = config.REWARD_DICT_LATE if config.REWARD_PRESET == "late_training" else config.REWARD_DICT
       
        # Env setup
        print("dang tao...")
        self.env, self.obs_shape, self.action_shape, self.properties = make_snake(
            num_envs=1,
            num_snakes=config.NUM_SNAKES,
            height=config.HEIGHT,
            width=config.WIDTH,
            snake_length=config.SNAKE_LENGTH,
            vision_range=config.VISION_RANGE,
            reward_dict=reward_dict
        )
       
        print("tao xong...")
        self.num_actions = 3
        self.agent_obs_shape = self.env.observation_space.shape
        print("self.env.observation_space.shape = ", self.env.observation_space.shape)
       
        # --- SHARED BRAIN ---
        # Khởi tạo MỘT mạng duy nhất cho tất cả agents
        self.policy_net = DQN(self.agent_obs_shape, self.num_actions).to(config.DEVICE)
        self.target_net = DQN(self.agent_obs_shape, self.num_actions).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
       
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        # Shared Buffer: Tất cả agent đẩy kinh nghiệm vào đây
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
       
        # Global Epsilon
        self.epsilon = config.EPSILON_START
       
        # Agents (Logic holders)
        self.agents = [SharedAgent(i, self.num_actions) for i in range(config.NUM_SNAKES)]
       
        # Logging & TensorBoard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(config.LOG_DIR, current_time)
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
       
        self.best_mean_reward = float('-inf')
        os.makedirs(config.SAVE_DIR, exist_ok=True)
       
    def update_model(self):
        """Update shared model using shared buffer"""
        if len(self.memory) < self.config.MIN_BUFFER_SIZE:
            return None
       
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
       
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.config.DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.config.DEVICE)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.config.DEVICE)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.config.DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.config.DEVICE)
       
        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
       
        # Max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.config.GAMMA * next_q_values
       
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)
       
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
       
        return loss.item()
 
    def train(self):
        print(f"Start training with SHARED MODEL on {self.config.DEVICE}...")
       
        if self.config.RESUME_FROM:
            self.load_checkpoint(self.config.RESUME_FROM)
            self.start_episode = self.config.RESUME_FROM + 1
 
        checkpoint_history = []
        global_step = 0 # Step tổng để vẽ biểu đồ loss mịn hơn
       
        for episode in range(self.start_episode, self.config.NUM_EPISODES + 1):
            obs = self.env.reset()
            dones = [False] * self.config.NUM_SNAKES
            ep_rewards = [0.0] * self.config.NUM_SNAKES
            step = 0
           
            episode_loss = []
           
            while not all(dones) and step < self.config.MAX_STEPS_PER_EPISODE:
                actions = []
                # Lấy action cho từng agent bằng mạng chung
                for i, agent in enumerate(self.agents):
                    if dones[i]:
                        actions.append(0)
                    else:
                        act = agent.select_action(obs[i], self.policy_net, self.epsilon, self.config.DEVICE)
                        actions.append(act)
               
                next_obs, rewards, next_dones, info = self.env.step(actions)
               
                # Store transitions
                for i in range(self.config.NUM_SNAKES):
                    if not dones[i]: # Nếu agent chưa chết trước đó
                        r = rewards[i]
                        # Penalty chết sớm
                        if next_dones[i] and step < self.config.EARLY_DEATH_THRESHOLD:
                            r += self.config.EARLY_DEATH_PENALTY
                       
                        self.memory.push(obs[i], actions[i], r, next_obs[i], next_dones[i])
                        ep_rewards[i] += r
               
                # Training step (Update model chung)
                loss = self.update_model()
                if loss is not None:
                    episode_loss.append(loss)
                    global_step += 1
               
                obs = next_obs
                dones = next_dones
                step += 1
           
            # --- End of Episode ---
           
            # Decay epsilon
            self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
           
            # Update target net
            if episode % self.config.TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
           
            # Metrics
            avg_ep_reward = np.mean(ep_rewards)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
           
            # --- TENSORBOARD LOGGING ---
            self.writer.add_scalar('Train/Mean_Reward', avg_ep_reward, episode)
            self.writer.add_scalar('Train/Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Train/Episode_Length', step, episode)
            if avg_loss > 0:
                self.writer.add_scalar('Train/Loss', avg_loss, episode)
           
            # Console Log
            if episode % 10 == 0:
                print(f"Ep {episode:5d} | Mean Reward: {avg_ep_reward:6.2f} | "
                      f"Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f} | Steps: {step}")
           
            # Save Checkpoint
            if self.config.SAVE_BEST_ONLY and episode >= 50:
                # Logic lưu best model dựa trên mean reward của 100 ep gần nhất
                # (Ở đây làm đơn giản là so sánh reward ep hiện tại cho code gọn,
                # thực tế nên dùng moving average)
                if avg_ep_reward > self.best_mean_reward:
                    self.best_mean_reward = avg_ep_reward
                    self.save_checkpoint("best")
                    print(f"  [BEST] New best mean reward: {avg_ep_reward:.2f}")
 
            if self.config.SAVE_FREQ and episode % self.config.SAVE_FREQ == 0:
                self.save_checkpoint(episode)
                checkpoint_history.append(episode)
                if len(checkpoint_history) > self.config.KEEP_LAST_N:
                    self.delete_checkpoint(checkpoint_history.pop(0))
 
        # Finish
        self.save_checkpoint("final")
        self.writer.close()
        print("Training Complete!")
 
    def save_checkpoint(self, tag):
        path = os.path.join(self.config.SAVE_DIR, f"shared_model_{tag}.pth")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_mean_reward': self.best_mean_reward
        }, path)
 
    def load_checkpoint(self, tag):
        path = os.path.join(self.config.SAVE_DIR, f"shared_model_{tag}.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.DEVICE, weights_only=False)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epsilon = ckpt['epsilon']
            if 'best_mean_reward' in ckpt:
                self.best_mean_reward = ckpt['best_mean_reward']
            print(f"Loaded checkpoint: {path}")
        else:
            print(f"Checkpoint not found: {path}")
 
    def delete_checkpoint(self, tag):
        path = os.path.join(self.config.SAVE_DIR, f"shared_model_{tag}.pth")
        if os.path.exists(path):
            os.remove(path)

# ========================== HELPER FUNCTIONS FOR EVALUATE ==========================
# Định nghĩa kênh giống greedy.py để dùng cho việc mask
class DQN_Evaluator:
    def __init__(self, config, model_class, checkpoint_tag="best"):
        self.config = config
        self.device = config.DEVICE
        
        # ================= DEFINITIONS =================
        self.CH_WALL = 0
        self.CH_FRUIT = 1
        self.CH_OTHER_HEAD = 2
        self.CH_OTHER_BODY = 3
        self.CH_OTHER_TAIL = 4
        self.CH_MY_HEAD = 5
        self.CH_MY_BODY = 6
        self.CH_MY_TAIL = 7

        self.DEADLY_CHANNELS = [
            self.CH_WALL,
            self.CH_OTHER_HEAD,
            self.CH_OTHER_BODY,
            self.CH_OTHER_TAIL,
            self.CH_MY_BODY,
            self.CH_MY_TAIL
        ]

        # ================= LOAD MODEL =================
        # Giả sử obs_shape chuẩn là (H, W, 8) hoặc (8, H, W) tùy config, 
        # ở đây ta lấy theo env giả lập hoặc truyền vào. 
        # Để an toàn, ta khởi tạo model khi bắt đầu eval hoặc truyền shape vào init.
        # Ở đây tôi khởi tạo placeholder, model thực tế sẽ load trong evaluate 
        # để đảm bảo đúng shape từ env, hoặc bạn có thể fix cứng shape.
        self.model = None 
        self.model_class = model_class
        self.checkpoint_path = os.path.join(config.SAVE_DIR, f"shared_model_{checkpoint_tag}.pth")

    def _load_model(self, obs_shape):
        """Helper để load model đúng shape từ env"""
        self.model = self.model_class(obs_shape, 3).to(self.device)
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['policy_net'])
            print(f"Eval loaded: {self.checkpoint_path}")
        else:
            print("checkpoint_loi = ", self.checkpoint_path)
            print("Warning: Evaluated with random weights (no checkpoint found)")
        self.model.eval()

    def get_current_direction(self, obs_i, head_pos):
        """Xác định hướng hiện tại dựa trên thân kề đầu."""
        hy, hx = head_pos
        # Tìm thân kề đầu để suy ra hướng
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            by, bx = hy - dy, hx - dx
            if 0 <= by < obs_i.shape[0] and 0 <= bx < obs_i.shape[1]:
                if obs_i[by, bx, self.CH_MY_BODY] == 1 or obs_i[by, bx, self.CH_MY_TAIL] == 1:
                    return (dy, dx)
        return (-1, 0) # Mặc định Up

    def count_reachable_space(self, obs, start_pos, limit=60):
        """Flood Fill đếm không gian an toàn."""
        q = deque([tuple(start_pos)])
        visited = set([tuple(start_pos)])
        count = 0
        H, W = obs.shape[0], obs.shape[1]
        
        while q and count < limit:
            curr_y, curr_x = q.popleft()
            count += 1
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = curr_y + dy, curr_x + dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                    # Check an toàn với các kênh nguy hiểm
                    if not any(obs[ny, nx, ch] == 1 for ch in self.DEADLY_CHANNELS):
                        visited.add((ny, nx))
                        q.append((ny, nx))
        return count

    def get_action(self, obs_i, current_dir, occupied_next_positions=set()):
        """
        Xử lý logic chọn action cho 1 con rắn:
        1. Mask các hướng chết (tường, thân, rắn địch).
        2. Mask các hướng mà rắn khác vừa xí chỗ (occupied_next_positions).
        3. Check Head-to-Head.
        4. Check Flood Fill (đường cụt).
        5. Inference Model.
        
        Returns:
            action (int): 0, 1, 2
            next_dir (tuple): (dy, dx) hướng mới
            next_pos (tuple): (ny, nx) vị trí đầu mới
        """
        # Tìm đầu rắn
        head_pos = np.argwhere(obs_i[:, :, self.CH_MY_HEAD] == 1)
        if len(head_pos) == 0:
            return 0, (0,0), None # Chết rồi hoặc lỗi
        hy, hx = head_pos[0]

        # Xác định hướng hiện tại nếu chưa có
        if current_dir is None:
            current_dir = self.get_current_direction(obs_i, (hy, hx))
        
        dy, dx = current_dir
        # Map action sang vector hướng: 0: Thẳng, 1: Trái, 2: Phải
        possible_moves = {0: (dy, dx), 1: (-dx, dy), 2: (dx, -dy)}
        
        deadly_actions = []
        H, W = obs_i.shape[:2]
        
        # Tính độ dài rắn (để so sánh với flood fill)
        my_length = (np.sum(obs_i[:, :, self.CH_MY_HEAD] == 1) +
                     np.sum(obs_i[:, :, self.CH_MY_BODY] == 1) +
                     np.sum(obs_i[:, :, self.CH_MY_TAIL] == 1))

        # --- Loop check 3 hành động ---
        for action in [0, 1, 2]:
            mdy, mdx = possible_moves[action]
            ny, nx = hy + mdy, hx + mdx
            
            # 1. Check Biên
            if not (0 <= ny < H and 0 <= nx < W):
                deadly_actions.append(action)
                continue
            
            # 2. Check va chạm với nước đi của rắn khác (VỪA QUYẾT ĐỊNH trong step này)
            if (ny, nx) in occupied_next_positions:
                deadly_actions.append(action)
                continue

            # 3. Check va chạm vật cản tĩnh (tường, thân rắn...)
            is_dead = False
            for ch in self.DEADLY_CHANNELS:
                if obs_i[ny, nx, ch] == 1:
                    is_dead = True
                    break
            if is_dead:
                deadly_actions.append(action)
                continue

            # 4. Check Head-to-Head Risk (Ô cạnh đầu rắn địch hiện tại)
            near_enemy_head = False
            for h_dy, h_dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                hy_check, hx_check = ny + h_dy, nx + h_dx
                if 0 <= hy_check < H and 0 <= hx_check < W:
                    if obs_i[hy_check, hx_check, self.CH_OTHER_HEAD] == 1:
                        near_enemy_head = True
                        break
            if near_enemy_head:
                deadly_actions.append(action)
                continue

            # 5. Check Loop/Dead End (Flood Fill logic)
            sim = obs_i.copy()
            sim[hy, hx, self.CH_MY_HEAD] = 0
            sim[hy, hx, self.CH_MY_BODY] = 1
            sim[ny, nx, :] = 0
            sim[ny, nx, self.CH_MY_HEAD] = 1
            
            eating_fruit = (obs_i[ny, nx, self.CH_FRUIT] == 1)
            # Nếu không ăn táo, đuôi sẽ co lại -> xóa đuôi cũ trong simulation
            if not eating_fruit:
                tail_pos = np.argwhere(obs_i[:, :, self.CH_MY_TAIL] == 1)
                if len(tail_pos) > 0:
                    ty, tx = tail_pos[0]
                    sim[ty, tx, self.CH_MY_TAIL] = 0

            # Flood fill
            free_space = self.count_reachable_space(sim, [ny, nx], limit=60)
            new_length = my_length + 1 if eating_fruit else my_length
            
            # Nếu không gian < độ dài rắn -> Chết (hoặc loop quá bé)
            if free_space < new_length:
                deadly_actions.append(action)

        # --- Model Inference ---
        # Chuyển obs sang tensor
        state_t = torch.tensor(obs_i, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze()
        
        # Mask các hành động chết (-inf)
        for act in deadly_actions:
            q_values[act] = float('-inf')

        # Nếu tất cả đều chết, chọn đại cái ít âm nhất (hoặc random 0)
        if len(deadly_actions) == 3:
             # Vẫn phải chọn 1 hướng để trả về, dù biết là chết
             act = torch.argmax(q_values).item() 
        else:
             act = torch.argmax(q_values).item()
        
        # Tính toán kết quả trả về
        final_dy, final_dx = possible_moves[act]
        final_ny, final_nx = hy + final_dy, hx + final_dx
        
        return act, (final_dy, final_dx), (final_ny, final_nx)

    def evaluate(self, num_episodes=1, render=True):
            env, obs_shape, action_shape, _ = make_snake(
                num_envs=1, num_snakes=self.config.NUM_SNAKES,
                height=self.config.HEIGHT, width=self.config.WIDTH,
                snake_length=self.config.SNAKE_LENGTH, vision_range=self.config.VISION_RANGE,
                reward_dict=self.config.REWARD_DICT
            )
            
            if render:
                env = RenderGUI(
                    env,
                    save_video=True,
                    video_path=f"snake_eval_{self.config.HEIGHT}x{self.config.WIDTH}.mp4",
                    fps=20
                )

            # Init Model (vì cần obs_shape từ env)
            if self.model is None:
                self._load_model(env.observation_space.shape)
            
            # Lưu tổng cộng để tính trung bình cuối cùng
            global_total_rewards = 0.0
            global_total_steps = 0.0
            
            print(f"\n--- Starting Evaluation: {num_episodes} Episodes ---")
            
            for ep in range(num_episodes):
                obs = env.reset()
                dones = [False] * self.config.NUM_SNAKES
                
                # Theo dõi cho từng episode
                ep_step_counter = 0
                ep_rewards = np.zeros(self.config.NUM_SNAKES)
                # Timelife riêng cho từng con rắn trong episode này
                snake_timelifes = np.zeros(self.config.NUM_SNAKES) 
                
                current_directions = [None] * self.config.NUM_SNAKES
                
                while not all(dones) and ep_step_counter < 1000:
                    if render:
                        env.render()
                        time.sleep(0.01) # Giảm nhẹ delay để eval nhanh hơn
                    
                    actions = []
                    occupied_next_positions = set()
                    
                    for i in range(self.config.NUM_SNAKES):
                        if dones[i]:
                            actions.append(0)
                            continue
                        
                        # Rắn còn sống thì cộng thêm 1 vào timelife
                        snake_timelifes[i] += 1
                        
                        act, new_dir, next_pos = self.get_action(
                            obs[i], 
                            current_directions[i], 
                            occupied_next_positions
                        )
                        
                        actions.append(act)
                        current_directions[i] = new_dir
                        
                        if next_pos is not None:
                            occupied_next_positions.add(next_pos)
                    
                    # Step Environment
                    obs, rewards, dones, _ = env.step(actions)
                    
                    for i in range(self.config.NUM_SNAKES):
                        ep_rewards[i] += rewards[i]
                    
                    ep_step_counter += 1
                
                # Tính toán kết quả Episode
                avg_ep_reward = np.mean(ep_rewards)
                avg_ep_timelife = np.mean(snake_timelifes)
                
                global_total_rewards += avg_ep_reward
                global_total_steps += avg_ep_timelife
                
                print(f"Ep {ep+1:3d}: Avg Reward: {avg_ep_reward:6.2f} | Avg Timelife: {avg_ep_timelife:5.1f} steps")
                
            # Kết quả tổng hợp cuối cùng
            final_avg_reward = global_total_rewards / num_episodes
            final_avg_timelife = global_total_steps / num_episodes
            
            print("-" * 50)
            print(f"FINAL RESULTS OVER {num_episodes} EPISODES:")
            print(f" >> Average Reward per Snake: {final_avg_reward:.2f}")
            print(f" >> Average Timelife per Snake: {final_avg_timelife:.2f} steps")
            print("-" * 50)
            
            env.close()
            return final_avg_reward, final_avg_timelife

class ExternalAgentBase:
    """
    Class cha cho các thuật toán cũ của bạn.
    Bạn cần wrap code cũ vào class kế thừa class này.
    """
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.name = f"Algo_{agent_id}"

    def get_action(self, obs):
        """
        Input: obs (H, W, C) numpy array
        Output: action (0: thẳng, 1: trái, 2: phải)
        """
        # --- PLACEHOLDER: BẠN HÃY GỌI THUẬT TOÁN CŨ CỦA BẠN Ở ĐÂY ---
        # Ví dụ: return my_old_greedy_algo(obs)
        return random.choice([0, 1, 2]) 

class PPOEnemy(ExternalAgentBase):
    def __init__(self, agent_id, model_path, env_instance):
        super().__init__(agent_id)
        self.name = f"PPO_Agent_{agent_id}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo thuật toán PPO và load weight
        from algorithm.ppo import PPO # Đảm bảo đường dẫn import đúng
        self.ppo = PPO(env_instance)
        self.ppo.load(model_path)
        self.ppo.model.to(self.device)
        self.ppo.model.eval()

    def get_action(self, obs):
        # Chuyển đổi observation sang tensor đúng định dạng PPO yêu cầu
        # Thông thường PPO trong marlenv yêu cầu batch dim ở đầu
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        # Nếu obs chưa có dim batch (H, W, C) -> (1, H, W, C)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            # Sử dụng phương thức lấy action từ model PPO của bạn
            action, _, _, _ = self.ppo.model.get_action_and_value(obs_t)
        
        # Trả về giá trị int (0, 1, hoặc 2)
        return action.cpu().item()

class HybridNEATEnemy(ExternalAgentBase):
    def __init__(self, agent_id, result_pickle_path):
        super().__init__(agent_id)
        self.name = f"Hybrid_NEAT_{agent_id}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load file kết quả
        if not os.path.exists(result_pickle_path):
            raise FileNotFoundError(f"Cannot find {result_pickle_path}")
            
        with open(result_pickle_path, 'rb') as f:
            data = pickle.load(f)

        # 1. Setup Feature Extractor (DQN Part)
        # Giả định shape là (20, 20, 3) - Cần khớp với lúc train
        dummy_shape = (1, 20, 20, 8) 
        self.dqn_fe = DQN(dummy_shape, 3).to(self.device)
        
        # Load weight cho DQN
        dqn_state = data.get('dqn_state_dict', None)
        # Xử lý key 'module.' nếu train bằng DataParallel
        clean_state = {k.replace('module.', ''): v for k, v in dqn_state.items()}
        self.dqn_fe.load_state_dict(clean_state)
        
        self.dqn_fe.eval() # Quan trọng: Set mode eval
        
        # 2. Setup Decision Head (NEAT Part)
        genome = data['neat_genome']
        config = data['neat_config']
        self.neat_net = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_action(self, obs):
        # B1: Đưa qua DQN Feature Extractor
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # Nếu obs là (H, W, C) -> Thêm batch dim -> (1, H, W, C)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
            
        with torch.no_grad():
            # forward_features trả về vector 128 chiều
            features = self.dqn_fe.forward_features(obs_t).cpu().numpy().flatten()
            
        # B2: Đưa qua NEAT Decision Head
        # NEAT nhận input là list/array 1 chiều
        output = self.neat_net.activate(features)
        
        # B3: Argmax để ra action
        return int(np.argmax(output))

class GreedyEnemy(ExternalAgentBase):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.name = f"Greedy_FruitSeeker_{agent_id}"
        
        # Giữ nguyên các định nghĩa channel từ code của bạn
        self.CH_WALL = 0
        self.CH_FRUIT = 1
        self.CH_OTHER_HEAD = 2
        self.CH_OTHER_BODY = 3
        self.CH_OTHER_TAIL = 4
        self.CH_MY_HEAD = 5
        self.CH_MY_BODY = 6
        self.CH_MY_TAIL = 7
        
        self.deadly_channels = [
            self.CH_WALL, self.CH_OTHER_HEAD, self.CH_OTHER_BODY, 
            self.CH_OTHER_TAIL, self.CH_MY_BODY, self.CH_MY_TAIL
        ]
        self.current_direction = None

    def get_action(self, obs):
        # COPY TOÀN BỘ LOGIC TRONG HÀM get_action CỦA BẠN VÀO ĐÂY
        head_pos = np.argwhere(obs[:, :, self.CH_MY_HEAD] == 1)
        if len(head_pos) == 0: return 0
        hy, hx = head_pos[0]

        # 1. Tìm táo gần nhất
        fruit_positions = np.argwhere(obs[:, :, self.CH_FRUIT] == 1)
        target_fruit = None
        if len(fruit_positions) > 0:
            dists = [abs(hy - fy) + abs(hx - fx) for fy, fx in fruit_positions]
            target_fruit = fruit_positions[np.argmin(dists)]

        # 2. Xử lý hướng hiện tại
        if self.current_direction is None:
            curr_dir = (-1, 0)
            for dy_nb, dx_nb in [(-1,0),(1,0),(0,-1),(0,1)]:
                by, bx = hy + dy_nb, hx + dx_nb
                if 0 <= by < obs.shape[0] and 0 <= bx < obs.shape[1]:
                    if obs[by, bx, self.CH_MY_BODY] == 1 or obs[by, bx, self.CH_MY_TAIL] == 1:
                        curr_dir = (hy - by, hx - bx)
                        break
            self.current_direction = curr_dir

        dy, dx = self.current_direction
        possible_moves = {0: (dy, dx), 1: (-dx, dy), 2: (dx, -dy)}
        scores = []

        # 3. Đánh giá nước đi
        for action in [0, 1, 2]:
            mdy, mdx = possible_moves[action]
            ny, nx = hy + mdy, hx + mdx
            
            if (ny < 0 or ny >= obs.shape[0] or nx < 0 or nx >= obs.shape[1]):
                scores.append(-float('inf'))
                continue

            is_dead = False
            for ch in self.deadly_channels:
                if obs[ny, nx, ch] == 1:
                    is_dead = True
                    break
            
            if is_dead:
                scores.append(-float('inf'))
                continue

            score = 0
            if target_fruit is not None:
                dist = abs(ny - target_fruit[0]) + abs(nx - target_fruit[1])
                score = -dist 
            scores.append(score)

        if all(s == -float('inf') for s in scores):
            chosen = 0
        else:
            max_s = max(scores)
            best_actions = [i for i, v in enumerate(scores) if v == max_s]
            chosen = random.choice(best_actions)

        self.current_direction = possible_moves[chosen]
        return chosen
    
class BattleArena(DQN_Evaluator):
    """
    Môi trường đấu trường: Agent 0 là DQN, Agent 1,2,3 là External Agents.
    Kế thừa DQN_Evaluator để tái sử dụng hàm 'get_action' thông minh (floodfill, safety check).
    """
    def __init__(self, config, dqn_model_class, external_agents, checkpoint_tag="best"):
        super().__init__(config, dqn_model_class, checkpoint_tag)
        self.external_agents = external_agents # List chứa 3 object agent khác
        
        # Đảm bảo ta có đúng 3 đối thủ
        assert len(self.external_agents) == 3, "Cần đúng 3 external agents cho kịch bản 1vs3"

    def run_battle(self, num_episodes=10, render=True):
            # 1. Khởi tạo môi trường và tên hiển thị cho từng thuật toán
            display_names = ["DQN (Main)", "PPO Agent", "Hybrid NEAT", "Greedy Bot"]
            
            env, _, _, _ = make_snake(
                num_envs=1, 
                num_snakes=4,
                height=self.config.HEIGHT, 
                width=self.config.WIDTH,
                snake_length=self.config.SNAKE_LENGTH, 
                vision_range=self.config.VISION_RANGE,
                reward_dict=self.config.REWARD_DICT,
                render_mode=render 
            )

            if render:
                env = RenderGUI(
                    env, 
                    save_video=True,
                    video_path="battle_results_4.mp4", 
                    fps=15
                )

            # 2. Các biến tích lũy để tính Mean (Trung bình)
            # Index: 0=DQN, 1=PPO, 2=NEAT, 3=Greedy
            total_rewards = np.zeros(4)
            total_lifetimes = np.zeros(4)

            if self.model is None:
                self._load_model(env.observation_space.shape)
            
            print(f"\n>>> STARTING BATTLE: {num_episodes} EPISODES <<<")

            for ep in range(num_episodes):
                obs = env.reset()
                dones = [False] * 4
                dqn_current_dir = None 
                
                # Biến lưu trữ riêng cho episode hiện tại
                ep_rewards = np.zeros(4)
                ep_lifetimes = np.zeros(4)
                
                steps = 0
                while not all(dones) and steps < self.config.MAX_STEPS_PER_EPISODE * 2:
                    if render:
                        env.render(agent_names=display_names)
                        time.sleep(0.01)
                    
                    actions = []
                    for i in range(4):
                        if dones[i]:
                            actions.append(0) # Ngừng hành động nếu đã chết
                        else:
                            ep_lifetimes[i] += 1 # Tăng lifetime cho agent còn sống
                            if i == 0: # Agent DQN
                                act, new_dir, _ = self.get_action(obs[0], dqn_current_dir, set())
                                actions.append(act)
                                dqn_current_dir = new_dir
                            else: # Các đối thủ (PPO, NEAT, Greedy)
                                ext_act = self.external_agents[i-1].get_action(obs[i])
                                actions.append(ext_act)

                    # Bước đi trong môi trường
                    obs, rewards, dones, info = env.step(actions)
                    
                    # Tích lũy reward nhận được tại mỗi step cho từng agent
                    for i in range(4):
                        ep_rewards[i] += rewards[i]
                    
                    steps += 1
                
                # Sau khi episode kết thúc, cộng vào tổng tích lũy
                total_rewards += ep_rewards
                total_lifetimes += ep_lifetimes
                
                print(f"Episode {ep+1:2d} Done. Steps: {steps}")

            # 3. TÍNH TOÁN VÀ LOG KẾT QUẢ TRUNG BÌNH (MEAN)
            print("\n" + "="*65)
            print(f"{'ALGORITHM':<20} | {'MEAN REWARD':<18} | {'MEAN LIFETIME':<15}")
            print("-" * 65)
            
            for i in range(4):
                mean_reward = total_rewards[i] / num_episodes
                mean_lifetime = total_lifetimes[i] / num_episodes
                
                # Log ra console theo định dạng bảng
                print(f"{display_names[i]:<20} | {mean_reward:>18.2f} | {mean_lifetime:>15.1f}")
                
            print("="*65 + "\n")
            env.close()

# ========================== MAIN ==========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "battle"])
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="final")
    parser.add_argument("--no-render", action="store_true")
   
    args = parser.parse_args()
    config = Config()
    config.NUM_EPISODES = args.episodes
    config.RESUME_FROM = args.resume
   
    if args.mode == "train":
        trainer = Trainer(config)
        trainer.train()
    elif args.mode == "eval":
        evaluator = DQN_Evaluator(config=config, model_class=DQN, checkpoint_tag="15500")
        evaluator.evaluate(num_episodes=20, render=False)
    
    elif args.mode == "battle":
        temp_env, _, _, _ = make_snake(num_envs=1, num_snakes=4, height=20, width=20)
        ppo_agent = PPOEnemy(
                agent_id=1, 
                model_path="snake_shared_policy_3_snakes_new_reward_ver3.pt", 
                env_instance=temp_env
        )

        # 2. Hybrid NEAT Agent
        hybrid_agent = HybridNEATEnemy(
            agent_id=2, 
            result_pickle_path="hybrid_neat_best.pkl"
        )

        # 3. Greedy Agent (Thuật toán tìm đường cơ bản của bạn)
        greedy_agent = GreedyEnemy(agent_id=3)

        # Gom danh sách đối thủ
        enemies = [ppo_agent, hybrid_agent, greedy_agent]

        # Khởi tạo Arena (DQN sẽ là Agent 0)
        arena = BattleArena(
            config=config, 
            dqn_model_class=DQN, 
            external_agents=enemies, 
            checkpoint_tag="15500"
        )
        
        # Chạy 50 trận đấu để lấy số liệu thống kê cuối cùng
        arena.run_battle(num_episodes=1, render=True)
        
        temp_env.close()
