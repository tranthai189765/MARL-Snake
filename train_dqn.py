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
 
from marlenv.marlenv.wrappers import make_snake, RenderGUI

# ========================== CONFIG ==========================
class Config:
    # Environment
    NUM_SNAKES = 4
    HEIGHT = 20
    WIDTH = 20
    SNAKE_LENGTH = 5
    VISION_RANGE = None
   
    # Training
    NUM_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 512
    BATCH_SIZE = 256
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
        'fruit': 10.0,
        'kill': 0.0,
        'lose': -50.0,
        'win': 0.0,
        'time': -0.03,
    }

   
    REWARD_DICT_LATE = {
        'fruit': 10.0,
        'kill': 0.0,
        'lose': -50.0,
        'win': 0.0,
        'time': -0.03,
    }

   
    # Checkpoints & Logs
    SAVE_FREQ = 500        
    SAVE_BEST_ONLY = True    
    KEEP_LAST_N = 2        
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
 
# ========================== EVALUATION ==========================
def evaluate(config, checkpoint_tag="best", num_episodes=10, render=True):
    env, obs_shape, action_shape, _ = make_snake(
        num_envs=1, num_snakes=config.NUM_SNAKES,
        height=config.HEIGHT, width=config.WIDTH,
        snake_length=config.SNAKE_LENGTH, vision_range=config.VISION_RANGE
    )
    if render:
        env = RenderGUI(env)
   
    # Load model
    obs_shape  = env.observation_space.shape
    model = DQN(obs_shape, 3).to(config.DEVICE)
    path = os.path.join(config.SAVE_DIR, f"shared_model_{checkpoint_tag}.pth")
   
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['policy_net'])
        print(f"Eval loaded: {path}")
    else:
        print("Warning: Evaluated with random weights (no checkpoint found)")
   
    model.eval()
   
    total_rewards = np.zeros(config.NUM_SNAKES)
   
    for ep in range(num_episodes):
        obs = env.reset()
        dones = [False] * config.NUM_SNAKES
        step = 0
        ep_rews = np.zeros(config.NUM_SNAKES)
       
        while not all(dones) and step < 500:
            if render:
                print("day nay dmm")
                env.render()
                time.sleep(0.05)
           
            actions = []
            with torch.no_grad():
                for i in range(config.NUM_SNAKES):
                    if dones[i]:
                        actions.append(0)
                    else:
                        # Inference
                        state_t = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
                        act = model(state_t).argmax().item()
                        actions.append(act)
           
            obs, rewards, dones, _ = env.step(actions)
            for i in range(config.NUM_SNAKES):
                ep_rews[i] += rewards[i]
            step += 1
       
        print(f"Ep {ep+1}: Rewards {ep_rews} | Steps {step}")
        total_rewards += ep_rews
   
    print(f"Average Rewards: {total_rewards/num_episodes}")
    env.close()
 
# ========================== MAIN ==========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=4500)
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
    else:
        evaluate(config, checkpoint_tag=args.checkpoint, render=True)
