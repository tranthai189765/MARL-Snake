"""
Deep Q-Learning Baseline cho MARL-Snake
========================================
Mỗi agent có một DQN riêng, học độc lập.
Observation: image grid (H, W, C)
Action: 0 (thẳng), 1 (trái), 2 (phải)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import time
from datetime import datetime

from marlenv.marlenv.wrappers import make_snake, RenderGUI

# ========================== CONFIG ==========================
class Config:
    # Environment
    NUM_SNAKES = 4
    HEIGHT = 20
    WIDTH = 20
    SNAKE_LENGTH = 5
    VISION_RANGE = 5  # None = full map, 5 = 11x11 local view
    
    # Training
    NUM_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 500
    BATCH_SIZE = 64
    GAMMA = 0.99  # Discount factor
    LR = 1e-4
    
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995
    
    # Replay Buffer
    BUFFER_SIZE = 100000
    MIN_BUFFER_SIZE = 1000 # Kích thước tối thiểu để bắt đầu training
    
    # Target Network
    TARGET_UPDATE_FREQ = 100  # Cập nhật target network mỗi N episodes
    
    # Reward shaping - Preset selection
    REWARD_PRESET = "default"  # "default" hoặc "late_training"
    
    # Early death penalty (phạt chết sớm)
    EARLY_DEATH_THRESHOLD = 10  # Nếu chết trong 10 bước đầu
    EARLY_DEATH_PENALTY = -1.0  # Phạt thêm -1.0
    
    # Reward dictionaries
    REWARD_DICT = {
        'fruit': 1.0,    # Ăn trái cây
        'kill': 2.0,     # Giết rắn khác
        'lose': -1.0,    # Chết
        'win': 0.5,      # Thắng (mỗi step)
        'time': 0.01,    # Thưởng sống sót
    }
    
    # Late training preset - Giảm tự hủy đầu game
    REWARD_DICT_LATE = {
        'fruit': 1.0,
        'kill': 2.0,
        'lose': -1.5,    # Phạt chết nặng hơn
        'win': 0.5,
        'time': 0.0,     # Tắt time reward - tập trung vào quality
    }
    
    # Save/Load - Checkpoint Strategy
    SAVE_FREQ = 500         
    SAVE_BEST_ONLY = True    
    KEEP_LAST_N = 3         
    SAVE_DIR = "checkpoints"
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
    """
    CNN-based DQN cho observation dạng image grid.
    Input: (batch, H, W, C) - sẽ được permute sang (batch, C, H, W)
    Output: Q-values cho mỗi action
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        h, w, c = input_shape
        # CNN layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Tính output size sau conv layers
        conv_out_size = h * w * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        # x: (batch, H, W, C) -> (batch, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float()
        
        # Normalize to [0, 1]
        x = x / 255.0 if x.max() > 1.0 else x
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# ========================== DQN AGENT ==========================
class DQNAgent:
    def __init__(self, agent_id, input_shape, num_actions, config):
        self.agent_id = agent_id
        self.config = config
        self.num_actions = num_actions
        
        # Networks
        self.policy_net = DQN(input_shape, num_actions).to(config.DEVICE)
        self.target_net = DQN(input_shape, num_actions).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        
        # Replay Buffer
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # Epsilon
        self.epsilon = config.EPSILON_START
        
        # Statistics
        self.losses = []
        self.rewards = []
        
        # Best model tracking
        self.best_avg_reward = float('-inf')
        
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.config.DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """Một bước cập nhật DQN"""
        if len(self.memory) < self.config.MIN_BUFFER_SIZE:
            return None
        
        # Sample batch
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.config.DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.config.DEVICE)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.config.DEVICE)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.config.DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.config.DEVICE)
        
        # Compute Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute max Q(s', a') from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.config.GAMMA * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_END, 
                          self.epsilon * self.config.EPSILON_DECAY)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_avg_reward': self.best_avg_reward,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.config.DEVICE, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        if 'best_avg_reward' in checkpoint:
            self.best_avg_reward = checkpoint['best_avg_reward']


# ========================== TRAINING ==========================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.start_episode = 1  # Episode bắt đầu training
        
        # Chọn reward dict dựa trên preset
        reward_dict = config.REWARD_DICT_LATE if config.REWARD_PRESET == "late_training" else config.REWARD_DICT
        
        # Tạo môi trường
        self.env, self.obs_shape, self.action_shape, self.properties = make_snake(
            num_envs=1,
            num_snakes=config.NUM_SNAKES,
            height=config.HEIGHT,
            width=config.WIDTH,
            snake_length=config.SNAKE_LENGTH,
            vision_range=config.VISION_RANGE,
            reward_dict=reward_dict
        )
        
        self.num_agents = config.NUM_SNAKES
        self.num_actions = self.action_shape[0]
        
        # Observation shape cho mỗi agent
        if config.VISION_RANGE:
            h = w = config.VISION_RANGE * 2 + 1
        else:
            h, w = config.HEIGHT, config.WIDTH
        # Lấy số channels từ properties hoặc tính từ obs
        self.agent_obs_shape = self.obs_shape  # (H, W, C)
        
        print(f"Observation shape per agent: {self.agent_obs_shape}")
        print(f"Number of actions: {self.num_actions}")
        print(f"Device: {config.DEVICE}")
        print(f"Reward preset: {config.REWARD_PRESET}")
        print(f"Reward dict: {reward_dict}")
        print(f"Early death penalty: {config.EARLY_DEATH_PENALTY} (threshold: {config.EARLY_DEATH_THRESHOLD} steps)")
        
        # Tạo agents
        self.agents = [
            DQNAgent(i, self.agent_obs_shape, self.num_actions, config)
            for i in range(self.num_agents)
        ]
        
        # Thống kê
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        self.episode_lengths = []
        self.wins = [0] * self.num_agents
        
        # Tạo thư mục save
        os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    def train(self):
        print("\n" + "="*60)
        print("BẮT ĐẦU TRAINING DQN CHO MARL-SNAKE")
        print("="*60)
        print(f"[CONFIG] Checkpoint Strategy:")
        print(f"   - Save periodic: {'Every ' + str(self.config.SAVE_FREQ) + ' eps' if self.config.SAVE_FREQ else 'Disabled'}")
        print(f"   - Save best only: {self.config.SAVE_BEST_ONLY}")
        print(f"   - Keep last N: {self.config.KEEP_LAST_N if self.config.KEEP_LAST_N else 'All'}")
        
        # Resume từ checkpoint nếu có
        if self.config.RESUME_FROM:
            resume_episode = self.config.RESUME_FROM
            if self.load_checkpoint_for_resume(resume_episode):
                self.start_episode = resume_episode + 1
                print(f"\n[RESUME] Tiếp tục từ episode {resume_episode}")
                print(f"   Training sẽ bắt đầu từ episode {self.start_episode}")
            else:
                print(f"\n[WARNING] Không tìm thấy checkpoint ep{resume_episode}, train từ đầu")
        
        print("="*60 + "\n")
        
        checkpoint_history = []  # Lưu danh sách episodes đã checkpoint
        
        for episode in range(self.start_episode, self.config.NUM_EPISODES + 1):
            obs = self.env.reset()
            dones = [False] * self.num_agents
            episode_reward = [0.0] * self.num_agents
            step = 0
            
            while not all(dones) and step < self.config.MAX_STEPS_PER_EPISODE:
                # Chọn actions cho mỗi agent
                actions = []
                for i, agent in enumerate(self.agents):
                    if dones[i]:
                        actions.append(0)  # Dead agent, action không quan trọng
                    else:
                        actions.append(agent.select_action(obs[i], training=True))
                
                # Thực hiện step
                next_obs, rewards, dones, info = self.env.step(actions)
                
                # Lưu transitions và cập nhật
                for i, agent in enumerate(self.agents):
                    reward = rewards[i]
                    
                    # Áp dụng early death penalty nếu chết sớm
                    if dones[i] and step < self.config.EARLY_DEATH_THRESHOLD:
                        reward += self.config.EARLY_DEATH_PENALTY
                    
                    if not dones[i] or step == 0:  # Lưu cả step cuối khi chết
                        agent.store_transition(
                            obs[i], actions[i], reward, next_obs[i], dones[i]
                        )
                    episode_reward[i] += reward
                    
                    # Update network
                    agent.update()
                
                obs = next_obs
                step += 1
            
            # End of episode
            for i, agent in enumerate(self.agents):
                agent.decay_epsilon()
                self.episode_rewards[i].append(episode_reward[i])
            self.episode_lengths.append(step)
            
            # Cập nhật target network
            if episode % self.config.TARGET_UPDATE_FREQ == 0:
                for agent in self.agents:
                    agent.update_target_network()
            
            # Logging
            if episode % 10 == 0:
                avg_rewards = [np.mean(r[-100:]) if r else 0 for r in self.episode_rewards]
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_epsilon = np.mean([a.epsilon for a in self.agents])
                
                print(f"Episode {episode:5d} | "
                      f"Rewards: {[f'{r:.2f}' for r in avg_rewards]} | "
                      f"Length: {avg_length:.1f} | "
                      f"ε: {avg_epsilon:.3f}")
            
            # ===== CHECKPOINT SAVING =====
            
            # 1. Lưu best model (nếu enable) - kiểm tra mỗi 50 episodes
            if self.config.SAVE_BEST_ONLY and episode >= 50 and episode % 50 == 0:
                avg_rewards = [np.mean(r[-100:]) if len(r) >= 50 else np.mean(r) if r else 0 for r in self.episode_rewards]
                for i, agent in enumerate(self.agents):
                    if avg_rewards[i] > agent.best_avg_reward:
                        agent.best_avg_reward = avg_rewards[i]
                        best_path = os.path.join(self.config.SAVE_DIR, f"agent_{i}_best.pth")
                        agent.save(best_path)
                        print(f"   [BEST] Agent {i} NEW BEST! Reward: {avg_rewards[i]:.2f}")
            
            # 2. Lưu periodic checkpoint (nếu enable)
            if self.config.SAVE_FREQ and episode % self.config.SAVE_FREQ == 0:
                self.save_checkpoint(episode)
                checkpoint_history.append(episode)
                
                # 3. Xóa checkpoint cũ (chỉ giữ N gần nhất)
                if self.config.KEEP_LAST_N and len(checkpoint_history) > self.config.KEEP_LAST_N:
                    old_episode = checkpoint_history.pop(0)
                    self.delete_checkpoint(old_episode)
        
        print("\n" + "="*60)
        print("[DONE] TRAINING HOAN THANH!")
        print("="*60)
        self.save_checkpoint("final")
        self.print_summary()
    
    def save_checkpoint(self, episode):
        """Lưu checkpoint và hiển thị thông báo"""
        for i, agent in enumerate(self.agents):
            path = os.path.join(self.config.SAVE_DIR, f"agent_{i}_ep{episode}.pth")
            agent.save(path)
        print(f"   [SAVE] Saved checkpoint: episode {episode}")
    
    def delete_checkpoint(self, episode):
        """Xóa checkpoint cũ để tiết kiệm disk"""
        for i in range(self.num_agents):
            path = os.path.join(self.config.SAVE_DIR, f"agent_{i}_ep{episode}.pth")
            if os.path.exists(path):
                os.remove(path)
        print(f"   [DELETE] Deleted old checkpoint: episode {episode}")
    
    def print_summary(self):
        """In tóm tắt training"""
        print("\n[SUMMARY] TRAINING SUMMARY:")
        print("-" * 60)
        for i, agent in enumerate(self.agents):
            avg_reward = np.mean(self.episode_rewards[i][-100:]) if self.episode_rewards[i] else 0
            total_transitions = len(agent.memory)
            best_reward_str = f"{agent.best_avg_reward:.2f}" if agent.best_avg_reward > float('-inf') else "N/A"
            print(f"Agent {i}:")
            print(f"  - Final avg reward (100-ep): {avg_reward:.2f}")
            print(f"  - Best avg reward ever:      {best_reward_str}")
            print(f"  - Final epsilon:             {agent.epsilon:.3f}")
            print(f"  - Total transitions stored:  {total_transitions}")
        
        avg_final_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        print(f"\nAverage episode length (final 100): {avg_final_length:.1f} steps")
        print("-" * 60)
    
    def load_checkpoint(self, episode):
        """Load checkpoint để eval (không cần đầy đủ state)"""
        for i, agent in enumerate(self.agents):
            path = os.path.join(self.config.SAVE_DIR, f"agent_{i}_ep{episode}.pth")
            if os.path.exists(path):
                agent.load(path)
        print(f"📂 Đã load checkpoint từ episode {episode}")
    
    def load_checkpoint_for_resume(self, episode):
        """Load checkpoint để tiếp tục training"""
        success = True
        for i, agent in enumerate(self.agents):
            path = os.path.join(self.config.SAVE_DIR, f"agent_{i}_ep{episode}.pth")
            if not os.path.exists(path):
                success = False
                break
            agent.load(path)
            best_str = f"{agent.best_avg_reward:.2f}" if agent.best_avg_reward > float('-inf') else "N/A"
            print(f"   [LOAD] Agent {i} from ep{episode} (epsilon={agent.epsilon:.3f}, best={best_str})")
        return success


# ========================== EVALUATION ==========================
def evaluate(config, checkpoint_episode="final", num_episodes=10, render=True, max_steps=500):
    """Đánh giá agents đã train"""
    
    # Tạo môi trường
    env, obs_shape, action_shape, properties = make_snake(
        num_envs=1,
        num_snakes=config.NUM_SNAKES,
        height=config.HEIGHT,
        width=config.WIDTH,
        snake_length=config.SNAKE_LENGTH,
        vision_range=config.VISION_RANGE,
        reward_dict=config.REWARD_DICT
    )
    
    if render:
        env = RenderGUI(env)
    
    num_actions = action_shape[0]
    
    # Load agents
    agents = []
    for i in range(config.NUM_SNAKES):
        agent = DQNAgent(i, obs_shape, num_actions, config)
        
        # Ưu tiên load best model nếu có
        best_path = os.path.join(config.SAVE_DIR, f"agent_{i}_best.pth")
        normal_path = os.path.join(config.SAVE_DIR, f"agent_{i}_ep{checkpoint_episode}.pth")
        
        if checkpoint_episode == "best" and os.path.exists(best_path):
            agent.load(best_path)
            best_str = f"{agent.best_avg_reward:.2f}" if agent.best_avg_reward > float('-inf') else "N/A"
            print(f"[OK] Loaded BEST model for Agent {i} (reward: {best_str})")
        elif os.path.exists(normal_path):
            agent.load(normal_path)
            print(f"[OK] Loaded checkpoint ep{checkpoint_episode} for Agent {i}")
        elif os.path.exists(best_path):
            agent.load(best_path)
            print(f"[OK] Checkpoint not found, loaded BEST model for Agent {i}")
        else:
            print(f"[WARN] No checkpoint found for Agent {i}, using untrained model!")
        
        agent.epsilon = 0.0  # No exploration during evaluation
        agents.append(agent)
    
    print(f"\n[EVAL] Max steps per episode: {max_steps}")
    
    # Evaluate
    total_rewards = [0.0] * config.NUM_SNAKES
    wins = [0] * config.NUM_SNAKES
    timeouts = 0  # Đếm số episode bị timeout
    
    for ep in range(num_episodes):
        obs = env.reset()
        dones = [False] * config.NUM_SNAKES
        episode_reward = [0.0] * config.NUM_SNAKES
        step = 0
        
        while not all(dones) and step < max_steps:
            if render:
                env.render()
                time.sleep(0.1)
            
            actions = []
            for i, agent in enumerate(agents):
                if dones[i]:
                    actions.append(0)
                else:
                    actions.append(agent.select_action(obs[i], training=False))
            
            obs, rewards, dones, info = env.step(actions)
            
            for i in range(config.NUM_SNAKES):
                episode_reward[i] += rewards[i]
            
            step += 1
        
        # Check timeout
        timeout_flag = ""
        if step >= max_steps and not all(dones):
            timeouts += 1
            timeout_flag = " [TIMEOUT]"
        
        for i in range(config.NUM_SNAKES):
            total_rewards[i] += episode_reward[i]
        
        # Tìm winner
        if 'rank' in info:
            winner_idx = info['rank'].index(1) if 1 in info['rank'] else -1
            if winner_idx >= 0:
                wins[winner_idx] += 1
        
        print(f"Episode {ep+1}: Rewards = {[f'{r:.2f}' for r in episode_reward]} | Steps: {step}{timeout_flag}")
    
    env.close()
    
    print("\n" + "="*60)
    print("[RESULT] KET QUA DANH GIA")
    print("="*60)
    for i in range(config.NUM_SNAKES):
        avg_reward = total_rewards[i]/num_episodes
        win_rate = wins[i]/num_episodes * 100
        print(f"Agent {i}:")
        print(f"  - Avg Reward: {avg_reward:.2f}")
        print(f"  - Wins: {wins[i]}/{num_episodes} ({win_rate:.1f}%)")
    
    if timeouts > 0:
        print(f"\n[WARN] {timeouts}/{num_episodes} episodes hit timeout ({max_steps} steps)")
    print("="*60)


# ========================== MAIN ==========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN Training cho MARL-Snake")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "both"])
    parser.add_argument("--episodes", type=int, default=5000, help="Tổng số episodes training")
    parser.add_argument("--resume", type=int, default=None, help="Resume từ episode (ví dụ: --resume 1800)")
    parser.add_argument("--reward-preset", type=str, default="default", 
                       choices=["default", "late_training"],
                       help="Reward preset: default (có time reward) hoặc late_training (phạt chết sớm nặng hơn)")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-eval-steps", type=int, default=500, help="Max steps per eval episode (prevent infinite loops)")
    parser.add_argument("--checkpoint", type=str, default="final")
    parser.add_argument("--no-render", action="store_true")
    
    args = parser.parse_args()
    
    config = Config()
    config.NUM_EPISODES = args.episodes
    config.RESUME_FROM = args.resume
    config.REWARD_PRESET = args.reward_preset
    
    if args.mode == "train" or args.mode == "both":
        trainer = Trainer(config)
        trainer.train()
    
    if args.mode == "eval" or args.mode == "both":
        evaluate(config, 
                checkpoint_episode=args.checkpoint, 
                num_episodes=args.eval_episodes,
                render=not args.no_render,
                max_steps=args.max_eval_steps)
