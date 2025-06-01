import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import time


########################################################
# Existing/Imported Functions from path3 (no omissions) #
########################################################

def read_asc_file(filename):
    metadata_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']
    metadata = {key: None for key in metadata_keys}

    with open(filename, 'r') as f:
        lines_read = 0
        while lines_read < 100:
            line = f.readline().strip().lower()
            if not line or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key in metadata_keys:
                if key in ['ncols', 'nrows']:
                    metadata[key] = int(parts[1])
                else:
                    metadata[key] = float(parts[1])
                lines_read += 1
            if all(metadata.values()):
                break
        data = np.loadtxt(f, dtype=np.float32)
    missing = [k for k, v in metadata.items() if v is None]
    if missing:
        raise ValueError(f"Missing metadata fields: {missing}")

    return data, metadata


def create_obstacle_map(data, metadata):
    nodata = metadata['nodata_value']
    obstacle_map = np.where((data == nodata) | (data > 0), 1, 0)
    return obstacle_map.astype(np.int8)


def grid_to_geo(row, col, metadata):
    x = metadata['xllcorner'] + col * metadata['cellsize']
    y = metadata['yllcorner'] + (metadata['nrows'] - row - 1) * metadata['cellsize']
    return (x, y)


def geo_to_grid(x, y, metadata):
    col = int((x - metadata['xllcorner']) // metadata['cellsize'])
    row = int(metadata['nrows'] - (y - metadata['yllcorner']) // metadata['cellsize'] - 1)
    if row < 0 or row >= metadata['nrows'] or col < 0 or col >= metadata['ncols']:
        raise ValueError("Coordinates out of bounds")
    return (row, col)


def calculate_path_length(path, metadata, is_3d=False):
    """
    计算路径的实际长度（以米为单位）
    """
    if not path or len(path) < 2:
        return 0.0

    total_length = 0.0

    for i in range(1, len(path)):
        if is_3d:
            # 3D路径计算
            prev_point = path[i - 1]
            curr_point = path[i]

            # 转换为地理坐标
            prev_x, prev_y = grid_to_geo(prev_point[0], prev_point[1], metadata)
            curr_x, curr_y = grid_to_geo(curr_point[0], curr_point[1], metadata)

            # 计算3D距离
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            dz = curr_point[2] - prev_point[2]

            segment_length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        else:
            # 2D路径计算
            prev_point = path[i - 1]
            curr_point = path[i]

            # 转换为地理坐标
            prev_x, prev_y = grid_to_geo(prev_point[0], prev_point[1], metadata)
            curr_x, curr_y = grid_to_geo(curr_point[0], curr_point[1], metadata)

            # 计算2D距离
            dx = curr_x - prev_x
            dy = curr_y - prev_y

            segment_length = np.sqrt(dx ** 2 + dy ** 2)

        total_length += segment_length

    return total_length




########################################################
# Classes from path3 for DQN Training (no omissions)    #
########################################################

class PathPlanningEnv:
    def __init__(self, obstacle_map, start, goal, is_3d=False, elevation_data=None, delta_z=5):
        self.grid = obstacle_map
        self.start = start
        self.goal = goal
        self.is_3d = is_3d
        self.elevation_data = elevation_data
        self.delta_z = delta_z
        self.current_pos = start
        self.actions_2d = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.actions_3d = self.actions_2d + [('up',), ('down',)]
        self.goal_pos = goal

    def reset(self):
        self.current_pos = self.start
        return self.get_state()

    def get_state(self):
        if self.is_3d:
            # 标准化当前位置和高度
            current = (self.current_pos[0] / self.grid.shape[0],
                       self.current_pos[1] / self.grid.shape[1],
                       self.current_pos[2] / 100)
            goal = (self.goal_pos[0] / self.grid.shape[0],
                    self.goal_pos[1] / self.grid.shape[1],
                    self.goal_pos[2] / 100)
        else:
            current = (self.current_pos[0] / self.grid.shape[0],
                       self.current_pos[1] / self.grid.shape[1])
            goal = (self.goal_pos[0] / self.grid.shape[0],
                    self.goal_pos[1] / self.grid.shape[1])

        return np.concatenate([current, goal])

    def step(self, action):
        done = False
        reward = 0
        prev_position = np.array(self.current_pos[:2])

        if self.is_3d:
            if action < 4:  # 水平移动
                dx, dy = self.actions_2d[action]
                new_x = self.current_pos[0] + dx
                new_y = self.current_pos[1] + dy
                new_z = self.current_pos[2]
            else:  # 垂直移动
                dz = self.delta_z if action == 4 else -self.delta_z
                new_x, new_y = self.current_pos[:2]
                new_z = self.current_pos[2] + dz
        else:
            dx, dy = self.actions_2d[action]
            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy

        # 边界检查
        if 0 <= new_x < self.grid.shape[0] and 0 <= new_y < self.grid.shape[1]:
            if self.grid[new_x, new_y] == 1:
                if self.is_3d:
                    if self.elevation_data[new_x, new_y] <= new_z:
                        self.current_pos = (new_x, new_y, new_z)
                    else:
                        reward = -100  # 高度不足惩罚
                        return self.get_state(), reward, done, {}
                else:
                    reward = -100
            else:
                self.current_pos = (new_x, new_y, new_z) if self.is_3d else (new_x, new_y)
        else:
            reward = -100  # 越界惩罚

        # 改进奖励设计
        new_position = np.array(self.current_pos[:2])
        target_position = np.array(self.goal_pos[:2])

        # 距离变化奖励
        prev_dist = np.linalg.norm(prev_position - target_position)
        new_dist = np.linalg.norm(new_position - target_position)
        distance_reward = (prev_dist - new_dist) * 5  # 强化距离缩短奖励

        # 基础生存奖励
        survival_penalty = -0.5

        reward = distance_reward + survival_penalty

        # 终点奖励
        if np.array_equal(new_position, target_position):
            reward += 1000
            done = True

        return self.get_state(), reward, done, {}


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, env, is_3d=False):
        self.env = env
        self.is_3d = is_3d
        self.state_dim = 6 if is_3d else 4  # 3D时包含高度信息
        self.action_dim = len(env.actions_3d) if is_3d else 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)  # 将模型迁移到 GPU
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)  # 同样迁移目标模型
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.best_path = None
        self.best_distance = float('inf')
        self.best_reward = -float('inf')
        self.loss_history = []

    def act(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes=500, render_interval=50, layer_min=None, layer_max=None):
        if self.best_path is None:
            self.best_path = [self.env.start]
            self.best_distance = np.linalg.norm(
                np.array(self.env.start[:2]) -
                np.array(self.env.goal[:2])
            )

        rewards_history = []
        steps_history = []
        success_count = 0  # 成功到达目标的次数
        last_episode_rewards = []
        last_episode_path = []

        for ep in range(episodes):
            if ep == episodes - 1:
                last_episode_rewards = []
                last_episode_path = []

            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            current_path = [self.env.current_pos]
            best_reward = 0
            no_improve_count = 0
            episode_losses = []

            while not done and steps < 10000:  # 最大步数限制
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                if ep == episodes - 1:
                    last_episode_rewards.append(reward)
                    last_episode_path.append(self.env.current_pos)

                self.remember(state, action, reward, next_state, done)

                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > 20:  # 连续20回合无改进
                    self.epsilon = min(0.5, self.epsilon + 0.1)  # 重新激活探索
                    no_improve_count = 0

                loss_value = self.replay()
                state = next_state
                total_reward += reward
                steps += 1
                current_path.append(self.env.current_pos)

                if loss_value > 0:
                    episode_losses.append(loss_value)

            # 检查是否成功到达目标
            final_pos = np.array(current_path[-1][:2])
            target_pos = np.array(self.env.goal[:2])
            current_distance = np.linalg.norm(final_pos - target_pos)

            # 如果到达目标或非常接近目标，算作成功
            if done or current_distance < 1.0:
                success_count += 1

            avg_loss_this_episode = np.mean(episode_losses) if episode_losses else 0.0
            self.loss_history.append(avg_loss_this_episode)

            if ep == episodes - 1:
                print("\n最后一回合分析:")
                print(f"总步数: {steps}")
                print(f"最终奖励: {total_reward:.1f}")
                self.plot_last_episode(last_episode_rewards, last_episode_path, self.env.grid)

            # 基于奖励更新最佳路径
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_path = current_path.copy()
                if layer_min:
                    self.save_model(f'best_{layer_min}_{layer_max}_path_model.pth')
                else:
                    self.save_model(f'best_{"3d" if self.is_3d else "2d"}_path_model.pth')

            rewards_history.append(total_reward)
            steps_history.append(steps)
            self.update_target()
            success_rate = (success_count / (ep + 1)) * 100
            print(f"回合 {ep + 1}/{episodes}, 奖励: {total_reward:.1f}, "
                  f"步数: {steps}, Epsilon: {self.epsilon:.3f}, "
                  f"成功率: {success_rate:.1f}%")
            # 进度显示
            if ep % render_interval == 0 or ep == episodes - 1:
                self.plot_progress(rewards_history, steps_history)


        # 计算最终成功率
        final_success_rate = (success_count / episodes) * 100

        # 返回训练统计信息
        training_stats = {
            'success_rate': final_success_rate,
            'total_episodes': episodes,
            'successful_episodes': success_count,
            'rewards_history': rewards_history,
            'steps_history': steps_history
        }

        return training_stats

    def plot_progress(self, rewards, steps):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.subplot(1, 2, 2)
        plt.plot(steps)
        plt.title('Steps per Episode')
        plt.tight_layout()
        plt.show()

    def plot_last_episode(self, step_rewards, path, obstacle_map):
        plt.figure(figsize=(16, 6))

        # === 左图：累计奖励趋势 ===
        plt.subplot(1, 2, 1)

        # 计算累计奖励
        cumulative_rewards = np.cumsum(step_rewards)

        # 主曲线
        main_line, = plt.plot(cumulative_rewards,
                              color='#2C5F8D',
                              linewidth=1.5,
                              alpha=0.8,
                              label='Total Reward')

        # 标注最大增长率点
        gradients = np.diff(cumulative_rewards)
        max_gradient_idx = np.argmax(gradients) + 1
        plt.scatter(max_gradient_idx, cumulative_rewards[max_gradient_idx],
                    color='#EE7621', s=80, zorder=5,
                    label='Max Reward Rate')

        # 终值标注线
        plt.axhline(y=cumulative_rewards[-1],
                    color='#7C878E', linestyle='--',
                    linewidth=1, alpha=0.7,
                    label=f'Final: {cumulative_rewards[-1]:.1f}')

        # 双坐标轴设置
        ax = plt.gca()
        ax.set_xlabel('Step Number', fontsize=12)
        ax.set_ylabel('Cumulative Reward', color='#2C5F8D', fontsize=12)
        ax.tick_params(axis='y', labelcolor='#2C5F8D')

        ax2 = ax.twinx()
        ax2.plot(gradients, color='#7C878E', alpha=0.6,
                 linestyle='--', label='Reward Rate')
        ax2.set_ylabel('Reward Rate (Δ/step)', color='#7C878E', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#7C878E')

        plt.title('Accumulated Reward Trend with Reward Rate', fontsize=14)
        lines = [main_line] + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper left')

        # 右图：路径可视化
        plt.subplot(1, 2, 2)
        # 绘制障碍物地图
        plt.imshow(obstacle_map, cmap='gray_r', origin='upper', alpha=0.6)

        # 提取路径坐标（处理2D/3D情况）
        if len(path) == 0:
            print("Warning: Empty path in last episode")
            return

        if self.is_3d:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
        else:
            try:
                rows, cols = zip(*[(p[0], p[1]) for p in path])
            except ValueError:
                rows, cols = [], []

        # 绘制路径
        if len(rows) > 1:
            plt.plot(cols, rows,
                     marker='.', markersize=8,
                     linestyle='-', linewidth=1.5,
                     color='dodgerblue', alpha=0.8,
                     label='Agent Path')

        # 标记起终点
        if len(rows) > 0:
            # 起点
            plt.scatter(cols[0], rows[0],
                        s=120, c='limegreen',
                        edgecolors='black', marker='o',
                        label='Start')
            # 终点
            plt.scatter(cols[-1], rows[-1],
                        s=120, c='orangered',
                        edgecolors='black', marker='X',
                        label='End' if rows[-1] != rows[0] else 'Start')

        plt.title('Navigation Trajectory', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        torch.save({
            'online': self.model.state_dict(),
            'target': self.target_model.state_dict(),
            'best_path': self.best_path,  # 新增路径保存
            'best_distance': self.best_distance,
            'best_reward': self.best_reward
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['online'])
        self.target_model.load_state_dict(checkpoint['target'])
        # 恢复路径状态
        self.best_path = checkpoint.get('best_path', None)
        self.best_distance = checkpoint.get('best_distance', float('inf'))
        self.best_reward = checkpoint.get('best_reward', -float('inf'))

    def get_path(self, max_steps=100000):
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # 禁用探索，完全依赖策略网络
        try:
            # 尝试获取完整路径
            path = []
            state = self.env.reset()
            path.append(self.env.current_pos)
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = self.act(state)
                next_state, _, done, _ = self.env.step(action)
                path.append(self.env.current_pos)
                state = next_state
                steps += 1

            # 验证路径是否完成
            final_pos = np.array(path[-1][:2])
            target_pos = np.array(self.env.goal[:2])
            success = np.array_equal(final_pos, target_pos)

            self.epsilon = original_epsilon

            # 返回最佳路径或当前未完成路径
            return {
                # 'path': path if success else self.best_path ,
                'path': path,
                'success': success,
                'final_distance': np.linalg.norm(np.array(final_pos) - np.array(target_pos))
            }
        except Exception as e:
            print(f"路径生成错误: {str(e)}")
            return {
                'path': [],
                'success': False,
                'final_distance': float('inf')
            }

    def plot_loss(self):
        """
        Plot the recorded average training loss per episode.
        """
        if not self.loss_history:
            print("No loss data to plot.")
            return

        plt.figure()
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Training Loss per Episode')
        plt.legend()
        plt.show()

    #########################################
    # Added/Modified Code for Layered 3D    #
    #########################################
def create_layered_obstacle_map(data, metadata, layer_min, layer_max):
    """
    For each grid cell, if building height >= layer_max, mark as obstacle.
    If building height < layer_min, mark as passable.
    If building height is between layer_min and layer_max, mark passable but heavy penalty
    (we'll account for penalty in the path cost or DQN reward).
    For demonstration, we'll store passable with penalty by marking it "2" (some special code).
    """
    nodata = metadata['nodata_value']
    layered_map = np.zeros_like(data, dtype=np.int8)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if val == nodata:
                # treat as obstacle (like original create_obstacle_map)
                layered_map[r, c] = 1
            else:
                # if building height >= layer_max => obstacle
                if val >= layer_max:
                    layered_map[r, c] = 1
                # if building height < layer_min => passable
                elif val < layer_min:
                    layered_map[r, c] = 0
                else:
                    # partial overlap => passable but add "flip penalty"
                    layered_map[r, c] = 2
    return layered_map

def layered_astar_2d(layered_map, start, end):
    """
    We'll adapt the A* for the layered map.
    If layered_map[r,c] == 1 => obstacle, cost infinite
    If layered_map[r,c] == 0 => passable, cost=1
    If layered_map[r,c] == 2 => partial building, cost=1 (we'll add penalty in cost)
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    closed_set = set()
    came_from = {}
    gscore = {start: 0}
    # cost function: if 2 => cost + 2 (some penalty for partial overlap)
    def cost_function(val):
        return 2.0 if val == 2 else 1.0

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    fscore = {start: manhattan(start, end)}
    heap = []
    heappush(heap, (fscore[start], start))

    while heap:
        _, current = heappop(heap)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)
        for dx, dy in neighbors:
            nr, nc = current[0] + dx, current[1] + dy
            if 0 <= nr < layered_map.shape[0] and 0 <= nc < layered_map.shape[1]:
                if layered_map[nr, nc] == 1:
                    continue
                tentative_g = gscore[current] + cost_function(layered_map[nr, nc])
                if (nr, nc) in closed_set and tentative_g >= gscore.get((nr, nc), float('inf')):
                    continue
                if tentative_g < gscore.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    gscore[(nr, nc)] = tentative_g
                    fscore[(nr, nc)] = tentative_g + manhattan((nr, nc), end)
                    heappush(heap, (fscore[(nr, nc)], (nr, nc)))
    return None

class LayeredPathPlanningEnv(PathPlanningEnv):
    """
    Inherits from PathPlanningEnv, but:
    - grid can have 3 states: 0 passable, 1 obstacle, 2 partial building
    - stepping onto 2 => extra penalty
    """
    def __init__(self, layered_map, start, goal):
        super().__init__(
            obstacle_map=layered_map,
            start=start,
            goal=goal,
            is_3d=False,  # We'll treat each layer as a 2D problem
        )
        self.layered_map = layered_map

    def step(self, action):
        # Override step to add penalty on "2" cells
        done = False
        reward = 0
        prev_position = np.array(self.current_pos[:2])
        dx, dy = self.actions_2d[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy

        if not (0 <= new_x < self.grid.shape[0] and 0 <= new_y < self.grid.shape[1]):
            # out of boundary
            reward = -100
            return self.get_state(), reward, done, {}

        cell_val = self.grid[new_x, new_y]
        if cell_val == 1:
            # obstacle
            reward = -100
        else:
            # passable
            self.current_pos = (new_x, new_y)
            # If partial building => flipping penalty
            if cell_val == 2:
                reward -= 15.0  # extra penalty

        new_position = np.array(self.current_pos[:2])
        target_position = np.array(self.goal_pos[:2])

        # distance-based reward
        prev_dist = np.linalg.norm(prev_position - target_position)
        new_dist = np.linalg.norm(new_position - target_position)
        distance_reward = (prev_dist - new_dist) * 5.0
        survival_penalty = -0.5
        reward += (distance_reward + survival_penalty)

        if np.array_equal(new_position, target_position):
            reward += 1000
            done = True

        return self.get_state(), reward, done, {}

############################################################
# Visualization Helpers (from path3, no omissions) + extras
############################################################

def plot_comparison(obstacle_map, all_paths, metadata):
    plt.figure(figsize=(14, 10))
    plt.imshow(obstacle_map, cmap='gray_r', origin='upper', alpha=0.7)

    style_config = {
        ('A* 2D', True): {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        ('A* 3D', True): {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        ('DQN 2D', True): {'color': 'green', 'linestyle': '--', 'linewidth': 2},
        ('DQN 3D', True): {'color': 'purple', 'linestyle': '--.', 'linewidth': 2},
        ('DQN 2D', False): {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
        ('DQN 3D', False): {'color': 'purple', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
        ('Layered A*', True): {'color': 'lightblue', 'linestyle': '-', 'linewidth': 2},
        ('Layered DQN', True): {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
    }

    for path_info in all_paths:
        method_name, path_data, is_3d = path_info

        if path_data is None or not isinstance(path_data, dict) or 'path' not in path_data:
            print(f"警告: {method_name} 路径数据无效")
            continue

        path = path_data.get('path', [])
        success = path_data.get('success', False)

        if len(path) < 2:
            print(f"警告: {method_name} 路径过短")
            continue

        try:
            if is_3d:
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
            else:
                rows, cols = zip(*path)
        except:
            print(f"路径坐标解析失败: {method_name}")
            continue

        style_key = (method_name, success)
        if style_key not in style_config:
            # fallback style
            style_key = ('Layered A*', True)

        plt.plot(cols, rows, label=f'{method_name} {"(success)" if success else "(best)"}',
                 **style_config.get(style_key, {}))

        end_marker = 'o' if success else 'X'
        plt.scatter(cols[-1], rows[-1],
                    c=style_config[style_key]['color'],
                    s=100, marker=end_marker,
                    edgecolors='black', zorder=5)

    plt.legend(title="Method:")
    plt.title('Path Planning Comparison', fontsize=14)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def generate_table(path, metadata, data, is_3d=False):
    table = []
    total_dist = 0.0
    prev_geo = None
    for step, point in enumerate(path):
        row, col = point[0], point[1]
        x, y = grid_to_geo(row, col, metadata)
        z = point[2] if is_3d else data[row, col]
        h = data[row, col]

        if step > 0:
            dx = x - prev_geo[0]
            dy = y - prev_geo[1]
            dz = z - prev_geo[2] if is_3d else 0
            dist = np.sqrt(dx ** 2 + dy ** 2 + (dz ** 2 if is_3d else 0))
            total_dist += dist

        table.append({
            'Step': step + 1,
            'X': x,
            'Y': y,
            'Z': z,
            'Terrain Height': h,
            'Cumulative Distance': total_dist
        })
        prev_geo = (x, y, z) if is_3d else (x, y, data[row, col])

    return table

def export_csv(table, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)

############################################################
# New Layered DQN Example Class (Optional demonstration)   #
############################################################

class LayeredDQNEnv(PathPlanningEnv):
    """
    Similar approach as LayeredPathPlanningEnv but we keep the DQN style environment.
    If grid cell is 2 => partial building => flipping penalty
    """
    def __init__(self, layered_map, start, goal):
        super().__init__(
            obstacle_map=layered_map,
            start=start,
            goal=goal,
            is_3d=False
        )

    def step(self, action):
        done = False
        reward = 0
        prev_position = np.array(self.current_pos[:2])

        dx, dy = self.actions_2d[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy

        if not (0 <= new_x < self.grid.shape[0] and 0 <= new_y < self.grid.shape[1]):
            reward = -100
            return self.get_state(), reward, done, {}

        cell_val = self.grid[new_x, new_y]
        if cell_val == 1:
            reward = -100
        else:
            self.current_pos = (new_x, new_y)
            if cell_val == 2:
                reward -= 15.0

        new_position = np.array(self.current_pos[:2])
        target_position = np.array(self.goal_pos[:2])
        prev_dist = np.linalg.norm(prev_position - target_position)
        new_dist = np.linalg.norm(new_position - target_position)
        distance_reward = (prev_dist - new_dist) * 5.0
        survival_penalty = -0.5
        reward += distance_reward + survival_penalty

        if np.array_equal(new_position, target_position):
            reward += 1000
            done = True

        return self.get_state(), reward, done, {}
