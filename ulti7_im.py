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
# Modified Classes with Combined Prior Knowledge Input #
########################################################

class ImprovedPathPlanningEnv:
    def __init__(self, obstacle_map, start, goal, is_3d=False, elevation_data=None, delta_z=5, max_z=150):
        self.grid = obstacle_map
        self.start = start
        self.goal = goal
        self.is_3d = is_3d
        self.elevation_data = elevation_data
        self.delta_z = delta_z
        self.current_pos = start
        self.goal_pos = goal
        self.max_z = max_z
        self.collision_count = 0
        self.episode_collisions = 0

        # 获取地图尺寸
        self.map_height, self.map_width = obstacle_map.shape

        # 初始化两张表格：当前位置表和先验知识表（障碍物+目标位置）
        self.current_position_table = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.prior_knowledge_table = np.zeros((self.map_height, self.map_width), dtype=np.float32)

        # 将障碍物和目标合并到先验知识表中
        # 障碍物标记为负值，目标标记为正值
        self.prior_knowledge_table = -1.0 * self.grid.astype(np.float32)  # 障碍物标记为 -1

        # 设置目标位置（标记为更高的正值）
        if self.is_3d:
            self.prior_knowledge_table[goal[0], goal[1]] = 15.0
        else:
            self.prior_knowledge_table[goal[0], goal[1]] = 5.0

        if self.is_3d:
            self.actions_3d = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if not (dx == 0 and dy == 0 and dz == 0):
                            if dz == 0:
                                self.actions_3d.append((dx, dy, 0))
                            if dz != 0 and dx == 0 and dy == 0:
                                self.actions_3d.append((0, 0, dz))
            self.num_actions = len(self.actions_3d)
        else:
            # 8 个方向: 上下左右 + 对角
            self.actions_2d = [
                (-1, 0),  # 上
                (1, 0),  # 下
                (0, -1),  # 左
                (0, 1),  # 右
                (-1, -1),  # 左上
                (-1, 1),  # 右上
                (1, -1),  # 左下
                (1, 1)  # 右下
            ]
            self.num_actions = len(self.actions_2d)

    def reset(self):
        self.current_pos = self.start

        # 重置当前位置表
        self.current_position_table.fill(0.0)
        self.episode_collisions = 0
        if self.is_3d:
            self.current_position_table[self.current_pos[0], self.current_pos[1]] = self.current_pos[2]
        else:
            self.current_position_table[self.current_pos[0], self.current_pos[1]] = 1.0

        return self.get_state()

    def get_state(self):
        """
        返回两张表格的组合作为状态
        形状: (2, map_height, map_width)
        """
        state = np.stack([
            self.current_position_table,
            self.prior_knowledge_table
        ], axis=0)
        return state

    def get_valid_actions(self):
        """
        根据当前位置和先验知识，返回有效的action列表
        """
        valid_actions = []

        if self.is_3d:
            x, y, z = self.current_pos

            for action_idx, (dx, dy, dz) in enumerate(self.actions_3d):
                nx, ny, nz = x + dx, y + dy, z + dz * self.delta_z

                # 检查边界
                if not (0 <= nx < self.map_height and 0 <= ny < self.map_width):
                    continue

                # 检查高度限制
                if nz > self.max_z:
                    continue

                # 检查障碍物（负值表示障碍物）
                if self.prior_knowledge_table[nx, ny] < 0:
                    # 如果是3D环境，检查是否可以飞越障碍物
                    if self.elevation_data is not None:
                        obstacle_height = self.elevation_data[nx, ny]
                        if obstacle_height + 5 > nz:  # 如果高度不够飞越
                            continue
                    else:
                        continue  # 没有高程数据时，不能通过障碍物

                # 如果通过所有检查，这个action是有效的
                valid_actions.append(action_idx)
        else:
            x, y = self.current_pos

            for action_idx, (dx, dy) in enumerate(self.actions_2d):
                nx, ny = x + dx, y + dy

                # 检查边界
                if not (0 <= nx < self.map_height and 0 <= ny < self.map_width):
                    continue

                # 检查障碍物（负值表示障碍物）
                if self.prior_knowledge_table[nx, ny] < 0:
                    continue  # 2D环境中不能通过障碍物

                # 如果通过所有检查，这个action是有效的
                valid_actions.append(action_idx)

        # 如果没有有效的action（被困），返回所有action让环境处理惩罚
        if not valid_actions:
            return list(range(self.num_actions))

        return valid_actions

    def step(self, action):
        done = False
        reward = 0
        prev_position = np.array(self.current_pos[:2])
        collision_occurred = False

        if self.is_3d:
            dx, dy, dz = self.actions_3d[action]
            if dz != 0:
                reward -= 5
            x, y, z = self.current_pos
            nx, ny, nz = x + dx, y + dy, z + dz * self.delta_z
            if nz > self.max_z:
                nz = self.max_z
                reward -= 20
        else:
            dx, dy = self.actions_2d[action]
            x, y = self.current_pos
            nx, ny = x + dx, y + dy

        # 边界检查
        if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
            if self.grid[nx, ny] == 1:
                if self.is_3d:
                    if self.elevation_data[nx, ny] + 5 <= nz:
                        # 更新当前位置表
                        self.current_position_table.fill(0.0)
                        self.current_position_table[nx, ny] = 1.0
                        self.current_pos = (nx, ny, nz)
                    else:
                        reward = -100
                        collision_occurred = True
                else:
                    reward = -100
                    collision_occurred = True
            elif self.grid[nx, ny] == 3:
                if self.is_3d:
                    if self.elevation_data[nx, ny] + 10 <= nz:
                        # 更新当前位置表
                        self.current_position_table.fill(0.0)
                        self.current_position_table[nx, ny] = 1.0
                        self.current_pos = (nx, ny, nz)
                    else:
                        reward = -5
                        # 更新当前位置表
                        self.current_position_table.fill(0.0)
                        self.current_position_table[nx, ny] = 1.0
                        self.current_pos = (nx, ny, nz)
                else:
                    reward = -5
                    # 更新当前位置表
                    self.current_position_table.fill(0.0)
                    self.current_position_table[nx, ny] = 1.0
                    self.current_pos = (nx, ny)
            else:
                # 更新当前位置表
                self.current_position_table.fill(0.0)
                self.current_position_table[nx, ny] = 1.0
                self.current_pos = (nx, ny, nz) if self.is_3d else (nx, ny)
        else:
            reward = -100
            collision_occurred = True

        # 改进奖励设计
        new_position = np.array(self.current_pos[:2])
        target_position = np.array(self.goal_pos[:2])

        # 距离变化奖励
        prev_dist = np.linalg.norm(prev_position - target_position)
        new_dist = np.linalg.norm(new_position - target_position)
        distance_reward = (prev_dist - new_dist) * 5

        # 基础生存奖励
        survival_penalty = -0.5

        reward = distance_reward + survival_penalty

        # 终点奖励
        if np.array_equal(new_position, target_position):
            reward += 1000
            done = True
        if collision_occurred:
            self.collision_count += 1
            self.episode_collisions += 1

        return self.get_state(), reward, done, {'collision': collision_occurred}


class ImprovedDQN(nn.Module):
    def __init__(self, map_height, map_width, output_dim, is_3d=False):
        super(ImprovedDQN, self).__init__()

        # 卷积层处理三张表格输入 (2, map_height, map_width)
        self.conv_layers = nn.Sequential(
            # 第一层卷积: 2通道 -> 16通道
            nn.Conv2d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积: 16通道 -> 32通道
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积: 32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层卷积: 64通道 -> 128通道
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化到1x1
        )

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x的形状应该是 (batch_size, 2, map_height, map_width)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x



class ImprovedDQNAgent:
    def __init__(self, env, is_3d=False):
        self.env = env
        self.is_3d = is_3d
        self.map_height = env.map_height
        self.map_width = env.map_width
        self.action_dim = env.num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建网络
        self.model = ImprovedDQN(self.map_height, self.map_width, self.action_dim, is_3d).to(self.device)
        self.target_model = ImprovedDQN(self.map_height, self.map_width, self.action_dim, is_3d).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64  # 减小batch size因为状态空间更大
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.best_path = None
        self.best_distance = float('inf')
        self.best_reward = -float('inf')
        self.loss_history = []
        self.total_steps = 0
        self.target_update_freq = 5000

    def act(self, state):
        # 获取当前状态下的有效actions
        valid_actions = self.env.get_valid_actions()

        if np.random.rand() < self.epsilon:
            # 随机选择一个有效的action
            return np.random.choice(valid_actions)

        # 确保状态的形状正确 (1, 2, map_height, map_width)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # 只考虑有效actions的Q值
        valid_q_values = q_values[0][valid_actions]
        best_valid_action_idx = valid_q_values.argmax().item()

        # 返回有效actions中Q值最大的action
        return valid_actions[best_valid_action_idx]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
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
        success_count = 0
        last_episode_rewards = []
        last_episode_path = []
        collision_history = []
        filtered_action_stats = []

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
            total_collisions = 0
            filtered_action_count = 0

            while not done and steps < 10000:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                # 记录碰撞
                if info.get('collision', False):
                    total_collisions += 1

                if ep == episodes - 1:
                    last_episode_rewards.append(reward)
                    last_episode_path.append(self.env.current_pos)

                self.remember(state, action, reward, next_state, done)

                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # if no_improve_count > 20:
                #     self.epsilon = min(0.5, self.epsilon + 0.1)
                #     no_improve_count = 0

                loss_value = self.replay()
                state = next_state
                total_reward += reward
                steps += 1
                self.total_steps += 1
                current_path.append(self.env.current_pos)

                if loss_value > 0:
                    episode_losses.append(loss_value)

                if self.total_steps % self.target_update_freq == 0:
                    self.update_target()

            # 检查是否成功到达目标
            final_pos = np.array(current_path[-1][:2])
            target_pos = np.array(self.env.goal[:2])
            current_distance = np.linalg.norm(final_pos - target_pos)
            collision_history.append(total_collisions)
            filtered_action_stats.append(filtered_action_count / steps if steps > 0 else 0)

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
                    self.save_model(f'improved_filtered_best_{layer_min}_{layer_max}_path_model.pth')
                else:
                    self.save_model(f'improved_filtered_best_{"3d" if self.is_3d else "2d"}_path_model.pth')

            rewards_history.append(total_reward)
            steps_history.append(steps)

            # 每10个episode更新一次target网络


            success_rate = (success_count / (ep + 1)) * 100
            print(f"回合 {ep + 1}/{episodes}, 奖励: {total_reward:.1f}, "
                  f"步数: {steps}, 碰撞次数: {total_collisions}, "
                  f"动作过滤率: {filtered_action_stats[-1]:.2%}, "
                  f"Epsilon: {self.epsilon:.3f}, "
                  f"成功率: {success_rate:.1f}%")

            if ep % render_interval == 0 or ep == episodes - 1:
                self.plot_progress(rewards_history, steps_history)

        final_success_rate = (success_count / episodes) * 100
        self.plot_collision_stats(collision_history, filtered_action_stats)

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

        # 左图：累计奖励趋势
        plt.subplot(1, 2, 1)
        cumulative_rewards = np.cumsum(step_rewards)
        main_line, = plt.plot(cumulative_rewards,
                              color='#2C5F8D',
                              linewidth=1.5,
                              alpha=0.8,
                              label='Total Reward')

        gradients = np.diff(cumulative_rewards)
        max_gradient_idx = np.argmax(gradients) + 1
        plt.scatter(max_gradient_idx, cumulative_rewards[max_gradient_idx],
                    color='#EE7621', s=80, zorder=5,
                    label='Max Reward Rate')

        plt.axhline(y=cumulative_rewards[-1],
                    color='#7C878E', linestyle='--',
                    linewidth=1, alpha=0.7,
                    label=f'Final: {cumulative_rewards[-1]:.1f}')

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
        plt.imshow(obstacle_map, cmap='gray_r', origin='upper', alpha=0.6)

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

        if len(rows) > 1:
            plt.plot(cols, rows,
                     marker='.', markersize=8,
                     linestyle='-', linewidth=1.5,
                     color='dodgerblue', alpha=0.8,
                     label='Agent Path')

        if len(rows) > 0:
            plt.scatter(cols[0], rows[0],
                        s=120, c='limegreen',
                        edgecolors='black', marker='o',
                        label='Start')
            plt.scatter(cols[-1], rows[-1],
                        s=120, c='orangered',
                        edgecolors='black', marker='X',
                        label='End' if rows[-1] != rows[0] else 'Start')

        plt.title('Navigation Trajectory', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_collision_stats(self, collisions, filter_rates):
        plt.figure(figsize=(12, 8))

        # 碰撞次数图表
        plt.subplot(2, 1, 1)
        plt.plot(collisions, 'r-', label='碰撞次数')
        plt.fill_between(range(len(collisions)),
                         collisions,
                         color='red', alpha=0.1)
        plt.ylabel('碰撞次数')
        plt.title('每回合碰撞次数统计')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加移动平均线
        window_size = max(1, len(collisions) // 20)
        moving_avg = np.convolve(collisions, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(collisions)), moving_avg,
                 'b--', linewidth=2, label=f'移动平均 ({window_size}回合)')
        plt.legend()

        # 动作过滤率图表
        plt.subplot(2, 1, 2)
        plt.plot(filter_rates, 'g-', label='动作过滤率')
        plt.fill_between(range(len(filter_rates)),
                         filter_rates,
                         color='green', alpha=0.1)
        plt.xlabel('回合数')
        plt.ylabel('过滤率')
        plt.title('有效动作过滤比例')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加移动平均线
        moving_avg_filter = np.convolve(filter_rates, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(filter_rates)), moving_avg_filter,
                 'b--', linewidth=2, label=f'移动平均 ({window_size}回合)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        torch.save({
            'online': self.model.state_dict(),
            'target': self.target_model.state_dict(),
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'best_reward': self.best_reward
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['online'])
        self.target_model.load_state_dict(checkpoint['target'])
        self.best_path = checkpoint.get('best_path', None)
        self.best_distance = checkpoint.get('best_distance', float('inf'))
        self.best_reward = checkpoint.get('best_reward', -float('inf'))

    def get_path(self, max_steps=100000):
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        try:
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

            final_pos = np.array(path[-1][:2])
            target_pos = np.array(self.env.goal[:2])
            success = np.array_equal(final_pos, target_pos)

            self.epsilon = original_epsilon

            return {
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

    def visualize_state_tables(self, state=None):
        """
        可视化两张状态表格
        """
        if state is None:
            state = self.env.get_state()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 当前位置表
        axes[0].imshow(state[0], cmap='Reds', origin='upper')
        axes[0].set_title('Current Position Table')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')

        # 先验知识表（障碍物+目标）
        im = axes[1].imshow(state[1], cmap='coolwarm', origin='upper')
        axes[1].set_title('Prior Knowledge (Obstacles + Goal)')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')

        # 添加颜色条
        cbar = fig.colorbar(im, ax=axes[1])
        cbar.set_label('Value (Negative=Obstacle, Positive=Goal)')

        plt.tight_layout()
        plt.show()

    def visualize_action_filtering(self, state=None):
        """
        可视化action过滤过程
        """
        if state is None:
            state = self.env.get_state()

        valid_actions = self.env.get_valid_actions()

        print(f"\n=== Action过滤分析 ===")
        print(f"当前位置: {self.env.current_pos}")
        print(f"总action数量: {self.env.num_actions}")
        print(f"有效action数量: {len(valid_actions)}")
        print(f"有效actions: {valid_actions}")

        if self.is_3d:
            print("\n3D Actions详情:")
            for i, (dx, dy, dz) in enumerate(self.env.actions_3d):
                status = "✓" if i in valid_actions else "✗"
                print(f"  Action {i}: ({dx:2d}, {dy:2d}, {dz:2d}) {status}")
        else:
            print("\n2D Actions详情:")
            for i, (dx, dy) in enumerate(self.env.actions_2d):
                status = "✓" if i in valid_actions else "✗"
                print(f"  Action {i}: ({dx:2d}, {dy:2d}) {status}")

        # 可视化当前位置周围的障碍物
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # 显示先验知识表
        prior_knowledge_view = state[1].copy()

        # 标记当前位置
        if self.is_3d:
            curr_row, curr_col = self.env.current_pos[0], self.env.current_pos[1]
        else:
            curr_row, curr_col = self.env.current_pos

        # 在当前位置周围5x5区域显示
        view_size = 5
        row_start = max(0, curr_row - view_size // 2)
        row_end = min(self.env.map_height, curr_row + view_size // 2 + 1)
        col_start = max(0, curr_col - view_size // 2)
        col_end = min(self.env.map_width, curr_col + view_size // 2 + 1)

        local_view = prior_knowledge_view[row_start:row_end, col_start:col_end]

        im = ax.imshow(local_view, cmap='coolwarm', origin='upper')
        fig.colorbar(im, ax=ax, label='Value (Negative=Obstacle, Positive=Goal)')

        # 标记当前位置
        local_curr_row = curr_row - row_start
        local_curr_col = curr_col - col_start
        ax.scatter(local_curr_col, local_curr_row, c='blue', s=200, marker='o', label='Current Position')

        # 标记可能的下一步位置
        if self.is_3d:
            actions = self.env.actions_3d
        else:
            actions = self.env.actions_2d

        for i, action in enumerate(actions):
            if self.is_3d:
                dx, dy, dz = action
            else:
                dx, dy = action

            next_row = curr_row + dx
            next_col = curr_col + dy

            # 检查是否在本地视图范围内
            if (row_start <= next_row < row_end and col_start <= next_col < col_end):
                local_next_row = next_row - row_start
                local_next_col = next_col - col_start

                if i in valid_actions:
                    ax.scatter(local_next_col, local_next_row, c='green', s=100, marker='s', alpha=0.7)
                else:
                    ax.scatter(local_next_col, local_next_row, c='red', s=100, marker='x', alpha=0.7)

        ax.set_title(f'Local Action View (Current: {curr_row}, {curr_col})')
        ax.legend(['Current Position', 'Valid Next', 'Invalid Next'])
        plt.show()


#########################################
# Layered 3D classes (updated)         #
#########################################

def create_layered_obstacle_map(data, metadata, layer_min, layer_max):
    nodata = metadata['nodata_value']
    layered_map = np.zeros_like(data, dtype=np.int8)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if val == nodata:
                layered_map[r, c] = 1
            else:
                if val >= layer_max:
                    layered_map[r, c] = 1
                elif val < layer_min:
                    layered_map[r, c] = 0
                else:
                    layered_map[r, c] = 2
    return layered_map


class ImprovedLayeredDQNEnv(ImprovedPathPlanningEnv):
    """
    继承改进的路径规划环境，针对分层处理
    """

    def __init__(self, layered_map, start, goal):
        super().__init__(
            obstacle_map=layered_map,
            start=start,
            goal=goal,
            is_3d=False
        )

        # 重新初始化先验知识表以适应分层
        self.prior_knowledge_table = np.zeros((self.map_height, self.map_width), dtype=np.float32)

        # 设置不同类型区域的值
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if self.grid[r, c] == 1:  # 障碍物
                    self.prior_knowledge_table[r, c] = -1.0
                elif self.grid[r, c] == 2:  # 部分建筑物
                    self.prior_knowledge_table[r, c] = -0.5
                # 其他区域保持为0

        # 设置目标位置
        self.prior_knowledge_table[goal[0], goal[1]] = 5.0

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
            # 更新当前位置表
            self.current_position_table.fill(0.0)
            self.current_position_table[new_x, new_y] = 1.0
            self.current_pos = (new_x, new_y)

            if cell_val == 2:
                reward -= 15.0  # 部分建筑物惩罚

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


############################################################
# Visualization Helpers                                   #
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
        ('Improved DQN 2D', True): {'color': 'cyan', 'linestyle': '-', 'linewidth': 2},
        ('Improved DQN 3D', True): {'color': 'magenta', 'linestyle': '-', 'linewidth': 2},
        ('Improved DQN 2D', False): {'color': 'cyan', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7},
        ('Improved DQN 3D', False): {'color': 'magenta', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7},
        ('Improved Filtered DQN 2D', True): {'color': 'lime', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered DQN 3D', True): {'color': 'gold', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered DQN 2D', False): {'color': 'lime', 'linestyle': '-', 'linewidth': 3, 'alpha': 0.7},
        ('Improved Filtered DQN 3D', False): {'color': 'gold', 'linestyle': '-', 'linewidth': 3, 'alpha': 0.7},
        ('Layered A*', True): {'color': 'lightblue', 'linestyle': '-', 'linewidth': 2},
        ('Layered DQN', True): {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 30-40m', True): {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 30-40m', False): {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 40-50m', True): {'color': 'green', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 40-50m', False): {'color': 'green', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 50-60m', True): {'color': 'purple', 'linestyle': '-', 'linewidth': 2},
        ('Improved Layered DQN 50-60m', False): {'color': 'purple', 'linestyle': '-', 'linewidth': 2},
        ('Improved Filtered Layered DQN 30-40m', True): {'color': 'darkorange', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered Layered DQN 30-40m', False): {'color': 'darkorange', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered Layered DQN 40-50m', True): {'color': 'darkgreen', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered Layered DQN 40-50m', False): {'color': 'darkgreen', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered Layered DQN 50-60m', True): {'color': 'darkviolet', 'linestyle': '-', 'linewidth': 3},
        ('Improved Filtered Layered DQN 50-60m', False): {'color': 'darkviolet', 'linestyle': '-', 'linewidth': 3},
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
            style_key = ('Layered A*', True)

        plt.plot(cols, rows, label=f'{method_name} {"(success)" if success else "(best)"}',
                 **style_config.get(style_key, {}))

        end_marker = 'o' if success else 'X'
        plt.scatter(cols[-1], rows[-1],
                    c=style_config[style_key]['color'],
                    s=100, marker=end_marker,
                    edgecolors='black', zorder=5)

    plt.legend(title="Method:")
    plt.title('Path Planning Comparison (Improved DQN with Action Filtering)', fontsize=14)
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


# 保持原有的类以便兼容
PathPlanningEnv = ImprovedPathPlanningEnv
DQNAgent = ImprovedDQNAgent
LayeredDQNEnv = ImprovedLayeredDQNEnv