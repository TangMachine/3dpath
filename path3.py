import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import csv
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from collections import deque
import os
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


def astar_2d(grid, start, end):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    closed_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
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
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == 1:
                    continue
                tentative_g = gscore[current] + 1
                if (nr, nc) in closed_set and tentative_g >= gscore.get((nr, nc), float('inf')):
                    continue
                if tentative_g < gscore.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    gscore[(nr, nc)] = tentative_g
                    fscore[(nr, nc)] = tentative_g + abs(nr - end[0]) + abs(nc - end[1])
                    heappush(heap, (fscore[(nr, nc)], (nr, nc)))
    return None


def astar_3d(data, metadata, start, end, delta_z):
    def heuristic(a, b):
        dx = (a[1] - b[1]) * metadata['cellsize']
        dy = (a[0] - b[0]) * metadata['cellsize']
        dz = a[2] - b[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    open_heap = []
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
    heappush(open_heap, (fscore[start], gscore[start], start))  # (f, g, node)
    came_from = {}

    while open_heap:
        current_f, current_g, current = heappop(open_heap)

        # Lazy A*: Skip if this node has been improved
        if current_g > gscore.get(current, float('inf')):
            continue

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # Generate neighbors with simplified cost calculation
        neighbors = []
        # Horizontal moves (4 directions)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row = current[0] + dx
            new_col = current[1] + dy
            if 0 <= new_row < data.shape[0] and 0 <= new_col < data.shape[1]:
                # Allow movement only if terrain height <= current z
                if data[new_row, new_col] <= current[2]:
                    neighbors.append(((new_row, new_col, current[2]), metadata['cellsize']))

        # Vertical moves (up/down)
        for dz in [delta_z, -delta_z]:
            new_z = current[2] + dz
            # Ensure altitude doesn't go below terrain
            if new_z >= data[current[0], current[1]]:
                neighbors.append(((current[0], current[1], new_z), abs(dz)))

        for neighbor, move_cost in neighbors:
            tentative_g = current_g + move_cost
            # Update if this path is better
            if tentative_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore_neighbor = tentative_g + heuristic(neighbor, end)
                heappush(open_heap, (fscore_neighbor, tentative_g, neighbor))

    return None


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
            if self.grid[new_x, new_y] == 0:
                if self.is_3d:
                    if self.elevation_data[new_x, new_y] <= new_z:
                        self.current_pos = (new_x, new_y, new_z)
                    else:
                        reward = -10  # 高度不足惩罚
                        return self.get_state(), reward, done, {}
                else:
                    self.current_pos = (new_x, new_y)
            else:
                reward = -10  # 障碍物惩罚
        else:
            reward = -10  # 越界惩罚

        #改进奖励设计
        new_position = np.array(self.current_pos[:2])
        target_position = np.array(self.goal_pos[:2])

        # 距离变化奖励
        prev_dist = np.linalg.norm(prev_position - target_position)
        new_dist = np.linalg.norm(new_position - target_position)
        distance_reward = (prev_dist - new_dist) * 5  # 强化距离缩短奖励

        # 基础生存奖励
        survival_penalty = -0.2

        reward = distance_reward + survival_penalty

        # 终点奖励
        if np.array_equal(new_position, target_position):
            reward += 500
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
        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
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


    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes=500, render_interval=50):
        rewards_history = []
        steps_history = []
        last_episode_rewards = []  # 新增：记录最后回合的奖励
        last_episode_path = []  # 新增：记录最后回合的路径


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

            while not done and steps < 10000:  # 最大步数限制
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                if ep == episodes - 1:
                    last_episode_rewards.append(reward)
                    last_episode_path.append(self.env.current_pos)
                    cumulative_rewards = np.cumsum(last_episode_rewards)
                self.remember(state, action, reward, next_state, done)
                if total_reward > best_reward:
                    best_reward = total_reward
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > 20:  # 连续20回合无改进
                    self.epsilon = min(0.5, self.epsilon + 0.1)  # 重新激活探索
                    no_improve_count = 0
                self.replay()
                state = next_state
                total_reward += reward
                steps += 1
                current_path.append(self.env.current_pos)

            # 更新最佳路径（基于距离）
            final_pos = np.array(current_path[-1][:2])
            target_pos = np.array(self.env.goal[:2])
            current_distance = np.linalg.norm(final_pos - target_pos)

            if ep == episodes - 1:
                print("\nFinal Episode Analysis:")
                print(f"Total Steps: {steps}")
                print(f"Final Reward: {total_reward:.1f}")
                self.plot_last_episode(last_episode_rewards, last_episode_path, self.env.grid)

            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_path = current_path.copy()
                self.save_model(f'best_{"3d" if self.is_3d  else "2d"}_path_model.pth')



                # 或者基于奖励更新（根据需求选择其一）
            # if total_reward > self.best_reward:
            #     self.best_reward = total_reward
            #     self.best_path = current_path.copy()


            rewards_history.append(total_reward)
            steps_history.append(steps)
            self.update_target()
            # 进度显示
            print(f"Episode {ep + 1}/{episodes}, Reward: {total_reward:.1f}, "
                  f"Steps: {steps}, Epsilon: {self.epsilon:.3f}")
            if ep % render_interval == 0 or ep == episodes - 1:
                print(f"Episode {ep + 1}/{episodes}, Reward: {total_reward:.1f}, "
                      f"Steps: {steps}, Epsilon: {self.epsilon:.3f}")
                self.plot_progress(rewards_history, steps_history)

        return rewards_history, steps_history

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
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        # 添加 weights_only=True 参数
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.target_model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    def get_path(self, max_steps=50000):
        original_epsilon = self.epsilon
        self.epsilon = 0  # 禁用探索，完全依赖策略网络
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
            final_pos = path[-1][:2]
            target_pos = self.env.goal[:2]
            success = np.array_equal(final_pos, target_pos)

            self.epsilon = original_epsilon

        # 返回最佳路径或当前未完成路径
            return {
                'path': path if success else self.best_path or [],
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


# 修改后的可视化函数
def plot_comparison(obstacle_map, all_paths, metadata):
    plt.figure(figsize=(14, 10))
    plt.imshow(obstacle_map, cmap='gray_r', origin='upper', alpha=0.7)

    # 样式配置（添加未完成路径样式）
    style_config = {
        ('A* 2D', True): {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        ('A* 3D', True): {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        ('DQN 2D', True): {'color': 'green', 'linestyle': '--', 'linewidth': 2},
        ('DQN 3D', True): {'color': 'purple', 'linestyle': '--.', 'linewidth': 2},
        ('DQN 2D', False): {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
        ('DQN 3D', False): {'color': 'purple', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7}
    }

    for path_info in all_paths:
        method_name, path_data, is_3d = path_info

        # 添加空路径检查
        if path_data is None or not isinstance(path_data, dict) or 'path' not in path_data:
            print(f"警告: {method_name} 路径数据无效")
            continue

        path = path_data.get('path', [])
        success = path_data.get('success', False)

        # 添加路径长度检查
        if len(path) < 2:
            print(f"警告: {method_name} 路径过短")
            continue

        path = path_data['path']
        success = path_data['success']
        style_key = (method_name, success)

        # 提取坐标
        try:
            if is_3d:
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                z_values = [p[2] for p in path]
            else:
                rows, cols = zip(*path)
        except:
            print(f"路径坐标解析失败: {method_name}")
            continue

        # 绘制路径
        plt.plot(cols, rows,
                 label=f'{method_name} {"(success)" if success else "(best)"}',
                 **style_config.get(style_key, {}))

        # 标记终点
        end_marker = 'o' if success else 'X'
        plt.scatter(cols[-1], rows[-1],
                    c=style_config[style_key]['color'],
                    s=100, marker=end_marker,
                    edgecolors='black', zorder=5)

    # 可视化增强
    plt.legend(title="Y:")
    plt.title('PathPlaning', fontsize=14)
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


if __name__ == "__main__":
    try:
        data, metadata = read_asc_file('output_dem.asc')
        obstacle_map = create_obstacle_map(data, metadata)

        # 定义起终点
        start_geo = (metadata['xllcorner'] + 100 * metadata['cellsize'],
                     metadata['yllcorner'] + (metadata['nrows'] - 200 - 1) * metadata['cellsize'])
        end_geo = (metadata['xllcorner'] + 500 * metadata['cellsize'],
                   metadata['yllcorner'] + (metadata['nrows'] - 400 - 1) * metadata['cellsize'])
        start_row, start_col = geo_to_grid(*start_geo, metadata)
        end_row, end_col = geo_to_grid(*end_geo, metadata)
        print(f"Start: {start_row}, {start_col} ")
        # 原有方法路径
        print("2D路径规划中...")
        path_2d = astar_2d(obstacle_map, (start_row, start_col), (end_row, end_col))
        delta_z = 6
        start_3d = (start_row, start_col, data[start_row, start_col] + delta_z)
        end_3d = (end_row, end_col, data[end_row, end_col] + delta_z)
        print("3D路径规划中...")
        path_3d = astar_3d(data, metadata, start_3d, end_3d, delta_z)

        dtrain = True
        # DQN路径规划
        if dtrain:
            def train_dqn_agent(is_3d=False):
                if is_3d:
                    env = PathPlanningEnv(obstacle_map, start_3d, end_3d,
                                      is_3d=True, elevation_data=data, delta_z=delta_z)
                else:
                    env = PathPlanningEnv(obstacle_map, (start_row, start_col), (end_row, end_col))

                agent = DQNAgent(env, is_3d=is_3d)
                print(f"Training {'3D' if is_3d else '2D'} DQN agent...")
                agent.train(episodes=50)
                agent.save_model(f'dqn_agent_{"3d" if is_3d else "2d"}.pth')
                return agent.get_path()

            dqn_2d_result = train_dqn_agent(is_3d=False)
            dqn_3d_result = train_dqn_agent(is_3d=True)

        else:
            # 加载预训练模型
            def load_dqn_agent(is_3d=False):
                if is_3d:
                    env = PathPlanningEnv(obstacle_map, start_3d, end_3d,
                                      is_3d=True, elevation_data=data, delta_z=delta_z)
                else:
                    env = PathPlanningEnv(obstacle_map, (start_row, start_col), (end_row, end_col))

                agent = DQNAgent(env, is_3d=is_3d)
                # agent.load_model(f'dqn_agent_{"3d" if is_3d else "2d"}.pth')
                model_path = f'best_{"3d" if is_3d else "2d"}_path_model.pth'
                # 添加文件存在检查
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件 {model_path} 不存在")
                # 添加模型校验
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    print(f"成功加载模型: {model_path}")
                    print(f"模型参数键: {checkpoint.keys()}")
                except Exception as e:
                    print(f"模型加载失败: {str(e)}")
                agent.load_model(model_path)
                return agent.get_path()
            dqn_2d_result = load_dqn_agent(is_3d=False)
            dqn_3d_result = load_dqn_agent(is_3d=True)


        # 统一路径数据结构
        all_paths = [
            ('A* 2D', {'path': path_2d, 'success': True}, False),
            ('A* 3D', {'path': path_3d, 'success': True}, True),
            ('DQN 2D', dqn_2d_result, False),
            ('DQN 3D', dqn_3d_result, True)
        ]

        # 调用可视化
        plot_comparison(obstacle_map, all_paths, metadata)
        # # 生成表格数据
        # tables = []
        # if path_2d:
        #     tables.append(('A* 2D', generate_table(path_2d, metadata, data)))
        # if path_3d:
        #     tables.append(('A* 3D', generate_table(path_3d, metadata, data, is_3d=True)))
        # if dqn_2d_result:
        #     tables.append(('DQN 2D', generate_table(dqn_2d_result, metadata, data)))
        # if dqn_3d_result:
        #     tables.append(('DQN 3D', generate_table(dqn_3d_result, metadata, data, is_3d=True)))
        #
        # # 导出CSV
        # i=0
        # for name, table in tables:
        #     i+=1
        #     export_csv(table, f'{i}_path.csv')


    except Exception as e:
        print(f"Error: {str(e)}")