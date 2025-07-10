import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 常量定义（基于论文简化）
C_F = 50  # 旅客平均延误损失（元/分钟·人），论文中设为50
MAX_ITERATIONS = 10000  # 最大训练迭代次数
BATCH_SIZE = 32  # PPO批次大小
GAMMA = 0.99  # 折扣因子
EPSILON = 0.2  # PPO裁剪参数
LEARNING_RATE = 0.001  # 学习率
HIDDEN_SIZE = 256  # 神经网络隐藏层大小
SAVE_MODEL_PATH = "flight_recovery_model.pt"  # 模型保存路径


# 数据加载和预处理函数
def load_data(excel_path):
    """
    从Excel加载数据，并转换为适合RL的格式。
    输入：Excel路径，列：[航班序号, 航班人数, 原定起飞时间, 恢复起飞时间集]
    返回：航班列表（每个航班为字典），恢复时间列表（固定时间点）
    """
    df = pd.read_excel(excel_path)
    flights = []
    recovery_times = []

    for idx, row in df.iterrows():
        flight_id = int(row[0])
        passengers = int(row[1])
        # scheduled_time = pd.to_datetime(row[2]).timestamp() / 60  # 转换为分钟（数值）
        scheduled_time = int(row[2])
        # 恢复时间集（假设第四列是列表或字符串，如"21:00,21:05,..."）
        recovery_time = int(row[3])


        flights.append({
            'id': flight_id,
            'passengers': passengers,
            'scheduled_time': scheduled_time
        })
        recovery_times.append(recovery_time)   # 所有航班共享同一恢复时间集（排序后分配）

    # 恢复时间集排序（升序）
    recovery_times_sorted = sorted(recovery_times)
    return flights, recovery_times_sorted


# 定义MDP环境
class FlightRecoveryEnv:
    def __init__(self, flights, recovery_times):
        self.flights = flights  # 初始航班列表（未排序）
        self.recovery_times = recovery_times  # 固定恢复时间集（升序）
        self.n_flights = len(flights)
        self.reset()

    def reset(self):
        """重置环境：随机初始排序，并分配恢复时间"""
        self.current_order = list(range(self.n_flights))
        random.shuffle(self.current_order)  # 随机初始排序
        self.assigned_times = self._assign_times(self.current_order)  # 分配恢复时间
        return self._get_state()

    def _assign_times(self, order):
        """根据排序顺序分配恢复时间：order[i] 的航班分配到 recovery_times[i]"""
        return [self.recovery_times[i] for i in order]

    def _get_state(self):
        """获取当前状态：航班特征矩阵（n_flights x 3维：人数、原定时间、当前分配的恢复时间）"""
        state = []
        for idx in self.current_order:
            flight = self.flights[idx]
            state.append([
                flight['passengers'],
                flight['scheduled_time'],
                self.assigned_times[idx]  # 当前分配的恢复时间
            ])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        执行动作（交换两个航班索引），更新状态，计算奖励。
        动作：tuple (i, j)，表示交换位置i和j的航班。
        返回：新状态，奖励，是否终止
        """
        # 保存旧状态和成本
        old_cost = self._compute_total_cost()

        # 执行交换动作
        i, j = action
        self.current_order[i], self.current_order[j] = self.current_order[j], self.current_order[i]
        self.assigned_times = self._assign_times(self.current_order)  # 更新分配时间

        # 计算新成本和奖励
        new_cost = self._compute_total_cost()
        reward = old_cost - new_cost  # 成本减少为正奖励
        done = False  # 单步不终止，由训练循环控制
        return self._get_state(), reward, done

    def _compute_total_cost(self):
        """计算当前排序的总延误成本（基于分配时间）"""
        total_cost = 0
        for idx in range(self.n_flights):
            flight_idx = self.current_order[idx]
            flight = self.flights[flight_idx]
            delay = abs(self.assigned_times[idx] - flight['scheduled_time'])  # 延误时间（分钟）
            cost = C_F * delay * flight['passengers']  # 简化成本公式
            total_cost += cost
        return total_cost

    def get_final_schedule(self):
        """获取最终排序结果：列表 of [航班序号, 人数, 原定时间, 恢复时间]"""
        schedule = []
        for idx in range(self.n_flights):
            flight_idx = self.current_order[idx]
            flight = self.flights[flight_idx]
            schedule.append([
                flight['id'],
                flight['passengers'],
                flight['scheduled_time'] ,
                self.assigned_times[idx] ,
            ])
        return schedule


# 定义策略网络（一维CNN + 全连接，基于论文）
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # 一维卷积层（提取序列特征）
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 全连接层
        self.fc1 = nn.Linear(32 * 100, HIDDEN_SIZE)  # 输入尺寸：通道*序列长度（假设序列长100）
        self.fc2 = nn.Linear(HIDDEN_SIZE, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) -> 转置为 (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)


# PPO代理（训练策略网络）
class PPOTrainer:
    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def train(self, max_iterations):
        """训练循环：收集轨迹，更新策略网络"""
        cost_history = []
        for iter in range(max_iterations):
            states, actions, rewards, log_probs = [], [], [], []
            state = self.env.reset()

            # 收集一批轨迹数据
            for _ in range(BATCH_SIZE):
                # 选择动作
                state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
                action_probs = self.policy_net(state_tensor).detach().numpy().flatten()

                # 动作空间：所有可能的航班对交换
                action_space = [(i, j) for i in range(self.env.n_flights) for j in range(i + 1, self.env.n_flights)]
                action_idx = np.random.choice(len(action_space), p=action_probs)
                action = action_space[action_idx]

                # 执行动作
                next_state, reward, done = self.env.step(action)

                # 存储数据
                states.append(state)
                actions.append(action_idx)
                rewards.append(reward)
                log_probs.append(np.log(action_probs[action_idx]))
                state = next_state

            # 计算优势函数
            rewards = np.array(rewards)
            discounted_rewards = []
            discounted_r = 0
            for r in rewards[::-1]:
                discounted_r = r + GAMMA * discounted_r
                discounted_rewards.insert(0, discounted_r)
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

            # 转换为Tensor
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
            adv_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)

            # PPO损失计算
            self.optimizer.zero_grad()
            new_probs = self.policy_net(states_tensor)
            new_log_probs = torch.log(new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            clip_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
            loss = -torch.min(ratio * adv_tensor, clip_ratio * adv_tensor).mean()

            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 记录成本
            current_cost = self.env._compute_total_cost()
            cost_history.append(current_cost)
            if iter % 100 == 0:
                print(f"Iteration {iter}, Total Cost: {current_cost:.2f}")

        # 保存模型
        torch.save(self.policy_net.state_dict(), SAVE_MODEL_PATH)
        print(f"Model saved to {SAVE_MODEL_PATH}")
        return cost_history


# 使用训练好的模型进行预测
def predict_optimal_schedule(flights, recovery_times, model_path=SAVE_MODEL_PATH):
    """加载模型并生成最优排序"""
    env = FlightRecoveryEnv(flights, recovery_times)
    policy_net = PolicyNetwork(input_dim=3,
                               output_dim=len(env.current_order) * (len(env.current_order) - 1) // 2)  # 输出维度 = 动作空间大小
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    state = env.reset()
    for _ in range(100):  # 运行多次步骤确保收敛
        state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        action_probs = policy_net(state_tensor).detach().numpy().flatten()
        action_space = [(i, j) for i in range(env.n_flights) for j in range(i + 1, env.n_flights)]
        action_idx = np.argmax(action_probs)
        action = action_space[action_idx]
        next_state, _, _ = env.step(action)
        state = next_state

    return env.get_final_schedule()


# 主函数：读取数据、训练、预测并输出
def main(excel_path, train_new_model=True):
    # 1. 加载数据
    flights, recovery_times = load_data(excel_path)
    print(f"Loaded {len(flights)} flights and {len(recovery_times)} recovery time slots.")

    # 2. 训练模型（或加载现有模型）
    env = FlightRecoveryEnv(flights, recovery_times)
    policy_net = PolicyNetwork(input_dim=3, output_dim=len(env.current_order) * (len(env.current_order) - 1) // 2)

    if train_new_model or not os.path.exists(SAVE_MODEL_PATH):
        print("Training new model...")
        trainer = PPOTrainer(env, policy_net)
        cost_history = trainer.train(MAX_ITERATIONS)
        # 绘制成本下降曲线（可选）
        plt.plot(cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Total Delay Cost')
        plt.title('Training Cost Reduction')
        plt.savefig('cost_reduction.png')
        plt.close()
    else:
        print(f"Loading pre-trained model from {SAVE_MODEL_PATH}")
        policy_net.load_state_dict(torch.load(SAVE_MODEL_PATH))

    # 3. 使用模型预测最优排序
    optimal_schedule = predict_optimal_schedule(flights, recovery_times)

    # 4. 输出排序结果表格
    result_df = pd.DataFrame(optimal_schedule, columns=['航班序号', '航班人数', '原定起飞时间', '恢复起飞时间'])
    result_df.to_excel('optimized_schedule.xlsx', index=False)
    print("Optimized schedule saved to 'optimized_schedule.xlsx'")

    return result_df


# 示例运行
if __name__ == "__main__":
    # 替换为您的Excel路径
    excel_path = "flight_data.xlsx"

    # 首次运行：训练模型并输出结果
    result_df = main(excel_path, train_new_model=True)
    print("\nOptimized Schedule:")
    print(result_df)

    # 下次运行其他数据时，可直接加载模型（train_new_model=False）
    # result_df_next = main("new_data.xlsx", train_new_model=False)