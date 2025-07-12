import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam
from datetime import datetime
import matplotlib.pyplot as plt

# 超参数优化
LEARNING_RATE = 0.01
GAMMA = 0.9
EPS_CLIP = 0.2
EPOCHS = 5000
BATCH_SIZE = 128


# 1. 数据预处理模块（完全重构）
def load_data():
    flights = pd.read_excel("flights_data.xlsx")
    loss_table = pd.read_excel("aircraft_loss.xlsx")
    merged = pd.merge(flights, loss_table, on='aircraft_type')

    # 修正1：确保处理的是副本
    state_data = merged[['passenger_count', 'avg_fare', 'max_capacity', 'load_factor', 'delay_loss']].copy()

    # 修正2：安全添加新列
    initial_delay = (merged['scheduled_departure'] - merged['scheduled_departure_available'])
    state_data.loc[:, 'initial_delay'] = initial_delay.abs().values
    state_data.loc[:, 'passenger_loss'] = 50  # 固定值

    # 时间处理保持不变
    available_times = merged['scheduled_departure_available'].values.astype(np.int64) // 10 ** 9

    return {
        'state': torch.tensor(state_data.values, dtype=torch.float32),
        'original_schedule': merged['scheduled_departure'].values.astype(np.int64) // 10 ** 9,
        'available_times': available_times
    }

# 2. 环境模拟器（完全重构）
class FlightEnv:
    def __init__(self, states, original_schedule, available_times):
        self.states = states.clone()
        self.original_schedule = original_schedule.copy()
        self.current_schedule = original_schedule.copy()
        self.available_times = available_times.copy()
        self.n_flights = len(states)

    def reset(self):
        """重置环境到初始状态"""
        self.current_schedule = self.original_schedule.copy()
        return self.states.clone(), self.original_schedule.copy()

    def _action_to_indices(self, action_idx):
        """将动作索引转换为航班索引对"""
        n = self.n_flights
        i = action_idx // (n - 1)
        j = action_idx % n
        if j >= i:
            j += 1
        return i, j

    def step(self, action_idx):
        """执行航班顺序交换动作"""
        i, j = self._action_to_indices(action_idx)

        # 交换航班顺序
        self.current_schedule[i], self.current_schedule[j] = self.current_schedule[j], self.current_schedule[i]

        # 计算新延误（基于可用起飞时间）
        delays = np.abs(self.current_schedule - self.available_times)

        # 更新状态中的延误时长
        new_states = self.states.clone()
        new_states[:, 0] = torch.tensor(delays)

        # 计算奖励
        current_loss = self._calc_total_loss(delays, new_states)
        prev_loss = self._calc_total_loss(self.states[:, 0].numpy(), self.states)
        reward = prev_loss - current_loss

        return new_states, reward, delays

    def _calc_total_loss(self, delays, states):
        """计算总损失（优化公式）"""
        # 状态向量: [延误, 票价, 乘客数, 座位数, 机型损失, 旅客损失]
        delays_hour = delays / 60
        passenger_disappoint = (delays_hour ** 2) ** (1 / 3) / 29  # 失望率d

        # 航空公司损失
        airline_loss = states[:, 3] * states[:, 4] * 0.1 * states[:, 1] * delays / states[:, 5]

        # 机场损失
        airport_loss = states[:, 4] * delays

        # 旅客损失
        passenger_loss = states[:, 6] * delays

        # 总损失计算
        total_loss = (passenger_disappoint + 1.0 * airline_loss + 1.0 * airport_loss + 1.1 * passenger_loss).sum()
        return total_loss.item()


# 3. 策略网络（保持不变）
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_dim - 1) * 8, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, n_actions)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze()

    def act(self, state, mask):
        logits = self(state)
        logits[~mask] = -1e8
        probs = torch.softmax(logits, dim=-1)
        return probs.multinomial(1).item()


# 4. PPO训练算法（优化训练逻辑）
def train():
    data = load_data()
    env = FlightEnv(data['state'], data['original_schedule'], data['available_times'])
    n_actions = env.n_flights * (env.n_flights - 1)
    policy = PolicyNetwork(data['state'].shape[1], n_actions)
    optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

    # 训练过程可视化
    loss_history = []

    for epoch in range(EPOCHS):
        state, _ = env.reset()
        rewards = []

        # 单次迭代轨迹
        for step in range(100):  # 固定步长
            mask = torch.ones(n_actions).bool()
            action_idx = policy.act(state, mask)
            next_state, reward, _ = env.step(action_idx)
            rewards.append(reward)
            state = next_state.clone()

        # PPO梯度更新
        loss = -torch.mean(torch.tensor(rewards))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())

        # 保存模型
        if epoch % 100 == 0:
            torch.save(policy.state_dict(), f"flight_model_epoch{epoch}.pt")
            print(f"Epoch {epoch}: Loss={loss.item():.2f}")

    # 绘制损失曲线
    plt.plot(loss_history)
    plt.title('Training Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()


# 5. 预测与输出（增强结果展示）
def predict(model_path):
    # 加载数据
    data = load_data()

    # 初始化策略网络
    n_actions = data['state'].shape[0] * (data['state'].shape[0] - 1)
    policy = PolicyNetwork(data['state'].shape[1], n_actions)
    policy.load_state_dict(torch.load(model_path))

    # 初始化环境
    env = FlightEnv(data['state'], data['original_schedule'], data['available_times'])
    state, schedule = env.reset()

    # 执行航班重排
    delay_history = []
    for step in range(50):  # 固定决策步数
        mask = torch.ones(n_actions).bool()
        action_idx = policy.act(state, mask)
        state, _, delays = env.step(action_idx)
        delay_history.append(delays.mean())

    # 构建输出表格
    result = pd.DataFrame({
        "flight_id": range(len(schedule)),
        "passenger_count": data['state'][:, 2].tolist(),
        "original_time": pd.to_datetime(data['original_schedule'], unit='s'),
        "available_time": pd.to_datetime(data['available_times'], unit='s'),
        "recovered_time": pd.to_datetime(env.current_schedule, unit='s'),
        "delay_minutes": delays
    })

    # 计算延误改善
    initial_delay = (result['original_time'] - result['available_time'])
    improvement = (initial_delay - result['delay_minutes']).mean()
    print(f"平均延误改善: {improvement:.2f} 分钟")

    # 保存结果
    result.to_excel("recovered_schedule.xlsx", index=False)

    # 延误改善可视化
    plt.figure(figsize=(10, 6))
    plt.plot(delay_history)
    plt.title('Delay Improvement During Recovery')
    plt.xlabel('Decision Step')
    plt.ylabel('Average Delay (minutes)')
    plt.savefig('delay_improvement.png')

    return result


# 主流程（增加异常处理）
if __name__ == "__main__":
    try:
        print("Starting training process...")
        train()

        print("\nStarting prediction with best model...")
        # 使用最终模型预测
        predict(f"flight_model_epoch{EPOCHS - 1}.pt")

        print("\nOperation completed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # 错误处理逻辑可根据需要扩展