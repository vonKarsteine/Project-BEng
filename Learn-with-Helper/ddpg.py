import numpy as np
import gymnasium as gym  # 这里可能是笔误，应该是 gym
import torch
import collections
import random
import copy
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd

# DDPG_Agent 类定义
class DDPG_Agent():

    def __init__(self,
                 action_scale: torch.tensor,  # 动作的缩放因子(数据类型为张量）
                 action_upper_bound: np.ndarray,  # 动作的上界（数据类型为多维数组）
                 action_lower_bound: np.ndarray,  # 动作的下界（数据类型为多维数组）
                 replay_buffer: collections.deque,  # 经验回放缓冲区（双端队列）
                 replay_start_size: int,  # 开始训练前的经验回放缓冲区大小
                 batch_size: int,  # 每个训练步骤中的批次大小
                 actor_update_frequent: int,  # 更新 actor 网络的频率
                 target_update_tau: float,  # 软更新参数
                 actor_network: torch.nn,  # actor 神经网络
                 critic_network: torch.nn,  # critic 神经网络
                 actor_optimizer: torch.optim,  # actor 网络优化器
                 critic_optimizer: torch.optim,  # critic 网络优化器
                 gamma: float = 0.9,  # 折扣因子
                 sigma_noise: float = 0.2,  # 噪声标准差
                 device: torch.device = torch.device("cpu")  # 设备类型，默认为 CPU
                 ) -> None:

        self.device = device  # 设置设备

        self.exp_counter = 0  # 经验计数器

        self.replay_buffer = replay_buffer  # 设置经验回放缓冲区
        self.replay_start_size = replay_start_size  # 设置开始训练前的经验回放缓冲区大小
        self.batch_size = batch_size  # 设置批次大小

        self.actor_update_frequent = actor_update_frequent  # 设置更新 actor 网络的频率
        self.target_update_tau = target_update_tau  # 设置软更新参数

        # 设置 critic 网络（主网络和目标网络）
        self.main_critic_network = critic_network.to(self.device)
        self.target_critic_network = copy.deepcopy(critic_network).to(self.device)

        # 设置 actor 网络（主网络和目标网络）
        self.main_actor_network = actor_network.to(self.device)
        self.target_actor_network = copy.deepcopy(actor_network).to(self.device)

        self.critic_optimizer = critic_optimizer  # 设置 critic 网络优化器
        self.actor_optimizer = actor_optimizer  # 设置 actor 网络优化器

        self.gamma = gamma  # 设置折扣因子
        self.sigma_noise = sigma_noise  # 设置噪声标准差
        self.action_scale = action_scale  # 设置动作缩放因子
        self.action_upper_bound = action_upper_bound  # 设置动作上界
        self.action_lower_bound = action_lower_bound  # 设置动作下界

    def get_behavior_action(self, obs: np.ndarray) -> np.ndarray:
        # 将观测转换为 PyTorch Tensor，并移到指定设备
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        # 通过主 actor 网络获取动作
        action = self.main_actor_network(obs)

        # 添加高斯噪声，以促进探索
        action += torch.normal(0, self.action_scale * self.sigma_noise)

        # 将动作转换为 NumPy 数组，并裁剪到指定的上下界
        action = action.cpu().detach().numpy().clip(self.action_lower_bound, self.action_upper_bound)


        return action  # 返回计算后的动作

    def soft_update_network(self, main_network: torch.nn, target_network: torch.nn) -> None:
        """软更新目标网络的参数"""
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            # 使用软更新规则更新目标网络的参数：目标网络参数 = 软更新参数 * 主网络参数 + (1 - 软更新参数) * 目标网络参数
            target_param.data.copy_(
                self.target_update_tau * main_param.data + (1.0 - self.target_update_tau) * target_param.data)

    def batch_Q_approximation(self,
                              obs: torch.tensor,
                              action: torch.tensor,
                              reward: torch.tensor,
                              next_obs: torch.tensor,
                              done: torch.tensor) -> None:
        """用于更新主 critic 网络"""

        # 获取当前状态动作对的 Q 值估计
        current_Q = self.main_critic_network(obs, action).squeeze(1)

        # 计算 TD 目标值
        TD_target = reward + (1 - done) * self.gamma * self.target_critic_network(next_obs, self.target_actor_network(
            next_obs)).squeeze(1)

        # 计算 critic 损失，使用均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(current_Q,
                                            TD_target.detach()))  # 将 TD_target 的梯度流断开，冻结目标网络的参数

        # 清零 critic 网络优化器的梯度
        self.critic_optimizer.zero_grad()

        # 反向传播并更新 critic 网络参数
        critic_loss.backward()
        self.critic_optimizer.step()

    def batch_actor_update(self, obs: torch.tensor) -> None:
        """用于更新主 actor 网络。
           在此之前，需要冻结主 critic 网络的参数"""

        # 将主 critic 网络的参数设置为不计算梯度
        for p in self.main_critic_network.parameters():
            p.requires_grad = False

        # 计算 actor 损失，目标是最大化 critic 的输出
        actor_loss = torch.mean(-self.main_critic_network(obs, self.main_actor_network(obs)))

        # 清零 actor 网络优化器的梯度
        self.actor_optimizer.zero_grad()

        # 反向传播并更新 actor 网络参数
        actor_loss.backward()
        self.actor_optimizer.step()

        """更新主 actor 网络后，
           需要解除主 critic 网络参数的冻结状态"""
        for p in self.main_critic_network.parameters():
            p.requires_grad = True

    def Q_approximation(self,
                        obs: np.ndarray,
                        action: int,
                        reward: float,
                        next_obs: np.ndarray,
                        done: bool) -> None:
        """在这里，我们继续使用 DQN 的框架，因为 DDPG 是 DQN 的扩展。
           最大的 Q 值由目标网络来近似。"""

        self.exp_counter += 1
        self.replay_buffer.append((obs, action, reward, next_obs, done))

        if len(self.replay_buffer) > self.replay_start_size:
            # 从经验回放缓冲区中随机采样一批数据
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)

            # 使用批次数据更新 critic 网络
            self.batch_Q_approximation(obs, action, reward, next_obs, done)  # 训练 critic

            if self.exp_counter % self.actor_update_frequent == 0:  # 每隔 actor_update_frequent 步更新一次 actor 网络
                # 使用批次数据更新 actor 网络，同时进行软更新目标网络
                self.batch_actor_update(obs)  # 训练 actor
                self.soft_update_network(self.main_critic_network,
                                         self.target_critic_network)  # 软更新目标 critic 网络
                self.soft_update_network(self.main_actor_network, self.target_actor_network)  # 软更新目标 actor 网络


class Actor_Network(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: torch.tensor, action_bias: torch.tensor) -> None:
        super(Actor_Network, self).__init__()

        # 定义第一个全连接层，输入维度为观测空间的维度，输出维度为64
        self.fc1 = torch.nn.Linear(obs_dim, 64)

        # 定义第二个全连接层，输入维度为64，输出维度为动作空间的维度
        self.fc2 = torch.nn.Linear(64, action_dim)

        self.action_scale = action_scale  # 动作缩放因子
        self.action_bias = action_bias  # 动作偏置

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)  # 第一个全连接层
        x = F.relu(x)  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 第二个全连接层
        x = torch.tanh(x)  # 使用 tanh 激活函数将输出缩放到 [-1, 1]

        return x * self.action_scale + self.action_bias  # 缩放并加上偏置，得到最终的动作输出


class Critic_Network(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(Critic_Network, self).__init__()

        # 定义第一个全连接层，输入维度为观测空间和动作空间的维度之和，输出维度为64
        self.fc1 = torch.nn.Linear(obs_dim + action_dim, 64)

        # 定义第二个全连接层，输入维度为64，输出维度为1
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, o: torch.tensor, a: torch.tensor) -> torch.tensor:
        x = torch.cat([o, a], dim=1)  # 将观测和动作连接在一起
        x = self.fc1(x)  # 第一个全连接层
        x = F.relu(x)  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 第二个全连接层

        return x  # 返回 critic 网络的输出


class ReplayBuffer():
    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp_data: tuple) -> None:
        """向经验回放缓冲区添加新的经验数据"""
        self.buffer.append(exp_data)

    def sample(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """从经验回放缓冲区中随机采样一个批次的数据"""
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)

        obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        """返回经验回放缓冲区的当前大小"""
        return len(self.buffer)


class TrainManager():

    def __init__(self,
                 env: gym.Env,
                 episode_num: int = 1000,
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.95,
                 sigma_noise: float = 0.2,
                 buffer_capacity: int = 2000,
                 replay_start_size: int = 200,
                 actor_update_frequent: int = 2,
                 target_update_tau: float = 1e-3,
                 batch_size: int = 32,
                 seed: int = 0,
                 my_device: str = "cpu"
                 ) -> None:
        """
        初始化 DDPG Agent 的训练管理器。

        参数：
        - env: Gym 环境对象
        - episode_num: 训练的总轮数
        - actor_lr: Actor 网络的学习率
        - critic_lr: Critic 网络的学习率
        - gamma: 折扣因子
        - sigma_noise: 动作噪声的标准差
        - buffer_capacity: 经验回放缓冲区的容量
        - replay_start_size: 开始训练前经验回放缓冲区中的样本数量
        - actor_update_frequent: 更新 Actor 网络的频率
        - target_update_tau: 软更新目标网络的参数时的权重
        - batch_size: 每次更新所使用的小批次大小
        - seed: 随机种子，以确保实验的可复现性
        - my_device: 使用的设备 ("cpu" 或 "cuda")
        """
        # 设置随机种子以保证实验的可复现性
        self.seed = seed  # 设置随机种子，用于保证实验的可复现性
        random.seed(self.seed)  # 设置 Python 内置的随机数生成器的种子
        torch.manual_seed(self.seed)  # 设置 PyTorch 的随机数生成器的种子
        torch.cuda.manual_seed_all(seed)  # 设置 PyTorch 在 CUDA 上的随机数生成器的种子
        np.random.seed(self.seed)  # 设置 NumPy 的随机数生成器的种子
        torch.backends.cudnn.deterministic = True  # 当使用 CUDA 加速时，设置 PyTorch 的 CuDNN 以保证实验的可复现性

        self.device = torch.device(my_device)  # 设置设备类型

        self.env = env
        _, _ = self.env.reset(seed=self.seed)
        self.episode_num = episode_num

        # 获取观测和动作空间的维度信息
        obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        action_dim = gym.spaces.utils.flatdim(env.action_space)

        # 获取动作空间的上下界信息
        action_upper_bound = env.action_space.high
        action_lower_bound = env.action_space.low

        # 计算动作空间的偏置和缩放因子
        # 计算动作的偏置，即上下界中点
        action_bias = (action_upper_bound + action_lower_bound) / 2.0
        # 将偏置转换为 PyTorch Tensor，并移到指定设备
        action_bias = torch.tensor(action_bias, dtype=torch.float32).to(self.device)
        # 计算动作的缩放因子，即上下界之差的一半
        action_scale = (action_upper_bound - action_lower_bound) / 2.0
        # 将缩放因子转换为 PyTorch Tensor，并移到指定设备
        action_scale = torch.tensor(action_scale, dtype=torch.float32).to(self.device)

        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(capacity=buffer_capacity, device=self.device)

        # 创建 actor 网络和 critic 网络
        actor_network = Actor_Network(obs_dim, action_dim,
                                      action_scale, action_bias).to(self.device)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=actor_lr)
        critic_network = Critic_Network(obs_dim, action_dim).to(self.device)
        critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=critic_lr)

        # 创建 DDPG 强化学习代理
        self.agent = DDPG_Agent(action_scale=action_scale,
                                action_upper_bound=action_upper_bound,
                                action_lower_bound=action_lower_bound,
                                replay_buffer=self.buffer,
                                replay_start_size=replay_start_size,
                                batch_size=batch_size,
                                actor_update_frequent=actor_update_frequent,
                                target_update_tau=target_update_tau,
                                actor_network=actor_network,
                                critic_network=critic_network,
                                actor_optimizer=actor_optimizer,
                                critic_optimizer=critic_optimizer,
                                gamma=gamma,
                                sigma_noise=sigma_noise,
                                device=self.device)

        # 初始化用于存储每个 episode 总奖励的数组
        self.episode_total_rewards = np.zeros(episode_num)
        self.index_episode = 0

    def train_episode(self) -> float:
        total_reward = 0
        obs, _ = self.env.reset()

        while True:
            action = self.agent.get_behavior_action(obs)  # 获取智能体的动作
            next_obs, reward, terminated, truncated, _ = self.env.step(action)  # 执行动作并获取下一个状态、奖励等信息
            done = terminated or truncated  # 判断是否终止当前 episode
            total_reward += reward  # 累计奖励

            # 使用经验更新智能体的 Q 值估计
            self.agent.Q_approximation(obs, action, reward, next_obs, done)

            obs = next_obs  # 更新当前观测
            if done:
                # 记录当前 episode 的总奖励，并更新索引
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break

        return total_reward  # 返回当前 episode 的总奖励

    def train(self) -> None:
        for e in range(self.episode_num):
            episode_reward = self.train_episode()
            if e % 50 == 0:
                print('Episode %s: Total Reward = %.2f' % (e, episode_reward))

    def plotting(self, smoothing_window: int = 50) -> None:
        """绘制随时间变化的 episode 奖励曲线"""
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.episode_total_rewards, label="Episode Reward")
        # 使用滑动平均对曲线进行平滑处理
        rewards_smoothed = pd.Series(self.episode_total_rewards).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, label="Episode Reward (Smoothed)")
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title("Episode Reward over Time")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    Manger = TrainManager(env=env,
                          episode_num=500,
                          actor_lr=1e-3,
                          critic_lr=1e-2,
                          gamma=0.95,
                          sigma_noise=0.01,
                          buffer_capacity=10000,
                          replay_start_size=1000,
                          actor_update_frequent=1,
                          target_update_tau=5e-3,
                          batch_size=64,
                          seed=0,
                          my_device="cpu"
                          )
    Manger.train()
    Manger.plotting()