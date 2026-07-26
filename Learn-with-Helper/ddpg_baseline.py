"""Classical DDPG baseline on uav-v0 (thesis Fig 10 comparison curve).

Reuses the student's DDPG_Agent / ReplayBuffer from ddpg.py, with the
400/300 hidden layers stated in policy_domain.Config, and logs periodic
greedy evaluations in the same format as test.py so plot_results.py can
overlay the curve with the LwH runs.

Usage:
    python ddpg_baseline.py --training-steps 1400000 --log-dir logs/ddpg
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ddpg import DDPG_Agent, ReplayBuffer
from environment import create_env
from utils import setup_logger


class Actor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden1=400, hidden2=300):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden1=400, hidden2=300):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim + action_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, 1)

    def forward(self, o, a):
        x = torch.cat([o, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(env, actor, episodes, max_episode_length):
    successes, total_reward = 0, 0.0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_episode_length):
            with torch.no_grad():
                action = actor(torch.from_numpy(state).float()).numpy()
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                if info.get('is_success', False):
                    successes += 1
                break
    return total_reward / episodes, successes / episodes


def main():
    parser = argparse.ArgumentParser(description='DDPG baseline on uav-v0')
    parser.add_argument('--env', default='uav-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--training-steps', type=int, default=int(1.4e6))
    parser.add_argument('--max-episode-length', type=int, default=2000)
    parser.add_argument('--cache-interval', type=int, default=50000,
                        help='evaluate every this many environment steps')
    parser.add_argument('--test-episodes', type=int, default=50)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--sigma-noise', type=float, default=0.2)
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--replay-start-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--log-dir', default=os.path.join('logs', 'ddpg'))
    parser.add_argument('--save-model-dir', default='trained_models/')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    setup_logger('ddpg_log', os.path.join(args.log_dir, '{}_log'.format(args.env)))
    log = logging.getLogger('ddpg_log')
    for k, v in vars(args).items():
        log.info('{0}: {1}'.format(k, v))

    train_env = create_env(args.env, args.seed)
    eval_env = create_env(args.env, -1)
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    device = torch.device('cpu')
    action_scale = torch.ones(action_dim)
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim, action_dim)
    agent = DDPG_Agent(
        action_scale=action_scale,
        action_upper_bound=train_env.action_space.high,
        action_lower_bound=train_env.action_space.low,
        replay_buffer=ReplayBuffer(capacity=args.buffer_capacity, device=device),
        replay_start_size=args.replay_start_size,
        batch_size=args.batch_size,
        actor_update_frequent=2,
        target_update_tau=5e-3,
        actor_network=actor,
        critic_network=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=args.critic_lr),
        gamma=args.gamma,
        sigma_noise=args.sigma_noise,
        device=device,
    )

    start_time = time.time()
    steps, episodes, next_eval = 0, 0, 0
    state = train_env.reset()
    eps_len = 0
    while steps < args.training_steps:
        action = agent.get_behavior_action(state)
        next_state, reward, done, info = train_env.step(action)
        eps_len += 1
        if eps_len >= args.max_episode_length:
            done = True
        agent.Q_approximation(state, action, reward, next_state, done)
        state = next_state
        steps += 1
        if done:
            episodes += 1
            eps_len = 0
            state = train_env.reset()

        if steps >= next_eval:
            reward_mean, success_rate = evaluate(
                eval_env, agent.main_actor_network,
                args.test_episodes, args.max_episode_length)
            log.info(
                "Time {0}, training episodes {1}, training steps {2}, reward episode {3}, "
                "success_rate {4}, model cached 0".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    episodes, steps, reward_mean, success_rate))
            torch.save(agent.main_actor_network.state_dict(),
                       os.path.join(args.save_model_dir, '{}_ddpg.dat'.format(args.env)))
            next_eval += args.cache_interval


if __name__ == '__main__':
    main()
