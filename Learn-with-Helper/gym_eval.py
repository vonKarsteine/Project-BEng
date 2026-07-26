from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_MLP
from player_util import Agent
import logging

parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='uav-v0',
    metavar='ENV',
    help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--render',
    action='store_true',
    help='Watch the episode in the vtk visualization window')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
parser.add_argument(
    '--model',
    default='MLP',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=1,
    metavar='SF',
    help='Choose whether to stack observations')
parser.add_argument(
    '--new-gym-eval',
    default=False,
    metavar='NGE',
    help='Create a gym evaluation for upload')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--reward-type',
    type=str,
    default='sparse',
    metavar='RT',
    help='reward type (default: 1)')

parser.add_argument(
    '--variance',
    type=float,
    nargs='+',
    metavar='V',
    help='variance of actions')

parser.add_argument(
    '--prior-decay',
    type=float,
    default=0.0005,
    metavar='PD',
    help='decay slope of prior (default: 0.0005)')

parser.add_argument(
    '--demo-type',
    type=str,
    default='uav',
    help='demonstration type')

args = parser.parse_args()

print("begin loading models")
saved_state = torch.load(
    os.path.join(args.load_model_dir, '{}.dat'.format(args.env)),
    map_location=lambda storage, loc: storage)
print("finished loading models")

torch.manual_seed(args.seed)

env = create_env(args.env, -1)
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None, -1)
player.model = A3C_MLP(env.observation_space, env.action_space, args.stack_frames)

player.model.load_state_dict(saved_state)

player.model.eval()
success_count = 0
for i_episode in range(args.num_episodes):
    speed = []
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.eps_len = 0
    reward_sum = 0
    while True:
        if args.render:
            returned = player.env.render()
            speed.append(returned[0])

        player.action_test()
        reward_sum += player.reward

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            if player.info.get('is_success', False):
                success_count += 1
            print('episode {0}: reward {1:.2f}, success {2}, length {3}'.format(
                i_episode, reward_sum, player.info.get('is_success', False),
                player.eps_len))
            break

print('evaluated {0} episodes: mean reward {1:.2f}, success rate {2:.2f}'.format(
    num_tests, reward_mean, success_count / num_tests))
