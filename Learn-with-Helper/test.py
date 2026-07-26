from __future__ import division
try:
    from setproctitle import setproctitle as ptitle
except ImportError:
    def ptitle(title):
        pass
import os
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_MLP
from player_util import Agent
import time
import logging
import queue


class Model_Buffer:
    def __init__(self, args):
        self.args = args
        self.model_buffer = queue.Queue(-1)
        self.flag = -1

    def put(self, model):
        training_steps = int(model.training_steps.weight.data.cpu().numpy()[0])
        training_episodes = int(model.training_steps.bias.data.cpu().numpy()[0])
        flag = int(training_steps / self.args.cache_interval)
        if flag > self.flag:
            self.flag = flag
            # clone: the state_dict holds references to the live shared-memory
            # tensors; without cloning every queued "snapshot" would alias the
            # newest weights and the evaluation would mislabel its checkpoints
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            self.model_buffer.put([snapshot,
                                   int(flag * self.args.cache_interval), int(training_episodes)])
        return self

    def get(self):
        if self.model_buffer.empty():
            return False
        else:
            return self.model_buffer.get()

    def get_flag(self):
        return self.flag

    def qsize(self):
        return self.model_buffer.qsize()

    def clear(self):
        self.model_buffer = queue.Queue(-1)
        self.flag = -1


def test(rank, args, shared_model):
    model_buffer = Model_Buffer(args)
    test_episodes = args.test_episodes
    ptitle('Test Agent')
    torch.set_num_threads(1)
    log = {}
    log_path = os.path.join(args.log_dir, '{}_log'.format(args.env))
    setup_logger('{}_log'.format(args.env), log_path)
    print("logfile check", log_path)

    print("logs in test", args.log_dir)

    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    # for i in range(100):
    #     log['{}_log'.format(args.env)].info('{0}'.format(i))

    # print('we prefix seed = -1 when testing')
    # args.seed = -1
    torch.manual_seed(args.seed)
    env = create_env(args.env, args.seed)
    # env = gym.make(args.env)
    # env.seed(args.seed)

    start_time = time.time()
    num_tests = 0
    training_steps = 0
    # last snapshot label that training can produce: labels are floored to
    # cache_interval multiples, so 'training_steps > args.training_steps'
    # would never fire and the tester would hang forever after training ends
    last_expected_label = (args.training_steps // args.cache_interval) * args.cache_interval
    player = Agent(None, env, args, None, rank)
    player.model = A3C_MLP(
        player.env.observation_space, player.env.action_space, args.stack_frames)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.done = True

    player.model.eval()

    is_model_empty = True
    is_testing = False
    while True:
        model_buffer.put(shared_model)
        if player.done and np.mod(num_tests, test_episodes) == 0 and not is_testing:
            reward_episode = 0
            success_rate = 0
            load_model = model_buffer.get()
            model_queue_size = model_buffer.qsize()
            if load_model:
                is_testing = True
                is_model_empty = False
                training_steps = load_model[1]
                training_episodes = load_model[2]
                player.model.load_state_dict(load_model[0])
            else:
                is_model_empty = True
                time.sleep(10)

        if not is_model_empty:
            player.action_test()
            # log['{}_log'.format(args.env)].info("test steps {}".format(1))
            reward_episode += player.reward
            # the env sets is_success on every terminal/truncated step, with
            # value False on crashes/timeouts — check the value, not the key
            if player.info.get('is_success', False):
                success_rate += 1

            if player.done:
                eps_len_temp = player.eps_len

                num_tests += 1
                player.eps_len = 0
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()

                if np.mod(num_tests, test_episodes) == 0:
                    is_testing = False
                    reward_episode = reward_episode / test_episodes
                    success_rate = success_rate / test_episodes
                    log['{}_log'.format(args.env)].info(
                        "Time {0}, training episodes {1}, training steps {2}, reward episode {3}, success_rate {4}, "
                        "model cached {5}"
                            .format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                                    training_episodes, training_steps, reward_episode, success_rate,
                                    model_queue_size))

                    # save model where gym_eval.py expects it
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, os.path.join(
                        args.save_model_dir, '{}.dat'.format(args.env)))
        if training_steps >= last_expected_label and not is_testing:
            break
