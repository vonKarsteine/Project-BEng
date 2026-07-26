from __future__ import division
import numpy as np
import gymnasium as gym
import gym_uav


def create_env(name, seed):
    env = frame_stack(name, seed)
    return env


class frame_stack:
    """Thin adapter exposing the old 4-tuple gym API on top of gymnasium."""

    def __init__(self, name, seed):
        self.name = name
        # disable_env_checker: the checker calls reset(seed=...) with
        # follow-up assertions the 2019-era env was never written for
        self.env = gym.make(name, disable_env_checker=True)
        # gymnasium removed env.seed(); UavDenseEnv keeps its own seed()
        # method (a no-op for seed < 0, preserving the original semantics)
        self.env.unwrapped.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        ob, _info = self.env.reset()
        return ob

    def step(self, action):
        ob, rew, done, truncated, info = self.env.step(action)
        # UavDenseEnv returns done as an int sum of termination flags
        return ob, rew, bool(done) or bool(truncated), info

    def render(self):
        return self.env.render()


class MaxMinFilter:
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs


class NormalizedEnv:
    def __init__(self):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def __call__(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

