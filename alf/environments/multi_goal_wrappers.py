# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections import OrderedDict
import random

import gym
import numpy as np
import cv2
import gin
from tf_agents.environments import wrappers


@gin.configurable
class RandomGoalWrapper(gym.Wrapper):
    """
    Generate a random goal and extend the observation to an augmented
    observation of the following form:
    obs_aug = {
        obs: obs_org,
        goal: goal
    }
    """

    def __init__(self, env, num_of_goals):
        """Create a RandomGoalWrapper object

        Args:
            env (gym.Env): the gym environment
            num_of_goals (int): total number of goals to sample from
        """
        super().__init__(env)
        self._num_of_goals = num_of_goals
        self._p_goal = np.full(self._num_of_goals, 1.0 / self._num_of_goals)

        # observation_goal = gym.spaces.Discrete(
        #     low=0, high=self._num_of_goals, shape=(1, ), dtype=np.int32)
        observation_goal = gym.spaces.Discrete(self._num_of_goals)
        # observation_obs = gym.spaces.Box(
        #     low=-10.0, high=10.0, shape=(2, ), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            obs=env.observation_space, goal=observation_goal)
        # self.observation_space = gym.spaces.Dict(
        #     obs=observation_obs, goal=observation_goal)

        self._goal = self.sample_goal()

    def sample_goal(self):
        """
        Samples goal according to goal probabilities in self._p_goal
        """
        return np.random.choice(self._num_of_goals, p=self._p_goal)
        #return 3

    def aug_obs_with_goal(self, obs_org):
        obs_aug = OrderedDict()
        obs_aug['obs'] = obs_org
        obs_aug['goal'] = np.asarray(self._goal)  # should be np.array??
        return obs_aug

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.aug_obs_with_goal(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        # resample the goal
        obs = self.env.reset(**kwargs)
        self._goal = self.sample_goal()
        obs = self.aug_obs_with_goal(obs)
        return obs
