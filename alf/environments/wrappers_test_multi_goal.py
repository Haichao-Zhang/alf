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

import gym
from gym import spaces
import numpy as np
import tensorflow as tf

from alf.environments.multi_goal_wrappers import RandomGoalWrapper


class FakeEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(
            shape=(32, 32), low=0, high=255, dtype=np.uint8)
        self.action_space = spaces.Box(
            shape=(1, ), low=-1, high=1, dtype=np.float32)

    def render(self, width=32, height=32, *args, **kwargs):
        del args
        del kwargs
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


class MultiGoalTest(tf.test.TestCase):
    def _create_env(self, num_of_goals):
        return RandomGoalWrapper(
            env=FakeEnvironment(), num_of_goals=num_of_goals)
        # env = FakeEnvironment()
        # return env

    def test_multi_goal(self):
        env = self._create_env(num_of_goals=1)
        obs = env.reset()
        all_fields = obs.keys()
        expected = {"obs", "goal"}
        assert all_fields == expected, "Result " + str(
            all_fields) + " doesn't match exptected " + str(expected)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
