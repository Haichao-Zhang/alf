# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
import numpy as np


def viz(s0, ac_seqs, gym_name='Pendulum-v0'):
    """Visualization function
    Args:
        s0: initial state
        ac_seqs: action sequence [T, ...]
    """
    env = gym.make(gym_name)
    env.reset(s0)
    # set initial state
    T = ac_seqs.shape[0]
    for t in range(T):
        img = env.render(mode='rgb_array')
        observation, reward, done, info = env.step(
            ac_seqs[t])  # take a random action
    env.close()


ac_seqs = np.array([[0], [0]])
s0 = np.array([0, 1, 0])
viz(s0, ac_seqs)
