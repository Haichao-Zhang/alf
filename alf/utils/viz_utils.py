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

import sys
sys.path.append('/mnt/DATA/work/RL/alf')

import gym
import numpy as np
import time


def get_img_cube(s0, ac_seqs, env):
    """Visualization function
    Args:
        s0: initial state
        ac_seqs: action sequence [T, ...]
    """
    T = ac_seqs.shape[0]
    # set initial state
    env.reset(init_obs=s0)
    img_init = env.render(mode='rgb_array')

    img_cube = np.zeros([T + 1] + list(img_init.shape))

    img_cube[0] = img_init

    for t in range(T):
        #env.render()
        #time.sleep(500)
        obs, reward, done, info = env.step(ac_seqs[t])  # take a random action
        img = env.render(mode='rgb_array')
        img_cube[t + 1] = obs
    env.close()


def merge_img_cube(img_cube, overlap_ratio):
    T = img_cube.shape[0]
    nr = img_cube.shape[1]
    nc = img_cube.shape[2]
    nb = img_cube.shape[3]

    overlap_nc = np.round(nc * overlap_ratio)
    new_nc = nc * T - overlap_nc * (T - 1)
    comp_img = np.zeros(nr, nc, nb)


env_name = 'Pendulum-v0'
env = gym.make(env_name)

ac_seqs = np.array([[0], [0]])
T = 100

action_dim = env.action_space.shape[0]
ac_seqs = np.zeros([T, action_dim])
for t in range(T):
    ac_seqs[t] = env.action_space.sample()
s0 = np.array([1, 0, 0])
img_cube = get_img_cube(s0, ac_seqs, env)

comp_img = merge_img_cube(img_cube, 0.2)
