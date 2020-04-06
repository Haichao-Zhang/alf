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

# import sys
# sys.path.append('/mnt/DATA/work/RL/alf')
import os
import gym
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

from alf.utils.common import add_method
from gym.envs.classic_control import PendulumEnv


@add_method(PendulumEnv)
def reset(self, init_obs=None):
    # added by hz 2020-03-02 15:31:57
    # will derive the initial state from given observation
    if init_obs is not None:
        #np.array([np.cos(theta), np.sin(theta), thetadot])
        c_theta = init_obs[0]
        s_theta = init_obs[1]
        theta = np.arctan2(s_theta, c_theta)
        thetadot = init_obs[2]
        init_state = np.array([theta, thetadot])
        self.state = init_state
    else:
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
    self.last_u = None
    return self._get_obs()


def save_to_np(tensor, file_path):
    # save np array
    tensor_np = tensor.cpu().numpy()
    np.save(file_path, tensor_np)


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
        img_cube[t + 1] = img
    env.close()
    return img_cube


def get_img_cube_obs(s0, obs_seqs, env):
    """Visualization function
    Args:
        s0: initial state
        obs_seqs: observation sequence [T, ...]
    """
    T = obs_seqs.shape[0]
    # set initial state
    env.reset(init_obs=s0)
    img_init = env.render(mode='rgb_array')

    img_cube = np.zeros([T + 1] + list(img_init.shape))

    img_cube[0] = img_init

    for t in range(T):
        #env.render()
        #time.sleep(500)
        st = obs_seqs[t, :]
        st = np.squeeze(st)
        env.reset(init_obs=st)
        img = env.render(mode='rgb_array')
        img_cube[t + 1] = img
    env.close()
    return img_cube


def merge_img_cube(img_cube, overlap_ratio):
    T = img_cube.shape[0]
    nr = img_cube.shape[1]
    nc = img_cube.shape[2]
    nb = img_cube.shape[3]

    overlap_nc = np.round(nc * overlap_ratio)
    nonoverlap_nc = int(nc - overlap_nc)
    new_nc = int(nc * T - overlap_nc * (T - 1))

    comp_img = np.zeros([nr, new_nc, nb])
    norm_img = np.zeros([nr, new_nc, nb])
    for t in range(T):
        comp_img[:, t * nonoverlap_nc:t * nonoverlap_nc +
                 nc, :] += img_cube[t, :, :].astype(float)
        norm_img[:, t * nonoverlap_nc:t * nonoverlap_nc + nc, :] += np.ones(
            [nr, nc, nb])

    comp_img = np.divide(comp_img, norm_img).astype(np.uint8)
    return comp_img


def vis_actions(s0, ac_seqs_pop, viz_path, name):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]

    # T = 25
    # ac_seqs = np.zeros([T, action_dim])
    # for t in range(T):
    #     ac_seqs[t] = env.action_space.sample()

    # s0 = np.array([0, 1, 0])

    # ac_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/ac_seq.mat.npy')  # [batch, pop, T]
    # obs_seqs = np.load('/mnt/DATA/work/RL/alf/obs_seqs.mat.npy')  # [1, 3]

    os.makedirs(viz_path, exist_ok=True)
    # population number
    pop_num = ac_seqs_pop.shape[1]

    for p in range(pop_num):
        ac_seqs = np.reshape(
            np.squeeze(ac_seqs_pop[0, p, :]), [-1, action_dim])
        img_cube = get_img_cube(s0, ac_seqs, env)
        comp_img = merge_img_cube(img_cube, 0.4)

        filename = 'plan_{}_{}_action.png'.format(name, p)
        img_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
        print(os.path.join(viz_path, filename))
        cv2.imwrite(os.path.join(viz_path, filename), img_bgr)


def vis_observations(obs_seqs_pop, ac_seqs_pop, viz_path, name):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]

    # T = 25
    # ac_seqs = np.zeros([T, action_dim])
    # for t in range(T):
    #     ac_seqs[t] = env.action_space.sample()

    # s0 = np.array([0, 1, 0])

    # ac_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/ac_seq.mat.npy')  # [batch, pop, T]
    # obs_seqs_pop = np.load('/mnt/DATA/work/RL/alf/obs_seqs.mat.npy')  # [1, 3]

    # ac_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/ac_seqs_latest.mat.npy')  # [batch, pop, T]
    # obs_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/obs_seqs_latest.mat.npy')  # [1, 3]

    s0 = obs_seqs_pop[:, 0, 0, :]
    s0 = np.squeeze(s0)

    obs_dim = obs_seqs_pop.shape[-1]

    os.makedirs(viz_path, exist_ok=True)
    # population number
    pop_num = ac_seqs_pop.shape[1]

    for p in range(pop_num):
        ac_seqs = np.reshape(
            np.squeeze(ac_seqs_pop[0, p, :]), [-1, action_dim])
        obs_seqs = np.reshape(np.squeeze(obs_seqs_pop[0, p, :]), [-1, obs_dim])
        img_cube = get_img_cube_obs(s0, obs_seqs, env)
        comp_img = merge_img_cube(img_cube, 0.4)

        filename = 'plan_{}_{}_obs.png'.format(name, p)
        img_bgr = cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR)
        print(os.path.join(viz_path, filename))
        cv2.imwrite(os.path.join(viz_path, filename), img_bgr)


def plot_actions(ac_seqs_pop, viz_path, name):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]

    os.makedirs(viz_path, exist_ok=True)
    # population number
    pop_num = ac_seqs_pop.shape[1]

    filename = 'plan_plot_{}_action.png'.format(name)

    for p in range(min(10, pop_num)):
        ac_seqs = np.reshape(
            np.squeeze(ac_seqs_pop[0, p, :]), [-1, action_dim])

        ac_seqs = ac_seqs.flatten()

        print(ac_seqs)

        plt.plot(ac_seqs, linewidth=2)
        #plt.show()

        print(os.path.join(viz_path, filename))
        plt.savefig(os.path.join(viz_path, filename))
        # cv2.imwrite(os.path.join(viz_path, filename), img_bgr)


def viz():
    # random
    # latest
    # beam (latest)
    # ac_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/ac_seq.mat.npy')  # [batch, pop, T]
    # obs_seqs_pop = np.load('/mnt/DATA/work/RL/alf/obs_seqs.mat.npy')  # [1, 3]
    ac_seqs_pop = np.load(
        '/mnt/DATA/work/RL/alf/ac_seqs_std_car_before_train.mat.npy'
    )  # [batch, pop, T]
    # obs_seqs_pop = np.load(
    #     '/mnt/DATA/work/RL/alf/obs_seqs_std.mat.npy')  # [1, 3]
    viz_path = '/home/haichaozhang/Documents/data/mbrl/viz_plot_car'

    plot_actions(ac_seqs_pop, viz_path, "action_before_train")

    # plt.plot([1, 2, 3, 4])


if __name__ == '__main__':
    viz()
