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

import gin
import math
import torch

# implement the respective reward functions for desired environments here


@gin.configurable
def reward_function_for_mountaincar_init(obs, action):
    """Function for computing reward for gym Pendulum environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """
    goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    goal_velocity = 0.0
    position = obs[:, 0]
    velocity = obs[:, 1]

    done_mask = (position >= goal_position) * (velocity >= goal_velocity)
    done_reward = torch.ones([obs.shape[0], 1]) * 1000.0
    reward = torch.zeros([obs.shape[0], 1])
    reward = torch.where(
        done_mask.view(-1, 1), done_reward, -action * action * 0.1)

    return reward


@gin.configurable
def reward_function_for_mountaincar(obs, action):
    """Function for computing reward for gym Pendulum environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """
    goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    goal_velocity = 0.0

    power = 0.0015
    max_speed = 0.07
    min_position = -1.2
    max_position = 0.6

    position = obs[:, 0:1]
    velocity = obs[:, 1:2]

    force = torch.clamp(action, -1.0, 1.0)

    velocity += force * power - 0.0025 * torch.cos(3 * position)
    velocity = torch.clamp(velocity, -max_speed, max_speed)

    position += velocity
    position = torch.clamp(position, min_position, max_position)

    reset_mask = (position == min_position) * (velocity < 0)
    velocity = velocity * reset_mask

    done_mask = (position >= goal_position) * (velocity >= goal_velocity)
    done_reward = torch.ones([obs.shape[0], 1]) * 1000.0
    reward = torch.zeros([obs.shape[0], 1])
    reward = torch.where(
        done_mask.view(-1, 1), done_reward, -action * action * 0.1)

    return reward
