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

import numpy as np
import torch
from alf.data_structures import TimeStep
from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self,
                 solution_dim,
                 population_size,
                 upper_bound=None,
                 lower_bound=None):
        """Creates a Random Optimizer

        Args:
            solution_dim (int): The dimensionality of the problem space
            population_size (int): The number of candidate solutions to be
                sampled at every iteration
            upper_bound (int|tf.Tensor): upper bounds for elements in solution
            lower_bound (int|tf.Tensor): lower bounds for elements in solution
        """
        super().__init__()
        self._solution_dim = solution_dim
        self._population_size = population_size
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound

    def obtain_solution(self, time_step: TimeStep, state):
        """Minimize the cost function provided

        Args:
            time_step (TimeStep): the initial time_step to start rollout
            state: input state to start rollout
        """
        init_obs = time_step.observation
        batch_size = init_obs.shape[0]
        solutions = torch.rand(
            batch_size, self._population_size, self._solution_dim) * (
                self._upper_bound - self._lower_bound) / 2.0 + (
                    self._upper_bound + self._lower_bound) / 2.0
        costs = self.cost_function(time_step, state, solutions)
        min_ind = torch.argmin(costs, dim=-1).long()
        solution = solutions.gather(-1, min_ind)
        return solution
