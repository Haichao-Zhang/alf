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
"""Normalized Advantage Function (NAF)."""

import numpy as np
import gin

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, TrainingInfo
from alf.nest import nest
from alf.networks import NafCriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, spec_utils

NafCriticState = namedtuple("NafCriticState", ['critic', 'target_critic'])
NafCriticInfo = namedtuple("NafCriticInfo", ["q_value", "target_q_value"])
NafState = namedtuple("NafState", ['critic'])
NafInfo = namedtuple("NafInfo", ["critic"], default_value=())
NafLossInfo = namedtuple('NafLossInfo', ('critic'))


@gin.configurable
class NafAlgorithm(OffPolicyAlgorithm):
    """Normalized Advantage Function (NAF).

    Reference:
    Gu et al "Continuous Deep Q-Learning with Model-based Acceleration"
    https://arxiv.org/pdf/1603.00748.pdf
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 critic_network: NafCriticNetwork,
                 env=None,
                 config: TrainerConfig = None,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 critic_optimizer=None,
                 gradient_clipping=None,
                 debug_summaries=False,
                 name="NafAlgorithm"):
        """Create a NafAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            critic_network (Network): The network will be called with
                call(observation, action).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck
                (OU) noise added in the default collect policy.
            ou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            critic_loss (None|OneStepTDLoss): an object for calculating critic
                loss. If None, a default OneStepTDLoss will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        train_state_spec = NafState(
            critic=NafCriticState(
                critic=critic_network.state_spec,
                target_critic=critic_network.state_spec))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)

        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_network])

        self._critic_network = critic_network
        self._target_critic_network = critic_network.copy()

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        self._ou_process = common.create_ou_process(action_spec, ou_stddev,
                                                    ou_damping)

        self._update_target = common.get_target_updater(
            models=[self._critic_network],
            target_models=[self._target_critic_network],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        mqv, state = self._critic_network((time_step.observation, None),
                                          state=state.critic)
        action = mqv[0]

        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)

        def _sample(a, ou):
            if torch.rand(1) < epsilon_greedy:
                return a + ou()
            else:
                return a

        noisy_action = nest.map_structure(_sample, action, self._ou_process)
        noisy_action = nest.map_structure(spec_utils.clip_to_spec,
                                          noisy_action, self._action_spec)
        state = empty_state._replace(critic=state)
        return AlgStep(output=noisy_action, state=state, info=NafInfo())

    def rollout_step(self, time_step: TimeStep, state=None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by DdpgAlgorithm")
        return self.predict_step(time_step, state, epsilon_greedy=1.0)

    def _critic_train_step(self, exp: Experience, state: NafCriticState):

        mqv_target, target_critic_state = self._target_critic_network(
            (exp.observation, None), state=state.target_critic)

        mqv, critic_state = self._critic_network((exp.observation, exp.action),
                                                 state=state.critic)

        action = mqv[0]
        q_value = mqv[2].view(-1)
        target_q_value = mqv_target[2].view(-1)

        state = NafCriticState(
            critic=critic_state, target_critic=target_critic_state)

        info = NafCriticInfo(q_value=q_value, target_q_value=target_q_value)

        return AlgStep(output=action, state=state, info=info)

    def train_step(self, exp: Experience, state: NafState):
        cirtic_step = self._critic_train_step(exp=exp, state=state.critic)
        return cirtic_step._replace(
            state=NafState(critic=cirtic_step.state),
            info=NafInfo(critic=cirtic_step.info))

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._critic_loss(
            training_info=training_info,
            value=training_info.info.critic.q_value,
            target_value=training_info.info.critic.target_q_value)
        return LossInfo(
            loss=critic_loss.loss, extra=NafLossInfo(critic=critic_loss.extra))

    def after_update(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network']
