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
import functools

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

NafCriticState = namedtuple(
    "NafCriticState",
    ['critic1', 'critic2', 'target_critic1', 'target_critic2'])
NafCriticInfo = namedtuple(
    "NafCriticInfo",
    ["q_value1", "q_value2", "target_q_value1", "target_q_value2"])
NafState = namedtuple("NafState", ['critic'])
NafInfo = namedtuple("NafInfo", ["critic"], default_value=())
NafLossInfo = namedtuple('NafLossInfo', ('critic1', 'critic2'))


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
                 ou_scale=1.0,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss_ctor=None,
                 target_update_tau=0.05,
                 target_update_period=1,
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
            critic_loss_ctor (None|OneStepTDLoss): an object for calculating critic
                loss. If None, a default OneStepTDLoss will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network.copy()
        critic_network2 = critic_network.copy()

        train_state_spec = NafState(
            critic=NafCriticState(
                critic1=critic_network.state_spec,
                critic2=critic_network.state_spec,
                target_critic1=critic_network.state_spec,
                target_critic2=critic_network.state_spec))

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
            self.add_optimizer(critic_optimizer,
                               [critic_network1, critic_network2])

        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2

        self._target_critic_network1 = critic_network.copy()
        self._target_critic_network2 = critic_network.copy()

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        # if critic_loss_ctor is None:
        #     critic_loss_ctor = functools.partial(
        #         OneStepTDLoss, debug_summaries=debug_summaries)
        self._critic_loss1 = OneStepTDLoss(
            debug_summaries=debug_summaries, name="critic_loss1")
        self._critic_loss2 = OneStepTDLoss(
            debug_summaries=debug_summaries, name="critic_loss2")

        self._ou_process = common.create_ou_process(action_spec, ou_scale,
                                                    ou_stddev, ou_damping)

        self._update_target = common.get_target_updater(
            models=[self._critic_network1, self._critic_network2],
            target_models=[
                self._target_critic_network1, self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        # how to handle multi-network
        mqv1, state1 = self._critic_network1((time_step.observation, None),
                                             state=state.critic)
        mqv2, state2 = self._critic_network2((time_step.observation, None),
                                             state=state.critic)
        if mqv1[2] < mqv2[2]:
            action = mqv1[0]
            state = state1
        else:
            action = mqv2[0]
            state = state2

        # action = mqv1[0]
        # state = state1

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
                                      "is not supported by NafAlgorithm")
        return self.predict_step(time_step, state, epsilon_greedy=1.0)

    def _critic_train_step(self, exp: Experience, state: NafCriticState):

        mqv1, critic1_state = self._critic_network1(
            (exp.observation, exp.action), state=state.critic1)

        mqv2, critic2_state = self._critic_network2(
            (exp.observation, exp.action), state=state.critic2)

        action_target1, target_critic1_state = self._target_critic_network1(
            (exp.observation, None), state=state.target_critic1, mode="action")

        action_target2, target_critic2_state = self._target_critic_network2(
            (exp.observation, None), state=state.target_critic2, mode="action")

        # double Q-learning (use action instead of exp.action)
        # mqv2, critic2_state = self._critic_network2(
        #     (exp.observation, exp.action), state=state.critic2)

        # swap action
        mqv_target1, target_critic1_state = self._target_critic_network1(
            (exp.observation, action_target1), state=state.target_critic1)
        mqv_target2, target_critic2_state = self._target_critic_network2(
            (exp.observation, action_target2), state=state.target_critic2)

        q_value1 = mqv1[1].view(-1)
        q_value2 = mqv2[1].view(-1)
        # target_q_value1 = mqv_target1[2].view(-1)
        # target_q_value2 = mqv_target2[2].view(-1)

        # target_q_value1 = mqv_target1[2].view(-1)
        # target_q_value2 = mqv_target2[2].view(-1)
        # BUG: should be 1
        target_q_value1 = mqv_target1[1].view(-1)
        target_q_value2 = mqv_target2[1].view(-1)

        # # TODO
        # target_q_value = torch.min(target_q_value1, target_q_value2)

        state = NafCriticState(
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = NafCriticInfo(
            q_value1=q_value1,
            q_value2=q_value2,
            target_q_value1=target_q_value1,
            target_q_value2=target_q_value2)

        return AlgStep(output=None, state=state, info=info)

    def _get_q_value(self, inputs, state=None):

        mqv1, critic_state1 = self._critic_network1(inputs, state=state)
        mqv2, critic_state = self._critic_network2(inputs, state=state)

        q_value = torch.min(mqv1[1], mqv2[1]).view(-1)

        return q_value, critic_state1

    def _get_state_value(self, inputs, state=None):

        mqv1, critic_state1 = self._critic_network1(inputs, state=state)
        mqv2, critic_state = self._critic_network2(inputs, state=state)

        state_value = torch.min(mqv1[2], mqv2[2]).view(-1)

        return state_value, critic_state1

    def train_step(self, exp: Experience, state: NafState):
        cirtic_step = self._critic_train_step(exp=exp, state=state.critic)
        return cirtic_step._replace(
            state=NafState(critic=cirtic_step.state),
            info=NafInfo(critic=cirtic_step.info))

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss1 = self._critic_loss1(
            training_info=training_info,
            value=training_info.info.critic.q_value1,
            target_value=training_info.info.critic.target_q_value1)
        critic_loss2 = self._critic_loss2(
            training_info=training_info,
            value=training_info.info.critic.q_value2,
            target_value=training_info.info.critic.target_q_value2)
        return LossInfo(
            loss=critic_loss1.loss + critic_loss2.loss,
            extra=NafLossInfo(
                critic1=critic_loss1.extra, critic2=critic_loss2.extra))

    def after_update(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network1', '_target_critic_network2']
