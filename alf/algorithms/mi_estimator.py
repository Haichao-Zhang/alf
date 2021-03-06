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
"""Mutual Information Estimator."""

import math

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks.utils import BatchSquash
from tf_agents.specs import tensor_spec
from tf_agents.utils.nest_utils import get_outer_rank
from tf_agents.utils import common as tfa_common

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.averager import ScalarAdaptiveAverager
from alf.utils.data_buffer import DataBuffer
from alf.utils.encoding_network import EncodingNetwork, TFAEncodingNetwork
from alf.utils.nest_utils import get_nest_batch_size
from alf.utils import math_ops
from alf.layers import NestConcatenate


class MIEstimator(Algorithm):
    """Mutual Infomation Estimator.

    Implements several mutual information estimator from
    Belghazi et al "Mutual Information Neural Estimation"
    http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf
    Hjelm et al "Learning Deep Representations by Mutual Information Estimation
    and Maximization" https://arxiv.org/pdf/1808.06670.pdf

    Currently 3 types of estimator are implemented, which are based on the
    following variational lower bounds:
    * 'DV':  sup_T E_P(T) - log E_Q(exp(T))
    * 'KLD': sup_T E_P(T) - E_Q(exp(T)) + 1
    * 'JSD': sup_T -E_P(softplus(-T))) - E_Q(solftplus(T)) + log(4)
    * 'ML': sup_q E_P(log(q(y|x)) - log(P(y)))

    where P is the joint distribution of X and Y, and Q is the product marginal
    distribution of P. Both DV and KLD are lower bounds for KLD(P||Q)=MI(X, Y).
    However, JSD is not a lower bound for mutual information, it is a lower
    bound for JSD(P||Q), which is closely correlated with MI as pointed out in
    Hjelm et al.

    For ML, P(y) is the margianl distribution of y, and it needs to be provided.
    The current implementation uses a normal distribution with diagonal variance
    for q(y|x). So it only support continous `y`. If P(y|x) can be reasonably
    approximated as an diagonal normal distribution and P(y) is known, then 'ML'
    may give better estimation for the mutual information.

    Assumming the function class of T is rich enough to represent any function,
    for KLD and JSD, T will converge to log(P/Q) and hence E_P(T) can also be
    used as an estimator of KLD(P||Q)=MI(X,Y). For DV, T will converge to
    log(P/Q) + c, where c=log E_Q(exp(T)).

    Among 'DV', 'KLD' and 'JSD',  'DV' and 'KLD' seem to give a better estimation
    of PMI than 'JSD'. But 'JSD' might be numerically more stable than 'DV' and
    'KLD' because of the use of softplus instead of exp. And 'DV' is more stable
    than 'KLD' because of the logarithm.

    Several strategies are implemented in order to estimate E_Q(.):
    * 'buffer': store y to a buffer and randomly retrieve samples from the
       buffer.
    * 'double_buffer': stroe both x and y to buffers and randomly retrieve
       samples from the two buffers.
    * 'shuffle': randomly shuffle batch y
    * 'shift': shift batch y by one sample, i.e.
      tf.concat([y[-1:, ...], y[0:-1, ...]], axis=0)
    * direct sampling: You can also provide the marginal distribution of y to
      train_step(). In this case, sampler is ignored and samples of y for
      estimating E_Q(.) are sampled from y_distribution.

    If you need the gradient of y, you should use sampler 'shift' and 'shuffle'.

    Among these, 'buffer' and 'shift' seem to perform better and 'shuffle'
    performs worst. 'buffer' incurs additional storage cost. 'shift' has the
    assumption that y samples from one batch are independent. If the additional
    memory is not a concern, we recommend 'buffer' sampler so that there is no
    need to worry about the assumption of independence.

    MIEstimator can be also used to estimate conditional mutual information
    MI(X,Y|Z) using 'KLD', 'JSD' or 'ML'. In this case, you should let `x` to
    represent X and Z, and `y` to represent Y. And when calling train_step(),
    you need to provide `y_distribution` which is the distribution P(Y|z).
    Note that 'DV' cannot be used for estimating conditional mutual information.
    See mi_estimator_test.py for example.
    """

    def __init__(self,
                 x_spec,
                 y_spec,
                 model=None,
                 fc_layers=(256, ),
                 sampler='buffer',
                 buffer_size=65536,
                 optimizer: tf.optimizers.Optimizer = None,
                 estimator_type='DV',
                 averager=ScalarAdaptiveAverager(),
                 name="MIEstimator"):
        """Create a MIEstimator.

        Args:
            x_spec (nested TensorSpec): spec of x
            y_spec (nested TensorSpec): spec of y
            model (Network): can be called as model([x, y]) and return a Tensor
                with shape=[batch_size, 1]. If None, a default MLP with
                fc_layers will be created.
            fc_layers (tuple[int]): size of hidden layers. Only used if model is
                None.
            sampler (str): type of sampler used to get samples from marginal
                distribution, should be one of ['buffer', 'double_buffer',
                'shuffle', 'shift']
            buffer_size (int): capacity of buffer for storing y for sampler
                'buffer' and 'double_buffer'
            optimzer (tf.optimizers.Optimzer): optimizer
            estimator_type (str): one of 'DV', 'KLD' or 'JSD'
            averager (EMAverager): averager used to maintain a moving average
                of exp(T). Only used for 'DV' estimator
            name (str): name of this estimator
        """
        assert estimator_type in ['ML', 'DV', 'KLD', 'JSD'
                                  ], "Wrong estimator_type %s" % estimator_type
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._x_spec = x_spec
        self._y_spec = y_spec
        if model is None:
            if estimator_type == 'ML':
                model = TFAEncodingNetwork(
                    name="MIEstimator",
                    input_tensor_spec=x_spec,
                    fc_layer_params=fc_layers,
                    preprocessing_combiner=NestConcatenate(axis=-1))
            else:
                model = EncodingNetwork(
                    name="MIEstimator",
                    input_tensor_spec=[x_spec, y_spec],
                    fc_layer_params=fc_layers,
                    last_layer_size=1)
        self._model = model
        self._type = estimator_type
        if sampler == 'buffer':
            self._y_buffer = DataBuffer(y_spec, capacity=buffer_size)
            self._sampler = self._buffer_sampler
        elif sampler == 'double_buffer':
            self._x_buffer = DataBuffer(x_spec, capacity=buffer_size)
            self._y_buffer = DataBuffer(y_spec, capacity=buffer_size)
            self._sampler = self._double_buffer_sampler
        elif sampler == 'shuffle':
            self._sampler = self._shuffle_sampler
        elif sampler == 'shift':
            self._sampler = self._shift_sampler
        else:
            raise TypeError("Wrong type for sampler %s" % sampler)

        if estimator_type == 'DV':
            self._mean_averager = averager
        if estimator_type == 'ML':
            assert isinstance(
                y_spec,
                tf.TensorSpec), ("Currently, 'ML' does "
                                 "not support nested y_spec: %s" % y_spec)
            assert tensor_spec.is_continuous(y_spec), (
                "Currently, 'ML' does "
                "not support discreted y_spec: %s" % y_spec)
            self._delta_loc_layer = tf.keras.layers.Dense(
                y_spec.shape[-1],
                kernel_initializer=tf.initializers.Zeros(),
                bias_initializer=tf.initializers.Zeros(),
                name='delta_loc_layer')
            self._delta_scale_layer = tf.keras.layers.Dense(
                y_spec.shape[-1],
                kernel_initializer=tf.initializers.Zeros(),
                bias_initializer=tf.keras.initializers.Constant(
                    value=math.log(math.e - 1)),
                name='delta_scale_layer')

    def _buffer_sampler(self, x, y):
        batch_size = get_nest_batch_size(y)
        if self._y_buffer.current_size >= batch_size:
            y1 = self._y_buffer.get_batch(batch_size)
            self._y_buffer.add_batch(y)
        else:
            self._y_buffer.add_batch(y)
            y1 = self._y_buffer.get_batch(batch_size)
        # It seems that tf.stop_gradient() should be unnesessary. But somehow
        # TF will crash without this stop_gradient
        return x, tf.nest.map_structure(tf.stop_gradient, y1)

    def _double_buffer_sampler(self, x, y):
        batch_size = get_nest_batch_size(y)
        self._x_buffer.add_batch(x)
        x1 = self._x_buffer.get_batch(batch_size)
        self._y_buffer.add_batch(y)
        y1 = self._y_buffer.get_batch(batch_size)
        return x1, y1

    def _shuffle_sampler(self, x, y):
        return x, math_ops.shuffle(y)

    def _shift_sampler(self, x, y):
        def _shift(y):
            return tf.concat([y[-1:, ...], y[0:-1, ...]], axis=0)

        return x, tf.nest.map_structure(_shift, y)

    def train_step(self, inputs, y_distribution=None, state=None):
        """Perform training on one batch of inputs.

        Args:
            inputs (tuple(nested Tensor, nested Tensor)): tuple of x and y
            y_distribution (nested tfp.distributions.Distribution): distribution
                for the marginal distribution of y. If None, will use the
                sampling method `sampler` provided at constructor to generate
                the samples for the marginal distribution of Y.
            state: not used
        Returns:
            AlgorithmStep
                outputs (Tensor): shape=[batch_size], its mean is the estimated
                    MI for estimator 'KL', 'DV' and 'KLD', and Jensen-Shannon
                    divergence for estimator 'JSD'
                state: not used
                info (LossInfo): info.loss is the loss
        """
        x, y = inputs

        if self._type == 'ML':
            return self._ml_step(x, y, y_distribution)

        num_outer_dims = get_outer_rank(x, self._x_spec)
        batch_squash = BatchSquash(num_outer_dims)
        x = batch_squash.flatten(x)
        y = batch_squash.flatten(y)
        if y_distribution is None:
            x1, y1 = self._sampler(x, y)
        else:
            x1 = x
            y1 = y_distribution.sample()
            y1 = batch_squash.flatten(y1)

        log_ratio = self._model([x, y])[0]
        t1 = self._model([x1, y1])[0]

        if self._type == 'DV':
            ratio = tf.math.exp(tf.minimum(t1, 20))
            mean = tf.stop_gradient(tf.reduce_mean(ratio))
            if self._mean_averager:
                self._mean_averager.update(mean)
                unbiased_mean = tf.stop_gradient(self._mean_averager.get())
            else:
                unbiased_mean = mean
            # estimated MI = reduce_mean(mi)
            # ratio/mean-1 does not contribute to the final estimated MI, since
            # mean(ratio/mean-1) = 0. We add it so that we can have an estimation
            # of the variance of the MI estimator
            mi = log_ratio - (tf.math.log(mean) + ratio / mean - 1)
            loss = ratio / unbiased_mean - log_ratio
        elif self._type == 'KLD':
            ratio = tf.math.exp(tf.minimum(t1, 20))
            mi = log_ratio - ratio + 1
            loss = -mi
        elif self._type == 'JSD':
            mi = -tf.nn.softplus(-log_ratio) - tf.nn.softplus(t1) + math.log(4)
            loss = -mi
        mi = batch_squash.unflatten(mi)
        loss = batch_squash.unflatten(loss)

        return AlgorithmStep(
            outputs=mi, state=(), info=LossInfo(loss, extra=()))

    def _ml_pmi(self, x, y, y_distribution):
        num_outer_dims = get_outer_rank(x, self._x_spec)
        hidden = self._model(x)[0]
        batch_squash = BatchSquash(num_outer_dims)
        hidden = batch_squash.flatten(hidden)
        delta_loc = self._delta_loc_layer(hidden)
        delta_scale = tf.nn.softplus(self._delta_scale_layer(hidden))
        delta_loc = batch_squash.unflatten(delta_loc)
        delta_scale = batch_squash.unflatten(delta_scale)
        y_given_x_dist = tfp.distributions.Normal(
            loc=y_distribution.loc + delta_loc,
            scale=y_distribution.scale * delta_scale)

        # Because Normal.event_shape is [], the result of Normal.log_prob() is
        # the probabilities of individual dimensions. So we need to use
        # tfa_common.log_probability() instead.
        # TODO: implement a normal distribution with non-scalar event shape.
        pmi = tfa_common.log_probability(y_given_x_dist, y, self._y_spec)
        pmi -= tf.stop_gradient(
            tfa_common.log_probability(y_distribution, y, self._y_spec))
        return pmi

    def _ml_step(self, x, y, y_distribution):
        pmi = self._ml_pmi(x, y, y_distribution)
        return AlgorithmStep(outputs=pmi, state=(), info=LossInfo(loss=-pmi))

    def calc_pmi(self, x, y, y_distribution=None):
        """Return estimated pointwise mutual information.

        The pointwise mutual information is defined as:
            log P(x|y)/P(x) = log P(y|x)/P(y)

        Args:
            x (tf.Tensor): x
            y (tf.Tensor): y
            y_distribution (tfp.distributions.Normal): needs to be provided for
                'ML' estimator.
        Returns:
            tf.Tensor: pointwise mutual information between x and y
        """
        if self._type == 'ML':
            assert y_distribution is not None, "y_distribution needs to be provided"
            return self._ml_pmi(x, y, y_distribution)
        log_ratio = self._model([x, y])[0]
        log_ratio = tf.squeeze(log_ratio, axis=-1)
        if self._type == 'DV':
            log_ratio -= tf.math.log(self._mean_averager.get())
        return log_ratio
