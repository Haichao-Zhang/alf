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

import gin
import numpy as np

import tensorflow_probability as tfp
import tensorflow as tf

from tf_agents.distributions.utils import SquashToSpecNormal
from tf_agents.specs import tensor_spec


@gin.configurable
def estimated_entropy(dist: tfp.distributions.Distribution,
                      seed=None,
                      assume_reparametrization=False,
                      num_samples=1,
                      check_numerics=False):
    """Estimate entropy by sampling.

    Use sampling to calculate entropy. The unbiased estimator for entropy is
    -log(p(x)) where x is an unbiased sample of p. However, the gradient of
    -log(p(x)) is not an unbiased estimator of the gradient of entropy. So we
    also calculate a value whose gradient is an unbiased estimator of the
    gradient of entropy. See docs/subtleties_of_estimating_entropy.py for
    detail.

    Args:
        dist (tfp.distributions.Distribution): concerned distribution
        seed (Any): Any Python object convertible to string, supplying the
            initial entropy.
        assume_reparametrization (bool): assume the sample from continuous
            distribution is generated by transforming a fixed distribution
            by a parameterized function. If we can assume this,
            entropy_for_gradient will have lower variance. We make the default
            to be False to be safe.
        num_samples (int): number of random samples used for estimating entropy.
        check_numerics (bool): If true, adds tf.debugging.check_numerics to
            help find NaN / Inf values. For debugging only.
    Returns:
        tuple of (entropy, entropy_for_gradient). entropy_for_gradient is for
        calculating gradient
    """
    sample_shape = (num_samples, )
    single_action = dist.sample(sample_shape=sample_shape, seed=seed)
    if single_action.dtype.is_floating and assume_reparametrization:
        entropy = -dist.log_prob(single_action)
        if check_numerics:
            entropy = tf.debugging.check_numerics(entropy, 'entropy')
        entropy = tf.reduce_mean(entropy, axis=0)
        entropy_for_gradient = entropy
    else:
        entropy = -dist.log_prob(tf.stop_gradient(single_action))
        if check_numerics:
            entropy = tf.debugging.check_numerics(entropy, 'entropy')
        entropy_for_gradient = -0.5 * tf.math.square(entropy)
        entropy = tf.reduce_mean(entropy, axis=0)
        entropy_for_gradient = tf.reduce_mean(entropy_for_gradient, axis=0)
    return entropy, entropy_for_gradient


def entropy_with_fallback(distributions, action_spec, seed=None):
    """Computes total entropy of nested distribution.

    If entropy() of a distribution is not implemented, this function will
    fallback to use sampling to calculate the entropy. It returns two values:
    (entropy, entropy_for_gradient).
    There are two situations:
    * entropy() is implemented. entropy is same as entropy_for_gradient.
    * entropy() is not implemented. We use sampling to calculate entropy. The
        unbiased estimator for entropy is -log(p(x)). However, the gradient of
        -log(p(x)) is not an unbiased estimator of the gradient of entropy. So
        we also calculate a value whose gradient is an unbiased estimator of
        the gradient of entropy. See estimated_entropy() for detail.

    Example:
        with tf.GradientTape() as tape:
            ent, ent_for_grad = entropy_with_fall_back(dist, action_spec)
        tf.summary.scalar("entropy", ent)
        grad = tape.gradient(ent_for_grad, weight)

    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
        action_spec (nested BoundedTensorSpec): A nested tuple representing the
            action spec.
        seed (Any): Any Python object convertible to string, supplying the
            initial entropy.
    Returns:
        tuple of (entropy, entropy_for_gradient). You should use entropy in
        situations where its value is needed, and entropy_for_gradient where
        you need to calculate the gradient of entropy.
    """
    seed_stream = tfp.util.SeedStream(seed=seed, salt='entropy_with_fallback')

    def _calc_outer_rank(dist: tfp.distributions.Distribution, action_spec):
        if isinstance(dist, SquashToSpecNormal):
            # SquashToSpecNormal does not implement the two necessary interface
            # functions of Distribution. So we have to use the original
            # distribution it transforms.
            # SquashToSpecNormal is used by NormalProjectionNetwork with
            # scale_distribution=True
            dist = dist.input_distribution
        return (dist.batch_shape.ndims + dist.event_shape.ndims -
                action_spec.shape.ndims)

    def _compute_entropy(dist: tfp.distributions.Distribution, action_spec):
        if isinstance(dist, SquashToSpecNormal):
            entropy, entropy_for_gradient = estimated_entropy(
                dist, seed=seed_stream())
        else:
            entropy = dist.entropy()
            entropy_for_gradient = entropy

        outer_rank = _calc_outer_rank(dist, action_spec)
        rank = entropy.shape.ndims
        reduce_dims = list(range(outer_rank, rank))
        entropy = tf.reduce_sum(input_tensor=entropy, axis=reduce_dims)
        entropy_for_gradient = tf.reduce_sum(
            input_tensor=entropy_for_gradient, axis=reduce_dims)
        return entropy, entropy_for_gradient

    entropies = list(
        map(_compute_entropy, tf.nest.flatten(distributions),
            tf.nest.flatten(action_spec)))
    entropies_for_gradient = [eg for e, eg in entropies]
    entropies = [e for e, eg in entropies]

    return tf.add_n(entropies), tf.add_n(entropies_for_gradient)


def calc_default_target_entropy(spec):
    """Calc default target entropy
    Args:
        spec (TensorSpec): action spec
    Returns:
    """
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = tensor_spec.is_continuous(spec)
    min_prob = 0.01
    log_mp = np.log(min_prob)
    # continuous: suppose the prob concentrates on a delta of 0.01*(M-m)
    # discrete: ignore the entry of 0.99 and uniformly distribute probs on rest
    e = np.sum([(np.log(M - m) + log_mp
                 if cont else min_prob * (np.log(M - m) - log_mp))
                for m, M, _ in min_max])
    return e


def calc_default_max_entropy(spec, fraction=0.8):
    """Calc default max entropy
    Args:
        spec (TensorSpec): action spec
        fraction (float): this fraction of the theoretical entropy upper bound
            will be used as the max entropy
    Returns:
        A default max entropy for adjusting the entropy weight
    """
    assert fraction <= 1.0 and fraction > 0
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = tensor_spec.is_continuous(spec)
    # use uniform distributions to compute upper bounds
    e = np.sum([(np.log(M - m) * (fraction if M - m > 1 else 1.0 / fraction)
                 if cont else np.log(M - m + 1) * fraction)
                for m, M, _ in min_max])
    return e
