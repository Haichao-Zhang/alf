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

import functools
import gin
import hashlib
import numpy as np
import numpy as onp  # tobe removed later
import torch
import torch.distributions as td
from torch.distributions.transforms import _InverseTransform
import torch.nn as nn

import alf.nest as nest
from alf.tensor_specs import TensorSpec

import collections
import weakref


class OUProcess(nn.Module):
    """A zero-mean Ornstein-Uhlenbeck process."""

    def __init__(self, initial_value, damping=0.15, stddev=0.2):
        """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.
        The Ornstein-Uhlenbeck process is a process that generates temporally
        correlated noise via a random walk with damping. This process describes
        the velocity of a particle undergoing brownian motion in the presence of
        friction. This can be useful for exploration in continuous action
        environments with momentum.
        The temporal update equation is:
        `x_next = (1 - damping) * x + N(0, std_dev)`
        Args:
        initial_value: Initial value of the process.
        damping: The rate at which the noise trajectory is damped towards the
            mean. We must have 0 <= damping <= 1, where a value of 0 gives an
            undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
            Hence in most applications a small non-zero value is appropriate.
        stddev: Standard deviation of the Gaussian component.
        """
        super(OUProcess, self).__init__()
        self._damping = damping
        self._stddev = stddev
        self._x = initial_value
        self._x.requires_grad = False

    def forward(self):
        noise = torch.randn_like(self._x) * self._stddev
        self._x.data.copy_((1 - self._damping) * self._x + noise)
        return self._x


def DiagMultivariateNormal(loc, scale_diag):
    """Create a Normal distribution with diagonal variance."""
    return td.Independent(td.Normal(loc, scale_diag), 1)


def _builder_independent(base_builder, reinterpreted_batch_ndims, **kwargs):
    return td.Independent(base_builder(**kwargs), reinterpreted_batch_ndims)


def _builder_transformed(base_builder, transforms, **kwargs):
    return td.TransformedDistribution(base_builder(**kwargs), transforms)


def _get_builder(obj):
    if type(obj) == td.Categorical:
        return td.Categorical, {'logits': obj.logits}
    elif type(obj) == td.Normal:
        return td.Normal, {'loc': obj.mean, 'scale': obj.stddev}
    elif type(obj) == td.Independent:
        builder, params = _get_builder(obj.base_dist)
        new_builder = functools.partial(_builder_independent, builder,
                                        obj.reinterpreted_batch_ndims)
        return new_builder, params
    elif isinstance(obj, TransformedDistribution):
        builder, params = _get_builder(obj.base_dist)
        new_builder = functools.partial(_builder_transformed, builder,
                                        obj.transforms)
        return new_builder, params
    else:
        raise ValueError("Unsupported value type: %s" % type(obj))


def extract_distribution_parameters(dist: td.Distribution):
    """Extract the input parameters of a distribution.

    Args:
        dist (Distribution): distribution from which to extract parameters
    Returns:
        the nest of the input parameter of the distribution
    """
    return _get_builder(dist)[1]


class DistributionSpec(object):
    def __init__(self, builder, input_params_spec):
        """Create a DistributionSpec instance.

        Args:
            builder (Callable): the function which is used to build the
                distribution. The returned value of `builder(input_params)`
                is a Distribution with input parameter as `input_params`
            input_params_spec (nested TensorSpec): the spec for the argument of
                `builder`
        """
        self.builder = builder
        self.input_params_spec = input_params_spec

    def build_distribution(self, input_params):
        """Build a Distribution using `input_params`

        Args:
            input_params (nested Tensor): the parameters for build the
                distribution. It should match `input_params_spec` provided as
                `__init__`
        Returns:
            A Distribution
        """
        nest.assert_same_structure(input_params, self.input_params_spec)
        return self.builder(**input_params)

    @classmethod
    def from_distribution(cls, dist, from_dim=0):
        """Create a DistributionSpec from a Distribution.
        Args:
            dist (Distribution): the Distribution from which the spec is
                extracted.
            from_dim (int): only use the dimenions from this. The reason of
                using `from_dim`>0 is that [0, from_dim) might be batch
                dimension in some scenario.
        Returns:
            DistributionSpec
        """
        builder, input_params = _get_builder(dist)
        input_param_spec = nest.map_structure(
            lambda tensor: TensorSpec.from_tensor(tensor, from_dim),
            input_params)
        return cls(builder, input_param_spec)


def extract_spec(nests, from_dim=1):
    """
    Extract TensorSpec or DistributionSpec for each element of a nested structure.
    It assumes that the first dimension of each element is the batch size.

    Args:
        nests (nested structure): each leaf node of the nested structure is a
            Tensor or Distribution of the same batch size
        from_dim (int): ignore dimension before this when constructing the spec.
    Returns:
        spec (nested structure): each leaf node of the returned nested spec is the
            corresponding spec (excluding batch size) of the element of `nest`
    """

    def _extract_spec(obj):
        if isinstance(obj, torch.Tensor):
            return TensorSpec.from_tensor(obj, from_dim)
        elif isinstance(obj, td.Distribution):
            return DistributionSpec.from_distribution(obj, from_dim)
        else:
            raise ValueError("Unsupported value type: %s" % type(obj))

    return nest.map_structure(_extract_spec, nests)


def to_distribution_param_spec(nests):
    """Convert the DistributionSpecs in nests to their parameter specs.

    Args:
        nests (nested DistributionSpec of TensorSpec):  Each DistributionSpec
            will be converted to a dictionary of the spec of its input Tensor
            parameters.
    Returns:
        A nest of TensorSpec/dict[TensorSpec]. Each leaf is a TensorSpec or a
        dict corresponding to one distribution, with keys as parameter name and
        values as TensorSpecs for the parameters.
    """

    def _to_param_spec(spec):
        if isinstance(spec, DistributionSpec):
            return spec.input_params_spec
        elif isinstance(spec, TensorSpec):
            return spec
        else:
            raise ValueError("Only TensorSpec or DistributionSpec is allowed "
                             "in nest, got %s. nest is %s" % (spec, nests))

    return nest.map_structure(_to_param_spec, nests)


def params_to_distributions(nests, nest_spec):
    """Convert distribution parameters to Distribution, keep Tensors unchanged.
    Args:
        nests (nested tf.Tensor): nested Tensor and dictionary of the Tensor
            parameters of Distribution. Typically, `nest` is obtained using
            `distributions_to_params()`
        nest_spec (nested DistributionSpec and TensorSpec): The distribution
            params will be converted to Distribution according to the
            corresponding DistributionSpec in nest_spec
    Returns:
        nested Distribution/Tensor
    """

    def _to_dist(spec, params):
        if isinstance(spec, DistributionSpec):
            return spec.build_distribution(params)
        elif isinstance(spec, TensorSpec):
            return params
        else:
            raise ValueError(
                "Only DistributionSpec or TensorSpec is allowed "
                "in nest_spec, got %s. nest_spec is %s" % (spec, nest_spec))

    return nest.map_structure_up_to(nest_spec, _to_dist, nest_spec, nests)


def distributions_to_params(nests):
    """Convert distributions to its parameters, keep Tensors unchanged.
    Only returns parameters that have torch.Tensor values.
    Args:
        nests (nested Distribution and Tensor): Each Distribution will be
            converted to dictionary of its Tensor parameters.
    Returns:
        A nest of Tensor/Distribution parameters. Each leaf is a Tensor or a
        dict corresponding to one distribution, with keys as parameter name and
        values as tensors containing parameter values.
    """

    def _to_params(dist_or_tensor):
        if isinstance(dist_or_tensor, td.Distribution):
            return extract_distribution_parameters(dist_or_tensor)
        elif isinstance(dist_or_tensor, torch.Tensor):
            return dist_or_tensor
        else:
            raise ValueError(
                "Only Tensor or Distribution is allowed in nest, ",
                "got %s. nest is %s" % (dist_or_tensor, nests))

    return nest.map_structure(_to_params, nests)


def compute_entropy(distributions):
    """Computes total entropy of nested distribution.
    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
    Returns:
        entropy
    """

    def _compute_entropy(dist: td.Distribution):
        entropy = dist.entropy()
        return entropy

    entropies = nest.map_structure(_compute_entropy, distributions)
    total_entropies = sum(nest.flatten(entropies))
    return total_entropies


def compute_log_probability(distributions, actions):
    """Computes log probability of actions given distribution.

    Args:
        distributions: A possibly batched tuple of distributions.
        actions: A possibly batched action tuple.

    Returns:
        A Tensor representing the log probability of each action in the batch.
    """

    def _compute_log_prob(single_distribution, single_action):
        single_log_prob = single_distribution.log_prob(single_action)
        return single_log_prob

    nest.assert_same_structure(distributions, actions)
    log_probs = nest.map_structure(_compute_log_prob, distributions, actions)
    total_log_probs = sum(nest.flatten(log_probs))
    return total_log_probs


def rsample_action_distribution(nested_distributions):
    """Sample actions from distributions with reparameterization-based sampling
        (rsample) to enable backpropagation.
    Args:
        nested_distributions (nested Distribution): action distributions.
    Returns:
        rsampled actions
    """
    assert all(nest.flatten(nest.map_structure(lambda d: d.has_rsample,
                nested_distributions))), \
            ("all the distributions need to support rsample in order to enable "
            "backpropagation")
    return nest.map_structure(lambda d: d.rsample(), nested_distributions)


def sample_action_distribution(nested_distributions):
    """Sample actions from distributions with conventional sampling without
        enabling backpropagation.
    Args:
        nested_distributions (nested Distribution): action distributions.
    Returns:
        sampled actions
    """
    return nest.map_structure(lambda d: d.sample(), nested_distributions)


def epsilon_greedy_sample(nested_distributions, eps=0.1):
    """Generate greedy sample that maximizes the probability.
    Args:
        nested_distributions (nested Distribution): distribution to sample from
        eps (float): a floating value in [0,1], representing the chance of
            action sampling instead of taking argmax. This can help prevent
            a dead loop in some deterministic environment like Breakout.
    Returns:
        (nested) Tensor
    """

    def greedy_fn(dist):
        # pytorch distribution has no 'mode' operation
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        if isinstance(dist, td.categorical.Categorical):
            greedy_action = torch.argmax(dist.logits, -1)
        elif isinstance(dist, td.normal.Normal):
            greedy_action = dist.mean
        else:
            raise NotImplementedError("Mode sampling not implemented for "
                                      "{cls}".format(cls=type(dist)))
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return sample_action_distribution(nested_distributions)
    else:
        return nest.map_structure(greedy_fn, nested_distributions)


def get_base_dist(dist):
    """Get the base distribution.

    Args:
        dist (td.Distribution)
    Returns:
        the base distribution if dist is td.Independent or
            td.TransformedDistribution, and dist if dist is td.Normal
    Raises:
        NotImplementedError if dist or its based distribution is not
            td.Normal, td.Independent or td.TransformedDistribution
    """
    if isinstance(dist, td.Normal) or isinstance(dist, td.Categorical):
        return dist
    elif isinstance(dist, (td.Independent, td.TransformedDistribution)):
        return get_base_dist(dist.base_dist)
    else:
        raise NotImplementedError(
            "Distribution type %s is not supported" % type(dist))


@gin.configurable
def estimated_entropy(dist, num_samples=1, check_numerics=False):
    """Estimate entropy by sampling.

    Use sampling to calculate entropy. The unbiased estimator for entropy is
    -log(p(x)) where x is an unbiased sample of p. However, the gradient of
    -log(p(x)) is not an unbiased estimator of the gradient of entropy. So we
    also calculate a value whose gradient is an unbiased estimator of the
    gradient of entropy. See docs/subtleties_of_estimating_entropy.py for
    detail.

    Args:
        dist (torch.distributions.Distribution): concerned distribution
        num_samples (int): number of random samples used for estimating entropy.
        check_numerics (bool): If true, adds tf.debugging.check_numerics to
            help find NaN / Inf values. For debugging only.
    Returns:
        tuple of (entropy, entropy_for_gradient). entropy_for_gradient is for
        calculating gradient
    """
    sample_shape = (num_samples, )
    if dist.has_rsample:
        single_action = dist.rsample(sample_shape=sample_shape)
    else:
        single_action = dist.sample(sample_shape=sample_shape)
    if single_action.dtype.is_floating_point and dist.has_rsample:
        entropy = -dist.log_prob(single_action)
        if check_numerics:
            assert torch.all(torch.isfinite(entropy))
        entropy = entropy.mean(dim=0)
        entropy_for_gradient = entropy
    else:
        entropy = -dist.log_prob(single_action.detach())
        if check_numerics:
            assert torch.all(torch.isfinite(entropy))
        entropy_for_gradient = -0.5 * entropy**2
        entropy = entropy.mean(dim=0)
        entropy_for_gradient = entropy_for_gradient.mean(dim=0)
    return entropy, entropy_for_gradient


# NOTE(hnyu): It might be possible to get a closed-form of entropy given a
# Normal as the base dist with only affine transformation?
# It's better (lower variance) than this estimated one.
#
# Something like what TFP does:
# https://github.com/tensorflow/probability/blob/356cfddef026b3339b8f2a81e600acd2ff8e22b4/tensorflow_probability/python/distributions/transformed_distribution.py#L636
# (Probably it's complicated, but we need to spend time figuring out if the
# current estimation is the best way to do this).


# Here, we compute entropy of transformed distributions using sampling.
def entropy_with_fallback(distributions, action_spec):
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
        ent, ent_for_grad = entropy_with_fall_back(dist, action_spec)
        alf.summary.scalar("entropy", ent)
        ent_for_grad.backward()
    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
        action_spec (nested BoundedTensorSpec): A nested tuple representing the
            action spec.
    Returns:
        tuple of (entropy, entropy_for_gradient). You should use entropy in
        situations where its value is needed, and entropy_for_gradient where
        you need to calculate the gradient of entropy.
    """

    def _compute_entropy(dist: td.Distribution, action_spec):
        if isinstance(dist, td.TransformedDistribution):
            # TransformedDistribution is used by NormalProjectionNetwork with
            # scale_distribution=True, in which case we estimate with sampling.
            entropy, entropy_for_gradient = estimated_entropy(dist)
        else:
            entropy = dist.entropy()
            entropy_for_gradient = entropy
        return entropy, entropy_for_gradient

    entropies = list(
        map(_compute_entropy, nest.flatten(distributions),
            nest.flatten(action_spec)))
    entropies, entropies_for_gradient = zip(*entropies)

    return sum(entropies), sum(entropies_for_gradient)


def calc_default_target_entropy(spec):
    """Calc default target entropy
    Args:
        spec (TensorSpec): action spec
    Returns:
    """
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = spec.is_continuous
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
    cont = spec.is_continuous
    # use uniform distributions to compute upper bounds
    e = np.sum([(np.log(M - m) * (fraction if M - m > 1 else 1.0 / fraction)
                 if cont else np.log(M - m + 1) * fraction)
                for m, M, _ in min_max])
    return e


#=============DEBUG code, to be removed
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms must be a Transform or list, but was {}".format(
                    transforms))
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] +
                        [t.event_dim for t in self.transforms])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super(TransformedDistribution, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        base_dist_batch_shape = batch_shape + self.base_dist.batch_shape[len(
            self.batch_shape):]
        new.base_dist = self.base_dist.expand(base_dist_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return self.transforms[
            -1].codomain if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.event_dim)
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y),
            event_dim - len(self.base_dist.event_shape))
        return log_prob

    def sample_log_p(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            print("------forward")
            print(x)
            x = transform(x)
        value = x

        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            print("=======inverse----")
            print(x)
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.event_dim)
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y),
            event_dim - len(self.base_dist.event_shape))

    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value


class _Mapping(
        collections.namedtuple('_Mapping', ['x', 'y', 'ildj', 'kwargs'])):
    """Helper class to make it easier to manage caching in `Bijector`."""

    def __new__(cls, x=None, y=None, ildj=None, kwargs=None):
        """Custom __new__ so namedtuple items have defaults.
    Args:
      x: `Tensor` or None. Input to forward; output of inverse.
      y: `Tensor` or None. Input to inverse; output of forward.
      ildj: `Tensor`. This is the (un-reduce_sum'ed) inverse log det jacobian.
      kwargs: Python dictionary. Extra args supplied to forward/inverse/etc
        functions.
    Returns:
      mapping: New instance of _Mapping.
    """
        return super(_Mapping, cls).__new__(cls, x, y, ildj, kwargs)

    @property
    def subkey(self):
        """Returns subkey used for caching (nested under either `x` or `y`)."""
        return self._deep_tuple(self.kwargs)

    def merge(self, x=None, y=None, ildj=None, kwargs=None, mapping=None):
        """Returns new _Mapping with args merged with self.
    Args:
      x: `Tensor` or None. Input to forward; output of inverse.
      y: `Tensor` or None. Input to inverse; output of forward.
      ildj: `Tensor`. This is the (un-reduce_sum'ed) inverse log det jacobian.
      kwargs: Python dictionary. Extra args supplied to forward/inverse/etc
        functions.
      mapping: Instance of _Mapping to merge. Can only be specified if no other
        arg is specified.
    Returns:
      mapping: New instance of `_Mapping` which has inputs merged with self.
    Raises:
      ValueError: if mapping and any other arg is not `None`.
    """
        if mapping is None:
            mapping = _Mapping(x=x, y=y, ildj=ildj, kwargs=kwargs)
        elif any(arg is not None for arg in [x, y, ildj, kwargs]):
            raise ValueError(
                'Cannot simultaneously specify mapping and individual '
                'arguments.')

        return _Mapping(
            x=self._merge(self.x, mapping.x),
            y=self._merge(self.y, mapping.y),
            ildj=self._merge(self.ildj, mapping.ildj),
            kwargs=self._merge(self.kwargs, mapping.kwargs, use_equals=True))

    def remove(self, field):
        """To support weak referencing, removes cache key from the cache value."""
        return _Mapping(
            x=None if field == 'x' else self.x,
            y=None if field == 'y' else self.y,
            ildj=self.ildj,
            kwargs=self.kwargs)

    def _merge(self, old, new, use_equals=False):
        """Helper to merge which handles merging one value."""

        def generic_to_array(x):
            if isinstance(x, np.generic):
                x = np.array(x)
            if isinstance(x, onp.ndarray):
                x.flags.writeable = False
            return x

        if old is None:
            return generic_to_array(new)
        if new is None:
            return generic_to_array(old)
        if (old == new) if use_equals else (old is new):
            return generic_to_array(old)
        raise ValueError('Incompatible values: %s != %s' % (old, new))

    def _deep_tuple(self, x):
        """Converts nested `tuple`, `list`, or `dict` to nested `tuple`."""
        if isinstance(x, dict):
            return self._deep_tuple(tuple(sorted(x.items())))
        elif isinstance(x, (list, tuple)):
            return tuple(map(self._deep_tuple, x))
        elif isinstance(x, torch.Tensor):
            return x.experimental_ref()

        return x


class WeakKeyDefaultDict(dict):
    """`WeakKeyDictionary` which always adds `defaultdict(dict)` in getitem."""

    # Q:Why not subclass `collections.defaultdict`?
    # Subclassing collections.defaultdict means we have a more complicated `repr`,
    # `str` which makes debugging the bijector cache more tedious. Additionally it
    # means we need to think about passing through __init__ args but manually
    # specifying the `default_factory`. That is, just overriding `__missing__`
    # ends up being a lot cleaner.

    # Q:Why not subclass `weakref.WeakKeyDictionary`?
    # `weakref.WeakKeyDictionary` has an even worse `repr`, `str` than
    # collections.defaultdict. Plus, since we want explicit control over how the
    # keys are created we need to override __getitem__ which is the only feature
    # of `weakref.WeakKeyDictionary` we're using.

    # This is the 'WeakKey' part.
    def __getitem__(self, key):
        weak_key = HashableWeakRef(key, self.pop)
        return super(WeakKeyDefaultDict, self).__getitem__(weak_key)

    # This is the 'DefaultDict' part.
    def __missing__(self, key):
        assert isinstance(key, HashableWeakRef)  # Can't happen.
        return super(WeakKeyDefaultDict, self).setdefault(key, {})

    # Everything that follows is only useful to help make debugging easier.

    def __contains__(self, key):
        return super(WeakKeyDefaultDict, self).__contains__(
            HashableWeakRef(key))

    # We don't want mutation except through __getitem__.

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def setdefault(self, *args, **kwargs):
        raise NotImplementedError()


class HashableWeakRef(weakref.ref):
    """weakref.ref which makes np.array objects hashable.
    We take care to ensure that a hash can still be provided in the case that the
    ref has been cleaned up. This ensures that the WeakKeyDefaultDict doesn't
    suffer memory leaks by failing to clean up HashableWeakRef key objects whose
    referrents have gone out of scope and been destroyed (as in
    https://github.com/tensorflow/probability/issues/647).
    """

    def __init__(self, referrent, callback=None):
        # Note that -1 is a safe sentinal value for detecting whether hash has been
        # initialized, since python doesn't allow hashes of -1. In particular,
        #
        # ```python
        # a = -1
        # print(hash(a))
        # ==> -2
        # ```
        self._last_known_hash = -1
        super(HashableWeakRef, self).__init__(referrent, callback)

    def __hash__(self):
        x = self()
        # If the ref has been cleaned up, fall back to the last known hash value.
        if x is None:
            if self._last_known_hash == -1:
                raise ValueError(
                    'HashableWeakRef\'s ref has been cleaned up but the hash was never '
                    'known. It may not be able to be cleaned up as a result.')
            return self._last_known_hash
        if not isinstance(x, onp.ndarray):
            result = id(x)
        elif isinstance(x, np.generic):
            raise ValueError('Unable to weakref np.generic')
        # Note: The following logic can never be reached by the public API because
        # the bijector base class always calls `convert_to_tensor` before accessing
        # the cache.
        else:
            x.flags.writeable = False
            result = hash(str(x.__array_interface__) + str(id(x)))
        self._last_known_hash = result
        return result

    def __repr__(self):
        return repr(self())

    def __str__(self):
        return str(self())

    def __eq__(self, other):
        # If either ref has been cleaned up, fall back to comparing the
        # HashableWeakRef instance ids, following what weakref equality checks do
        # (https://github.com/python/cpython/blob/master/Objects/weakrefobject.c#L196)
        if self() is None or other() is None:
            return id(self) == id(other)
        x = self()
        if isinstance(x, np.generic):
            raise ValueError('Unable to weakref np.generic')
        y = other()
        ids_are_equal = id(x) == id(y)
        if not isinstance(x, onp.ndarray):
            return ids_are_equal
        return (isinstance(y, onp.ndarray)
                and x.__array_interface__ == y.__array_interface__
                and ids_are_equal)


class StableTransform(Transform, object):
    """
    Abstract class for invertable transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.
    Caching is useful for transforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::
        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.
    However the following will error when caching due to dependency reversal::
        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x
    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.
    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
        event_dim (int): Number of dimensions that are correlated together in
            the transform ``event_shape``. This should be 0 for pointwise
            transforms, 1 for transforms that act jointly on vectors, 2 for
            transforms that act jointly on matrices, etc.
    """
    bijective = False
    event_dim = 0

    def __init__(self):
        super(StableTransform, self).__init__()
        self._inv = None
        # self._from_y = self._no_dependency(WeakKeyDefaultDict())
        # self._from_x = self._no_dependency(WeakKeyDefaultDict())
        self._from_y = WeakKeyDefaultDict()
        self._from_x = WeakKeyDefaultDict()

    def _cache_by_x(self, mapping):
        """Helper which stores new mapping info in the forward dict."""
        # Merging from lookup is an added check that we're not overwriting anything
        # which is not None.
        mapping = mapping.merge(
            mapping=self._lookup(mapping.x, mapping.y, mapping.kwargs))
        if mapping.x is None:
            raise ValueError('Caching expects x to be known, i.e., not None.')
        self._from_x[mapping.x][mapping.subkey] = mapping.remove('x')

    def _cache_by_y(self, mapping):
        """Helper which stores new mapping info in the inverse dict."""
        # Merging from lookup is an added check that we're not overwriting anything
        # which is not None.
        mapping = mapping.merge(
            mapping=self._lookup(mapping.x, mapping.y, mapping.kwargs))
        if mapping.y is None:
            raise ValueError('Caching expects y to be known, i.e., not None.')
        self._from_y[mapping.y][mapping.subkey] = mapping.remove('y')

    def _cache_update(self, mapping):
        """Helper which updates only those cached entries that already exist."""
        if mapping.x is not None and mapping.subkey in self._from_x[mapping.x]:
            self._cache_by_x(mapping)
        if mapping.y is not None and mapping.subkey in self._from_y[mapping.y]:
            self._cache_by_y(mapping)

    def _lookup(self, x=None, y=None, kwargs=None):
        """Helper which retrieves mapping info from forward/inverse dicts."""
        mapping = _Mapping(x=x, y=y, kwargs=kwargs)
        subkey = mapping.subkey
        if x is not None:
            # We removed x at caching time. Add it back if we lookup successfully.
            mapping = self._from_x[x].get(subkey, mapping).merge(x=x)
        if y is not None:
            # We removed y at caching time. Add it back if we lookup successfully.
            mapping = self._from_y[y].get(subkey, mapping).merge(y=y)
        return mapping

    # pytorch code -----------------
    @property
    def inv(self):
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    @property
    def sign(self):
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
        raise NotImplementedError

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def __call__(self, x):
        """
        Computes the transform `x => y`.
        """
        if not self.bijective:  # No caching for non-injective
            return self._call(x)
        mapping = self._lookup(x=x)
        if mapping.y is not None:
            return mapping.y
        mapping = mapping.merge(y=self._call(x))
        # It's most important to cache the y->x mapping, because computing
        # inverse(forward(y)) may be numerically unstable / lossy. Caching the
        # x->y mapping only saves work. Since python doesn't support ephemerons,
        # we cannot be simultaneously weak-keyed on both x and y, so we choose y.
        self._cache_by_y(mapping)
        return mapping.y

    def _inv_call(self, y):
        """
        Inverts the transform `y => x`.
        """
        if not self.bijective:  # No caching for non-injective
            return self._inverse(y)
        mapping = self._lookup(y=y)
        if mapping.x is not None:
            return mapping.x
        mapping = mapping.merge(x=self._inverse(y))
        # It's most important to cache the x->y mapping, because computing
        # forward(inverse(y)) may be numerically unstable / lossy. Caching the
        # y->x mapping only saves work. Since python doesn't support ephemerons,
        # we cannot be simultaneously weak-keyed on both x and y, so we choose x.
        self._cache_by_x(mapping)
        return mapping.x

    def _call(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class StableTanh(Transform):
    """Invertable transformation (bijector) that computes `Y = tanh(X)`,
    therefore `Y in (-1, 1)`.

    This can be achieved by an affine transform of the Sigmoid transformation,
    i.e., it is equivalent to applying a list of transformations sequentially:
        ```
        transforms = [td.AffineTransform(loc=0, scale=2)
                        td.SigmoidTransform(),
                        td.AffineTransform(
                            loc=-1,
                            scale=2]
        ```
    However, using the `StableTanh` transformation directly is more numerically
    stable.
    """
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        # We use cache by default as it is numerically unstable for inversion
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, StableTanh)

    def _call(self, x):
        y = torch.tanh(x)
        return y
        # return torch.clamp(y, -0.99999997, 0.99999997)

    def _inverse(self, y):
        # Based on https://github.com/tensorflow/agents/commit/dfb8c85a01d65832b05315928c010336df13f7b9#diff-a572e559b953f965c5c2cd1b9ded2c7b

        # 0.99999997 is the maximum value such that atanh(x) is valid for both
        # float32 and float64
        def _atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        y = torch.where(
            torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        return _atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (
            torch.log(torch.tensor(2.0, dtype=x.dtype, requires_grad=False)) -
            x - nn.functional.softplus(-2.0 * x))
