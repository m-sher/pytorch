# mypy: allow-untyped-defs
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Triangular"]


class Triangular(Distribution):
    r"""
    Creates a Triangular distribution parameterized by :attr:`low`, :attr:`high`, and :attr:`peak`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Triangular(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.5]))
        >>> m.sample()  # Triangular distributed with low=0, high=1, peak=0.5
        tensor([ 0.1046])

    Args:
        low (float or Tensor): lower range (inclusive)
        high (float or Tensor): upper range (exclusive), must satisfy high > low
        peak (float or Tensor): mode of the distribution, must satisfy low <= peak <= high
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
        "peak": constraints.dependent(is_discrete=False, event_dim=0),
    }
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return (self.low + self.high + self.peak) / 3

    @property
    def mode(self) -> Tensor:
        return self.peak

    @property
    def variance(self) -> Tensor:
        a, b, c = self.low, self.high, self.peak
        return (a.pow(2) + b.pow(2) + c.pow(2) - a * b - a * c - b * c) / 18

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        peak: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.low, self.high, self.peak = broadcast_all(low, high, peak)
        if isinstance(low, _Number) and isinstance(high, _Number) and isinstance(peak, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Triangular, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        new.peak = self.peak.expand(batch_shape)
        super(Triangular, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        a, b, c = self.low, self.high, self.peak
        fc = (c - a) / (b - a)
        # Inverse CDF: left side when u < F(c), right side otherwise
        left = a + ((b - a) * (c - a) * u).sqrt()
        right = b - ((b - a) * (b - c) * (1 - u)).sqrt()
        return torch.where(u < fc, left, right)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        a, b, c = self.low, self.high, self.peak
        # PDF = 2(x-a)/((b-a)(c-a)) for a <= x <= c
        #     = 2(b-x)/((b-a)(b-c)) for c < x <= b
        left_dens = 2 * (value - a) / ((b - a) * (c - a))
        right_dens = 2 * (b - value) / ((b - a) * (b - c))
        dens = torch.where(value <= c, left_dens, right_dens)
        in_support = (value >= a) & (value <= b)
        return torch.where(in_support, dens.log(), torch.full_like(dens, -inf))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        a, b, c = self.low, self.high, self.peak
        left_cdf = (value - a).pow(2) / ((b - a) * (c - a))
        right_cdf = 1 - (b - value).pow(2) / ((b - a) * (b - c))
        result = torch.where(value <= c, left_cdf, right_cdf)
        result = torch.where(value < a, torch.zeros_like(result), result)
        result = torch.where(value > b, torch.ones_like(result), result)
        return result

    def icdf(self, value):
        a, b, c = self.low, self.high, self.peak
        fc = (c - a) / (b - a)
        left = a + ((b - a) * (c - a) * value).sqrt()
        right = b - ((b - a) * (b - c) * (1 - value)).sqrt()
        return torch.where(value < fc, left, right)

    def entropy(self):
        # Entropy of triangular distribution: 0.5 - log(2 / (high - low))
        # = 0.5 + log((high - low) / 2)
        return 0.5 + (self.high - self.low).log() - math.log(2)
