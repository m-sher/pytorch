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

    The support is the closed interval ``[low, high]``. The mode is at :attr:`peak`,
    which must satisfy ``low <= peak <= high`` and ``high > low``.

    When :attr:`peak` equals :attr:`low` or :attr:`high`, the distribution degenerates
    to a right- or left-triangular density (supported by SciPy's ``triang`` with
    ``c in {0, 1}``).

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Triangular(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.5]))
        >>> m.sample()  # Triangular distributed with low=0, high=1, peak=0.5
        tensor([ 0.1046])

    Args:
        low (float or Tensor): lower range (inclusive)
        high (float or Tensor): upper range (inclusive), must satisfy high > low
        peak (float or Tensor): mode of the distribution, must satisfy low <= peak <= high
    """

    has_rsample = True

    @property
    def arg_constraints(self):
        return {
            "low": constraints.less_than(self.high),
            "high": constraints.greater_than(self.low),
            "peak": constraints.interval(self.low, self.high),
        }

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
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        a, b, c = self.low, self.high, self.peak
        width = b - a
        # PDF = 2(x-a)/((b-a)(c-a)) for a <= x <= c  (when c > a)
        #     = 2(b-x)/((b-a)(b-c)) for c < x <= b  (when c < b)
        # Boundary modes (c==a or c==b): only one side is active; the zero-width side is unused.
        left_denom = width * (c - a)
        right_denom = width * (b - c)
        left_dens = torch.where(
            c > a,
            2 * (value - a) / left_denom.clamp(min=torch.finfo(value.dtype).tiny),
            torch.zeros_like(value),
        )
        right_dens = torch.where(
            c < b,
            2 * (b - value) / right_denom.clamp(min=torch.finfo(value.dtype).tiny),
            torch.zeros_like(value),
        )
        # When peak==low: density is only on the right branch (right-triangular)
        # When peak==high: density is only on the left branch (left-triangular)
        dens = torch.where(value <= c, left_dens, right_dens)
        dens = torch.where((c == a) & (value >= a) & (value <= b), right_dens, dens)
        dens = torch.where((c == b) & (value >= a) & (value <= b), left_dens, dens)
        in_support = (value >= a) & (value <= b)
        return torch.where(in_support, dens.clamp(min=0).log(), torch.full_like(dens, -inf))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        a, b, c = self.low, self.high, self.peak
        width = b - a
        left_cdf = torch.where(
            c > a,
            (value - a).pow(2) / (width * (c - a)),
            torch.zeros_like(value),
        )
        right_cdf = torch.where(
            c < b,
            1 - (b - value).pow(2) / (width * (b - c)),
            torch.ones_like(value),
        )
        result = torch.where(value <= c, left_cdf, right_cdf)
        # peak == low: F(x) = 1 - ((b-x)/(b-a))^2
        result = torch.where(
            c == a,
            1 - (b - value).pow(2) / width.pow(2),
            result,
        )
        # peak == high: F(x) = ((x-a)/(b-a))^2
        result = torch.where(
            c == b,
            (value - a).pow(2) / width.pow(2),
            result,
        )
        result = torch.where(value < a, torch.zeros_like(result), result)
        result = torch.where(value > b, torch.ones_like(result), result)
        return result.clamp(0, 1)

    def icdf(self, value):
        a, b, c = self.low, self.high, self.peak
        width = b - a
        fc = torch.where(c > a, (c - a) / width, torch.zeros_like(width))
        left = a + (width * (c - a).clamp(min=0) * value).sqrt()
        right = b - (width * (b - c).clamp(min=0) * (1 - value)).sqrt()
        result = torch.where(value < fc, left, right)
        # peak == low: only right branch (fc == 0)
        result = torch.where(c == a, b - (width.pow(2) * (1 - value)).sqrt(), result)
        # peak == high: only left branch (fc == 1)
        result = torch.where(c == b, a + (width.pow(2) * value).sqrt(), result)
        return result

    def entropy(self):
        # Entropy of triangular distribution: 0.5 + log((high - low) / 2)
        return 0.5 + (self.high - self.low).log() - math.log(2)
