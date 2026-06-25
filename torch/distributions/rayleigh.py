# mypy: allow-untyped-defs
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import euler_constant
from torch.types import _Number, _size


__all__ = ["Rayleigh"]


class Rayleigh(Distribution):
    r"""
    Creates a Rayleigh distribution parameterized by :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Rayleigh(torch.tensor([1.0]))
        >>> m.sample()  # Rayleigh distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the distribution (must be positive)
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(math.pi / 2)

    @property
    def mode(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (4 - math.pi) / 2

    @property
    def stddev(self) -> Tensor:
        return self.scale * math.sqrt((4 - math.pi) / 2)

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale = torch.as_tensor(scale)
        if isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Rayleigh, _instance)
        batch_shape = torch.Size(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Rayleigh, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.scale.dtype)
        u = torch.rand(shape, dtype=self.scale.dtype, device=self.scale.device)
        u = u.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return self.scale * (-2 * u.log()).sqrt()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        scale_sq = self.scale.pow(2)
        # log(x / sigma^2) - x^2 / (2 sigma^2)
        log_prob = value.log() - scale_sq.log() - value.pow(2) / (2 * scale_sq)
        return torch.where(
            value >= 0,
            log_prob,
            torch.full_like(log_prob, -inf),
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cdf = 1 - torch.exp(-value.pow(2) / (2 * self.scale.pow(2)))
        return torch.where(value >= 0, cdf, torch.zeros_like(cdf))

    def icdf(self, value):
        finfo = torch.finfo(self.scale.dtype)
        value = value.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return self.scale * (-2 * (1 - value).log()).sqrt()

    def entropy(self):
        # 1 + log(sigma / sqrt(2)) + gamma/2, where gamma is Euler's constant
        return 1 + self.scale.log() - 0.5 * math.log(2) + 0.5 * euler_constant
