# mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Logistic"]


class Logistic(Distribution):
    r"""
    Creates a Logistic distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Logistic(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Logistic distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (location parameter)
        scale (float or Tensor): scale of the distribution (must be positive)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return (self.scale * math.pi).pow(2) / 3

    @property
    def stddev(self) -> Tensor:
        return self.scale * math.pi / math.sqrt(3)

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Logistic, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Logistic, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        u = u.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return self.loc + self.scale * (u.log() - (-u).log1p())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return -z - 2 * torch.nn.functional.softplus(-z) - self.scale.log()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return torch.sigmoid(z)

    def icdf(self, value):
        finfo = torch.finfo(self.loc.dtype)
        value = value.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return self.loc + self.scale * (value.log() - (1 - value).log())

    def entropy(self):
        return self.scale.log() + 2.0
