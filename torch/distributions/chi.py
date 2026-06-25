# mypy: allow-untyped-defs
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform


__all__ = ["Chi"]


class Chi(TransformedDistribution):
    r"""
    Creates a Chi distribution parameterized by shape parameter :attr:`df` (degrees of freedom).

    This is equivalent to taking the square root of a ``Chi2(df)`` random variable,
    or equivalently ``Gamma(alpha=0.5*df, beta=0.5)`` under a square-root transform.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi(torch.tensor([1.0]))
        >>> m.sample()  # Chi distributed with df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom (shape parameter, must be positive)
    """

    arg_constraints = {"df": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Gamma

    def __init__(
        self,
        df: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # Chi(df) = sqrt(Chi2(df)) = sqrt(Gamma(0.5*df, 0.5))
        base_dist = Gamma(0.5 * df, 0.5, validate_args=False)
        half = base_dist.rate.new_full((), 0.5)
        super().__init__(base_dist, PowerTransform(half), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Chi, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def df(self) -> Tensor:
        return self.base_dist.concentration * 2

    @property
    def mean(self) -> Tensor:
        # sqrt(2) * Gamma((df+1)/2) / Gamma(df/2)
        half_df = self.df / 2
        return math.sqrt(2) * torch.exp(
            torch.lgamma(half_df + 0.5) - torch.lgamma(half_df)
        )

    @property
    def mode(self) -> Tensor:
        # sqrt(df - 1) for df >= 1, else 0
        return torch.where(
            self.df >= 1,
            (self.df - 1).clamp(min=0).sqrt(),
            torch.zeros_like(self.df),
        )

    @property
    def variance(self) -> Tensor:
        return self.df - self.mean.pow(2)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # log(2^(1-k/2) / Gamma(k/2) * x^(k-1) * exp(-x^2/2))
        # where k = df
        k = self.df
        half_k = k / 2
        log_prob = (
            (1 - half_k) * math.log(2)
            - torch.lgamma(half_k)
            + (k - 1) * value.log()
            - 0.5 * value.pow(2)
        )
        return torch.where(
            value >= 0,
            log_prob,
            torch.full_like(log_prob, -inf),
        )

    def entropy(self):
        # H = lgamma(k/2) + 0.5*(k - log(2) - (k-1)*digamma(k/2))
        k = self.df
        half_k = k / 2
        return (
            torch.lgamma(half_k)
            + 0.5 * (k - math.log(2) - (k - 1) * torch.digamma(half_k))
        )
