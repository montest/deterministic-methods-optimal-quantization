import numpy as np

from cmath import inf
from typing import Union
from scipy.stats import norm
from dataclasses import dataclass, field

from univariate.voronoi_quantization import VoronoiQuantization1D


@dataclass
class NormalVoronoiQuantization(VoronoiQuantization1D):
    lower_bound_support: float = field(init=False, default=-inf)
    upper_bound_support: float = field(init=False, default=inf)
    mean: float = field(init=False, default=0)
    variance: float = field(init=False, default=1)

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return norm.cdf(x)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return norm.pdf(x)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        return -self.pdf(x)

    def lr(self, N: int, n: int, max_iter: int):
        a = 2.0 * N
        b = np.pi / float(N * N)
        return a / float(a + b * (n + 1.))
