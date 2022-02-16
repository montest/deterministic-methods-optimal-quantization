import numpy as np

from typing import Union
from scipy.stats import uniform
from dataclasses import dataclass, field

from univariate.voronoi_quantization import VoronoiQuantization1D


@dataclass
class UniformVoronoiQuantization(VoronoiQuantization1D):

    lower_bound_support: float = field(init=False, default=0)
    upper_bound_support: float = field(init=False, default=1)
    variance: float = field(init=False, default=1/12)

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return uniform.cdf(x)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return uniform.pdf(x)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        return 0.5 * x ** 2

    def lr(self, N: int, n: int):
        return 1 / float(n + 1.)
