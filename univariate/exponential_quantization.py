import numpy as np

from cmath import inf
from typing import Union
from dataclasses import dataclass, field

from univariate.voronoi_quantization import VoronoiQuantization1D


@dataclass
class ExponentialVoronoiQuantization(VoronoiQuantization1D):
    lambda_: float = field(default=1)

    lower_bound_support: float = field(init=False, default=-inf)
    upper_bound_support: float = field(init=False, default=inf)
    mean: float = field(init=False)
    variance: float = field(init=False)

    def __post_init__(self):
        self.mean = 1./self.lambda_
        self.variance = 1./(self.lambda_ ** 2)

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return self.lambda_ * np.exp(-self.lambda_*x)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return 1. - np.exp(-self.lambda_*x)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        to_return = - np.exp(-self.lambda_*x)*(x+self.mean) + self.mean
        if type(x) == float and x == inf:
            to_return = self.mean
        # If x is an array then it is supposed to be sorted hence if there is an inf value, it is the last value and it
        # should appear only once in the case of optimal quantization.
        if type(x) == np.ndarray and x[-1] == inf:
            to_return[-1] = self.mean
        return to_return

