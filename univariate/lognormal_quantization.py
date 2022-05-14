import numpy as np

from cmath import inf
from typing import Union
from scipy.stats import lognorm, norm
from dataclasses import dataclass, field

from univariate.voronoi_quantization import VoronoiQuantization1D


@dataclass
class LogNormalVoronoiQuantization(VoronoiQuantization1D):
    sigma: float = field(default=1.)

    lower_bound_support: float = field(init=False, default=0)
    upper_bound_support: float = field(init=False, default=inf)
    mean: float = field(init=False)
    variance: float = field(init=False)

    def __post_init__(self):
        self.mean = np.exp(0.5 * self.sigma**2)
        self.variance = (np.exp(self.sigma**2)-1.0) * np.exp(self.sigma**2)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return lognorm.pdf(x, s=self.sigma)

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return lognorm.cdf(x, s=self.sigma)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        return self.mean * norm.cdf(np.log(x) / self.sigma - self.sigma)
