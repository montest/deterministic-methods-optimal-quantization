import numpy as np

from typing import Union
from scipy.stats import uniform
from dataclasses import dataclass, field

from univariate.voronoi_quantization import VoronoiQuantization1D


@dataclass
class UniformVoronoiQuantization(VoronoiQuantization1D):
    lower_bound_support: float = field(init=False, default=0)
    upper_bound_support: float = field(init=False, default=1)
    mean: float = field(init=False, default=1.0/2.0)
    variance: float = field(init=False, default=1.0/12.0)

    def optimal_quantization(self, N):
        centroids = np.linspace((2.0 - 1.0) / (2.0 * N), (2.0 * N - 1.0) / (2.0 * N), N)
        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return uniform.cdf(x)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return uniform.pdf(x)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        return 0.5 * x ** 2

