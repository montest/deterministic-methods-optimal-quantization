from cmath import inf
from dataclasses import dataclass, field
from typing import List, Union
from numpy.linalg import inv

import numpy as np
from scipy.stats import norm

Point: float

@dataclass
class GaussianQuantization:

    lower_bound_support: float = field(init=False, default=-inf)
    upper_bound_support: float = field(init=False, default=inf)

    def variance(self):
        return 1

    # Cumulative Distribution Function
    def cdf(self, x: Union[float, np.ndarray]):
        return norm.cdf(x)

    # Probabilty Density Function
    def pdf(self, x: Union[float, np.ndarray]):
        return norm.pdf(x)

    # First Partial Moment
    def fpm(self, x: Union[float, np.ndarray]):
        return -self.pdf(x)

    def build_mid_points(self, centroids: np.ndarray) -> np.ndarray:
        mid_points = 0.5 * (centroids[1:] + centroids[:-1])
        mid_points = np.insert(mid_points, 0, self.lower_bound_support)
        mid_points = np.append(mid_points, self.upper_bound_support)
        return mid_points

    def mean_of_each_cell(self, mid_points: np.ndarray) -> np.ndarray:
        tempMom = self.fpm(mid_points)
        mean_on_each_cell = tempMom[1:] - tempMom[:-1]
        return mean_on_each_cell

    def proba_of_each_cell(self, mid_points: np.ndarray) -> np.ndarray:
        tempProb = self.cdf(mid_points)
        proba_of_each_cell = tempProb[1:] - tempProb[:-1]
        return proba_of_each_cell

    def distortion(self, centroids: np.ndarray) -> float:
        mid_points = self.build_mid_points(centroids)

        # First term is variance of random variable
        to_return = self.variance()

        # Second term is 2 * \sum_i x_i * E [ X \1_{X \in C_i} ]
        mean_of_each_cell = self.mean_of_each_cell(mid_points)
        to_return -= 2. * (centroids * mean_of_each_cell).sum()

        # Third ans last term is E [ \widehat X^2 ]
        proba_of_each_cell = self.proba_of_each_cell(mid_points)
        to_return += (centroids * centroids * proba_of_each_cell).sum()

        return 0.5 * to_return

    def gradient_distortion(self, centroids: np.ndarray) -> np.ndarray:
        mid_points = self.build_mid_points(centroids)
        to_return = 2. * (centroids * self.proba_of_each_cell(mid_points) - self.mean_of_each_cell(mid_points))
        return to_return

    def hessian_distortion(self, centroids: np.ndarray):
        N = len(centroids)
        result = np.zeros((N, N))
        mid_points = self.build_mid_points(centroids)
        proba_of_each_cell = self.proba_of_each_cell(mid_points)
        tempDens = self.pdf(mid_points)

        a = 0.0
        for i in range(N-1):
            result[i, i] = 2 * proba_of_each_cell[i] - tempDens[i] * a
            a = (centroids[i+1]-centroids[i]) * 0.5
            result[i, i] -= tempDens[i+1] * a
            result[i, i+1] = - a * tempDens[i+1]
            result[i+1, i] = result[i, i+1]
        result[N-1, N-1] = 2 * proba_of_each_cell[N-1] - tempDens[N-1] * a
        return result

    def newton_raphson_step(self, centroids: np.ndarray) -> np.ndarray:
        hessian = self.hessian_distortion(centroids)
        inv_hessian = inv(hessian)
        gradient = self.gradient_distortion(centroids)
        return centroids - np.dot(inv_hessian, gradient)

    def newton_raphson_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):
            centroids = self.newton_raphson_step(centroids)
            centroids.sort()
            print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

    def lr(self, N: int, n: int):
        a = 4.0 * N
        b = np.pi ** 2 / float(N * N)
        return a / float(a + b * (n + 1.))

    def mean_field_clvq_step(self, centroids: np.ndarray, step: int) -> np.ndarray:
        gradient = self.gradient_distortion(centroids)
        return centroids - self.lr(len(centroids), step) * gradient

    def mean_field_clvq_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):
            centroids = self.mean_field_clvq_step(centroids, i)
            centroids.sort()
            print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

    def deterministic_lloyd_step(self, centroids: np.ndarray) -> np.ndarray:
        mid_points = self.build_mid_points(centroids)
        mean_of_each_cell = self.mean_of_each_cell(mid_points)
        proba_of_each_cell = self.proba_of_each_cell(mid_points)
        return mean_of_each_cell / proba_of_each_cell

    def deterministic_lloyd_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):
            centroids = self.deterministic_lloyd_step(centroids)
            print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities