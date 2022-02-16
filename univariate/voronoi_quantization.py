import numpy as np

from typing import Union
from numpy.linalg import inv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VoronoiQuantization1D(ABC):

    lower_bound_support: float = field(init=False)
    upper_bound_support: float = field(init=False)
    variance: float = field(init=False)

    # Cumulative Distribution Function
    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    # Probabilty Density Function
    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    # First Partial Moment
    @abstractmethod
    def fpm(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    def distortion(self, centroids: np.ndarray) -> float:
        mid_points = self.build_mid_points(centroids)

        # First term is variance of random variable
        to_return = self.variance

        # Second term is 2 * \sum_i x_i * E [ X \1_{X \in C_i} ]
        mean_of_each_cell = self.mean_of_each_cell(mid_points)
        to_return -= 2. * (centroids * mean_of_each_cell).sum()

        # Third and last term is E [ \widehat X^2 ]
        proba_of_each_cell = self.proba_of_each_cell(mid_points)
        to_return += (centroids * centroids * proba_of_each_cell).sum()

        return 0.5 * to_return

    @abstractmethod
    def lr(self, N: int, n: int):
        pass

    # Optimization methods
    def newton_raphson_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):
            inv_hessian = inv(self.hessian_distortion(centroids))
            gradient = self.gradient_distortion(centroids)
            centroids = centroids - np.dot(inv_hessian, gradient)
            centroids.sort()  # we sort the centroids because Newton-Raphson does not always preserve the order
            # print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

    def mean_field_clvq_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):

            gradient = self.gradient_distortion(centroids)
            lr = self.lr(len(centroids), i)
            centroids = centroids - lr * gradient

            centroids.sort()
            # print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

    def deterministic_lloyd_method(self, centroids: np.ndarray, nbr_iterations: int):
        centroids.sort()

        for i in range(nbr_iterations):
            mid_points = self.build_mid_points(centroids)
            mean_of_each_cell = self.mean_of_each_cell(mid_points)
            proba_of_each_cell = self.proba_of_each_cell(mid_points)
            centroids = mean_of_each_cell / proba_of_each_cell

            # print(f"Distortion at step {i+1}: {self.distortion(centroids)}")

        probabilities = self.proba_of_each_cell(self.build_mid_points(centroids))
        return centroids, probabilities

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

    def gradient_distortion(self, centroids: np.ndarray) -> np.ndarray:
        mid_points = self.build_mid_points(centroids)
        to_return = centroids * self.proba_of_each_cell(mid_points) - self.mean_of_each_cell(mid_points)
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
        return 0.5 * result


