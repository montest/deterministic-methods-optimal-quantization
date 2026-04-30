import scipy
import numpy as np

import sys
import os
from typing import List, Literal, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from loguru import logger


def _configure_default_logger() -> None:
    # Set INFO as the default level for this project.
    # (Loguru defaults to DEBUG; we want INFO unless the caller overrides.)
    #
    # You can override via environment variable:
    # - DETERMINISTIC_QUANTIZATION_LOG_LEVEL=WARNING
    # - DETERMINISTIC_QUANTIZATION_LOG_LEVEL=DEBUG
    level = os.getenv("DETERMINISTIC_QUANTIZATION_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=level)


_configure_default_logger()


@dataclass
class VoronoiQuantization1D(ABC):
    mean: float = field(init=False)
    variance: float = field(init=False)

    lower_bound_support: float = field(init=False)
    upper_bound_support: float = field(init=False)

    def get_vertices(
        self,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute the vertices of the Voronoi quantizer given the centroids.

        :param centroids:
        :return: list of vertices
        """
        vertices = 0.5 * (centroids[1:] + centroids[:-1])
        vertices = np.insert(vertices, 0, self.lower_bound_support)
        vertices = np.append(vertices, self.upper_bound_support)
        return vertices

    def distortion(
        self,
        centroids: np.ndarray,
    ) -> float:
        """Compute the quadratic distortion for a given quantizer.

        :param centroids:
        :return: distortion
        """
        vertices = self.get_vertices(centroids)

        # First term is variance of random variable
        to_return = self.variance + self.mean**2

        # Second term is 2 * \sum_i x_i * E [ X \1_{X \in C_i} ]
        mean_of_each_cell = self.cells_expectation(vertices)
        to_return -= 2.0 * (centroids * mean_of_each_cell).sum()

        # Third and last term is E [ \widehat X^2 ]
        proba_of_each_cell = self.cells_probability(vertices)
        to_return += (centroids**2 * proba_of_each_cell).sum()

        return 0.5 * to_return

    def gradient_distortion(
        self,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute the quadratic distortion's gradient for a given quantizer.

        :param centroids:
        :return: a list of size N containing the gradients
        """
        vertices = self.get_vertices(centroids)
        to_return = centroids * self.cells_probability(vertices) - self.cells_expectation(vertices)
        return to_return

    def hessian_distortion(
        self,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute the quadratic distortion's Hessian for a given quantizer.

        :param centroids:
        :return: an array of size (N, N) containing the hessian
        """
        N = len(centroids)
        result = np.zeros((N, N))
        vertices = self.get_vertices(centroids)
        proba_of_each_cell = self.cells_probability(vertices)
        tempDens = self.pdf(vertices)

        a = 0.0
        for i in range(N - 1):
            result[i, i] = 2.0 * proba_of_each_cell[i] - tempDens[i] * a
            a = (centroids[i + 1] - centroids[i]) * 0.5
            result[i, i] -= tempDens[i + 1] * a
            result[i, i + 1] = -a * tempDens[i + 1]
            result[i + 1, i] = result[i, i + 1]
        result[N - 1, N - 1] = 2 * proba_of_each_cell[N - 1] - tempDens[N - 1] * a
        return result

    ## Optimization methods ##

    def deterministic_lloyd_method(
        self,
        centroids: np.ndarray,
        nbr_iterations: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        if nbr_iterations == 0:
            return centroids, self.cells_probability(self.get_vertices(centroids)), []

        logger.info(
            "Start Lloyd (N={}, iterations={})",
            len(centroids),
            nbr_iterations,
        )
        distortions = []
        for i in range(nbr_iterations):
            vertices = self.get_vertices(centroids)
            mean_of_each_cell = self.cells_expectation(vertices)
            proba_of_each_cell = self.cells_probability(vertices)
            centroids = mean_of_each_cell / proba_of_each_cell
            distortions.append(self.distortion(centroids))
        probabilities = self.cells_probability(self.get_vertices(centroids))
        logger.info("End Lloyd (final_distortion={})", distortions[-1] if distortions else None)
        return centroids, probabilities, distortions

    def mean_field_clvq_method(
        self,
        centroids: np.ndarray,
        nbr_iterations: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        logger.info(
            "Start mean-field CLVQ (N={}, iterations={})",
            len(centroids),
            nbr_iterations,
        )
        distortions = []
        for i in range(nbr_iterations):
            gradient = self.gradient_distortion(centroids)
            lr = self.lr(len(centroids), i, nbr_iterations)
            centroids = centroids - lr * gradient

            centroids.sort()
            distortions.append(self.distortion(centroids))
        probabilities = self.cells_probability(self.get_vertices(centroids))
        logger.info("End mean-field CLVQ (final_distortion={})", distortions[-1] if distortions else None)
        return centroids, probabilities, distortions

    def newton_raphson_method(
        self,
        centroids: np.ndarray,
        nbr_iterations: int,
        num_warmup_iterations: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        logger.info(
            "Start Newton–Raphson (warmup_lloyd={}, iterations={}, N={})",
            num_warmup_iterations,
            nbr_iterations,
            len(centroids),
        )
        centroids, probas, distortions = self.deterministic_lloyd_method(centroids, num_warmup_iterations)
        for i in range(nbr_iterations - num_warmup_iterations):
            hessian = self.hessian_distortion(centroids)
            gradient = self.gradient_distortion(centroids)
            inv_hessian_dot_grad = scipy.linalg.solve(hessian, gradient, assume_a="sym")
            centroids = centroids - inv_hessian_dot_grad
            centroids.sort()  # we sort the centroids because Newton-Raphson does not always preserve the order
            distortions.append(self.distortion(centroids))

        probabilities = self.cells_probability(self.get_vertices(centroids))
        logger.info("End Newton–Raphson (final_distortion={})", distortions[-1] if distortions else None)
        return centroids, probabilities, distortions

    def newton_raphson_method_with_levenberg_marquardt(
        self,
        centroids: np.ndarray,
        nbr_iterations: int,
        lambda_0: float = 1.0,
        num_warmup_iterations: int = 20,
        diagonal_term_type: Literal["identity", "hessian"] = "identity",
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        logger.info(
            "Start NR+LM (warmup_lloyd={}, iterations={}, N={}, lambda_0={}, diagonal_term_type={})",
            num_warmup_iterations,
            nbr_iterations,
            len(centroids),
            lambda_0,
            diagonal_term_type,
        )
        centroids, probas, distortions = self.deterministic_lloyd_method(centroids, num_warmup_iterations)
        lambda_ = lambda_0
        current_distortion = self.distortion(centroids)
        max_inner = 10
        for i in range(num_warmup_iterations, nbr_iterations):
            hessian = self.hessian_distortion(centroids)
            gradient = self.gradient_distortion(centroids)
            improved = False
            inner_tries = 0
            for _ in range(max_inner):
                inner_tries += 1
                if diagonal_term_type == "identity":
                    new_hessian = hessian + lambda_ * np.eye(len(centroids))
                elif diagonal_term_type == "hessian":
                    new_hessian = hessian + lambda_ * hessian
                else:
                    raise ValueError(f"Invalid diagonal term type: {diagonal_term_type}")
                try:
                    inv_hessian_dot_grad = scipy.linalg.solve(new_hessian, gradient, assume_a="sym")
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        "NR+LM step {}/{}: solve failed ({}), increasing lambda from {} to {}",
                        i + 1,
                        nbr_iterations,
                        type(e).__name__,
                        lambda_,
                        lambda_ * 10,
                    )
                    lambda_ = lambda_ * 10
                    continue
                candidate_centroids = centroids - inv_hessian_dot_grad
                candidate_centroids.sort()
                candidate_distortion = self.distortion(candidate_centroids)
                if candidate_distortion < current_distortion:
                    current_distortion = candidate_distortion
                    distortions.append(current_distortion)
                    centroids = candidate_centroids
                    improved = True
                    break
                lambda_ = lambda_ * 10
                logger.info("NR+LM step {}/{}: no improvement, increasing lambda to {}", i + 1, nbr_iterations, lambda_)
            if not improved:
                distortions.append(current_distortion)
                logger.warning(
                    "NR+LM step {}/{}: no_improvement after {} tries (lambda_ ended at {})",
                    i + 1,
                    nbr_iterations,
                    inner_tries,
                    lambda_,
                )
            lambda_ = lambda_ * 0.1
            logger.info("NR+LM step {}/{}: decreasing lambda to {}", i + 1, nbr_iterations, lambda_)
        probabilities = self.cells_probability(self.get_vertices(centroids))
        logger.info("End NR+LM (final_distortion={})", distortions[-1] if distortions else None)
        return centroids, probabilities, distortions

    def lr(
        self,
        N: int,
        n: int,
        max_iter: int,
    ) -> float:
        return 0.1

    def cells_expectation(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """Compute the expectation of $X$ on each cell using the first partial moment function

        :param vertices:
        :return: list of size N containing $\forall i \in \{ 1, \dots, N \}, \mathbb{E} (X \1_{X \in C_i (\Gamma_N) } )$
        """
        first_partial_moment = self.fpm(vertices)
        mean_on_each_cell = first_partial_moment[1:] - first_partial_moment[:-1]
        return mean_on_each_cell

    def cells_probability(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """Compute the probabilities of $X$ on each cell using the cumulative distribution function (that are the
        probabilities of the quantizer $\widehat X^N$)

        :param vertices:
        :return: list of size N containing $\forall i \in \{ 1, \dots, N \}, \mathbb{P} (X \in C_i (\Gamma_N) } )$
        """
        cumulated_probability = self.cdf(vertices)
        proba_of_each_cell = cumulated_probability[1:] - cumulated_probability[:-1]
        return proba_of_each_cell

    @abstractmethod
    def pdf(
        self,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Probabilty Density Function, can take a float or a list/array as input and returns
        $$
            x \rightarrow f_X ( x )
        $$
        It needs be implemented in the derived class.
        """
        pass

    @abstractmethod
    def cdf(
        self,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Cumulative Distribution Function, can take a float or a list/array as input and returns
        $$
            x \rightarrow \mathbb{P} ( X \leq x )
        $$
        It needs be implemented in the derived class.
        """
        pass

    @abstractmethod
    def fpm(
        self,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """First Partial Moment, can take a float or a list/array as input and returns
        $$
            x \rightarrow \mathbb{E} [ X \mathbb{1}_{X \leq x} ]
        $$
        It needs be implemented in the derived class.
        """
        pass
