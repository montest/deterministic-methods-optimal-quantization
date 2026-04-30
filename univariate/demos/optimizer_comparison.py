"""Helpers for comparing optimizer distortion curves."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Dict, List, Literal

import numpy as np

from univariate.voronoi_quantization import VoronoiQuantization1D

MethodKey = Literal["lloyd", "mfclvq", "nr", "nrlm"]
DiagonalTermType = Literal["identity", "hessian"]


def compare_methods(
    quantizer: VoronoiQuantization1D,
    initial_centroids: np.ndarray,
    n_iter: int,
    methods: Iterable[MethodKey],
    *,
    nr_num_warmup_iterations: int = 20,
    nrlm_num_warmup_iterations: int = 20,
) -> Dict[str, List[float]]:
    """Return distortion curves for the selected optimizers.

    Each method starts from a fresh copy of ``initial_centroids``.

    Notes:
    - Use ``n_iter >= nrlm_num_warmup_iterations`` if you include ``\"nrlm\"`` (NR+LM runs
      ``nrlm_num_warmup_iterations`` Lloyd warmup iterations, then ``n_iter - warmup`` main steps,
      producing a series of length ``n_iter``).
    - Use ``n_iter >= nr_num_warmup_iterations`` if you include ``\"nr\"`` (Newton–Raphson uses
      ``nr_num_warmup_iterations`` Lloyd warmup iterations).
    """
    initial = np.asarray(initial_centroids, dtype=float)

    method_specs: dict[MethodKey, tuple[str, Callable[..., tuple]]] = {
        "lloyd": ("Lloyd", quantizer.deterministic_lloyd_method),
        "mfclvq": ("Mean-field CLVQ", quantizer.mean_field_clvq_method),
        "nr": ("Newton–Raphson", quantizer.newton_raphson_method),
        "nrlm": ("Newton–Raphson (LM)", quantizer.newton_raphson_method_with_levenberg_marquardt),
    }

    out: Dict[str, List[float]] = {}
    for key in methods:
        label, fn = method_specs[key]
        centroids = initial.copy()
        if key == "nr":
            _, _, distortions = fn(centroids, n_iter, num_warmup_iterations=nr_num_warmup_iterations)
        elif key == "nrlm":
            _, _, distortions = fn(centroids, n_iter, num_warmup_iterations=nrlm_num_warmup_iterations)
        else:
            _, _, distortions = fn(centroids, n_iter)
        out[label] = distortions
    return out


def compare_all_methods(
    quantizer: VoronoiQuantization1D,
    initial_centroids: np.ndarray,
    n_iter: int,
) -> Dict[str, List[float]]:
    """Return distortion curves for Lloyd, mean-field CLVQ, Newton–Raphson, and NR+LM."""
    return compare_methods(quantizer, initial_centroids, n_iter, methods=("lloyd", "mfclvq", "nr", "nrlm"))


def compare_nrlm_sweep(
    quantizer: VoronoiQuantization1D,
    initial_centroids: np.ndarray,
    n_iter: int,
    lambda_0_values: Iterable[float],
    diagonal_term_types: Iterable[DiagonalTermType] = ("identity", "hessian"),
    *,
    nrlm_num_warmup_iterations: int = 20,
) -> Dict[tuple[DiagonalTermType, float], List[float]]:
    """Return distortion curves for a sweep of NR+LM hyperparameters.

    Keys are (diagonal_term_type, lambda_0). Each run starts from a copy of
    ``initial_centroids``.
    """
    initial = np.asarray(initial_centroids, dtype=float)
    out: Dict[tuple[DiagonalTermType, float], List[float]] = {}
    for diag in diagonal_term_types:
        for lam0 in lambda_0_values:
            centroids = initial.copy()
            _, _, distortions = quantizer.newton_raphson_method_with_levenberg_marquardt(
                centroids,
                n_iter,
                lambda_0=float(lam0),
                num_warmup_iterations=nrlm_num_warmup_iterations,
                diagonal_term_type=diag,
            )
            out[(diag, float(lam0))] = distortions
    return out
