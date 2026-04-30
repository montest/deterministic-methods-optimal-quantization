import numpy as np
import argparse

from univariate.exponential_quantization import ExponentialVoronoiQuantization
from univariate.lognormal_quantization import LogNormalVoronoiQuantization
from univariate.normal_quantization import NormalVoronoiQuantization
from univariate.uniform_quantization import UniformVoronoiQuantization

np.set_printoptions(precision=5)
np.set_printoptions(linewidth=np.inf)


def print_distortion_curve(distortions: list[float]) -> None:
    print("Distortion by iteration")
    print(f"{'step':>6}  {'distortion':>16}")
    print("-" * 26)
    for step, value in enumerate(distortions, start=1):
        print(f"{step:>6}  {value:>16.8f}")


if __name__ == "__main__":
    np.random.seed(0)

    quantizers = {
        "normal": NormalVoronoiQuantization,
        "lognormal": LogNormalVoronoiQuantization,
        "uniform": UniformVoronoiQuantization,
        "exponential": ExponentialVoronoiQuantization,
    }

    random_sampling = {
        "normal": np.random.normal,
        "lognormal": np.random.lognormal,
        "uniform": np.random.uniform,
        "exponential": np.random.exponential,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--size", type=int, help="Size of quantizer", required=True)
    parser.add_argument(
        "-d",
        "--distribution",
        type=str,
        choices=["normal", "lognormal", "uniform", "exponential"],
        help="Distribution of the quantizer to build",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["lloyd", "mfclvq", "nr", "nrlm"],
        help="Optimization method (`mfclvq` = mean field CLVQ, `nr` = Newton–Raphson, `nrlm` = NR with Levenberg–Marquardt)",
        required=True,
    )
    parser.add_argument(
        "-n", "--nbr_iter", type=int, help="Number of iteration to apply of the optimization method", required=True
    )
    parser.add_argument(
        "--print-distortions",
        action="store_true",
        help="Print a table of distortion at each iteration (after optimization)",
    )

    args = parser.parse_args()

    quantization = quantizers.get(args.distribution)()

    centroids = random_sampling.get(args.distribution)(size=args.size)
    optimizers = {
        "lloyd": quantization.deterministic_lloyd_method,
        "mfclvq": quantization.mean_field_clvq_method,
        "nr": quantization.newton_raphson_method,
        "nrlm": quantization.newton_raphson_method_with_levenberg_marquardt,
    }
    centroids, probas, distortions = optimizers.get(args.method)(centroids, args.nbr_iter)

    if args.print_distortions:
        print_distortion_curve(distortions)
        print()

    print(f"centroids  : {centroids}")
    print(f"probas     : {probas}")
    print(f"Distortion : {distortions[-1]}")
    print(f"Gradient   : {quantization.gradient_distortion(centroids)}")
