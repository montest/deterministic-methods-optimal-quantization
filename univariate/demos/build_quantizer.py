import numpy as np
import argparse

from univariate.exponential_quantization import ExponentialVoronoiQuantization
from univariate.lognormal_quantization import LogNormalVoronoiQuantization
from univariate.normal_quantization import NormalVoronoiQuantization
from univariate.uniform_quantization import UniformVoronoiQuantization

np.set_printoptions(precision=5)
np.set_printoptions(linewidth=np.inf)


if __name__ == "__main__":
    np.random.seed(0)

    quantizers = {
        'normal': NormalVoronoiQuantization,
        'lognormal': LogNormalVoronoiQuantization,
        'uniform': UniformVoronoiQuantization,
        'exponential': ExponentialVoronoiQuantization,
    }

    random_sampling = {
        'normal': np.random.normal,
        'lognormal': np.random.lognormal,
        'uniform': np.random.uniform,
        'exponential': np.random.exponential
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--size", type=int, help="Size of quantizer", required=True)
    parser.add_argument("-d", "--distribution", type=str, choices=['normal', 'lognormal', 'uniform', 'exponential'], help="Distribution of the quantizer to build", required=True)
    parser.add_argument("-m", "--method", type=str, choices=['lloyd', 'mfclvq', 'nr'], help="Optimization method (`mfclvq` stands for mean field clvq and `nr` stands for Newton Raphson)", required=True)
    parser.add_argument("-n", "--nbr_iter", type=int, help="Number of iteration to apply of the optimization method", required=True)
    parser.add_argument("-p", "--should_print", type=bool, help="Decide if the distortion should be printed at each iteration", required=False, default=False)

    args = parser.parse_args()

    quantization = quantizers.get(args.distribution)()

    centroids = random_sampling.get(args.distribution)(size=args.size)
    optimizers = {
        'lloyd': quantization.deterministic_lloyd_method,
        'mfclvq': quantization.mean_field_clvq_method,
        'nr': quantization.newton_raphson_method,
    }
    centroids, probas = optimizers.get(args.method)(centroids, args.nbr_iter, args.should_print)

    print(f"centroids  : {centroids}")
    print(f"probas     : {probas}")
    print(f"Distortion : {quantization.distortion(centroids)}")
    print(f"Gradient   : {quantization.gradient_distortion(centroids)}")
