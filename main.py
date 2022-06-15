import numpy as np

from univariate.exponential_quantization import ExponentialVoronoiQuantization
from univariate.normal_quantization import NormalVoronoiQuantization
from univariate.uniform_quantization import UniformVoronoiQuantization

np.set_printoptions(precision=5)
np.set_printoptions(linewidth=np.inf)


def initialize_quantizer(N):
    np.random.seed(0)
    centroids = np.random.exponential(1, N)   # Initialize the Voronoi Quantizer
    # centroids = np.random.normal(0, 1, N)   # Initialize the Voronoi Quantizer
    # centroids = np.random.uniform(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()
    return centroids


if __name__ == "__main__":
    N = 100
    # quantization = UniformVoronoiQuantization()
    # centroids, probas = quantization.optimal_quantization(N)
    # print(f"\nDistortion: {quantization.distortion(centroids)}")

    # print(f"Optimal quantization: \n{centroids}\nwith weights \n{probas}")

    quantization = ExponentialVoronoiQuantization()
    # quantization = NormalVoronoiQuantization()

    centroids = initialize_quantizer(N)
    print(centroids)

    centroids, probas = quantization.deterministic_lloyd_method(centroids, 1)
    centroids, probas = quantization.newton_raphson_method(centroids, 10)
    print(f"\nDistortion: {quantization.distortion(centroids)}")
    print(f"Gradient: {quantization.gradient_distortion(centroids)}")

    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())

    centroids = initialize_quantizer(N)
    centroids, probas = quantization.deterministic_lloyd_method(centroids, 10000)
    print(f"\nDistortion: {quantization.distortion(centroids)}")
    print(f"Gradient: {quantization.gradient_distortion(centroids)}")

    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())

    centroids = initialize_quantizer(N)
    centroids, probas = quantization.mean_field_clvq_method(centroids, 1000000)
    print(f"\nDistortion: {quantization.distortion(centroids)}")
    print(f"Gradient: {quantization.gradient_distortion(centroids)}")

    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())
