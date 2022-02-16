import numpy as np

from univariate.normal_quantization import NormalVoronoiQuantization

if __name__ == "__main__":

    N = 10
    normal_quantization = NormalVoronoiQuantization()
    np.random.seed(0)
    centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()

    centroids, probas = normal_quantization.deterministic_lloyd_method(centroids, 5)
    centroids, probas = normal_quantization.newton_raphson_method(centroids, 100)
    print(f"\nDistortion: {normal_quantization.distortion(centroids)}")
    print(f"Gradient: {normal_quantization.gradient_distortion(centroids)}")


    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())

    np.random.seed(0)
    centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()
    centroids, probas = normal_quantization.deterministic_lloyd_method(centroids, 1000)
    print(f"\nDistortion: {normal_quantization.distortion(centroids)}")
    print(f"Gradient: {normal_quantization.gradient_distortion(centroids)}")

    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())

    np.random.seed(0)
    centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()
    centroids, probas = normal_quantization.mean_field_clvq_method(centroids, 50000)
    print(f"\nDistortion: {normal_quantization.distortion(centroids)}")
    print(f"Gradient: {normal_quantization.gradient_distortion(centroids)}")

    print(centroids)
    print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())
