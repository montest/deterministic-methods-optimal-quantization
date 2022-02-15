import numpy as np

from gaussian_quantization import GaussianQuantization

if __name__ == "__main__":

    N = 10
    gaussian_quantization = GaussianQuantization()
    # np.random.seed(0)
    # centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    # centroids.sort()
    # centroids, probas = gaussian_quantization.deterministic_lloyd_method(centroids, 5)
    #
    # centroids, probas = gaussian_quantization.newton_raphson_method(centroids, 100)
    #
    # print(centroids)
    # print(probas)
    # print(probas.sum())
    # print((centroids*probas).sum())
    # print((centroids*centroids*probas).sum())

    np.random.seed(0)
    centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()
    centroids, probas = gaussian_quantization.deterministic_lloyd_method(centroids, 1000)

    print(centroids)
    print(probas)
    print(probas.sum())
    print((centroids*probas).sum())
    print((centroids*centroids*probas).sum())

    np.random.seed(0)
    centroids = np.random.normal(0, 1, N)  # Initialize the Voronoi Quantizer
    centroids.sort()
    centroids, probas = gaussian_quantization.mean_field_clvq_method(centroids, 10000)

    print(centroids)
    print(probas)
    print(probas.sum())
    print((centroids*probas).sum())
    print((centroids*centroids*probas).sum())
