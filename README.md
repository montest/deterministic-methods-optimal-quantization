# Deterministic methods for optimal quantization

The aim of this repository is to illustrate the ideas developed in my blog post [Deterministic Numerical Methods for Optimal Voronoï Quantization: The one-dimensional case](https://montest.github.io/2022/06/21/DeterministicdMethodsForOptimQuantifUnivariates/) where I focus on real valued random variables and explain how to efficiently build optimal quantizers in dimension 1. The main idea is to create an abstract class VoronoiQuantization1D that will contains all generic methods that can be used in order to optimize an optimal quantizer as well as some useful methods in order to compute the distortion, its gradient and hessian. And then implement the methods specific to the distribution of $X$ in the derived classes (e.g NormalVoronoiQuantization, UniformVoronoiQuantization, LogNormalVoronoiQuantization, ExponentialVoronoiQuantization).

My current Python version is `Python 3.9.13` and the required packages are detailed in `requirements.txt`

## Command for building optimal quantizers using different optimization methods

`N` is the size of the quantizer, `n` is the number of steps and `m` is the method chosen (`mfclvq` for Mean Field CLVQ, `lloyd` for Lloyd method and `nr` for Newton Raphson method).

### Normal distribution
```
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m mfclvq
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m lloyd
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m nr
```

### Log-Normal distribution
```
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m mfclvq
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m lloyd
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m nr
```

### Exponential distribution
```
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m mfclvq
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m lloyd
python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m nr
```

### Uniform distribution
```
python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m mfclvq
python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m lloyd
python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m nr
```