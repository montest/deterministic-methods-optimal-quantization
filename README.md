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