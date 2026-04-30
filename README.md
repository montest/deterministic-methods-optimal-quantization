# Deterministic methods for optimal quantization

The aim of this repository is to illustrate the ideas developed in my blog post [Deterministic Numerical Methods for Optimal Voronoï Quantization: The one-dimensional case](https://montest.github.io/2022/06/21/DeterministicdMethodsForOptimQuantifUnivariates/) where I focus on real valued random variables and explain how to efficiently build optimal quantizers in dimension 1. The main idea is to create an abstract class VoronoiQuantization1D that will contains all generic methods that can be used in order to optimize an optimal quantizer as well as some useful methods in order to compute the distortion, its gradient and hessian. And then implement the methods specific to the distribution of $X$ in the derived classes (e.g NormalVoronoiQuantization, UniformVoronoiQuantization, LogNormalVoronoiQuantization, ExponentialVoronoiQuantization).

## Environment

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Python **3.10 or newer** is required (`requires-python` in [`pyproject.toml`](pyproject.toml)). Dependencies are declared in `pyproject.toml` and pinned in [`uv.lock`](uv.lock).

From the repository root:

```bash
uv sync
```

That creates or updates a `.venv` with the locked packages. You can run commands with `uv run` (no need to activate the virtual environment first), or activate `.venv` and use `python` as usual.

To work with the **optimizer comparison notebooks** under [`notebooks/`](notebooks/), install dev dependencies (Jupyter, Matplotlib) and keep the process **working directory at the repository root** so `import univariate` works:

```bash
uv sync --group dev
uv run jupyter lab
```

Then open e.g. `notebooks/normal.ipynb` from JupyterLab’s file browser.

### Running all notebooks (CI / locally)

To execute all notebooks headlessly:

```bash
scripts/run_notebooks.sh
```

This script runs `jupyter nbconvert --execute` via `uv` and **overwrites the notebooks in-place** with executed outputs (plots and logs).

Note: the comparison notebooks use **Bokeh** for interactive plots. GitHub’s notebook viewer does not execute JavaScript, so you won't see interactive Bokeh plots rendered on GitHub—open the notebooks in JupyterLab or export to HTML if you need shareable interactive output.

### Logging

This project uses [loguru](https://github.com/Delgan/loguru). Default logging level is **INFO**. You can override it with:

```bash
export DETERMINISTIC_QUANTIZATION_LOG_LEVEL=WARNING  # or DEBUG
```

If you don't want logs captured into notebook outputs when running `scripts/run_notebooks.sh`, you can run:

```bash
DETERMINISTIC_QUANTIZATION_LOG_LEVEL=ERROR scripts/run_notebooks.sh
```

If you need a `requirements.txt` for another tool, generate one from the lockfile with:

```bash
uv export --format requirements-txt -o requirements.txt
```

## Command for building optimal quantizers using different optimization methods

`N` is the size of the quantizer, `n` is the number of steps and `m` is the method chosen (`mfclvq` for Mean Field CLVQ, `lloyd` for Lloyd, `nr` for Newton–Raphson, and `nrlm` for Newton–Raphson with Levenberg–Marquardt damping).

### Normal distribution
```
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m mfclvq
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m lloyd
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m nr
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d normal -m nrlm
```

### Log-Normal distribution
```
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m mfclvq
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m lloyd
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m nr
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d lognormal -m nrlm
```

### Exponential distribution
```
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m mfclvq
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m lloyd
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m nr
uv run python -m univariate.demos.build_quantizer -N 10 -n 1000 -d exponential -m nrlm
```

### Uniform distribution
```
uv run python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m mfclvq
uv run python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m lloyd
uv run python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m nr
uv run python -m univariate.demos.build_quantizer -N 50 -n 1000 -d uniform -m nrlm
```
