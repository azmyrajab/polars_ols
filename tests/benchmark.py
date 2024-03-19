import polars_ols as pls  # import package to register the .least_squares namespace
import polars as pl
import numpy as np
import statsmodels.formula.api as smf
import pyperf


def _make_data() -> pl.DataFrame:
    x = np.random.normal(size=(10_000, 5)).astype("float32")
    eps = np.random.normal(size=10_000, scale=0.1).astype("float32")
    return pl.DataFrame(data=x, schema=[f"x{i + 1}" for i in range(5)]).with_columns(
        y=pl.lit(x.sum(1) + eps)
    )


def benchmark_least_squares(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(
            pl.col("y")
            .least_squares.ols(*[pl.col(c) for c in data.columns if c != "y"])
            .alias("predictions")
        )
        .collect()
    )


def benchmark_least_squares_numpy(data: pl.DataFrame):
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return data.lazy().with_columns(predictions=pl.lit(x @ coef).alias("predictions")).collect()


def benchmark_ridge(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(
            pl.col("y")
            .least_squares.ridge(*[pl.col(c) for c in data.columns if c != "y"], alpha=0.0001)
            .alias("predictions")
        )
        .collect()
    )


def benchmark_ridge_numpy(data: pl.DataFrame):
    alpha: float = 0.0001
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    xtx = x.T @ x + np.eye(x.shape[1]) * alpha
    xty = x.T @ y
    coef = np.linalg.solve(xtx, xty)
    return data.lazy().with_columns(predictions=pl.lit(x @ coef).alias("predictions")).collect()


def benchmark_wls_from_formula(data: pl.DataFrame):
    (
        data.lazy()
        .with_columns(pl.lit(1.0).alias("sample_weights"))
        .with_columns(
            pl.col("y")
            .least_squares.from_formula(
                "x1 + x2 + x3 + x4 + x5", sample_weights=pl.col("sample_weights")
            )
            .alias("predictions")
        )
    )


def benchmark_wls_from_formula_statsmodels(data: pl.DataFrame):
    predictions = (
        smf.wls(
            data=data,
            formula="y ~ x1 + x2 + x3 + x4 + x5",
            weights=np.ones(len(data), dtype="float32"),
        )
        .fit()
        .predict()
        .to_numpy()
    )
    return data.lazy().with_columns(predictions=pl.lit(predictions)).collect()


df = _make_data()
runner = pyperf.Runner()
runner.bench_func("benchmark_least_squares", benchmark_least_squares, df)
runner.bench_func("benchmark_least_squares_numpy", benchmark_least_squares_numpy, df)
runner.bench_func("benchmark_ridge", benchmark_ridge, df)
runner.bench_func("benchmark_ridge_numpy", benchmark_ridge_numpy, df)
runner.bench_func("benchmark_wls_from_formula", benchmark_wls_from_formula, df)
runner.bench_func(
    "benchmark_wls_from_formula_statsmodels", benchmark_wls_from_formula_statsmodels, df
)