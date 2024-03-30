import numpy as np
import polars as pl
import pyperf
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet

import polars_ols as pls  # import package to register the .least_squares namespace


def _make_data(n_features: int = 5) -> pl.DataFrame:
    x = np.random.normal(size=(2_000, n_features)).astype("float32")
    eps = np.random.normal(size=2_000, scale=0.1).astype("float32")
    return pl.DataFrame(data=x, schema=[f"x{i + 1}" for i in range(n_features)]).with_columns(
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
    return (
        data.lazy()
        .with_columns(pl.lit(1.0).alias("sample_weights"))
        .with_columns(
            pl.col("y")
            .least_squares.from_formula(
                "x1 + x2 + x3 + x4 + x5", sample_weights=pl.col("sample_weights")
            )
            .alias("predictions")
        )
    ).collect()


def benchmark_wls_from_formula_statsmodels(data: pl.DataFrame):
    predictions = (
        smf.wls(
            data=data,
            formula="y ~ x1 + x2 + x3 + x4 + x5",
            weights=np.ones(len(data), dtype="float32"),
        )
        .fit()
        .predict()
    )
    return data.lazy().with_columns(predictions=pl.lit(predictions)).collect()


def benchmark_elastic_net(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(
            pl.col("y").least_squares.elastic_net(
                *[pl.col(c) for c in data.columns if c != "y"],
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=1_000,
            )
        )
        .collect()
    )


def benchmark_elastic_net_sklearn(data: pl.DataFrame):
    mdl = ElasticNet(fit_intercept=False, alpha=0.1, l1_ratio=0.5, max_iter=1_000)
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    mdl.fit(x, y)
    return data.lazy().with_columns(predictions=pl.lit(mdl.predict(x))).collect()


def benchmark_recursive_least_squares(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(
            pl.col("y").least_squares.rls(
                *[pl.col(c) for c in data.columns if c != "y"],
                half_life=252,
            )
        )
        .collect()
    )


def benchmark_rolling_least_squares(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(
            pl.col("y").least_squares.rolling_ols(
                *[pl.col(c) for c in data.columns if c != "y"],
                window_size=252,
                min_periods=2,
                alpha=0.0001,
            )
        )
        .collect()
    )


if __name__ == "__main__":
    # example: python tests/benchmark.py --quiet --fast
    # we run the benchmarks in python (as opposed to rust) so that overhead of pyO3 is included
    df = _make_data()
    runner = pyperf.Runner()
    runner.bench_func("benchmark_least_squares", benchmark_least_squares, df)
    runner.bench_func("benchmark_ridge", benchmark_ridge, df)
    runner.bench_func("benchmark_wls_from_formula", benchmark_wls_from_formula, df)
    runner.bench_func("benchmark_elastic_net", benchmark_elastic_net, df)
    runner.bench_func("benchmark_recursive_least_squares", benchmark_recursive_least_squares, df)
    runner.bench_func("benchmark_rolling_least_squares", benchmark_rolling_least_squares, df)

    # runner.bench_func("benchmark_least_squares_numpy", benchmark_least_squares_numpy, df)
    # runner.bench_func("benchmark_ridge_numpy", benchmark_ridge_numpy, df)
    # runner.bench_func(
    #     "benchmark_wls_from_formula_statsmodels", benchmark_wls_from_formula_statsmodels, df
    # )
    # runner.bench_func("benchmark_elastic_net_sklearn", benchmark_elastic_net_sklearn, df)
