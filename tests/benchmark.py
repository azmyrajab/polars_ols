import numpy as np
import polars as pl
import pyperf
import scipy
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet, Ridge
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.regression.rolling import RollingOLS

import polars_ols as pls  # import package to register the .least_squares namespace
from polars_ols.least_squares import SolveMethod


def _make_data(n_samples: int = 2_000, n_features: int = 5) -> pl.DataFrame:
    x = np.random.normal(size=(n_samples, n_features))
    eps = np.random.normal(size=n_samples, scale=0.1)
    return pl.DataFrame(data=x, schema=[f"x{i + 1}" for i in range(n_features)]).with_columns(
        y=pl.lit(x.sum(1) + eps)
    )


def benchmark_least_squares(data: pl.DataFrame, solve_method: SolveMethod = "qr"):
    return (
        data.lazy()
        .with_columns(
            pl.col("y")
            .least_squares.ols(
                *[pl.col(c) for c in data.columns if c != "y"], solve_method=solve_method
            )
            .alias("predictions")
        )
        .collect()
    )


def benchmark_least_squares_numpy_svd(data: pl.DataFrame):
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return data.lazy().with_columns(predictions=pl.lit(x @ coef).alias("predictions")).collect()


def benchmark_least_squares_numpy_qr(data: pl.DataFrame):
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    q, r = np.linalg.qr(x)
    # make use of the fact that r is upper triangular --> use back substitution (solve_triangular)
    coef = scipy.linalg.solve_triangular(r, q.T @ y, check_finite=False)
    return data.lazy().with_columns(predictions=pl.lit(x @ coef).alias("predictions")).collect()


def benchmark_ridge(data: pl.DataFrame, solve_method: SolveMethod = "chol"):
    return (
        data.lazy()
        .with_columns(
            pl.col("y")
            .least_squares.ridge(
                *[pl.col(c) for c in data.columns if c != "y"],
                alpha=0.0001,
                solve_method=solve_method,
            )
            .alias("predictions")
        )
        .collect()
    )


def benchmark_ridge_numpy(data: pl.DataFrame):
    alpha: float = 0.0001
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    xtx = x.T @ x + np.eye(x.shape[1]) * alpha
    xty = x.T @ y
    coef = np.linalg.solve(xtx, xty)  # this is similar to Ridge(solve_method="cholesky")
    return data.lazy().with_columns(predictions=pl.lit(x @ coef).alias("predictions")).collect()


def benchmark_ridge_sklearn(data: pl.DataFrame, solve_method: str = "cholesky"):
    alpha: float = 0.0001
    y, x = data.select("y").to_numpy().flatten(), data.select(pl.all().exclude("y")).to_numpy()
    mdl = Ridge(fit_intercept=False, alpha=alpha, solver=solve_method).fit(x, y)
    return (
        data.lazy().with_columns(predictions=pl.lit(x @ mdl.coef_).alias("predictions")).collect()
    )


def benchmark_wls_from_formula(data: pl.DataFrame):
    return (
        data.lazy()
        .with_columns(pl.lit(1.0).alias("sample_weights"))
        .with_columns(
            pl.col("y")
            .least_squares.from_formula(
                " + ".join(c for c in data.columns if "x" in c),
                sample_weights=pl.col("sample_weights"),
            )
            .alias("predictions")
        )
    ).collect()


def benchmark_wls_from_formula_statsmodels(data: pl.DataFrame):
    x_cols = " + ".join(c for c in data.columns if "x" in c)
    predictions = (
        smf.wls(
            data=data,
            formula=f"y ~ {x_cols}",
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
                l1_ratio=0.5,  # same as sklearn default setting
                max_iter=1_000,  # same as sklearn default setting
                tol=1.0e-4,  # same as sklearn default setting
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
    features = [pl.col(c) for c in data.columns if c != "y"]
    return (
        data.lazy()
        .with_columns(
            pl.col("y").least_squares.rolling_ols(
                *features,
                window_size=252,
                min_periods=len(features),
                null_policy="skip",
            )
        )
        .collect()
    )


def benchmark_rolling_least_squares_statsmodels(data: pl.DataFrame):
    x = data.select(pl.all().exclude("y")).to_numpy()
    res = RollingOLS(df["y"].to_numpy(), x, window=252, min_nobs=x.shape[1], expanding=True).fit(
        params_only=True
    )
    return data.lazy().with_columns(predictions=pl.lit((res.params * x).sum(1))).collect()


def benchmark_recursive_least_squares_statsmodels(data: pl.DataFrame):
    x = data.select(pl.all().exclude("y")).to_numpy()
    res = RecursiveLS(
        df["y"].to_numpy(),
        x,
    ).fit()
    return data.lazy().with_columns(predictions=pl.lit((res.params * x).sum(1))).collect()


if __name__ == "__main__":
    # example: python tests/benchmark.py --quiet --fast
    # we run the benchmarks in python (as opposed to rust) so that overhead of pyO3 is included
    df = _make_data(n_features=5, n_samples=2_000)
    runner = pyperf.Runner()
    runner.bench_func("benchmark_least_squares_qr", benchmark_least_squares, df, "qr")
    runner.bench_func("benchmark_least_squares_svd", benchmark_least_squares, df, "svd")
    runner.bench_func("benchmark_ridge_cholesky", benchmark_ridge, df, "chol")
    # runner.bench_func("benchmark_ridge_svd", benchmark_ridge, df, "svd")
    # runner.bench_func("benchmark_wls_from_formula", benchmark_wls_from_formula, df)
    # runner.bench_func("benchmark_elastic_net", benchmark_elastic_net, df)
    # runner.bench_func("benchmark_recursive_least_squares", benchmark_recursive_least_squares, df)
    # runner.bench_func("benchmark_rolling_least_squares", benchmark_rolling_least_squares, df)

    runner.bench_func("benchmark_least_squares_numpy_qr", benchmark_least_squares_numpy_qr, df)
    runner.bench_func("benchmark_least_squares_numpy_svd", benchmark_least_squares_numpy_svd, df)
    runner.bench_func("benchmark_ridge_sklearn_cholesky", benchmark_ridge_sklearn, df, "cholesky")
    # runner.bench_func("benchmark_ridge_sklearn_svd", benchmark_ridge_sklearn, df, "svd")
    # runner.bench_func(
    #     "benchmark_wls_from_formula_statsmodels", benchmark_wls_from_formula_statsmodels, df
    # )
    # runner.bench_func("benchmark_elastic_net_sklearn", benchmark_elastic_net_sklearn, df)
    # runner.bench_func(
    #     "benchmark_recursive_least_squares_statsmodels",
    #     benchmark_recursive_least_squares_statsmodels,
    #     df,
    # )
    # runner.bench_func(
    #     "benchmark_rolling_least_squares_statsmodels",
    #     benchmark_rolling_least_squares_statsmodels,
    #     df,
    # )
