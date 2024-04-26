from typing import Optional, Tuple

import numpy as np
import polars as pl
import pytest
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet, Ridge
from statsmodels.regression.rolling import RollingOLS

from polars_ols import (
    NullPolicy,
    OLSKwargs,
    SolveMethod,
    compute_least_squares,
    compute_least_squares_from_formula,
    compute_multi_target_least_squares,
)
from polars_ols.utils import timer


def _make_data(
    n_samples: int = 5_000,
    n_features: int = 2,
    n_groups: Optional[int] = None,
    scale: float = 0.1,
    sparsity: float = 0.0,
    add_missing: bool = False,
    missing_columns: Optional[Tuple[str, ...]] = None,
) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_samples, n_features))
    eps = rng.normal(size=n_samples, scale=scale)

    df = pl.DataFrame(data=x, schema=[f"x{i + 1}" for i in range(n_features)]).with_columns(
        y=pl.lit(x[:, : int(n_features * (1.0 - sparsity))].sum(1) + eps)
    )

    if n_groups is not None:
        df = df.with_columns(group=pl.lit(rng.integers(n_groups, size=n_samples)))

    if add_missing:

        def insert_nulls(val):
            return None if rng.random() < 0.1 else val

        columns = missing_columns or df.columns
        df = df.with_columns(
            *(pl.col(c).map_elements(insert_nulls, return_dtype=pl.Float64) for c in columns)
        )
    return df


@pytest.mark.parametrize("solve_method", ("qr", "svd", "chol", "lu", None))
def test_ols(solve_method: SolveMethod):
    import os

    os.environ["POLARS_VERBOSE"] = "1"
    df = _make_data(n_samples=1_000, n_features=2)
    # compute OLS w/ polars-ols
    with timer(f"\nOLS {solve_method}", precision=5):
        for _ in range(1_000):
            expr = compute_least_squares(
                pl.col("y"),
                pl.col("x1"),
                pl.col("x2"),
                ols_kwargs=OLSKwargs(solve_method=solve_method),
            ).alias("predictions")
            df = df.lazy().with_columns(expr).collect()
    # compute OLS w/ lstsq numpy
    with timer("OLS numpy", precision=5):
        for _ in range(1_000):
            x, y = df.select("x1", "x2").to_numpy(), df.select("y").to_numpy().flatten()
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            df = df.with_columns(predictions2=pl.lit(x @ coef).flatten())
    assert np.allclose(df["predictions"], df["predictions2"], atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.parametrize(
    "alpha,mode,null_policy",
    [
        (0.0, "residuals", "ignore"),
        (0.0, "residuals", "drop"),
        (0.0001, "residuals", "drop_y_zero_x"),
        (0.01, "residuals", "drop_zero"),
    ],
)
def test_multi_target_regression(alpha, mode, null_policy):
    df = _make_data(
        n_samples=10_000,
        n_features=3,
        add_missing=null_policy not in {"zero", "ignore"},
        missing_columns=("x1",),
        n_groups=3,
    )
    df = df.with_columns(
        pl.struct(
            y1=pl.col("x1") + pl.col("x2") + pl.col("x3"),
            y2=pl.col("x1") - pl.col("x2") + pl.col("x3"),
            y3=-pl.col("x1") + pl.col("x2") - pl.col("x3"),
        ).alias("y")
    )
    ols_kwargs = OLSKwargs(null_policy=null_policy, solve_method="svd", alpha=alpha)

    with timer(f"compute multi-target (alpha={alpha}, null_policy={null_policy}, mode={mode})"):
        multi_target = df.select(
            compute_multi_target_least_squares(
                "y",
                pl.col("^x.*$"),
                mode=mode,
                ols_kwargs=ols_kwargs,
            )
            .over("group")
            .alias(mode)
        )

    with timer("compute multiple linear regressions"):
        expected = df.unnest("y").select(
            compute_least_squares(
                target,
                pl.col("^x.*$"),
                mode=mode,
                ols_kwargs=ols_kwargs,
            )
            .over("group")
            .alias(target)
            for target in ("y1", "y2", "y3")
        )

    assert np.allclose(multi_target.unnest(mode), expected, equal_nan=True)


def test_fit_missing_data_coefficients():
    df = _make_data(add_missing=True)

    # test rust zero policy is sane
    assert np.allclose(
        df.select(
            pl.col("y").least_squares.ols(pl.col("^x.*$"), null_policy="zero", mode="coefficients")
        ).unnest("coefficients"),
        df.fill_null(0.0)
        .select(
            pl.col("y").least_squares.ols(
                pl.col("^x.*$"), null_policy="ignore", mode="coefficients"
            )
        )
        .unnest("coefficients"),
    )

    # test rust drop (any) policy is sane
    assert np.allclose(
        df.select(
            pl.col("y").least_squares.ols(pl.col("^x.*$"), null_policy="drop", mode="coefficients")
        ).unnest("coefficients"),
        df.drop_nulls()
        .select(
            pl.col("y").least_squares.ols(
                pl.col("^x.*$"), null_policy="ignore", mode="coefficients"
            )
        )
        .unnest("coefficients"),
    )

    # test rust drop_y_zero_x policy is sane
    assert np.allclose(
        df.select(
            pl.col("y").least_squares.ols(
                pl.col("^x.*$"), null_policy="drop_y_zero_x", mode="coefficients"
            )
        ).unnest("coefficients"),
        df.drop_nulls(subset=["y"])
        .fill_null(0.0)
        .select(
            pl.col("y").least_squares.ols(
                pl.col("^x.*$"), null_policy="ignore", mode="coefficients"
            )
        )
        .unnest("coefficients"),
    )


@pytest.mark.parametrize("null_policy", ["drop", "drop_zero", "drop_y_zero_x"])
def test_fit_missing_data_predictions_and_residuals(null_policy: NullPolicy):
    df = _make_data(add_missing=True)

    # check predictions logic
    with timer("numpy lstsq w/ manual nan policy", precision=5):
        # compute desired behaviour for "drop" in numpy:
        x, y = (
            df.select("x1", "x2").to_numpy(),
            df.select("y").to_numpy().flatten(),
        )  # notice implicit null -> nan converting to numpy

        # 1) compute mask, drop invalid rows, compute coefficients
        if null_policy == "drop_y_zero_x":
            # drop_y will only drop rows based on the target
            is_valid = ~np.isnan(y)
            x_fit, y_fit = np.nan_to_num(x[is_valid, :]), y[is_valid]
        else:
            # drop or drop_zero will drop any invalid row in fitting
            is_valid = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
            x_fit, y_fit = x[is_valid, :], y[is_valid]

        coef = np.linalg.lstsq(x_fit, y_fit, rcond=None)[0]

        # in order to broadcast (valid) predictions/residuals to the dimensions of original data; we
        # use the original (un-dropped) x for computing predictions.
        # The behaviours which are provided for handling nulls in prediction context:
        # a) null_policy="drop": ensure predictions are always masked to null
        #                        if any feature or target is null.
        # b) null_policy="drop_zero": using coefficients estimated from dropped data, always produce
        #                             valid predictions by zero filling features
        #                             (this allows extrapolation).
        # c) null_policy="drop_y_zero_x": similar to "drop_zero" but where coefficients have been
        #                                 estimated based on the validity of y alone.
        is_nan_x = np.isnan(x)
        x_predict = x.copy()
        x_predict[is_nan_x] = 0.0  # fill x nans with zero
        predictions_expected = x_predict @ coef

        # if user selected "drop" ensure that predictions are masked based on
        # validity of both features & targets.
        if null_policy == "drop":
            predictions_expected[~is_valid] = np.nan
        # otherwise, if "drop_zero" / "drop_y_zero_x", compute predictions based on zero-filled
        # features which allows extrapolation.

    with timer("polars-ols w/ null_policy", precision=5):
        predictions = df.select(
            predictions=pl.col("y").least_squares.ols(
                pl.col("x1"), pl.col("x2"), null_policy=null_policy, mode="predictions"
            )
        )

    assert np.allclose(
        predictions.to_numpy().flatten(),
        predictions_expected.flatten(),
        rtol=1.0e-4,
        atol=1.0e-4,
        equal_nan=True,
    )

    # check residuals logic
    residuals_expected = y - predictions_expected  # no need to copy y
    residuals = df.select(
        residuals=pl.col("y").least_squares.ols(
            pl.col("x1"), pl.col("x2"), null_policy=null_policy, mode="residuals"
        )
    )
    assert np.allclose(
        residuals.to_numpy().flatten(), residuals_expected, rtol=1.0e-4, atol=1.0e-4, equal_nan=True
    )


def test_all_empty_data():
    df = pl.DataFrame(
        {
            "A": [None, 2, None, 4],
            "B": [1, None, 3, None],
        }
    )
    df = df.with_columns(
        pl.col("A")
        .least_squares.ols(
            pl.col("B"),
            mode="residuals",
            null_policy="drop",
            solve_method="svd",
        )
        .alias("residuals")
    )
    assert df["residuals"].is_null().all()


@pytest.mark.parametrize("n_features", (2, 10, 100, 1_000))
def test_fit_wide(n_features: int):
    df = _make_data(n_samples=10, n_features=n_features, scale=1.0e-4)
    features = [pl.col(f) for f in df.columns if f.startswith("x")]
    df = df.with_columns(
        # for p > k: OLS implementation will automatically use LAPACK SVD which should handle
        # over-determined problems
        pl.col("y").least_squares.ols(*features, mode="coefficients").alias("coef_ols"),
        # p >> k, cholesky of X.T @ X may fail: ridge will gracefully
        # fall back to LU with partial pivoting - which should handle over-determined problems
        pl.col("y")
        .least_squares.ridge(*features, mode="coefficients", alpha=1.0e-5)
        .alias("coef_ridge"),
        # lasso/elastic-net use coordinate-descent and so should work always
        pl.col("y")
        .least_squares.lasso(
            *features, mode="coefficients", alpha=1.0e-6, tol=1.0e-8, max_iter=3_000
        )
        .alias("coef_lasso"),
    )
    c_ols = df.select(
        pl.corr(
            pl.col("coef_ols").least_squares.predict(*features).alias("predictions"),
            pl.col("y"),
        )
    )
    c_ridge = df.select(
        pl.corr(
            pl.col("coef_ridge").least_squares.predict(*features).alias("predictions"),
            pl.col("y"),
        )
    )
    c_lasso = df.select(
        pl.corr(
            pl.col("coef_lasso").least_squares.predict(*features).alias("predictions"),
            pl.col("y"),
        )
    )
    assert c_ols.item() == pytest.approx(1.0, rel=1.0e-5, abs=1.0e-5)
    assert c_ridge.item() == pytest.approx(1.0, rel=1.0e-5, abs=1.0e-5)
    assert c_lasso.item() == pytest.approx(1.0, rel=1.0e-5, abs=1.0e-5)


@pytest.mark.parametrize(
    "n_features, solve_method",
    [
        (10, "svd"),  # n < k
        (99, "svd"),  # n = k
        (1_000, "svd"),  # n > k
        (90, "qr"),  # n < k
    ],
)
def test_fit_multi_collinear(n_features: int, solve_method: str):
    n_samples = 100
    # note that:
    # - only SVD solver (equivalent to lstsq) ensures minimum norm solution in case of collinearity
    last_feature_name = f"x{n_features}"
    multicollinear_feature_name = f"x{n_features + 1}"

    df = _make_data(n_samples=n_samples, n_features=n_features, scale=1.0e-4)
    df = df.with_columns((pl.col(last_feature_name) + 1.0e-12).alias(multicollinear_feature_name))

    features = [pl.col(f) for f in df.columns if f.startswith("x")]

    coef = (
        df.select(
            pl.col("y").least_squares.ols(
                *features, mode="coefficients", solve_method=solve_method, rcond=1.0e-16
            )  # machine precision rcond
        )
        .unnest("coefficients")
        .to_numpy()
        .flatten()
    )

    x, y = df.select(pl.all().exclude("y")).to_numpy(), df["y"].to_numpy()
    coef_expected = np.linalg.lstsq(x, y, rcond=1.0e-16)[0]  # similar to -1 (machine precision)

    if solve_method == "svd":
        assert np.allclose(
            coef, coef_expected, rtol=1.0e-2, atol=1.0e-2
        ), f"norm coef={np.linalg.norm(coef)}, norm expected: {np.linalg.norm(coef_expected)}"
        assert np.allclose(x @ coef, x @ coef_expected, rtol=1.0e-4, atol=1.0e-4)
    else:
        assert ~np.isnan(coef).any()
        # other methods will *not* same coefficients
        assert np.linalg.norm(coef) != np.linalg.norm(coef_expected)
        # but should not 'blow up' for QR
        assert np.allclose(x @ coef, x @ coef_expected, rtol=1.0e-4, atol=1.0e-4)


def test_coefficients_ols():
    df = _make_data()
    coef = (
        df.select(
            pl.col("y")
            .least_squares.from_formula("x1 + x2 -1", mode="coefficients")
            .alias("coefficients")
        )
        .unnest("coefficients")
        .to_numpy()[0]
    )
    assert np.allclose(coef, [1.0, 1.0], atol=1.0e-2, rtol=1.0e-2)


def test_coefficients_ols_groups():
    df = _make_data(n_groups=10)
    coef_group = (
        df.select(
            "group",
            pl.col("y")
            .least_squares.ols(pl.col("^x.*$"), mode="coefficients")
            .over("group")
            .alias("coefficients"),
        )
        .unique()
        .unnest("coefficients")
    )
    assert len(coef_group) == 10

    coef_group_1 = (
        df.filter(pl.col("group") == 1)
        .select(
            pl.col("y")
            .least_squares.ols(pl.col("^x.*$"), mode="coefficients")
            .alias("coefficients")
        )
        .unnest("coefficients")
    )
    assert np.allclose(coef_group.filter(pl.col("group") == 1).select("x1", "x2"), coef_group_1)


def test_coefficients_shape_broadcast():
    df = _make_data(n_samples=5_000, n_groups=10)
    assert df.select(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .alias("coefficients")
    ).shape == (1, 1)

    assert df.with_columns(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .alias("coefficients")
    ).shape == (5_000, 5)

    df_group = df.select(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .over("group")
        .alias("coefficients"),
        "group",
    )
    assert df_group.shape == (5_000, 2)
    assert df_group.unique().shape == (10, 2)

    assert df.with_columns(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .over("group")
        .alias("coefficients")
    ).shape == (5_000, 5)


def test_ols_residuals():
    df = _make_data()
    residuals = df.select(
        pl.col("y").least_squares.from_formula("x1 + x2 -1", mode="residuals")
    ).to_numpy()
    x, y = df.select("x1", "x2").to_numpy(), df["y"].to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    assert np.allclose(residuals.flatten(), y - x @ coef, rtol=1.0e-4, atol=1.0e-4)


def test_ols_intercept():
    df = _make_data()
    expr = compute_least_squares(pl.col("y"), pl.col("x1"), pl.col("x2"), add_intercept=True).alias(
        "predictions"
    )
    y_hat = df.select(expr).to_numpy().flatten()
    expected = smf.ols(formula="y ~ x1 + x2", data=df).fit().predict(df).to_numpy()
    assert np.allclose(y_hat, expected, atol=1.0e-4, rtol=1.0e-4)


def test_least_squares_from_formula():
    weights = np.random.uniform(0, 1, size=5_000)
    weights /= weights.mean()
    df = _make_data().with_columns(sample_weights=pl.lit(weights))

    expr = compute_least_squares_from_formula(
        "y ~ x1 + x2",  # patsy includes intercept by default
        sample_weights=pl.col("sample_weights"),
    ).alias("predictions")

    expected = (
        smf.wls("y ~ x1 + x2", data=df, weights=df["sample_weights"].to_numpy())
        .fit()
        .predict(df)
        .to_numpy()
    )
    assert np.allclose(df.select(expr).to_numpy().flatten(), expected, rtol=1.0e-4, atol=1.0e-4)


@pytest.mark.parametrize("solve_method", ("svd", "chol"))
def test_ridge(solve_method: str):
    df = _make_data()
    alpha: float = 0.01

    with timer(f"\nridge python benchmark - {solve_method}"):
        for _ in range(1_000):
            x = df.select("x1", "x2").to_numpy()
            y = df.select("y").to_numpy().flatten()
            if solve_method == "chol":
                coef_expected = np.linalg.solve((x.T @ x) + np.eye(x.shape[1]) * alpha, x.T @ y)
            elif solve_method == "svd":
                coef_expected = (
                    Ridge(fit_intercept=False, solver="svd", alpha=alpha).fit(x, y).coef_
                )
            else:
                raise ValueError()
            expected = x @ coef_expected

    with timer(f"ridge rust {solve_method}"):
        for _ in range(1_000):
            expr = compute_least_squares(
                pl.col("y"),
                pl.col("x1"),
                pl.col("x2"),
                ols_kwargs=OLSKwargs(alpha=alpha, solve_method=solve_method),
            ).alias("predictions")
            df = df.lazy().with_columns(expr).collect()
    assert np.allclose(df["predictions"].to_numpy(), expected, rtol=1.0e-4, atol=1.0e-4)


def test_wls():
    array = np.random.normal(size=(10_000, 2))
    df = pl.DataFrame(
        {
            "y": array.sum(axis=1)
            + np.hstack(
                [np.random.normal(size=8_000, scale=10.0), np.random.normal(size=2_000, scale=0.1)]
            ),
            "x1": array[:, 0],
            "x2": array[:, 1],
        }
    )

    weights = np.hstack([np.ones(8_000) * 1.0 / 10**2, np.ones(2_000) * 1.0 / 0.1**2])
    weights /= weights.mean()

    df = df.with_columns(sample_weight=weights)

    expr_wls = compute_least_squares(
        pl.col("y"),
        pl.col("x1"),
        pl.col("x2"),
        sample_weights=pl.col("sample_weight"),
    ).alias("predictions_wls")
    expr_ols = compute_least_squares(
        pl.col("y"),
        pl.col("x1"),
        pl.col("x2"),
    ).alias("predictions_ols")
    df = df.lazy().with_columns(expr_wls, expr_ols).collect()

    y_hat_wls = smf.wls(data=df, formula="y ~ x1 + x2 -1", weights=weights).fit().predict()
    y_hat_ols = smf.ols(data=df, formula="y ~ x1 + x2 -1").fit().predict()

    assert np.allclose(df["predictions_ols"].to_numpy(), y_hat_ols, rtol=1.0e-4, atol=1.0e-4)
    assert np.allclose(df["predictions_wls"].to_numpy(), y_hat_wls, rtol=1.0e-4, atol=1.0e-4)


def test_least_squares_namespace():
    df = _make_data().with_columns(sample_weight=pl.lit(1.0))
    alpha: float = 0.0
    ols = pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2")).alias("ols")
    ridge = pl.col("y").least_squares.ridge(pl.col("x1"), pl.col("x2"), alpha=alpha).alias("ridge")
    wls = (
        pl.col("y")
        .least_squares.wls(pl.col("x1"), pl.col("x2"), sample_weights=pl.col("sample_weight"))
        .alias("wls")
    )
    formula = pl.col("y").least_squares.from_formula("x1 + x2 - 1").alias("formula")
    df = df.lazy().select(ols, ridge, wls, formula).collect()

    # ensure all of the above are equivalent
    assert np.allclose(df.corr(), 1.0)


@pytest.mark.parametrize(
    "n_features,sparsity,alpha,solve_method",
    [
        (2, 0.0, 0.1, "cd"),
        (100, 0.5, 0.3, "cd"),
        (300, 0.5, 1.0, "cd"),
        (500, 0.9, 0.1, "cd"),
        (1_000, 0.9, 0.3, "cd_active_set"),
    ],
)
def test_elastic_net(n_features: int, sparsity: float, alpha: float, solve_method: SolveMethod):
    df = _make_data(n_features=n_features, sparsity=sparsity)
    features = df.select(pl.all().exclude("y")).columns

    with timer("\nelastic net sklearn"):
        mdl = ElasticNet(fit_intercept=False, alpha=alpha, l1_ratio=0.5, max_iter=1_000, tol=0.0001)
        x, y = df.select(*features), df.select("y")
        mdl.fit(x, y)
        predictions_1 = mdl.predict(x).flatten()

    with timer("elastic net polars-ols"):
        predictions_2 = (
            df.lazy()
            .select(
                pl.col("y").least_squares.elastic_net(
                    *features,
                    mode="predictions",
                    l1_ratio=0.5,
                    alpha=alpha,
                    max_iter=1_000,
                    tol=0.0001,
                    solve_method=solve_method,
                )
            )
            .collect()
            .to_numpy()
            .flatten()
        )
    assert np.allclose(predictions_1, predictions_2, rtol=1.0e-4, atol=1.0e-4)


def test_elastic_net_non_negative():
    df = _make_data()
    mdl = ElasticNet(
        fit_intercept=False, alpha=0.1, l1_ratio=0.5, max_iter=1_000, tol=0.0001, positive=True
    )
    mdl.fit(df.select(pl.col("x1"), -pl.col("x2")), df.select("y"))

    coef = (
        df.lazy()
        .select(
            pl.col("y")
            .least_squares.elastic_net(
                pl.col("x1"),
                -pl.col("x2"),
                mode="coefficients",
                l1_ratio=0.5,
                alpha=0.1,
                max_iter=1_000,
                tol=0.0001,
                positive=True,
            )
            .alias("coefficients")
        )
        .unnest("coefficients")
        .collect()
        .to_numpy()
        .flatten()
    )
    assert np.allclose(mdl.coef_, coef, rtol=1.0e-4, atol=1.0e-4)


def test_recursive_least_squares():
    df = _make_data()
    rng = np.random.default_rng(0)

    def insert_nulls(val):
        return None if rng.random() < 0.1 else val

    df = df.with_columns(
        *(pl.col(c).map_elements(insert_nulls, return_dtype=pl.Float64) for c in df.columns)
    )

    # expanding OLS
    coef_rls = (
        df.lazy()
        .select(
            pl.col("y")
            .least_squares.rls(
                pl.col("x1"),
                pl.col("x2"),
                mode="coefficients",
                # equivalent to expanding window (no forgetting)
                half_life=None,
                # arbitrarily weak L2 (diffuse) prior
                initial_state_covariance=1_000_000.0,
                null_policy="drop",
            )
            .alias("coefficients")
        )
        .unnest("coefficients")
        .collect()
        .to_numpy()
    )

    # full sample OLS
    coef_ols = (
        df.lazy()
        .select(
            pl.col("y")
            .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients", null_policy="drop")
            .alias("coefficients")
        )
        .unnest("coefficients")
        .collect()
        .to_numpy()
        .flatten()
    )

    # the two models should be equivalent in full sample
    assert np.allclose(coef_rls[-1], coef_ols, rtol=1.0e-4, atol=1.0e-4)


def test_recursive_least_squares_prior():
    df = _make_data()

    # equivalent to expanding OLS w/ L2 prior towards [0.5, 0.5]
    coef_rls_prior = (
        df.lazy()
        .select(
            pl.col("y")
            .least_squares.rls(
                pl.col("x1"),
                pl.col("x2"),
                mode="coefficients",
                # equivalent to expanding window (no forgetting)
                half_life=None,
                initial_state_covariance=1.0e-6,  # arbitrarily strong L2 prior
                initial_state_mean=[0.25, 0.25],  # custom prior
            )
            .alias("coefficients")
        )
        .unnest("coefficients")
        .collect()
        .to_numpy()
    )

    # given few samples and strong prior strength, the coefficients are nearly
    # identical to the prior
    assert np.allclose(coef_rls_prior[0], [0.25, 0.25], rtol=1.0e-3, atol=1.0e-3)
    assert np.allclose(coef_rls_prior[10], [0.25, 0.25], rtol=1.0e-3, atol=1.0e-3)

    # as number of samples seen grows, the coefficients start to diverge from prior
    # & eventually converge to ground truth.
    assert not np.allclose(coef_rls_prior[-1], [0.5, 0.5], rtol=1.0e-4, atol=1.0e-4)


@pytest.mark.parametrize(
    "window_size,min_periods,use_woodbury",
    [
        (2, 2, False),
        (10, 2, False),
        (10, 2, True),
        (63, 5, False),
        (252, 5, False),
        (252, 5, True),
    ],
)
def test_rolling_least_squares(window_size: int, min_periods: int, use_woodbury: bool):
    df = _make_data(n_samples=1_000, add_missing=True, missing_columns=("y",))

    with timer("\nrolling ols"):
        coef_rolling = (
            df.lazy()
            .select(
                pl.col("y")
                .least_squares.rolling_ols(
                    pl.col("x1"),
                    pl.col("x2"),
                    mode="coefficients",
                    window_size=window_size,
                    min_periods=min_periods,
                    use_woodbury=use_woodbury,
                    null_policy="drop_window",  # equivalent to mode='missing' in statsmodels
                )
                .alias("coefficients")
            )
            .unnest("coefficients")
            .collect()
        )
    with timer("rolling ols statsmodels"):
        mdl = RollingOLS(
            df["y"].to_numpy(),
            df[["x1", "x2"]].to_numpy(),
            window=window_size,
            min_nobs=min_periods,
            expanding=True,
            missing="drop",
        ).fit()
        coef_statsmodels = (
            pl.DataFrame(mdl.params, schema=["x1", "x2"])
            .fill_nan(None)
            .with_columns(pl.all().forward_fill())
        )

    assert np.allclose(
        coef_rolling.to_numpy(),
        coef_statsmodels.to_numpy(),
        rtol=1.0e-3,
        atol=1.0e-3,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "min_periods,expected",
    [
        (999, 2),
        (1_000, 1),
        (1_001, 0),
    ],
)
def test_rolling_ols_insufficient_data(min_periods: int, expected: int):
    df = _make_data(n_samples=1_000)

    with timer("\nrolling ols"):
        coef_rolling = (
            df.lazy()
            .select(
                pl.col("y")
                .least_squares.rolling_ols(
                    pl.col("x1"),
                    pl.col("x2"),
                    mode="coefficients",
                    window_size=2_000,
                    min_periods=min_periods,
                    use_woodbury=False,
                    null_policy="drop_window",  # equivalent to mode='missing' in statsmodels
                )
                .alias("coefficients")
            )
            .unnest("coefficients")
            .collect()
        )

    assert coef_rolling.count().max_horizontal().item() == expected


@pytest.mark.parametrize("window_size", (21, 252))
def test_rolling_window_drop(window_size: int):
    df = _make_data(n_samples=1_000, add_missing=True, missing_columns=("y",)).with_row_index()
    df_drop = df.drop_nulls()

    df_merged = (
        df_drop.lazy()
        .select(
            "index",
            pl.col("y")
            .least_squares.rolling_ols(
                "x1", "x2", window_size=window_size, mode="predictions", null_policy="drop"
            )
            .alias("predictions_1"),
        )
        .join_asof(
            df.lazy().select(
                "index",
                pl.col("y")
                .least_squares.rolling_ols(
                    "x1",
                    "x2",
                    window_size=window_size,
                    mode="predictions",
                    null_policy="drop",
                )
                .alias("predictions_2"),
            ),
            on="index",
        )
    ).collect()

    assert np.allclose(df_merged["predictions_1"], df_merged["predictions_2"], equal_nan=True)


def test_moving_window_regressions_over():
    df = _make_data(n_groups=10)

    df = (
        (
            df.lazy().select(
                "group",
                pl.col("y")
                .least_squares.rolling_ols(
                    pl.col("x1"),
                    pl.col("x2"),
                    mode="coefficients",
                    window_size=1_000_000,
                    min_periods=2,
                    # larger than data window size equivalent to expanding window
                )
                .over("group")
                .alias("coef_rolling_ols_group"),
                pl.col("y")
                .least_squares.rls(
                    pl.col("x1"),
                    pl.col("x2"),
                    half_life=None,
                    initial_state_covariance=1.0e6,
                    mode="coefficients",
                    # no forgetting factor + diffuse prior equivalent to expanding window
                )
                .over("group")
                .alias("coef_rls_group"),
                pl.col("y")
                .least_squares.ols(
                    pl.col("x1"),
                    pl.col("x2"),
                    mode="coefficients",
                )
                .over("group")
                .alias("coef_ols_group"),  # full sample OLS per group
            )
        )
        .collect()
        .rechunk()
    )

    # As of the last sample per group: RLS & rolling regression should behave identically
    # to full sample OLS per group (the way they were set up above)
    df_last = df.group_by("group").last()

    assert np.allclose(
        df_last.unnest("coef_ols_group").select("x1", "x2"),
        df_last.unnest("coef_rolling_ols_group").select("x1", "x2"),
    )
    assert np.allclose(
        df_last.unnest("coef_ols_group").select("x1", "x2"),
        df_last.unnest("coef_rls_group").select("x1", "x2"),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_predict():
    df = _make_data(n_groups=1)
    df_test = _make_data(n_samples=20, n_groups=1).drop("y")

    # estimate coefficients
    df_coefficients = (
        df.lazy()
        .select(
            "group",
            pl.col("y")
            .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
            .over("group")
            .alias("coefficients"),
        )
        .unique()
    )

    # compute predictions as a rust expression
    predictions = (
        df_test.lazy()
        .join(df_coefficients, on="group")
        .select(
            pl.col("coefficients").least_squares.predict(
                pl.col("x1"),
                pl.col("x2"),
                name="predictions",
                null_policy="zero",
            )
        )
        .collect()
        .to_numpy()
        .flatten()
    )

    # compare to predictions computed as dot product of train coefficients with test data
    expected = (
        df_test.drop("group")
        .to_numpy()
        .dot(df_coefficients.drop("group").unnest("coefficients").collect().to_numpy().T)
        .flatten()
    )
    assert np.allclose(predictions, expected)


def test_predict_formula():
    df = _make_data()
    df = (
        df.lazy()
        .with_columns(
            coefficients=pl.col("y").least_squares.from_formula("x1 + x2", mode="coefficients"),
            predictions_1=pl.col("y").least_squares.from_formula("x1 + x2", mode="predictions"),
        )
        .with_columns(
            predictions_2=pl.col("coefficients").least_squares.predict_from_formula("x1 + x2")
        )
    ).collect()
    assert np.allclose(df["predictions_1"], df["predictions_2"])


def test_predict_complex():
    df = _make_data(n_groups=10)
    df = (
        df.lazy()
        .with_columns(
            predictions_1=pl.col("y")
            .least_squares.rls(pl.col("x1"), pl.col("x2"), mode="predictions")
            .over("group"),
            coefficients=pl.col("y")
            .least_squares.rls(pl.col("x1"), pl.col("x2"), mode="coefficients")
            .over("group"),
        )
        .with_columns(
            predictions_2=pl.col("coefficients").least_squares.predict(pl.col("x1"), pl.col("x2"))
        )
        .collect()
    )
    assert np.allclose(df["predictions_1"], df["predictions_2"])
