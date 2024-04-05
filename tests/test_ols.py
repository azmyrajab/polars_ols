import time
from contextlib import contextmanager

import numpy as np
import polars as pl
import pytest
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet
from statsmodels.regression.rolling import RollingOLS

from polars_ols import OLSKwargs, compute_least_squares, least_squares_from_formula


@contextmanager
def timer(msg: str | None = None, precision: int = 3) -> float:
    start = time.perf_counter()
    end = start
    yield lambda: end - start
    msg = f"{msg or 'Took'}: {time.perf_counter() - start:.{precision}f} seconds"
    print(msg)
    end = time.perf_counter()


def _make_data(n: int = 10_000, n_groups: int | None = None):
    rng = np.random.default_rng(0)
    array = rng.normal(size=(n, 2))
    data = {
        "y": array.sum(axis=1) + rng.normal(size=n, scale=0.1),
        "x1": array[:, 0],
        "x2": array[:, 1],
    }
    if n_groups is not None:
        data |= {"group": rng.integers(n_groups, size=n)}
    return pl.DataFrame(data).with_columns(pl.col(pl.FLOAT_DTYPES).cast(pl.Float32))


def test_ols():
    df = _make_data()
    # compute OLS solution w/ polars (via QR in rust) [2.331s]
    with timer("OLS rust"):
        for _ in range(1_000):
            expr = compute_least_squares(pl.col("y"), pl.col("x1"), pl.col("x2")).alias(
                "predictions"
            )
            df = df.lazy().with_columns(expr).collect()

    # compute OLS w/ lstsq numpy [4.583s]
    with timer("OLS numpy"):
        for _ in range(1_000):
            x, y = df.select("x1", "x2").to_numpy(), df.select("y").to_numpy().flatten()
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            df = df.with_columns(predictions2=pl.lit(x @ coef).flatten())

    assert np.allclose(df["predictions"], df["predictions2"], atol=1.0e-4, rtol=1.0e-4)


def test_fit_missing_data():
    df = _make_data()
    rng = np.random.default_rng(0)

    def insert_nulls(val):
        return None if rng.random() < 0.1 else val

    df = df.with_columns(
        *(pl.col(c).map_elements(insert_nulls, return_dtype=pl.Float32) for c in df.columns)
    )

    with pytest.raises(pl.exceptions.ComputeError):
        df.select(pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2"), null_policy="ignore"))

    # test zero policy is sane
    assert np.allclose(
        df.select(pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2"), null_policy="zero")),
        df.fill_null(0.0).select(
            pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2"), null_policy="ignore")
        ),
    )

    # test drop policy is sane
    assert np.allclose(
        df.select(pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2"), null_policy="drop")),
        df.drop_nulls().select(
            pl.col("y").least_squares.ols(pl.col("x1"), pl.col("x2"), null_policy="ignore")
        ),
    )


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
            .least_squares.from_formula("x1 + x2 -1", mode="coefficients")
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
            .least_squares.from_formula("x1 + x2 -1", mode="coefficients")
            .alias("coefficients")
        )
        .unnest("coefficients")
    )
    assert np.allclose(coef_group.filter(pl.col("group") == 1).select("x1", "x2"), coef_group_1)


def test_coefficients_shape_broadcast():
    df = _make_data(n=10_000, n_groups=10)
    assert df.select(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .alias("coefficients")
    ).shape == (1, 1)

    assert df.with_columns(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .alias("coefficients")
    ).shape == (10_000, 5)

    df_group = df.select(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .over("group")
        .alias("coefficients"),
        "group",
    )
    assert df_group.shape == (10_000, 2)
    assert df_group.unique().shape == (10, 2)

    assert df.with_columns(
        pl.col("y")
        .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
        .over("group")
        .alias("coefficients")
    ).shape == (10_000, 5)


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
    weights = np.random.uniform(0, 1, size=10_000)
    weights /= weights.mean()
    df = _make_data().with_columns(sample_weights=pl.lit(weights)).cast(pl.Float32)

    expr = least_squares_from_formula(
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


def test_ridge():
    df = _make_data()
    alpha: float = 0.01

    with timer("ridge python"):
        for _ in range(1_000):
            x = df.select("x1", "x2").to_numpy()
            y = df.select("y").to_numpy().flatten()
            coef_expected = np.linalg.solve((x.T @ x) + np.eye(x.shape[1]) * alpha, x.T @ y)
            expected = x @ coef_expected

    with timer("ridge rust"):
        for _ in range(1_000):
            expr = compute_least_squares(
                pl.col("y"),
                pl.col("x1"),
                pl.col("x2"),
                ols_kwargs=OLSKwargs(alpha=alpha),
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
    ).cast(pl.Float32)

    weights = np.hstack([np.ones(8_000) * 1.0 / 10**2, np.ones(2_000) * 1.0 / 0.1**2])
    weights /= weights.mean()

    df = df.with_columns(sample_weight=weights).cast(pl.Float32)

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


def test_elastic_net():
    df = _make_data()
    mdl = ElasticNet(fit_intercept=False, alpha=0.1, l1_ratio=0.5, max_iter=1_000, tol=0.0001)
    mdl.fit(df.select(pl.all().exclude("y")), df.select("y"))

    coef = (
        df.lazy()
        .select(
            pl.col("y")
            .least_squares.from_formula(
                "x1 + x2 -1",
                mode="coefficients",
                l1_ratio=0.5,
                alpha=0.1,
                max_iter=1_000,
                tol=0.0001,
            )
            .alias("coefficients")
        )
        .unnest("coefficients")
        .collect()
        .to_numpy()
        .flatten()
    )
    assert np.allclose(mdl.coef_, coef, rtol=1.0e-4, atol=1.0e-4)


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
            .least_squares.ols(pl.col("x1"), pl.col("x2"), mode="coefficients")
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


def test_rolling_least_squares():
    df = _make_data()
    with timer("rolling ols"):
        coef_rolling = (
            df.lazy()
            .select(
                pl.col("y")
                .least_squares.rolling_ols(
                    pl.col("x1"),
                    pl.col("x2"),
                    mode="coefficients",
                    window_size=252,
                    min_periods=2,
                )
                .alias("coefficients")
            )
            .unnest("coefficients")
            .collect()
            .to_numpy()
        )
    with timer("rolling ols statsmodels"):
        mdl = RollingOLS(
            df["y"].to_numpy(), df[["x1", "x2"]].to_numpy(), window=252, min_nobs=2, expanding=True
        ).fit()
    assert np.allclose(coef_rolling[1:], mdl.params[1:].astype("float32"), rtol=1.0e-3, atol=1.0e-3)


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
    df_test = _make_data(n=20, n_groups=1)

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
                pl.col("x1"), pl.col("x2"), name="predictions"
            )
        )
        .collect()
        .to_numpy()
    )

    # compare to predictions computed as dot product of train coefficients with test data
    expected = (
        df_test.drop("group")
        .to_numpy()
        .dot(df_coefficients.unnest("coefficients").collect().to_numpy().T)
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
