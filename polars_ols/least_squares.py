from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

from polars_ols.utils import build_expressions_from_patsy_formula, parse_into_expr

__all__ = [
    "compute_least_squares",
    "compute_recursive_least_squares",
    "compute_rolling_least_squares",
    "least_squares_from_formula",
    "OLSKwargs",
    "RLSKwargs",
    "RollingKwargs",
]


@dataclass
class OLSKwargs:
    """Specifies parameters relevant for regularized linear models: LASSO / Ridge / ElasticNet."""

    alpha: Optional[float] = 0.0
    l1_ratio: Optional[float] = None
    max_iter: Optional[int] = 1_000
    tol: Optional[float] = 0.0001
    positive: Optional[bool] = False  # if True, imposes non-negativity constraint on coefficients

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLSKwargs:
    """Specifies parameters of Recursive Least Squares models."""

    half_life: Optional[float] = None
    initial_state_covariance: Optional[float] = 10.0
    initial_state_mean: Optional[List[float] | float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RollingKwargs:
    """Specifies parameters of Rolling OLS model."""

    window_size: int = 1_000_000  # defaults to expanding OLS
    min_periods: Optional[int] = None
    use_woodbury: Optional[bool] = None
    alpha: Optional[float] = None  # optional ridge alpha

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pre_process_data(
    target: pl.Expr, *features: pl.Expr, sample_weights: Optional[pl.Expr], add_intercept: bool
):
    """Pre-processes the input data by casting it to float32 and scaling it with sample weights if
     provided.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.

    Returns:
        Tuple containing the pre-processed target, features, and sample weights.
    """
    target = parse_into_expr(target).cast(pl.Float32)
    features = [f.cast(pl.Float32) for f in features]
    if add_intercept:
        features += [target.mul(0.0).add(1.0).alias("intercept")]
    sqrt_w = 1.0
    if sample_weights is not None:
        sqrt_w = sample_weights.cast(pl.Float32).sqrt()
        target *= sqrt_w
        features = [expr * sqrt_w for expr in features]
    return target, features, sqrt_w


def compute_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    ols_kwargs: Optional[OLSKwargs] = None,
) -> pl.Expr:
    """Performs least squares regression.

    Depending on parameters provided this method supports a combination of sample weighting (WLS),
     L1/L2 regularization,
     and/or non-negativity constraint on coefficients.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        ols_kwargs: Additional keyword arguments specific for regularized OLS models. These include:
            - "alpha": Regularization strength. Default is 0.0.
                      Expected dtype: float.
            - "l1_ratio": Mixing parameter for ElasticNet regularization (0 for Ridge, 1 for LASSO).
                          Default is None (equivalent to Ridge regression).
                          Expected dtype: float or None.
            - "max_iter": Maximum number of iterations. Defaults to 1000 iterations.
                          Expected dtype: int or None.
            - "tol": Tolerance for convergence criterion. Defaults to 0.0001.
                     Expected dtype: float or None.
            - "positive": Whether to enforce non-negativity constraints on coefficients.
                          Default is None (equivalent to False, i.e. no constraint on coefficients).
                          Expected dtype: bool or None.
    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in {
        "predictions",
        "residuals",
        "coefficients",
    }, "'mode' must be one of {predictions, residuals, coefficients}"
    target, features, sqrt_w = _pre_process_data(
        target, *features, sample_weights=sample_weights, add_intercept=add_intercept
    )

    ols_kwargs: OLSKwargs = ols_kwargs or OLSKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="least_squares_coefficients",
            args=[target, *features],
            kwargs=ols_kwargs.to_dict(),
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="least_squares",
                args=[target, *features],
                kwargs=ols_kwargs.to_dict(),
                is_elementwise=False,
            )
            / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")


def least_squares_from_formula(
    formula: str,
    sample_weights: Optional[pl.Expr] = None,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    **kwargs,
) -> pl.Expr:
    """Performs least squares regression using a formula.

    Depending on choice of additional kwargs dispatches either rolling, recursive,
    on static least squares compute functions.

    Args:
        formula: Patsy-style formula string.
        **kwargs: Additional keyword arguments for the least squares function.

    Returns:
        Resulting expression based on the formula.
    """
    expressions, add_intercept = build_expressions_from_patsy_formula(
        formula, include_dependent_variable=True
    )

    # resolve additional kwargs and relevant ols compute function
    if kwargs.get("half_life"):
        rls_kwargs: RLSKwargs = RLSKwargs(**kwargs)
        func = partial(compute_recursive_least_squares, rls_kwargs=rls_kwargs)
    elif kwargs.get("window_size"):
        rolling_kwargs: RollingKwargs = RollingKwargs(**kwargs)
        func = partial(compute_rolling_least_squares, rolling_kwargs=rolling_kwargs)
    else:
        ols_kwargs: OLSKwargs = OLSKwargs(**kwargs)
        func = partial(compute_least_squares, ols_kwargs=ols_kwargs)

    return func(
        expressions[0],
        *expressions[1:],
        add_intercept=add_intercept,
        sample_weights=sample_weights,
        mode=mode,
    )


def compute_recursive_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    rls_kwargs: Optional[RLSKwargs] = None,
) -> pl.Expr:
    """Performs an efficient recursive least squares regression (RLS).

    Defaults to RLS with forgetting factor of 1.0 and a high initial state variance: equivalent to
     efficient 'streaming' expanding window OLS.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        rls_kwargs: Additional keyword arguments for the recursive least squares model.
            - "half_life": Half-life parameter for exponential forgetting. Default is None
            (no forgetting).
            Expected dtype: float or None.
            - "initial_state_covariance":
            Scalar representing which behaves like an L2 regularization parameter. Larger values
             correspond to larger prior uncertainty around mean vector of state (inversely
             proportional to strength of equivalent L2 penalty).
            Defaults to 10.
            Expected dtype: float or None.
            - "initial_state_mean": Initial mean vector of the state.
                                    Default is zero vector.
                                    Expected dtype: List[float] or None.

    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in {
        "predictions",
        "residuals",
        "coefficients",
    }, "'mode' must be one of {predictions, residuals, coefficients}"
    target, features, sqrt_w = _pre_process_data(
        target, *features, sample_weights=sample_weights, add_intercept=add_intercept
    )
    rls_kwargs: RLSKwargs = rls_kwargs or RLSKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="recursive_least_squares_coefficients",
            args=[target, *features],
            kwargs=rls_kwargs.to_dict(),
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="recursive_least_squares",
                args=[target, *features],
                kwargs=rls_kwargs.to_dict(),
                is_elementwise=False,
            )
            / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")


def compute_rolling_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    rolling_kwargs: Optional[RollingKwargs] = None,
) -> pl.Expr:
    """Performs least squares regression in a rolling window fashion.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        rolling_kwargs: Additional keyword arguments for the rolling least squares model.
            - "window_size": The size of the rolling window.
            - "min_periods": The minimum number of observations required to produce estimates.
            - "use_woodbury": Whether to use Woodbury matrix identity for faster computation.
                              Defaults to True if num_features > 10.
            - "alpha": L2 Regularization strength. Default is 0.0.
                      Expected dtype: float.

    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in {
        "predictions",
        "residuals",
        "coefficients",
    }, "'mode' must be one of {predictions, residuals, coefficients}"
    target, features, sqrt_w = _pre_process_data(
        target, *features, sample_weights=sample_weights, add_intercept=add_intercept
    )
    rolling_kwargs: RollingKwargs = rolling_kwargs or RollingKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="rolling_least_squares_coefficients",
            args=[target, *features],
            kwargs=rolling_kwargs.to_dict(),
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="rolling_least_squares",
                args=[target, *features],
                kwargs=rolling_kwargs.to_dict(),
                is_elementwise=False,
            )
            / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")
