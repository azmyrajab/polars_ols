from pathlib import Path
from typing import Literal, Optional

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

from polars_ols.utils import build_expressions_from_patsy_formula, parse_into_expr

__all__ = ["least_squares", "least_squares_from_formula", "recursive_least_squares"]


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


def least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    **ols_kwargs,
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
        **ols_kwargs: Additional keyword arguments for the OLS model. These include:
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
    # handle additional model specific kwargs
    defaults = {
        "alpha": 0.0,
        "l1_ratio": None,
        "max_iter": None,
        "tol": None,
        "positive": None,
    }
    supported_parameters = set(defaults)
    assert set(ols_kwargs).issubset(supported_parameters), (
        f"only the following parameters are supported {supported_parameters}, "
        f"the following are not {set(ols_kwargs).difference(supported_parameters)} "
    )
    kwargs = {**defaults, **ols_kwargs}

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="least_squares_coefficients",
            args=[target, *features],
            kwargs=kwargs,
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="least_squares",
                args=[target, *features],
                kwargs=kwargs,
                is_elementwise=False,
            )
            / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")


def least_squares_from_formula(formula: str, **ols_kwargs) -> pl.Expr:
    """Performs ordinary least squares regression using a formula.

    Args:
        formula: Patsy-style formula string.
        **ols_kwargs: Additional keyword arguments for the least squares function.

    Returns:
        Resulting expression based on the formula.
    """
    expressions, add_intercept = build_expressions_from_patsy_formula(
        formula, include_dependent_variable=True
    )
    return least_squares(
        expressions[0], *expressions[1:], add_intercept=add_intercept, **ols_kwargs
    )


def recursive_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
    **rls_kwargs,
):
    """Performs an efficient recursive least squares regression (RLS).

    Defaults to RLS with forgetting factor of 1.0 and a high initial state variance: equivalent to
     efficient 'streaming' expanding window OLS.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        **rls_kwargs: Additional keyword arguments for the recursive least squares model.
        These include:
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

    # handle additional model specific kwargs
    defaults = {
        "half_life": None,
        "initial_state_covariance": None,
        "initial_state_mean": None,
    }
    supported_parameters = set(defaults)
    assert set(rls_kwargs).issubset(
        supported_parameters
    ), f"only the following parameters are supported {supported_parameters}"
    kwargs = {**defaults, **rls_kwargs}

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="recursive_least_squares_coefficients",
            args=[target, *features],
            kwargs=kwargs,
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="recursive_least_squares",
                args=[target, *features],
                kwargs=kwargs,
                is_elementwise=False,
            )
            / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")
