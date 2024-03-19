from pathlib import Path

from typing import Literal, Optional

import polars as pl
from polars.type_aliases import IntoExpr
from polars.plugins import register_plugin_function
from polars_ols.utils import parse_into_expr, build_expressions_from_patsy_formula

__all__ = ["pl_least_squares", "pl_least_squares_from_formula"]


def pl_least_squares(
        target: IntoExpr,
        *features: pl.Expr,
        sample_weights: Optional[pl.Expr] = None,
        add_intercept: bool = False,
        mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
        **kwargs,
) -> pl.Expr:
    assert mode in {"predictions", "residuals", "coefficients"}
    target = parse_into_expr(target).cast(pl.Float32)
    features = [f.cast(pl.Float32) for f in features]
    if add_intercept:
        features += [target.mul(0.0).add(1.0).alias("intercept")]
    sqrt_w = 1.0
    if sample_weights is not None:
        sqrt_w = sample_weights.cast(pl.Float32).sqrt()
        target *= sqrt_w
        features = [expr * sqrt_w for expr in features]

    defaults = {
        "alpha": 0.0,
        "l1_ratio": None,
        "max_iter": None,
        "tol": None,
    }
    kwargs = {**defaults, **kwargs}

    if mode == "coefficients":
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="pl_least_squares_coefficients",
            args=[target, *features],
            kwargs=kwargs,
            is_elementwise=False,
            changes_length=True,
        )
    else:
        predictions = (
                register_plugin_function(
                    plugin_path=Path(__file__).parent,
                    function_name="pl_least_squares",
                    args=[target, *features],
                    kwargs=kwargs,
                    is_elementwise=False,
                )
                / sqrt_w
        )  # undo the sqrt(w) scaling implicit in predictions (:= scaled_features @ coef)
        if mode == "predictions":
            return predictions
        else:
            return (target - predictions).alias("residuals")


def pl_least_squares_from_formula(formula: str, **kwargs) -> pl.Expr:
    expressions, add_intercept = build_expressions_from_patsy_formula(
        formula, include_dependent_variable=True
    )
    return pl_least_squares(expressions[0], *expressions[1:], add_intercept=add_intercept, **kwargs)
