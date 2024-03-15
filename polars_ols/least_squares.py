from typing import Literal

import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

from polars_ols.utils import parse_into_expr, build_expressions_from_patsy_formula

lib = _get_shared_lib_location(__file__)

__all__ = ["pl_least_squares", "pl_least_squares_from_formula"]


def pl_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    ridge_alpha: float = 0.0,
    sample_weights: pl.Expr | None = None,
    ridge_solve_method: Literal["svd", "solve"] = "solve",
    add_intercept: bool = False,
) -> pl.Expr:
    target = parse_into_expr(target).cast(pl.Float32)
    features = [f.cast(pl.Float32) for f in features]
    if add_intercept:
        features += [target.mul(0.0).add(1.0).alias("intercept")]
    sqrt_w = 1.0
    if sample_weights is not None:
        sqrt_w = sample_weights.cast(pl.Float32).sqrt()
        target *= sqrt_w
        features = [expr * sqrt_w for expr in features]
    return (
        target.register_plugin(
            lib=lib,
            symbol="pl_least_squares",
            args=features,
            kwargs={
                "ridge_alpha": ridge_alpha,
                "ridge_solve_method": ridge_solve_method,
            },
            is_elementwise=False,
        )
        / sqrt_w
    )  # undo the sqrt(w) scaling implicit in predictions (:= scaled_features @ coef)


def pl_least_squares_from_formula(formula: str, **kwargs) -> pl.Expr:
    expressions, add_intercept = build_expressions_from_patsy_formula(
        formula, include_dependent_variable=True
    )
    return pl_least_squares(expressions[0], *expressions[1:], add_intercept=add_intercept, **kwargs)
