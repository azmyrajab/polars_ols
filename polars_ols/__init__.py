from typing import Literal, Optional

import polars as pl

from polars_ols.least_squares import (
    OLSKwargs,
    RLSKwargs,
    RollingKwargs,
    compute_least_squares,
    compute_recursive_least_squares,
    compute_rolling_least_squares,
    least_squares_from_formula,
)
from polars_ols.utils import build_expressions_from_patsy_formula

__all__ = [
    "compute_least_squares",
    "compute_recursive_least_squares",
    "compute_rolling_least_squares",
    "LeastSquares",
    "least_squares_from_formula",
]


@pl.api.register_expr_namespace("least_squares")
class LeastSquares:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def least_squares(
        self,
        *features: pl.Expr,
        sample_weights: Optional[pl.Expr] = None,
        add_intercept: bool = False,
        mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
        **ols_kwargs,
    ) -> pl.Expr:
        return compute_least_squares(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            ols_kwargs=OLSKwargs(**ols_kwargs),
        )

    def ols(self, *features: pl.Expr, **kwargs) -> pl.Expr:
        return self.least_squares(*features, **kwargs)

    def wls(self, *features: pl.Expr, sample_weights: pl.Expr, **kwargs) -> pl.Expr:
        return self.least_squares(*features, sample_weights=sample_weights, **kwargs)

    def ridge(self, *features: pl.Expr, alpha: float, **kwargs) -> pl.Expr:
        return self.least_squares(*features, alpha=alpha, l1_ratio=0.0, **kwargs)

    def lasso(self, *features: pl.Expr, alpha: float, **kwargs) -> pl.Expr:
        return self.least_squares(*features, alpha=alpha, l1_ratio=1.0, **kwargs)

    def elastic_net(self, *features: pl.Expr, alpha: float, l1_ratio: float = 0.5, **kwargs):
        return self.least_squares(*features, alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    def rls(
        self,
        *features: pl.Expr,
        sample_weights: Optional[pl.Expr] = None,
        add_intercept: bool = False,
        mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
        **rls_kwargs,
    ):
        return compute_recursive_least_squares(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            rls_kwargs=RLSKwargs(**rls_kwargs),
        )

    def rolling_ols(
        self,
        *features: pl.Expr,
        sample_weights: Optional[pl.Expr] = None,
        add_intercept: bool = False,
        mode: Literal["predictions", "residuals", "coefficients"] = "predictions",
        **rolling_kwargs,
    ):
        return compute_rolling_least_squares(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            rolling_kwargs=RollingKwargs(**rolling_kwargs),
        )

    def expanding_ols(self, *features: pl.Expr, **kwargs):
        return self.rls(*features, half_life=None, **kwargs)

    def from_formula(self, formula: str, **kwargs) -> pl.Expr:
        features, add_intercept = build_expressions_from_patsy_formula(
            formula, include_dependent_variable=False
        )
        if kwargs.get("half_life"):
            return self.rls(*features, add_intercept=add_intercept, **kwargs)
        elif kwargs.get("window_size"):
            return self.rolling_ols(*features, add_intercept=add_intercept, **kwargs)
        else:
            return self.least_squares(*features, add_intercept=add_intercept, **kwargs)
