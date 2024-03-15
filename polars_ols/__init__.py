import polars as pl
from polars_ols.utils import build_expressions_from_patsy_formula
from polars_ols.least_squares import pl_least_squares, pl_least_squares_from_formula

__all__ = [
    "pl_least_squares",
    "pl_least_squares_from_formula",
    "LeastSquares",
]


@pl.api.register_expr_namespace("least_squares")
class LeastSquares:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def least_squares(self, *features: pl.Expr, **kwargs) -> pl.Expr:
        return pl_least_squares(self._expr, *features, **kwargs)

    def ols(self, *features: pl.Expr, **kwargs) -> pl.Expr:
        return self.least_squares(*features, **kwargs)

    def wls(self, *features: pl.Expr, sample_weights: pl.Expr, **kwargs) -> pl.Expr:
        return self.least_squares(*features, sample_weights=sample_weights, **kwargs)

    def ridge(self, *features: pl.Expr, alpha: float, **kwargs) -> pl.Expr:
        return self.least_squares(*features, ridge_alpha=alpha, **kwargs)

    def from_formula(self, formula: str, **kwargs) -> pl.Expr:
        features, add_intercept = build_expressions_from_patsy_formula(
            formula, include_dependent_variable=False
        )
        return self.least_squares(*features, add_intercept=add_intercept, **kwargs)
