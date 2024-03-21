import polars as pl
from polars_ols.utils import build_expressions_from_patsy_formula
from polars_ols.least_squares import least_squares, least_squares_from_formula, recursive_least_squares

__all__ = [
    "least_squares",
    "least_squares_from_formula",
    "LeastSquares",
]


@pl.api.register_expr_namespace("least_squares")
class LeastSquares:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def least_squares(self, *features: pl.Expr, **kwargs) -> pl.Expr:
        return least_squares(self._expr, *features, **kwargs)

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

    def rls(self, *features: pl.Expr, half_life: float | None, **kwargs):
        return recursive_least_squares(self._expr, *features, half_life=half_life, **kwargs)

    def expanding_ols(self, *features: pl.Expr, **kwargs):
        return self.rls(*features, half_life=None, **kwargs)

    def from_formula(self, formula: str, **kwargs) -> pl.Expr:
        features, add_intercept = build_expressions_from_patsy_formula(
            formula, include_dependent_variable=False
        )
        return self.least_squares(*features, add_intercept=add_intercept, **kwargs)
