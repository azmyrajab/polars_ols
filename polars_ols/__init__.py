import polars as pl
from polars_ols.least_squares import pl_least_squares
from functools import reduce
from typing import Sequence

__all__ = [
    "pl_least_squares",
    "pl_least_squares_from_formula",
    "LeastSquares",
]


def _build_expressions_from_patsy_formula(formula: str, include_dependent_variable: bool = False
                                          ) -> (Sequence[pl.Expr], bool):
    try:
        import patsy as pa
    except ImportError:
        raise NotImplementedError("'patsy' needs to be installed in your python environment in order to use "
                                  "formula api")
    desc = pa.ModelDesc.from_formula(formula)

    if include_dependent_variable:
        assert len(desc.lhs_termlist) == 1, "must provide exactly one LHS variable"
        terms = desc.lhs_termlist + desc.rhs_termlist
    else:
        assert len(desc.lhs_termlist) == 0, "can not provide LHS variables in this context"
        terms = desc.rhs_termlist

    add_intercept: bool = not("-1" in formula)

    expressions = []
    for term in terms:
        if any("C(" in f.code for f in term.factors):
            raise NotImplementedError("building patsy categories into polars expressions is not supported")
        if len(term.factors) == 1:
            expressions.append(pl.col(term.factors[0].code))
        elif len(term.factors) >= 2:
            expr = reduce((lambda x, y: x * pl.col(y)), (f.code for f in term.factors), pl.lit(1))
            expressions.append(expr.alias(":".join(f.code for f in term.factors)))
    return expressions, add_intercept


def pl_least_squares_from_formula(formula: str, **kwargs) -> pl.Expr:
    expressions, add_intercept = _build_expressions_from_patsy_formula(formula, include_dependent_variable=True)
    return pl_least_squares(expressions[0], *expressions[1:],  add_intercept=add_intercept, **kwargs)


@pl.api.register_expr_namespace("least_squares")
class LeastSquares:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def least_squares(self, *features: pl.Expr, **kwargs) -> pl.Expr:
        return pl_least_squares(self._expr, *features, **kwargs)

    def ols(self, *features: pl.Expr) -> pl.Expr:
        return self.least_squares(*features)

    def wls(self, *features: pl.Expr, sample_weights: pl.Expr) -> pl.Expr:
        return self.least_squares(*features, sample_weights=sample_weights)

    def ridge(self, *features: pl.Expr, alpha: float) -> pl.Expr:
        return self.least_squares(*features, ridge_alpha=alpha)

    def from_formula(self, formula: str, **kwargs) -> pl.Expr:
        features, add_intercept = _build_expressions_from_patsy_formula(formula, include_dependent_variable=False)
        return self.least_squares(*features, add_intercept=add_intercept, **kwargs)
