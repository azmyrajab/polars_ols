from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Sequence, Tuple

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr, PolarsDataType


def parse_into_expr(
    expr: IntoExpr,
    *,
    str_as_lit: bool = False,
    list_as_lit: bool = True,
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    expr
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to `False` (default),
        strings are parsed as column names.
    list_as_lit
        Interpret list input as a lit literal, If set to `False`,
        lists are parsed as `Series` literals.
    dtype
        If the input is expected to resolve to a literal with a known dtype, pass
        this to the `lit` constructor.

    Returns
    -------
    polars.Expr
    """
    if isinstance(expr, pl.Expr):
        pass
    elif isinstance(expr, str) and not str_as_lit:
        expr = pl.col(expr)
    elif isinstance(expr, list) and not list_as_lit:
        expr = pl.lit(pl.Series(expr), dtype=dtype)
    else:
        expr = pl.lit(expr, dtype=dtype)

    return expr


def build_expressions_from_patsy_formula(
    formula: str, include_dependent_variable: bool = False
) -> Tuple[Sequence[pl.Expr], bool]:
    try:
        import patsy as pa
    except ImportError as e:
        raise NotImplementedError(
            "'patsy' needs to be installed in your python environment in order to use "
            "formula api"
        ) from e
    desc = pa.ModelDesc.from_formula(formula)

    if include_dependent_variable:
        assert len(desc.lhs_termlist) == 1, "must provide exactly one LHS variable"
        terms = desc.lhs_termlist + desc.rhs_termlist
    else:
        assert len(desc.lhs_termlist) == 0, "can not provide LHS variables in this context"
        terms = desc.rhs_termlist

    add_intercept: bool = "-1" not in formula

    expressions = []
    for term in terms:
        if any("C(" in f.code for f in term.factors):
            raise NotImplementedError(
                "building patsy categories into polars expressions is not supported"
            )
        if len(term.factors) == 1:
            expressions.append(pl.col(term.factors[0].code))
        elif len(term.factors) >= 2:
            expr = reduce((lambda x, y: x * pl.col(y)), (f.code for f in term.factors), pl.lit(1))
            expressions.append(expr.alias(":".join(f.code for f in term.factors)))
    return expressions, add_intercept
