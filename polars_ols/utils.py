from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache, reduce
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr, PolarsDataType


__all__ = [
    "parse_into_expr",
    "build_expressions_from_patsy_formula",
    "timer",
]


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


@lru_cache(maxsize=100)
def build_expressions_from_patsy_formula(
    formula: str, include_dependent_variable: bool = False
) -> Tuple[Sequence[pl.Expr], bool]:
    """Builds polars LHS and/or RHS expressions given a patsy formula.

    Only a subset of supported features are supported, in particular:
    - simple target to feature formula w/ interaction variables and intercept are
    fully supported (e.g. 'y ~ x1 + x2:x3 -1')
    - external functions applied to columns are not yet supported (e.g. "log(x1)")
    - categorical features can not be supported, you have to 'pivot' yourself (e.g. "C(group)")

    Example:
        >>> ex, intercept = build_expressions_from_patsy_formula("y ~ x1 + x2 + x3:x4",
        ... include_dependent_variable=True)
        >>> assert str(pl.col("y")) == str(ex[0])
        >>> assert str(pl.col("x1")) == str(ex[1])
        >>> assert str((1 * pl.col("x3") * pl.col("x4")).alias("x3:x4")) == str(ex[-1])
    """
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


@contextmanager
def timer(msg: Optional[str] = None, precision: int = 3) -> float:
    start = time.perf_counter()
    end = start
    yield lambda: end - start
    msg = f"{msg or 'Took'}: {(time.perf_counter() - start) * 1_000:.{precision}f} ms"
    print(msg)
    end = time.perf_counter()
