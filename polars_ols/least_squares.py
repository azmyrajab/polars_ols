import logging
from dataclasses import asdict, dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, get_args

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

from polars_ols.utils import build_expressions_from_patsy_formula, parse_into_expr

logger = logging.getLogger(__name__)

__all__ = [
    "compute_least_squares",
    "compute_recursive_least_squares",
    "compute_rolling_least_squares",
    "least_squares_from_formula",
    "predict",
    "OLSKwargs",
    "RLSKwargs",
    "RollingKwargs",
    "NullPolicy",
    "OutputMode",
]

NullPolicy = Literal["zero", "drop", "drop_y_zero_x", "ignore"]
OutputMode = Literal["predictions", "residuals", "coefficients"]

_VALID_NULL_POLICIES: Set[NullPolicy] = set(get_args(NullPolicy))
_VALID_OUTPUT_MODES: Set[OutputMode] = set(get_args(OutputMode))


@dataclass
class OLSKwargs:
    """Specifies parameters relevant for regularized linear models: LASSO / Ridge / ElasticNet.

    Attributes:
        alpha: Regularization strength. Defaults to 0.0.
        l1_ratio: Mixing parameter for ElasticNet regularization (0 for Ridge, 1 for LASSO).
            Defaults to None (equivalent to Ridge regression).
        max_iter: Maximum number of iterations. Defaults to 1000 iterations.
        tol: Tolerance for convergence criterion. Defaults to 0.0001.
        positive: Whether to enforce non-negativity constraints on coefficients.
            Defaults to False (no constraint on coefficients).
    """

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
    target: pl.Expr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr],
    add_intercept: bool,
    null_policy: NullPolicy,
):
    """Pre-processes the input data by casting it to float32 and scaling it with sample weights if
     provided.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        null_policy: literal defining a strategy for handling missing data.

    Returns:
        Tuple containing the pre-processed target, features, and sample weights.
    """
    target = parse_into_expr(target).cast(pl.Float32)
    features = [f.cast(pl.Float32) for f in features]

    # handle nulls
    match null_policy:
        case "zero":
            target = target.fill_null(0.0)
            features = [f.fill_null(0.0) for f in features]
        case "drop_y_zero_x":
            # drops rows based on missing targets, and otherwise zero's out any
            # remaining missing features
            is_valid = target.is_not_null()
            target = target.filter(is_valid)
            features = [f.filter(is_valid).fill_null(0.0) for f in features]
        case "drop":
            # drops rows if either target or any feature is missing
            is_valid_y = target.is_not_null()
            is_valid_x = reduce(lambda x, y: x.is_not_null() & y.is_not_null(), features)
            is_valid = is_valid_y & is_valid_x
            features = [f.filter(is_valid) for f in features]
            target = target.filter(is_valid)
        case "ignore":
            # only choose this if nulls are already handled upstream
            pass
        case _:
            raise NotImplementedError(
                f"null_policy: '{null_policy}' is not supported. "
                f"It must be one of '{_VALID_NULL_POLICIES}'"
            )

    # handle intercept
    if add_intercept:
        if any(f.meta.output_name == "const" for f in features):
            logger.info("feature named 'const' already detected, assuming it is an intercept")
        else:
            features += [target.mul(0.0).add(1.0).alias("const")]
    # handle sample weights
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
    mode: OutputMode = "predictions",
    null_policy: NullPolicy = "ignore",
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
        null_policy: Strategy for handling missing data ("zero", "drop", "ignore").
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
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of '{_VALID_OUTPUT_MODES}'"
    target, features, sqrt_w = _pre_process_data(
        target,
        *features,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
        null_policy=null_policy,
    )

    ols_kwargs: OLSKwargs = ols_kwargs or OLSKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="least_squares_coefficients",
                args=[target, *features],
                kwargs=ols_kwargs.to_dict(),
                is_elementwise=False,
                changes_length=True,
                returns_scalar=True,
            )
            .alias("coefficients")
            .struct.rename_fields([f.meta.output_name() for f in features])
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
            return target - predictions


def compute_recursive_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: OutputMode = "predictions",
    null_policy: NullPolicy = "ignore",
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
        null_policy: Strategy for handling missing data ("zero", "drop", "ignore").
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
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of '{_VALID_OUTPUT_MODES}'"
    target, features, sqrt_w = _pre_process_data(
        target,
        *features,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
        null_policy=null_policy,
    )
    rls_kwargs: RLSKwargs = rls_kwargs or RLSKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="recursive_least_squares_coefficients",
                args=[target, *features],
                kwargs=rls_kwargs.to_dict(),
                is_elementwise=False,
            )
            .alias("coefficients")
            .struct.rename_fields([f.meta.output_name() for f in features])
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
            return target - predictions


def compute_rolling_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: OutputMode = "predictions",
    null_policy: NullPolicy = "ignore",
    rolling_kwargs: Optional[RollingKwargs] = None,
) -> pl.Expr:
    """Performs least squares regression in a rolling window fashion.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        null_policy: Strategy for handling missing data ("zero", "drop", "ignore").
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
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of '{_VALID_OUTPUT_MODES}'"
    target, features, sqrt_w = _pre_process_data(
        target,
        *features,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
        null_policy=null_policy,
    )
    rolling_kwargs: RollingKwargs = rolling_kwargs or RollingKwargs()

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        return (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="rolling_least_squares_coefficients",
                args=[target, *features],
                kwargs=rolling_kwargs.to_dict(),
                is_elementwise=False,
            )
            .alias("coefficients")
            .struct.rename_fields([f.meta.output_name() for f in features])
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
            return target - predictions


def least_squares_from_formula(
    formula: str,
    sample_weights: Optional[pl.Expr] = None,
    mode: OutputMode = "predictions",
    null_policy: NullPolicy = "ignore",
    **kwargs,
) -> pl.Expr:
    """Performs least squares regression using a formula.

    Depending on choice of additional kwargs dispatches either rolling, recursive,
    on static least squares compute functions.

    Args:
        formula: Patsy-style formula string.
        sample_weights: Optional expression representing sample weights.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        null_policy: Strategy for handling missing data ("zero", "drop", "ignore").
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
        null_policy=null_policy,
        mode=mode,
    )


def predict(
    coefficients: IntoExpr,
    *features: pl.Expr,
    add_intercept: bool = False,
    name: Optional[str] = None,
) -> pl.Expr:
    """Helper which computes predictions as a product of (aligned) coefficients with features.

    Args:
        coefficients: Polars expression returning a coefficients struct.
        *features: variable number of feature expressions.
        add_intercept: boolean indicating if a constant should be added to features.
        name: optional str defining an alias for computed predictions expression.

    Returns:
        polars expression denoting computed predictions.
    """
    if add_intercept:
        if any(f.meta.output_name == "const" for f in features):
            logger.warning("feature named 'const' already detected, assuming it is the intercept")
        else:
            features += (features[-1].fill_null(0.0).mul(0.0).add(1.0).alias("const"),)

    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="predict",
        args=[coefficients, *(f.fill_null(0.0).cast(pl.Float32) for f in features)],
        is_elementwise=False,
    ).alias(name or "predictions")
