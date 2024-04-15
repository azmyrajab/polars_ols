from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Union,
    get_args,
)

from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    import polars as pl
    from polars.type_aliases import IntoExpr

from polars_ols.utils import build_expressions_from_patsy_formula, parse_into_expr

logger = logging.getLogger(__name__)

__all__ = [
    # fitting and prediction functions
    "compute_least_squares",
    "compute_recursive_least_squares",
    "compute_rolling_least_squares",
    "compute_least_squares_from_formula",
    "predict",
    # model specific parameters
    "OLSKwargs",
    "RLSKwargs",
    "RollingKwargs",
    # types controlling general modelling behaviour
    "NullPolicy",
    "OutputMode",
    "SolveMethod",
]

NullPolicy = Literal[
    "zero",  # simply zero fills nulls in both targets & features
    "drop",  # drops any rows with nulls in fitting and masks associated predictions with nulls
    "ignore",  # use this option if nulls are already handled upstream
    "drop_zero",  # drops any rows with nulls in fitting, but then computes predictions
    # with zero filled features. Use this to allow for extrapolation.
    "drop_y_zero_x",  # only drops rows with null targets and fill any null features with zero
    "skip",  # only relevant to rolling window regression: this causes any observations with null
    # to be omitted and only valid observations within the fixed window are used.
]
OutputMode = Literal["predictions", "residuals", "coefficients"]
SolveMethod = Literal["qr", "svd", "chol", "lu", "cd"]

_VALID_NULL_POLICIES: Set[NullPolicy] = set(get_args(NullPolicy))
_VALID_OUTPUT_MODES: Set[OutputMode] = set(get_args(OutputMode))
_VALID_SOLVE_METHODS: Set[SolveMethod] = set(get_args(SolveMethod)).union({None})


@dataclass
class Kwargs:
    null_policy: NullPolicy = "ignore"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        # rust code does validate all options, but prefer to fail, on some, early
        assert (
            self.null_policy in _VALID_NULL_POLICIES
        ), f"'null_policy' must be one of {_VALID_NULL_POLICIES}. You passed: {self.null_policy}"


@dataclass
class OLSKwargs(Kwargs):
    """Specifies parameters relevant for regularized linear models: LASSO / Ridge / ElasticNet.

    Attributes:
        alpha: Regularization strength. Defaults to 0.0.
        l1_ratio: Mixing parameter for ElasticNet regularization (0 for Ridge, 1 for LASSO).
            Defaults to None (equivalent to Ridge regression).
        max_iter: Maximum number of iterations. Defaults to 1000 iterations.
        tol: Tolerance for convergence criterion. Defaults to 1.e-5.
        positive: Whether to enforce non-negativity constraints on coefficients.
            Defaults to False (no constraint on coefficients).
        null_policy: Strategy for handling missing data. Defaults to "ignore".
        solve_method: Algorithm used for computing least squares solution.
            Defaults to None, where a recommended default method is chosen based on problem
            specifics.
        rcond: Optional float specifying cut-off ratio for small singular values. Only relevant for
               "SVD" solve methods. Defaults to None, where it is chosen as per
                numpy lstsq convention.
    """

    alpha: Optional[float] = 0.0
    l1_ratio: Optional[float] = None
    max_iter: Optional[int] = 1_000
    tol: Optional[float] = 1.0e-5
    positive: Optional[bool] = False  # if True, imposes non-negativity constraint on coefficients
    solve_method: Optional[SolveMethod] = None
    rcond: Optional[float] = None

    def __post_init__(self):
        # rust code does validate all options, but prefer to fail, on some, early
        valid_ols_policies = _VALID_NULL_POLICIES - {"skip"}  # skip does not make
        # sense outside of moving window models.
        assert (
            self.null_policy in valid_ols_policies
        ), f"'null_policy' must be one of {valid_ols_policies}. You passed: {self.null_policy}"
        assert (
            self.solve_method in _VALID_SOLVE_METHODS
        ), f"'solve_method' must be one of {_VALID_SOLVE_METHODS}. You passed: {self.solve_method}"


@dataclass
class RLSKwargs(Kwargs):
    """Specifies parameters of Recursive Least Squares models.

    Attributes:
        half_life: Half-life parameter for exponential forgetting. Defaults to None
            (no forgetting).
        initial_state_covariance: Scalar representing which behaves like an L2 regularization
                                  parameter. Larger values correspond to larger prior uncertainty
                                  around mean vector of state (inversely proportional to strength
                                  of equivalent L2 penalty). Defaults to 10.
        initial_state_mean: Initial mean vector of the state. Defaults to None.
        null_policy: Strategy for handling missing data. Defaults to "drop".
    """

    half_life: Optional[float] = None
    initial_state_covariance: Optional[float] = 10.0
    initial_state_mean: Union[Optional[List[float], float]] = None
    null_policy: NullPolicy = "drop"


@dataclass
class RollingKwargs(Kwargs):
    """Specifies parameters of Rolling OLS model.

    Attributes:
        window_size: The size of the rolling window.
        min_periods: The minimum number of observations required to produce estimates.
        use_woodbury: Whether to use Woodbury matrix identity for faster computation.
                      Defaults to True if num_features > 10.
        alpha: L2 Regularization strength. Default is 0.0.
        null_policy: Strategy for handling missing data. Defaults to "ignore".
    """

    window_size: int = 1_000_000  # defaults to expanding OLS
    min_periods: Optional[int] = None
    use_woodbury: Optional[bool] = None
    alpha: Optional[float] = None  # optional ridge alpha
    null_policy: NullPolicy = "drop"


def _pre_process_data(
    target: pl.Expr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr],
    add_intercept: bool,
):
    """Pre-processes the input data by adding an intercept to features and applying sample weights,
    if specified.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.

    Returns:
        Tuple containing the pre-processed target, features, and sample weights.
    """
    target = parse_into_expr(target)
    features = [parse_into_expr(f) for f in features]
    # handle intercept
    if add_intercept:
        if any(f.meta.output_name == "const" for f in features):
            logger.info("feature named 'const' already detected, assuming it is an intercept")
        else:
            features.append(target.fill_null(0.0).mul(0.0).add(1.0).alias("const"))
    # handle sample weights
    sqrt_w: Optional[pl.Expr] = None
    if sample_weights is not None:
        sqrt_w = sample_weights.sqrt()
        target *= sqrt_w
        features = [(expr * sqrt_w) for expr in features]
    return target, features, sqrt_w


def _register_least_squares_plugin(
    target: pl.Expr,
    *features: pl.Expr,
    mode: OutputMode,
    function_name: str,
    ols_kwargs: Kwargs,
    returns_scalar_coefficients: bool = False,
    **kwargs,
):
    # pre-process features and targets prior to fitting
    target = parse_into_expr(target)
    target_fit, features_fit, sqrt_w = _pre_process_data(target, *features, **kwargs)

    # register either coefficient or prediction plugin functions
    if mode == "coefficients":
        # TODO: remove 'rename_fields' after https://github.com/pola-rs/pyo3-polars/issues/79
        #  it currently breaks input_wildcard_expansion=True correctly returning a struct.
        return (
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name=f"{function_name}_coefficients",
                args=[target_fit, *features_fit],
                kwargs=ols_kwargs.to_dict(),
                is_elementwise=False,
                changes_length=returns_scalar_coefficients,
                returns_scalar=returns_scalar_coefficients,
                input_wildcard_expansion=True,
            )
            .alias("coefficients")
            .struct.rename_fields([f.meta.output_name() for f in features_fit])
        )
    else:
        predictions = register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name=function_name,
            args=[target_fit, *features_fit],
            kwargs=ols_kwargs.to_dict(),
            is_elementwise=False,
            input_wildcard_expansion=True,
        )
        if sqrt_w is not None:
            predictions /= sqrt_w  # undo the scaling implicit in WLS weighting
        if mode == "predictions":
            return predictions
        else:
            return target - predictions


def compute_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: OutputMode = "predictions",
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
        ols_kwargs: Additional keyword arguments specific for regularized OLS models. See OLSKwargs.
    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of {_VALID_OUTPUT_MODES}"
    ols_kwargs: OLSKwargs = ols_kwargs or OLSKwargs()
    # register either coefficient or prediction plugin functions
    return _register_least_squares_plugin(
        target,
        *features,
        mode=mode,
        function_name="least_squares",
        ols_kwargs=ols_kwargs,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
        returns_scalar_coefficients=True,
    )


def compute_recursive_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: OutputMode = "predictions",
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
        rls_kwargs: Additional keyword arguments for the recursive least squares model.
                    See RLSKwargs.

    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of {_VALID_OUTPUT_MODES}"
    rls_kwargs: RLSKwargs = rls_kwargs or RLSKwargs()
    # register either coefficient or prediction plugin functions
    return _register_least_squares_plugin(
        target,
        *features,
        mode=mode,
        function_name="recursive_least_squares",
        ols_kwargs=rls_kwargs,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
    )


def compute_rolling_least_squares(
    target: IntoExpr,
    *features: pl.Expr,
    sample_weights: Optional[pl.Expr] = None,
    add_intercept: bool = False,
    mode: OutputMode = "predictions",
    rolling_kwargs: Optional[RollingKwargs] = None,
) -> pl.Expr:
    """Performs least squares regression in a rolling window fashion.

    Args:
        target: The target expression.
        *features: Variable number of feature expressions.
        sample_weights: Optional expression representing sample weights.
        add_intercept: Whether to add an intercept column.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
        rolling_kwargs: Additional keyword arguments for the rolling least squares model.
                        See RollingKwargs.

    Returns:
        Resulting expression based on the chosen mode.
    """
    assert mode in _VALID_OUTPUT_MODES, f"'mode' must be one of {_VALID_OUTPUT_MODES}"
    rolling_kwargs: RollingKwargs = rolling_kwargs or RollingKwargs()
    # register either coefficient or prediction plugin functions
    expr = _register_least_squares_plugin(
        target,
        *features,
        mode=mode,
        function_name="rolling_least_squares",
        ols_kwargs=rolling_kwargs,
        sample_weights=sample_weights,
        add_intercept=add_intercept,
    )
    if mode in {"predictions", "residuals"}:
        expr = expr.fill_nan(None)
    return expr


def compute_least_squares_from_formula(
    formula: str,
    sample_weights: Optional[pl.Expr] = None,
    mode: OutputMode = "predictions",
    **kwargs,
) -> pl.Expr:
    """Performs least squares regression using a formula.

    Depending on choice of additional kwargs dispatches either rolling, recursive,
    on static least squares compute functions.

    Args:
        formula: Patsy-style formula string.
        sample_weights: Optional expression representing sample weights.
        mode: Mode of operation ("predictions", "residuals", "coefficients").
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
        mode=mode,
    )


def predict(
    coefficients: IntoExpr,
    *features: pl.Expr,
    null_policy: NullPolicy = "zero",
    add_intercept: bool = False,
    name: Optional[str] = None,
) -> pl.Expr:
    """Helper which computes predictions as a product of (aligned) coefficients with features.

    Args:
        coefficients: Polars expression returning a coefficients struct.
        *features: variable number of feature expressions.
        null_policy: specifies how nulls in features are handled. Defaults to zero filling.
        add_intercept: boolean indicating if a constant should be added to features.
        name: optional str defining an alias for computed predictions expression.

    Returns:
        polars expression denoting computed predictions.
    """
    assert null_policy in _VALID_NULL_POLICIES, "'null_policy' must be one of {drop, ignore, zero}"
    if add_intercept:
        if any(f.meta.output_name == "const" for f in features):
            logger.warning("feature named 'const' already detected, assuming it is the intercept")
        else:
            features += (features[-1].fill_null(0.0).mul(0.0).add(1.0).alias("const"),)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="predict",
        args=[coefficients, *features],
        kwargs={"null_policy": null_policy},
        is_elementwise=False,
        input_wildcard_expansion=True,
    ).alias(name or "predictions")


def convert_series_to_struct(*inputs: pl.Expr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="inputs_to_struct",
        args=inputs,
    )
