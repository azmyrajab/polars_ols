from __future__ import annotations

from typing import List, Optional, Union

import polars as pl

from polars_ols.least_squares import (
    NullPolicy,
    OLSKwargs,
    OutputMode,
    RLSKwargs,
    RollingKwargs,
    SolveMethod,
    compute_least_squares,
    compute_least_squares_from_formula,
    compute_multi_target_least_squares,
    compute_recursive_least_squares,
    compute_rolling_least_squares,
    predict,
)
from polars_ols.utils import build_expressions_from_patsy_formula

__all__ = [
    "compute_least_squares",
    "compute_recursive_least_squares",
    "compute_rolling_least_squares",
    "compute_multi_target_least_squares",
    "compute_least_squares_from_formula",
    "LeastSquares",
]

ExprOrStr = Union[pl.Expr, str]


@pl.api.register_expr_namespace("least_squares")
class LeastSquares:
    """Registers the `.least_squares` namespace and provide entry points for the various models
     supported by this extension package.

     The below are parameters common to all models:
        - `sample_weights`: Optional expression representing sample weights.
        - `add_intercept`: Whether to add an intercept column.
        - `mode`: Mode of operation ("predictions", "residuals", "coefficients").
        - `null_policy`: Strategy for handling missing data, it can be:
            "ignore": does no null handling - use this option if nulls are already handled upstream.
            "zero": simply zero fills nulls in both targets & features prior to fitting.
            "drop": drops any rows (across targets or features) which have nulls prior to computing
                    coefficients. For non moving-window models, it then masks associated predictions
                    with nulls. For moving-window models coefficients are forward-filled.
            "drop_y_zero_x": Similar to "drop", but only rows with null targets are masked.
                             If features contain remaining nulls, it fills them with zeros.
                             This option allows extrapolation.
            "drop_zero": Masks rows with null targets or features similar in fitting coefficients.
                         similar to "drop". However, for predictions it uses zero filled features
                         dotted with those coefficients instead of masking with null.
                         This option allows extrapolation.

    Remaining parameters are model specific, see `OLSKwargs`, `RLSKwargs`, & `RollingKwargs`
    dataclasses for details.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def least_squares(
        self,
        *features: ExprOrStr,
        sample_weights: Optional[ExprOrStr] = None,
        add_intercept: bool = False,
        mode: OutputMode = "predictions",
        null_policy: NullPolicy = "ignore",
        solve_method: Optional[SolveMethod] = None,
        multi_target: bool = False,
        **ols_kwargs,
    ) -> pl.Expr:
        """Perform least squares regression.

        :param features: Variable number of feature expressions.
        :param sample_weights: Optional expression representing sample weights.
        :param add_intercept: Whether to add an intercept column.
        :param mode: Mode of operation ("predictions", "residuals", "coefficients").
        :param null_policy: Strategy for handling missing data.
        :param solve_method: Algorithm used for computing least squares solution.
                             Defaults to None, where a recommended default method is chosen based
                             on parametrization of the least-squares problem.
                             It can be one of ("qr", "svd", "chol", "lu", "cd").
        :param multi_target: boolean indicating if the target expression is multi-target struct.
        :param ols_kwargs: Additional, optional, model specific kwargs. See OLSKwargs.
        """
        ols_func = compute_least_squares if not multi_target else compute_multi_target_least_squares
        return ols_func(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            ols_kwargs=OLSKwargs(null_policy=null_policy, solve_method=solve_method, **ols_kwargs),
        )

    def ols(self, *features: ExprOrStr, **kwargs) -> pl.Expr:
        """Performs ordinary least squares. Alias for `least_squares`."""
        return self.least_squares(*features, **kwargs)

    def multi_target_ols(self, *features: ExprOrStr, **kwargs) -> pl.Expr:
        return self.least_squares(*features, multi_target=True, **kwargs)

    def wls(self, *features: ExprOrStr, sample_weights: ExprOrStr, **kwargs) -> pl.Expr:
        """Performs weighted least squares. Alias for `least_squares`.

        :param features: Variable number of feature expressions.
        :param sample_weights: optional expression representing sample weights.
        :param kwargs: additional kwargs passed to `least_squares` function.
        """
        return self.least_squares(*features, sample_weights=sample_weights, **kwargs)

    def ridge(self, *features: ExprOrStr, alpha: float, **kwargs) -> pl.Expr:
        """Performs ridge regression. Alias for `least_squares`.

        :param features: Variable number of feature expressions.
        :param alpha: L2 regularization parameter. Must be non-negative.
        :param kwargs: additional kwargs passed to `least_squares` function.
        """
        return self.least_squares(*features, alpha=alpha, l1_ratio=0.0, **kwargs)

    def lasso(self, *features: ExprOrStr, alpha: float, **kwargs) -> pl.Expr:
        """Performs lasso regression. Alias for `least_squares`.

        :param features: Variable number of feature expressions.
        :param alpha: L1 regularization parameter. Must be non-negative.
        :param kwargs: additional kwargs passed to `least_squares` function.
        """
        return self.least_squares(*features, alpha=alpha, l1_ratio=1.0, **kwargs)

    def elastic_net(
        self,
        *features: ExprOrStr,
        alpha: float,
        l1_ratio: float = 0.5,
        positive: bool = False,
        **kwargs,
    ):
        """Performs lasso regression. Alias for `least_squares`.

        The implementation uses cyclic coordinate descent.

        :param features: Variable number of feature expressions.
        :param alpha: Regularization strength parameter. Must be non-negative.
        :param l1_ratio: mixing parameter for elastic net regularization. Must be in [0., 1.]
                         (0. for Ridge, 1. for LASSO).
        :param positive: if set, enforces non-negativity constraints on coefficients.
        :param kwargs: additional kwargs passed to `least_squares` function.
        """
        return self.least_squares(
            *features, alpha=alpha, l1_ratio=l1_ratio, positive=positive, **kwargs
        )

    def rls(
        self,
        *features: ExprOrStr,
        sample_weights: Optional[ExprOrStr] = None,
        add_intercept: bool = False,
        mode: OutputMode = "predictions",
        null_policy: NullPolicy = "drop",
        half_life: Optional[float] = None,
        initial_state_covariance: Optional[float] = 10.0,
        initial_state_mean: Union[Optional[List[float], float]] = None,
    ):
        """Performs recursive least squares estimation.

        This model incrementally updates the least squares coefficients per observation
        in an efficient manner.

        General parameters:
        :param features: Variable number of feature expressions.
        :param sample_weights: Optional expression representing sample weights.
        :param add_intercept: Whether to add an intercept column.
        :param mode: Mode of operation ("predictions", "residuals", "coefficients").
        :param null_policy: Strategy for handling missing data. Recursive least squares model
                            will always forward fill coefficients in cased of masked observations.

        Model specific parameters:
        :param half_life: Half-life parameter for exponential forgetting. Defaults to None, which is
                          equivalent to expanding window least-squares (no forgetting).
        :param initial_state_mean: Denotes a prior for the mean value of the coefficients.
                          It can be either a scalar or a list. Defaults to None (no prior).
        :param initial_state_covariance: Scalar which denotes a prior of the uncertainty (variance)
                          around `initial_state_mean`. It can be thought of as inversely
                          proportional to the strength of an L2 penalty in a ridge regression.
                          Defaults to 10, intended to be a weak 'diffuse' prior.
        """
        return compute_recursive_least_squares(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            rls_kwargs=RLSKwargs(
                null_policy=null_policy,
                half_life=half_life,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
            ),
        )

    def rolling_ols(
        self,
        *features: ExprOrStr,
        window_size: int,
        sample_weights: Optional[ExprOrStr] = None,
        add_intercept: bool = False,
        mode: OutputMode = "predictions",
        null_policy: NullPolicy = "drop",
        min_periods: Optional[int] = None,
        use_woodbury: Optional[bool] = None,
        alpha: Optional[float] = None,
    ):
        """Performs rolling window least squares estimation by incrementally updating its state.

        General parameters:
        :param features: Variable number of feature expressions.
        :param sample_weights: Optional expression representing sample weights.
        :param add_intercept: Whether to add an intercept column.
        :param mode: Mode of operation ("predictions", "residuals", "coefficients").
        :param null_policy: Strategy for handling missing data. See `NullPolicy` for all options.
            Specific to the rolling window context:
                - "drop_window": This policy is specific to rolling ols. For every fixed rolling
                 window of data: drop rows with null values and only use remaining observations
                 for estimation. If number of valid observations per window doesn't exceed
                 min_periods: forward fill last previous coefficients. Note that this behavior is
                 similar to statsmodels RollingOLS' missing='drop' behaviour.
                - "drop": Logically equivalent to dropping any rows with null rows,
                computing rolling window on that data, then re-aligning with forward fill to the
                original data shape. I.e. it always operates on the last `window_size`
                valid observations.

        Model specific parameters:
        :param window_size: The size of the rolling window.
        :param min_periods: The minimum number of observations required to produce estimates.
        :param use_woodbury: Whether to use the Woodbury matrix identity for faster computation,
                             this allows updating inv(XTX) directly avoiding an expensive inversion
                             if number of features is large.
                             Defaults to None: which triggers Woodbury updates if num_features > 60.
        :param alpha: L2 Regularization strength. Default is 0.0.
        """
        return compute_rolling_least_squares(
            self._expr,
            *features,
            sample_weights=sample_weights,
            add_intercept=add_intercept,
            mode=mode,
            rolling_kwargs=RollingKwargs(
                window_size=window_size,
                min_periods=min_periods,
                use_woodbury=use_woodbury,
                alpha=alpha,
                null_policy=null_policy,
            ),
        )

    def expanding_ols(self, *features: ExprOrStr, **kwargs):
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

    def predict(
        self,
        *features: ExprOrStr,
        name: Optional[str] = None,
        add_intercept: bool = False,
        null_policy: NullPolicy = "zero",
    ) -> pl.Expr:
        return predict(
            self._expr,
            *features,
            add_intercept=add_intercept,
            name=name,
            null_policy=null_policy,
        )

    def predict_from_formula(self, formula: str, name: Optional[str] = None) -> pl.Expr:
        features, add_intercept = build_expressions_from_patsy_formula(
            formula, include_dependent_variable=False
        )
        has_const = any(f.meta.output_name == "const" for f in features)
        add_intercept &= not has_const
        return self.predict(*features, name=name, add_intercept=add_intercept)
