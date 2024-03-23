# Polars OLS
## Least squares extension in Polars

Supports linear model estimation in [Polars](https://www.pola.rs/).

This package provides efficient rust implementations of common linear
regression variants (OLS, WLS, Ridge, Elastic Net, Recursive least squares, Non-negative least squares) and exposes
them as simple polars expressions which can easily be integrated into your workflow.

### Why?

1. **High Performance**: implementations are written in rust and make use of optimized rust linear-algebra crates & LAPACK routines. See benchmark section.
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to other formats or multiple libraries (e.g. numpy, sklearn) to do simple linear regressions.
Chain least squares formulae like any other expression in polars.
3. **Efficient Implementation**:
   - Numerically stable algorithms are chosen where possible (e.g. QR, Cholesky).
   - Flexible model specification allows arbitrary combination of sample weighting, L1/L2 regularization, & non-negativity constraints on parameters.
   - Efficient rank-1 update algorithms used for moving window regressions.
4. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
5. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) which automatically converts to equivalent polars expressions.

Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/). Then simply run the below to install the `polars-ols` extension:
```console
pip install polars-ols
```

API & Examples
------------

Importing `polars_ols` will register the namespace `least_squares` provided by this package.
You can build models either by either specifying columns (`pl.col(...)`) for your targets and features or using
the formula api (patsy syntax). See below for examples.
```python
import polars as pl
import polars_ols as pls

ols_expr = pl.col("y").least_squares.from_formula("x1 + x2").over("group").alias("predictions_ols")
lasso_expr = pl.col("y").least_squares.lasso(pl.col("x1"), pl.col("x2"), alpha=0.01).alias("predictions_lasso")
```

Alternatively, you also access public methods provided by the package as well:

```python
import polars as pl
import polars_ols as pls

enet_expr = pls.least_squares_from_formula("y ~ x1 + x2",
                                          sample_weights=pl.col("weights"),  # specify column with sample weights (WLS)
                                          alpha=0.0001,   # specify regularization parameter
                                          l1_ratio=0.5,  # set 50% L1 penalty & 50% L2 penalty
                                          positive=True,  # enforce non-negativity on coefficients
                                          ).over("group").alias("predictions_enet")
```

The methods provided by the `least_squares` namespace return polars expressions which your polars dataframe
can access normally:

```python
df = pl.DataFrame({"y": [1.16, -2.16, -1.57, 0.21, 0.22, 1.6, -2.11, -2.92, -0.86, 0.47],
                   "x1": [0.72, -2.43, -0.63, 0.05, -0.07, 0.65, -0.02, -1.64, -0.92, -0.27],
                   "x2": [0.24, 0.18, -0.95, 0.23, 0.44, 1.01, -2.08, -1.36, 0.01, 0.75],
                   "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                   "weights": [0.34, 0.97, 0.39, 0.8, 0.57, 0.41, 0.19, 0.87, 0.06, 0.34],
                   })
predictions = df.with_columns(ols_expr.round(2), enet_expr.round(2), lasso_expr.round(2))

print(predictions)
```
```
shape: (10, 8)
┌───────┬───────┬───────┬───────┬─────────┬─────────────────┬──────────────────┬───────────────────┐
│ y     ┆ x1    ┆ x2    ┆ group ┆ weights ┆ predictions_ols ┆ predictions_enet ┆ predictions_lasso │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---     ┆ ---             ┆ ---              ┆ ---               │
│ f64   ┆ f64   ┆ f64   ┆ i64   ┆ f64     ┆ f32             ┆ f32              ┆ f32               │
╞═══════╪═══════╪═══════╪═══════╪═════════╪═════════════════╪══════════════════╪═══════════════════╡
│ 1.16  ┆ 0.72  ┆ 0.24  ┆ 1     ┆ 0.34    ┆ 0.97            ┆ 0.93             ┆ 0.93              │
│ -2.16 ┆ -2.43 ┆ 0.18  ┆ 1     ┆ 0.97    ┆ -2.23           ┆ -2.19            ┆ -2.18             │
│ -1.57 ┆ -0.63 ┆ -0.95 ┆ 1     ┆ 0.39    ┆ -1.54           ┆ -1.52            ┆ -1.54             │
│ 0.21  ┆ 0.05  ┆ 0.23  ┆ 1     ┆ 0.8     ┆ 0.29            ┆ 0.27             ┆ 0.27              │
│ 0.22  ┆ -0.07 ┆ 0.44  ┆ 1     ┆ 0.57    ┆ 0.37            ┆ 0.35             ┆ 0.36              │
│ 1.6   ┆ 0.65  ┆ 1.01  ┆ 2     ┆ 0.41    ┆ 1.6             ┆ 1.62             ┆ 1.62              │
│ -2.11 ┆ -0.02 ┆ -2.08 ┆ 2     ┆ 0.19    ┆ -2.11           ┆ -2.08            ┆ -2.05             │
│ -2.92 ┆ -1.64 ┆ -1.36 ┆ 2     ┆ 0.87    ┆ -2.91           ┆ -2.92            ┆ -2.92             │
│ -0.86 ┆ -0.92 ┆ 0.01  ┆ 2     ┆ 0.06    ┆ -0.87           ┆ -0.87            ┆ -0.88             │
│ 0.47  ┆ -0.27 ┆ 0.75  ┆ 2     ┆ 0.34    ┆ 0.48            ┆ 0.49             ┆ 0.47              │
└───────┴───────┴───────┴───────┴─────────┴─────────────────┴──────────────────┴───────────────────┘
```

A `mode` parameter is used to set the type of the output returned by all methods (`"predictions", "residuals", "coefficients"`).
Defaults to returning predictions.

In case `"coefficients"` is passed the output's shape is the number of features specified, and otherwise
the output will match the input's length. See below:

```python
coefficients = df.select(pl.col("y").least_squares.from_formula("x1 + x2", mode="coefficients")
                         .alias("coefficients").round(2))

print(coefficients)
```
```
shape: (3, 1)
┌──────────────┐
│ coefficients │
│ ---          │
│ f32          │
╞══════════════╡
│ 0.98         │ <-- x1
│ 0.99         │ <-- x2
│ 0.0          │ <-- intercept added by formula api
└──────────────┘
```

The only exception is dynamic models (e.g. recursive or rolling window least squares) where the coefficients as of each
sample are recorded in a `list[f32]` dtype (the length of each list is number of features).

```python
coefficients_rls = df.select(pl.col("y").least_squares.rls(
                pl.col("x1"),
                pl.col("x2"),
                mode="coefficients",
                half_life=None, # equivalent to expanding window (no forgetting)
                initial_state_covariance=0.25,  # L2 prior
                initial_state_mean=[0.0, 0.0],  # custom prior
            ).alias("coefficients_rls"))

print(coefficients_rls)
```
```
shape: (10, 1)
┌──────────────────────┐
│ coefficients_rls     │
│ ---                  │
│ list[f32]            │
╞══════════════════════╡
│ [0.182517, 0.060839] │
│ [0.583966, 0.010787] │
│ [0.646492, 0.233397] │
│ [0.646885, 0.239023] │
│ [0.645457, 0.252556] │
│ [0.686584, 0.395498] │
│ [0.666145, 0.647716] │
│ [0.759064, 0.727013] │
│ [0.77024, 0.723971]  │
│ [0.765996, 0.73275]  │
└──────────────────────┘
```

Supported Models
------------

Currently, this extension package supports the following variants:
- Ordinary Least Squares: ```least_squares.ols```
- Weighted Least Squares: ```least_squares.wls```
- Regularized Least Squares (Lasso / Ridge / Elastic Net) ```least_squares.{lasso, ridge, elastic_net}```
- Non-negative Least Squares: ```least_squares.nnls```
- Formula API: ```least_squares.from_formula```
- Recursive Least Squares: ```least_squares.rls```

Notice that an arbitrary combination of sample_weights, L1/L2 penalties, and non-negativity constraint can be used with
both the ```least_squares.from_formula``` and ```least_squares.least_squares``` methods outside of `least_squares.rls`.
