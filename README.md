# Polars OLS
## Least squares extension in Polars

Supports linear model estimation in [Polars](https://www.pola.rs/).

This package provides efficient rust implementations of common linear
regression variants (OLS, WLS, Ridge, Elastic Net, Non-negative least squares, Recursive least squares) and exposes
them as simple polars expressions which can easily be integrated into your workflow.

### Why?

1. **High Performance**: implementations are written in rust and make use of optimized rust linear-algebra crates & LAPACK routines. See _benchmark_ section.
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to other formats or multiple libraries (e.g. numpy, sklearn) to do simple linear regressions.
Chain least squares formulae like any other expression in polars.
3. **Efficient Implementations**:
   - Numerically stable algorithms are chosen where possible (e.g. QR, Cholesky).
   - Flexible model specification allows arbitrary combination of sample weighting, L1/L2 regularization, & non-negativity constraints on parameters.
   - Efficient rank-1 update algorithms used for moving window regressions.
4. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
5. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) which automatically converts to equivalent polars expressions.

Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/). Then run the below to install the `polars-ols` extension:
```console
pip install polars-ols
```

API & Examples
------------

Importing `polars_ols` will register the namespace `least_squares` provided by this package.
You can build models either by either specifying polars expressions (e.g. `pl.col(...)`) for your targets and features or using
the formula api (patsy syntax). All models support the following general (optional) arguments:
- `mode` - a literal which determines the type of output produced by the model
- `add_intercept` - a boolean specifying if an intercept feature should be added to the features
- `sample_weights` - a column or expression providing non-negative weights applied to the samples

Remaining parameters are model specific, for example `alpha` penalty parameter used by regularized least squares models.

See below for basic usage examples.
Please refer to the [tests](./tests/test_ols.py) or [demo notebook](./notebooks/polars_ols_demo.ipynb) for detailed examples.

```python
import polars as pl
import polars_ols as pls  # registers 'least_squares' namespace

df = pl.DataFrame({"y": [1.16, -2.16, -1.57, 0.21, 0.22, 1.6, -2.11, -2.92, -0.86, 0.47],
                   "x1": [0.72, -2.43, -0.63, 0.05, -0.07, 0.65, -0.02, -1.64, -0.92, -0.27],
                   "x2": [0.24, 0.18, -0.95, 0.23, 0.44, 1.01, -2.08, -1.36, 0.01, 0.75],
                   "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                   "weights": [0.34, 0.97, 0.39, 0.8, 0.57, 0.41, 0.19, 0.87, 0.06, 0.34],
                   })

lasso_expr = pl.col("y").least_squares.lasso(pl.col("x1"), pl.col("x2"), alpha=0.0001, add_intercept=True).over("group")
wls_expr = pls.least_squares_from_formula("y ~ x1 + x2 -1", sample_weights=pl.col("weights"))

predictions = df.with_columns(lasso_expr.round(2).alias("predictions_lasso"),
                              wls_expr.round(2).alias("predictions_wls"))

print(predictions.head(5))
```
```
shape: (5, 7)
┌───────┬───────┬───────┬───────┬─────────┬───────────────────┬─────────────────┐
│ y     ┆ x1    ┆ x2    ┆ group ┆ weights ┆ predictions_lasso ┆ predictions_wls │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---     ┆ ---               ┆ ---             │
│ f64   ┆ f64   ┆ f64   ┆ i64   ┆ f64     ┆ f32               ┆ f32             │
╞═══════╪═══════╪═══════╪═══════╪═════════╪═══════════════════╪═════════════════╡
│ 1.16  ┆ 0.72  ┆ 0.24  ┆ 1     ┆ 0.34    ┆ 0.97              ┆ 0.93            │
│ -2.16 ┆ -2.43 ┆ 0.18  ┆ 1     ┆ 0.97    ┆ -2.23             ┆ -2.18           │
│ -1.57 ┆ -0.63 ┆ -0.95 ┆ 1     ┆ 0.39    ┆ -1.54             ┆ -1.54           │
│ 0.21  ┆ 0.05  ┆ 0.23  ┆ 1     ┆ 0.8     ┆ 0.29              ┆ 0.27            │
│ 0.22  ┆ -0.07 ┆ 0.44  ┆ 1     ┆ 0.57    ┆ 0.37              ┆ 0.36            │
└───────┴───────┴───────┴───────┴─────────┴───────────────────┴─────────────────┘
```

The `mode` parameter is used to set the type of the output returned by all methods (`"predictions", "residuals", "coefficients"`).
It defaults to returning predictions matching the input's length.

In case `"coefficients"` is set the output's shape is the number of features specified, see below:

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

Supported Models
------------

Currently, this extension package supports the following variants:
- Ordinary Least Squares: ```least_squares.ols```
- Weighted Least Squares: ```least_squares.wls```
- Regularized Least Squares (Lasso / Ridge / Elastic Net) ```least_squares.{lasso, ridge, elastic_net}```
- Non-negative Least Squares: ```least_squares.nnls```

As well as efficient implementations of moving window models:
- Recursive Least Squares: ```least_squares.rls```
- Rolling / Expanding Window OLS: ```least_squares.{rolling_ols, expanding_ols}```

An arbitrary combination of sample_weights, L1/L2 penalties, and non-negativity constraints can be specified with
the ```least_squares.from_formula``` and ```least_squares.least_squares``` entry-points.

Benchmark
------------
Usual caveat of benchmarks being platform and set-up dependent apply, but the below should still be indicative of the
type of performance to expect when using this package.
