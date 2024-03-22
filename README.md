# Polars OLS
## Least squares extension in Polars

Support for a variety of linear models (OLS, WLS, Ridge, Elastic Net, etc.) in [Polars](https://www.pola.rs/).

### Why?

1. **High Performance**: implementations are written in rust and make use of high performance rust linear-algebra crates & LAPACK routines.
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to other formats or multiple libraries (e.g. numpy, sklearn) to do simple linear regressions. Chain least squares like any other expression in your workflow.
3. **Efficient Implementation**:
   - Numerically stable algorithms are chosen where possible (e.g. QR, Cholesky)
   - Flexible model specification allows arbitrary combination of sample weighting, L1/L2 regularization, & non-negativity constraints on parameters
   - Efficient rank-1 update algorithms used for moving window regressions
4. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
5. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) which automatically converts to equivalent polars expressions.

Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/).

Then, you'll need to install `polars-ols`:
```console
pip install polars-ols
```

API
------------

Importing `polars_ols` will register the namespace `least_squares` provided by this package.
You can build models either by specifying columns for target and features or using the formula api (patsy syntax).
```python
import polars as pl
import polars_ols as pls

ols_expr = pl.col("y").least_squares.from_formula("x1 + x2 + x3").over("group").alias("predictions_ols")
lasso_expr = pl.col("y").least_squares.lasso(pl.col("x1"), pl.col("x2"), pl.col("x3"), alpha=0.01).alias("predictions_lasso")
```

Alternatively, you also access public methods provided by the package as well:

```python
import polars as pl
import polars_ols as pls

wls_expr = pls.least_squares_from_formula("y ~ x1 + x2 + x3",
                                          sample_weights=pl.col("weights"),
                                          alpha=0.0001,
                                          l1_weight=0.5,
                                          ).over("group").alias("predictions_wls")
```

The methods provided by the `least_squares` namespace return polars expressions which your polars dataframe
can use normally:

```python
predictions = df.select(ols_expr, wls_expr, lasso_expr)
```

A `mode` parameter is used to set the type of the output returned by all methods (`"predictions", "residuals", "coefficients"`).
Defaults to returning predictions.

Currently, this extension package supports:
- Ordinary Least Squares: ```least_squares.ols```
- Weighted Least Squares: ```least_squares.wls```
- Regularized Least Squares (Lasso / Ridge / Elastic Net) ```least_squares.{lasso, ridge, elastic_net}```
- Non-negative Least Squares: ```least_squares.nnls```
- Formula API: ```least_squares.from_formula```

Notice that an arbitrary combination of sample_weights, L1/L2 penalties, and non-negativity constraint can be used with
both the ```least_squares.from_formula``` and ```least_squares.least_squares``` methods.

Examples
-------------

See `/tests` for further detailed examples.
