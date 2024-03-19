# Polars OLS
## Least squares extension in Polars

Support for a variety of linear models (OLS, WLS, Ridge, Elastic Net, etc.) in [Polars](https://www.pola.rs/).

### Why? 

1. **Performance**: implementations are written in rust and highly optimized making use of rust linear-algebra crates & LAPACK routines.
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to other formats or multiple libraries (e.g. numpy, sklearn) to do simple linear regressions. Chain least squares like any other expression in your workflow.
3. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
4. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) and automatically converts to equivalent polars expressions.

Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/).

Then, you'll need to install `polars-ols`:
```console
pip install polars-ols
```

API
------------

Importing `polars_ols` will register the namespace `least_squares` provided by this package:
```python
import polars as pl
import polars_ols as pls

ols_expr = pl.col("y").least_squares.from_formula("x1 + x2 + x3").over("group").alias("predictions_ols")
```

Alternatively, you also access public methods provided by the package:
```python
import polars as pl
import polars_ols as pls

wls_expr = pls.pl_least_squares_from_formula("y ~ x1 + x2 + x3",
                                             sample_weights=pl.col("weights")
                                             ).over("group").alias("predictions_wls")
```

The methods provided by the `least_squares` namespace return normal polars expressions which your polars dataframe can use
```python
predictions = df.select(ols_expr, wls_expr)
```



Examples
-------------

See `/tests` for further detailed examples.
