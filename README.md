# Polars OLS
## Least squares extension in Polars

Support for a variety of linear models (OLS, WLS, Ridge, etc.) in [Polars](https://www.pola.rs/).

### Why? 

1. **Performance**: implementations are written in rust and highly optimized making use of rust linear-algebra crates & LAPACK
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to other formats or libraries (e.g. numpy, sklearn) to do simple linear regressions. Chain least squares like any other expression in your workflow.
3. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
4. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) and automatically converts to equivalent polars expressions.

Installation
------------


First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/).

Then, you'll need to install `polars-ols`:
```console
pip install polars-ols
```

Examples
-------------

See `/tests` for further detailed examples.
