#![allow(clippy::unit_arg, clippy::unused_unit)]
use ndarray::{Array, Array1, Array2, Axis};
use polars::error::{PolarsResult, polars_err};
use polars::prelude::{NamedFromOwned, Series};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::least_squares::{solve_ols_qr, solve_ridge};


/// Convert a slice of polars series into target & feature ndarray objects.
pub fn convert_polars_to_ndarray(inputs: &[Series]) -> (Array1<f32>, Array2<f32>) {
    let m = inputs.len();
    let n = inputs[0].len();

    assert!(m > 1, "must pass at least 2 series");

    // prepare targets & features ndarrays. assume first series is targets and rest are features.
    let y = inputs[0]
        .f32()
        .expect("Failed to convert polars series to f32 array")
        .to_ndarray()
        .expect("Failed to convert f32 series to ndarray")
        .to_owned();

    // note that this was faster than converting polars series -> polars dataframe -> to_ndarray
    let mut x: Array<f32, _> = Array::zeros((n, m - 1));
    x.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(j, mut col)| {
            // Convert Series to ndarray
            let s = inputs[j + 1]
                .f32()
                .expect("Failed to convert polars series to f32 array")
                .to_ndarray()
                .expect("Failed to convert f32 series to ndarray");
            col.assign(&s);
        });
    (y, x)
}

#[derive(Deserialize)]
pub struct OLSKwargs {
    ridge_alpha: f32,
    ridge_solve_method: String,
}

/// Computes linear predictions and returns a polars series.
fn make_predictions(features: &Array2<f32>, coefficients: Array1<f32>) -> Series {
    Series::from_vec("predictions", features.dot(&coefficients).to_vec())
}

fn _get_least_squares_coefficients(
    targets: &Array1<f32>,
    features: &Array2<f32>,
    kwargs: OLSKwargs,
) -> Array1<f32> {
    assert!(
        kwargs.ridge_alpha >= 0.,
        "alpha must be strictly positive or zero"
    );
    if kwargs.ridge_alpha > 0. {
        solve_ridge(
            targets,
            features,
            kwargs.ridge_alpha,
            Some(&kwargs.ridge_solve_method),
        )
    } else {
        solve_ols_qr(targets, features)
    }
}

#[polars_expr(output_type = Float32)]
fn pl_least_squares(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let coefficients = _get_least_squares_coefficients(&y, &x, kwargs);
    Ok(make_predictions(&x, coefficients))
}

#[polars_expr(output_type = Float32)]
fn pl_least_squares_coefficients(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    Ok(Series::from_vec(
        "coefficients",
        _get_least_squares_coefficients(&y, &x, kwargs).to_vec(),
    ))
}
