#![allow(clippy::unit_arg, clippy::unused_unit)]

use ndarray::{Array, Array1, Array2, Axis};
use polars::datatypes::{DataType, Field, Float32Type};
use polars::error::{polars_err, PolarsResult};
use polars::frame::DataFrame;
use polars::prelude::{IndexOrder, IntoSeries, NamedFromOwned, Series};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::least_squares::{
    solve_elastic_net, solve_ols_qr, solve_recursive_least_squares, solve_ridge, solve_rolling_ols,
};

/// convert a slice of polars series into a 2D feature array.
fn construct_features_array(inputs: &[Series]) -> Array2<f32> {
    let m = inputs.len();
    let n = inputs[0].len();
    // Prepare features ndarray
    let mut x: Array<f32, _> = Array::zeros((n, m));
    x.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(j, mut col)| {
            // Convert Series to ndarray
            let s = inputs[j]
                .f32()
                .expect("Failed to convert polars series to f32 array")
                .to_ndarray()
                .expect("Failed to convert f32 series to ndarray");
            col.assign(&s);
        });
    x
}

/// Convert a slice of polars series into target & feature ndarray objects.
pub fn convert_polars_to_ndarray(inputs: &[Series]) -> (Array1<f32>, Array2<f32>) {
    let m = inputs.len();
    assert!(m > 1, "must pass at least 2 series");

    // prepare targets & features ndarrays. assume first series is targets and rest are features.
    let y = inputs[0]
        .f32()
        .expect("Failed to convert polars series to f32 array")
        .to_ndarray()
        .expect("Failed to convert f32 series to ndarray")
        .to_owned();

    // note that this was faster than converting polars series -> polars dataframe -> to_ndarray
    // assume first series is targets and rest are features.
    let x = construct_features_array(&inputs[1..]);
    assert_eq!(
        x.len_of(Axis(0)),
        y.len(),
        "all input series passed must be of equal length"
    );

    (y, x)
}

fn coefficients_struct_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    // the first input field denotes the target, which we need not carry in output struct
    Ok(Field::new(
        "coefficients",
        DataType::Struct(input_fields[1..].to_vec()),
    ))
}

/// Convert the coefficients into a Polars series of struct dtype.
fn coefficients_to_struct_series(coefficients: &Array2<f32>) -> Series {
    // Convert 2D ndarray into DataFrame
    let df: DataFrame = DataFrame::new(
        coefficients
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(i, col)| Series::from_vec(&i.to_string(), col.to_vec()))
            .collect::<Vec<Series>>(),
    )
    .unwrap();
    // Convert DataFrame to a Series of struct dtype
    df.into_struct("coefficients").into_series()
}

// fn list_float_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = Field::new(
//         input_fields[0].name(),
//         DataType::List(Box::new(DataType::Float32)),
//     );
//     Ok(field.clone())
// }
// fn coefficients_to_series_list(coefficients: &Array2<f32>) -> Series {
//     // convert 2d ndarray into Series of List[f32]
//     let mut chunked_builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
//         "",
//         coefficients.len_of(Axis(0)),
//         coefficients.len_of(Axis(1)),
//         DataType::Float32,
//     );
//     for row in coefficients.axis_iter(Axis(0)) {
//         match row.as_slice() {
//             Some(row) => chunked_builder.append_slice(row),
//             None => chunked_builder.append_slice(&row.to_vec()),
//         }
//     }
//     chunked_builder.finish().into_series()
// }

/// Computes linear predictions and returns a polars series.
fn make_predictions(features: &Array2<f32>, coefficients: Array1<f32>, name: &str) -> Series {
    Series::from_vec(name, features.dot(&coefficients).to_vec())
}

fn convert_option_vec_to_array1(opt_vec: Option<Vec<f32>>) -> Option<Array1<f32>> {
    opt_vec.map(Array1::from)
}

#[derive(Deserialize)]
pub struct OLSKwargs {
    alpha: Option<f32>,
    l1_ratio: Option<f32>,
    max_iter: Option<usize>,
    tol: Option<f32>,
    positive: Option<bool>,
}

#[derive(Deserialize)]
pub struct RLSKwargs {
    half_life: Option<f32>,
    initial_state_covariance: Option<f32>,
    initial_state_mean: Option<Vec<f32>>, // in python list[f32] | None is equivalent
}

#[derive(Deserialize)]
pub struct RollingKwargs {
    window_size: usize,
    min_periods: Option<usize>,
    use_woodbury: Option<bool>,
    alpha: Option<f32>,
}

fn _get_least_squares_coefficients(
    targets: &Array1<f32>,
    features: &Array2<f32>,
    kwargs: OLSKwargs,
) -> Array1<f32> {
    let alpha = kwargs.alpha.unwrap_or(0.0);
    let positive = kwargs.positive.unwrap_or(false);
    if alpha == 0. && !positive {
        solve_ols_qr(targets, features)
    } else if alpha > 0. && kwargs.l1_ratio.unwrap_or(0.0) == 0. && !positive {
        solve_ridge(targets, features, alpha)
    } else {
        solve_elastic_net(
            targets,
            features,
            alpha,
            kwargs.l1_ratio,
            kwargs.max_iter,
            kwargs.tol,
            kwargs.positive,
        )
    }
}

#[polars_expr(output_type=Float32)]
fn least_squares(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let coefficients = _get_least_squares_coefficients(&y, &x, kwargs);
    Ok(make_predictions(&x, coefficients, inputs[0].name()))
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn least_squares_coefficients(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    // force into 1 x K 2-d array, so that we can return a series of struct
    let coefficients = _get_least_squares_coefficients(&y, &x, kwargs).insert_axis(Axis(0));
    // let series = coefficients_to_series_list(&coefficients);
    let series = coefficients_to_struct_series(&coefficients);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn recursive_least_squares_coefficients(
    inputs: &[Series],
    kwargs: RLSKwargs,
) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let initial_state_mean = convert_option_vec_to_array1(kwargs.initial_state_mean);
    let coefficients = solve_recursive_least_squares(
        &y,
        &x,
        kwargs.half_life,
        kwargs.initial_state_covariance,
        initial_state_mean,
    );
    // let series = coefficients_to_series_list(&coefficients);
    let series = coefficients_to_struct_series(&coefficients);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float32)]
fn recursive_least_squares(inputs: &[Series], kwargs: RLSKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let coefficients = solve_recursive_least_squares(
        &y,
        &x,
        kwargs.half_life,
        kwargs.initial_state_covariance,
        None,
    );
    let predictions = (&x * &coefficients).sum_axis(Axis(1));
    Ok(Series::from_vec(inputs[0].name(), predictions.to_vec()))
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn rolling_least_squares_coefficients(
    inputs: &[Series],
    kwargs: RollingKwargs,
) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let coefficients = solve_rolling_ols(
        &y,
        &x,
        kwargs.window_size,
        kwargs.min_periods,
        kwargs.use_woodbury,
        kwargs.alpha,
    );
    // let series = coefficients_to_series_list(&coefficients);
    let series = coefficients_to_struct_series(&coefficients);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float32)]
fn rolling_least_squares(inputs: &[Series], kwargs: RollingKwargs) -> PolarsResult<Series> {
    let (y, x) = convert_polars_to_ndarray(inputs);
    let coefficients = solve_rolling_ols(
        &y,
        &x,
        kwargs.window_size,
        kwargs.min_periods,
        kwargs.use_woodbury,
        kwargs.alpha,
    );
    let predictions = (&x * &coefficients).sum_axis(Axis(1));
    Ok(Series::from_vec(inputs[0].name(), predictions.to_vec()))
}

/// This function provides a convenience expression to multiply fitted coefficients with features,
/// which may be particularly useful in case predicting on test data
/// (otherwise use direct prediction functions).
#[polars_expr(output_type=Float32)]
fn predict(inputs: &[Series]) -> PolarsResult<Series> {
    // The first input is always assumed to be the coefficient struct, and the remaining
    // input series are assumed to be an equivalent number of features.
    let coefficients_df: DataFrame = inputs[0]
        .struct_()
        .expect("the first input series to predict function must be of dtype struct!")
        .clone()
        .unnest();
    let features = construct_features_array(&inputs[1..]);
    let coefficients: Array2<f32> = coefficients_df
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let predictions = (&features * &coefficients).sum_axis(Axis(1));
    Ok(Series::from_vec(inputs[0].name(), predictions.to_vec()))
}
