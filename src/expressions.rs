#![allow(clippy::unit_arg, clippy::unused_unit)]

use ndarray::{Array, Array1, Array2, Axis};
use polars::datatypes::{DataType, Field, Float32Type};
use polars::error::{polars_err, PolarsResult};
use polars::frame::DataFrame;
use polars::prelude::{
    BooleanChunked, FillNullStrategy, IndexOrder, IntoSeries, NamedFrom, NamedFromOwned, Series,
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::str::FromStr;

use crate::least_squares::{
    solve_elastic_net, solve_ols, solve_recursive_least_squares, solve_ridge, solve_rolling_ols,
    SolveMethod,
};

/// convert a slice of polars series into a 2D feature array.
fn construct_features_array(inputs: &[Series], fill_zero: bool) -> Array2<f32> {
    let m = inputs.len();
    let n = inputs[0].len();
    // Prepare features ndarray
    let mut x: Array<f32, _> = Array::zeros((n, m));
    x.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(j, mut col)| {
            if fill_zero {
                // Convert Series to ndarray
                let filled = inputs[j].fill_null(FillNullStrategy::Zero).unwrap();
                let s = filled
                    .f32()
                    .expect("Failed to convert polars series to f32 array")
                    .to_ndarray()
                    .expect("Failed to convert f32 series to ndarray");
                col.assign(&s);
            } else {
                // Convert Series to ndarray
                let s = inputs[j]
                    .f32()
                    .expect("Failed to convert polars series to f32 array")
                    .to_ndarray()
                    .expect("Failed to convert f32 series to ndarray");
                col.assign(&s);
            }
        });
    x
}

/// Convert a slice of polars series into target & feature ndarray objects.
pub fn convert_polars_to_ndarray(
    inputs: &[Series],
    null_policy: &NullPolicy,
    is_valid: Option<&BooleanChunked>,
) -> (Array1<f32>, Array2<f32>) {
    let m = inputs.len();
    assert!(m > 1, "must pass at least 2 series");

    // handle nulls according to the specified null policy
    let mut filtered_inputs = Vec::new();
    handle_nulls(inputs, null_policy, is_valid, &mut filtered_inputs);

    // prepare targets & features ndarrays. assume first series is targets and rest are features.
    let y = filtered_inputs[0]
        .f32()
        .expect("Failed to convert polars series to f32 array")
        .to_ndarray()
        .expect("Failed to convert f32 series to ndarray")
        .to_owned();

    // note that this was faster than converting polars series -> polars dataframe -> to_ndarray
    // assume first series is targets and rest are features.
    let x = construct_features_array(&filtered_inputs[1..], false);
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

fn mask_predictions(predictions: Vec<f32>, is_valid_mask: &BooleanChunked) -> Vec<Option<f32>> {
    is_valid_mask
        .iter()
        .zip(&predictions)
        // apply validity mask to predictions
        .map(|(valid, &prediction)| {
            if valid.unwrap_or(false) {
                Some(prediction)
            } else {
                None
            }
        })
        .collect()
}

/// Computes linear predictions and returns a Polars series.
///
/// # Arguments
/// * `features` - A 2D array representing the features.
/// * `coefficients` - A 1D array representing the coefficients.
/// * `is_valid_mask` - An optional BooleanChunked series representing the validity mask.
/// * `name` - A string specifying the name of the resulting series.
///
/// # Returns
/// A Polars Series containing the computed predictions.
fn make_predictions(
    features: &Array2<f32>,
    coefficients: &Array1<f32>,
    is_valid_mask: Option<&BooleanChunked>,
    name: &str,
) -> Series {
    // compute dot product of (zero-filled) features with coefficients
    let predictions = features.dot(coefficients).to_vec();
    if let Some(is_valid) = is_valid_mask {
        // is_valid mask has been passed: when true retain values and otherwise mask with None.
        let masked_predictions: Vec<Option<f32>> = mask_predictions(predictions, is_valid);
        Series::new(name, &masked_predictions)
    } else {
        // no mask is provided, return predictions as-is
        Series::from_vec(name, predictions)
    }
}

fn convert_option_vec_to_array1(opt_vec: Option<Vec<f32>>) -> Option<Array1<f32>> {
    opt_vec.map(Array1::from)
}

#[derive(Debug, PartialEq)]
pub enum NullPolicy {
    Zero,
    Drop,
    Ignore,
    DropZero,
    DropYZeroX,
}

impl FromStr for NullPolicy {
    type Err = ();

    fn from_str(input: &str) -> Result<NullPolicy, Self::Err> {
        match input {
            "zero" => Ok(NullPolicy::Zero),
            "drop" => Ok(NullPolicy::Drop),
            "ignore" => Ok(NullPolicy::Ignore),
            "drop_y_zero_x" => Ok(NullPolicy::DropYZeroX),
            "drop_zero" => Ok(NullPolicy::DropZero),
            _ => Err(()),
        }
    }
}

fn compute_is_valid_mask(inputs: &[Series], null_policy: &NullPolicy) -> Option<BooleanChunked> {
    match null_policy {
        // Compute the intersection of all non-null rows across input series
        NullPolicy::Drop | NullPolicy::DropZero => {
            let is_valid_mask = inputs[0].is_not_null();
            Some(
                inputs[1..]
                    .iter()
                    .fold(is_valid_mask, |acc, s| acc & s.is_not_null()),
            )
        }
        // Compute non-null mask based on the first input series (i.e. targets)
        NullPolicy::DropYZeroX => Some(inputs[0].is_not_null()),
        _ => None,
    }
}

/// Handles null values in the input series based on the specified null policy.
///
/// # Arguments
///
/// * `inputs` - A slice of input series to be processed.
/// * `null_policy` - The null handling policy to be applied.
/// * `is_valid_mask` - A boolean array which specifies, based on the chosen null policy,
///                     which row samples are valid.
/// * `outputs` - A mutable reference to a vector of series where null values have been handled
///               according to the specified policy. If no null handling is required
///               (NullPolicy::Ignore), `outputs` will contain a reference to the original `inputs`
fn handle_nulls(
    inputs: &[Series],
    null_policy: &NullPolicy,
    is_valid_mask: Option<&BooleanChunked>,
    outputs: &mut Vec<Series>,
) {
    match null_policy {
        NullPolicy::Zero => {
            // Zero out any nulls across all input series
            outputs.extend(
                inputs
                    .iter()
                    .map(|s| s.fill_null(FillNullStrategy::Zero).unwrap()),
            );
        }
        NullPolicy::DropYZeroX => {
            // Compute non-null mask based on the first input series (i.e. targets)
            let is_valid_mask = is_valid_mask.unwrap();
            // Apply mask to all series, then additionally fill any remaining nulls with zero
            outputs.extend(inputs.iter().map(|s| {
                s.filter(is_valid_mask)
                    .expect("Failed to filter input series with targets not-null mask!")
                    .fill_null(FillNullStrategy::Zero)
                    .unwrap()
            }));
        }
        NullPolicy::Drop | NullPolicy::DropZero => {
            // Compute the intersection of all non-null rows across input series
            let is_valid_mask = is_valid_mask.unwrap();
            // Apply mask to all input series
            outputs.extend(inputs.iter().map(|s| {
                s.filter(is_valid_mask)
                    .expect("Failed to filter input series with common not-null mask!")
            }));
        }
        // For `Ignore`, simply assign inputs to outputs
        // this approach of working with references should avoid copying unnecessarily
        NullPolicy::Ignore => outputs.extend_from_slice(inputs),
    }
}

#[derive(Deserialize)]
pub struct OLSKwargs {
    alpha: Option<f32>,
    l1_ratio: Option<f32>,
    max_iter: Option<usize>,
    tol: Option<f32>,
    positive: Option<bool>,
    solve_method: Option<String>,
    null_policy: Option<String>,
}

#[derive(Deserialize)]
pub struct RLSKwargs {
    half_life: Option<f32>,
    initial_state_covariance: Option<f32>,
    initial_state_mean: Option<Vec<f32>>, // in python list[f32] | None is equivalent
    null_policy: Option<String>,
}

#[derive(Deserialize)]
pub struct RollingKwargs {
    window_size: usize,
    min_periods: Option<usize>,
    use_woodbury: Option<bool>,
    alpha: Option<f32>,
    null_policy: Option<String>,
}

#[derive(Deserialize)]
pub struct PredictKwargs {
    null_policy: Option<String>,
}

pub trait HasNullPolicy {
    fn get_null_policy(&self) -> NullPolicy;
}

macro_rules! impl_has_null_policy {
    ($($struct:ident),*) => {
        $(
            impl HasNullPolicy for $struct {
                fn get_null_policy(&self) -> NullPolicy {
                    self.null_policy.as_ref().map(|s| NullPolicy::from_str(s.as_str())
                    .expect("Invalid null_policy detected!")).unwrap_or(NullPolicy::Ignore)
                }
            }
        )*
    };
}

impl_has_null_policy!(OLSKwargs, RLSKwargs, RollingKwargs, PredictKwargs);

fn _get_least_squares_coefficients(
    targets: &Array1<f32>,
    features: &Array2<f32>,
    kwargs: OLSKwargs,
) -> Array1<f32> {
    let alpha = kwargs.alpha.unwrap_or(0.0);
    let positive = kwargs.positive.unwrap_or(false);
    let solve_method = kwargs
        .solve_method
        .map(|s| SolveMethod::from_str(s.as_str()).expect("invalid solve_method detected!"));
    if alpha == 0.
        && !positive
        && matches!(
            solve_method,
            None | Some(SolveMethod::SVD) | Some(SolveMethod::QR)
        )
    {
        solve_ols(targets, features, solve_method)
    } else if alpha >= 0. && kwargs.l1_ratio.unwrap_or(0.0) == 0. && !positive {
        solve_ridge(targets, features, alpha, solve_method)
    } else {
        solve_elastic_net(
            targets,
            features,
            alpha,
            kwargs.l1_ratio,
            kwargs.max_iter,
            kwargs.tol,
            kwargs.positive,
            solve_method,
        )
    }
}

#[polars_expr(output_type=Float32)]
fn least_squares(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy);
    let (y_fit, x_fit) = convert_polars_to_ndarray(inputs, &null_policy, is_valid.as_ref());
    let coefficients = _get_least_squares_coefficients(&y_fit, &x_fit, kwargs);

    if matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero) {
        // absent additional filtering: features for fitting is the same as for prediction
        Ok(make_predictions(
            &x_fit,
            &coefficients,
            is_valid.as_ref(),
            inputs[0].name(),
        ))
    } else {
        // ensure that predictions broadcast to the same shape as original inputs (don't drop rows)
        let x_predict = construct_features_array(&inputs[1..], true);
        if null_policy == NullPolicy::Drop {
            // if null policy is drop: mask invalid rows with is_valid BooleanChunked
            Ok(make_predictions(
                &x_predict,
                &coefficients,
                is_valid.as_ref(),
                inputs[0].name(),
            ))
        } else {
            // Otherwise always produce valid predictions as dot product of zero-filled features w/
            // estimated coefficients.
            Ok(make_predictions(
                &x_predict,
                &coefficients,
                None,
                inputs[0].name(),
            ))
        }
    }
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn least_squares_coefficients(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy);
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, is_valid.as_ref());
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
    let null_policy = kwargs.get_null_policy();
    assert!(
        matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero),
        "null policies which drop rows are not yet supported for RLS"
    );
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, None);
    let initial_state_mean = convert_option_vec_to_array1(kwargs.initial_state_mean);
    let coefficients = solve_recursive_least_squares(
        &y,
        &x,
        kwargs.half_life,
        kwargs.initial_state_covariance,
        initial_state_mean,
    );
    let series = coefficients_to_struct_series(&coefficients);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float32)]
fn recursive_least_squares(inputs: &[Series], kwargs: RLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    assert!(
        matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero),
        "null policies which drop rows are not yet supported for RLS"
    );
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, None);
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
    let null_policy = kwargs.get_null_policy();
    assert!(
        matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero),
        "null policies which drop rows are not yet supported for rolling least squares"
    );
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, None);
    let coefficients = solve_rolling_ols(
        &y,
        &x,
        kwargs.window_size,
        kwargs.min_periods,
        kwargs.use_woodbury,
        kwargs.alpha,
    );
    let series = coefficients_to_struct_series(&coefficients);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float32)]
fn rolling_least_squares(inputs: &[Series], kwargs: RollingKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    assert!(
        matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero),
        "null policies which drop rows are not yet supported for rolling least Squares"
    );
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, None);
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
fn predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    // The first input is always assumed to be the coefficient struct, and the remaining
    // input series are assumed to be an equivalent number of features.
    let coefficients_df: DataFrame = inputs[0]
        .struct_()
        .expect("the first input series to predict function must be of dtype struct!")
        .clone()
        .unnest();
    // compute predictions assuming zero filled features
    let features = construct_features_array(&inputs[1..], null_policy != NullPolicy::Ignore);
    let coefficients: Array2<f32> = coefficients_df
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let predictions = (&features * &coefficients).sum_axis(Axis(1)).to_vec();

    if null_policy == NullPolicy::Drop {
        // If user has opted for "Drop" policy: mask predictions
        let is_valid = compute_is_valid_mask(inputs, &null_policy).unwrap();
        Ok(Series::new(
            inputs[0].name(),
            mask_predictions(predictions, &is_valid),
        ))
    } else {
        // Otherwise, simply return predictions
        Ok(Series::from_vec(inputs[0].name(), predictions))
    }
}
