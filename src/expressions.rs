#![allow(clippy::unit_arg, clippy::unused_unit)]
use ndarray::{s, Array, Array1, Array2, Axis};
use polars::datatypes::{DataType, Field, Float64Type};
use polars::error::{polars_err, PolarsResult};
use polars::frame::DataFrame;
use polars::prelude::{
    lit, BooleanChunked, FillNullStrategy, IndexOrder, IntoLazy, IntoSeries, NamedFrom,
    NamedFromOwned, Series, NULL,
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::str::FromStr;

use crate::least_squares::{
    solve_elastic_net, solve_multi_target, solve_ols, solve_recursive_least_squares, solve_ridge,
    solve_rolling_ols, NullPolicy, SolveMethod,
};

/// convert a slice of polars series into a 2D feature array.
fn construct_features_array(inputs: &[Series], fill_zero: bool) -> Array2<f64> {
    let m = inputs.len();
    let n = inputs[0].len();
    // Prepare features ndarray
    let mut x: Array<f64, _> = Array::zeros((n, m));
    x.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(j, mut col)| {
            if fill_zero {
                // Convert Series to ndarray
                let s = inputs[j]
                    .cast(&DataType::Float64)
                    .expect("failed to cast inputs to f64")
                    .fill_null(FillNullStrategy::Zero)
                    .unwrap();
                let s = s.rechunk();
                let s = s
                    .f64()
                    .unwrap()
                    // .expect("Failed to convert polars series to f64 array")
                    .to_ndarray()
                    .expect("Failed to convert f64 series to ndarray");
                col.assign(&s);
            } else {
                let s = inputs[j]
                    .cast(&DataType::Float64)
                    .expect("failed to cast inputs to f64");
                let s = s.rechunk();
                // Convert Series to ndarray
                let s = s
                    .f64()
                    .unwrap()
                    // .expect("Failed to convert polars series to f64 array")
                    .to_ndarray()
                    .expect("Failed to convert f64 series to ndarray");
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
) -> (Array1<f64>, Array2<f64>) {
    let m = inputs.len();
    assert!(m > 1, "must pass at least 2 series");

    // handle nulls according to the specified null policy
    let mut filtered_inputs = Vec::new();
    handle_nulls(inputs, null_policy, is_valid, &mut filtered_inputs);

    // prepare targets & features ndarrays. assume first series is targets and rest are features.
    let y = filtered_inputs[0]
        .cast(&DataType::Float64)
        .expect("Failed to cast targets series to f64");
    let y = y.rechunk();
    let y = y
        .f64()
        .expect("Failed to convert polars series to f64 array")
        .to_ndarray()
        .expect("Failed to convert f64 series to ndarray")
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
fn convert_array_to_struct_series(
    array: &Array2<f64>,
    feature_names: &[&str],
    name: Option<&str>,
) -> Series {
    // Convert 2D ndarray into DataFrame
    let df: DataFrame = DataFrame::new(
        array
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(i, col)| {
                // TODO: clean below up once https://github.com/pola-rs/pyo3-polars/issues/79
                //  is resolved
                let i_str = i.to_string();
                let name = if feature_names[i].is_empty() {
                    i_str.as_ref()
                } else {
                    feature_names[i]
                };
                // let col = col.map(|x| if x.is_nan() { None } else { Some(x) });
                Series::from_vec(name, col.to_vec())
            })
            .collect::<Vec<Series>>(),
    )
    .unwrap()
    .lazy()
    .fill_nan(lit(NULL))
    .collect()
    .unwrap();
    // Convert DataFrame to a Series of struct dtype
    df.into_struct(name.unwrap_or("coefficients")).into_series()
}

fn mask_predictions(predictions: Vec<f64>, is_valid_mask: &BooleanChunked) -> Vec<Option<f64>> {
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

enum Coefficients {
    Array1(Array1<f64>),
    Array2(Array2<f64>),
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
    features: &Array2<f64>,
    coefficients: &Coefficients,
    is_valid_mask: Option<&BooleanChunked>,
    name: &str,
) -> Series {
    // compute dot product of (zero-filled) features with coefficients
    let predictions = match coefficients {
        Coefficients::Array1(coefficients) => features.dot(coefficients).to_vec(),
        Coefficients::Array2(coefficients) => (features * coefficients).sum_axis(Axis(1)).to_vec(),
    };

    if let Some(is_valid) = is_valid_mask {
        // is_valid mask has been passed: when true retain values and otherwise mask with None.
        let masked_predictions: Vec<Option<f64>> = mask_predictions(predictions, is_valid);
        Series::new(name, &masked_predictions)
    } else {
        // no mask is provided, return predictions as-is
        Series::from_vec(name, predictions)
    }
}

fn convert_option_vec_to_array1(opt_vec: Option<Vec<f64>>) -> Option<Array1<f64>> {
    opt_vec.map(Array1::from)
}

fn compute_is_valid_mask(
    inputs: &[Series],
    null_policy: &NullPolicy,
    multi_target_index: Option<usize>,
) -> Option<BooleanChunked> {
    let multi_target_index = multi_target_index.unwrap_or(0);
    match null_policy {
        // Compute the intersection of all non-null rows across input series
        NullPolicy::Drop | NullPolicy::DropZero | NullPolicy::DropWindow => {
            let is_valid_mask = inputs[0].is_not_null();
            Some(
                inputs[1..]
                    .iter()
                    .fold(is_valid_mask, |acc, s| acc & s.is_not_null()),
            )
        }
        // Compute non-null mask based on the first input series (i.e. targets)
        NullPolicy::DropYZeroX => match multi_target_index {
            0 => Some(inputs[0].is_not_null()),
            _ => Some(
                inputs[1..multi_target_index]
                    .iter()
                    .fold(inputs[0].is_not_null(), |acc, s| acc & s.is_not_null()),
            ),
        },
        _ => None,
    }
}

fn convert_is_valid_mask_to_vec(is_valid: &Option<BooleanChunked>, n_samples: usize) -> Vec<bool> {
    if let Some(boolean_chunked) = is_valid {
        assert_eq!(
            boolean_chunked.len(),
            n_samples,
            "length of is_valid mask must match number of samples"
        );
        boolean_chunked
            .iter()
            .map(|opt_bool| opt_bool.unwrap_or(false))
            .collect()
    } else {
        vec![true; n_samples]
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
        NullPolicy::Drop | NullPolicy::DropZero | NullPolicy::DropWindow => {
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
    alpha: Option<f64>,
    l1_ratio: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    positive: Option<bool>,
    solve_method: Option<String>,
    null_policy: Option<String>,
    rcond: Option<f64>,
}

#[derive(Deserialize)]
pub struct RLSKwargs {
    half_life: Option<f64>,
    initial_state_covariance: Option<f64>,
    initial_state_mean: Option<Vec<f64>>, // in python list[f64] | None is equivalent
    null_policy: Option<String>,
}

#[derive(Deserialize)]
pub struct RollingKwargs {
    window_size: usize,
    min_periods: Option<usize>,
    use_woodbury: Option<bool>,
    alpha: Option<f64>,
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
    targets: &Array1<f64>,
    features: &Array2<f64>,
    kwargs: OLSKwargs,
) -> Array1<f64> {
    // handle degenerate case of no data
    if features.is_empty() {
        return Array1::zeros(features.len_of(Axis(1)));
    }

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
        solve_ols(targets, features, solve_method, kwargs.rcond)
    } else if alpha >= 0. && kwargs.l1_ratio.unwrap_or(0.0) == 0. && !positive {
        solve_ridge(targets, features, alpha, solve_method, kwargs.rcond)
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

#[polars_expr(output_type=Float64)]
fn least_squares(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let (y_fit, x_fit) = convert_polars_to_ndarray(inputs, &null_policy, is_valid.as_ref());
    let coefficients =
        Coefficients::Array1(_get_least_squares_coefficients(&y_fit, &x_fit, kwargs));

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
    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let (y, x) = convert_polars_to_ndarray(inputs, &null_policy, is_valid.as_ref());
    // force into 1 x K 2-d array, so that we can return a series of struct
    let coefficients = _get_least_squares_coefficients(&y, &x, kwargs).insert_axis(Axis(0));
    // convert coefficients to a polars struct
    let feature_names: Vec<&str> = inputs[1..].iter().map(|input| input.name()).collect();
    assert_eq!(
        feature_names.len(),
        coefficients.len_of(Axis(1)),
        "number of coefficients must match number of features!"
    );
    let series = convert_array_to_struct_series(&coefficients, &feature_names, None);
    Ok(series.with_name("coefficients"))
}

fn multi_target_struct_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let dtype = input_fields[0].dtype.to_owned();
    assert!(
        dtype.is_struct(),
        "the first series in a multi-target regression \
        must be of polars struct dtype with each field corresponding to an output"
    );
    Ok(Field::new("predictions", dtype))
}

#[polars_expr(output_type_func=multi_target_struct_dtype)]
fn multi_target_least_squares(inputs: &[Series], kwargs: OLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();

    // compute targets
    let targets_struct = inputs[0]
        .struct_()
        .expect("the first series in a multi-target regression must be a struct");
    let targets_df = targets_struct.to_owned().unnest();
    let target_series = targets_df.iter().as_slice();

    // concatenate multi-target and feature series
    let series = [target_series, &inputs[1..]].concat();

    // define number of target series
    let m = target_series.len();

    // compute validity mask
    let is_valid = compute_is_valid_mask(&series, &null_policy, Some(m));

    // handle nulls according to the specified null policy (this should just work for multi-target)
    let mut filtered_series = Vec::new();
    handle_nulls(
        &series,
        &null_policy,
        is_valid.as_ref(),
        &mut filtered_series,
    );

    // compute filtered features & targets array
    let features_array = construct_features_array(&filtered_series[m..], true);
    let targets_array = construct_features_array(&filtered_series[..m], true);

    // this should be K x M, where K is number of features
    let coefficients =
        solve_multi_target(&targets_array, &features_array, kwargs.alpha, kwargs.rcond);
    assert_eq!(coefficients.nrows(), features_array.ncols());
    assert_eq!(coefficients.ncols(), targets_array.ncols());

    let target_names: Vec<&str> = target_series.iter().map(|s| s.name()).collect();

    if matches!(null_policy, NullPolicy::Ignore | NullPolicy::Zero) {
        let predictions = features_array.dot(&coefficients);
        Ok(convert_array_to_struct_series(
            &predictions,
            &target_names,
            Some("predictions"),
        ))
    } else {
        // compute unfiltered features array and predictions assuming zero filled features
        let features_array_predict = construct_features_array(&inputs[1..], true);
        let mut predictions = features_array_predict.dot(&coefficients);

        // re-mask invalid predictions
        if null_policy == NullPolicy::Drop {
            if let Some(is_valid) = is_valid {
                for (i, valid) in is_valid.iter().enumerate() {
                    if !valid.unwrap_or(false) {
                        predictions.slice_mut(s![i, ..]).fill(f64::NAN);
                    }
                }
            }
        }

        Ok(convert_array_to_struct_series(
            &predictions,
            &target_names,
            Some("predictions"),
        ))
    }
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn recursive_least_squares_coefficients(
    inputs: &[Series],
    kwargs: RLSKwargs,
) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();

    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let is_valid = convert_is_valid_mask_to_vec(&is_valid, inputs[0].len());

    let (y, x) = convert_polars_to_ndarray(inputs, &NullPolicy::Zero, None);
    let initial_state_mean = convert_option_vec_to_array1(kwargs.initial_state_mean);
    let coefficients = solve_recursive_least_squares(
        &y,
        &x,
        kwargs.half_life,
        kwargs.initial_state_covariance,
        initial_state_mean,
        &is_valid,
    );
    // convert coefficients to a polars struct
    let feature_names: Vec<&str> = inputs[1..].iter().map(|input| input.name()).collect();
    assert_eq!(
        feature_names.len(),
        coefficients.len_of(Axis(1)),
        "number of coefficients must match number of features!"
    );
    let series = convert_array_to_struct_series(&coefficients, &feature_names, None);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float64)]
fn recursive_least_squares(inputs: &[Series], kwargs: RLSKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let is_valid_vec = convert_is_valid_mask_to_vec(&is_valid, inputs[0].len());
    let (y, x) = convert_polars_to_ndarray(inputs, &NullPolicy::Zero, None);

    let coefficients = Coefficients::Array2(solve_recursive_least_squares(
        &y,
        &x,
        kwargs.half_life,
        kwargs.initial_state_covariance,
        None,
        &is_valid_vec,
    ));

    Ok(make_predictions(
        &x,
        &coefficients,
        is_valid.as_ref(),
        inputs[0].name(),
    ))
}

#[polars_expr(output_type_func=coefficients_struct_dtype)]
fn rolling_least_squares_coefficients(
    inputs: &[Series],
    kwargs: RollingKwargs,
) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let is_valid = convert_is_valid_mask_to_vec(&is_valid, inputs[0].len());
    let (y, x) = convert_polars_to_ndarray(inputs, &NullPolicy::Zero, None);
    let coefficients = solve_rolling_ols(
        &y,
        &x,
        kwargs.window_size,
        kwargs.min_periods,
        kwargs.use_woodbury,
        kwargs.alpha,
        &is_valid,
        null_policy,
    );
    // convert coefficients to a polars struct
    let feature_names: Vec<&str> = inputs[1..].iter().map(|input| input.name()).collect();
    assert_eq!(
        feature_names.len(),
        coefficients.len_of(Axis(1)),
        "number of coefficients must match number of features!"
    );
    let series = convert_array_to_struct_series(&coefficients, &feature_names, None);
    Ok(series.with_name("coefficients"))
}

#[polars_expr(output_type=Float64)]
fn rolling_least_squares(inputs: &[Series], kwargs: RollingKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    let is_valid = compute_is_valid_mask(inputs, &null_policy, None);
    let is_valid_vec = convert_is_valid_mask_to_vec(&is_valid, inputs[0].len());
    let (y, x) = convert_polars_to_ndarray(inputs, &NullPolicy::Zero, None);
    let coefficients = Coefficients::Array2(solve_rolling_ols(
        &y,
        &x,
        kwargs.window_size,
        kwargs.min_periods,
        kwargs.use_woodbury,
        kwargs.alpha,
        &is_valid_vec,
        null_policy,
    ));

    Ok(make_predictions(
        &x,
        &coefficients,
        is_valid.as_ref(),
        inputs[0].name(),
    ))
}

/// This function provides a convenience expression to multiply fitted coefficients with features,
/// which may be particularly useful in case predicting on test data
/// (otherwise use direct prediction functions).
#[polars_expr(output_type=Float64)]
fn predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let null_policy = kwargs.get_null_policy();
    // The first input is always assumed to be the coefficient struct, and the remaining
    // input series are assumed to be an equivalent number of features.
    let coefficients_df: DataFrame = inputs[0]
        .struct_()
        .expect("the first input series to predict function must be of dtype struct!")
        .clone()
        .unnest();

    assert_eq!(
        coefficients_df.shape().1,
        inputs[1..].len(),
        "number of coefficients must match number of features!"
    );

    // compute predictions assuming zero filled features
    let features = construct_features_array(&inputs[1..], null_policy != NullPolicy::Ignore);
    let coefficients: Array2<f64> = coefficients_df
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let predictions = (&features * &coefficients).sum_axis(Axis(1)).to_vec();

    if null_policy == NullPolicy::Drop {
        // If user has opted for "Drop" policy: mask predictions
        let is_valid = compute_is_valid_mask(inputs, &null_policy, None).unwrap();
        Ok(Series::new(
            inputs[0].name(),
            mask_predictions(predictions, &is_valid),
        ))
    } else {
        // Otherwise, simply return predictions
        Ok(Series::from_vec(inputs[0].name(), predictions))
    }
}
