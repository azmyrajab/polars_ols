use ndarray::{Array1, Array2};
use faer::prelude::*;
use faer::Side;
use faer_ext::{IntoFaer, IntoNdarray};


/// Computes the Mean Squared Error (MSE) between actual and predicted values.
/// MSE is the average of the squared differences between actual and predicted values.
/// Formula: MSE = Σ(actual - predicted)^2 / n
fn compute_mse(targets: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..targets.len() {
        sum += (targets[i] - predicted[i]).powi(2);
    }
    sum / targets.len() as f64
}

/// Computes the Mean Absolute Error (MAE) between actual and predicted values.
/// MAE is the average of the absolute differences between actual and predicted values.
/// Formula: MAE = Σ|actual - predicted| / n
fn compute_mae(targets: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..targets.len() {
        sum += (targets[i] - predicted[i]).abs();
    }
    sum / targets.len() as f64
}

/// Computes the R-squared (R2) coefficient of determination between actual and predicted values.
/// Formula: R2 = 1 - (Σ(actual - predicted)^2 / Σ(actual - mean(actual))^2)
fn compute_r2(targets: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mean = targets.mean().unwrap_or(0.);
    let ss_total = targets.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    let ss_residual = targets
        .iter()
        .zip(predicted.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>();
    1.0 - (ss_residual / ss_total)
}

fn compute_standard_errors(residuals: &Array1<f64>, features: &Array2<f64>) -> Array1<f64> {

    let ssr = residuals.iter().map(|&r| r * r).sum::<f64>();

    let n = residuals.len() as f64;
    let df = n - features.ncols() as f64;
    let mse = ssr / df;  // estimate error variance

    // compute the covariance matrix of predictors
    let covariance_matrix = features.t().dot(features) / n;

    let covariance_matrix = covariance_matrix.view().into_faer();

    let cholesky = covariance_matrix.cholesky(Side::Lower);
    let variance_covariance_matrix = covariance_matrix * mse;

    // Extract standard errors from the diagonal of the variance-covariance matrix
    let standard_errors = variance_covariance_matrix.diag().map(|&x| x.sqrt());

    standard_errors
}
