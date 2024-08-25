use faer::solvers::SolverCore;
use faer::Side::Lower;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{Array1, Array2, ArrayView2};
use statrs::distribution::{ContinuousCDF, StudentsT};

pub struct ResidualMetrics {
    pub mse: f64,
    pub mae: f64,
    pub r2: f64,
}

/// Computes the Mean Squared Error (MSE), Mean Absolute Error (MAE), and
/// R-squared (R2) coefficient of determination between actual and predicted values.
pub fn compute_residual_metrics(targets: &Array1<f64>, predicted: &Array1<f64>) -> ResidualMetrics {
    let mean = targets.mean().unwrap_or(0.0);
    let mut sum_sq_error = 0.0;
    let mut sum_abs_error = 0.0;
    let mut sum_sq_total = 0.0;

    for i in 0..targets.len() {
        let actual = targets[i];
        let pred = predicted[i];
        let error = actual - pred;
        sum_sq_error += error.powi(2);
        sum_abs_error += error.abs();
        sum_sq_total += (actual - mean).powi(2);
    }

    let mse = sum_sq_error / targets.len() as f64;
    let mae = sum_abs_error / targets.len() as f64;
    let r2 = 1.0 - (sum_sq_error / sum_sq_total);

    ResidualMetrics { mse, mae, r2 }
}

/// Struct to hold standard errors and t-values for regression coefficients.
pub struct FeatureMetrics {
    pub standard_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
}

fn t_value_to_p_value(t_value: f64, df: f64) -> f64 {
    let t_dist =
        StudentsT::new(0.0, 1.0, df).expect("Invalid parameters for StudentT distribution");
    2.0 * (1.0 - t_dist.cdf(t_value.abs()))
}

// Function to compute the trace of a square matrix
fn trace(matrix: ArrayView2<f64>) -> f64 {
    // Ensure the matrix is square
    assert_eq!(matrix.nrows(), matrix.ncols(), "Matrix must be square!");

    // Sum the diagonal elements
    let mut trace_sum = 0.0;
    for i in 0..matrix.nrows() {
        trace_sum += matrix[(i, i)];
    }

    trace_sum
}

/// Computes standard errors and t-values for regression coefficients using Ridge Regression.
///
/// # Arguments
/// - `features`: Feature matrix (X) of shape (n_samples, n_features).
/// - `targets`: Target vector (y) of length n_samples.
/// - `lambda`: Regularization parameter for Ridge Regression.
///
/// # Returns
/// - `FeatureMetrics` containing standard errors and t-values.
///
/// # Errors
/// Returns an error if matrix inversion fails or degrees of freedom are invalid.
pub fn compute_feature_metrics(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    lambda: f64,
) -> FeatureMetrics {
    let n = features.nrows() as f64;
    let p = features.ncols() as f64;

    // Compute X^T X
    let xtx = features.t().dot(features);

    // Add λI for regularization
    let identity: Array2<f64> = Array2::eye(xtx.nrows());
    let xtx_reg = &xtx + &identity.mapv(|v| v * lambda);

    // Convert xtx_reg to a faer matrix
    let xtx_faer = xtx_reg.view().into_faer();

    // Invert (X^T X + λI) using faer
    let xtx_inv_faer = xtx_faer
        .cholesky(Lower)
        .expect("could not compute cholesky")
        .inverse();
    let xtx_inv = xtx_inv_faer.as_ref().into_ndarray();

    // Compute X^T y
    let xty = features.t().dot(targets);

    // Compute coefficients: β = (X^T X + λI)^(-1) X^T y
    let coefficients = xtx_inv.dot(&xty);

    // Compute predictions and residuals
    let predictions = features.dot(&coefficients);
    let residuals = targets - &predictions;

    // Estimate variance: σ^2 = RSS / (n - p + k)
    let rss = residuals.mapv(|r| r.powi(2)).sum();
    let df = if lambda > 0.0 {
        n - trace(xtx_inv) // .trace().unwrap_or(0.0)
    } else {
        n - p
    };

    assert!(
        df > 0.0,
        "Degrees of freedom <= 0. Cannot compute standard errors."
    );
    let sigma_squared = rss / df;

    // Standard Errors: sqrt(diag(σ^2 * (X^T X + λI)^(-1)))
    let standard_errors = xtx_inv.diag().mapv(|v| (sigma_squared * v.abs()).sqrt());

    // T-values: coefficients / standard errors
    let t_values = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&coef, &se)| coef / se)
        .collect::<Array1<f64>>();

    let p_values: Array1<f64> = t_values
        .iter()
        .map(|&t| t_value_to_p_value(t, df))
        .collect();

    FeatureMetrics {
        standard_errors,
        t_values,
        p_values,
    }
}
