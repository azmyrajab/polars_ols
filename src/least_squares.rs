use faer::prelude::SpSolverLstsq;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{s, Array, Array1, Array2, NewAxis};
use ndarray_linalg::{Norm, SolveC};


/// Solves an ordinary least squares problem using QR using faer
/// Inputs: features (2d ndarray), targets (1d ndarray)
/// Outputs: 1-d OLS coefficients
pub fn solve_ols_qr(y: &Array1<f32>, x: &Array2<f32>) -> Array1<f32> {
    // compute least squares solution via QR
    let x_faer = x.view().into_faer();
    let y_faer = y.slice(s![.., NewAxis]).into_faer();
    let coefficients = x_faer.qr().solve_lstsq(&y_faer);
    coefficients
        .as_ref()
        .into_ndarray()
        .slice(s![.., 0])
        .to_owned()
}

/// Solves a ridge regression problem of the form: ||y - x B|| + alpha * ||B||
/// Inputs: features (2d ndarray), targets (1d ndarray), ridge alpha scalar
pub fn solve_ridge(
    y: &Array1<f32>,
    x: &Array2<f32>,
    alpha: f32,
) -> Array1<f32> {
    assert!(alpha > 0., "alpha must be strictly positive");
    let x_t = &x.t();
    let x_t_x = x_t.dot(x);
    let x_t_y = x_t.dot(y);
    let eye = Array::eye(x_t_x.shape()[0]);
    let ridge_matrix = &x_t_x + &eye * alpha;
    {
        ridge_matrix
                .solvec(&x_t_y)
                .expect("failed to solve normal equations (cholesky)")
    }
}

fn soft_threshold(x: &f32, alpha: f32) -> f32 {
    x.signum() * (x.abs() - alpha).max(0.0)
}

pub fn solve_elastic_net(
    y: &Array1<f32>,
    x: &Array2<f32>,
    alpha: f32,
    l1_ratio: Option<f32>,
    max_iter: Option<usize>,
    tol: Option<f32>,
) -> Array1<f32> {
    let l1_ratio = l1_ratio.unwrap_or(0.5);
    let max_iter = max_iter.unwrap_or(1_000);
    let tol = tol.unwrap_or(0.0001);
    assert!(l1_ratio >= 0. && l1_ratio <= 1., "l1_ratio must be strictly between 0 and 1");

    let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
    let mut w = Array1::<f32>::zeros(n_features);
    let xtx = x.t().dot(x);
    let mut residuals = y.to_owned(); // Initialize residuals
    let alpha = alpha * n_samples as f32;

    for _ in 0..max_iter {
        let w_old = w.clone();
        for j in 0..n_features {
            let xj = x.slice(s![.., j]);
            // Naive update: add contribution of current feature to residuals
            residuals = &residuals + &xj * w[j];
            w[j] = soft_threshold(&xj.dot(&residuals.view()), alpha * l1_ratio)
                / (xtx[[j, j]] + alpha * (1.0 - l1_ratio));
            // Naive update: subtract contribution of current feature from residuals
            residuals = &residuals - &xj * w[j];
        }
        if (&w - &w_old).norm_l2() < tol {
            break;
        }
    }
    w
}
