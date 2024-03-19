use faer::prelude::SpSolverLstsq;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{s, Array, Array1, Array2, NewAxis};
use ndarray_linalg::{LeastSquaresSvd, SolveC, SVD};

/// Solves an ordinary least squares problem using ndarray and LAPACK SGELSD.
/// Inputs: features (2d ndarray), targets (1d ndarray)
/// Outputs: 1-d OLS coefficients
#[allow(dead_code)]
pub fn solve_ols_lapack(y: &Array1<f32>, x: &Array2<f32>) -> Array1<f32> {
    // compute least squares solution via LAPACK SGELSD (divide and conquer SVD)
    let solution = x.least_squares(y).expect("failed to solve least squares");
    solution.solution
}

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
/// Inputs: features (2d ndarray), targets (1d ndarray), ridge alpha scalar, & solve method string
pub fn solve_ridge(
    y: &Array1<f32>,
    x: &Array2<f32>,
    alpha: f32,
    method: Option<&str>,
) -> Array1<f32> {
    assert!(alpha > 0., "alpha must be strictly positive");

    // default to directly solving normal equations if un-specified
    let method = method.unwrap_or("solve");
    assert!(
        ["solve", "svd"].contains(&method),
        "'method' must be either 'solve' or 'svd'"
    );

    let k = std::cmp::min(x.shape()[1], x.shape()[0]);

    let coefficients = match method {
        "svd" => {
            let (u, s, v) = x.svd(true, true).unwrap();
            let u = u.unwrap();
            let v = v.unwrap();
            let ut_y = u.slice(s![.., ..k]).t().dot(y);
            let d = &s / (&s * &s + alpha);
            let d_ut_y = d * &ut_y;
            v.t().dot(&d_ut_y)
        }
        "solve" => {
            let x_t = &x.t();
            let x_t_x = x_t.dot(x);
            let x_t_y = x_t.dot(y);
            let eye = Array::eye(x_t_x.shape()[0]);
            let ridge_matrix = &x_t_x + &eye * alpha;
            ridge_matrix
                .solvec(&x_t_y)
                .expect("failed to solve normal equations (cholesky)")
        }
        _ => {
            panic!("unsupported solve method {method} passed")
        }
    };
    coefficients
}


