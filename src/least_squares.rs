use faer::prelude::SpSolverLstsq;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{s, Array, Array1, Array2, ArrayView1, Axis, NewAxis};
use ndarray_linalg::{Inverse, InverseC, Norm, SolveC};

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

/// Solves the normal equations: (X^T X) coefficients = X^T Y
fn solve_normal_equations(xtx: &Array2<f32>, xty: &Array1<f32>) -> Array1<f32> {
    // attempt to solve via cholesky making use of X.T X being SPD
    match xtx.solvec(xty) {
        Ok(coefficients) => coefficients,
        Err(_) => {
            // else fallback to QR decomposition
            solve_ols_qr(xty, xtx)
        }
    }
}

/// Solves a ridge regression problem of the form: ||y - x B|| + alpha * ||B||
/// Inputs: features (2d ndarray), targets (1d ndarray), ridge alpha scalar
pub fn solve_ridge(y: &Array1<f32>, x: &Array2<f32>, alpha: f32) -> Array1<f32> {
    assert!(alpha > 0., "alpha must be strictly positive");
    let x_t = &x.t();
    let x_t_x = x_t.dot(x);
    let x_t_y = x_t.dot(y);
    let eye = Array::eye(x_t_x.shape()[0]);
    let ridge_matrix = &x_t_x + &eye * alpha;
    solve_normal_equations(&ridge_matrix, &x_t_y)
}

fn soft_threshold(x: &f32, alpha: f32, positive: bool) -> f32 {
    let mut result = x.signum() * (x.abs() - alpha).max(0.0);
    if positive {
        result = result.max(0.0);
    }
    result
}

/// Solves an elastic net regression problem of the form: 1 / (2 * n_samples) * ||y - Xw||_2
/// + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2
pub fn solve_elastic_net(
    y: &Array1<f32>,
    x: &Array2<f32>,
    alpha: f32,            // strictly positive regularization parameter
    l1_ratio: Option<f32>, // scalar strictly between 0 (full ridge) and 1 (full lasso)
    max_iter: Option<usize>,
    tol: Option<f32>,       // controls convergence criteria between iterations
    positive: Option<bool>, // enforces non-negativity constraint
) -> Array1<f32> {
    let l1_ratio = l1_ratio.unwrap_or(0.5);
    let max_iter = max_iter.unwrap_or(1_000);
    let tol = tol.unwrap_or(0.0001);
    let positive = positive.unwrap_or(false);

    assert!(alpha > 0., "'alpha' must be strictly positive");
    assert!(
        (0. ..=1.).contains(&l1_ratio),
        "'l1_ratio' must be strictly between 0. and 1."
    );

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
            w[j] = soft_threshold(&xj.dot(&residuals.view()), alpha * l1_ratio, positive)
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

pub struct RecursiveLeastSquares {
    forgetting_factor: f32, // exponential decay factor
    coef: Array1<f32>,      // coefficient vector
    p: Array2<f32>,         // state covariance
    k: Array1<f32>,         // kalman gain
}

impl RecursiveLeastSquares {
    pub fn new(
        num_features: usize,
        lam: f32,
        half_life: Option<f32>,
        initial_state_mean: Option<Array1<f32>>,
    ) -> Self {
        // calculate forgetting_factor based on the value of half_life, default to 1.0
        // (expanding ols)
        let forgetting_factor = if let Some(half_life) = half_life {
            0.5f32.ln() / half_life
        } else {
            1.0
        };

        let coef = Array1::<f32>::zeros(num_features);
        let p = Array2::<f32>::eye(num_features) * lam;
        let k = Array1::<f32>::zeros(num_features);
        let coef = initial_state_mean.unwrap_or(coef);
        RecursiveLeastSquares {
            forgetting_factor,
            coef,
            p,
            k,
        }
    }

    pub fn update(&mut self, x: &Array1<f32>, y: f32) {
        let r = 1.0 + x.t().dot(&self.p).dot(x) / self.forgetting_factor;
        self.k
            .assign(&(&self.p.dot(x) / (r * self.forgetting_factor)));
        let residuals = y - x.dot(&self.coef);
        self.coef.assign(&(&self.coef + &(&self.k * residuals)));
        // unfortunately can't find equivalent of 'np.outer' in ndarray
        let k_ = &self.k.view().insert_axis(Axis(1)); // K x 1
        self.p
            .assign(&(&self.p / self.forgetting_factor - k_.dot(&k_.t()) * r));
    }

    pub fn predict(&self, x: &Array1<f32>) -> f32 {
        x.dot(&self.coef)
    }
}

/// Solves an online least squares problem updating coefficients with every sample.
///
/// This function performs online least squares regression, updating the coefficients
/// with every sample. It uses the Recursive Least Squares (RLS) algorithm to adaptively
/// update the coefficients as new data arrives.
///
/// # Arguments
/// * `y` - A reference to a one-dimensional array containing the target values.
/// * `x` - A reference to a two-dimensional array containing the input features.
/// * `half_life` - An optional parameter representing the half-life of forgetting past information
///                 in the Recursive Least Squares algorithm. A smaller half-life places more
///                 weight on recent samples.
/// * `initial_state_covariance` - An optional parameter representing the initial covariance
///                                 matrix of the state estimation. Default value is 10.0.
/// * `initial_state_mean` - An optional parameter representing the initial mean vector of the
///                           state estimation. If not provided, it is initialized to zeros.
///
/// # Returns
/// A two-dimensional array containing the updated coefficients of the linear regression model.
pub fn solve_recursive_least_squares(
    y: &Array1<f32>,
    x: &Array2<f32>,
    half_life: Option<f32>,
    initial_state_covariance: Option<f32>,
    initial_state_mean: Option<Array1<f32>>,
) -> Array2<f32> {
    let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
    let mut recursive_least_squares = RecursiveLeastSquares::new(
        n_features,
        initial_state_covariance.unwrap_or(10.0),
        half_life,
        initial_state_mean,
    );
    // let mut predictions = Array1::<f32>::zeros(n_samples);
    let mut coefficients = Array2::<f32>::zeros((n_samples, n_features));

    for t in 0..n_samples {
        let y_t = y[t];
        let x_t = x.slice(s![t, ..]).to_owned();
        recursive_least_squares.update(&x_t, y_t);
        coefficients
            .slice_mut(s![t, ..])
            .assign(&recursive_least_squares.coef.view());
        // predictions[t] = recursive_least_squares.predict(&x_t);
    }
    coefficients
}

fn outer_product(u: &ArrayView1<f32>, v: &ArrayView1<f32>) -> Array2<f32> {
    // Reshape u and v to have a shape of (n, 1) and (1, m) respectively
    let u_reshaped = u.insert_axis(Axis(1));
    let v_reshaped = v.insert_axis(Axis(0));

    // Compute the outer product using broadcasting and dot product
    u_reshaped.dot(&v_reshaped)
}

fn inv_diag(c: &Array2<f32>) -> Array2<f32> {
    let s = c.raw_dim();
    assert!(s[0] == s[1]);
    let mut res: Array2<f32> = Array2::zeros(s);
    for i in 0..s[0] {
        res[(i, i)] = c[(i, i)].recip();
    }
    res
}

/// Computes the Woodbury update of a matrix A_inv using matrices U, V, B_inv.
///
/// The Woodbury update is a method to efficiently update the inverse of a matrix A_inv
/// when adding or removing rows or columns from the original matrix.
///
/// The Woodbury update formula is given by:
///
/// ```text
/// (A + U C V)^{-1} = A^{-1} + A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
/// ```
pub fn woodbury_update(
    a_inv: &Array2<f32>,
    u: &Array2<f32>,
    c: &Array2<f32>,
    v: &Array2<f32>,
    c_is_diag: Option<bool>,
) -> Array2<f32> {
    // Check if c_is_diag is Some(true)
    let inv_c = if let Some(true) = c_is_diag {
        inv_diag(c)
    } else {
        c.inv().unwrap()
    };
    // compute V inv(A)
    let v_inv_a = v.dot(a_inv);
    let inv_a_u = a_inv.dot(u);
    // compute term (C^{-1} + V A^{-1} U)^{-1}
    let intermediate = (inv_c + v.dot(&inv_a_u)).inv().unwrap();
    a_inv - inv_a_u.dot(&intermediate).dot(&v_inv_a)
}

/// Function to update inv(X^TX) by x_update array of rank r using Woodbury Identity.
fn update_xtx_inv(xtx_inv: &Array2<f32>, x_update: &Array2<f32>) -> Array2<f32> {
    // Reshape x_new and x_old for Woodbury update
    let u = x_update.t().to_owned(); // K x r
    let v = u.t().to_owned(); // r x K
    let c = Array2::eye(u.shape()[1]); // Identity matrix r x r

    // Apply Woodbury update
    woodbury_update(xtx_inv, &u, &c, &v, Some(true))
}

pub fn solve_rolling_ols(
    y: &Array1<f32>,
    x: &Array2<f32>,
    window_size: usize,
    min_periods: Option<usize>,
    use_woodbury: Option<bool>,
) -> Array2<f32> {
    let n = x.shape()[0];
    let k = x.shape()[1]; // Number of independent variables
    let min_periods = min_periods.unwrap_or(1);
    let use_woodbury = use_woodbury.unwrap_or(false);
    let mut coefficients = Array2::zeros((n, k));

    // Initialize X^T X, inv(X.T X), and X^T Y
    let x_window = x.slice(s![..window_size, ..]);
    let y_window = y.slice(s![..window_size]);
    let mut xty = x_window.t().dot(&y_window);
    let mut xtx = x_window.t().dot(&x_window);

    // Use woodbury to propagate inv(X.T X) & (X.T Y)
    if use_woodbury {
        // assign warm-up coefficients
        let mut xtx_inv = xtx.invc().expect("could not compute inverse");
        let coef_warmup = xtx_inv.t().dot(&xty);
        coefficients
            .slice_mut(s![min_periods, ..])
            .assign(&coef_warmup);

        // Slide the window and update coefficients
        for i in min_periods + 1..n {
            let i_start = i.saturating_sub(window_size);

            let x_new = x.row(i);

            if i > window_size {
                let x_prev = x.row(i_start);

                // create rank 2 update array
                let mut x_update = ndarray::stack(Axis(0), &[x_prev, x_new]).unwrap(); // 2 x K

                // multiply x_old row by -1.0 (subtract the previous contribution)
                x_update.row_mut(0).mapv_inplace(|elem| -elem);

                // update inv(XTX) and XTY
                xtx_inv = update_xtx_inv(&xtx_inv, &x_update);
                xty = xty + &x_new * y[i] - &x_prev * y[i_start];
            } else {
                let x_update = x_new.insert_axis(Axis(0)).into_owned(); // 1 x K
                xtx_inv = update_xtx_inv(&xtx_inv, &x_update);
                xty = xty + &x_new * y[i];
            }

            let coefficients_i = xtx_inv.dot(&xty);
            coefficients.slice_mut(s![i, ..]).assign(&coefficients_i);
        }
    } else {
        // propagate X.T X & X.T Y and solve normal equations at every time step

        // assign warm-up coefficients
        let coef_warmup = solve_normal_equations(&xtx, &xty);
        coefficients
            .slice_mut(s![min_periods, ..])
            .assign(&coef_warmup);

        // Slide the window and update coefficients
        for i in min_periods + 1..n {
            let i_start = i.saturating_sub(window_size);

            // update XTX w/ latest data point
            let x_new = x.row(i);

            // Add new contributions
            xtx += &outer_product(&x_new, &x_new);
            xty = xty - &x_new * y[i];

            // Subtract the previous contribution
            if i > window_size {
                let x_prev = x.row(i_start);
                xtx -= &outer_product(&x_prev, &x_prev);
                xty = xty - &x_prev * y[i_start];
            }

            // update coefficients
            let coefficients_i = solve_normal_equations(&xtx, &xty);
            coefficients.slice_mut(s![i, ..]).assign(&coefficients_i);
        }
    }
    coefficients
}
