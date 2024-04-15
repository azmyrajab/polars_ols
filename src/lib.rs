use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

mod expressions;
pub mod least_squares;

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use ndarray_linalg::assert_close_l2;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use polars::datatypes::DataType::Float64;
    use polars::prelude::*;

    use crate::expressions::convert_polars_to_ndarray;
    use crate::least_squares::{
        inv, outer_product, solve_elastic_net, solve_ols, solve_recursive_least_squares,
        solve_ridge, solve_rolling_ols, update_xtx_inv, woodbury_update, NullPolicy, SolveMethod,
    };

    fn make_data(null_policy: Option<NullPolicy>) -> (Array1<f64>, Array2<f64>) {
        let null_policy = null_policy.unwrap_or(NullPolicy::Ignore);
        let x1 = Series::from_vec(
            "x1",
            Array::random(10_000, Normal::new(0., 1.).unwrap()).to_vec(),
        )
        .cast(&Float64)
        .unwrap();
        let x2 = Series::from_vec(
            "x2",
            Array::random(10_000, Normal::new(0., 1.).unwrap()).to_vec(),
        )
        .cast(&Float64)
        .unwrap();
        let y = (&x1 + &x2).with_name("y");

        convert_polars_to_ndarray(&[y.clone(), x1, x2], &null_policy, None)
    }

    #[test]
    fn test_ols() {
        let (targets, features) = make_data(None);
        let coefficients_1 = solve_ols(&targets, &features, None, None);
        let coefficients_2 = solve_ols(&targets, &features, Some(SolveMethod::SVD), None);
        let expected = array![1., 1.];
        assert_close_l2!(&coefficients_1, &coefficients_2, 0.001);
        assert_close_l2!(&coefficients_1, &expected, 0.001);
    }

    #[test]
    fn test_ridge() {
        let (targets, features) = make_data(None);
        let coefficients_1 = solve_ridge(&targets, &features, 10.0, None, None);
        let coefficients_2 = solve_ridge(&targets, &features, 10.0, Some(SolveMethod::SVD), None);
        let expected = array![0.999, 0.999];
        assert_close_l2!(&coefficients_1, &coefficients_2, 0.001);
        assert_close_l2!(&coefficients_1, &expected, 0.001);
    }

    #[test]
    fn test_elastic_net() {
        let (targets, features) = make_data(None);
        let coefficients = solve_elastic_net(
            &targets,
            &features,
            0.001,
            Some(0.5),
            None,
            None,
            None,
            None,
        );
        let expected = array![0.999, 0.999];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_recursive_least_squares() {
        let (targets, features) = make_data(None);
        let is_valid = vec![true; targets.len()];

        let coefficients = solve_recursive_least_squares(
            &targets,
            &features,
            Some(252.0),
            Some(0.01),
            None,
            &is_valid,
        );
        let expected = array![1.0, 1.0];
        println!("{:?}", coefficients.slice(s![0, ..]));
        println!("{:?}", coefficients.slice(s![-1, ..]));
        assert_close_l2!(&coefficients.slice(s![-1, ..]), &expected, 0.0001);
    }

    #[test]
    fn test_rolling_least_squares() {
        let (targets, features) = make_data(None);
        let is_valid = vec![true; targets.len()];

        let coefficients = solve_rolling_ols(
            &targets,
            &features,
            1_000usize,
            Some(100usize),
            Some(false),
            None,
            &is_valid,
            NullPolicy::DropWindow,
        );
        let expected: Array1<f64> = array![1.0, 1.0];
        println!("{:?}", coefficients.slice(s![0, ..]));
        println!("{:?}", coefficients.slice(s![-1, ..]));
        assert_close_l2!(&coefficients.slice(s![-1, ..]), &expected, 0.0001);
    }

    #[test]
    fn test_woodbury_update() {
        // Test matrices
        let a = array![[0.5, 0.2], [0.0, 0.5]]; // A^{-1}
        let a_inv = inv(&a, false);
        let u = array![[1.0, 2.0], [3.0, 4.0]]; // U
        let c = array![[1.0, 0.0], [0.0, 1.0]]; // C
        let v = array![[1.0, 0.0], [0.0, 1.0]]; // V

        // Expected result
        let expected_result = inv(&(&a + &u.dot(&c).dot(&v)), false);

        // Compute the Woodbury update
        let result = woodbury_update(&a_inv, &u, &c, &v, Some(true));

        // test confirms: inv(A + UCV) == A{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

        // Compare with expected result
        assert_close_l2!(&result, &expected_result, 0.00001);
    }

    #[test]
    fn test_update_xtx_inv() {
        // Test matrices
        let x = Array2::<f64>::random((252, 5), Normal::new(0., 1.).unwrap());

        let xtx = x.t().dot(&x);
        let mut xtx_inv = inv(&xtx, true);

        let x_new = array![0.5, 2., -0.3, 0.1, 0.2];
        let x_new = x_new.view(); // new data point
        let x_old = x.slice(s![0, ..]); // old data point

        // create rank 2 update array
        let x_update = ndarray::stack(Axis(0), &[x_old, x_new]).unwrap().clone(); // 2 x K

        let c: Array2<f64> = array![[-1., 0.], [0., 1.]]; // subtract x_old, add x_new

        // update xtx inv with [x_old, x_new] using woodbury
        xtx_inv = update_xtx_inv(&xtx_inv, &x_update, Some(&c));

        // test non fancy xtx update + invert
        let expected = inv(
            &(&xtx - &outer_product(&x_old, &x_old) + &outer_product(&x_new, &x_new)),
            true,
        );
        assert_close_l2!(&xtx_inv, &expected, 0.00001);
    }
}

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[pymodule]
#[pyo3(name = "_polars_ols")]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
