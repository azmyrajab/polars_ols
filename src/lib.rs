mod expressions;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[cfg(test)]
mod tests {
    use polars::datatypes::DataType::Float32;
    use polars::prelude::*;
    use ndarray_rand::rand_distr::Normal;
    use ndarray::prelude::*;
    use ndarray_linalg::{assert_close_l2};
    use ndarray_rand::RandomExt;
    use crate::expressions::{convert_polars_to_ndarray, solve_ols_lapack, solve_ols_qr, solve_ridge};

    fn make_data() -> (Series, Series, Series) {
        // let y = Series::new("targets", [3, 3, 3, 4]).cast(&Float32).unwrap();
        // let x1 = Series::new("x1", [1, 1, 1, 1]).cast(&Float32).unwrap();
        // let x2 = Series::new("x2", [0, 0, 0, 1]).cast(&Float32).unwrap();
        let x1 = Series::from_vec(
            "x1", Array::random(10_000, Normal::new(0., 10.).unwrap()).to_vec(),
        ).cast(&Float32).unwrap();
        let x2 = Series::from_vec(
            "x2", Array::random(10_000, Normal::new(0., 10.).unwrap()).to_vec(),
        ).cast(&Float32).unwrap();
        let y = (&x1 + &x2).with_name("y");
        (y, x1, x2)
    }

    #[test]
    fn test_ols_lapack() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_ols_lapack(&targets, &features);
        let expected = array![1., 1.];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_ols_qr() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_ols_qr(&targets, &features);
        let expected = array![1., 1.];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_ridge() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_ridge(&targets, &features, 1_000.0, Some("svd"));
        let expected = array![0.999, 0.999];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

}