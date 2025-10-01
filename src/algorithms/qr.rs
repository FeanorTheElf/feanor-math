use crate::algorithms::matmul::ComputeInnerProduct;
use crate::field::{Field, FieldStore};
use crate::matrix::*;
use crate::ring::*;
use crate::rings::approx_real::{ApproxRealField, SqrtRing};

use std::cmp::min;

#[stability::unstable(feature = "enable")]
pub trait QRDecompositionField: Field {

    ///
    /// Given a matrix `A`, computes an orthogonal matrix `Q` and an upper triangular
    /// matrix `R` with `A = Q R`. The function writes `Q diag(x_1, ..., x_n)` to `q` and
    /// `diag(1/x_1, ..., 1/x_n) R` to `matrix`, and returns `x_1^2, ..., x_n^2`, where
    /// `x_1, ..., x_n` are the elements on the diagonal of `R`.
    /// 
    /// Returning the values as given above instead of just `Q` and `R` is done
    /// to avoid the computation of square-roots, which may not be supported by the
    /// underlying ring. If it is supported, you can use [`QRDecompositionField::q_r_decomposition()`]
    /// instead. Note that this means that `diag(x_1^2, ..., x_n^2)` and `R`
    /// are the LDL-decomposition of `A^T A`.
    /// 
    fn q_d_r_decomposition<V1, V2>(&self, matrix: SubmatrixMut<V1, Self::Element>, q: SubmatrixMut<V2, Self::Element>) -> Vec<Self::Element>
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>;

    ///
    /// Given a symmetric matrix `A`, computes a strict lower triangular matrix `L` and
    /// a diagonal matrix `D` such that `A = L D L^T`. The function writes `L` to `matrix`
    /// and returns the diagonal elements of `D`.
    /// 
    fn l_d_l_decomposition<V>(&self, matrix: SubmatrixMut<V, Self::Element>) -> Vec<Self::Element>
        where V: AsPointerToSlice<Self::Element>
    {
        l_d_l_decomposition_impl(RingRef::new(self), matrix)
    }
       
    ///
    /// Given a matrix `A`, computes an orthogonal matrix `Q` and an upper triangular
    /// matrix `R` with `A = Q R`. These are returned in `matrix` and `q`, respectively.
    /// 
    /// Note that if the ring is not a [`SqrtRing`], you can still use [`QRDecompositionField::q_d_r_decomposition()`].
    ///  
    fn q_r_decomposition<V1, V2>(&self, mut matrix: SubmatrixMut<V1, Self::Element>, mut q: SubmatrixMut<V2, Self::Element>)
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>, Self: SqrtRing
    {
        let d = self.q_d_r_decomposition(matrix.reborrow(), q.reborrow());
        for (i, scale_sqr) in d.into_iter().enumerate() {
            let scale = self.sqrt(scale_sqr);
            let scale_inv = self.div(&self.one(), &scale);
            for j in 0..matrix.col_count() {
                self.mul_assign_ref(matrix.at_mut(i, j), &scale);
            }
            for k in 0..q.row_count() {
                self.mul_assign_ref(q.at_mut(k, i), &scale_inv);
            }
        }
    }
}

fn l_d_l_decomposition_impl<R, V>(ring: R, mut matrix: SubmatrixMut<V, El<R>>) -> Vec<El<R>>
    where R: RingStore, 
        R::Type: Field,
        V: AsPointerToSlice<El<R>>
{
    assert_eq!(matrix.row_count(), matrix.col_count());
    let n = matrix.row_count();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let pivot = ring.clone_el(matrix.at(i, i));
        if !ring.get_ring().is_approximate() && ring.is_zero(&pivot) {
            panic!("matrix is singular")
        }
        let pivot_inv = ring.div(&ring.one(), matrix.at(i, i));
        for j in i..n {
            ring.mul_assign_ref(matrix.at_mut(j, i), &pivot_inv);
        }
        for k in (i + 1)..n {
            for l in k..n {
                let subtract = ring.mul_ref_snd(ring.mul_ref(matrix.as_const().at(k, i), matrix.as_const().at(l, i)), &pivot);
                ring.sub_assign(matrix.at_mut(l, k), subtract);
            }
        }
        result.push(pivot);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            *matrix.at_mut(i, j) = ring.zero();
        }
    }
    return result;
}

impl<R: ApproxRealField + SqrtRing> QRDecompositionField for R {

    default fn q_d_r_decomposition<V1, V2>(&self, mut matrix: SubmatrixMut<V1, Self::Element>, mut q: SubmatrixMut<V2, Self::Element>) -> Vec<Self::Element>
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>
    {
        self.q_r_decomposition(matrix.reborrow(), q.reborrow());
        return unimplemented!();
    }

    default fn l_d_l_decomposition<V>(&self, matrix: SubmatrixMut<V, Self::Element>) -> Vec<Self::Element>
        where V: AsPointerToSlice<Self::Element>
    {
        l_d_l_decomposition_impl(RingRef::new(self), matrix)
    }

    default fn q_r_decomposition<V1, V2>(&self, mut matrix: SubmatrixMut<V1, Self::Element>, mut q: SubmatrixMut<V2, Self::Element>)
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>
    {
        let ring = RingRef::new(self);
        let m = matrix.row_count();
        let n = matrix.col_count();
        assert_eq!(m, q.row_count());
        assert_eq!(m, q.col_count());
        for i in 0..m {
            for j in 0..m {
                *q.at_mut(i, j) = if i == j { self.one() } else { self.zero() };
            }
        }

        for i in 0..min(n, m) {
            let norm_sqr = <_ as RingStore>::sum(&ring, (i..m).map(|k| ring.pow(ring.clone_el(matrix.at(k, i)), 2)));
            let norm = self.sqrt(self.clone_el(&norm_sqr));
            let alpha = if self.is_neg(matrix.at(i, i)) {
                self.clone_el(&norm)
            } else {
                self.negate(self.clone_el(&norm))
            };
            let x1_minus_alpha = ring.sub_ref(matrix.at(i, i), &alpha);
            // the value `| x - alpha * e1 |^2 / 2`
            let half_norm_sqr_x_minus_alpha = self.sub(norm_sqr, self.mul_ref(&alpha, matrix.at(i, i)));
            let half_norm_sqr_x_minus_alpha_inv = self.div(&ring.one(), &half_norm_sqr_x_minus_alpha);

            // update matrix
            let (mut pivot, mut rest) = matrix.reborrow().submatrix(i..m, i..n).split_cols(0..1, 1..(n - i));
            for j in 0..(n - i - 1) {
                let scale = ring.mul_ref_fst(
                    &half_norm_sqr_x_minus_alpha_inv, 
                    <_ as ComputeInnerProduct>::inner_product_ref(self, [(&x1_minus_alpha, rest.at(0, j))].into_iter().chain((1..(m - i)).map(|k| (pivot.at(k, 0), rest.at(k, j))))),
                );
                ring.sub_assign(rest.at_mut(0, j), ring.mul_ref(&scale, &x1_minus_alpha));
                for k in 1..(m - i) {
                    ring.sub_assign(rest.at_mut(k, j), ring.mul_ref(&scale, pivot.at(k, 0)));
                }
            }

            // update q
            let mut rest = q.reborrow().restrict_cols(i..m);
            for j in 0..m {
                let scale = ring.mul_ref_fst(
                    &half_norm_sqr_x_minus_alpha_inv, 
                    <_ as ComputeInnerProduct>::inner_product_ref(self, [(&x1_minus_alpha, rest.at(j, 0))].into_iter().chain((1..(m - i)).map(|k| (pivot.at(k, 0), rest.at(j, k))))),
                );
                ring.sub_assign(rest.at_mut(j, 0), ring.mul_ref(&scale, &x1_minus_alpha));
                for k in 1..(m - i) {
                    ring.sub_assign(rest.at_mut(j, k), ring.mul_ref(&scale, pivot.at(k, 0)));
                }
            }

            // update pivot
            for k in 1..(m - i) {
                *pivot.at_mut(k, 0) = self.zero();
            }
            *pivot.at_mut(0, 0) = alpha;
        }
    }
}

#[cfg(test)]
use crate::algorithms::matmul::STANDARD_MATMUL;
#[cfg(test)]
use crate::matrix::{TransposableSubmatrix, TransposableSubmatrixMut};
#[cfg(test)]
use crate::algorithms::matmul::MatmulAlgorithm;
#[cfg(test)]
use crate::matrix::format_matrix;
#[cfg(test)]
use crate::rings::approx_real::float::Real64;

#[cfg(test)]
fn assert_is_correct_qr<V1, V2, V3>(original: Submatrix<V1, f64>, q: Submatrix<V2, f64>, r: Submatrix<V3, f64>)
    where V1: AsPointerToSlice<f64>, V2: AsPointerToSlice<f64>, V3: AsPointerToSlice<f64>
{
    let m = q.row_count();
    let n = r.col_count();
    assert_eq!(m, original.row_count());
    assert_eq!(n, original.col_count());
    assert_eq!(m, r.row_count());
    let mut product = OwnedMatrix::zero(m, n, Real64::RING);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(q),
        TransposableSubmatrix::from(r),
        TransposableSubmatrixMut::from(product.data_mut()),
        Real64::RING
    );
    for i in 0..m {
        for j in 0..n {
            if !(Real64::RING.get_ring().is_approx_eq(*original.at(i, j), *product.at(i, j), 100)) {
                println!("product does not match; Q, R are");
                println!("{}", format_matrix(m, m, |i, j| q.at(i, j), Real64::RING));
                println!("and");
                println!("{}", format_matrix(m, n, |i, j| r.at(i, j), Real64::RING));
                println!("the product is");
                println!("{}", format_matrix(m, n, |i, j| product.at(i, j), Real64::RING));
                panic!();
            }
        }
    }
    let mut product = OwnedMatrix::zero(m, m, Real64::RING);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(q).transpose(),
        TransposableSubmatrix::from(q),
        TransposableSubmatrixMut::from(product.data_mut()),
        Real64::RING
    );
    for i in 0..m {
        for j in 0..m {
            let expected = if i == j { 1. } else { 0. };
            if !(Real64::RING.get_ring().is_approx_eq(expected, *product.at(i, j), 100)) {
                println!("Q is not orthogonal");
                println!("{}", format_matrix(m, m, |i, j| q.at(i, j), Real64::RING));
                panic!();
            }
        }
    }

    for j in 0..n {
        for i in (j + 1)..m {
            if !(Real64::RING.get_ring().is_approx_eq(0., *r.at(i, j), 100)) {
                println!("R is not upper triangular");
                println!("{}", format_matrix(m, n, |i, j| r.at(i, j), Real64::RING));
                panic!();
            }
        }
    }
}

#[cfg(test)]
fn assert_is_correct_ldl<V1, V2>(original: Submatrix<V1, f64>, l: Submatrix<V2, f64>, d: &[f64])
    where V1: AsPointerToSlice<f64>, V2: AsPointerToSlice<f64>
{
    let n = l.col_count();
    assert_eq!(n, l.row_count());
    assert_eq!(n, original.col_count());
    assert_eq!(n, original.row_count());
    let l_scaled = OwnedMatrix::from_fn(n, n, |i, j| *l.at(i, j) * d[j]);
    let mut product = OwnedMatrix::zero(n, n, Real64::RING);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(l_scaled.data()),
        TransposableSubmatrix::from(l).transpose(),
        TransposableSubmatrixMut::from(product.data_mut()),
        Real64::RING
    );
    for i in 0..n {
        for j in 0..n {
            if !(Real64::RING.get_ring().is_approx_eq(*original.at(i, j), *product.at(i, j), 100)) {
                println!("product does not match; L is");
                println!("{}", format_matrix(n, n, |i, j| l.at(i, j), Real64::RING));
                println!("D is diag{:?} and the product LDL^T is", d);
                println!("{}", format_matrix(n, n, |i, j| product.at(i, j), Real64::RING));
                panic!();
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if !(Real64::RING.get_ring().is_approx_eq(0., *l.at(i, j), 100)) {
                println!("L is not lower triangular");
                println!("{}", format_matrix(n, n, |i, j| l.at(i, j), Real64::RING));
                panic!();
            }
        }
    }
}

#[test]
fn test_qr() {
    let RR = Real64::RING;
    let a = OwnedMatrix::new_with_shape(vec![0., 1., 1., 0.], 2, 2);
    let mut r = a.clone_matrix(RR);
    let mut q = OwnedMatrix::zero(2, 2, RR);
    RR.get_ring().q_r_decomposition(r.data_mut(), q.data_mut());
    assert_is_correct_qr(a.data(), q.data(), r.data());

    let a = OwnedMatrix::new_with_shape(vec![1., 2., 3., 4., 5., 6.], 3, 2);
    let mut r = a.clone_matrix(RR);
    let mut q = OwnedMatrix::zero(3, 3, RR);
    RR.get_ring().q_r_decomposition(r.data_mut(), q.data_mut());
    assert_is_correct_qr(a.data(), q.data(), r.data());

    let a = OwnedMatrix::new_with_shape(vec![1., 2., 3., 4., 5., 6.], 2, 3);
    let mut r = a.clone_matrix(RR);
    let mut q = OwnedMatrix::zero(2, 2, RR);
    RR.get_ring().q_r_decomposition(r.data_mut(), q.data_mut());
    assert_is_correct_qr(a.data(), q.data(), r.data());

    let a = OwnedMatrix::new_with_shape((1..31).map(|x| x as f64 * if x % 2 == 0 { -1.0 } else { 1.0 }).collect::<Vec<_>>(), 6, 5);
    let mut r = a.clone_matrix(RR);
    let mut q = OwnedMatrix::zero(6, 6, RR);
    RR.get_ring().q_r_decomposition(r.data_mut(), q.data_mut());
    assert_is_correct_qr(a.data(), q.data(), r.data());
}

#[test]
fn test_ldl() {
    let RR = Real64::RING;
    let a = OwnedMatrix::new_with_shape(vec![5., 1., 1., 5.], 2, 2);
    let mut l = a.clone_matrix(RR);
    let d = RR.get_ring().l_d_l_decomposition(l.data_mut());
    assert_is_correct_ldl(a.data(), l.data(), &d);

    let a = OwnedMatrix::new_with_shape(vec![1., 2., 3., 2., 6., 5., 3., 5., 20.], 3, 3);
    let mut l = a.clone_matrix(RR);
    let d = RR.get_ring().l_d_l_decomposition(l.data_mut());
    assert_is_correct_ldl(a.data(), l.data(), &d);
    
    let mut a = OwnedMatrix::zero(5, 5, RR);
    let factor = OwnedMatrix::new((0..25).map(|c| (c as f64).powi(2)).collect(), 5);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(factor.data()),
        TransposableSubmatrix::from(factor.data()).transpose(),
        TransposableSubmatrixMut::from(a.data_mut()),
        RR
    );
    let mut l = a.clone_matrix(RR);
    let d = RR.get_ring().l_d_l_decomposition(l.data_mut());
    assert_is_correct_ldl(a.data(), l.data(), &d);

    let a = OwnedMatrix::new_with_shape(vec![1., 2., 3., 2., 6., 5., 3., 5., -20.], 3, 3);
    let mut l = a.clone_matrix(RR);
    let d = RR.get_ring().l_d_l_decomposition(l.data_mut());
    assert_is_correct_ldl(a.data(), l.data(), &d);
}