use strassen::naive_matmul;

use crate::matrix::{AsPointerToSlice, TransposableSubmatrix, TransposableSubmatrixMut};
use crate::ring::*;

pub mod strassen;

///
/// Trait to allow rings to provide specialized implementations for inner products, i.e.
/// the sums `sum_i a[i] * b[i]`.
/// 
#[stability::unstable(feature = "enable")]
pub trait ComputeInnerProduct: RingBase {

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element;
}

impl<R: ?Sized + RingBase> ComputeInnerProduct for R {

    default fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product(els.map(|(l, r)| (self.clone_el(l), r)))
    }

    default fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product_ref_fst(els.map(|(l, r)| (l, self.clone_el(r))))
    }

    default fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        self.sum(els.map(|(l, r)| self.mul(l, r)))
    }
}

///
/// Trait for objects that can compute a matrix multiplications over a fixed ring.
/// 
#[stability::unstable(feature = "enable")]
pub trait MatmulAlgorithm<R: ?Sized + RingBase> {

    ///
    /// Computes the matrix product of `lhs` and `rhs`, and adds the result to `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] += sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn add_matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>;
         
    ///
    /// Computes the matrix product of `lhs` and `rhs`, and stores the result in `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] = sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    ///    
    fn matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        for i in 0..dst.row_count() {
            for j in 0..dst.col_count() {
                *dst.at_mut(i, j) = ring.zero();
            }
        }
        self.add_matmul(lhs, rhs, dst, ring);
    }
}

#[stability::unstable(feature = "enable")]
pub const STANDARD_MATMUL: DirectMatmulAlgorithm = DirectMatmulAlgorithm;

#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
pub struct DirectMatmulAlgorithm;

impl<R: ?Sized + RingBase> MatmulAlgorithm<R> for DirectMatmulAlgorithm {

    fn add_matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: &R
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        naive_matmul::<_, _, _, _, true, T1, T2, T3>(lhs, rhs, dst, ring)
    }

    fn matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: &R
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        naive_matmul::<_, _, _, _, false, T1, T2, T3>(lhs, rhs, dst, ring)
    }
}
