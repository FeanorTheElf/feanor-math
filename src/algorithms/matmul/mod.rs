use crate::{matrix::{AsPointerToSlice, Submatrix, SubmatrixMut}, ring::*};

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
    fn add_matmul<V1, V2, V3>(&self, lhs: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, dst: SubmatrixMut<V3, R::Element>, ring: &R)
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
    fn matmul<V1, V2, V3>(&self, lhs: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
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
    
    ///
    /// Computes the matrix product of `lhs` transposed and `rhs`, and adds the result to `dst`.
    /// 
    /// This requires that `lhs` is a `kxn` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] += sum_l lhs[l, i] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn add_matmul_fst_transposed<V1, V2, V3>(&self, lhs_T: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>;

    ///
    /// Computes the matrix product of `lhs` transposed and `rhs`, and stores the result in `dst`.
    /// 
    /// This requires that `lhs` is a `kxn` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] = sum_l lhs[l, i] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn matmul_fst_transposed<V1, V2, V3>(&self, lhs_T: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        for i in 0..dst.row_count() {
            for j in 0..dst.col_count() {
                *dst.at_mut(i, j) = ring.zero();
            }
        }
        self.add_matmul_fst_transposed(lhs_T, rhs, dst, ring);
    }

    ///
    /// Computes the matrix product of `lhs` and `rhs` transposed, and adds the result to `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `mxk` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] += sum_l lhs[i, l] * rhs[j, l]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn add_matmul_snd_transposed<V1, V2, V3>(&self, lhs: Submatrix<V1, R::Element>, rhs_T: Submatrix<V2, R::Element>, dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>;
    
    ///
    /// Computes the matrix product of `lhs` and `rhs` transposed, and stores the result in `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `mxk` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] = sum_l lhs[i, l] * rhs[j, l]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn matmul_snd_transposed<V1, V2, V3>(&self, lhs: Submatrix<V1, R::Element>, rhs_T: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        for i in 0..dst.row_count() {
            for j in 0..dst.col_count() {
                *dst.at_mut(i, j) = ring.zero();
            }
        }
        self.add_matmul_snd_transposed(lhs, rhs_T, dst, ring);
    }
}

#[stability::unstable(feature = "enable")]
pub const STANDARD_MATMUL: DirectMatmulAlgorithm = DirectMatmulAlgorithm;

#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
pub struct DirectMatmulAlgorithm;

impl<R: ?Sized + RingBase> MatmulAlgorithm<R> for DirectMatmulAlgorithm {

    fn add_matmul<V1,V2,V3>(&self, lhs: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        assert_eq!(lhs.row_count(), dst.row_count());
        assert_eq!(rhs.col_count(), dst.col_count());
        assert_eq!(lhs.col_count(), rhs.row_count());
        for i in 0..lhs.row_count() {
            for j in 0..rhs.col_count() {
                ring.add_assign(dst.at_mut(i, j), <_ as ComputeInnerProduct>::inner_product_ref(ring, (0..lhs.col_count()).map(|k| (lhs.at(i, k), rhs.at(k, j)))));
            }
        }
    }

    fn add_matmul_fst_transposed<V1,V2,V3>(&self, lhs_T: Submatrix<V1, R::Element>, rhs: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        assert_eq!(lhs_T.col_count(), dst.row_count());
        assert_eq!(rhs.col_count(), dst.col_count());
        assert_eq!(lhs_T.row_count(), rhs.row_count());
        for i in 0..lhs_T.col_count() {
            for j in 0..rhs.col_count() {
                ring.add_assign(dst.at_mut(i, j), <_ as ComputeInnerProduct>::inner_product_ref(ring, (0..lhs_T.row_count()).map(|k| (lhs_T.at(k, i), rhs.at(k, j)))));
            }
        }
    }
    
    fn add_matmul_snd_transposed<V1,V2,V3>(&self, lhs: Submatrix<V1, R::Element>, rhs_T: Submatrix<V2, R::Element>, mut dst: SubmatrixMut<V3, R::Element>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        assert_eq!(lhs.row_count(), dst.row_count());
        assert_eq!(rhs_T.row_count(), dst.col_count());
        assert_eq!(lhs.col_count(), rhs_T.col_count());
        for i in 0..lhs.row_count() {
            for j in 0..rhs_T.row_count() {
                ring.add_assign(dst.at_mut(i, j), <_ as ComputeInnerProduct>::inner_product_ref(ring, (0..lhs.col_count()).map(|k| (lhs.at(i, k), rhs_T.at(j, k)))));
            }
        }
    }
}
