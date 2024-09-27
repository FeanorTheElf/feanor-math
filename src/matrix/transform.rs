use crate::ring::*;

use super::{AsPointerToSlice, SubmatrixMut};

///
/// A trait for a "target" that can "consume" elementary operations on matrices.
///  
/// This is mainly used during algorithms that work on matrices, since in many cases
/// they transform matrices using elementary row or column operations, and have to
/// accumulate data depending on these operations.
/// 
pub trait TransformTarget<R>
    where R: ?Sized + RingBase
{
    ///
    /// The transformation given by the matrix `A` with `A[k, l]` being
    ///  - `1` if `k = l` and `k != i, j`
    ///  - `transform[0]` if `(k, l) = (i, i)`
    ///  - `transform[1]` if `(k, l) = (i, j)`
    ///  - `transform[2]` if `(k, l) = (j, i)`
    ///  - `transform[3]` if `(k, l) = (j, j)`
    ///  - `0` otherwise
    /// 
    /// In other words, the matrix looks like
    /// ```text
    /// | 1  ...  0                       |
    /// | ⋮        ⋮                       |
    /// | 0  ...  1                       |
    /// |    A             B              | <- i-th row
    /// |            1  ...  0            |
    /// |            ⋮        ⋮            |
    /// |            0  ...  1            |
    /// |    C             D              | <- j-th row
    /// |                       1  ...  0 |
    /// |                       ⋮        ⋮ |
    /// |                       0  ...  1 |
    ///      ^ i-th col    ^ j-th col
    /// ```
    /// where `transform = [A, B, C, D]`.
    /// 
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]);

    ///
    /// The transformation corresponding to subtracting `factor` times the `src`-th row
    /// resp. col from the `dst`-th row resp. col.
    /// 
    /// More precisely, the `(k, l)`-th entry of the transform matrix is defined to be
    ///  - `1` if `k == l`
    ///  - `-factor` if `k == dst, l == src`
    ///  - `0` otherwise
    /// 
    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        self.transform(ring, src, dst, &[ring.one(), ring.zero(), ring.negate(ring.clone_el(factor)), ring.one()])
    }

    ///
    /// The transformation corresponding to the permutation matrix swapping `i`-th and `j`-th row
    /// resp. column.
    /// 
    /// More precisely, the `(k, l)`-th entry of the transform matrix is defined to be
    ///  - `1` if `k == l, k != i, k != j`
    ///  - `1` if `k == i, l == j`
    ///  - `1` if `k == j, l == i`
    ///  - `0` otherwise
    /// 
    fn swap<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize) {
        self.transform(ring, i, j, &[ring.zero(), ring.one(), ring.one(), ring.zero()])
    }
}

///
/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the left, i.e. applied to the rows of the matrix.
/// 
pub struct TransformRows<'a, V, R>(pub SubmatrixMut<'a, V, R::Element>, pub &'a R)
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase;

///
/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the right, i.e. applied to the cols of the matrix.
/// 
pub struct TransformCols<'a, V, R>(pub SubmatrixMut<'a, V, R::Element>, pub &'a R)
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase;

impl<'a, V, R> TransformTarget<R> for TransformRows<'a, V, R>
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let A = &mut self.0;
        for l in 0..A.col_count() {
            let (new_i, new_j) = (
                ring.add(ring.mul_ref(A.at(i, l), &transform[0]), ring.mul_ref(A.at(j, l), &transform[1])),
                ring.add(ring.mul_ref(A.at(i, l), &transform[2]), ring.mul_ref(A.at(j, l), &transform[3]))
            );
            *A.at_mut(i, l) = new_i;
            *A.at_mut(j, l) = new_j;
        }
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &<R as RingBase>::Element) {
        let A = &mut self.0;
        for j in 0..A.col_count() {
            let to_sub = ring.mul_ref(factor, A.at(src, j));
            ring.sub_assign(A.at_mut(dst, j), to_sub);
        }
    }
}

impl<'a, V, R> TransformTarget<R> for TransformCols<'a, V, R>
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let A = &mut self.0;
        for l in 0..A.row_count() {
            let (new_i, new_j) = (
                ring.add(ring.mul_ref(A.at(l, i), &transform[0]), ring.mul_ref(A.at(l, j), &transform[1])),
                ring.add(ring.mul_ref(A.at(l, i), &transform[2]), ring.mul_ref(A.at(l, j), &transform[3]))
            );
            *A.at_mut(l, i) = new_i;
            *A.at_mut(l, j) = new_j;
        }
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &<R as RingBase>::Element) {
        let A = &mut self.0;
        for i in 0..A.row_count() {
            let to_sub = ring.mul_ref(factor, A.at(i, src));
            ring.sub_assign(A.at_mut(i, dst), to_sub);
        }
    }
}
