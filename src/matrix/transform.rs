use std::marker::PhantomData;

use tracing::instrument;

use super::{AsPointerToSlice, OwnedMatrix, SubmatrixMut};
use crate::prelude::*;

/// A trait for a "target" that can "consume" elementary operations on matrices.
///  
/// This is mainly used during algorithms that work on matrices, since in many cases
/// they transform matrices using elementary row or column operations, and have to
/// accumulate data depending on these operations.
///
/// # Left- and right-transforms
///
/// In most aspects, a [`TransformTarget`] is just fed a matrix via its factorization.
/// However, there is a difference if the factors are accumulated from the left or the
/// right. Concretely
///  - If an algorithm feeds a [`TransformTarget`] a left-transform, the transform target receives
///    elementary matrices `L1, ..., Lr`, and the complete transform is then left-multiplication by
///    the matrix `L = Lr ... L1` (notice the reversal coming from the left-multiplication).
///  - If an algorithm feeds a [`TransformTarget`] a right-transform, the transform target receives
///    elementary matrices `R1, ..., Rr`, and the complete transform is then right-multiplication by
///    the matrix `R = R1^T ... Rr^T` (notice the transpose).
pub trait TransformTarget<R>
where
    R: ?Sized + RingBase,
{
    /// The transformation corresponds to replacing two rows (in the context of a left-transform)
    /// resp. two columns (in the context of a right-transform) by linear combinations of them.
    ///
    /// Concretely, in the context of a left-transform, the transformation given by the matrix `A`
    /// with `A[k, l]` being
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
    /// |    a             b              | <- i-th row
    /// |            1  ...  0            |
    /// |            ⋮        ⋮            |
    /// |            0  ...  1            |
    /// |    c             c              | <- j-th row
    /// |                       1  ...  0 |
    /// |                       ⋮        ⋮ |
    /// |                       0  ...  1 |
    ///      ^ i-th col    ^ j-th col
    /// ```
    /// where `transform = [a, b, c, d]`.
    fn transform<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]);

    /// The transformation corresponding to subtracting `factor` times the `src`-th row
    /// (in the context of a left-transform) resp. col from the `dst`-th row resp. col
    /// (in the context of a right transform).
    ///
    /// More precisely, in the context of a left-transform, the `(k, l)`-th entry of the transform
    /// matrix is defined to be
    ///  - `1` if `k == l`
    ///  - `-factor` if `k == dst, l == src`
    ///  - `0` otherwise
    fn subtract<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        self.transform(
            ring,
            src,
            dst,
            &[ring.one(), ring.zero(), ring.negate(factor.clone()), ring.one()],
        )
    }

    /// The transformation corresponding to the permutation matrix swapping `i`-th and `j`-th row
    /// (in the context of a left-transform) resp. column (in the context of a right transform).
    ///
    /// More precisely, the `(k, l)`-th entry of the transform matrix is defined to be
    ///  - `1` if `k == l, k != i, k != j`
    ///  - `1` if `k == i, l == j`
    ///  - `1` if `k == j, l == i`
    ///  - `0` otherwise
    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        self.transform(ring, i, j, &[ring.zero(), ring.one(), ring.one(), ring.zero()])
    }
}

impl<'a, T, R> TransformTarget<R> for &'a mut T
where
    R: ?Sized + RingBase,
    T: TransformTarget<R>,
{
    fn transform<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T as TransformTarget<R>>::transform(*self, ring, i, j, transform)
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T as TransformTarget<R>>::subtract(*self, ring, src, dst, factor)
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T as TransformTarget<R>>::swap(*self, ring, i, j)
    }
}

/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the left, i.e. applied to the rows of the matrix.
pub struct TransformRows<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    target: SubmatrixMut<'a, V, R::Element>,
}

/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the right, i.e. applied to the cols of the matrix.
pub struct TransformCols<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    target: SubmatrixMut<'a, V, R::Element>,
}

impl<'a, V, R> TransformRows<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    pub fn new(target: SubmatrixMut<'a, V, R::Element>) -> Self { Self { target } }
}

impl<'a, V, R> TransformCols<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    pub fn new(target: SubmatrixMut<'a, V, R::Element>) -> Self { Self { target } }
}

impl<'a, V, R> TransformTarget<R> for TransformRows<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    #[instrument(skip_all, level = "trace")]
    fn transform<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        i: usize,
        j: usize,
        transform: &[<R as RingBase>::Element; 4],
    ) {
        let A = &mut self.target;
        for l in 0..A.col_count() {
            let (new_i, new_j) = (
                ring.add(
                    ring.mul_ref(A.at(i, l), &transform[0]),
                    ring.mul_ref(A.at(j, l), &transform[1]),
                ),
                ring.add(
                    ring.mul_ref(A.at(i, l), &transform[2]),
                    ring.mul_ref(A.at(j, l), &transform[3]),
                ),
            );
            *A.at_mut(i, l) = new_i;
            *A.at_mut(j, l) = new_j;
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn subtract<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        src: usize,
        dst: usize,
        factor: &<R as RingBase>::Element,
    ) {
        let A = &mut self.target;
        for j in 0..A.col_count() {
            let to_sub = ring.mul_ref(factor, A.at(src, j));
            ring.sub_assign(A.at_mut(dst, j), to_sub);
        }
    }
}

impl<'a, V, R> TransformTarget<R> for TransformCols<'a, V, R>
where
    V: AsPointerToSlice<R::Element>,
    R: ?Sized + RingBase,
{
    #[instrument(skip_all, level = "trace")]
    fn transform<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        i: usize,
        j: usize,
        transform: &[<R as RingBase>::Element; 4],
    ) {
        let A = &mut self.target;
        for l in 0..A.row_count() {
            let (new_i, new_j) = (
                ring.add(
                    ring.mul_ref(A.at(l, i), &transform[0]),
                    ring.mul_ref(A.at(l, j), &transform[1]),
                ),
                ring.add(
                    ring.mul_ref(A.at(l, i), &transform[2]),
                    ring.mul_ref(A.at(l, j), &transform[3]),
                ),
            );
            *A.at_mut(l, i) = new_i;
            *A.at_mut(l, j) = new_j;
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn subtract<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        src: usize,
        dst: usize,
        factor: &<R as RingBase>::Element,
    ) {
        let A = &mut self.target;
        for i in 0..A.row_count() {
            let to_sub = ring.mul_ref(factor, A.at(i, src));
            ring.sub_assign(A.at_mut(i, dst), to_sub);
        }
    }
}

pub struct InvertTransform<T> {
    delegate_inverted: T,
}

impl<T> InvertTransform<T> {
    pub fn new(delegate_inverted: T) -> Self { Self { delegate_inverted } }
}

impl<R, T> TransformTarget<R> for InvertTransform<T>
where
    R: ?Sized + RingBase + DivisibilityRing,
    T: TransformTarget<R>,
{
    fn transform<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        i: usize,
        j: usize,
        transform: &[<R as RingBase>::Element; 4],
    ) {
        let det = ring.sub(
            ring.mul_ref(&transform[0], &transform[3]),
            ring.mul_ref(&transform[1], &transform[2]),
        );
        assert!(
            ring.is_unit(&det),
            "InvertTransform requires that all elemetary transforms passed to the target are invertible"
        );
        let det_inv = ring.invert(&det).unwrap();
        let inv_transform = [
            ring.mul_ref(&transform[3], &det_inv),
            ring.negate(ring.mul_ref(&transform[1], &det_inv)),
            ring.negate(ring.mul_ref(&transform[2], &det_inv)),
            ring.mul_ref(&transform[0], &det_inv),
        ];
        self.delegate_inverted.transform(ring, i, j, &inv_transform);
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        src: usize,
        dst: usize,
        factor: &<R as RingBase>::Element,
    ) {
        self.delegate_inverted
            .subtract(ring, src, dst, &ring.negate(factor.clone()));
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        self.delegate_inverted.swap(ring, i, j)
    }
}

pub struct TransposeTransform<T> {
    delegate_transposed: T,
}

impl<T> TransposeTransform<T> {
    pub fn new(delegate_transposed: T) -> Self { Self { delegate_transposed } }
}

impl<R, T> TransformTarget<R> for TransposeTransform<T>
where
    R: ?Sized + RingBase + DivisibilityRing,
    T: TransformTarget<R>,
{
    fn transform<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        i: usize,
        j: usize,
        transform: &[<R as RingBase>::Element; 4],
    ) {
        let transposed_transform = [
            transform[0].clone(),
            transform[2].clone(),
            transform[1].clone(),
            transform[3].clone(),
        ];
        self.delegate_transposed.transform(ring, i, j, &transposed_transform);
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(
        &mut self,
        ring: S,
        src: usize,
        dst: usize,
        factor: &<R as RingBase>::Element,
    ) {
        self.delegate_transposed.subtract(ring, dst, src, factor);
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        self.delegate_transposed.swap(ring, i, j)
    }
}

enum Transform<R>
where
    R: ?Sized + RingBase,
{
    General(usize, usize, [R::Element; 4]),
    Subtract(usize, usize, R::Element),
    Swap(usize, usize),
}

#[stability::unstable(feature = "enable")]
pub struct TransformList<R>
where
    R: ?Sized + RingBase,
{
    transforms: Vec<Transform<R>>,
    row_count: usize,
}

impl<R> TransformList<R>
where
    R: ?Sized + RingBase,
{
    #[stability::unstable(feature = "enable")]
    pub fn new(row_count: usize) -> Self {
        Self {
            row_count,
            transforms: Vec::new(),
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn replay<S: Copy + RingStore<Ring = R>, T: TransformTarget<R>>(&self, ring: S, mut target: T) {
        for transform in &self.transforms {
            match transform {
                Transform::General(i, j, matrix) => target.transform(ring, *i, *j, matrix),
                Transform::Subtract(src, dst, factor) => target.subtract(ring, *src, *dst, factor),
                Transform::Swap(i, j) => target.swap(ring, *i, *j),
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn replay_reversed<S: Copy + RingStore<Ring = R>, T: TransformTarget<R>>(&self, ring: S, mut target: T) {
        for transform in self.transforms.iter().rev() {
            match transform {
                Transform::General(i, j, matrix) => target.transform(ring, *i, *j, matrix),
                Transform::Subtract(src, dst, factor) => target.subtract(ring, *src, *dst, factor),
                Transform::Swap(i, j) => target.swap(ring, *i, *j),
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn to_matrix<S: Copy + RingStore<Ring = R>>(&self, ring: S) -> OwnedMatrix<R::Element> {
        let mut result = OwnedMatrix::identity(self.row_count, self.row_count, ring);
        self.replay(ring, TransformRows::new(result.data_mut()));
        return result;
    }
}

impl<R> TransformTarget<R> for TransformList<R>
where
    R: ?Sized + RingBase,
{
    fn transform<S: Copy + RingStore<Ring = R>>(
        &mut self,
        _ring: S,
        i: usize,
        j: usize,
        transform: &[<R as RingBase>::Element; 4],
    ) {
        debug_assert!(i < self.row_count);
        debug_assert!(j < self.row_count);
        self.transforms
            .push(Transform::General(i, j, std::array::from_fn(|k| transform[k].clone())))
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(
        &mut self,
        _ring: S,
        src: usize,
        dst: usize,
        factor: &<R as RingBase>::Element,
    ) {
        debug_assert!(src < self.row_count);
        debug_assert!(dst < self.row_count);
        self.transforms.push(Transform::Subtract(src, dst, factor.clone()))
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, _ring: S, i: usize, j: usize) {
        debug_assert!(i < self.row_count);
        debug_assert!(j < self.row_count);
        self.transforms.push(Transform::Swap(i, j))
    }
}

impl<R> TransformTarget<R> for ()
where
    R: ?Sized + RingBase,
{
    fn transform<S: Copy + RingStore<Ring = R>>(&mut self, _: S, _: usize, _: usize, _: &[R::Element; 4]) {}

    fn subtract<S: Copy + RingStore<Ring = R>>(&mut self, _: S, _: usize, _: usize, _: &R::Element) {}

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, _: S, _: usize, _: usize) {}
}

/// A [`TransformTarget`] that forwards all transforms to a fixed
/// delegate, but offsets every row/column index by a given value.
pub struct OffsetTransformIndex<R, T>
where
    R: ?Sized + RingBase,
    T: TransformTarget<R>,
{
    delegate: T,
    index_offset: usize,
    ring: PhantomData<R>,
}

impl<R, T> OffsetTransformIndex<R, T>
where
    R: ?Sized + RingBase,
    T: TransformTarget<R>,
{
    /// Creates a new [`OffsetTransformIndex`] that forwards all transforms to `delegate`.
    pub fn new(delegate: T, offset: usize) -> Self {
        Self {
            delegate,
            index_offset: offset,
            ring: PhantomData,
        }
    }
}

impl<R, T> TransformTarget<R> for OffsetTransformIndex<R, T>
where
    R: ?Sized + RingBase,
    T: TransformTarget<R>,
{
    fn transform<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T as TransformTarget<R>>::transform(
            &mut self.delegate,
            ring,
            i + self.index_offset,
            j + self.index_offset,
            transform,
        );
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T as TransformTarget<R>>::subtract(
            &mut self.delegate,
            ring,
            src + self.index_offset,
            dst + self.index_offset,
            factor,
        );
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T as TransformTarget<R>>::swap(&mut self.delegate, ring, i + self.index_offset, j + self.index_offset);
    }
}

/// A [`TransformTarget`] that forwards all transforms to
/// two fixed delegates.
pub struct DuplicateTransforms<R, T1, T2>
where
    R: ?Sized + RingBase,
    T1: TransformTarget<R>,
    T2: TransformTarget<R>,
{
    delegate1: T1,
    delegate2: T2,
    ring: PhantomData<R>,
}

impl<R, T1, T2> DuplicateTransforms<R, T1, T2>
where
    R: ?Sized + RingBase,
    T1: TransformTarget<R>,
    T2: TransformTarget<R>,
{
    /// Creates a new [`DuplicateTransforms`] that forwards all transforms to `first` and `second`.
    pub fn new(first: T1, second: T2) -> Self {
        Self {
            delegate1: first,
            delegate2: second,
            ring: PhantomData,
        }
    }
}

impl<R, T1, T2> TransformTarget<R> for DuplicateTransforms<R, T1, T2>
where
    R: ?Sized + RingBase,
    T1: TransformTarget<R>,
    T2: TransformTarget<R>,
{
    fn transform<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T1 as TransformTarget<R>>::transform(&mut self.delegate1, ring, i, j, transform);
        <T2 as TransformTarget<R>>::transform(&mut self.delegate2, ring, i, j, transform);
    }

    fn subtract<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T1 as TransformTarget<R>>::subtract(&mut self.delegate1, ring, src, dst, factor);
        <T2 as TransformTarget<R>>::subtract(&mut self.delegate2, ring, src, dst, factor);
    }

    fn swap<S: Copy + RingStore<Ring = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T1 as TransformTarget<R>>::swap(&mut self.delegate1, ring, i, j);
        <T2 as TransformTarget<R>>::swap(&mut self.delegate2, ring, i, j);
    }
}
