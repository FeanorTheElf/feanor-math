use std::marker::PhantomData;

use crate::ring::*;

use super::{AsPointerToSlice, OwnedMatrix, SubmatrixMut};

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

impl<'a, T, R> TransformTarget<R> for &'a mut T
    where R: ?Sized + RingBase,
        T: TransformTarget<R>
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T as TransformTarget<R>>::transform(*self, ring, i, j, transform)
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T as TransformTarget<R>>::subtract(*self, ring, src, dst, factor)
    }

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T as TransformTarget<R>>::swap(*self, ring, i, j)
    }
}

///
/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the left, i.e. applied to the rows of the matrix.
/// 
/// TODO: at next breaking release, remove the reference to the ring
/// 
pub struct TransformRows<'a, V, R>(pub SubmatrixMut<'a, V, R::Element>, pub &'a R)
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase;

///
/// Wraps a [`SubmatrixMut`] to get a [`TransformTarget`]. Every transform is multiplied to
/// the wrapped matrix from the right, i.e. applied to the cols of the matrix.
/// 
/// TODO: at next breaking release, remove the reference to the ring
/// 
pub struct TransformCols<'a, V, R>(pub SubmatrixMut<'a, V, R::Element>, pub &'a R)
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase;

impl<'a, V, R> TransformTarget<R> for TransformRows<'a, V, R>
    where V: AsPointerToSlice<R::Element>, R: ?Sized + RingBase
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        assert!(ring.get_ring() == self.1);
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
        assert!(ring.get_ring() == self.1);
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
        assert!(ring.get_ring() == self.1);
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
        assert!(ring.get_ring() == self.1);
        let A = &mut self.0;
        for i in 0..A.row_count() {
            let to_sub = ring.mul_ref(factor, A.at(i, src));
            ring.sub_assign(A.at_mut(i, dst), to_sub);
        }
    }
}

enum Transform<R>
    where R: ?Sized + RingBase
{
    General(usize, usize, [R::Element; 4]),
    Subtract(usize, usize, R::Element),
    Swap(usize, usize)
}

#[stability::unstable(feature = "enable")]
pub struct TransformList<R>
    where R: ?Sized + RingBase
{
    transforms: Vec<Transform<R>>,
    row_count: usize
}

impl<R> TransformList<R>
    where R: ?Sized + RingBase
{    
    #[stability::unstable(feature = "enable")]
    pub fn new(row_count: usize) -> Self {
        Self {
            row_count: row_count,
            transforms: Vec::new()
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn replay<S: Copy + RingStore<Type = R>, T: TransformTarget<R>>(&self, ring: S, mut target: T) {
        for transform in &self.transforms {
            match transform {
                Transform::General(i, j, matrix) => target.transform(ring, *i, *j, matrix),
                Transform::Subtract(src, dst, factor) => target.subtract(ring, *src, *dst, factor),
                Transform::Swap(i, j) => target.swap(ring, *i, *j)
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn replay_transposed<S: Copy + RingStore<Type = R>, T: TransformTarget<R>>(&self, ring: S, mut target: T) {
        for transform in self.transforms.iter().rev() {
            match transform {
                Transform::General(i, j, matrix) => {
                    target.transform(ring, *i, *j, &[
                        ring.clone_el(&matrix[0]),
                        ring.clone_el(&matrix[2]),
                        ring.clone_el(&matrix[1]),
                        ring.clone_el(&matrix[3])
                    ])
                },
                Transform::Subtract(src, dst, factor) => target.subtract(ring, *dst, *src, factor),
                Transform::Swap(i, j) => target.swap(ring, *i, *j)
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn to_matrix<S: Copy + RingStore<Type = R>>(&self, ring: S) -> OwnedMatrix<R::Element> {
        let mut result = OwnedMatrix::identity(self.row_count, self.row_count, ring);
        self.replay(ring, TransformRows(result.data_mut(), ring.get_ring()));
        return result;
    }
}

impl<R> TransformTarget<R> for TransformList<R>
    where R: ?Sized + RingBase
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        debug_assert!(i < self.row_count);
        debug_assert!(j < self.row_count);
        self.transforms.push(Transform::General(i, j, std::array::from_fn(|k| ring.clone_el(&transform[k]))))
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &<R as RingBase>::Element) {
        debug_assert!(src < self.row_count);
        debug_assert!(dst < self.row_count);
        self.transforms.push(Transform::Subtract(src, dst, ring.clone_el(factor)))
    }

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, _ring: S, i: usize, j: usize) {
        debug_assert!(i < self.row_count);
        debug_assert!(j < self.row_count);
        self.transforms.push(Transform::Swap(i, j))
    }
}

impl<R> TransformTarget<R> for ()
    where R: ?Sized + RingBase
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, _: S, _: usize, _: usize, _: &[R::Element; 4]) {}

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, _: S, _: usize, _: usize, _: &R::Element) {}

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, _: S, _: usize, _: usize) {}
}

///
/// A [`TransformTarget`] that forwards all transforms to a fixed
/// delegate, but offsets every row/column index by a given value.
/// 
pub struct OffsetTransformIndex<R, T>
    where R: ?Sized + RingBase,
        T: TransformTarget<R>
{
    delegate: T,
    index_offset: usize,
    ring: PhantomData<R>
}

impl<R, T> OffsetTransformIndex<R, T>
    where R: ?Sized + RingBase,
        T: TransformTarget<R>
{
    ///
    /// Creates a new [`OffsetTransformIndex`] that forwards all transforms to `delegate`.
    /// 
    pub fn new(delegate: T, offset: usize) -> Self {
        Self {
            delegate: delegate, 
            index_offset: offset, 
            ring: PhantomData
        }
    }
}

impl<R, T> TransformTarget<R> for OffsetTransformIndex<R, T>
    where R: ?Sized + RingBase,
        T: TransformTarget<R>
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T as TransformTarget<R>>::transform(&mut self.delegate, ring, i + self.index_offset, j + self.index_offset, transform);
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T as TransformTarget<R>>::subtract(&mut self.delegate, ring, src + self.index_offset, dst + self.index_offset, factor);
    }

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T as TransformTarget<R>>::swap(&mut self.delegate, ring, i + self.index_offset, j + self.index_offset);
    }
}

///
/// A [`TransformTarget`] that forwards all transforms to
/// two fixed delegates.
/// 
pub struct DuplicateTransforms<R, T1, T2>
    where R: ?Sized + RingBase,
        T1: TransformTarget<R>,
        T2: TransformTarget<R>
{
    delegate1: T1,
    delegate2: T2,
    ring: PhantomData<R>
}

impl<R, T1, T2> DuplicateTransforms<R, T1, T2>
    where R: ?Sized + RingBase,
        T1: TransformTarget<R>,
        T2: TransformTarget<R>
{
    ///
    /// Creates a new [`DuplicateTransforms`] that forwards all transforms to `first` and `second`.
    /// 
    pub fn new(first: T1, second: T2) -> Self {
        Self {
            delegate1: first,
            delegate2: second, 
            ring: PhantomData
        }
    }
}

impl<R, T1, T2> TransformTarget<R> for DuplicateTransforms<R, T1, T2>
    where R: ?Sized + RingBase,
        T1: TransformTarget<R>,
        T2: TransformTarget<R>
{
    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[R::Element; 4]) {
        <T1 as TransformTarget<R>>::transform(&mut self.delegate1, ring, i, j, transform);
        <T2 as TransformTarget<R>>::transform(&mut self.delegate2, ring, i, j, transform);
    }

    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, ring: S, src: usize, dst: usize, factor: &R::Element) {
        <T1 as TransformTarget<R>>::subtract(&mut self.delegate1, ring, src, dst, factor);
        <T2 as TransformTarget<R>>::subtract(&mut self.delegate2, ring, src, dst, factor);
    }

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize) {
        <T1 as TransformTarget<R>>::swap(&mut self.delegate1, ring, i, j);
        <T2 as TransformTarget<R>>::swap(&mut self.delegate2, ring, i, j);
    }
}
