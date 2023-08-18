use super::*;
use super::vec_fn::VectorFn;

use std::ops::{RangeBounds, Bound, Index, IndexMut};
use std::marker::PhantomData;

///
/// Trait for all vector views that "shrink" the section of the vector
/// that is represented, without changing the type. This is mainly important
/// in combination with recursive algorithms.
/// 
/// # Example
/// 
/// Assuming we wanted to implement a (very stupid) recursive variant of summing
/// all values in a vector. We would like to do it as
/// ```ignore
/// # use feanor_math::vector::*;
/// # use feanor_math::vector::subvector::*;
/// 
/// // Compiler error: overflow evaluating the requirement `&Vec<i64>: Sized
/// fn sum<V: VectorView<i64> + Copy>(vector: V) -> i64 {
///     if vector.len() == 0 { 0 } else { sum(Subvector::new(vector).subvector(1..)) + *vector.at(0) }
/// }
/// 
/// assert_eq!(7, sum(&[1, 1, 1, 1, 1, 1, 1][..]));
/// ```
/// but this clearly cannot work - this can never be monomorphized.
/// Instead, use
/// ```
/// # use feanor_math::vector::*;
/// # use feanor_math::vector::subvector::*;
/// 
/// // This works!
/// fn sum<V: SelfSubvectorView<i64> + Copy>(vector: V) -> i64 {
///     if vector.len() == 0 { 0 } else { sum(vector.subvector(1..)) + *vector.at(0) }
/// }
/// 
/// assert_eq!(7, sum(&[1, 1, 1, 1, 1, 1, 1][..]));
/// ```
/// 
/// ## The mutable case
/// 
/// In the mutable case, this is much more difficult, as we cannot have vectors that are `Copy`.
/// Simple examples still work:
/// ```
/// # use feanor_math::vector::*;
/// # use feanor_math::vector::subvector::*;
/// 
/// fn inc<V: VectorViewMut<i64> + SelfSubvectorView<i64>>(mut vector: V) {
///     if vector.len() > 0 { 
///         *vector.at_mut(0) += 1;
///         inc(vector.subvector(1..));
///     }
/// }
/// 
/// let mut data = [1, 1, 1, 0, 0, 0];
/// inc(Subvector::new(&mut data));
/// assert_eq!([2, 2, 2, 1, 1, 1], data);
/// ```
/// But it is a problem that [`SelfSubvectorView::subvector()`] moves the current object.
/// In particular, the following does not work:
/// ```ignore
/// # use feanor_math::vector::*;
/// # use feanor_math::vector::subvector::*;
/// 
/// fn inc<V: VectorViewMut<i64> + SelfSubvectorView<i64>>(mut vector: V) {
///     if vector.len() > 0 { 
///         inc(vector.subvector(1..));
///         *vector.at_mut(0) += 1;
///     }
/// }
/// 
/// let mut data = [1, 1, 1, 0, 0, 0];
/// inc(Subvector::new(&mut data));
/// assert_eq!([2, 2, 2, 1, 1, 1], data);
/// ```
/// Currently, there is no solution implemented for this, as in most cases mutable
/// slices do the job. However, it might be solved with a `BorrowableMut`-trait in
/// the future.
/// 
pub trait SelfSubvectorView<T: ?Sized>: VectorView<T> {

    fn subvector<R: RangeBounds<usize>>(self, range: R) -> Self;
}

impl<'a, T> SelfSubvectorView<T> for &'a [T] {

    fn subvector<R: RangeBounds<usize>>(self, range: R) -> Self {
        self.index((range.start_bound().cloned(), range.end_bound().cloned()))
    }
}

impl<'a, T> SelfSubvectorView<T> for &'a mut [T] {

    fn subvector<R: RangeBounds<usize>>(self, range: R) -> Self {
        self.index_mut((range.start_bound().cloned(), range.end_bound().cloned()))
    }
}

///
/// A view on a part of another vector view.
/// 
pub struct Subvector<T: ?Sized, V> 
    where V: VectorView<T>
{
    from: usize,
    to: usize,
    base: V,
    element: PhantomData<T>
}

impl<T: ?Sized, V> Subvector<T, V>
    where V: VectorView<T>
{
    pub fn new(base: V) -> Self {
        Subvector { 
            from: 0, 
            to: base.len(), 
            base: base, 
            element: PhantomData 
        }
    }
}

impl<T: ?Sized, V> VectorView<T> for Subvector<T, V>
    where V: VectorView<T>
{
    fn len(&self) -> usize {
        self.to - self.from
    }

    fn at(&self, i: usize) -> &T {
        debug_assert!(i < self.len());
        self.base.at(i + self.from)
    }
}

impl<T: ?Sized, V> VectorViewMut<T> for Subvector<T, V>
    where V: VectorViewMut<T>
{
    fn at_mut(&mut self, i: usize) -> &mut T {
        debug_assert!(i < self.len());
        self.base.at_mut(i + self.from)
    }
}

impl<T, V> SwappableVectorViewMut<T> for Subvector<T, V>
    where V: SwappableVectorViewMut<T>
{
    fn swap(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.len());
        debug_assert!(j < self.len());
        self.base.swap(i + self.from, j + self.from);
    }
}

impl<T: ?Sized, V> SelfSubvectorView<T> for Subvector<T, V>
    where V: VectorView<T>
{
    fn subvector<R: RangeBounds<usize>>(mut self, range: R) -> Self {
        let from = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x + 1,
            Bound::Unbounded => 0
        };
        assert!(from <= self.len());
        let to = match range.end_bound() {
            Bound::Included(x) => *x - 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len()
        };
        assert!(to <= self.len());
        assert!(to >= from);
        self.to = to + self.from;
        self.from = from + self.from;
        return self;
    }
}

impl<T: ?Sized, V> Copy for Subvector<T, V>
    where V: VectorView<T> + Copy
{}

impl<T: ?Sized, V> Clone for Subvector<T, V>
    where V: VectorView<T> + Clone
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            from: self.from,
            to: self.to,
            element: PhantomData
        }
    }
}
pub trait SelfSubvectorFn<T>: VectorFn<T> {

    fn subvector<R: RangeBounds<usize>>(self, range: R) -> Self;
}

pub struct SubvectorFn<T, V> 
    where V: VectorFn<T>
{
    from: usize,
    to: usize,
    base: V,
    element: PhantomData<T>
}

impl<T, V> SubvectorFn<T, V>
    where V: VectorFn<T>
{
    pub fn new(base: V) -> Self {
        SubvectorFn { 
            from: 0, 
            to: base.len(), 
            base: base, 
            element: PhantomData 
        }
    }
}

impl<T, V> VectorFn<T> for SubvectorFn<T, V>
    where V: VectorFn<T>
{
    fn len(&self) -> usize {
        self.to - self.from
    }

    fn at(&self, i: usize) -> T {
        debug_assert!(i < self.len());
        self.base.at(i + self.from)
    }
}

impl<T, V> SelfSubvectorFn<T> for SubvectorFn<T, V>
    where V: VectorFn<T>
{
    fn subvector<R: RangeBounds<usize>>(mut self, range: R) -> Self {
        let from = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x + 1,
            Bound::Unbounded => 0
        };
        assert!(from <= self.len());
        let to = match range.end_bound() {
            Bound::Included(x) => *x - 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len()
        };
        assert!(to <= self.len());
        assert!(to >= from);
        self.to = to + self.from;
        self.from = from + self.from;
        return self;
    }
}

impl<T, V> Copy for SubvectorFn<T, V>
    where V: VectorFn<T> + Copy
{}

impl<T, V> Clone for SubvectorFn<T, V>
    where V: VectorFn<T> + Clone
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            from: self.from,
            to: self.to,
            element: PhantomData
        }
    }
}