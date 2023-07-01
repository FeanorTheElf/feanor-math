use super::*;
use super::vec_fn::VectorFn;

use std::ops::{RangeBounds, Bound, Index, IndexMut};
use std::marker::PhantomData;

pub trait SelfSubvectorView<T>: VectorView<T> {

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

pub struct Subvector<T, V> 
    where V: VectorView<T>
{
    from: usize,
    to: usize,
    base: V,
    element: PhantomData<T>
}

impl<T, V> Subvector<T, V>
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

impl<T, V> VectorView<T> for Subvector<T, V>
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

impl<T, V> VectorViewMut<T> for Subvector<T, V>
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

impl<T, V> SelfSubvectorView<T> for Subvector<T, V>
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

impl<T, V> Copy for Subvector<T, V>
    where V: VectorView<T> + Copy
{}

impl<T, V> Clone for Subvector<T, V>
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