use std::{marker::PhantomData, ops::{RangeBounds, Bound}};

use crate::vector::*;

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
