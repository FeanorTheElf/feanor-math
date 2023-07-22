use std::{marker::PhantomData, cmp::{min, max}};

use super::*;
use crate::vector::subvector::*;

pub struct Chain<V1, V2, T>
    where V1: VectorView<T>, V2: VectorView<T>
{
    el: PhantomData<T>,
    first: V1,
    second: V2
}

impl<V1, V2, T> Chain<V1, V2, T>
    where V1: VectorView<T>, V2: VectorView<T>
{
    pub fn new(first: V1, second: V2) -> Self {
        Chain { 
            el: PhantomData, 
            first: first, 
            second: second
        }
    }
}

impl<V1, V2, T> VectorView<T> for Chain<V1, V2, T>
    where V1: VectorView<T>, V2: VectorView<T>
{
    fn len(&self) -> usize {
        self.first.len() + self.second.len()
    }

    fn at(&self, i: usize) -> &T {
        if i >= self.first.len() {
            self.second.at(i - self.first.len())
        } else {
            self.first.at(i)
        }
    }
}

impl<V1, V2, T> VectorViewMut<T> for Chain<V1, V2, T>
    where V1: VectorViewMut<T>, V2: VectorViewMut<T>
{
    fn at_mut(&mut self, i: usize) -> &mut T {
        if i >= self.first.len() {
            self.second.at_mut(i - self.first.len())
        } else {
            self.first.at_mut(i)
        }
    }
}

impl<V1, V2, T> SwappableVectorViewMut<T> for Chain<V1, V2, T>
    where V1: SwappableVectorViewMut<T>, V2: SwappableVectorViewMut<T>
{
    fn swap(&mut self, i: usize, j: usize) {
        if i >= self.first.len() && j >= self.first.len() {
            self.second.swap(i - self.first.len(), j - self.first.len())
        } else if i < self.first.len() && j < self.first.len() {
            self.first.swap(i, j);
        } else {
            std::mem::swap(self.first.at_mut(min(i, j)), self.second.at_mut(max(i, j)));
        }
    }
}

impl<V1, V2, T> SelfSubvectorView<T> for Chain<V1, V2, T>
    where V1: SelfSubvectorView<T>, V2: SelfSubvectorView<T>
{
    fn subvector<R: std::ops::RangeBounds<usize>>(self, range: R) -> Self {
        let start = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(n) => *n + 1,
            std::ops::Bound::Included(n) => *n
        };
        let end = match range.end_bound() {
            std::ops::Bound::Excluded(n) => *n,
            std::ops::Bound::Included(n) => *n + 1,
            std::ops::Bound::Unbounded => self.len()
        };
        assert!(start <= end);
        Chain { 
            el: PhantomData, 
            second: self.second.subvector((min(self.first.len(), start) - self.first.len())..(min(self.first.len(), end) - self.first.len())),
            first: self.first.subvector(range)
        }
    }
}

