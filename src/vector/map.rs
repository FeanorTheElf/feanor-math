use std::marker::PhantomData;

use crate::vector::*;

use super::{vec_fn::VectorFn, subvector::{SelfSubvectorFn, SelfSubvectorView}};

pub struct Map<V, F, T> 
    where V: VectorView<T>
{
    base_view: V,
    accessor: F,
    base_element: PhantomData<T>
}

impl<V, F, T, U> Map<V, F, T>
    where V: VectorView<T>, F: Fn(&T) -> &U
{
    pub const fn new(base_view: V, accessor: F) -> Self {
        Map {
            base_view: base_view,
            accessor: accessor,
            base_element: PhantomData
        }
    }
}

impl<V, F, T, U> Clone for Map<V, F, T>
    where V: VectorView<T> + Clone, F: Clone + Fn(&T) -> &U
{
    fn clone(&self) -> Self {
        Map::new(self.base_view.clone(), self.accessor.clone())
    }
}

impl<V, F, T, U> Copy for Map<V, F, T>
    where V: VectorView<T> + Copy, F: Copy + Fn(&T) -> &U
{}

impl<V, F, T, U> VectorView<U> for Map<V, F, T>
    where V: VectorView<T>, F: Fn(&T) -> &U
{
    fn at(&self, i: usize) -> &U {
        (self.accessor)(self.base_view.at(i))
    }

    fn len(&self) -> usize {
        self.base_view.len()
    }
}

impl<V, F, T, U> SelfSubvectorView<U> for Map<V, F, T>
    where V: SelfSubvectorView<T>, F: Fn(&T) -> &U
{
    fn subvector<R: std::ops::RangeBounds<usize>>(self, range: R) -> Self {
        Map { base_view: self.base_view.subvector(range), accessor: self.accessor, base_element: self.base_element }
    }
}

pub struct MapMut<V, F, G, T> 
    where V: VectorView<T>
{
    base_view: V,
    accessor: F,
    accessor_mut: G,
    base_element: PhantomData<T>
}

impl<V, F, G, T, U> MapMut<V, F, G, T>
    where V: VectorView<T>, F: Fn(&T) -> &U, G: Fn(&mut T) -> &mut U
{
    pub const fn new(base_view: V, accessor: F, accessor_mut: G) -> Self {
        MapMut {
            base_view: base_view,
            accessor: accessor,
            accessor_mut: accessor_mut,
            base_element: PhantomData
        }
    }
}

impl<V, F, G, T, U> VectorView<U> for MapMut<V, F, G, T>
    where V: VectorView<T>, F: Fn(&T) -> &U, G: Fn(&mut T) -> &mut U
{
    fn at(&self, i: usize) -> &U {
        (self.accessor)(self.base_view.at(i))
    }

    fn len(&self) -> usize {
        self.base_view.len()
    }
}

impl<V, F, G, T, U> VectorViewMut<U> for MapMut<V, F, G, T>
    where V: VectorViewMut<T>, F: Fn(&T) -> &U, G: Fn(&mut T) -> &mut U
{
    fn at_mut(&mut self, i: usize) -> &mut U {
        (self.accessor_mut)(self.base_view.at_mut(i))
    }
}

impl<V, F, G, T, U> SwappableVectorViewMut<U> for MapMut<V, F, G, T>
    where V: SwappableVectorViewMut<T>, F: Fn(&T) -> &U, G: Fn(&mut T) -> &mut U
{
    fn swap(&mut self, i: usize, j: usize) {
        self.base_view.swap(i, j)
    }
}

impl<V, F, G, T, U> Clone for MapMut<V, F, G, T>
    where V: VectorViewMut<T> + Clone, F: Clone + Fn(&T) -> &U, G: Clone + Fn(&mut T) -> &mut U
{
    fn clone(&self) -> Self {
        MapMut::new(self.base_view.clone(), self.accessor.clone(), self.accessor_mut.clone())
    }
}

impl<V, F, G, T, U> Copy for MapMut<V, F, G, T>
    where V: VectorViewMut<T> + Copy, F: Copy + Fn(&T) -> &U, G: Copy + Fn(&mut T) -> &mut U
{}

pub struct MapFn<V, F, T> 
    where V: VectorFn<T>
{
    base_view: V,
    accessor: F,
    base_element: PhantomData<T>
}

impl<V, F, T, U> MapFn<V, F, T>
    where V: VectorFn<T>, F: Fn(T) -> U
{
    pub const fn new(base_view: V, accessor: F) -> Self {
        MapFn {
            base_view: base_view,
            accessor: accessor,
            base_element: PhantomData
        }
    }
}

impl<V, F, T, U> Clone for MapFn<V, F, T>
    where V: VectorFn<T> + Clone, F: Clone + Fn(T) -> U
{
    fn clone(&self) -> Self {
        MapFn::new(self.base_view.clone(), self.accessor.clone())
    }
}

impl<V, F, T, U> Copy for MapFn<V, F, T>
    where V: VectorFn<T> + Copy, F: Copy + Fn(T) -> U
{}

impl<V, F, T, U> VectorFn<U> for MapFn<V, F, T>
    where V: VectorFn<T>, F: Fn(T) -> U
{
    fn at(&self, i: usize) -> U {
        (self.accessor)(self.base_view.at(i))
    }

    fn len(&self) -> usize {
        self.base_view.len()
    }
}

impl<V, F, T, U> SelfSubvectorFn<U> for MapFn<V, F, T>
    where V: SelfSubvectorFn<T>, F: Fn(T) -> U
{
    fn subvector<R: std::ops::RangeBounds<usize>>(self, range: R) -> Self {
        MapFn { base_view: self.base_view.subvector(range), accessor: self.accessor, base_element: self.base_element }
    }
}
