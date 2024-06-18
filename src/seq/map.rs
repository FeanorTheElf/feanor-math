use std::marker::PhantomData;

use super::{SwappableVectorViewMut, VectorFn, VectorView, VectorViewMut};

pub struct VectorViewMap<V: VectorView<T>, T: ?Sized, U: ?Sized, F: for<'a> Fn(&'a T) -> &'a U> {
    base: V,
    mapping_fn: F,
    elements: PhantomData<(*const T, *const U)>
}

impl<V: Clone + VectorView<T>, T: ?Sized, U: ?Sized, F: Clone + for<'a> Fn(&'a T) -> &'a U> Clone for VectorViewMap<V, T, U, F> {
    
    fn clone(&self) -> Self {
        Self {
            mapping_fn: self.mapping_fn.clone(),
            base: self.base.clone(),
            elements: PhantomData
        }
    }
}

impl<V: Copy + VectorView<T>, T: ?Sized, U: ?Sized, F: Copy + for<'a> Fn(&'a T) -> &'a U> Copy for VectorViewMap<V, T, U, F> {}

impl<V: VectorView<T>, T: ?Sized, U: ?Sized, F: for<'a> Fn(&'a T) -> &'a U> VectorViewMap<V, T, U, F> {

    pub fn new(base: V, mapping_fn: F) -> Self {
        Self {
            base: base,
            mapping_fn: mapping_fn,
            elements: PhantomData
        }
    }
}

impl<V: VectorView<T>, T: ?Sized, U: ?Sized, F: for<'a> Fn(&'a T) -> &'a U> VectorView<U> for VectorViewMap<V, T, U, F> {

    fn at(&self, i: usize) -> &U {
        (self.mapping_fn)(self.base.at(i))
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}

pub struct VectorViewMapMut<V: VectorViewMut<T>, T: ?Sized, U: ?Sized, F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U> {
    base: V,
    mapping_fns: (F_const, F_mut),
    elements: PhantomData<(*const T, *const U)>
}

impl<V: VectorViewMut<T>, T: ?Sized, U: ?Sized, F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U> VectorViewMapMut<V, T, U, F_const, F_mut> {

    pub fn new(base: V, mapping_fns: (F_const, F_mut)) -> Self {
        Self {
            base: base,
            mapping_fns: mapping_fns,
            elements: PhantomData
        }
    }
}

impl<V: VectorViewMut<T>, T: ?Sized, U: ?Sized, F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U> VectorView<U> for VectorViewMapMut<V, T, U, F_const, F_mut> {

    fn at(&self, i: usize) -> &U {
        (self.mapping_fns.0)(self.base.at(i))
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}

impl<V: VectorViewMut<T>, T: ?Sized, U: ?Sized, F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U> VectorViewMut<U> for VectorViewMapMut<V, T, U, F_const, F_mut> {

    fn at_mut(&mut self, i: usize) -> &mut U {
        (self.mapping_fns.1)(self.base.at_mut(i))
    }
}

impl<V: SwappableVectorViewMut<T>, T: ?Sized, U: ?Sized, F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U> SwappableVectorViewMut<U> for VectorViewMapMut<V, T, U, F_const, F_mut> {

    fn swap(&mut self, i: usize, j: usize) {
        self.base.swap(i, j)
    }
}

pub struct VectorFnMap<V: VectorFn<T>, T, U, F: Fn(T) -> U> {
    base: V,
    mapping_fn: F,
    elements: PhantomData<(*const T, *const U)>
}

impl<V: Clone + VectorFn<T>, T, U, F: Clone + Fn(T) -> U> Clone for VectorFnMap<V, T, U, F> {
    
    fn clone(&self) -> Self {
        Self {
            mapping_fn: self.mapping_fn.clone(),
            base: self.base.clone(),
            elements: PhantomData
        }
    }
}

impl<V: Copy + VectorFn<T>, T, U, F: Copy + Fn(T) -> U> Copy for VectorFnMap<V, T, U, F> {}

impl<V: VectorFn<T>, T, U, F: Fn(T) -> U> VectorFnMap<V, T, U, F> {
    
    pub fn new(base: V, mapping_fn: F) -> Self {
        Self {
            base: base,
            mapping_fn: mapping_fn,
            elements: PhantomData
        }
    }
}

impl<V: VectorFn<T>, T, U, F: Fn(T) -> U> VectorFn<U> for VectorFnMap<V, T, U, F> {

    fn at(&self, i: usize) -> U {
        (self.mapping_fn)(self.base.at(i))
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}