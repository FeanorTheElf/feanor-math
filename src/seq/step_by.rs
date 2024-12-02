use std::marker::PhantomData;

use super::*;

#[derive(Debug)]
pub struct StepBy<V: VectorView<T>, T: ?Sized> {
    base: V,
    step_by: usize,
    element: PhantomData<T>
}

impl<V: VectorView<T>, T: ?Sized> StepBy<V, T> {

    pub fn new(base: V, step_by: usize) -> Self {
        assert!(step_by > 0);
        Self {
            base: base,
            step_by: step_by,
            element: PhantomData
        }
    }
}

impl<V: Clone + VectorView<T>, T: ?Sized> Clone for StepBy<V, T> {

    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            step_by: self.step_by,
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorView<T>, T: ?Sized> Copy for StepBy<V, T> {}

impl<V: VectorView<T>, T: ?Sized> VectorView<T> for StepBy<V, T> {

    fn len(&self) -> usize {
        if self.base.len() == 0 {
            0
        } else {
            (self.base.len() - 1) / self.step_by + 1
        }
    }

    fn at(&self, i: usize) -> &T {
        self.base.at(i * self.step_by)
    }
}

impl<V: VectorViewMut<T>, T: ?Sized> VectorViewMut<T> for StepBy<V, T> {

    fn at_mut(&mut self, i: usize) -> &mut T {
        self.base.at_mut(i * self.step_by)
    }
}

impl<V: SwappableVectorViewMut<T>, T: ?Sized> SwappableVectorViewMut<T> for StepBy<V, T> {

    fn swap(&mut self, i: usize, j: usize) {
        self.base.swap(i * self.step_by, j * self.step_by)
    }
}

#[derive(Debug)]
pub struct StepByFn<V: VectorFn<T>, T> {
    base: V,
    step_by: usize,
    element: PhantomData<T>
}

impl<V: VectorFn<T>, T> StepByFn<V, T> {

    pub fn new(base: V, step_by: usize) -> Self {
        assert!(step_by > 0);
        Self {
            base: base,
            step_by: step_by,
            element: PhantomData
        }
    }
}

impl<V: Clone + VectorFn<T>, T> Clone for StepByFn<V, T> {

    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            step_by: self.step_by,
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorFn<T>, T> Copy for StepByFn<V, T> {}

impl<V: VectorFn<T>, T> VectorFn<T> for StepByFn<V, T> {

    fn len(&self) -> usize {
        if self.base.len() == 0 {
            0
        } else {
            (self.base.len() - 1) / self.step_by + 1
        }
    }

    fn at(&self, i: usize) -> T {
        self.base.at(i * self.step_by)
    }
}

#[test]
fn test_step_by() {
    let vec = [0, 1, 2, 3, 4, 5, 6, 7];
    let zero: [i32; 0] = [];
    assert_eq!(0, zero.step_by_view(1).len());
    assert_eq!(4, vec.step_by_view(2).len());
    assert_eq!(3, vec.step_by_view(3).len());
    assert_eq!(6, *vec.step_by_view(2).at(3));
    assert_eq!(0, *vec.step_by_view(3).at(0));
    assert_eq!(3, *vec.step_by_view(3).at(1));
}

#[test]
fn test_step_by_fn() {
    let vec = [0, 1, 2, 3, 4, 5, 6, 7].into_copy_els();
    let zero: CloneElFn<[i32; 0], _, _> = [].into_copy_els();
    assert_eq!(0, zero.step_by_fn(1).len());
    assert_eq!(4, vec.step_by_fn(2).len());
    assert_eq!(3, vec.step_by_fn(3).len());
    assert_eq!(6, vec.step_by_fn(2).at(3));
    assert_eq!(0, vec.step_by_fn(3).at(0));
    assert_eq!(3, vec.step_by_fn(3).at(1));
}