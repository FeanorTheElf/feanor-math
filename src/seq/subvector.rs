use std::marker::PhantomData;

use super::{SelfSubvectorFn, SelfSubvectorView, SwappableVectorViewMut, VectorFn, VectorView, VectorViewMut};

pub struct SubvectorView<V: VectorView<T>, T: ?Sized> {
    begin: usize,
    end: usize,
    base: V,
    element: PhantomData<T>
}

impl<V: Clone + VectorView<T>, T: ?Sized> Clone for SubvectorView<V, T> {
    
    fn clone(&self) -> Self {
        Self {
            begin: self.begin,
            end: self.end,
            base: self.base.clone(),
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorView<T>, T: ?Sized> Copy for SubvectorView<V, T> {}

impl<V: VectorView<T>, T: ?Sized> SubvectorView<V, T> {

    pub fn new(base: V) -> Self {
        Self {
            begin: 0,
            end: base.len(),
            base: base,
            element: PhantomData
        }
    }
}

impl<V: VectorView<T>, T: ?Sized> VectorView<T> for SubvectorView<V, T> {
    
    fn at(&self, i: usize) -> &T {
        assert!(i < self.len());
        self.base.at(i + self.begin)
    }

    fn len(&self) -> usize {
        self.end - self.begin
    }
}

impl<V: VectorViewMut<T>, T: ?Sized> VectorViewMut<T> for SubvectorView<V, T> {

    fn at_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len());
        self.base.at_mut(i + self.begin)
    }
}

impl<V: SwappableVectorViewMut<T>, T: ?Sized> SwappableVectorViewMut<T> for SubvectorView<V, T> {

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.base.swap(i + self.begin, j + self.begin)
    }
}

impl<V: VectorView<T>, T: ?Sized> SelfSubvectorView<T> for SubvectorView<V, T> {

    fn restrict_full(mut self, range: std::ops::Range<usize>) -> Self {
        assert!(range.end <= self.len());
        debug_assert!(range.start <= range.end);
        self.end = self.begin + range.end;
        self.begin = self.begin + range.start;
        return self;
    }
}

pub struct SubvectorFn<V: VectorFn<T>, T> {
    begin: usize,
    end: usize,
    base: V,
    element: PhantomData<T>
}

impl<V: Clone + VectorFn<T>, T> Clone for SubvectorFn<V, T> {
    
    fn clone(&self) -> Self {
        Self {
            begin: self.begin,
            end: self.end,
            base: self.base.clone(),
            element: PhantomData
        }
    }
}

impl<V: Copy + VectorFn<T>, T> Copy for SubvectorFn<V, T> {}

impl<V: VectorFn<T>, T> SubvectorFn<V, T> {

    pub fn new(base: V) -> Self {
        Self {
            begin: 0,
            end: base.len(),
            base: base,
            element: PhantomData
        }
    }
}

impl<V: VectorFn<T>, T> VectorFn<T> for SubvectorFn<V, T> {
    
    fn at(&self, i: usize) -> T {
        assert!(i < self.len());
        self.base.at(i + self.begin)
    }

    fn len(&self) -> usize {
        self.end - self.begin
    }
}

impl<V: VectorFn<T>, T> SelfSubvectorFn<T> for SubvectorFn<V, T> {
    
    fn restrict_full(mut self, range: std::ops::Range<usize>) -> Self {
        assert!(range.end <= self.len());
        debug_assert!(range.start <= range.end);
        self.end = self.begin + range.end;
        self.begin = self.begin + range.start;
        return self;
    }
}

#[test]
fn test_subvector_ranges() {
    let a = SubvectorView::new([0, 1, 2, 3, 4]);
    assert_eq!(3, a.restrict(0..3).len());
    assert_eq!(3, a.restrict(0..=2).len());
    assert_eq!(5, a.restrict(0..).len());
    assert_eq!(5, a.restrict(..).len());
    assert_eq!(2, a.restrict(3..).len());
}

#[test]
fn test_subvector_subvector() {
    let a = SubvectorView::new([0, 1, 2, 3, 4]);
    let b = a.restrict(1..4);
    assert_eq!(3, b.len());
    assert_eq!(1, *b.at(0));
    assert_eq!(2, *b.at(1));
    assert_eq!(3, *b.at(2));
}

#[test]
#[should_panic]
fn test_subvector_subvector_oob() {
    let a = SubvectorView::new([0, 1, 2, 3, 4]);
    let b = a.restrict(1..4);
    b.restrict(0..4);
}

#[test]
fn test_subvector_fn_ranges() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].into_fn(|x| *x));
    assert_eq!(3, a.restrict(0..3).len());
    assert_eq!(3, a.restrict(0..=2).len());
    assert_eq!(5, a.restrict(0..).len());
    assert_eq!(5, a.restrict(..).len());
    assert_eq!(2, a.restrict(3..).len());
}

#[test]
fn test_subvector_fn_subvector() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].into_fn(|x| *x));
    let b = a.restrict(1..4);
    assert_eq!(3, b.len());
    assert_eq!(1, b.at(0));
    assert_eq!(2, b.at(1));
    assert_eq!(3, b.at(2));
}

#[test]
#[should_panic]
fn test_subvector_fn_subvector_oob() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].into_fn(|x| *x));
    let b = a.restrict(1..4);
    b.restrict(0..4);
}