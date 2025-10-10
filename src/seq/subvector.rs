use std::marker::PhantomData;
use std::fmt::Debug;

use super::{SelfSubvectorFn, SelfSubvectorView, SparseVectorViewOperation, SwappableVectorViewMut, VectorFn, VectorView, VectorViewMut, VectorViewSparse};

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

    fn specialize_sparse<Op: SparseVectorViewOperation<T, Self>>(op: Op) -> Op::Output {

        struct WrapSubvector<T: ?Sized, V: VectorView<T>, Op: SparseVectorViewOperation<T, SubvectorView<V, T>>> {
            op: Op,
            element: PhantomData<T>,
            vector: PhantomData<V>
        }

        impl<T: ?Sized, V: VectorView<T>, Op: SparseVectorViewOperation<T, SubvectorView<V, T>>> SparseVectorViewOperation<T, V> for WrapSubvector<T, V, Op> {

            type Output = Op::Output;

            fn execute(self) -> Self::Output
                where V: VectorViewSparse<T>
            {
                self.op.execute()
            }

            fn fallback(self) -> Self::Output {
                self.op.fallback()
            }
        }

        V::specialize_sparse(WrapSubvector { op: op, element: PhantomData, vector: PhantomData })
    }

    fn as_slice<'a>(&'a self) -> Option<&'a [T]>
        where T: Sized
    {
        self.base.as_slice().map(|slice| &slice[self.begin..self.end])
    }
}

impl<V: VectorView<T> + Debug, T: ?Sized> Debug for SubvectorView<V, T> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubvectorView")
            .field("begin", &self.begin)
            .field("end", &self.end)
            .field("base", &self.base)
            .finish()
    }
}

pub struct FilterWithinRangeIter<'a, T: ?Sized, I>
    where T: 'a,
        I: Iterator<Item = (usize, &'a T)>
{
    it: I,
    begin: usize,
    end: usize
}

impl<'a, T: ?Sized, I> Iterator for FilterWithinRangeIter<'a, T, I>
    where T: 'a,
        I: Iterator<Item = (usize, &'a T)>
{
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.by_ref().filter(|(i, _)| *i >= self.begin && *i < self.end).next()
    }
}

impl<V: VectorViewSparse<T>, T: ?Sized> VectorViewSparse<T> for SubvectorView<V, T> {

    type Iter<'a> = FilterWithinRangeIter<'a, T, V::Iter<'a>>
        where Self: 'a, T: 'a;

    fn nontrivial_entries<'a>(&'a self) -> Self::Iter<'a> {
        FilterWithinRangeIter {
            it: self.base.nontrivial_entries(),
            begin: self.begin,
            end: self.end
        }
    }
}

impl<V: VectorViewMut<T>, T: ?Sized> VectorViewMut<T> for SubvectorView<V, T> {

    fn at_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len());
        self.base.at_mut(i + self.begin)
    }

    fn as_slice_mut<'a>(&'a mut self) -> Option<&'a mut [T]>
        where T: Sized
    {
        self.base.as_slice_mut().map(|slice| &mut slice[self.begin..self.end])
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

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use super::sparse::SparseMapVector;

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
    _ = b.restrict(0..4);
}

#[test]
fn test_subvector_fn_ranges() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].clone_els_by(|x| *x));
    assert_eq!(3, a.restrict(0..3).len());
    assert_eq!(3, a.restrict(0..=2).len());
    assert_eq!(5, a.restrict(0..).len());
    assert_eq!(5, a.restrict(..).len());
    assert_eq!(2, a.restrict(3..).len());
}

#[test]
fn test_subvector_fn_subvector() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].clone_els_by(|x| *x));
    let b = a.restrict(1..4);
    assert_eq!(3, b.len());
    assert_eq!(1, b.at(0));
    assert_eq!(2, b.at(1));
    assert_eq!(3, b.at(2));
}

#[test]
#[should_panic]
fn test_subvector_fn_subvector_oob() {
    let a = SubvectorFn::new([0, 1, 2, 3, 4].clone_els_by(|x| *x));
    let b = a.restrict(1..4);
    _ = b.restrict(0..4);
}

#[test]
fn test_subvector_sparse() {
    let mut sparse_vector = SparseMapVector::new(1000, StaticRing::<i64>::RING);
    *sparse_vector.at_mut(6) = 6;
    *sparse_vector.at_mut(20) = 20;
    *sparse_vector.at_mut(256) = 256;
    *sparse_vector.at_mut(257) = 257;

    let subvector = SubvectorView::new(sparse_vector).restrict(20..=256);

    struct Verify(SubvectorView<SparseMapVector<StaticRing<i64>>, i64>);

    impl SparseVectorViewOperation<i64, SubvectorView<SparseMapVector<StaticRing<i64>>, i64>> for Verify {

        type Output = ();

        fn execute(self) -> Self::Output
            where SubvectorView<SparseMapVector<StaticRing<i64>>, i64>: VectorViewSparse<i64>
        {
            assert!(
                vec![(20, &20), (256, &256)] == self.0.nontrivial_entries().collect::<Vec<_>>() ||
                vec![(256, &256), (20, &20)] == self.0.nontrivial_entries().collect::<Vec<_>>()
            );
        }

        fn fallback(self) -> Self::Output {
            unreachable!()
        }
    }

    VectorView::specialize_sparse(Verify(subvector));
}