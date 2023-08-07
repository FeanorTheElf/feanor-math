use std::{marker::PhantomData, ops::Range};

use crate::ring::{RingStore, RingBase};

use super::{VectorView, map::MapFn, subvector::{SelfSubvectorView, SelfSubvectorFn}};

///
/// A trait for objects that have the structure of a one-dimensional array,
/// and can produce objects at each entry. In other words, this is like a 
/// "random-access iterator".
/// 
/// # Related traits
/// 
/// If the entries are owned by the object, consider using the trait [crate::vector::VectorView].
/// Instead of returning entries by value, it returns entries by reference.
/// 
/// # Blanket implementations
/// 
/// There are many kinds of blanket implementations thinkable, e.g.
/// ```ignore
/// impl<T: Clone, V> VectorFn<T> for VectorView<T> { ... }
/// ```
/// or
/// ```ignore
/// impl<'a, T, V> VectorFn<&'a T> for &'a VectorView<T> { ... }
/// ```
/// However, these do not represent the standard use cases and clutter the space of
/// possible implementations. Instead, use the function [`crate::vector::VectorView::as_fn()`].
/// 
pub trait VectorFn<T> {
    
    fn len(&self) -> usize;
    fn at(&self, i: usize) -> T;

    fn map<U, F: Fn(T) -> U>(self, f: F) -> MapFn<Self, F, T>
        where Self: Sized
    {
        MapFn::new(self, f)
    }

    fn to_vec(&self) -> Vec<T> {
        Iterator::map(0..self.len(), |i| self.at(i)).collect()
    }
}

pub trait IntoVectorFn<T> {

    type Target: VectorFn<T>;

    fn into_fn(self) -> Self::Target;
}

pub struct VectorViewFn<V, T>
    where T: Clone,
        V: VectorView<T>
{
    base: V,
    element: PhantomData<T>
}

impl<V, T> VectorViewFn<V, T>
    where T: Clone,
        V: VectorView<T>
{
    pub fn new(base: V) -> Self {
        VectorViewFn { base: base, element: PhantomData }
    }
}

impl<V, T> SelfSubvectorFn<T> for VectorViewFn<V, T>
    where T: Clone,
        V: SelfSubvectorView<T>
{
    fn subvector<R: std::ops::RangeBounds<usize>>(self, range: R) -> Self {
        VectorViewFn { base: self.base.subvector(range), element: PhantomData }
    }
}

impl<V, T> VectorFn<T> for VectorViewFn<V, T>
    where T: Clone,
        V: VectorView<T>
{
    fn at(&self, i: usize) -> T {
        self.base.at(i).clone()
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}

impl<V, T> Clone for VectorViewFn<V, T>
    where T: Clone,
        V: Clone + VectorView<T>
{
    fn clone(&self) -> Self {
        Self::new(self.base.clone())
    }
}

impl<V, T> Copy for VectorViewFn<V, T>
    where T: Clone,
        V: Copy + VectorView<T>
{}

impl<'a, T, V: ?Sized> VectorFn<T> for &'a V 
    where V: VectorFn<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> T {
        V::at(*self, i)
    }
}

impl<'a, T, V: ?Sized> VectorFn<T> for &'a mut V 
    where V: VectorFn<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> T {
        V::at(*self, i)
    }
}

pub struct RingElVectorViewFn<R, V, T>
    where R: RingStore,
        R::Type: RingBase<Element = T>,
        V: VectorView<T>
{
    ring: R,
    base: V,
    element: PhantomData<T>
}

impl<R, V, T> RingElVectorViewFn<R, V, T>
    where R: RingStore,
        R::Type: RingBase<Element = T>,
        V: VectorView<T>
{
    pub fn new(base: V, ring: R) -> Self {
        RingElVectorViewFn { ring: ring, base: base, element: PhantomData }
    }
}

impl<R, V, T> SelfSubvectorFn<T> for RingElVectorViewFn<R, V, T>
    where R: RingStore,
        R::Type: RingBase<Element = T>,
        V: SelfSubvectorView<T>
{
    fn subvector<S: std::ops::RangeBounds<usize>>(self, range: S) -> Self {
        RingElVectorViewFn { ring: self.ring, base: self.base.subvector(range), element: PhantomData }
    }
}

impl<R, V, T> VectorFn<T> for RingElVectorViewFn<R, V, T>
    where R: RingStore,
        R::Type: RingBase<Element = T>,
        V: VectorView<T>
{
    fn at(&self, i: usize) -> T {
        self.ring.clone_el(self.base.at(i))
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}

impl<R, V, T> Clone for RingElVectorViewFn<R, V, T>
    where R: Clone + RingStore,
        R::Type: RingBase<Element = T>,
        V: Clone + VectorView<T>
{
    fn clone(&self) -> Self {
        Self::new(self.base.clone(), self.ring.clone())
    }
}

impl<R, V, T> Copy for RingElVectorViewFn<R, V, T>
    where R: Copy + RingStore,
        R::Type: RingBase<Element = T>,
        V: Copy + VectorView<T>
{}

pub struct RangeFn(Range<usize>);

impl IntoVectorFn<usize> for Range<usize> {

    type Target = RangeFn;

    fn into_fn(self) -> Self::Target {
        RangeFn(self)
    }
}

impl VectorFn<usize> for RangeFn {

    fn len(&self) -> usize {
        self.0.end - self.0.start
    }

    fn at(&self, i: usize) -> usize {
        assert!(i >= self.0.start && i < self.0.end);
        return i;
    }
}

impl<T, V> IntoVectorFn<T> for V
    where T: Clone, V: VectorView<T>
{
    type Target = VectorViewFn<V, T>;

    fn into_fn(self) -> Self::Target {
        VectorViewFn::new(self)
    }
}