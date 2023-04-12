pub mod map;
pub mod stride;

use std::ops::{RangeBounds, Bound, Index, IndexMut};
use std::marker::PhantomData;

use self::map::{Map, MapMut};
use self::stride::Stride;

///
/// A trait for objects that provides read access to a 1-dimensional
/// array of objects.
/// 
/// # Related traits
/// 
/// If the entries are not owned by the object, but e.g. produced on the fly,
/// or just "associated" in a more general sense, the trait [crate::vector::VectorFn]
/// can be used.
/// Furthermore, for mutable access, use [crate::vector::VectorViewMut].
/// 
pub trait VectorView<T> {

    fn len(&self) -> usize;
    fn at(&self, i: usize) -> &T;

    fn map<U, F: Fn(&T) -> &U>(self, f: F) -> Map<Self, F, T>
        where Self: Sized
    {
        Map::new(self, f)
    }

    fn stride(self, stride: usize) -> Stride<T, Self>
        where Self: Sized
    {
        Stride::new(self, stride)
    }
}

///
/// A trait for objects that provides mutable access to a 1-dimensional
/// array of objects.
/// 
/// # Related traits
/// 
/// If only immutable access is provided, use [crate::vector::VectorViewMut].
/// 
pub trait VectorViewMut<T>: VectorView<T> {

    fn at_mut(&mut self, i: usize) -> &mut T;

    fn map_mut<U, F: Fn(&T) -> &U, G: Fn(&mut T) -> &mut U>(self, f: F, f_mut: G) -> MapMut<Self, F, G, T>
        where Self: Sized
    {
        MapMut::new(self, f, f_mut)
    }
}

pub trait SwappableVectorViewMut<T>: VectorViewMut<T> {

    fn swap(&mut self, i: usize, j: usize);
}

///
/// A trait for objects that have the structure of a one-dimensional array,
/// and can produce objects at each entry.
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
/// possible implementations. For now, we just provide
/// ```ignore
/// impl<T: Copy, V> VectorFn<T> for VectorView<T> { ... }
/// ```
/// 
pub trait VectorFn<T> {
    
    fn len(&self) -> usize;
    fn at(&self, i: usize) -> T;
}

impl<T> VectorView<T> for Vec<T> {
    
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn at(&self, i: usize) -> &T {
        &(*self)[i]
    }
}

impl<T> VectorViewMut<T> for Vec<T> {
    
    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut (*self)[i]
    }
}

impl<T> SwappableVectorViewMut<T> for Vec<T> {
    
    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(&mut self[..], i, j);
    }
}

impl<T> VectorView<T> for [T] {
    
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn at(&self, i: usize) -> &T {
        &(*self)[i]
    }
}

impl<T> VectorViewMut<T> for [T] {
    
    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut (*self)[i]
    }
}

impl<T> SwappableVectorViewMut<T> for [T] {
    
    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(self, i, j);
    }
}


impl<T, const N: usize> VectorView<T> for [T; N] {
    
    fn len(&self) -> usize {
        N
    }

    fn at(&self, i: usize) -> &T {
        &(*self)[i]
    }
}

impl<T, const N: usize> VectorViewMut<T> for [T; N] {
    
    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut (*self)[i]
    }
}

impl<T, const N: usize> SwappableVectorViewMut<T> for [T; N] {
    
    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(&mut self[..], i, j);
    }
}

impl<'a, T, V: ?Sized> VectorView<T> for &'a V 
    where V: VectorView<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> &T {
        V::at(*self, i)
    }
}

impl<'a, T, V: ?Sized> VectorView<T> for &'a mut V 
    where V: VectorView<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> &T {
        V::at(*self, i)
    }
}

impl<'a, T, V: ?Sized> VectorViewMut<T> for &'a mut V 
    where V: VectorViewMut<T>
{
    fn at_mut(&mut self, i: usize) -> &mut T {
        V::at_mut(*self, i)
    }
}

impl<'a, T, V: ?Sized> SwappableVectorViewMut<T> for &'a mut V 
    where V: SwappableVectorViewMut<T>
{
    fn swap(&mut self, i: usize, j: usize) {
        V::swap(*self, i, j);
    }
}

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
