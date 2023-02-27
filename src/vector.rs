use std::ops::{RangeBounds, Index, IndexMut};

///
/// A trait for objects that provides read access to a 1-dimensional
/// array of objects.
/// 
/// # Related traits
/// If the entries are not owned by the object, but e.g. produced on the fly,
/// or just "associated" in a more general sense, the trait [crate::vector::VectorFn]
/// can be used.
/// Furthermore, for mutable access, use [crate::vector::VectorViewMut].
/// 
pub trait VectorView<T> {

    fn len(&self) -> usize;
    fn at(&self, i: usize) -> &T;
}

///
/// A trait for objects that provides mutable access to a 1-dimensional
/// array of objects.
/// 
/// # Related traits
/// If only immutable access is provided, use [crate::vector::VectorViewMut].
/// 
pub trait VectorViewMut<T>: VectorView<T> {

    fn at_mut(&mut self, i: usize) -> &mut T;
}

pub trait SwappableVectorViewMut<T>: VectorViewMut<T> {

    fn swap(&mut self, i: usize, j: usize);
}

///
/// A trait for objects that have the structure of a one-dimensional array,
/// and can produce objects at each entry.
/// 
/// # Related traits
/// If the entries are owned by the object, consider using the trait [crate::vector::VectorView].
/// Instead of returning entries by value, it returns entries by reference.
/// 
/// # Blanket implementations
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