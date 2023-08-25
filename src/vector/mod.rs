pub mod map;
pub mod stride;
pub mod chain;
pub mod permute;
pub mod sparse;
pub mod vec_fn;
pub mod subvector;

use std::marker::PhantomData;

use crate::ring::{RingStore, RingBase};

use self::chain::Chain;
use self::map::{Map, MapMut};
use self::stride::Stride;
use self::vec_fn::RingElVectorViewFn;

///
/// A trait for objects that provides read access to a 1-dimensional
/// array of objects.
/// 
/// # Related traits
/// 
/// If the entries are not owned by the object, but e.g. produced on the fly,
/// or just "associated" in a more general sense, the trait [vec_fn::VectorFn]
/// can be used.
/// Furthermore, for mutable access, use [VectorViewMut].
/// 
pub trait VectorView<T: ?Sized> {

    fn len(&self) -> usize;
    fn at(&self, i: usize) -> &T;

    fn map<U: ?Sized, F: Fn(&T) -> &U>(self, f: F) -> Map<Self, F, T>
        where Self: Sized
    {
        Map::new(self, f)
    }

    fn stride(self, stride: usize) -> Stride<T, Self>
        where Self: Sized
    {
        Stride::new(self, stride)
    }

    fn chain<V>(self, rhs: V) -> Chain<Self, V, T> 
        where Self: Sized, V: VectorView<T>
    {
        Chain::new(self, rhs)
    }

    fn as_el_fn<R>(self, ring: R) -> RingElVectorViewFn<R, Self, T>
        where Self: Sized,
            R: RingStore,
            R::Type: RingBase<Element = T>,
            T: Sized
    {
        RingElVectorViewFn::new(self, ring)
    }

    fn iter<'a>(&'a self) -> VectorViewIter<'a, Self, T> {
        VectorViewIter { begin: 0, end: self.len(), base: self, item: PhantomData }
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
/// # Iterators
/// 
/// Note that we cannot provide an `iter_mut()` function - it is not clear that
/// the underlying vector allows elements at different indices to be borrowed
/// mutably at once (e.g. sparse implementations). 
/// This is of course different in the immutable case. 
/// 
pub trait VectorViewMut<T: ?Sized>: VectorView<T> {

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

pub trait VectorViewSparse<T: ?Sized>: VectorView<T> {

    type Iter<'a>: Iterator<Item = (usize, &'a T)>
        where Self: 'a, T: 'a;

    fn nontrivial_entries<'a>(&'a self) -> Self::Iter<'a>;
}

impl<T> VectorView<T> for (T, T) {
    
    fn len(&self) -> usize {
        2
    }

    fn at(&self, i: usize) -> &T {
        match i {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("out of range")
        }
    }
}

impl<T> VectorViewMut<T> for (T, T) {

    fn at_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            _ => panic!("out of range")
        }
    }
}

impl<T> SwappableVectorViewMut<T> for (T, T) {

    fn swap(&mut self, i: usize, j: usize) {
        match (i, j) {
            (0, 1) | (1, 0) => { std::mem::swap(&mut self.0, &mut self.1) },
            (0, 0) | (1, 1) => {},
            _ => panic!("out of range")
        }
    }
}

impl<T> VectorView<T> for (T, T, T) {
    
    fn len(&self) -> usize {
        3
    }

    fn at(&self, i: usize) -> &T {
        match i {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("out of range")
        }
    }
}

impl<T> VectorViewMut<T> for (T, T, T) {

    fn at_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            _ => panic!("out of range")
        }
    }
}

impl<T> SwappableVectorViewMut<T> for (T, T, T) {

    fn swap(&mut self, i: usize, j: usize) {
        match (i, j) {
            (0, 1) | (1, 0) => { std::mem::swap(&mut self.0, &mut self.1) },
            (0, 2) | (2, 0) => { std::mem::swap(&mut self.0, &mut self.2) },
            (2, 1) | (1, 2) => { std::mem::swap(&mut self.2, &mut self.1) },
            (0, 0) | (1, 1) | (2, 2) => {},
            _ => panic!("out of range")
        }
    }
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

impl<'a, T: ?Sized, V: ?Sized> VectorView<T> for &'a V 
    where V: VectorView<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> &T {
        V::at(*self, i)
    }
}

impl<'a, T: ?Sized, V: ?Sized> VectorView<T> for &'a mut V 
    where V: VectorView<T>
{
    fn len(&self) -> usize {
        V::len(*self)
    }

    fn at(&self, i: usize) -> &T {
        V::at(*self, i)
    }
}

impl<'a, T: ?Sized, V: ?Sized> VectorViewMut<T> for &'a mut V 
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

pub struct VectorViewIter<'a, V: ?Sized, T: ?Sized>
    where V: 'a + VectorView<T>, T: 'a
{
    begin: usize,
    end: usize,
    base: &'a V,
    item: PhantomData<T>
}

impl<'a, V: ?Sized, T: ?Sized> Iterator for VectorViewIter<'a, V, T>
    where V: 'a + VectorView<T>, T: 'a
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.begin != self.end {
            let result = self.base.at(self.begin);
            self.begin += 1;
            return Some(result);
        } else {
            return None;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, V: ?Sized, T: ?Sized> DoubleEndedIterator for VectorViewIter<'a, V, T>
    where V: 'a + VectorView<T>, T: 'a
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.begin != self.end {
            self.end -= 1;
            return Some(self.base.at(self.end));
        } else {
            return None;
        }
    }
}

impl<'a, V: ?Sized, T: ?Sized> ExactSizeIterator for VectorViewIter<'a, V, T>
    where V: 'a + VectorView<T>, T: 'a
{
    fn len(&self) -> usize {
        self.end - self.begin
    }
}

impl<'a, V: ?Sized, T: ?Sized> Clone for VectorViewIter<'a, V, T>
    where V: 'a + VectorView<T>, T: 'a
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: ?Sized, T: ?Sized> Copy for VectorViewIter<'a, V, T>
    where V: 'a + VectorView<T>, T: 'a
{}
