
mod conversion;
mod map;

pub mod step_by;
pub mod subvector;
pub mod permute;
pub mod sparse;

use std::ops::{Bound, Range, RangeBounds};

pub use conversion::{CloneElFn, VectorViewFn, VectorFnIter};
pub use map::{VectorFnMap, VectorViewMap, VectorViewMapMut};
use step_by::{StepBy, StepByFn};

use crate::ring::*;

///
/// A trait for objects that provides random-position read access to a 1-dimensional 
/// sequence (or vector) of elements. 
/// 
/// # Related traits
/// 
/// Other traits that represent sequences are 
///  - [`ExactSizeIterator`]: Returns elements by value; Since elements are moved, each
///    element is returned only once, and they must be queried in order.
///  - [`VectorFn`]: Also returns elements by value, but assumes that the underlying structure
///    produces a new element whenever a position is queried. This allows accessing positions
///    multiple times and in a random order, but depending on the represented items, it might
///    require cloning an element on each access.
/// 
/// Apart from that, there are also the subtraits [`VectorViewMut`] and [`SwappableVectorViewMut`]
/// that allow mutating the underlying sequence (but still don't allow moving elements out).
/// Finally, there is [`SelfSubvectorView`], which directly supports taking subvectors.
/// 
/// # Example
/// ```
/// # use feanor_math::seq::*;
/// fn compute_sum<V: VectorView<i32>>(vec: V) -> i32 {
///     let mut result = 0;
///     for i in 0..vec.len() {
///         result += vec.at(i);
///     }
///     return result;
/// }
/// assert_eq!(10, compute_sum([1, 2, 3, 4]));
/// assert_eq!(10, compute_sum(vec![1, 2, 3, 4]));
/// assert_eq!(10, compute_sum(&[1, 2, 3, 4, 5][..4]));
/// ```
/// 
pub trait VectorView<T: ?Sized> {

    fn len(&self) -> usize;
    fn at(&self, i: usize) -> &T;

    ///
    /// Calls `op` with `self` if this vector view supports sparse access.
    /// Otherwise, `()` is returned.
    /// 
    /// This is basically a workaround that enables users to specialize on
    /// `V: VectorViewSparse`, even though specialization currently does not support
    /// this.
    /// 
    fn specialize_sparse<Op: SparseVectorViewOperation<T>>(&self, _op: Op) -> Result<Op::Output, ()> {
        Err(())
    }

    ///
    /// Returns an iterator over all elements in this vector.
    /// 
    /// NB: Not called `iter()` to prevent name conflicts, since many containers (e.g. `Vec<T>`)
    /// have a function `iter()` and implement [`VectorView`]. As a result, whenever [`VectorView`]
    /// is in scope, calling any one `iter()` would require fully-qualified call syntax.
    /// 
    fn as_iter<'a>(&'a self) -> VectorFnIter<VectorViewFn<'a, Self, T>, &'a T> {
        VectorFnIter::new(self.as_fn())
    }

    ///
    /// Converts this vector into a [`VectorFn`] that yields references `&T`.
    /// 
    fn as_fn<'a>(&'a self) -> VectorViewFn<'a, Self, T> {
        VectorViewFn::new(self)
    }

    ///
    /// Converts this vector into a [`VectorFn`] that clones elements on access.
    /// 
    fn into_ring_el_fn<R: RingStore>(self, ring: R) -> CloneElFn<Self, T, CloneRingEl<R>>
        where Self: Sized, T: Sized, R::Type: RingBase<Element = T>
    {
        self.into_fn(CloneRingEl(ring))
    }

    ///
    /// Converts this vector into a [`VectorFn`] that clones elements on access.
    /// 
    fn as_ring_el_fn<'a, R: RingStore>(&'a self, ring: R) -> CloneElFn<&'a Self, T, CloneRingEl<R>>
        where T: Sized, 
            R::Type: RingBase<Element = T>
    {
        self.into_ring_el_fn(ring)
    }

    ///
    /// Converts this vector into a [`VectorFn`] that clones elements on access.
    /// 
    fn into_fn<F>(self, clone_entry: F) -> CloneElFn<Self, T, F>
        where Self: Sized, T: Sized, F: Fn(&T) -> T
    {
        CloneElFn::new(self, clone_entry)
    }

    ///
    /// Creates a new [`VectorView`] whose elements are the results of the given function
    /// applied to the elements of this vector.
    /// 
    /// The most common use case is a projection on contained elements. Since [`VectorView`]s
    /// provide elements by reference, this is much less powerful than [`Iterator::map()`] or
    /// [`VectorFn::map_fn()`], since the function cannot return created elements.
    /// 
    /// # Example
    /// ```
    /// use feanor_math::seq::*;
    /// fn foo<V: VectorView<i64>>(data: V) {
    ///     // some logic
    /// }
    /// let data = vec![Some(1), Some(2), Some(3)];
    /// // the `as_ref()` is necessary, since we have to return a reference
    /// foo(data.map(|x| x.as_ref().unwrap()));
    /// ```
    /// 
    fn map<F: for<'a> Fn(&'a T) -> &'a U, U>(self, func: F) -> VectorViewMap<Self, T, U, F>
        where Self: Sized
    {
        VectorViewMap::new(self, func)
    }

    fn step_by(self, step_by: usize) -> StepBy<Self, T>
        where Self: Sized
    {
        StepBy::new(self, step_by)
    }
}

///
/// View on a sequence type that stores its data in a sparse format.
/// This clearly requires that the underlying type `T` has some notion
/// of a "zero" element.
/// 
pub trait VectorViewSparse<T: ?Sized>: VectorView<T> {

    type Iter<'a>: Iterator<Item = (usize, &'a T)>
        where Self: 'a, 
            T: 'a;

    ///
    /// Returns an iterator over all elements of the sequence with their indices
    /// that are "nonzero" (`T` must have an appropriate notion of zero).
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::seq::*;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::seq::sparse::*;
    /// let mut vector = SparseMapVector::new(10, StaticRing::<i64>::RING);
    /// *vector.at_mut(2) = 100;
    /// assert_eq!(vec![(2, 100)], vector.nontrivial_entries().map(|(i, x)| (i, *x)).collect::<Vec<_>>());
    /// ```
    /// 
    fn nontrivial_entries<'a>(&'a self) -> Self::Iter<'a>;
}

pub trait SparseVectorViewOperation<T: ?Sized> {

    type Output;

    fn execute<V: VectorViewSparse<T>>(self, vector: V) -> Self::Output;
}

fn range_within<R: RangeBounds<usize>>(len: usize, range: R) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(i) => {
            assert!(*i <= len);
            *i
        },
        Bound::Excluded(i) => {
            assert!(*i <= len);
            *i + 1
        }
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Included(i) => {
            assert!(*i >= start);
            assert!(*i < len);
            *i + 1
        },
        Bound::Excluded(i) => {
            assert!(*i >= start);
            assert!(*i <= len);
            *i
        }
    };
    return start..end;
}

///
/// Trait for [`VectorView`]s that support shrinking, i.e. transforming the
/// vector into a subvector of itself.
/// 
/// Note that you can easily get a subvector of a vector by using [`subvector::SubvectorView`],
/// but this will wrap the original type. This makes [`subvector::SubvectorView`] unsuitable
/// for some applications, like recursive algorithms.
/// 
/// Note also that [`SelfSubvectorView::restrict()`] consumes the current object, thus
/// it is most useful for vectors that implement [`Clone`]/[`Copy`], in particular for references
/// to vectors.
/// 
/// This is the [`VectorView`]-counterpart to [`SelfSubvectorFn`].
/// 
/// # Example
/// ```
/// # use feanor_math::seq::*;
/// # use feanor_math::seq::subvector::*;
/// fn compute_sum_recursive<V: SelfSubvectorView<i32>>(vec: V) -> i32 {
///     if vec.len() == 0 {
///         0
///     } else {
///         *vec.at(0) + compute_sum_recursive(vec.restrict(1..))
///     }
/// }
/// assert_eq!(10, compute_sum_recursive(SubvectorView::new([1, 2, 3, 4])));
/// assert_eq!(10, compute_sum_recursive(SubvectorView::new(vec![1, 2, 3, 4])));
/// assert_eq!(10, compute_sum_recursive(SubvectorView::new(&[1, 2, 3, 4, 5][..4])));
/// ```
/// 
pub trait SelfSubvectorView<T: ?Sized>: Sized + VectorView<T> {

    ///
    /// Returns a [`SelfSubvectorView`] that represents the elements within the given range
    /// of this vector.
    /// 
    fn restrict_full(self, range: Range<usize>) -> Self;

    ///
    /// Returns a [`SelfSubvectorView`] that represents the elements within the given range
    /// of this vector.
    /// 
    fn restrict<R: RangeBounds<usize>>(self, range: R) -> Self {
        let range_full = range_within(self.len(), range);
        self.restrict_full(range_full)
    }
}

impl<T: ?Sized, V: ?Sized + VectorView<T>> VectorView<T> for Box<V> {

    fn len(&self) -> usize {
        (**self).len()
    }

    fn at(&self, i: usize) -> &T {
        (**self).at(i)
    }

    fn specialize_sparse<Op: SparseVectorViewOperation<T>>(&self, op: Op) -> Result<Op::Output, ()> {
        (**self).specialize_sparse(op)
    }
}

impl<T: ?Sized, V: ?Sized + VectorViewMut<T>> VectorViewMut<T> for Box<V> {

    fn at_mut(&mut self, i: usize) -> &mut T {
        (**self).at_mut(i)
    }
}


impl<T: ?Sized, V: ?Sized + VectorViewSparse<T>> VectorViewSparse<T> for Box<V> {
    
    type Iter<'b> = V::Iter<'b>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        (**self).nontrivial_entries()
    }
}

impl<'a, T: ?Sized, V: ?Sized + VectorView<T>> VectorView<T> for &'a V {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn at(&self, i: usize) -> &T {
        (**self).at(i)
    }

    fn specialize_sparse<Op: SparseVectorViewOperation<T>>(&self, op:Op) -> Result<Op::Output, ()> {
        (**self).specialize_sparse(op)
    }
}

impl<'a, T: ?Sized, V: ?Sized + VectorViewSparse<T>> VectorViewSparse<T> for &'a V {
    type Iter<'b> = V::Iter<'b>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        (**self).nontrivial_entries()
    }
}

impl<'a, T: ?Sized, V: ?Sized + VectorView<T>> VectorView<T> for &'a mut V {

    fn len(&self) -> usize {
        (**self).len()
    }

    fn at(&self, i: usize) -> &T {
        (**self).at(i)
    }

    fn specialize_sparse<Op: SparseVectorViewOperation<T>>(&self, op:Op) -> Result<Op::Output, ()> {
        (**self).specialize_sparse(op)
    }
}

impl<'a, T: ?Sized, V: ?Sized + VectorViewMut<T>> VectorViewMut<T> for &'a mut V {

    fn at_mut(&mut self, i: usize) -> &mut T {
        (**self).at_mut(i)
    }
}

impl<'a, T: ?Sized, V: ?Sized + VectorViewSparse<T>> VectorViewSparse<T> for &'a mut V {

    type Iter<'b> = V::Iter<'b>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        (**self).nontrivial_entries()
    }
}

impl<'a, T: ?Sized, V: ?Sized + SwappableVectorViewMut<T>> SwappableVectorViewMut<T> for &'a mut V {

    fn swap(&mut self, i: usize, j: usize) {
        (**self).swap(i, j)
    }
}

///
/// A trait for [`VectorView`]s that allow mutable access to one element at a time.
/// 
/// Note that a fundamental difference to many containers (like `&mut [T]`) is that
/// this trait only defines functions that give a mutable reference to one element at
/// a time. In particular, it is intentionally impossible to have a mutable reference
/// to multiple elements at once. This enables implementations like sparse vectors,
/// e.g. [`sparse::SparseMapVector`].
/// 
pub trait VectorViewMut<T: ?Sized>: VectorView<T> {

    fn at_mut(&mut self, i: usize) -> &mut T;

    fn map_mut<F_const: for<'a> Fn(&'a T) -> &'a U, F_mut: for<'a> FnMut(&'a mut T) -> &'a mut U, U>(self, map_const: F_const, map_mut: F_mut) -> VectorViewMapMut<Self, T, U, F_const, F_mut>
        where Self: Sized
    {
        VectorViewMapMut::new(self, (map_const, map_mut))
    }
}

///
/// A trait for [`VectorViewMut`]s that support swapping of two elements.
/// 
/// Since [`VectorViewMut`] is not necessarily able to return two mutable
/// references to different entries, supporting swapping is indeed a stronger
/// property than just being a [`VectorViewMut`].
/// 
pub trait SwappableVectorViewMut<T: ?Sized>: VectorViewMut<T> {

    fn swap(&mut self, i: usize, j: usize);
}

///
/// A trait for objects that provides random-position access to a 1-dimensional 
/// sequence (or vector) of elements that returned by value. 
/// 
/// # Related traits
/// 
/// Other traits that represent sequences are 
///  - [`ExactSizeIterator`]: Also returns elements by value; However, to avoid copying elements,
///    an `ExactSizeIterator` returns every item only once, and only in the order of the underlying
///    vector.
///  - [`VectorView`]: Returns only references to the underlying data, but also supports random-position
///    access. Note that `VectorView<T>` is not the same as `VectorFn<&T>`, since the lifetime of returned
///    references `&T` in the case of `VectorView` is the lifetime of the vector, but in the case of 
///    `VectorFn`, it must be a fixed lifetime parameter.
/// 
/// Finally, there is the subtrait [`SelfSubvectorFn`], which directly supports taking subvectors.
/// 
/// # Example
/// ```
/// # use feanor_math::seq::*;
/// fn compute_sum<V: VectorFn<usize>>(vec: V) -> usize {
///     let mut result = 0;
///     for i in 0..vec.len() {
///         result += vec.at(i);
///     }
///     return result;
/// }
/// assert_eq!(10, compute_sum(1..5));
/// assert_eq!(10, compute_sum([1, 2, 3, 4].into_fn(|x| *x)));
/// assert_eq!(10, compute_sum(vec![1, 2, 3, 4].into_fn(|x| *x)));
/// assert_eq!(10, compute_sum((&[1, 2, 3, 4, 5][..4]).into_fn(|x| *x)));
/// ```
/// 
pub trait VectorFn<T> {

    fn len(&self) -> usize;
    fn at(&self, i: usize) -> T;

    ///
    /// Produces an iterator over the elements of this [`VectorFn`].
    /// 
    /// This transfers ownership of the object to the iterator. If this
    /// is not desired, consider using [`VectorFn::iter()`].
    /// 
    /// Note that [`VectorFn`]s do not necessarily implement [`IntoIterator`] and
    /// instead use this function. The reason for that is twofold:
    ///  - the only way of making all types implementing [`VectorFn`]s to also implement [`IntoIterator`]
    ///    would be to define `VectorFn` as a subtrait of `IntoIterator`. However, this conflicts with the
    ///    decision to have [`VectorFn`] have the element type as generic parameter, since [`IntoIterator`] 
    ///    uses an associated type.
    ///  - If the above problem could somehow be circumvented, for types that implement both [`Iterator`]
    ///    and [`VectorFn`] (like [`Range`]), calling `into_iter()` would then require fully-qualified call
    ///    syntax, which is very unwieldy.
    /// 
    fn into_iter(self) -> VectorFnIter<Self, T>
        where Self: Sized
    {
        VectorFnIter::new(self)
    }

    ///
    /// Produces an iterator over the elements of this [`VectorFn`].
    /// 
    /// See also [`VectorFn::into_iter()`] if a transfer of ownership is required.
    /// 
    fn iter<'a>(&'a self) -> VectorFnIter<&'a Self, T> {
        self.into_iter()
    }

    ///
    /// NB: Named `map_fn` to avoid conflicts with `map` of [`Iterator`]
    /// 
    fn map_fn<F: Fn(T) -> U, U>(self, func: F) -> VectorFnMap<Self, T, U, F>
        where Self: Sized
    {
        VectorFnMap::new(self, func)
    }

    ///
    /// NB: Named `step_by_fn` to avoid conflicts with `step_by` of [`Iterator`]
    /// 
    fn step_by_fn(self, step_by: usize) -> StepByFn<Self, T>
        where Self: Sized
    {
        StepByFn::new(self, step_by)
    }
}

///
/// Trait for [`VectorFn`]s that support shrinking, i.e. transforming the
/// vector into a subvector of itself.
/// 
/// Note that you can easily get a subvector of a vector by using [`subvector::SubvectorFn`],
/// but this will wrap the original type. This makes [`subvector::SubvectorFn`] unsuitable
/// for some applications, like recursive algorithms.
/// 
/// Note also that [`SelfSubvectorFn::restrict()`] consumes the current object, thus
/// it is most useful for vectors that implement [`Clone`]/[`Copy`], in particular for references
/// to vectors.
/// 
/// This is the [`VectorFn`]-counterpart to [`SelfSubvectorView`].
/// 
/// ## Default impls
/// 
/// As opposed to [`VectorView`], there are no implementations of [`VectorFn`] for standard
/// containers like `Vec<T>`, `&[T]` etc. This is because it is not directly clear whether elements
/// should be cloned on access, or whether a `VectorFn<&T>` is desired. Instead, use the appropriate
/// functions [`VectorView::as_fn()`] or [`VectorView::into_fn()`] to create a [`VectorFn`].
/// An exception is made for `Range<usize>`, which directly implements `VectorFn`. This allows
/// for yet another way of creating arbitrary `VectorFn`s by using `(0..len).map_fn(|i| ...)`.
/// 
/// # Example
/// ```
/// # use feanor_math::seq::*;
/// # use feanor_math::seq::subvector::*;
/// fn compute_sum_recursive<V: SelfSubvectorFn<usize>>(vec: V) -> usize {
///     if vec.len() == 0 {
///         0
///     } else {
///         vec.at(0) + compute_sum_recursive(vec.restrict(1..))
///     }
/// }
/// assert_eq!(10, compute_sum_recursive(SubvectorFn::new([1, 2, 3, 4].into_fn(|x| *x))));
/// assert_eq!(10, compute_sum_recursive(SubvectorFn::new(vec![1, 2, 3, 4].into_fn(|x| *x))));
/// assert_eq!(10, compute_sum_recursive(SubvectorFn::new((&[1, 2, 3, 4, 5][..4]).into_fn(|x| *x))));
/// ```
/// 
pub trait SelfSubvectorFn<T>: Sized + VectorFn<T> {

    ///
    /// Returns a [`SelfSubvectorFn`] that represents the elements within the given range
    /// of this vector.
    /// 
    fn restrict_full(self, range: Range<usize>) -> Self;

    ///
    /// Returns a [`SelfSubvectorFn`] that represents the elements within the given range
    /// of this vector.
    /// 
    fn restrict<R: RangeBounds<usize>>(self, range: R) -> Self {
        let range_full = range_within(self.len(), range);
        self.restrict_full(range_full)
    }
}

impl<'a, T, V: ?Sized + VectorFn<T>> VectorFn<T> for &'a V {

    fn len(&self) -> usize {
        (**self).len()
    }

    fn at(&self, i: usize) -> T {
        (**self).at(i)
    }
}

impl<T> VectorView<T> for [T] {

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn at(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<'a, T> SelfSubvectorView<T> for &'a [T] {

    fn restrict_full(self, range: Range<usize>) -> Self {
        &self[range]
    }
}

impl<'a, T> SelfSubvectorView<T> for &'a mut [T] {

    fn restrict_full(self, range: Range<usize>) -> Self {
        &mut self[range]
    }
}

impl<T> VectorViewMut<T> for [T] {

    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self[i]
    }
}

impl<T> SwappableVectorViewMut<T> for [T] {

    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(self, i, j)
    }
}

impl<T> VectorView<T> for Vec<T> {

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn at(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<T> VectorViewMut<T> for Vec<T> {

    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self[i]
    }
}

impl<T> SwappableVectorViewMut<T> for Vec<T> {

    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(self, i, j)
    }
}

impl<T, const N: usize> VectorView<T> for [T; N] {

    fn len(&self) -> usize {
        N
    }

    fn at(&self, i: usize) -> &T {
        &self[i]
    }
}

impl<T, const N: usize> VectorViewMut<T> for [T; N] {

    fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self[i]
    }
}

impl<T, const N: usize> SwappableVectorViewMut<T> for [T; N] {

    fn swap(&mut self, i: usize, j: usize) {
        <[T]>::swap(self, i, j)
    }
}

impl VectorFn<usize> for Range<usize> {

    fn at(&self, i: usize) -> usize {
        assert!(i < <_ as VectorFn<_>>::len(self));
        self.start + i
    }

    fn len(&self) -> usize {
        self.end - self.start
    }
}

///
/// A wrapper around a [`RingStore`] that is callable with signature `(&El<R>) -> El<R>`, 
/// and will clone the given ring element when called.
/// 
/// In order to be compatible with [`crate::iters::multi_cartesian_product()`], it
/// additionally is also callable with signature `(usize, &El<R>) -> El<R>`. In this
/// case, the first parameter is ignored.
/// 
#[derive(Copy, Clone)]
pub struct CloneRingEl<R: RingStore>(pub R);

impl<'a, R: RingStore> FnOnce<(&'a El<R>,)> for CloneRingEl<R> {

    type Output = El<R>;

    extern "rust-call" fn call_once(self, args: (&'a El<R>,)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, R: RingStore> FnMut<(&'a El<R>,)> for CloneRingEl<R> {

    extern "rust-call" fn call_mut(&mut self, args: (&'a El<R>,)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, R: RingStore> Fn<(&'a El<R>,)> for CloneRingEl<R> {

    extern "rust-call" fn call(&self, args: (&'a El<R>,)) -> Self::Output {
        self.0.clone_el(args.0)
    }
}

impl<'a, R: RingStore> FnOnce<(usize, &'a El<R>,)> for CloneRingEl<R> {

    type Output = El<R>;

    extern "rust-call" fn call_once(self, args: (usize, &'a El<R>,)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, R: RingStore> FnMut<(usize, &'a El<R>,)> for CloneRingEl<R> {

    extern "rust-call" fn call_mut(&mut self, args: (usize, &'a El<R>,)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, R: RingStore> Fn<(usize, &'a El<R>,)> for CloneRingEl<R> {

    extern "rust-call" fn call(&self, args: (usize, &'a El<R>,)) -> Self::Output {
        self.0.clone_el(args.1)
    }
}

#[test]
fn test_vector_fn_iter() {
    let vec = vec![1, 2, 4, 8, 16];
    assert_eq!(vec, vec.as_fn().into_iter().copied().collect::<Vec<_>>());
}