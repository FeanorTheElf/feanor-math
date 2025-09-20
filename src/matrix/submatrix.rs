use std::fmt::{Formatter, Debug};
use std::ops::{Deref, Range};
use std::marker::PhantomData;
use std::ptr::{addr_of_mut, NonNull};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, DataMut, Ix2};

use crate::seq::{VectorView, VectorViewMut, SwappableVectorViewMut};

///
/// Trait for objects that can be considered a contiguous part of memory. In particular,
/// the pointer returned by `get_pointer()` should be interpreted as the pointer to the first
/// element of a range of elements of type `T` (basically a C-style array). In some
/// sense, this is thus the unsafe equivalent of `Deref<Target = [T]>`.
/// 
/// # Safety
/// 
/// Since we use this to provide iterators that do not follow the natural layout of
/// the data, the following restrictions are necessary:
///  - Calling multiple times `get_pointer()` on the same reference is valid, and
///    all resulting pointers are valid to be dereferenced.
///  - In the above situation, we may also keep multiple mutable references that were
///    obtained by dereferencing the pointers, *as long as they don't alias, i.e. refer to 
///    different elements*. 
///  - If `Self: Sync` then `T: Send` and the above situation should be valid even if
///    the pointers returned by `get_pointer()` are produced and used from different threads. 
/// 
pub unsafe trait AsPointerToSlice<T> {

    ///
    /// Returns a pointer to the first element of multiple, contiguous `T`s.
    /// 
    /// # Safety
    /// 
    /// Requires that `self_` is a pointer to a valid object of this type. Note that
    /// it is legal to call this function while there exist mutable references to `T`s
    /// that were obtained by dereferencing an earlier result of `get_pointer()`. This
    /// means that in some situations, the passed `self_` may not be dereferenced without
    /// violating the aliasing rules.
    /// 
    /// However, it must be guaranteed that no mutable reference to an part of `self_` that
    /// is not pointed to by a result of `get_pointer()` exists. Immutable references may exist.
    /// 
    /// For additional detail, see the trait-level doc [`AsPointerToSlice`].
    /// 
    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T>;
}

unsafe impl<T> AsPointerToSlice<T> for Vec<T> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        // Safe, because "This method guarantees that for the purpose of the aliasing model, this 
        // method does not materialize a reference to the underlying slice" (quote from the doc of 
        // [`Vec::as_mut_ptr()`])
        unsafe {
            NonNull::new((*self_.as_ptr()).as_mut_ptr()).unwrap()
        }
    }
}

///
/// Newtype for `[T; SIZE]` that implements `Deref<Target = [T]>` so that it can be used
/// to store columns and access them through [`Submatrix`].
/// 
/// This is necessary, since [`Submatrix::from_2d`] requires that `V: Deref<Target = [T]>`.
/// 
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DerefArray<T, const SIZE: usize> {
    /// The wrapped array
    pub data: [T; SIZE]
}

impl<T: std::fmt::Debug, const SIZE: usize> std::fmt::Debug for DerefArray<T, SIZE> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<T, const SIZE: usize> From<[T; SIZE]> for DerefArray<T, SIZE> {

    fn from(value: [T; SIZE]) -> Self {
        Self { data: value }
    }
}

impl<'a, T, const SIZE: usize> From<&'a [T; SIZE]> for &'a DerefArray<T, SIZE> {

    fn from(value: &'a [T; SIZE]) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<'a, T, const SIZE: usize> From<&'a mut [T; SIZE]> for &'a mut DerefArray<T, SIZE> {

    fn from(value: &'a mut [T; SIZE]) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<T, const SIZE: usize> Deref for DerefArray<T, SIZE> {

    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data[..]
    }
}

unsafe impl<T, const SIZE: usize> AsPointerToSlice<T> for DerefArray<T, SIZE> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        unsafe { 
            let self_ptr = self_.as_ptr();
            let data_ptr = addr_of_mut!((*self_ptr).data);
            NonNull::new((*data_ptr).as_mut_ptr()).unwrap()
        }
    }
}

///
/// Represents a contiguous batch of `T`s by their first element.
/// In other words, a pointer to the batch is equal to a pointer to 
/// the first value.
/// 
#[repr(transparent)]
#[allow(missing_debug_implementations)]
pub struct AsFirstElement<T>(T);

unsafe impl<'a, T> AsPointerToSlice<T> for AsFirstElement<T> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        unsafe { std::mem::transmute(self_) }
    }
}

///
/// A submatrix that works on raw pointers, thus does not care about mutability
/// and borrowing. It already takes care about bounds checking and indexing.
/// Nevertheless, it is quite difficult to use this correctly, best not use it at
/// all. I mainly made it public to allow doctests.
/// 
/// More concretely, when having a 2d-structure, given by a sequence of `V`s, we
/// can consider a rectangular sub-block. This is encapsulated by SubmatrixRaw.
/// 
/// # Safety
/// 
/// The individual safety contracts are described at the corresponding functions.
/// However, in total be careful when actuall transforming the entry pointers
/// (given by [`SubmatrixRaw::entry_at`] or [`SubmatrixRaw::row_at`]) into references.
/// Since `SubmatrixRaw` does not borrow-check (and is `Copy`!), it is easy to create
/// aliasing pointers, that must not be converted into references.
/// 
/// ## Example of illegal use
/// ```rust
/// # use feanor_math::matrix::*;
/// # use core::ptr::NonNull;
/// let mut data = [1, 2, 3];
/// // this is actuall safe and intended use
/// let mut matrix = unsafe { SubmatrixRaw::<AsFirstElement<i64>, i64>::new(std::mem::transmute(NonNull::new(data.as_mut_ptr()).unwrap()), 1, 3, 0, 3) };
/// // this is safe, but note that ptr1 and ptr2 alias...
/// let mut ptr1: NonNull<i64> = matrix.entry_at(0, 0);
/// let mut ptr2: NonNull<i64> = matrix.entry_at(0, 0);
/// // this is UB now!
/// let (ref1, ref2) = unsafe { (ptr1.as_mut(), ptr2.as_mut()) };
/// ```
///
#[stability::unstable(feature = "enable")]
pub struct SubmatrixRaw<V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<*mut T>,
    rows: NonNull<V>,
    row_count: usize,
    row_step: isize,
    col_start: usize,
    col_count: usize
}

///
/// Requiring `T: Sync` is the more conservative choice. If `SubmatrixRaw`
/// acts as a mutable reference, we would only require `T: Send`, but we also
/// want `SubmatrixRaw` to be usable as an immutable reference, thus it can be
/// shared between threads, which requires `T: Sync`.
/// 
unsafe impl<V, T> Send for SubmatrixRaw<V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Sync
{}

unsafe impl<V, T> Sync for SubmatrixRaw<V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Sync
{}

impl<V, T> Clone for SubmatrixRaw<V, T> 
    where V: AsPointerToSlice<T>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, T> Copy for SubmatrixRaw<V, T> 
    where V: AsPointerToSlice<T>
{}

impl<V, T> SubmatrixRaw<V, T> 
    where V: AsPointerToSlice<T>
{
    ///
    /// Create a new SubmatrixRaw object.
    /// 
    /// # Safety
    /// 
    /// We require that each pointer `rows.offset(row_step * i)` for `0 <= i < row_count` points to a
    /// valid object and can be dereferenced. Furthermore, if `ptr` is the pointer returned by `[`AsPointerToSlice::get_pointer()`]`,
    /// then `ptr.offset(i + cols_start)` must point to a valid `T` for `0 <= i < col_count`.
    /// 
    /// Furthermore, we require any two of these (for different i) to represent disjunct "slices", i.e. if they
    /// give pointers `ptr1` and `ptr2` (via [`AsPointerToSlice::get_pointer()`]), then `ptr1.offset(cols_start + k)` and
    /// `ptr2.offset(cols_start + l)` for `0 <= k, l < col_count` never alias.
    /// 
    #[stability::unstable(feature = "enable")]
    pub unsafe fn new(rows: NonNull<V>, row_count: usize, row_step: isize, cols_start: usize, col_count: usize) -> Self {
        Self {
            entry: PhantomData,
            row_count: row_count,
            row_step: row_step,
            rows: rows,
            col_start: cols_start,
            col_count
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_rows(mut self, rows: Range<usize>) -> Self {
        assert!(rows.start <= rows.end);
        assert!(rows.end <= self.row_count);
        // this is safe since we require (during the constructor) that all pointers `rows.offset(i * row_step)`
        // are valid for `0 <= i < row_count`. Technically, this is not completely legal, as in the case 
        // `rows.start == rows.end == row_count`, the resulting pointer might point outside of the allocated area
        // - this is legal only when we are exactly one byte after it, but if `row_step` has a weird value, this does
        // not work. However, `row_step` has suitable values in all safe interfaces.
        unsafe {
            self.row_count = rows.end - rows.start;
            self.rows = self.rows.offset(rows.start as isize * self.row_step);
        }
        self
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_cols(mut self, cols: Range<usize>) -> Self {
        assert!(cols.end <= self.col_count);
        self.col_count = cols.end - cols.start;
        self.col_start += cols.start;
        self
    }

    ///
    /// Returns a pointer to the `row`-th row of the matrix.
    /// Be carefull about aliasing when making this into a reference!
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn row_at(&self, row: usize) -> NonNull<[T]> {
        assert!(row < self.row_count);
        // this is safe since `row < row_count` and we require `rows.offset(row * row_step)` to point
        // to a valid element of `V`
        let row_ref = unsafe {
            V::get_pointer(self.rows.offset(row as isize * self.row_step))
        };
        // similarly safe by constructor requirements
        unsafe {
            NonNull::slice_from_raw_parts(row_ref.offset(self.col_start as isize), self.col_count)
        }
    }

    ///
    /// Returns a pointer to the `(row, col)`-th entry of the matrix.
    /// Be carefull about aliasing when making this into a reference!
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn entry_at(&self, row: usize, col: usize) -> NonNull<T> {
        assert!(row < self.row_count, "Row index {} out of range 0..{}", row, self.row_count);
        assert!(col < self.col_count, "Col index {} out of range 0..{}", col, self.col_count);
        // this is safe since `row < row_count` and we require `rows.offset(row * row_step)` to point
        // to a valid element of `V`
        let row_ref = unsafe {
            V::get_pointer(self.rows.offset(row as isize * self.row_step))
        };
        // similarly safe by constructor requirements
        unsafe {
            row_ref.offset(self.col_start as isize + col as isize)
        }
    }
}

///
/// An immutable view on a column of a matrix [`Submatrix`]. 
/// 
pub struct Column<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a T>,
    raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T> Column<'a, V, T>
    where V: AsPointerToSlice<T>
{
    ///
    /// Creates a new column object representing the given submatrix. Thus,
    /// the submatrix must only have one column.
    /// 
    /// # Safety
    /// 
    /// Since `Column` represents immutable borrowing, callers of this method
    /// must ensure that for the lifetime `'a`, there are no mutable references
    /// to any object pointed to by `raw_data` (this includes both the "row descriptors"
    /// `V` and the actual elements `T`).
    /// 
    unsafe fn new(raw_data: SubmatrixRaw<V, T>) -> Self {
        assert!(raw_data.col_count == 1);
        Self {
            entry: PhantomData,
            raw_data: raw_data
        }
    }
}

impl<'a, V, T: Debug> Debug for Column<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut output = f.debug_list();
        for x in self.as_iter() {
            _ = output.entry(x);
        }
        output.finish()
    }
}

impl<'a, V, T> Clone for Column<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V, T> Copy for Column<'a, V, T>
    where V: AsPointerToSlice<T>
{}

impl<'a, V, T> VectorView<T> for Column<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn len(&self) -> usize {
        self.raw_data.row_count
    }

    fn at(&self, i: usize) -> &T {
        // safe since we assume that there are no mutable references to `raw_data` 
        unsafe {
            self.raw_data.entry_at(i, 0).as_ref()
        }
    }
}

///
///
/// A mutable view on a column of a matrix [`SubmatrixMut`]. 
/// 
/// Clearly must not be Copy/Clone.
/// 
pub struct ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T: Debug> Debug for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_const().fmt(f)
    }
}

impl<'a, V, T> ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    ///
    /// Creates a new column object representing the given submatrix. Thus,
    /// the submatrix must only have one column.
    /// 
    /// # Safety
    /// 
    /// Since `ColumnMut` represents mutable borrowing, callers of this method
    /// must ensure that for the lifetime `'a`, there are no other references
    /// to any matrix entry pointed to by `raw_data` (meaning the "content" elements
    /// `T`). It is allowed to have immutable references to the "row descriptors" `V`
    /// (assuming they are strictly different from the content `T`).
    /// 
    unsafe fn new(raw_data: SubmatrixRaw<V, T>) -> Self {
        assert!(raw_data.col_count == 1);
        Self {
            entry: PhantomData,
            raw_data: raw_data
        }
    }
    
    ///
    /// "Reborrows" the [`ColumnMut`], which is somewhat like cloning the submatrix, but
    /// disallows the cloned object to be used while its copy is alive. This is necessary
    /// to follow the aliasing rules of Rust.
    /// 
    pub fn reborrow<'b>(&'b mut self) -> ColumnMut<'b, V, T> {
        ColumnMut {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    ///
    /// Returns mutable references to two different entries of the column.
    /// 
    /// This requires `i != j`, otherwise the function will panic. This
    /// is necessary, since otherwise the two references would violate the
    /// borrowing rules of Rust.
    /// 
    pub fn two_entries<'b>(&'b mut self, i: usize, j: usize) -> (&'b mut T, &'b mut T) {
        assert!(i != j);
        // safe since i != j
        unsafe {
            (self.raw_data.entry_at(i, 0).as_mut(), self.raw_data.entry_at(j, 0).as_mut())
        }
    }
    
    ///
    /// Returns an immutable view on this matrix column.
    /// 
    pub fn as_const<'b>(&'b self) -> Column<'b, V, T> {
        Column {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }
}

unsafe impl<'a, V, T> Send for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T> + Sync, T: Send
{}

impl<'a, V, T> VectorView<T> for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn len(&self) -> usize {
        self.raw_data.row_count
    }

    fn at(&self, i: usize) -> &T {
        // safe since we assume that there are no other references to `raw_data` 
        unsafe {
            self.raw_data.entry_at(i, 0).as_ref()
        }
    }
}

///
/// Iterator over mutable references to the entries of a column
/// of a matrix [`SubmatrixMut`].
/// 
#[allow(missing_debug_implementations)]
pub struct ColumnMutIter<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    column_mut: ColumnMut<'a, V, T>
}

impl<'a, V, T> Iterator for ColumnMutIter<'a, V, T>
    where V: AsPointerToSlice<T>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.column_mut.raw_data.row_count > 0 {
            let mut result = self.column_mut.raw_data.entry_at(0, 0);
            self.column_mut.raw_data = self.column_mut.raw_data.restrict_rows(1..self.column_mut.raw_data.row_count);
            // safe since for the result lifetime, one cannot legally access this value using only the new value of `self.column_mut.raw_data`
            unsafe {
                Some(result.as_mut())
            }
        } else {
            None
        }
    }
}

impl<'a, V, T> IntoIterator for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    type Item = &'a mut T;
    type IntoIter = ColumnMutIter<'a, V, T>;

    fn into_iter(self) -> ColumnMutIter<'a, V, T> {
        ColumnMutIter { column_mut: self }
    }
}

impl<'a, V, T> VectorViewMut<T> for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn at_mut<'b>(&'b mut self, i: usize) -> &'b mut T {
        // safe since self is borrow mutably
        unsafe {
            self.raw_data.entry_at(i, 0).as_mut()
        }
    }
}

impl<'a, V, T> SwappableVectorViewMut<T> for ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    fn swap(&mut self, i: usize, j: usize) {
        if i != j {
            // safe since i != j, so these pointers do not alias; I think it is also safe for
            // zero sized type, even though it is slightly weird since the pointer might point
            // to the same location even if i != j
            unsafe {
                std::mem::swap(self.raw_data.entry_at(i, 0).as_mut(), self.raw_data.entry_at(j, 0).as_mut());
            }
        }
    }
}

///
/// Immutable view on a matrix that stores elements of type `T`.
/// 
/// Note that a [`Submatrix`] never owns the data, but always behaves like
/// a reference. In particular, this means that [`Submatrix`] is `Copy`,
/// unconditionally. The equivalent to a mutable reference is given by
/// [`SubmatrixMut`].
/// 
/// This view is designed to work with various underlying representations
/// of the matrix, as described by [`AsPointerToSlice`].
/// 
pub struct Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    entry: PhantomData<&'a T>,
    raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T> Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    ///
    /// Returns the submatrix that references only the entries whose row resp. column
    /// indices are within the given ranges.
    /// 
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        self.restrict_rows(rows).restrict_cols(cols)
    }

    ///
    /// Returns the submatrix that references only the entries whose row indices are within the given range.
    /// 
    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_rows(rows)
        }
    }

    ///
    /// Consumes the submatrix and produces a reference to its `(i, j)`-th entry.
    /// 
    /// In most cases, you will use [`Submatrix::at()`] instead, but in rare occasions,
    /// `into_at()` might be necessary to provide a reference whose lifetime is not
    /// coupled to the lifetime of the submatrix object.
    /// 
    pub fn into_at(self, i: usize, j: usize) -> &'a T {
        &self.into_row_at(i)[j]
    }
    
    ///
    /// Returns a reference to the `(i, j)`-th entry of this matrix.
    /// 
    pub fn at<'b>(&'b self, i: usize, j: usize) -> &'b T {
        &self.row_at(i)[j]
    }

    ///
    /// Returns the submatrix that references only the entries whose column indices are within the given range.
    /// 
    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_cols(cols)
        }
    }

    ///
    /// Returns an iterator over all rows of the matrix.
    /// 
    pub fn row_iter(self) -> impl 'a + Clone + ExactSizeIterator<Item = &'a [T]> {
        (0..self.raw_data.row_count).map(move |i| 
        // safe since there are no immutable references to self.raw_data    
        unsafe {
            self.raw_data.row_at(i).as_ref()
        })
    }

    ///
    /// Returns an iterator over all columns of the matrix.
    /// 
    pub fn col_iter(self) -> impl 'a + Clone + ExactSizeIterator<Item = Column<'a, V, T>> {
        (0..self.raw_data.col_count).map(move |j| {
            debug_assert!(j < self.raw_data.col_count);
            let mut result_raw = self.raw_data;
            result_raw.col_start += j;
            result_raw.col_count = 1;
            // safe since there are no immutable references to self.raw_data
            unsafe {
                return Column::new(result_raw);
            }
        })
    }

    ///
    /// Consumes the submatrix and produces a reference to its `i`-th row.
    /// 
    /// In most cases, you will use [`Submatrix::row_at()`] instead, but in rare occasions,
    /// `into_row_at()` might be necessary to provide a reference whose lifetime is not
    /// coupled to the lifetime of the submatrix object.
    /// 
    pub fn into_row_at(self, i: usize) -> &'a [T] {
        // safe since there are no mutable references to self.raw_data
        unsafe {
            self.raw_data.row_at(i).as_ref()
        }
    }

    ///
    /// Returns a view on the `i`-th row of this matrix.
    /// 
    pub fn row_at<'b>(&'b self, i: usize) -> &'b [T] {
        // safe since there are no immutable references to self.raw_data
        unsafe {
            self.raw_data.row_at(i).as_ref()
        }
    }

    ///
    /// Consumes the submatrix and produces a reference to its `j`-th column.
    /// 
    /// In most cases, you will use [`Submatrix::col_at()`] instead, but in rare occasions,
    /// `into_col_at()` might be necessary to provide a reference whose lifetime is not
    /// coupled to the lifetime of the submatrix object.
    /// 
    pub fn into_col_at(self, j: usize) -> Column<'a, V, T> {
        assert!(j < self.raw_data.col_count);
        let mut result_raw = self.raw_data;
        result_raw.col_start += j;
        result_raw.col_count = 1;
        // safe since there are no immutable references to self.raw_data
        unsafe {
            return Column::new(result_raw);
        }
    }

    ///
    /// Returns a view on the `j`-th column of this matrix.
    /// 
    pub fn col_at<'b>(&'b self, j: usize) -> Column<'b, V, T> {
        assert!(j < self.raw_data.col_count);
        let mut result_raw = self.raw_data;
        result_raw.col_start += j;
        result_raw.col_count = 1;
        // safe since there are no immutable references to self.raw_data
        unsafe {
            return Column::new(result_raw);
        }
    }

    ///
    /// Returns the number of columns of this matrix.
    /// 
    pub fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    ///
    /// Returns the number of rows of this matrix.
    /// 
    pub fn row_count(&self) -> usize {
        self.raw_data.row_count
    }
}

impl<'a, V, T: Debug> Debug for Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut output = f.debug_list();
        for row in self.row_iter() {
            _ = output.entry(&row);
        }
        output.finish()
    }
}

impl<'a, V, T> Clone for Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V, T> Copy for Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{}

///
/// Mutable view on a matrix that stores elements of type `T`.
/// 
/// As for [`Submatrix`], a [`SubmatrixMut`] never owns the data, but
/// behaves similarly to a mutable reference.
/// 
/// This view is designed to work with various underlying representations
/// of the matrix, as described by [`AsPointerToSlice`].
/// 
pub struct SubmatrixMut<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T> SubmatrixMut<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    ///
    /// Returns the submatrix that references only the entries whose row resp. column
    /// indices are within the given ranges.
    /// 
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        self.restrict_rows(rows).restrict_cols(cols)
    }

    ///
    /// Returns the submatrix that references only the entries whose row indices are within the given range.
    /// 
    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_rows(rows)
        }
    }

    ///
    /// Returns the submatrix that references only the entries whose column indices are within the given range.
    /// 
    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_cols(cols)
        }
    }

    ///
    /// Returns the two submatrices referencing the entries whose row index is within the given ranges.
    /// 
    /// The ranges must not overlap, so that the returned matrices don't alias the same entry.
    /// 
    pub fn split_rows(self, fst_rows: Range<usize>, snd_rows: Range<usize>) -> (Self, Self) {
        assert!(fst_rows.end <= snd_rows.start || snd_rows.end <= fst_rows.start);
        (
            Self {
                entry: PhantomData,
                raw_data: self.raw_data.restrict_rows(fst_rows)
            },
            Self {
                entry: PhantomData,
                raw_data: self.raw_data.restrict_rows(snd_rows)
            },
        )
    }

    ///
    /// Returns the two submatrices referencing the entries whose column index is within the given ranges.
    /// 
    /// The ranges must not overlap, so that the returned matrices don't alias the same entry.
    /// 
    pub fn split_cols(self, fst_cols: Range<usize>, snd_cols: Range<usize>) -> (Self, Self) {
        assert!(fst_cols.end <= snd_cols.start || snd_cols.end <= fst_cols.start);
        (
            Self {
                entry: PhantomData,
                raw_data: self.raw_data.restrict_cols(fst_cols)
            },
            Self {
                entry: PhantomData,
                raw_data: self.raw_data.restrict_cols(snd_cols)
            },
        )
    }

    ///
    /// Returns an iterator over (mutable views onto) the rows of this matrix.
    /// 
    pub fn row_iter(self) -> impl 'a + ExactSizeIterator<Item = &'a mut [T]> {
        (0..self.raw_data.row_count).map(move |i| 
        // safe since each access goes to a different location
        unsafe {
            self.raw_data.row_at(i).as_mut()
        })
    }

    ///
    /// Returns an iterator over (mutable views onto) the columns of this matrix.
    /// 
    pub fn col_iter(self) -> impl 'a + ExactSizeIterator<Item = ColumnMut<'a, V, T>> {
        (0..self.raw_data.col_count).map(move |j| {
            let mut result_raw = self.raw_data;
            result_raw.col_start += j;
            result_raw.col_count = 1;
            // safe since each time, the `result_raw` don't overlap
            unsafe {
                return ColumnMut::new(result_raw);
            }
        })
    }

    ///
    /// Consumes the submatrix and produces a reference to its `(i, j)`-th entry.
    /// 
    /// In most cases, you will use [`SubmatrixMut::at_mut()`] instead, but in rare occasions,
    /// `into_at_mut()` might be necessary to provide a reference whose lifetime is not
    /// coupled to the lifetime of the submatrix object.
    /// 
    pub fn into_at_mut(self, i: usize, j: usize) -> &'a mut T {
        &mut self.into_row_mut_at(i)[j]
    }

    ///
    /// Returns a mutable reference to the `(i, j)`-th entry of this matrix.
    /// 
    pub fn at_mut<'b>(&'b mut self, i: usize, j: usize) -> &'b mut T {
        &mut self.row_mut_at(i)[j]
    }

    ///
    /// Returns a reference to the `(i, j)`-th entry of this matrix.
    /// 
    pub fn at<'b>(&'b self, i: usize, j: usize) -> &'b T {
        self.as_const().into_at(i, j)
    }

    ///
    /// Returns a view onto the `i`-th row of this matrix.
    /// 
    pub fn row_at<'b>(&'b self, i: usize) -> &'b [T] {
        self.as_const().into_row_at(i)
    }

    ///
    /// Consumes the submatrix and produces a reference to its `i`-th row.
    /// 
    /// In most cases, you will use [`SubmatrixMut::row_mut_at()`] instead, but in rare occasions,
    /// `into_row_mut_at()` might be necessary to provide a reference whose lifetime is not
    /// coupled to the lifetime of the submatrix object.
    /// 
    pub fn into_row_mut_at(self, i: usize) -> &'a mut [T] {
        // safe since self is exists borrowed for 'a
        unsafe {
            self.raw_data.row_at(i).as_mut()
        }
    }

    ///
    /// Returns a mutable view onto the `i`-th row of this matrix.
    /// 
    pub fn row_mut_at<'b>(&'b mut self, i: usize) -> &'b mut [T] {
        self.reborrow().into_row_mut_at(i)
    }

    ///
    /// Returns a view onto the `i`-th column of this matrix.
    /// 
    pub fn col_at<'b>(&'b self, j: usize) -> Column<'b, V, T> {
        self.as_const().into_col_at(j)
    }

    ///
    /// Returns a mutable view onto the `j`-th row of this matrix.
    /// 
    pub fn col_mut_at<'b>(&'b mut self, j: usize) -> ColumnMut<'b, V, T> {
        assert!(j < self.raw_data.col_count);
        let mut result_raw = self.raw_data;
        result_raw.col_start += j;
        result_raw.col_count = 1;
        // safe since self is mutably borrowed for 'b
        unsafe {
            return ColumnMut::new(result_raw);
        }
    }

    ///
    /// "Reborrows" the [`SubmatrixMut`], which is somewhat like cloning the submatrix, but
    /// disallows the cloned object to be used while its copy is alive. This is necessary
    /// to follow the aliasing rules of Rust.
    /// 
    pub fn reborrow<'b>(&'b mut self) -> SubmatrixMut<'b, V, T> {
        SubmatrixMut {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    ///
    /// Returns an immutable view on the data of this matrix.
    /// 
    pub fn as_const<'b>(&'b self) -> Submatrix<'b, V, T> {
        Submatrix {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    ///
    /// Returns the number of columns of this matrix.
    /// 
    pub fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    ///
    /// Returns the number of rows of this matrix.
    /// 
    pub fn row_count(&self) -> usize {
        self.raw_data.row_count
    }
}

impl<'a, V, T: Debug> Debug for SubmatrixMut<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_const().fmt(f)
    }
}

impl<'a, T> SubmatrixMut<'a, AsFirstElement<T>, T> {

    ///
    /// Creates a view on the given data slice, interpreting it as a matrix of given shape.
    /// Assumes row-major order, i.e. contigous subslices of `data` will be the rows of the
    /// resulting matrix.
    /// 
    pub fn from_1d(data: &'a mut [T], row_count: usize, col_count: usize) -> Self {
        assert_eq!(row_count * col_count, data.len());
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(std::mem::transmute(NonNull::new(data.as_mut_ptr()).unwrap_unchecked()), row_count, col_count as isize, 0, col_count)
            }
        }
    }

    #[doc(cfg(feature = "ndarray"))]
    #[cfg(feature = "ndarray")]
    pub fn from_ndarray<S>(data: &'a mut ArrayBase<S, Ix2>) -> Self
        where S: DataMut<Elem = T>
    {
        assert!(data.is_standard_layout());
        let (nrows, ncols) = (data.nrows(), data.ncols());
        return Self::new(data.as_slice_mut().unwrap(), nrows, ncols);
    }
}

impl<'a, V: AsPointerToSlice<T> + Deref<Target = [T]>, T> SubmatrixMut<'a, V, T> {

    ///
    /// Interprets the given slice of slices as a matrix, by using the elements
    /// of the outer slice as the rows of the matrix.
    /// 
    pub fn from_2d(data: &'a mut [V]) -> Self {
        assert!(data.len() > 0);
        let row_count = data.len();
        let col_count = data[0].len();
        for row in data.iter() {
            assert_eq!(col_count, row.len());
        }
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(NonNull::new(data.as_mut_ptr() as *mut _).unwrap_unchecked(), row_count, 1, 0, col_count)
            }
        }
    }
}

impl<'a, T> Submatrix<'a, AsFirstElement<T>, T> {

    ///
    /// Creates a view on the given data slice, interpreting it as a matrix of given shape.
    /// Assumes row-major order, i.e. contigous subslices of `data` will be the rows of the
    /// resulting matrix.
    /// 
    pub fn from_1d(data: &'a [T], row_count: usize, col_count: usize) -> Self {
        assert_eq!(row_count * col_count, data.len());
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(std::mem::transmute(NonNull::new(data.as_ptr() as *mut T).unwrap_unchecked()), row_count, col_count as isize, 0, col_count)
            }
        }
    }

    #[doc(cfg(feature = "ndarray"))]
    #[cfg(feature = "ndarray")]
    pub fn from_ndarray<S>(data: &'a ArrayBase<S, Ix2>) -> Self
        where S: DataMut<Elem = T>
    {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        return Self::new(data.as_slice().unwrap(), nrows, ncols);
    }
}

impl<'a, V: AsPointerToSlice<T> + Deref<Target = [T]>, T> Submatrix<'a, V, T> {

    ///
    /// Interprets the given slice of slices as a matrix, by using the elements
    /// of the outer slice as the rows of the matrix.
    /// 
    pub fn from_2d(data: &'a [V]) -> Self {
        assert!(data.len() > 0);
        let row_count = data.len();
        let col_count = data[0].len();
        for row in data.iter() {
            assert_eq!(col_count, row.len());
        }
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(NonNull::new(data.as_ptr() as *mut _).unwrap_unchecked(), row_count, 1, 0, col_count)
            }
        }
    }
}

#[cfg(test)]
fn assert_submatrix_eq<V: AsPointerToSlice<T>, T: PartialEq + Debug, const N: usize, const M: usize>(expected: [[T; M]; N], actual: &mut SubmatrixMut<V, T>) {
    assert_eq!(N, actual.row_count());
    assert_eq!(M, actual.col_count());
    for i in 0..N {
        for j in 0..M {
            assert_eq!(&expected[i][j], actual.at(i, j));
            assert_eq!(&expected[i][j], actual.as_const().at(i, j));
        }
    }
}

#[cfg(test)]
fn with_testmatrix_vec<F>(f: F)
    where F: FnOnce(SubmatrixMut<Vec<i64>, i64>)
{
    let mut data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15]
    ];
    let matrix = SubmatrixMut::<Vec<_>, _>::from_2d(&mut data[..]);
    f(matrix)
}

#[cfg(test)]
fn with_testmatrix_array<F>(f: F)
    where F: FnOnce(SubmatrixMut<DerefArray<i64, 5>, i64>)
{
    let mut data = vec![
        DerefArray::from([1, 2, 3, 4, 5]),
        DerefArray::from([6, 7, 8, 9, 10]),
        DerefArray::from([11, 12, 13, 14, 15])
    ];
    let matrix = SubmatrixMut::<DerefArray<_, 5>, _>::from_2d(&mut data[..]);
    f(matrix)
}

#[cfg(test)]
fn with_testmatrix_linmem<F>(f: F)
    where F: FnOnce(SubmatrixMut<AsFirstElement<i64>, i64>)
{
    let mut data = vec![
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ];
    let matrix = SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut data[..], 3, 5);
    f(matrix)
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
fn with_testmatrix_ndarray<F>(f: F)
    where F: FnOnce(SubmatrixMut<AsFirstElement<i64>, i64>)
{
    use ndarray::array;

    let mut data = array![
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ];
    let matrix = SubmatrixMut::<AsFirstElement<_>, _>::from_ndarray(&mut data);
    f(matrix)
}

#[cfg(not(feature = "ndarray"))]
#[cfg(test)]
fn with_testmatrix_ndarray<F>(_: F)
    where F: FnOnce(SubmatrixMut<AsFirstElement<i64>, i64>)
{
    // do nothing
}

#[cfg(test)]
fn test_submatrix<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    assert_submatrix_eq([[2, 3], [7, 8]], &mut matrix.reborrow().submatrix(0..2, 1..3));
    assert_submatrix_eq([[8, 9, 10]], &mut matrix.reborrow().submatrix(1..2, 2..5));
    assert_submatrix_eq([[8, 9, 10], [13, 14, 15]], &mut matrix.reborrow().submatrix(1..3, 2..5));

    let (mut left, mut right) = matrix.split_cols(0..3, 3..5);
    assert_submatrix_eq([[1, 2, 3], [6, 7, 8], [11, 12, 13]], &mut left);
    assert_submatrix_eq([[4, 5], [9, 10], [14, 15]], &mut right);
}

#[test]
fn test_submatrix_wrapper() {
    with_testmatrix_vec(test_submatrix);
    with_testmatrix_array(test_submatrix);
    with_testmatrix_linmem(test_submatrix);
    with_testmatrix_ndarray(test_submatrix);
}

#[cfg(test)]
fn test_submatrix_mutate<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    let (mut left, mut right) = matrix.split_cols(0..3, 3..5);
    assert_submatrix_eq([[1, 2, 3], [6, 7, 8], [11, 12, 13]], &mut left);
    assert_submatrix_eq([[4, 5], [9, 10], [14, 15]], &mut right);
    *left.at_mut(1, 1) += 1;
    *right.at_mut(0, 0) += 1;
    *right.at_mut(2, 1) += 1;
    assert_submatrix_eq([[1, 2, 3], [6, 8, 8], [11, 12, 13]], &mut left);
    assert_submatrix_eq([[5, 5], [9, 10], [14, 16]], &mut right);

    let (mut top, mut bottom) = left.split_rows(0..1, 1..3);
    assert_submatrix_eq([[1, 2, 3]], &mut top);
    assert_submatrix_eq([[6, 8, 8], [11, 12, 13]], &mut bottom);
    *top.at_mut(0, 0) -= 1;
    *top.at_mut(0, 2) += 3;
    *bottom.at_mut(0, 2) -= 1;
    *bottom.at_mut(1, 0) += 3;
    assert_submatrix_eq([[0, 2, 6]], &mut top);
    assert_submatrix_eq([[6, 8, 7], [14, 12, 13]], &mut bottom);
}

#[test]
fn test_submatrix_mutate_wrapper() {
    with_testmatrix_vec(test_submatrix_mutate);
    with_testmatrix_array(test_submatrix_mutate);
    with_testmatrix_linmem(test_submatrix_mutate);
    with_testmatrix_ndarray(test_submatrix_mutate);
}

#[cfg(test)]
fn test_submatrix_col_iter<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    {
        let mut it = matrix.reborrow().col_iter();
        assert_eq!(vec![2, 7, 12], it.by_ref().skip(1).next().unwrap().into_iter().map(|x| *x).collect::<Vec<_>>());
        assert_eq!(vec![4, 9, 14], it.by_ref().skip(1).next().unwrap().into_iter().map(|x| *x).collect::<Vec<_>>());
        let mut last_col = it.next().unwrap();
        for x in last_col.reborrow() {
            *x *= 2;
        }
        assert_eq!(vec![10, 20, 30], last_col.into_iter().map(|x| *x).collect::<Vec<_>>());
    }
    assert_submatrix_eq([
        [1, 2, 3, 4, 10],
        [6, 7, 8, 9, 20],
        [11, 12, 13, 14, 30]], 
        &mut matrix
    );
    
    let (left, _right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = left.col_iter();
        let mut col1 = it.next().unwrap();
        let mut col2 = it.next().unwrap();
        assert!(it.next().is_none());
        assert_eq!(vec![1, 6, 11], col1.as_iter().map(|x| *x).collect::<Vec<_>>());
        assert_eq!(vec![2, 7, 12], col2.as_iter().map(|x| *x).collect::<Vec<_>>());
        assert_eq!(vec![1, 6, 11], col1.reborrow().into_iter().map(|x| *x).collect::<Vec<_>>());
        assert_eq!(vec![2, 7, 12], col2.reborrow().into_iter().map(|x| *x).collect::<Vec<_>>());
        *col1.into_iter().skip(1).next().unwrap() += 5;
    }
    assert_submatrix_eq([
        [1, 2, 3, 4, 10],
        [11, 7, 8, 9, 20],
        [11, 12, 13, 14, 30]], 
        &mut matrix
    );

    let (_left, right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = right.col_iter();
        let mut col = it.next().unwrap();
        assert!(it.next().is_none());
        assert_eq!(vec![4, 9, 14], col.reborrow().as_iter().map(|x| *x).collect::<Vec<_>>());
        *col.into_iter().next().unwrap() += 3;
    }
    assert_submatrix_eq([
        [1, 2, 3, 7, 10],
        [11, 7, 8, 9, 20],
        [11, 12, 13, 14, 30]], 
        &mut matrix
    );
}

#[test]
fn test_submatrix_col_iter_wrapper() {
    with_testmatrix_vec(test_submatrix_col_iter);
    with_testmatrix_array(test_submatrix_col_iter);
    with_testmatrix_linmem(test_submatrix_col_iter);
    with_testmatrix_ndarray(test_submatrix_col_iter);
}

#[cfg(test)]
fn test_submatrix_row_iter<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    {
        let mut it = matrix.reborrow().row_iter();
        assert_eq!(&[6, 7, 8, 9, 10], it.by_ref().skip(1).next().unwrap());
        let row = it.next().unwrap();
        assert!(it.next().is_none());
        row[1] += 6;
        row[4] *= 2;
    }
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 18, 13, 14, 30]], 
        &mut matrix
    );
    let (mut left, mut right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = left.reborrow().row_iter();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert_eq!(&[1, 2], row1);
        assert_eq!(&[6, 7], row2);
    }
    {
        let mut it = left.reborrow().row_iter();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert_eq!(&[1, 2], row1);
        assert_eq!(&[6, 7], row2);
        row2[1] += 1;
    }
    assert_submatrix_eq([[1, 2], [6, 8], [11, 18]], &mut left);
    {
        right = right.submatrix(1..3, 0..1);
        let mut it = right.reborrow().row_iter();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert_eq!(&[9], row1);
        assert_eq!(&[14], row2);
        row1[0] += 1;
    }
    assert_submatrix_eq([[10], [14]], &mut right);
}

#[test]
fn test_submatrix_row_iter_wrapper() {
    with_testmatrix_vec(test_submatrix_row_iter);
    with_testmatrix_array(test_submatrix_row_iter);
    with_testmatrix_linmem(test_submatrix_row_iter);
    with_testmatrix_ndarray(test_submatrix_row_iter);
}

#[cfg(test)]
fn test_submatrix_col_at<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    assert_eq!(&[2, 7, 12], &matrix.col_at(1).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[2, 7, 12], &matrix.as_const().col_at(1).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.as_const().col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);

    {
        let (mut top, mut bottom) = matrix.reborrow().restrict_rows(0..2).split_rows(0..1, 1..2);
        assert_eq!(&[1], &top.col_mut_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[1], &top.as_const().col_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[1], &top.col_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[5], &top.col_mut_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[5], &top.as_const().col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[5], &top.col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);

        assert_eq!(&[6], &bottom.col_mut_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[6], &bottom.as_const().col_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[6], &bottom.col_at(0).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[10], &bottom.col_mut_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[10], &bottom.as_const().col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[10], &bottom.col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
    }
}

#[test]
fn test_submatrix_col_at_wrapper() {
    with_testmatrix_vec(test_submatrix_col_at);
    with_testmatrix_array(test_submatrix_col_at);
    with_testmatrix_linmem(test_submatrix_col_at);
    with_testmatrix_ndarray(test_submatrix_col_at);
}

#[cfg(test)]
fn test_submatrix_row_at<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    assert_eq!(&[2, 7, 12], &matrix.col_at(1).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[2, 7, 12], &matrix.as_const().col_at(1).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.as_const().col_at(4).as_iter().copied().collect::<Vec<_>>()[..]);

    {
        let (mut left, mut right) = matrix.reborrow().restrict_cols(1..5).split_cols(0..2, 2..4);
        assert_eq!(&[2, 3], left.row_mut_at(0));
        assert_eq!(&[4, 5], right.row_mut_at(0));
        assert_eq!(&[2, 3], left.as_const().row_at(0));
        assert_eq!(&[4, 5], right.as_const().row_at(0));
        assert_eq!(&[2, 3], left.row_at(0));
        assert_eq!(&[4, 5], right.row_at(0));

        assert_eq!(&[7, 8], left.row_mut_at(1));
        assert_eq!(&[9, 10], right.row_mut_at(1));
        assert_eq!(&[7, 8], left.as_const().row_at(1));
        assert_eq!(&[9, 10], right.as_const().row_at(1));
        assert_eq!(&[7, 8], left.row_at(1));
        assert_eq!(&[9, 10], right.row_at(1));
    }
}

#[test]
fn test_submatrix_row_at_wrapper() {
    with_testmatrix_vec(test_submatrix_row_at);
    with_testmatrix_array(test_submatrix_row_at);
    with_testmatrix_linmem(test_submatrix_row_at);
    with_testmatrix_ndarray(test_submatrix_row_at);
}
