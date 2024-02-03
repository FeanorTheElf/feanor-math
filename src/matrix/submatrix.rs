use std::ops::Range;
use std::marker::PhantomData;
use std::ptr::NonNull;
use crate::ring::*;

use crate::vector::SwappableVectorViewMut;
use crate::vector::{VectorView, VectorViewMut};

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
///    obtained by dereferencing the pointers, *as long as they refer to different elements
///    of the slice*. 
///  - If `Self: Sync` then `T: Send` and the above situation should be valid even if
///    the pointers returned by `get_pointer()` are produced and used from different threads. 
/// 
pub unsafe trait AsPointerToSlice<T> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T>;
}

unsafe impl<T> AsPointerToSlice<T> for Vec<T> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        let self_ref = unsafe {
            self_.as_ref()
        };
        NonNull::new(self_ref.as_ptr() as *mut T).unwrap()
    }
}

unsafe impl<'a, T, const N: usize> AsPointerToSlice<T> for [T; N] {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        let self_ref = unsafe {
            self_.as_ref()
        };
        NonNull::new(self_ref.as_ptr() as *mut T).unwrap()
    }
}

///
/// Represents a contiguous batch of `T`s by their first element.
/// In other words, a pointer to the batch is equal to a pointer to 
/// the first value.
/// 
#[repr(transparent)]
pub struct AsFirstElement<T>(T);

unsafe impl<'a, T> AsPointerToSlice<T> for AsFirstElement<T> {

    unsafe fn get_pointer(self_: NonNull<Self>) -> NonNull<T> {
        std::mem::transmute(NonNull::from(self_))
    }
}

///
/// A submatrix that works on raw pointers, thus does not care about mutability
/// and borrowing. It already takes care about bounds checking and indexing.
/// Nevertheless, it is quite difficult to use this correctly, best not use it at
/// all. I mainly made it public to allow doctests.
/// 
/// More concretely, when having a 2d-structure, given by a sequence of `V`s, we
/// can consider a square sub-block. This is encapsulated by SubmatrixRaw.
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
/// ```
/// # use feanor_math::matrix::submatrix::*;
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
pub struct SubmatrixRaw<V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<*mut T>,
    pub rows: NonNull<V>,
    pub row_count: usize,
    pub row_step: isize,
    pub col_start: usize,
    pub col_count: usize
}

///
/// Requiring `T: Sync` is the more conservative choice. If `SubmatrixRaw`
/// acts as a mutable reference, we would only require `T: Send`, but we also
/// want `SubmatrixRaw` to be usable as an immutable reference, thus it can be
/// shared between threads, which requires `T: Sync`.
/// 
/// This makes implementing [`SubmatrixMut::concurrent_row_iter()`] and [`SubmatrixMut::concurrent_col_iter()`]
/// slightly more complicated.
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


    fn restrict_rows(mut self, rows: Range<usize>) -> Self {
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

    fn restrict_cols(mut self, cols: Range<usize>) -> Self {
        assert!(cols.end <= self.col_count);
        self.col_count = cols.end - cols.start;
        self.col_start += cols.start;
        self
    }

    ///
    /// Returns a pointer to the `row`-th row of the matrix.
    /// Be carefull about aliasing when making this into a reference!
    /// 
    fn row_at(&self, row: usize) -> NonNull<[T]> {
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
    pub fn entry_at(&self, row: usize, col: usize) -> NonNull<T> {
        assert!(row < self.row_count);
        assert!(col < self.col_count);
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
/// An immutable view on a column of a matrix. 
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
/// A mutable view on a column of a matrix. 
/// 
/// Clearly must not be Copy/Clone.
/// 
pub struct ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    raw_data: SubmatrixRaw<V, T>
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
    
    pub fn reborrow<'b>(&'b mut self) -> ColumnMut<'b, V, T> {
        ColumnMut {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    pub fn two_entries<'b>(&'b mut self, i: usize, j: usize) -> (&'b mut T, &'b mut T) {
        assert!(i != j);
        // safe since i != j
        unsafe {
            (self.raw_data.entry_at(i, 0).as_mut(), self.raw_data.entry_at(j, 0).as_mut())
        }
    }
    
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

pub struct Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    entry: PhantomData<&'a T>,
    raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T> Submatrix<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        self.restrict_rows(rows).restrict_cols(cols)
    }

    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_rows(rows)
        }
    }

    pub fn at<'b>(&'b self, i: usize, j: usize) -> &'b T {
        &self.row_at(i)[j]
    }

    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_cols(cols)
        }
    }

    pub fn row_iter(self) -> impl 'a + Clone + ExactSizeIterator<Item = &'a [T]> {
        (0..self.raw_data.row_count).map(move |i| 
        // safe since there are no immutable references to self.raw_data    
        unsafe {
            self.raw_data.row_at(i).as_ref()
        })
    }

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

    pub fn row_at<'b>(&'b self, i: usize) -> &'b [T] {
        // safe since there are no immutable references to self.raw_data
        unsafe {
            self.raw_data.row_at(i).as_ref()
        }
    }

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

    pub fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    pub fn row_count(&self) -> usize {
        self.raw_data.row_count
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

pub struct SubmatrixMut<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    pub raw_data: SubmatrixRaw<V, T>
}

impl<'a, V, T> SubmatrixMut<'a, V, T>
    where V: 'a + AsPointerToSlice<T>
{
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        self.restrict_rows(rows).restrict_cols(cols)
    }

    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_rows(rows)
        }
    }

    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        Self {
            entry: PhantomData,
            raw_data: self.raw_data.restrict_cols(cols)
        }
    }

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

    pub fn row_iter(self) -> impl 'a + ExactSizeIterator<Item = &'a mut [T]> {
        (0..self.raw_data.row_count).map(move |i| 
        // safe since each access goes to a different location
        unsafe {
            self.raw_data.row_at(i).as_mut()
        })
    }

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

    pub fn at<'b>(&'b mut self, i: usize, j: usize) -> &'b mut T {
        &mut self.row_at(i)[j]
    }

    pub fn row_at<'b>(&'b mut self, i: usize) -> &'b mut [T] {
        // safe since self is mutably borrowed for 'b
        unsafe {
            self.raw_data.row_at(i).as_mut()
        }
    }

    pub fn col_at<'b>(&'b mut self, j: usize) -> ColumnMut<'b, V, T> {
        assert!(j < self.raw_data.col_count);
        let mut result_raw = self.raw_data;
        result_raw.col_start += j;
        result_raw.col_count = 1;
        // safe since self is mutably borrowed for 'b
        unsafe {
            return ColumnMut::new(result_raw);
        }
    }

    pub fn reborrow<'b>(&'b mut self) -> SubmatrixMut<'b, V, T> {
        SubmatrixMut {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    pub fn as_const<'b>(&'b self) -> Submatrix<'b, V, T> {
        Submatrix {
            entry: PhantomData,
            raw_data: self.raw_data
        }
    }

    pub fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    pub fn row_count(&self) -> usize {
        self.raw_data.row_count
    }
}

impl<'a, V, T> SubmatrixMut<'a, V, T>
    where V: 'a + Sync + AsPointerToSlice<T>,
        T: Send
{
    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_row_iter(self) -> impl 'a + ExactSizeIterator<Item = &'a mut [T]> {
        self.row_iter()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_col_iter(self) -> impl 'a + ExactSizeIterator<Item = ColumnMut<'a, V, T>> {
        self.col_iter()
    }
    
    #[cfg(feature = "parallel")]
    pub fn concurrent_row_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut [T]> {
        
        struct AccessIthRow<'a, V, T>
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            entry: PhantomData<&'a ()>,
            raw_data: SubmatrixRaw<V, T>
        }

        unsafe impl<'a, V, T> Sync for AccessIthRow<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {}

        unsafe impl<'a, V, T> Send for AccessIthRow<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {}

        impl<'a, V, T> FnOnce<(usize,)> for AccessIthRow<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            type Output = &'a mut [T];

            extern "rust-call" fn call_once(self, args: (usize,)) -> Self::Output {
                self.call(args)
            }
        }

        impl<'a, V, T> FnMut<(usize,)> for AccessIthRow<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            extern "rust-call" fn call_mut(&mut self, args: (usize,)) -> Self::Output {
                self.call(args)
            }
        }

        impl<'a, V, T> Fn<(usize,)> for AccessIthRow<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            extern "rust-call" fn call(&self, args: (usize,)) -> Self::Output {
                unsafe {
                    self.raw_data.row_at(args.0).as_mut()
                }
            }
        }

        rayon::iter::ParallelIterator::map(
            rayon::iter::IntoParallelIterator::into_par_iter(0..self.row_count()),
            AccessIthRow { entry: PhantomData, raw_data: self.raw_data }
        )
    }

    #[cfg(feature = "parallel")]
    pub fn concurrent_col_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColumnMut<'a, V, T>> {
        
        struct AccessIthCol<'a, V, T>
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            entry: PhantomData<&'a ()>,
            raw_data: SubmatrixRaw<V, T>
        }

        unsafe impl<'a, V, T> Sync for AccessIthCol<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {}

        unsafe impl<'a, V, T> Send for AccessIthCol<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {}

        impl<'a, V, T> FnOnce<(usize,)> for AccessIthCol<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            type Output = ColumnMut<'a, V, T>;

            extern "rust-call" fn call_once(self, args: (usize,)) -> Self::Output {
                self.call(args)
            }
        }

        impl<'a, V, T> FnMut<(usize,)> for AccessIthCol<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            extern "rust-call" fn call_mut(&mut self, args: (usize,)) -> Self::Output {
                self.call(args)
            }
        }

        impl<'a, V, T> Fn<(usize,)> for AccessIthCol<'a, V, T> 
            where V: Sync + AsPointerToSlice<T>, T: 'a + Send
        {
            extern "rust-call" fn call(&self, args: (usize,)) -> Self::Output {
                let mut result_raw = self.raw_data;
                result_raw.col_start += args.0;
                result_raw.col_count = 1;
                unsafe {
                    return ColumnMut::new(result_raw);
                }
            }
        }

        rayon::iter::ParallelIterator::map(
            rayon::iter::IntoParallelIterator::into_par_iter(0..self.col_count()),
            AccessIthCol { entry: PhantomData, raw_data: self.raw_data }
        )
    }
}

impl<'a, V, T> Submatrix<'a, V, T>
    where V: 'a + Sync + AsPointerToSlice<T>,
        T: Sync
{
    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_row_iter(self) -> impl 'a + ExactSizeIterator<Item = &'a [T]> {
        self.row_iter()
    }
    
    #[cfg(feature = "parallel")]
    pub fn concurrent_row_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a [T]> {
        rayon::iter::ParallelIterator::map(
            rayon::iter::IntoParallelIterator::into_par_iter(0..self.row_count()),
            move |i| unsafe { self.raw_data.row_at(i).as_ref() }
        )
    }
}

impl<'a, T> SubmatrixMut<'a, AsFirstElement<T>, T> {

    pub fn new(data: &'a mut [T], row_count: usize, col_count: usize) -> Self {
        assert_eq!(row_count * col_count, data.len());
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(std::mem::transmute(NonNull::new(data.as_mut_ptr()).unwrap_unchecked()), row_count, col_count as isize, 0, col_count)
            }
        }
    }
}

impl<'a, T> SubmatrixMut<'a, Vec<T>, T> {

    pub fn new(data: &'a mut [Vec<T>]) -> Self {
        assert!(data.len() > 0);
        let row_count = data.len();
        let col_count = data[0].len();
        for row in data.iter_mut() {
            assert_eq!(col_count, row.len());
        }
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(NonNull::new(data.as_mut_ptr()).unwrap_unchecked(), row_count, 1, 0, col_count)
            }
        }
    }
}

impl<'a, T, const N: usize> SubmatrixMut<'a, [T; N], T> {

    pub fn new(data: &'a mut [[T; N]]) -> Self {
        assert!(data.len() > 0);
        let row_count = data.len();
        let col_count = data[0].len();
        for row in data.iter_mut() {
            assert_eq!(col_count, row.len());
        }
        unsafe {
            Self {
                entry: PhantomData,
                raw_data: SubmatrixRaw::new(NonNull::new(data.as_mut_ptr()).unwrap_unchecked(), row_count, 1, 0, col_count)
            }
        }
    }
}

impl<'a, V, R> super::Matrix<R> for Submatrix<'a, V, R::Element>
    where V: 'a + AsPointerToSlice<R::Element>, R: RingBase
{
    fn at(&self, i: usize, j: usize) -> &<R as RingBase>::Element {
        self.row_at(i).at(j)
    }

    fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    fn row_count(&self) -> usize {
        self.raw_data.row_count
    }
}

impl<'a, V, R> super::Matrix<R> for SubmatrixMut<'a, V, R::Element>
    where V: 'a + AsPointerToSlice<R::Element>, R: RingBase
{
    fn at(&self, i: usize, j: usize) -> &<R as RingBase>::Element {
        unsafe {
            self.raw_data.entry_at(i, j).as_ref()
        }
    }

    fn col_count(&self) -> usize {
        self.raw_data.col_count
    }

    fn row_count(&self) -> usize {
        self.raw_data.row_count
    }
}

#[cfg(test)]
use std::fmt::Debug;

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
    let matrix = SubmatrixMut::<Vec<_>, _>::new(&mut data[..]);
    f(matrix)
}

#[cfg(test)]
fn with_testmatrix_array<F>(f: F)
    where F: FnOnce(SubmatrixMut<[i64; 5], i64>)
{
    let mut data = vec![
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ];
    let matrix = SubmatrixMut::<[_; 5], _>::new(&mut data[..]);
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
    let matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut data[..], 3, 5);
    f(matrix)
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
    *left.at(1, 1) += 1;
    *right.at(0, 0) += 1;
    *right.at(2, 1) += 1;
    assert_submatrix_eq([[1, 2, 3], [6, 8, 8], [11, 12, 13]], &mut left);
    assert_submatrix_eq([[5, 5], [9, 10], [14, 16]], &mut right);

    let (mut top, mut bottom) = left.split_rows(0..1, 1..3);
    assert_submatrix_eq([[1, 2, 3]], &mut top);
    assert_submatrix_eq([[6, 8, 8], [11, 12, 13]], &mut bottom);
    *top.at(0, 0) -= 1;
    *top.at(0, 2) += 3;
    *bottom.at(0, 2) -= 1;
    *bottom.at(1, 0) += 3;
    assert_submatrix_eq([[0, 2, 6]], &mut top);
    assert_submatrix_eq([[6, 8, 7], [14, 12, 13]], &mut bottom);
}

#[test]
fn test_submatrix_mutate_wrapper() {
    with_testmatrix_vec(test_submatrix_mutate);
    with_testmatrix_array(test_submatrix_mutate);
    with_testmatrix_linmem(test_submatrix_mutate);
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
        assert_eq!(vec![1, 6, 11], col1.iter().map(|x| *x).collect::<Vec<_>>());
        assert_eq!(vec![2, 7, 12], col2.iter().map(|x| *x).collect::<Vec<_>>());
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
        assert_eq!(vec![4, 9, 14], col.reborrow().iter().map(|x| *x).collect::<Vec<_>>());
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
}

#[cfg(test)]
fn test_submatrix_col_at<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    assert_eq!(&[2, 7, 12], &matrix.col_at(1).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[2, 7, 12], &matrix.as_const().col_at(1).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.col_at(4).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.as_const().col_at(4).iter().copied().collect::<Vec<_>>()[..]);

    {
        let (mut top, mut bottom) = matrix.reborrow().restrict_rows(0..2).split_rows(0..1, 1..2);
        assert_eq!(&[1], &top.col_at(0).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[1], &top.as_const().col_at(0).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[5], &top.col_at(4).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[5], &top.as_const().col_at(4).iter().copied().collect::<Vec<_>>()[..]);

        assert_eq!(&[6], &bottom.col_at(0).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[6], &bottom.as_const().col_at(0).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[10], &bottom.col_at(4).iter().copied().collect::<Vec<_>>()[..]);
        assert_eq!(&[10], &bottom.as_const().col_at(4).iter().copied().collect::<Vec<_>>()[..]);
    }
}

#[test]
fn test_submatrix_col_at_wrapper() {
    with_testmatrix_vec(test_submatrix_col_at);
    with_testmatrix_array(test_submatrix_col_at);
    with_testmatrix_linmem(test_submatrix_col_at);
}

#[cfg(test)]
fn test_submatrix_row_at<V: AsPointerToSlice<i64>>(mut matrix: SubmatrixMut<V, i64>) {
    assert_submatrix_eq([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ], &mut matrix);
    assert_eq!(&[2, 7, 12], &matrix.col_at(1).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[2, 7, 12], &matrix.as_const().col_at(1).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.col_at(4).iter().copied().collect::<Vec<_>>()[..]);
    assert_eq!(&[5, 10, 15], &matrix.as_const().col_at(4).iter().copied().collect::<Vec<_>>()[..]);

    {
        let (mut left, mut right) = matrix.reborrow().restrict_cols(1..5).split_cols(0..2, 2..4);
        assert_eq!(&[2, 3], left.row_at(0));
        assert_eq!(&[4, 5], right.row_at(0));
        assert_eq!(&[2, 3], left.as_const().row_at(0));
        assert_eq!(&[4, 5], right.as_const().row_at(0));

        assert_eq!(&[7, 8], left.row_at(1));
        assert_eq!(&[9, 10], right.row_at(1));
        assert_eq!(&[7, 8], left.as_const().row_at(1));
        assert_eq!(&[9, 10], right.as_const().row_at(1));
    }
}

#[test]
fn test_submatrix_row_at_wrapper() {
    with_testmatrix_vec(test_submatrix_row_at);
    with_testmatrix_array(test_submatrix_row_at);
    with_testmatrix_linmem(test_submatrix_row_at);
}
