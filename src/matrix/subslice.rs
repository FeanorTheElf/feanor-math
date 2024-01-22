use std::ops::Range;
use std::ops::Index;
use std::{iter::FusedIterator, ops::IndexMut};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::cmp::min;

use crate::vector::{VectorView, VectorViewMut};

pub trait AsPointerToSlice<T> {

    unsafe fn get_pointer(&self) -> NonNull<T>;
    fn len(&self) -> usize;
}

impl<T> AsPointerToSlice<T> for Vec<T> {

    unsafe fn get_pointer(&self) -> NonNull<T> {
        NonNull::new(self.as_ptr() as *mut T).unwrap()
    }
    
    fn len(&self) -> usize {
        <Vec<T>>::len(self)
    }
}

impl<'a, T> AsPointerToSlice<T> for &'a mut [T] {

    unsafe fn get_pointer(&self) -> NonNull<T> {
        NonNull::new(self.as_ptr() as *mut T).unwrap()
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

pub struct ColumnMutIter<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    data: ColumnMut<'a, V, T>
}

impl<'a, V, T> Iterator for ColumnMutIter<'a, V, T>
    where V: AsPointerToSlice<T>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.len > 0 {
            let result = unsafe { ColumnMut::unsafe_at(NonNull::from(&self.data), 0).as_mut() };
            self.data.rows = unsafe { NonNull::new(self.data.rows.as_ptr().offset(1)).unwrap_unchecked() };
            self.data.len -= 1;
            return Some(result);
        } else {
            return None;
        }
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let actual_advance = min(n, self.data.len);
        self.data.rows = unsafe { NonNull::new(self.data.rows.as_ptr().offset(actual_advance as isize)).unwrap_unchecked() };
        self.data.len -= actual_advance;
        if n == actual_advance {
            return Ok(());
        } else {
            return Err(NonZeroUsize::new(n - actual_advance).unwrap());
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, V, T> ExactSizeIterator for ColumnMutIter<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    fn len(&self) -> usize {
        self.data.len
    }
}

impl<'a, V, T> FusedIterator for ColumnMutIter<'a, V, T> 
    where V: AsPointerToSlice<T>
{}

pub struct ColumnMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    rows: NonNull<V>,
    len: usize,
    col_index: isize
}

unsafe impl<'a, V, T> Send for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Send
{}

unsafe impl<'a, V, T> Sync for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Sync
{}

impl<'a, V, T> ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    pub fn reborrow<'b>(&'b mut self) -> ColumnMut<'b, V, T>
        where 'a: 'b
    {
        ColumnMut { entry: PhantomData, rows: self.rows, len: self.len, col_index: self.col_index }
    }

    unsafe fn unsafe_at(this: NonNull<Self>, index: usize) -> NonNull<T> {
        // the row might be shared, but we only access it via an immutable reference;
        // it must be guaranteed that the `self.col_index`-th entry of `row.get_pointer()`
        // is not shared
        let this_ref = { this.as_ref() };
        debug_assert!(index < this_ref.len);
        let row = unsafe { this_ref.rows.as_ptr().offset(index as isize).as_ref().unwrap_unchecked() };
        unsafe { NonNull::new(row.get_pointer().as_ptr().offset(this_ref.col_index)).unwrap_unchecked() }
    }
}

impl<'a, V, T> Index<usize> for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.at(index)
    }
}

impl<'a, V, T> IndexMut<usize> for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.at_mut(index)
    }
}

impl<'a, V, T> IntoIterator for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    type IntoIter = ColumnMutIter<'a, V, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        ColumnMutIter {
            data: self,
            entry: PhantomData
        }
    }
}

impl<'a, V, T> VectorView<T> for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    fn at(&self, i: usize) -> &T {
        assert!(i < self.len);
        unsafe { Self::unsafe_at(NonNull::from(self), i).as_ref() }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, V, T> VectorViewMut<T> for ColumnMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    fn at_mut(&mut self, i: usize) -> &mut T {
        unsafe { Self::unsafe_at(NonNull::from(self), i).as_mut() }
    }
}

pub struct SubmatrixMut<'a, V, T>
    where V: AsPointerToSlice<T>
{
    entry: PhantomData<&'a mut T>,
    rows: NonNull<V>,
    len: usize,
    cols_start: usize,
    cols_end: usize
}

unsafe impl<'a, V, T> Send for SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Send
{}

unsafe impl<'a, V, T> Sync for SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T> + Sync, T: Sync
{}

impl<'a, V, T> SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T>, T: 'a
{
    pub fn new(rows: &'a mut [V]) -> Self {
        let len = <[V]>::len(rows);
        let col_count = if len == 0 {
            0
        } else {
            rows[0].len()
        };
        assert!(rows.iter().all(|r| r.len() == col_count));
        return SubmatrixMut {
            cols_end: col_count,
            cols_start: 0,
            entry: PhantomData,
            len: len,
            rows: NonNull::new(rows.as_mut_ptr()).unwrap()
        };
    }

    unsafe fn unsafe_rows(data: NonNull<V>, row_offset: usize) -> NonNull<V> {
        unsafe { NonNull::new(data.as_ptr().offset(row_offset as isize)).unwrap_unchecked() }
    }

    unsafe fn unsafe_row(data: NonNull<V>, row: usize, cols_start: usize, cols_end: usize) -> NonNull<[T]> {
        let row_ref = unsafe { Self::unsafe_rows(data, row).as_ref() };
        let begin_entry = unsafe { NonNull::new(row_ref.get_pointer().as_ptr().offset(cols_start as isize)).unwrap_unchecked() };
        NonNull::slice_from_raw_parts(begin_entry, cols_end - cols_start)
    }

    pub fn reborrow<'b>(&'b mut self) -> SubmatrixMut<'b, V, T>
        where 'a: 'b
    {
        Self {
            cols_end: self.cols_end,
            entry: PhantomData,
            rows: self.rows,
            len: self.len,
            cols_start: self.cols_start
        }
    }

    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        assert!(rows.end <= self.len);
        assert!(cols.end <= self.cols_end - self.cols_start);
        Self {
            cols_end: cols.end + self.cols_start,
            cols_start: cols.start + self.cols_start,
            entry: PhantomData,
            len: rows.end - rows.start,
            rows: unsafe { Self::unsafe_rows(self.rows, rows.start) },
        }
    }

    pub fn split_rows(self, fst: Range<usize>, snd: Range<usize>) -> (SubmatrixMut<'a, V, T>, SubmatrixMut<'a, V, T>) {
        assert!(fst.end <= snd.start || snd.end <= fst.start);
        assert!(fst.end <= self.len);
        assert!(snd.end <= self.len);
        (
            SubmatrixMut {
                entry: PhantomData,
                rows: unsafe { Self::unsafe_rows(self.rows, fst.start) },
                cols_start: self.cols_start,
                cols_end: self.cols_end,
                len: fst.end - fst.start
            },
            SubmatrixMut {
                entry: PhantomData,
                rows: unsafe { Self::unsafe_rows(self.rows, snd.start) },
                cols_start: self.cols_start,
                cols_end: self.cols_end,
                len: snd.end - snd.start
            }
        )
    }

    pub fn split_cols(self, fst: Range<usize>, snd: Range<usize>) -> (SubmatrixMut<'a, V, T>, SubmatrixMut<'a, V, T>) {
        assert!(fst.end <= snd.start || snd.end <= fst.start);
        assert!(fst.end <= self.cols_end - self.cols_start);
        assert!(snd.end <= self.cols_end - self.cols_start);
        (
            SubmatrixMut {
                entry: PhantomData,
                rows: self.rows,
                len: self.len,
                cols_start: fst.start + self.cols_start,
                cols_end: fst.end + self.cols_start
            },
            SubmatrixMut {
                entry: PhantomData,
                rows: self.rows,
                len: self.len,
                cols_start: snd.start + self.cols_start,
                cols_end: snd.end + self.cols_start
            }
        )
    }

    pub fn row_count(&self) -> usize {
        self.len
    }

    pub fn col_count(&self) -> usize {
        self.cols_end - self.cols_start
    }
}

impl<'a, V, T> Index<usize> for SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len);
        unsafe { Self::unsafe_row(self.rows, index, self.cols_start, self.cols_end).as_ref() }
    }
}

impl<'a, V, T> IndexMut<usize> for SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len);
        unsafe { Self::unsafe_row(self.rows, index, self.cols_start, self.cols_end).as_mut() }
    }
}

impl<'a, V, T> SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T>
{
    pub fn row_iter<'b>(&'b self) -> impl 'b + Clone + ExactSizeIterator<Item = &'b [T]> {
        (0..self.row_count()).map(move |i| unsafe { Self::unsafe_row(self.rows, i, self.cols_start, self.cols_end).as_ref() })
    }

    pub fn row_iter_mut<'b>(&'b mut self) -> impl 'b + ExactSizeIterator<Item = &'b mut [T]> {
        let data = self.rows;
        let cols_start = self.cols_start;
        let cols_end = self.cols_end;
        (0..self.row_count()).map(move |i| unsafe { Self::unsafe_row(data, i, cols_start, cols_end).as_mut() })
    }

    pub fn col_iter_mut<'b>(&'b mut self) -> impl 'b + ExactSizeIterator<Item = ColumnMut<'b, V, T>> {
        (self.cols_start..self.cols_end).map(|col_index| ColumnMut {
            col_index: col_index as isize,
            entry: PhantomData,
            rows: self.rows,
            len: self.len
        })
    }
}

impl<'a, V, T> SubmatrixMut<'a, V, T> 
    where V: AsPointerToSlice<T> + Send + Sync, T: Send + Sync
{
    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_row_iter_mut<'b>(&'b mut self) -> impl 'b + ExactSizeIterator<Item = &'b mut [T]> {
        self.row_iter_mut()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_col_iter_mut<'b>(&'b mut self) -> impl 'b + ExactSizeIterator<Item = ColumnMut<'b, V, T>> {
        self.col_iter_mut()
    }

    #[cfg(feature = "parallel")]
    pub fn concurrent_row_iter_mut<'b>(&'b mut self) -> impl 'b + rayon::iter::IndexedParallelIterator<Item = &'b mut [T]> {
        use std::num::NonZeroIsize;

        let data: NonZeroIsize = unsafe { std::mem::transmute(self.rows) };
        let cols_start = self.cols_start;
        let cols_end = self.cols_end;
        <_ as rayon::iter::ParallelIterator>::map(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(0..self.row_count()), move |row| unsafe { Self::unsafe_row(std::mem::transmute(data), row, cols_start, cols_end).as_mut() })
    }

    #[cfg(feature = "parallel")]
    pub fn concurrent_col_iter_mut<'b>(&'b mut self) -> impl 'b + rayon::iter::IndexedParallelIterator<Item = ColumnMut<'b, V, T>> {
        
        let rows_ptr: std::num::NonZeroIsize = unsafe { std::mem::transmute(self.rows) };
        let len = self.len;
        let col_range = self.cols_start..self.cols_end;

        <_ as rayon::iter::ParallelIterator>::map(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(col_range), move |col_index| ColumnMut {
            col_index: col_index as isize,
            entry: PhantomData,
            rows: unsafe { std::mem::transmute(rows_ptr) },
            len: len
        })
    }
}

#[cfg(test)]
use std::fmt::Debug;

#[cfg(test)]
fn assert_submatrix_eq<V: AsPointerToSlice<T>, T: PartialEq + Debug, const N: usize, const M: usize>(expected: [[T; M]; N], actual: &SubmatrixMut<V, T>) {
    assert_eq!(N, actual.row_count());
    assert_eq!(M, actual.col_count());
    for i in 0..N {
        for j in 0..M {
            assert_eq!(&expected[i][j], &actual[i][j]);
        }
    }
}

#[test]
fn test_submatrix() {
    let mut data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    let mut matrix = SubmatrixMut::new(&mut data);
    assert_submatrix_eq([[2, 3], [7, 8]], &matrix.reborrow().submatrix(0..2, 1..3));
    assert_submatrix_eq([[8, 9, 10]], &matrix.reborrow().submatrix(1..2, 2..5));
    assert_submatrix_eq([[8, 9, 10], [13, 14, 15]], &matrix.reborrow().submatrix(1..3, 2..5));

    let (left, right) = matrix.split_cols(0..3, 3..5);
    assert_submatrix_eq([[1, 2, 3], [6, 7, 8], [11, 12, 13]], &left);
    assert_submatrix_eq([[4, 5], [9, 10], [14, 15]], &right);
}

#[test]
fn test_submatrix_mutate() {
    let mut data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    let matrix = SubmatrixMut::new(&mut data);
    let (mut left, mut right) = matrix.split_cols(0..3, 3..5);
    assert_submatrix_eq([[1, 2, 3], [6, 7, 8], [11, 12, 13]], &left);
    assert_submatrix_eq([[4, 5], [9, 10], [14, 15]], &right);
    left[1][1] += 1;
    right[0][0] += 1;
    right[2][1] += 1;
    assert_submatrix_eq([[1, 2, 3], [6, 8, 8], [11, 12, 13]], &left);
    assert_submatrix_eq([[5, 5], [9, 10], [14, 16]], &right);

    let (mut top, mut bottom) = left.split_rows(0..1, 1..3);
    assert_submatrix_eq([[1, 2, 3]], &top);
    assert_submatrix_eq([[6, 8, 8], [11, 12, 13]], &bottom);
    top[0][0] -= 1;
    top[0][2] += 3;
    bottom[0][2] -= 1;
    bottom[1][0] += 3;
    assert_submatrix_eq([[0, 2, 6]], &top);
    assert_submatrix_eq([[6, 8, 7], [14, 12, 13]], &bottom);
}

#[test]
fn test_submatrix_col_iter() {
    let mut data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    let mut matrix = SubmatrixMut::new(&mut data);
    {
        let mut it = matrix.col_iter_mut();
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
        &matrix
    );
    
    let (mut left, _right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = left.col_iter_mut();
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
        &matrix
    );

    let (_left, mut right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = right.col_iter_mut();
        let mut col = it.next().unwrap();
        assert!(it.next().is_none());
        assert_eq!(vec![4, 9, 14], col.reborrow().iter().map(|x| *x).collect::<Vec<_>>());
        *col.into_iter().next().unwrap() += 3;
    }
    assert_submatrix_eq([
        [1, 2, 3, 7, 10],
        [11, 7, 8, 9, 20],
        [11, 12, 13, 14, 30]], 
        &matrix
    );
}

#[test]
fn test_submatrix_row_iter() {
    let mut data = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    let mut matrix = SubmatrixMut::new(&mut data);
    {
        let mut it = matrix.row_iter_mut();
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
        &matrix
    );
    let (mut left, mut right) = matrix.reborrow().split_cols(0..2, 3..4);
    {
        let mut it = left.row_iter();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert_eq!(&[1, 2], row1);
        assert_eq!(&[6, 7], row2);
    }
    {
        let mut it = left.row_iter_mut();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert_eq!(&[1, 2], row1);
        assert_eq!(&[6, 7], row2);
        row2[1] += 1;
    }
    assert_submatrix_eq([[1, 2], [6, 8], [11, 18]], &left);
    {
        right = right.submatrix(1..3, 0..1);
        let mut it = right.row_iter_mut();
        let row1 = it.next().unwrap();
        let row2 = it.next().unwrap();
        assert_eq!(&[9], row1);
        assert_eq!(&[14], row2);
        row1[0] += 1;
    }
    assert_submatrix_eq([[10], [14]], &right);
}