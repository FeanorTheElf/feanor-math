
use std::{ops::Range, marker::PhantomData};

pub struct Column<'a, T> {
    element: PhantomData<&'a mut T>,
    pointer: *mut Vec<T>,
    col: usize,
    len: usize
}

unsafe impl<'a, T> Send for Column<'a, T> {}

impl<'a, T> Column<'a, T> {

    pub fn iter_mut<'b>(&'b mut self) -> impl 'b + Iterator<Item = &'a mut T> {
        (0..self.len).map(|i| unsafe { &mut *(*self.pointer.offset(i as isize)).as_mut_ptr().offset(self.col as isize) })
    }

    pub fn iter<'b>(&'b self) -> impl 'b + Clone + Iterator<Item = &'a T> {
        (0..self.len).map(|i| unsafe { &*(*self.pointer.offset(i as isize)).as_mut_ptr().offset(self.col as isize) })
    }
}

#[cfg(not(feature = "parallel"))]
pub fn column_iterator<'a, T>(data: &'a mut [Vec<T>], cols: Range<usize>) -> impl 'a + Iterator<Item = Column<'a, T>> {
    assert!(data.len() > 0);
    let col_count = data[0].len();
    let row_count = data.len();
    let data_ptr = data.as_mut_ptr();
    for i in 0..row_count {
        assert!(data[i].len() == col_count);
    }
    cols.map(move |j| Column { element: PhantomData, pointer: data_ptr, len: row_count, col: j })
}

#[cfg(feature = "parallel")]
pub fn column_iterator<'a, T>(data: &'a mut [Vec<T>], cols: Range<usize>) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = Column<'a, T>> 
    where T: Send
{
    assert!(data.len() > 0);
    let col_count = data[0].len();
    let row_count = data.len();
    let data_ptr = data.as_mut_ptr() as isize;
    for i in 0..row_count {
        assert!(data[i].len() == col_count);
    }
    <_ as rayon::iter::ParallelIterator>::map(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(cols), move |j| Column { element: PhantomData, pointer: data_ptr as *mut _, len: row_count, col: j })
}

#[cfg(feature = "parallel")]
pub fn potential_parallel_for_each<D, T, F, G, S>(data: D, init_thread: G, body: F)
    where F: Fn(&mut S, usize, T) + Send + Sync,
        G: Fn() -> S + Send + Sync,
        T: Send,
        D: rayon::iter::IntoParallelIterator<Item = T>,
        <D as rayon::iter::IntoParallelIterator>::Iter: rayon::iter::IndexedParallelIterator
{
    <_ as rayon::iter::ParallelIterator>::for_each_init(<_ as rayon::iter::IndexedParallelIterator>::enumerate(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(data)), init_thread, |state, (i, el)| body(state, i, el))
}

#[cfg(not(feature = "parallel"))]
pub fn potential_parallel_for_each<D, T, F, G, S>(data: D, init_thread: G, body: F)
    where F: Fn(&mut S, usize, T) + Send + Sync,
        G: Fn() -> S + Send + Sync,
        D: IntoIterator<Item = T>
{
    let mut state = init_thread();
    for (i, el) in data.into_iter().enumerate() {
        body(&mut state, i, el);
    }
}