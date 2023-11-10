
use std::ops::Range;

#[cfg(not(feature = "parallel"))]
pub fn column_iterator<'a, T>(data: &'a mut [Vec<T>], cols: Range<usize>) -> impl 'a + Iterator<Item = impl 'a + Iterator<Item = &'a mut T>> {
    assert!(data.len() > 0);
    let col_count = data[0].len();
    let row_count = data.len();
    let data_ptr = data.as_mut_ptr();
    for i in 0..row_count {
        assert!(data[i].len() == col_count);
    }
    cols.map(move |j| (0..row_count).map(move |i| unsafe {
        &mut *(*data_ptr.offset(i as isize)).as_mut_ptr().offset(j as isize)
    }))
}

#[cfg(feature = "parallel")]
pub fn column_iterator<'a, T>(data: &'a mut [Vec<T>], cols: Range<usize>) -> impl 'a + rayon::iter::ParallelIterator<Item = impl 'a + Iterator<Item = &'a mut T>> 
    where T: Sync
{
    assert!(data.len() > 0);
    let col_count = data[0].len();
    let row_count = data.len();
    let data_ptr = data.as_mut_ptr() as isize;
    for i in 0..row_count {
        assert!(data[i].len() == col_count);
    }
    <_ as rayon::iter::ParallelIterator>::map(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(cols), move |j| (0..row_count).map(move |i| unsafe {
        &mut *(*(data_ptr as *mut Vec<T>).offset(i as isize)).as_mut_ptr().offset(j as isize)
    }))
}

#[cfg(feature = "parallel")]
pub fn potential_parallel_for_each<D, T, F, G, S>(data: D, init_thread: G, body: F)
    where F: Fn(&mut S, T) + Send + Sync,
        G: Fn() -> S + Send + Sync,
        T: Send,
        D: rayon::iter::IntoParallelIterator<Item = T>
{
    <_ as rayon::iter::ParallelIterator>::for_each_init(<_ as rayon::iter::IntoParallelIterator>::into_par_iter(data), init_thread, body)
}

#[cfg(not(feature = "parallel"))]
pub fn potential_parallel_for_each<D, T, F, G, S>(data: D, init_thread: G, body: F)
    where F: Fn(&mut S, T) + Send + Sync,
        G: Fn() -> S + Send + Sync,
        D: IntoIterator<Item = T>
{
    let mut state = init_thread();
    for el in data {
        body(&mut state, el);
    }
}