
#[stability::unstable(feature = "enable")]
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

#[stability::unstable(feature = "enable")]
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