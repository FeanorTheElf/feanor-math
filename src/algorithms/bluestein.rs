use crate::ring::*;
use crate::algorithms;
use crate::vector::VectorViewMut;

pub struct BluesteinFFTTable<R>
    where R: RingStore
{
    ring: R,
    m_fft_table: algorithms::cooley_tuckey::FFTTableCooleyTuckey<R>,
    b_fft: Vec<El<R>>,
    root_of_unity: El<R>,
    inv_root_of_unity: El<R>,
    n: usize
}

impl<R> BluesteinFFTTable<R> 
    where R: RingStore
{
    fn fft_base<V, S, const INV: bool>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingStore, S::Type: CanonicalHom<R::Type>
    {

    }
}