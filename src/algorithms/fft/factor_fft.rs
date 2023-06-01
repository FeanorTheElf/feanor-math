use crate::{ring::*, mempool::*};
use crate::algorithms::fft::*;
use crate::vector::*;

pub struct FFTTableGenCooleyTuckey<R, T1, T2, M = AllocatingMemoryProvider> 
    where R: RingStore,
        T1: FFTTable<R>,
        T2: FFTTable<R>,
        M: MemoryProvider<El<R>>
{
    twiddle_factors: M::Object,
    inv_twiddle_factors: M::Object,
    left_table: T1,
    right_table: T2
}

impl<R, T1, T2, M> FFTTableGenCooleyTuckey<R, T1, T2, M>
    where R: RingStore,
        T1: FFTTable<R>,
        T2: FFTTable<R>,
        M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(root_of_unity: El<R>, left_table: T1, right_table: T2, memory_provider: M) -> Self {
        unimplemented!()
    }
}

impl<R, T1, T2, M> FFTTable<R> for FFTTableGenCooleyTuckey<R, T1, T2, M>
    where R: RingStore,
        T1: FFTTable<R>,
        T2: FFTTable<R>,
        M: MemoryProvider<El<R>>
{
    fn len(&self) -> usize {
        self.left_table.len() * self.right_table.len()
    }

    fn ring(&self) -> &R {
        self.left_table.ring()
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<R as RingStore>::Type>, V: VectorViewMut<El<S>>
    {
        let hom = ring.can_hom(self.ring()).unwrap();
        for i in 0..self.left_table.len() {
            self.left_table.unordered_fft(Subvector::new(&mut values).subvector((i * self.right_table.len())..((i + 1) * self.right_table.len())), &ring);
        }
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.twiddle_factors.at(i), hom.raw_hom());
        }
        for i in 0..self.right_table.len() {
            self.right_table.unordered_fft(Subvector::new(&mut values).subvector(i..).stride(self.left_table.len()), &ring);
        }
    }

    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<R as RingStore>::Type>, V: VectorViewMut<El<S>> 
    {
        unimplemented!()    
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        unimplemented!()
    }

    fn fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<R as RingStore>::Type>, V: SwappableVectorViewMut<El<S>> 
    {
        unimplemented!()
    }

    fn inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<R as RingStore>::Type>, V: SwappableVectorViewMut<El<S>> 
    {
        unimplemented!()
    }
}