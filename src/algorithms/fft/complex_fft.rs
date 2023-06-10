use crate::mempool::MemoryProvider;
use crate::ring::*;
use crate::rings::float_complex::Complex64;
use crate::vector::SwappableVectorViewMut;

use super::FFTTable;

///
/// A wrapper around an FFT table for f64-based complex numbers, 
/// that provides easier and safer access
/// 
pub struct Complex64FFTTable<F: FFTTable>
    where F::Ring: RingStore<Type = Complex64>
{
    base_table: F
}

pub trait ErrorEstimate {
    fn expected_absolute_error(&self, input_bound: f64) -> f64;
}

impl<F: FFTTable> FFTTable for Complex64FFTTable<F>
    where F::Ring: RingStore<Type = Complex64>
{
    type Ring = F::Ring;

    fn fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
            where S: RingStore, 
                S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
                V: SwappableVectorViewMut<crate::ring::El<S>>,
                M: MemoryProvider<El<S>>
    {
        self.base_table.fft(values, ring, memory_provider)
    }

    fn inv_fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
            where S: RingStore, 
                S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
                V: SwappableVectorViewMut<El<S>>,
                M: MemoryProvider<El<S>>
    {
        self.base_table.inv_fft(values, ring, memory_provider)     
    }

    fn unordered_fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
            where S: RingStore, 
                S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
                V: crate::vector::VectorViewMut<El<S>>,
                M: MemoryProvider<El<S>>
    {
        self.base_table.unordered_fft(values, ring, memory_provider)    
    }

    fn len(&self) -> usize {
        self.base_table.len()
    }

    fn ring(&self) -> &Self::Ring {
        self.base_table.ring()
    }

    fn root_of_unity(&self) -> &El<Self::Ring> {
        self.base_table.root_of_unity()
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        self.base_table.unordered_fft_permutation(i)
    }

    fn unordered_inv_fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
            where S: RingStore, 
                S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
                V: crate::vector::VectorViewMut<El<S>>,
                M: MemoryProvider<El<S>>
    {
        self.base_table.unordered_inv_fft(values, ring, memory_provider)
    }
}