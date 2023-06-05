use crate::{ring::*, vector::*, mempool::*};

pub mod cooley_tuckey;
pub mod bluestein;
pub mod factor_fft;
pub mod primitive;

pub trait FFTTable {

    type Ring: RingStore;

    fn len(&self) -> usize;
    fn ring(&self) -> &Self::Ring;
    fn root_of_unity(&self) -> &El<Self::Ring>;

    ///
    /// On input `i`, returns `j` such that `unordered_fft(values)[i]` contains the evaluation
    /// at `zeta^j` of values.
    /// 
    fn unordered_fft_permutation(&self, i: usize) -> usize;

    fn fft<V, S, M>(&self, mut values: V, ring: S, memory_provider: &M)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            M: MemoryProvider<El<S>>
    {
        self.unordered_fft(&mut values, ring, memory_provider);
        permute::permute_inv(&mut values, |i| self.unordered_fft_permutation(i), &AllocatingMemoryProvider);
    }
        
    fn inv_fft<V, S, M>(&self, mut values: V, ring: S, memory_provider: &M)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            M: MemoryProvider<El<S>>
    {
        permute::permute(&mut values, |i| self.unordered_fft_permutation(i), &AllocatingMemoryProvider);
        self.unordered_inv_fft(&mut values, ring, memory_provider);
    }

    ///
    /// Computes the FFT of the given values, but the output values are arbitrarily permuted
    /// (in a way compatible with [`FFTTable::unordered_inv_fft()`]).
    /// 
    /// This supports any given ring, as long as the precomputed values stored in the table are
    /// also contained in the new ring. The result is wrong however if the canonical homomorphism
    /// `R -> S` does not map the N-th root of unity to a primitive N-th root of unity.
    /// 
    /// Note that the FFT of a sequence `a_0, ..., a_(N - 1)` is defined as `Fa_k = sum_i a_i z^(-ik)`
    /// where `z` is an N-th root of unity.
    /// 
    fn unordered_fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            M: MemoryProvider<El<S>>;
        
    fn unordered_inv_fft<V, S, M>(&self, values: V, ring: S, memory_provider: &M)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            M: MemoryProvider<El<S>>;
}
