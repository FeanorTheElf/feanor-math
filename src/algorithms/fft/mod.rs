use crate::{ring::*, vector::*};

pub mod cooley_tuckey;
pub mod bluestein;
pub mod factor_fft;

pub trait FFTTable<R: RingStore> {

    fn len(&self) -> usize;
    fn ring(&self) -> &R;

    fn fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<R::Type>, V: SwappableVectorViewMut<El<S>>;
        
    fn inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<R::Type>, V: SwappableVectorViewMut<El<S>>;

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
    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<R::Type>, V: VectorViewMut<El<S>>;
        
    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<R::Type>, V: VectorViewMut<El<S>>;
}