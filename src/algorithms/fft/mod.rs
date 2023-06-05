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

    fn fft<V, S>(&self, mut values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: SwappableVectorViewMut<El<S>>
    {
        self.unordered_fft(&mut values, ring);
        permute::permute_inv(&mut values, |i| self.unordered_fft_permutation(i), &AllocatingMemoryProvider);
    }
        
    fn inv_fft<V, S>(&self, mut values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: SwappableVectorViewMut<El<S>>
    {
        permute::permute(&mut values, |i| self.unordered_fft_permutation(i), &AllocatingMemoryProvider);
        self.unordered_inv_fft(&mut values, ring);
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
    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>>;
        
    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>>;
}

pub trait FFTTableDyn<R>
    where R: RingBase
{
    fn dyn_len(&self) -> usize;
    fn dyn_root_of_unity(&self, ring: &R) -> R::Element;
    fn dyn_fft(&self, values: &mut [R::Element], ring: &R);
    fn dyn_inv_fft(&self, values: &mut [R::Element], ring: &R);
    fn dyn_unordered_fft(&self, values: &mut [R::Element], ring: &R);
    fn dyn_unordered_inv_fft(&self, values: &mut [R::Element], ring: &R);
}

impl<R, F> FFTTableDyn<R> for F
    where F: FFTTable, R: CanonicalHom<<F::Ring as RingStore>::Type> + SelfIso
{
    fn dyn_len(&self) -> usize {
        <Self as FFTTable>::len(self)
    }

    fn dyn_root_of_unity(&self, ring: &R) -> R::Element {
        ring.map_in_ref(self.ring().get_ring(), self.root_of_unity(), &ring.has_canonical_hom(self.ring().get_ring()).unwrap())
    }

    fn dyn_fft(&self, values: &mut [<R as RingBase>::Element], ring: &R) {
        self.fft(values, &RingRef::new(ring));
    }
    
    fn dyn_inv_fft(&self, values: &mut [<R as RingBase>::Element], ring: &R) {
        self.inv_fft(values, &RingRef::new(ring));
    }

    fn dyn_unordered_fft(&self, values: &mut [<R as RingBase>::Element], ring: &R) {
        self.unordered_fft(values, &RingRef::new(ring));
    }
    
    fn dyn_unordered_inv_fft(&self, values: &mut [<R as RingBase>::Element], ring: &R) {
        self.unordered_inv_fft(values, &RingRef::new(ring));
    }
}