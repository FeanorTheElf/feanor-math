use std::ops::Deref;

use crate::ring::*;
use crate::seq::*;

pub mod cooley_tuckey;
pub mod bluestein;
pub mod factor_fft;
pub mod complex_fft;

///
/// Trait for objects that can perform a fast fourier transform over some ring. 
/// 
/// Usually fast implementations of FFTs have to store a lot of precomputed data
/// (e.g. powers of roots of unity), hence they should be represented as objects
/// implementing this trait.
/// 
/// # Note on equality
/// 
/// If you choose to implement [`PartialEq`] for an FFTTable, and `F == G`, then
/// `F` and `G` should satisfy the following properties:
///  - `F.ring() == G.ring()`, i.e. elements can be transferred between rings
///    without applying homomorphisms
///  - `F.len() == G.len()`
///  - `F.root_of_unity() == G.root_of_unity()`
///  - `F.unordered_fft_permutation(i) == G.unordered_fft_permutation(i)` for all `i`
/// In other words, `F` and `G` must have exactly the same output for `unordered_fft`
/// (and thus `fft`, `inv_fft`, ...) on same inputs.
/// 
pub trait FFTTable {

    type Ring: ?Sized + RingStore;

    ///
    /// This FFTTable can compute the FFT of arrays of this length.
    /// 
    fn len(&self) -> usize;

    ///
    /// The underlying ring whose roots of unity are used by the FFT.
    /// 
    fn ring(&self) -> &Self::Ring;

    ///
    /// The root of unity used for the FFT. While all primitive `n`-th roots
    /// of unity can be used equally for computing a Fourier transform, the 
    /// concrete one used determines the order of the output values.
    /// 
    /// See also [`FFTTable::unordered_fft_permutation`].
    /// 
    fn root_of_unity(&self) -> &El<Self::Ring>;

    ///
    /// On input `i`, returns `j` such that `unordered_fft(values)[i]` contains the evaluation
    /// at `zeta^j` of values. Here `zeta` is the value returned by [`FFTTable::root_of_unity()`]
    /// 
    fn unordered_fft_permutation(&self, i: usize) -> usize;

    ///
    /// The inverse of [`FFTTable::unordered_fft_permutation()`], i.e. for all i, have
    /// `self.unordered_fft_permutation_inv(self.unordered_fft_permutation(i)) == i`.
    /// 
    fn unordered_fft_permutation_inv(&self, i: usize) -> usize;

    ///
    /// Computes the Fourier transform of the given `values` over the ring [`Self::Ring`].
    /// The output is in standard order, i.e. the `i`-th output element is the evaluation
    /// of the input at `self.root_of_unity()^-i` (note the `-`, which is standard
    /// convention for Fourier transforms).
    /// 
    /// # Panics
    /// 
    /// This function panics if `values.len() != self.len()`.
    ///
    fn fft<V>(&self, mut values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        self.unordered_fft(&mut values);
        permute::permute_inv(&mut values, |i| self.unordered_fft_permutation(i));
    }
        
    ///
    /// Computes the Fourier transform of the given `values` over the ring [`Self::Ring`].
    /// The output is in standard order, i.e. the `i`-th output element is the evaluation
    /// of the input at `self.root_of_unity()^i`, divided by `self.len()`.
    /// 
    /// # Panics
    /// 
    /// This function panics if `values.len() != self.len()`.
    ///
    fn inv_fft<V>(&self, mut values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        permute::permute(&mut values, |i| self.unordered_fft_permutation(i));
        self.unordered_inv_fft(&mut values);
    }

    ///
    /// Computes the Fourier transform of the given values, but the output values are arbitrarily permuted
    /// (in a way compatible with [`FFTTable::unordered_inv_fft()`]).
    /// 
    /// Note that the FFT of a sequence `a_0, ..., a_(N - 1)` is defined as `Fa_k = sum_i a_i z^(-ik)`
    /// where `z` is an N-th root of unity.
    /// 
    fn unordered_fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>;
    
    ///
    /// Inverse to [`Self::unordered_fft()`], with basically the same contract.
    /// 
    fn unordered_inv_fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>;
}

impl<T> FFTTable for T
    where T: Deref, T::Target: FFTTable
{
    type Ring = <T::Target as FFTTable>::Ring;
    
    fn len(&self) -> usize {
        self.deref().len()
    }

    fn ring(&self) -> &Self::Ring {
        self.deref().ring()
    }

    fn root_of_unity(&self) -> &El<Self::Ring> {
        self.deref().root_of_unity()
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        self.deref().unordered_fft_permutation(i)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        self.deref().unordered_fft_permutation_inv(i)
    }

    fn fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        self.deref().fft(values)
    }

    fn inv_fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        self.deref().inv_fft(values)
    }

    fn unordered_fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        self.deref().unordered_fft(values)
    }

    fn unordered_inv_fft<V>(&self, values: V)
        where V: SwappableVectorViewMut<El<Self::Ring>>
    {
        self.deref().unordered_inv_fft(values)
    }
}