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
///  - `F` and `G` support the same rings
///  - `F.len() == G.len()`
///  - `F.root_of_unity(ring) == G.root_of_unity(ring)` for each supported ring `ring`
///  - `F.unordered_fft_permutation(i) == G.unordered_fft_permutation(i)` for all `i`
/// In other words, `F` and `G` must have exactly the same output for `unordered_fft`
/// (and thus `fft`, `inv_fft`, ...) on same inputs (w.r.t. equality given by the
/// base rings, which are equal).
/// 
pub trait FFTAlgorithm<R: ?Sized + RingBase> {

    ///
    /// This FFTTable can compute the FFT of arrays of this length.
    /// 
    fn len(&self) -> usize;

    ///
    /// The root of unity used for the FFT. While all primitive `n`-th roots
    /// of unity can be used equally for computing a Fourier transform, the 
    /// concrete one used determines the order of the output values.
    /// 
    /// See also [`FFTAlgorithm::unordered_fft_permutation()`].
    /// 
    fn root_of_unity<S>(&self, ring: S) -> &R::Element
        where S: RingStore<Type = R> + Copy;

    ///
    /// On input `i`, returns `j` such that `unordered_fft(values)[i]` contains the evaluation
    /// at `zeta^(-j)` of values (note the `-`, which is standard convention for Fourier transforms).
    /// Here `zeta` is the value returned by [`FFTAlgorithm::root_of_unity()`].
    /// 
    /// # Example
    /// ```text
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::algorithms::*;
    /// # use feanor_math::field::*;
    /// # use feanor_math::algorithms::fft::*;
    /// let ring = Zn::new(17);
    /// let fft = cooley_tuckey::CooleyTuckeyFFT::for_zn(&ring, 3);
    /// let (zero, one) = (ring.zero(), ring.one());
    /// let mut values = [zero, one, one, zero, zero, zero, zero, zero];
    /// fft.unordered_fft(&mut values, &ring);
    /// for i in 0..8 {
    ///     let evaluation_at = ring.pow(ring.invert(fft.root_of_unity()).unwrap(), fft.unordered_fft_permutation(i));
    ///     assert_el_eq!(ring, ring.add(evaluation_at, ring.pow(evaluation_at, 2)), &values[i]);
    /// }
    /// ```
    /// 
    fn unordered_fft_permutation(&self, i: usize) -> usize;

    ///
    /// The inverse of [`FFTAlgorithm::unordered_fft_permutation()`], i.e. for all i, have
    /// `self.unordered_fft_permutation_inv(self.unordered_fft_permutation(i)) == i`.
    /// 
    fn unordered_fft_permutation_inv(&self, i: usize) -> usize;

    ///
    /// Computes the Fourier transform of the given `values` over the specified ring.
    /// The output is in standard order, i.e. the `i`-th output element is the evaluation
    /// of the input at `self.root_of_unity()^(-i)` (note the `-`, which is standard
    /// convention for Fourier transforms).
    /// 
    /// # Panics
    /// 
    /// This function panics if `values.len() != self.len()`.
    ///
    fn fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        self.unordered_fft(&mut values, ring);
        permute::permute_inv(&mut values, |i| self.unordered_fft_permutation(i));
    }
        
    ///
    /// Computes the Fourier transform of the given `values` over the specified ring.
    /// The output is in standard order, i.e. the `i`-th output element is the evaluation
    /// of the input at `self.root_of_unity()^i`, divided by `self.len()`.
    /// 
    /// # Panics
    /// 
    /// This function panics if `values.len() != self.len()`.
    ///
    fn inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        permute::permute(&mut values, |i| self.unordered_fft_permutation(i));
        self.unordered_inv_fft(&mut values, ring);
    }

    ///
    /// Computes the Fourier transform of the given values, but the output values are arbitrarily permuted
    /// (in a way compatible with [`FFTAlgorithm::unordered_inv_fft()`]).
    /// 
    /// Note that the FFT of a sequence `a_0, ..., a_(N - 1)` is defined as `Fa_k = sum_i a_i z^(-ik)`
    /// where `z` is an N-th root of unity.
    /// 
    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<R::Element>,
            S: RingStore<Type = R> + Copy;
    
    ///
    /// Inverse to [`FFTAlgorithm::unordered_fft()`], with basically the same contract.
    /// 
    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<R::Element>,
           S: RingStore<Type = R> + Copy;
}

impl<T, R> FFTAlgorithm<R> for T
    where R: ?Sized + RingBase,
        T: Deref, 
        T::Target: FFTAlgorithm<R>
{
    fn len(&self) -> usize {
        self.deref().len()
    }

    fn root_of_unity<S>(&self, ring: S) -> &R::Element
        where S: RingStore<Type = R> + Copy
    {
        self.deref().root_of_unity(ring)
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        self.deref().unordered_fft_permutation(i)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        self.deref().unordered_fft_permutation_inv(i)
    }

    fn fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R as RingBase>::Element>,
            S: RingStore<Type = R> + Copy 
    {
        self.deref().fft(values, ring)
    }

    fn inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R as RingBase>::Element>,
            S: RingStore<Type = R> + Copy 
    {
        self.deref().inv_fft(values, ring)
    }

    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R as RingBase>::Element>,
            S: RingStore<Type = R> + Copy 
    {
        self.deref().unordered_fft(values, ring)
    }

    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R as RingBase>::Element>,
            S: RingStore<Type = R> + Copy 
    {
        self.deref().unordered_inv_fft(values, ring)
    }
}