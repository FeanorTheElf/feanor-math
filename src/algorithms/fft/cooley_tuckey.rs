use crate::algorithms::unity_root::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::rings::zn::*;
use crate::seq::SwappableVectorViewMut;
use crate::ring::*;
use crate::seq::VectorViewMut;
use crate::homomorphism::*;
use crate::algorithms::fft::*;
use crate::algorithms;
use crate::rings::float_complex::*;
use super::complex_fft::*;

///
/// An optimized implementation of the Cooley-Tuckey FFT algorithm, to compute
/// the Fourier transform of an array with power-of-two length.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// // this ring has a 256-th primitive root of unity
/// let ring = zn_64::Zn::new(257);
/// let fft_table = CooleyTuckeyFFT::for_zn(ring, 8).unwrap();
/// let mut data = [ring.one()].into_iter().chain((0..255).map(|_| ring.zero())).collect::<Vec<_>>();
/// fft_table.unordered_fft(&mut data, &ring);
/// assert_el_eq!(ring, ring.one(), data[0]);
/// assert_el_eq!(ring, ring.one(), data[1]);
/// ```
/// 
pub struct CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>
{
    hom: H,
    root_of_unity: R_main::Element,
    log2_n: usize,
    // stores the powers of root_of_unity in special bitreversed order
    root_of_unity_list: Vec<R_twiddle::Element>,
    // stores the powers of inv_root_of_unity in special bitreversed order
    inv_root_of_unity_list: Vec<R_twiddle::Element>
}

///
/// Assumes that `index` has only the least significant `bits` bits set.
/// Then computes the value that results from reversing the least significant `bits`
/// bits.
/// 
pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits().checked_shr(usize::BITS - bits as u32).unwrap_or(0)
}

impl<R_main, H> CooleyTuckeyFFT<R_main, Complex64Base, H> 
    where R_main: ?Sized + RingBase,
        H: Homomorphism<Complex64Base, R_main>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the complex field, using the given homomorphism
    /// to connect the ring implementation for twiddles with the main ring implementation.
    /// 
    /// This function is mainly provided for parity with other rings, since in the complex case
    /// it currently does not make much sense to use a different homomorphism than the identity.
    /// Hence, it is simpler to use [`CooleyTuckeyFFT::for_complex()`].
    /// 
    pub fn for_complex_with_hom(hom: H, log2_n: usize) -> Self {
        let CC = *hom.domain().get_ring();
        Self::new_with_pows_with_hom(hom, |i| CC.root_of_unity(i, 1 << log2_n), log2_n)
    }
}

impl<R> CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<R>> 
    where R: RingStore<Type = Complex64Base>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the complex field.
    /// 
    pub fn for_complex(ring: R, log2_n: usize) -> Self {
        Self::for_complex_with_hom(ring.into_identity(), log2_n)
    }
}

impl<R> CooleyTuckeyFFT<R::Type, R::Type, Identity<R>> 
    where R: RingStore,
        R::Type: DivisibilityRing
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given ring, using the given root of unity
    /// as base. Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_hom(ring.into_identity(), root_of_unity, log2_n)
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given ring, using the passed function to
    /// provide the necessary roots of unity.
    /// 
    /// Concretely, `root_of_unity_pow(i)` should return `z^i`, where `z` is a `2^log2_n`-th
    /// primitive root of unity.
    /// 
    pub fn new_with_pows<F>(ring: R, root_of_unity_pow: F, log2_n: usize) -> Self 
        where F: FnMut(i64) -> El<R>
    {
        Self::new_with_pows_with_hom(ring.into_identity(), root_of_unity_pow, log2_n)
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for a prime field, assuming it has a characteristic
    /// congruent to 1 modulo `2^log2_n`.
    /// 
    pub fn for_zn(ring: R, log2_n: usize) -> Option<Self>
        where R::Type: ZnRing
    {
        Self::for_zn_with_hom(ring.into_identity(), log2_n)
    }
}

impl<R_main, R_twiddle, H> CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given rings, using the given root of unity.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    pub fn new_with_hom(hom: H, root_of_unity: R_twiddle::Element, log2_n: usize) -> Self {
        let ring = hom.domain();
        let mut root_of_unity_pow = |i: i64| if i >= 0 {
            ring.pow(ring.clone_el(&root_of_unity), i as usize)
        } else {
            ring.invert(&ring.pow(ring.clone_el(&root_of_unity), (-i) as usize)).unwrap()
        };

        // cannot call new_with_mem_and_pows() because of borrowing conflict
        assert!(ring.is_commutative());
        assert!(!hom.domain().get_ring().is_approximate());
        assert!(is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        assert!(is_prim_root_of_unity_pow2(&hom.codomain(), &hom.map(root_of_unity_pow(1)), log2_n));

        let root_of_unity_list = Self::create_root_of_unity_list(ring.get_ring(), &mut root_of_unity_pow, log2_n);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(ring.get_ring(), |i| root_of_unity_pow(-i), log2_n);
        let root_of_unity = root_of_unity_pow(1);

        CooleyTuckeyFFT {
            root_of_unity: hom.map(root_of_unity), 
            hom, 
            log2_n, 
            root_of_unity_list, 
            inv_root_of_unity_list
        }
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given rings, using the given function to create
    /// the necessary powers of roots of unity. This is the most generic way to create [`CooleyTuckeyFFT`].
    /// 
    /// Concretely, `root_of_unity_pow(i)` should return `z^i`, where `z` is a `2^log2_n`-th
    /// primitive root of unity.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    pub fn new_with_pows_with_hom<F>(hom: H, mut root_of_unity_pow: F, log2_n: usize) -> Self 
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let ring = hom.domain();
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity_pow2(&hom.codomain(), &hom.map(root_of_unity_pow(1)), log2_n));

        let root_of_unity_list = Self::create_root_of_unity_list(ring.get_ring(), &mut root_of_unity_pow, log2_n);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(ring.get_ring(), |i| root_of_unity_pow(-i), log2_n);
        let root_of_unity = root_of_unity_pow(1);
        
        CooleyTuckeyFFT {
            root_of_unity: hom.map(root_of_unity), 
            hom, 
            log2_n, 
            root_of_unity_list, 
            inv_root_of_unity_list
        }
    }

    fn create_root_of_unity_list<F>(ring: &R_twiddle, mut root_of_unity_pow: F, log2_n: usize) -> Vec<R_twiddle::Element>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        // in fact, we could choose this to have only length `(1 << log2_n) - 1`, but a power of two length is probably faster
        let mut root_of_unity_list = (0..(1 << log2_n)).map(|_| ring.zero()).collect::<Vec<_>>();
        let mut index = 0;
        for s in 0..log2_n {
            let m = 1 << s;
            let log2_group_size = log2_n - s;
            for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                let current_twiddle = root_of_unity_pow(m * bitreverse(i_bitreverse, log2_group_size) as i64);
                root_of_unity_list[index] = current_twiddle;
                index += 1;
            }
        }
        assert_eq!(index, (1 << log2_n) - 1);
        return root_of_unity_list;
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given prime fields, assuming they have
    /// a characteristic congruent to 1 modulo `2^log2_n`.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    pub fn for_zn_with_hom(hom: H, log2_n: usize) -> Option<Self>
        where R_twiddle: ZnRing
    {
        let ring_as_field = hom.domain().as_field().ok().unwrap();
        let root_of_unity = ring_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&ring_as_field, log2_n)?);
        drop(ring_as_field);
        Some(Self::new_with_hom(hom, root_of_unity, log2_n))
    }

    pub fn bitreverse_permute_inplace<V, T>(&self, mut values: V) 
        where V: SwappableVectorViewMut<T>
    {
        assert!(values.len() == 1 << self.log2_n);
        for i in 0..(1 << self.log2_n) {
            if bitreverse(i, self.log2_n) < i {
                values.swap(i, bitreverse(i, self.log2_n));
            }
        }
    }
}

impl<R_main, R_twiddle, H> PartialEq for CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.log2_n == other.log2_n &&
            self.ring().eq_el(self.root_of_unity(self.ring().get_ring()), other.root_of_unity(self.ring().get_ring()))
    }
}

impl<R_main, R_twiddle, H> Clone for CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone
{
    fn clone(&self) -> Self {
        Self {
            hom: self.hom.clone(),
            inv_root_of_unity_list: self.inv_root_of_unity_list.iter().map(|x| self.hom.domain().clone_el(x)).collect(),
            root_of_unity: self.hom.codomain().clone_el(&self.root_of_unity),
            log2_n: self.log2_n,
            root_of_unity_list: self.root_of_unity_list.iter().map(|x| self.hom.domain().clone_el(x)).collect()
        }
    }
}

///
/// A helper trait that defines the Cooley-Tuckey butterfly operation.
/// It is default-implemented for all rings, but for increase FFT performance, some rings
/// might wish to provide a specialization.
/// 
pub trait CooleyTuckeyButterfly<S>: RingBase
    where S: ?Sized + RingBase
{
    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + twiddle * values[i2], values[i1] - twiddle * values[i2])`
    /// 
    fn butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: &H, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);

    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + values[i2], (values[i1] - values[i2]) * twiddle)`
    /// 
    fn inv_butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: &H, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);
}

impl<R, S> CooleyTuckeyButterfly<S> for R
    where S: ?Sized + RingBase, R: ?Sized + RingBase
{
    #[inline(always)]
    default fn butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: &H, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        hom.mul_assign_ref_map(values.at_mut(i2), twiddle);
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
    }

    #[inline(always)]
    default fn inv_butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: &H, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
        hom.mul_assign_ref_map(values.at_mut(i2), twiddle);
    }
}

impl<R_main, R_twiddle, H> CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom.codomain()
    }

    ///
    /// Optimized implementation of the inplace Cooley-Tuckey FFT algorithm.
    /// Note that setting `INV = true` will perform an inverse fourier transform,
    /// except that the division by `n` is not included.
    /// 
    fn unordered_fft_dispatch<V, const INV: bool>(&self, values: &mut V)
        where V: VectorViewMut<R_main::Element> 
    {
        assert!(values.len() == (1 << self.log2_n));

        let hom = &self.hom;
        let R = hom.codomain();

        for step in 0..self.log2_n {

            let (log2_m, log2_group_size_half) = if !INV {
                (self.log2_n - step - 1, step)  
            } else {
                (step, self.log2_n - step - 1)
            };
            let group_size_half = 1 << log2_group_size_half;
            let m = 1 << log2_m;
            let two_m = 2 << log2_m;
            const UNROLL_COUNT: usize = 4;

            if group_size_half < UNROLL_COUNT {

                for k in 0..(1 << log2_m) {

                    let mut root_of_unity_index = (1 << self.log2_n) - 2 * group_size_half;

                    // 
                    // we want to compute a bitreverse_fft_inplace for `v_k, v_(k + m), v_(k + 2m), ..., v_(k + n - m)`;
                    // call this sequence a1
                    //
                    // we already have a bitreverse fft of `v_k, v_(k + 2m), v_(k + 4m), ..., v_(k + n - 2m) `
                    // and `v_(k + m), v_(k + 3m), v_(k + 5m), ..., v_(k + n - m)` in the corresponding entries;
                    // call these sequences a1 and a2
                    //
                    // Note that a1_i is stored in `(k + 2m * bitrev(i, n/m))` and a2_i in `(k + m + 2m * bitrev(i, n/m))`;
                    // We want to store a_i in `(k + m + m * bitrev(i, 2n/m))`
                    //
                    for i_bitreverse in 0..group_size_half {
                        //
                        // we want to compute `(a_i, a_(i + group_size/2)) = (a1_i + z^i a2_i, a1_i - z^i a2_i)`
                        //
                        // in bitreverse order, have
                        // `i_bitreverse     = bitrev(i, group_size) = 2 bitrev(i, group_size/2)` and
                        // `i_bitreverse + 1 = bitrev(i + group_size/2, group_size) = 2 bitrev(i, group_size/2) + 1`
                        //
                        let index1 = i_bitreverse * two_m + k;
                        let index2 = index1 + m;
    
                        if !INV {
                            let current_twiddle = &self.inv_root_of_unity_list[root_of_unity_index];
                            R.get_ring().butterfly(hom, values, current_twiddle, index1, index2);
                        } else {
                            let current_twiddle = &self.root_of_unity_list[root_of_unity_index];
                            R.get_ring().inv_butterfly(hom, values, current_twiddle, index1, index2);
                        }
                        root_of_unity_index += 1;
                    }
                }

            } else {
            
                // same but loop is unrolled

                for k in 0..m {

                    let mut root_of_unity_index = (1 << self.log2_n) - 2 * group_size_half;
                    let mut index1 = k;

                    for _ in (0..group_size_half).step_by(UNROLL_COUNT) {
                        for _ in 0..UNROLL_COUNT {

                            if !INV {
                                let current_twiddle = &self.inv_root_of_unity_list[root_of_unity_index];
                                R.get_ring().butterfly(hom, values, current_twiddle, index1, index1 + m);
                            } else {
                                let current_twiddle = &self.root_of_unity_list[root_of_unity_index];
                                R.get_ring().inv_butterfly(hom, values, current_twiddle, index1, index1 + m);
                            }
                            root_of_unity_index += 1;
                            index1 += two_m;

                        }
                    }
                }
            }
        }
    }
}

impl<R_main, R_twiddle, H> FFTAlgorithm<R_main> for CooleyTuckeyFFT<R_main, R_twiddle, H> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    fn len(&self) -> usize {
        1 << self.log2_n
    }

    fn root_of_unity(&self, ring: &R_main) -> &R_main::Element {
        assert!(ring == self.ring().get_ring(), "unsupported ring");
        &self.root_of_unity
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        self.unordered_fft(&mut values, ring);
        self.bitreverse_permute_inplace(&mut values);
    }

    fn inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        self.bitreverse_permute_inplace(&mut values);
        self.unordered_inv_fft(&mut values, ring);
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        self.unordered_fft_dispatch::<V, false>(&mut values);
    }
    
    fn unordered_inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        self.unordered_fft_dispatch::<V, true>(&mut values);
        let inv = self.hom.domain().invert(&self.hom.domain().int_hom().map(1 << self.log2_n)).unwrap();
        for i in 0..values.len() {
            self.hom.mul_assign_ref_map(values.at_mut(i), &inv);

        }
    }
}

impl<H> FFTErrorEstimate for CooleyTuckeyFFT<Complex64Base, Complex64Base, H> 
    where H: Homomorphism<Complex64Base, Complex64Base>
{
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        // each butterfly doubles the error, and then adds up to 
        let butterfly_absolute_error = input_bound * (root_of_unity_error() + f64::EPSILON);
        // the operator inf-norm of the FFT is its length
        return 2. * self.len() as f64 * butterfly_absolute_error + self.len() as f64 * input_error;
    }
}

#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;
#[cfg(test)]
use crate::rings::zn::zn_big;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::field::*;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_bitreverse_fft_inplace_basic() {
    let ring = Fp::<5>::RING;
    let z = ring.int_hom().map(2);
    let fft = CooleyTuckeyFFT::new(ring, ring.div(&1, &z), 2);
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace_advanced() {
    let ring = Fp::<17>::RING;
    let z = ring.int_hom().map(3);
    let fft = CooleyTuckeyFFT::new(ring, z, 4);
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_inv_fft_inplace() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(&ring, 4).unwrap();
    let values: [u64; 16] = [1, 2, 3, 2, 1, 0, 17 - 1, 17 - 2, 17 - 1, 0, 1, 2, 3, 4, 5, 6];
    let mut work = values;
    fft.unordered_fft(&mut work, ring);
    fft.unordered_inv_fft(&mut work, ring);
    assert_eq!(&work, &values);
}

#[test]
fn test_for_zn() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));

    let ring = Fp::<97>::RING;
    let fft = CooleyTuckeyFFT::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
}

#[cfg(test)]
fn run_fft_bench_round<R, S, H>(fft: &CooleyTuckeyFFT<S, R, H>, data: &Vec<S::Element>, copy: &mut Vec<S::Element>)
    where R: ZnRing, S: ZnRing, H: Homomorphism<R, S>
{
    copy.clear();
    copy.extend(data.iter().map(|x| fft.ring().clone_el(x)));
    fft.unordered_fft(&mut copy[..], &fft.ring());
    fft.unordered_inv_fft(&mut copy[..], &fft.ring());
    assert_el_eq!(fft.ring(), copy[0], data[0]);
}

#[cfg(test)]
const BENCH_SIZE_LOG2: usize = 13;

#[bench]
fn bench_fft(bencher: &mut test::Bencher) {
    let ring = zn_big::Zn::new(StaticRing::<i128>::RING, 1073872897);
    let fft = CooleyTuckeyFFT::for_zn(&ring, BENCH_SIZE_LOG2).unwrap();
    let data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    bencher.iter(|| {
        run_fft_bench_round(&fft, &data, &mut copy)
    });
}

#[bench]
fn bench_fft_zn64_fastmul(bencher: &mut test::Bencher) {
    let ring = zn_64::Zn::new(1073872897);
    let fastmul_ring = zn_64::ZnFastmul::new(ring);
    let fft = CooleyTuckeyFFT::for_zn_with_hom(ring.into_can_hom(fastmul_ring).ok().unwrap(), BENCH_SIZE_LOG2).unwrap();
    let data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    bencher.iter(|| {
        run_fft_bench_round(&fft, &data, &mut copy)
    });
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for log2_n in [4, 7, 11, 15] {
        let fft = CooleyTuckeyFFT::new_with_pows(CC, |x| CC.root_of_unity(x, 1 << log2_n), log2_n);
        let mut array = (0..(1 << log2_n)).map(|i|  CC.root_of_unity(i as i64, 1 << log2_n)).collect::<Vec<_>>();
        fft.fft(&mut array, CC);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}

#[test]
fn test_size_1_fft() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(&ring, 0).unwrap();
    let values: [u64; 1] = [3];
    let mut work = values;
    fft.unordered_fft(&mut work, ring);
    assert_eq!(&work, &values);
    fft.unordered_inv_fft(&mut work, ring);
    assert_eq!(&work, &values);
    assert_eq!(0, fft.unordered_fft_permutation(0));
    assert_eq!(0, fft.unordered_fft_permutation_inv(0));
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_cooley_tuckey_butterfly<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(ring: R, base: S, edge_case_elements: I, test_twiddle: &El<S>)
    where R::Type: CanHomFrom<S::Type>,
        S::Type: DivisibilityRing
{
    let test_inv_twiddle = base.invert(&test_twiddle).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();
    let hom = ring.can_hom(&base).unwrap();

    for a in &elements {
        for b in &elements {

            let mut vector = [ring.clone_el(a), ring.clone_el(b)];
            ring.get_ring().butterfly(&hom, &mut vector, &test_twiddle, 0, 1);
            assert_el_eq!(ring, ring.add_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &vector[0]);
            assert_el_eq!(ring, ring.sub_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &vector[1]);

            ring.get_ring().inv_butterfly(&hom, &mut vector, &test_inv_twiddle, 0, 1);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(a, 2), &vector[0]);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(b, 2), &vector[1]);

            let mut vector = [ring.clone_el(a), ring.clone_el(b)];
            ring.get_ring().butterfly(&hom, &mut vector, &test_twiddle, 1, 0);
            assert_el_eq!(ring, ring.add_ref_fst(b, ring.mul_ref_fst(a, hom.map_ref(test_twiddle))), &vector[1]);
            assert_el_eq!(ring, ring.sub_ref_fst(b, ring.mul_ref_fst(a, hom.map_ref(test_twiddle))), &vector[0]);

            ring.get_ring().inv_butterfly(&hom, &mut vector, &test_inv_twiddle, 1, 0);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(a, 2), &vector[0]);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(b, 2), &vector[1]);
        }
    }
}

#[test]
fn test_butterfly() {
    generic_test_cooley_tuckey_butterfly(zn_static::F17, zn_static::F17, zn_static::F17.elements(), &get_prim_root_of_unity_pow2(zn_static::F17, 4).unwrap());
}