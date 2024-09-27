use std::alloc::Allocator;
use std::alloc::Global;

use crate::algorithms::fft::FFTAlgorithm;
use crate::algorithms::unity_root::is_prim_root_of_unity;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::primitive_int::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::algorithms;
use crate::rings::zn::*;use crate::rings::float_complex::*;
use crate::algorithms::fft::complex_fft::*;
use crate::seq::SwappableVectorViewMut;

///
/// Bluestein's FFT algorithm (also known as Chirp-Z-transform) to compute the Fourier
/// transform of arbitrary length (including prime numbers).
/// 
pub struct BluesteinFFT<R_main, R_twiddle, H, A = Global>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>, 
        A: Allocator + Clone
{
    hom: H,
    m_fft_table: algorithms::fft::cooley_tuckey::CooleyTuckeyFFT<R_main, R_twiddle, H>,
    tmp_mem_allocator: A,
    ///
    /// This is the bitreverse fft of a part of the sequence b_i := z^(i^2) where
    /// z is a 2n-th root of unity.
    /// In particular, we choose the part b_i for 1 - n < i < n. Clearly, the value
    /// at a negative index i must be stored at index (i + m). The other values are
    /// irrelevant.
    /// 
    b_bitreverse_fft: Vec<R_twiddle::Element>,
    /// the powers `zeta^(-i * i)` for `zeta` the 2n-th root of unity
    inv_root_of_unity_2n: Vec<R_twiddle::Element>,
    /// contrary to expectations, this should be an n-th root of unity and inverse to `inv_root_of_unity^2`
    root_of_unity_n: R_main::Element,
    n: usize
}

impl<H, A> BluesteinFFT<Complex64Base, Complex64Base, H, A>
    where H: Homomorphism<Complex64Base, Complex64Base> + Clone, 
        A: Allocator + Clone
{
    ///
    /// Creates an [`BluesteinFFT`] for the complex field, using the given homomorphism
    /// to connect the ring implementation for twiddles with the main ring implementation.
    /// 
    /// This function is mainly provided for parity with other rings, since in the complex case
    /// it currently does not make much sense to use a different homomorphism than the identity.
    /// Hence, it is simpler to use [`BluesteinFFT::for_complex()`].
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`, where `2^log2_m` is the smallest power of two that is `> 2n`.
    /// 
    pub fn for_complex_with_hom(hom: H, n: usize, tmp_mem_allocator: A) -> Self{
        let ZZ = StaticRing::<i64>::RING;
        let CC = Complex64::RING;
        let log2_m = ZZ.abs_log2_ceil(&(2 * n as i64 + 1)).unwrap();
        Self::new_with_pows_with_hom(hom, |x| CC.root_of_unity(x, 2 * n as i64), |x| CC.root_of_unity(x, 1 << log2_m), n, log2_m, tmp_mem_allocator)
    }
}

impl<R, A> BluesteinFFT<Complex64Base, Complex64Base, Identity<R>, A>
    where R: RingStore<Type = Complex64Base> + Clone, 
        A: Allocator + Clone
{
    ///
    /// Creates an [`BluesteinFFT`] for the complex field.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`, where `2^log2_m` is the smallest power of two that is `> 2n`.
    /// 
    pub fn for_complex(ring: R, n: usize, tmp_mem_allocator: A) -> Self{
        Self::for_complex_with_hom(ring.into_identity(), n, tmp_mem_allocator)
    }
}

impl<R, A> BluesteinFFT<R::Type, R::Type, Identity<R>, A>
    where R: RingStore + Clone,
        R::Type: DivisibilityRing,
        A: Allocator + Clone
{
    ///
    /// Creates an [`BluesteinFFT`] for the given ring, using the given root of unity
    /// as base.
    /// 
    /// It is necessary that `root_of_unity_2n` is a primitive `2n`-th root of unity, and
    /// `root_of_unity_m` is a `2^log2_m`-th root of unity, where `2^log2_m > 2n`.
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`.
    /// 
    pub fn new(ring: R, root_of_unity_2n: El<R>, root_of_unity_m: El<R>, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self {
        Self::new_with_hom(ring.into_identity(), root_of_unity_2n, root_of_unity_m, n, log2_m, tmp_mem_allocator)
    }

    ///
    /// Creates an [`BluesteinFFT`] for the given ring, using the passed function to
    /// provide the necessary roots of unity.
    /// 
    /// Concretely, `root_of_unity_2n_pows(i)` should return `z^i`, where `z` is a `2n`-th
    /// primitive root of unity, and `root_of_unity_m_pows(i)` should return `w^i` where `w`
    /// is a `2^log2_m`-th primitive root of unity, where `2^log2_m > 2n`.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`.
    /// 
    pub fn new_with_pows<F, G>(ring: R, root_of_unity_2n_pows: F, root_of_unity_m_pows: G, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self 
        where F: FnMut(i64) -> El<R>,
            G: FnMut(i64) -> El<R>
    {
        Self::new_with_pows_with_hom(ring.into_identity(), root_of_unity_2n_pows, root_of_unity_m_pows, n, log2_m, tmp_mem_allocator)
    }

    ///
    /// Creates an [`BluesteinFFT`] for a prime field, assuming it has suitable roots of
    /// unity.
    /// 
    /// Concretely, this requires that the characteristic `p` is congruent to 1 modulo
    /// `2^log2_m n`, where `2^log2_m` is the smallest power of two that is `> 2n`.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`, where `2^log2_m` is the smallest power of two that is `> 2n`.
    /// 
    pub fn for_zn(ring: R, n: usize, tmp_mem_allocator: A) -> Option<Self>
        where R::Type: ZnRing
    {
        Self::for_zn_with_hom(ring.into_identity(), n, tmp_mem_allocator)
    }
}

impl<R_main, R_twiddle, H, A> BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone, 
        A: Allocator + Clone
{
    ///
    /// Creates an [`BluesteinFFT`] for the given rings, using the given root of unity.
    /// 
    /// It is necessary that `root_of_unity_2n` is a primitive `2n`-th root of unity, and
    /// `root_of_unity_m` is a `2^log2_m`-th root of unity, where `2^log2_m > 2n`.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`.
    /// 
    pub fn new_with_hom(hom: H, root_of_unity_2n: R_twiddle::Element, root_of_unity_m: R_twiddle::Element, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self {
        // we cannot cannot call `new_with_mem_and_pows` because of borrowing conflicts 

        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);
        assert!(!hom.domain().get_ring().is_approximate());
        assert!(is_prim_root_of_unity(hom.domain(), &root_of_unity_2n, 2 * n));
        assert!(is_prim_root_of_unity(hom.codomain(), &hom.map_ref(&root_of_unity_2n), 2 * n));
        assert!(is_prim_root_of_unity(hom.domain(), &root_of_unity_m, 1 << log2_m));
        assert!(is_prim_root_of_unity(hom.codomain(), &hom.map_ref(&root_of_unity_m), 1 << log2_m));

        let root_of_unity_2n_pows = |x: i64| if x >= 0 {
            hom.domain().pow(hom.domain().clone_el(&root_of_unity_2n), x as usize % (2 * n))
        } else {
            hom.domain().invert(&hom.domain().pow(hom.domain().clone_el(&root_of_unity_2n), (-x) as usize % (2 * n))).unwrap()
        };
        
        let mut b = Self::create_b_array(hom.domain().get_ring(), root_of_unity_2n_pows, n, 1 << log2_m);
        let inv_root_of_unity_2n = (0..n).map(|i| root_of_unity_2n_pows(-((i * i) as i64))).collect::<Vec<_>>();
        let root_of_unity_n = hom.codomain().pow(hom.map_ref(&root_of_unity_2n), 2);

        let m_fft_table_base = algorithms::fft::cooley_tuckey::CooleyTuckeyFFT::new(hom.domain(), hom.domain().clone_el(&root_of_unity_m), log2_m);
        m_fft_table_base.unordered_fft(&mut b[..], &hom.domain());
        drop(m_fft_table_base);

        // we actually require that this has the same ordering of outputs as `m_fft_table_base`, but since it's Cooley-Tuckey, both are bitreversed
        let m_fft_table = algorithms::fft::cooley_tuckey::CooleyTuckeyFFT::new_with_hom(hom.clone(), root_of_unity_m, log2_m);
        return BluesteinFFT { 
            m_fft_table: m_fft_table, 
            b_bitreverse_fft: b, 
            inv_root_of_unity_2n: inv_root_of_unity_2n, 
            root_of_unity_n: root_of_unity_n,
            n: n,
            tmp_mem_allocator: tmp_mem_allocator,
            hom: hom
        };
    }
    
    fn create_b_array<F>(ring: &R_twiddle, mut root_of_unity_2n_pows: F, n: usize, m: usize) -> Vec<R_twiddle::Element>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let mut b = (0..m).map(|_| ring.zero()).collect::<Vec<_>>();
        b[0] = ring.one();
        for i in 1..n {
            b[i] = root_of_unity_2n_pows((i * i) as i64 % (2 * n) as i64);
            b[m - i] = ring.clone_el(&b[i]);
        }
        return b;
    }

    ///
    /// Creates an [`BluesteinFFT`] for the given rings, using the given function to create
    /// the necessary powers of roots of unity. This is the most generic way to create [`BluesteinFFT`].
    /// 
    /// Concretely, `root_of_unity_2n_pows(i)` should return `z^i`, where `z` is a `2n`-th
    /// primitive root of unity, and `root_of_unity_m_pows(i)` should return `w^i` where `w`
    /// is a `2^log2_m`-th primitive root of unity, where `2^log2_m > 2n`.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`
    /// 
    pub fn new_with_pows_with_hom<F, G>(hom: H, mut root_of_unity_2n_pows: F, mut root_of_unity_m_pows: G, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self
        where F: FnMut(i64) -> R_twiddle::Element,
            G: FnMut(i64) -> R_twiddle::Element
    {
        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);
        assert!(hom.domain().get_ring().is_approximate() || is_prim_root_of_unity(hom.domain(), &root_of_unity_2n_pows(1), 2 * n));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity(hom.codomain(), &hom.map(root_of_unity_2n_pows(1)), 2 * n));
        assert!(hom.domain().get_ring().is_approximate() || is_prim_root_of_unity(hom.domain(), &root_of_unity_m_pows(1), 1 << log2_m));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity(hom.codomain(), &hom.map(root_of_unity_m_pows(1)), 1 << log2_m));

        let mut b = Self::create_b_array(hom.domain().get_ring(), &mut root_of_unity_2n_pows, n, 1 << log2_m);
        let inv_root_of_unity_2n = (0..n).map(|i| root_of_unity_2n_pows(-((i * i) as i64))).collect::<Vec<_>>();
        let root_of_unity_n = hom.map(root_of_unity_2n_pows(2));

        let m_fft_table_base = algorithms::fft::cooley_tuckey::CooleyTuckeyFFT::new_with_pows(hom.domain(), &mut root_of_unity_m_pows, log2_m);
        m_fft_table_base.unordered_fft(&mut b[..], &hom.domain());
        drop(m_fft_table_base);

        // we actually require that this has the same ordering of outputs as `m_fft_table_base`, but since it's Cooley-Tuckey, both are bitreversed
        let m_fft_table = algorithms::fft::cooley_tuckey::CooleyTuckeyFFT::new_with_pows_with_hom(hom.clone(), root_of_unity_m_pows, log2_m);
        return BluesteinFFT { 
            m_fft_table: m_fft_table, 
            b_bitreverse_fft: b, 
            inv_root_of_unity_2n: inv_root_of_unity_2n, 
            root_of_unity_n: root_of_unity_n,
            n: n,
            tmp_mem_allocator: tmp_mem_allocator,
            hom: hom
        };
    }

    ///
    /// Creates an [`BluesteinFFT`] for the given prime fields, assuming they have suitable
    /// roots of unity.
    /// 
    /// Concretely, this requires that the characteristic `p` is congruent to 1 modulo
    /// `2^log2_m n`, where `2^log2_m` is the smallest power of two that is `> 2n`.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// The given `tmp_mem_allocator` will be used for allocating temporary memory during the
    /// FFT computations, but not for storing precomputed data that will live as long as the FFT table
    /// itself. Currently, it suffices if `tmp_mem_allocator` supports the allocation of arrays `[R_main::Element]`
    /// of length `2^log2_m`, where `2^log2_m` is the smallest power of two that is `> 2n`. 
    /// 
    pub fn for_zn_with_hom(hom: H, n: usize, tmp_mem_allocator: A) -> Option<Self>
        where R_twiddle: ZnRing
    {
        let ZZ = StaticRing::<i64>::RING;
        let log2_m = ZZ.abs_log2_ceil(&(2 * n as i64 + 1)).unwrap();
        let ring_as_field = hom.domain().as_field().ok().unwrap();
        let root_of_unity_2n = ring_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity(&ring_as_field, 2 * n)?);
        let root_of_unity_m = ring_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&ring_as_field, log2_m)?);
        drop(ring_as_field);
        Some(Self::new_with_hom(hom, root_of_unity_2n, root_of_unity_m, n, log2_m, tmp_mem_allocator))
    }
}

impl<R_main, R_twiddle, H, A> BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>, 
        A: Allocator + Clone
{
    ///
    /// Computes the FFT of the given values using Bluestein's algorithm, using only the passed
    /// buffer as temporary storage.
    /// 
    /// This will not allocate additional memory, as opposed to [`BluesteinFFT::fft()`] etc.
    /// 
    /// Basically, the idea is to write an FFT of any length (e.g. prime length) as a convolution,
    /// and compute the convolution efficiently using a power-of-two FFT (e.g. with the Cooley-Tuckey 
    /// algorithm).
    /// 
    pub fn fft_base<V, W, const INV: bool>(&self, mut values: V, mut buffer: W)
        where V: SwappableVectorViewMut<R_main::Element>, 
            W: SwappableVectorViewMut<R_main::Element>
    {
        assert_eq!(values.len(), self.n);
        assert_eq!(buffer.len(), self.m_fft_table.len());

        let ring = self.hom.codomain();

        // set buffer to the zero-padded sequence values_i * z^(-i^2/2)
        for i in 0..self.n {
            let value = if INV {
                values.at((self.n - i) % self.n)
            } else {
                values.at(i)
            };
            *buffer.at_mut(i) = ring.clone_el(value);
            self.hom.mul_assign_ref_map(buffer.at_mut(i), &self.inv_root_of_unity_2n[i]);
        }
        for i in self.n..self.m_fft_table.len() {
            *buffer.at_mut(i) = ring.zero();
        }

        // perform convoluted product with b using a power-of-two fft
        self.m_fft_table.unordered_fft(&mut buffer, &self.ring());
        for i in 0..self.m_fft_table.len() {
            self.hom.mul_assign_ref_map(buffer.at_mut(i), &self.b_bitreverse_fft[i]);
        }
        self.m_fft_table.unordered_inv_fft(&mut buffer, &self.ring());

        // write values back, and multiply them with a twiddle factor
        for i in 0..self.n {
            *values.at_mut(i) = std::mem::replace(buffer.at_mut(i), ring.zero());
            self.hom.mul_assign_ref_map(values.at_mut(i), &self.inv_root_of_unity_2n[i]);
        }

        if INV {
            // finally, scale by 1/n
            let scale = self.hom.map(self.hom.domain().checked_div(&self.hom.domain().one(), &self.hom.domain().int_hom().map(self.n as i32)).unwrap());
            for i in 0..values.len() {
                ring.mul_assign_ref(values.at_mut(i), &scale);
            }
        }
    }

    fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom.codomain()
    }

}

impl<R_main, R_twiddle, H, A> PartialEq for BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>, 
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.n == other.n &&
            self.ring().eq_el(self.root_of_unity(self.ring()), other.root_of_unity(self.ring()))
    }
}

impl<R_main, R_twiddle, H, A> FFTAlgorithm<R_main> for BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>, 
        A: Allocator + Clone
{
    fn len(&self) -> usize {
        self.n
    }

    fn root_of_unity<S: RingStore<Type = R_main> + Copy>(&self, ring: S) -> &R_main::Element {
        assert!(self.ring().get_ring() == ring.get_ring(), "unsupported ring");
        &self.root_of_unity_n
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        i
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        i
    }

    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        let mut buffer = Vec::with_capacity_in(self.m_fft_table.len(), self.tmp_mem_allocator.clone());
        buffer.extend((0..self.m_fft_table.len()).map(|_| self.hom.codomain().zero()));
        self.fft_base::<_, _, false>(values, &mut buffer[..]);
    }

    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        let mut buffer = Vec::with_capacity_in(self.m_fft_table.len(), self.tmp_mem_allocator.clone());
        buffer.extend((0..self.m_fft_table.len()).map(|_| self.hom.codomain().zero()));
        self.fft_base::<_, _, true>(values, &mut buffer[..]);
    }

    fn fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        self.unordered_fft(values, ring);
    }

    fn inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        self.unordered_inv_fft(values, ring);
    }
}

impl<H, A> FFTErrorEstimate for BluesteinFFT<Complex64Base, Complex64Base, H, A>
    where H: Homomorphism<Complex64Base, Complex64Base>, 
        A: Allocator + Clone
{
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        let error_after_twiddling = input_error + input_bound * (root_of_unity_error() + f64::EPSILON);
        let error_after_fft = self.m_fft_table.expected_absolute_error(input_bound, error_after_twiddling);
        let b_bitreverse_fft_error = self.m_fft_table.expected_absolute_error(1., root_of_unity_error());
        // now the values are increased by up to a factor of m, so use `input_bound * m` instead
        let new_input_bound = input_bound * self.m_fft_table.len() as f64;
        let b_bitreverse_fft_bound = self.m_fft_table.len() as f64;
        let error_after_mul = new_input_bound * b_bitreverse_fft_error + b_bitreverse_fft_bound * error_after_fft + f64::EPSILON * new_input_bound * b_bitreverse_fft_bound;
        let error_after_inv_fft = self.m_fft_table.expected_absolute_error(new_input_bound * b_bitreverse_fft_bound, error_after_mul) / self.m_fft_table.len() as f64;
        let error_end = error_after_inv_fft + new_input_bound * (root_of_unity_error() + f64::EPSILON);
        return error_end;
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;

#[test]
fn test_fft_base() {
    let ring = Zn::<241>::RING;
    // a 5-th root of unity is 91 
    let fft = BluesteinFFT::new(ring, ring.int_hom().map(36), ring.int_hom().map(111), 5, 4, Global);
    let mut values = [1, 3, 2, 0, 7];
    let mut buffer = [0; 16];
    fft.fft_base::<_, _, false>(&mut values, &mut buffer);
    let expected = [13, 137, 202, 206, 170];
    assert_eq!(expected, values);
}

#[test]
fn test_fft_fastmul() {
    let ring = zn_64::Zn::new(241);
    let fastmul_ring = zn_64::ZnFastmul::new(ring);
    let fft = BluesteinFFT::new_with_hom(ring.can_hom(&fastmul_ring).unwrap(), fastmul_ring.int_hom().map(36), fastmul_ring.int_hom().map(111), 5, 4, Global);
    let mut values: [_; 5] = std::array::from_fn(|i| ring.int_hom().map([1, 3, 2, 0, 7][i]));
    fft.fft(&mut values, ring);
    let expected: [_; 5] = std::array::from_fn(|i| ring.int_hom().map([13, 137, 202, 206, 170][i]));
    for i in 0..values.len() {
        assert_el_eq!(ring, expected[i], values[i]);
    }
}

#[test]
fn test_inv_fft_base() {
    let ring = Zn::<241>::RING;
    // a 5-th root of unity is 91 
    let fft = BluesteinFFT::new(ring, ring.int_hom().map(36), ring.int_hom().map(111), 5, 4, Global);
    let values = [1, 3, 2, 0, 7];
    let mut work = values;
    let mut buffer = [0; 16];
    fft.fft_base::<_, _, false>(&mut work, &mut buffer);
    fft.fft_base::<_, _, true>(&mut work, &mut buffer);
    assert_eq!(values, work);
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for (p, _log2_m) in [(5, 4), (53, 7), (1009, 11)] {
        let fft = BluesteinFFT::for_complex(&CC, p, Global);
        let mut array = (0..(p as usize)).map(|i| CC.root_of_unity(i as i64, p as i64)).collect::<Vec<_>>();
        fft.fft(&mut array, CC);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}