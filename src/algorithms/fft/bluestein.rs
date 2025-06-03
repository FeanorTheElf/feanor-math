use std::alloc::Allocator;
use std::alloc::Global;
use std::fmt::Debug;

use crate::algorithms::fft::FFTAlgorithm;
use crate::algorithms::unity_root::*;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::integer::IntegerRingStore;
use crate::primitive_int::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::zn::*;
use crate::rings::float_complex::*;
use crate::algorithms::fft::complex_fft::*;
use crate::seq::SwappableVectorViewMut;

type BaseFFT<R_main, R_twiddle, H, A> = CooleyTuckeyFFT<R_main, R_twiddle, H, A>;

///
/// Bluestein's FFT algorithm (also known as Chirp-Z-transform) to compute the Fourier
/// transform of arbitrary length (including prime numbers).
/// 
pub struct BluesteinFFT<R_main, R_twiddle, H, A = Global>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone, 
        A: Allocator + Clone
{
    m_fft_table: BaseFFT<R_main, R_twiddle, H, A>,
    b_unordered_fft: Vec<R_twiddle::Element>,
    twiddles: Vec<R_twiddle::Element>,
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
    pub fn for_complex_with_hom(hom: H, n: usize, tmp_mem_allocator: A) -> Self{
        let ZZ = StaticRing::<i64>::RING;
        let CC = Complex64::RING;
        let n_i64: i64 = n.try_into().unwrap();
        let log2_m = ZZ.abs_log2_ceil(&(2 * n_i64 + 1)).unwrap();
        Self::new_with_pows_with_hom(hom, |x| CC.root_of_unity(x, 2 * n_i64), |x| CC.root_of_unity(x, 1 << log2_m), n, log2_m, tmp_mem_allocator)
    }
}

impl<R, A> BluesteinFFT<Complex64Base, Complex64Base, Identity<R>, A>
    where R: RingStore<Type = Complex64Base> + Clone, 
        A: Allocator + Clone
{
    ///
    /// Creates an [`BluesteinFFT`] for the complex field.
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
    pub fn new_with_hom(hom: H, root_of_unity_2n: R_twiddle::Element, root_of_unity_m: R_twiddle::Element, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self {
        let hom_copy = hom.clone();
        let twiddle_ring = hom_copy.domain();
        return Self::new_with_pows_with_hom(
            hom, 
            |i: i64| if i >= 0 {
                twiddle_ring.pow(twiddle_ring.clone_el(&root_of_unity_2n), i as usize % (2 * n))
            } else {
                twiddle_ring.invert(&twiddle_ring.pow(twiddle_ring.clone_el(&root_of_unity_2n), (-i) as usize % (2 * n))).unwrap()
            }, 
            |i: i64| if i >= 0 {
                twiddle_ring.pow(twiddle_ring.clone_el(&root_of_unity_m), i as usize)
            } else {
                twiddle_ring.invert(&twiddle_ring.pow(twiddle_ring.clone_el(&root_of_unity_m), (-i) as usize)).unwrap()
            }, 
            n, 
            log2_m, 
            tmp_mem_allocator
        );
    }

    ///
    /// Creates an [`BluesteinFFT`] for the given rings, using the given function to create
    /// the necessary powers of roots of unity.
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
    pub fn new_with_pows_with_hom<F, G>(hom: H, mut root_of_unity_2n_pows: F, mut root_of_unity_m_pows: G, n: usize, log2_m: usize, tmp_mem_allocator: A) -> Self
        where F: FnMut(i64) -> R_twiddle::Element,
            G: FnMut(i64) -> R_twiddle::Element
    {
        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);
        assert!(hom.domain().get_ring().is_approximate() || is_prim_root_of_unity(hom.domain(), &root_of_unity_m_pows(1), 1 << log2_m));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity(hom.codomain(), &hom.map(root_of_unity_m_pows(1)), 1 << log2_m));

        let m_fft_table = CooleyTuckeyFFT::create(
            hom, 
            &mut root_of_unity_m_pows, 
            log2_m, 
            tmp_mem_allocator
        );
        return Self::create(m_fft_table, |i| root_of_unity_2n_pows(2 * i), n);
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
    pub fn for_zn_with_hom(hom: H, n: usize, tmp_mem_allocator: A) -> Option<Self>
        where R_twiddle: ZnRing
    {
        let ring_as_field = hom.domain().as_field().ok().unwrap();
        let root_of_unity_2n = ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity(&ring_as_field, 2 * n)?);
        let log2_m = StaticRing::<i64>::RING.abs_log2_ceil(&(n * 2).try_into().unwrap()).unwrap();
        let root_of_unity_m = ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(&ring_as_field, log2_m)?);
        return Some(Self::new_with_hom(hom, root_of_unity_2n, root_of_unity_m, n, log2_m, tmp_mem_allocator));
    }
    
    ///
    /// Most general way to construct a [`BluesteinFFT`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create<F>(m_fft_table: BaseFFT<R_main, R_twiddle, H, A>, mut root_of_unity_n_pows: F, n: usize) -> Self
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let hom = m_fft_table.hom().clone();
        let m = m_fft_table.len();
        assert!(m >= 2 * n);
        assert!(n % 2 == 1);
        assert!(hom.codomain().is_commutative());
        assert!(hom.domain().get_ring().is_approximate() || is_prim_root_of_unity(hom.domain(), &root_of_unity_n_pows(1), n));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity(hom.codomain(), &hom.map(root_of_unity_n_pows(1)), n));

        let (twiddle_fft, old_hom) = m_fft_table.change_ring(hom.domain().identity());

        let half_mod_n = (n + 1) / 2;
        let mut b: Vec<_> = (0..n).map(|i| root_of_unity_n_pows(TryInto::<i64>::try_into(i * i * half_mod_n).unwrap())).collect();
        b.resize_with(m, || hom.domain().zero());
        
        twiddle_fft.unordered_fft(&mut b, hom.domain());

        let twiddles = (0..n).map(|i| root_of_unity_n_pows(-TryInto::<i64>::try_into(i * i * half_mod_n).unwrap())).collect::<Vec<_>>();
        let root_of_unity_n = hom.map(root_of_unity_n_pows(1));

        return BluesteinFFT { 
            m_fft_table: twiddle_fft.change_ring(old_hom).0, 
            b_unordered_fft: b, 
            twiddles: twiddles, 
            root_of_unity_n: root_of_unity_n,
            n: n
        };
    }

    #[stability::unstable(feature = "enable")]
    pub fn allocator(&self) -> &A {
        self.m_fft_table.allocator()
    }

    ///
    /// Computes the FFT of the given values using Bluestein's algorithm, using only the passed
    /// buffer as temporary storage.
    /// 
    /// This will not allocate additional memory, as opposed to [`BluesteinFFT::fft()`] etc.
    /// 
    /// Basically, the idea is to write an FFT of any length (e.g. prime length) as a convolution,
    /// and compute the convolution efficiently using a power-of-two FFT (e.g. with the Cooley-Tukey 
    /// algorithm).
    /// 
    /// TODO: At next breaking release, make this private
    /// 
    pub fn fft_base<V, W, const INV: bool>(&self, values: V, _buffer: W)
        where V: SwappableVectorViewMut<R_main::Element>, 
            W: SwappableVectorViewMut<R_main::Element>
    {
        if INV {
            self.unordered_inv_fft(values, self.ring());
        } else {
            self.unordered_fft(values, self.ring());
        }
    }

    fn fft_base_impl<V, A2, const INV: bool>(&self, mut values: V, mut buffer: Vec<R_main::Element, A2>)
        where V: SwappableVectorViewMut<R_main::Element>,
            A2: Allocator
    {
        assert_eq!(values.len(), self.n);
        assert_eq!(buffer.len(), self.m_fft_table.len());

        let ring = self.m_fft_table.hom().codomain();

        // set buffer to the zero-padded sequence values_i * z^(-i^2/2)
        for i in 0..self.n {
            let value = if INV {
                values.at((self.n - i) % self.n)
            } else {
                values.at(i)
            };
            buffer[i] = self.hom().mul_ref_map(value, &self.twiddles[i]);
        }
        for i in self.n..self.m_fft_table.len() {
            buffer[i] = ring.zero();
        }
 
        self.m_fft_table.unordered_truncated_fft(&mut buffer, self.n * 2);
        for i in 0..self.m_fft_table.len() {
            self.hom().mul_assign_ref_map(&mut buffer[i], &self.b_unordered_fft[i]);
        }
        self.m_fft_table.unordered_truncated_fft_inv(&mut buffer, self.n * 2);
        
        // make the normal convolution into a cyclic convolution of length n by taking it modulo `x^n - 1`
        let (buffer1, buffer2) = buffer[..(2 * self.n)].split_at_mut(self.n);
        for (a, b) in buffer1.iter_mut().zip(buffer2.iter_mut()) {
            ring.add_assign_ref(a, b);
        }

        // write values back, and multiply them with a twiddle factor
        for (i, x) in buffer.into_iter().enumerate().take(self.n) {
            *values.at_mut(i) = self.hom().mul_ref_snd_map(x, &self.twiddles[i]);
        }

        if INV {
            // finally, scale by 1/n
            let scale = self.hom().map(self.hom().domain().checked_div(&self.hom().domain().one(), &self.hom().domain().int_hom().map(self.n.try_into().unwrap())).unwrap());
            for i in 0..values.len() {
                ring.mul_assign_ref(values.at_mut(i), &scale);
            }
        }
    }

    fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom().codomain()
    }

    fn hom(&self) -> &H {
        self.m_fft_table.hom()
    }
}

impl<R_main, R_twiddle, H, A> PartialEq for BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone, 
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.n == other.n &&
            self.ring().eq_el(self.root_of_unity(self.ring()), other.root_of_unity(self.ring()))
    }
}

impl<R_main, R_twiddle, H, A> Debug for BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase + Debug,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone, 
        A: Allocator + Clone
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bluestein FFT of length {} over ring {:?}", self.n, self.ring().get_ring())
    }
}

impl<R_main, R_twiddle, H, A> FFTAlgorithm<R_main> for BluesteinFFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone, 
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
        let mut buffer = Vec::with_capacity_in(self.m_fft_table.len(), self.allocator().clone());
        buffer.extend((0..self.m_fft_table.len()).map(|_| self.ring().zero()));
        self.fft_base_impl::<_, _, false>(values, buffer);
    }

    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        let mut buffer = Vec::with_capacity_in(self.m_fft_table.len(), self.allocator().clone());
        buffer.extend((0..self.m_fft_table.len()).map(|_| self.ring().zero()));
        self.fft_base_impl::<_, _, true>(values, buffer);
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
    where H: Homomorphism<Complex64Base, Complex64Base> + Clone, 
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
    let fastmul_ring = zn_64::ZnFastmul::new(ring).unwrap();
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
        let mut array = (0..p).map(|i| CC.root_of_unity(i.try_into().unwrap(), p.try_into().unwrap())).collect::<Vec<_>>();
        fft.fft(&mut array, CC);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}

#[cfg(test)]
const BENCH_SIZE: usize = 1009;

#[bench]
fn bench_bluestein(bencher: &mut test::Bencher) {
    let ring = zn_64::Zn::new(18597889);
    let fastmul_ring = zn_64::ZnFastmul::new(ring).unwrap();
    let embedding = ring.can_hom(&fastmul_ring).unwrap();
    let ring_as_field = ring.as_field().ok().unwrap();
    let root_of_unity = fastmul_ring.coerce(&ring, ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity(&ring_as_field, 2 * BENCH_SIZE).unwrap()));
    let fastmul_ring_as_field = fastmul_ring.as_field().ok().unwrap();
    let fft = BluesteinFFT::new_with_hom(
        embedding.clone(), 
        root_of_unity, 
        fastmul_ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(&fastmul_ring_as_field, 11).unwrap()), 
        BENCH_SIZE, 
        11, 
        Global
    );
    let data = (0..BENCH_SIZE).map(|i| ring.int_hom().map(i as i32)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(BENCH_SIZE);
    bencher.iter(|| {
        copy.clear();
        copy.extend(data.iter().map(|x| ring.clone_el(x)));
        fft.unordered_fft(std::hint::black_box(&mut copy[..]), &ring);
        fft.unordered_inv_fft(std::hint::black_box(&mut copy[..]), &ring);
        assert_el_eq!(ring, copy[0], data[0]);
    });
}
