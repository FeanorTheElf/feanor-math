use std::ops::DerefMut;

use crate::algorithms::fft::FFTTable;
use crate::algorithms::unity_root::is_prim_root_of_unity;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::mempool::AllocatingMemoryProvider;
use crate::mempool::MemoryProvider;
use crate::primitive_int::*;
use crate::ring::*;
use crate::algorithms;
use crate::rings::zn::*;use crate::rings::float_complex::*;
use crate::vector::VectorViewMut;
use super::complex_fft::*;

pub struct FFTTableBluestein<R, M: MemoryProvider<El<R>> = AllocatingMemoryProvider>
    where R: RingStore
{
    m_fft_table: algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey<R, M>,
    ///
    /// This is the bitreverse fft of a part of the sequence b_i := z^(i^2) where
    /// z is a 2n-th root of unity.
    /// In particular, we choose the part b_i for 1 - n < i < n. Clearly, the value
    /// at a negative index i must be stored at index (i + m). The other values are
    /// irrelevant.
    /// 
    b_bitreverse_fft: M::Object,
    /// the powers `zeta^(-i * i)` for `zeta` the 2n-th root of unity
    inv_root_of_unity_2n: M::Object,
    /// contrary to expectations, this should be an n-th root of unity and inverse to `inv_root_of_unity^2`
    root_of_unity_n: El<R>,
    n: usize
}

impl<R> FFTTableBluestein<R> 
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity_2n: El<R>, root_of_unity_m: El<R>, n: usize, log2_m: usize) -> Self {
        Self::new_with_mem(ring, root_of_unity_2n, root_of_unity_m, n, log2_m, &AllocatingMemoryProvider)
    }

    pub fn for_zn(ring: R, n: usize) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        Self::for_zn_with_mem(ring, n, &AllocatingMemoryProvider)
    }
}

impl<R, M> FFTTableBluestein<R, M> 
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing,
        M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(ring: R, root_of_unity_2n: El<R>, root_of_unity_m: El<R>, n: usize, log2_m: usize, memory_provider: &M) -> Self {
        // we cannot cannot call `new_with_mem_and_pows` because of borrowing conflicts 

        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity(&ring, &root_of_unity_2n, 2 * n));

        let root_of_unity_2n_pows = |x: i64| if x >= 0 {
            ring.pow(ring.clone_el(&root_of_unity_2n), x as usize % (2 * n))
        } else {
            ring.invert(&ring.pow(ring.clone_el(&root_of_unity_2n), (-x) as usize % (2 * n))).unwrap()
        };
        
        let mut b = Self::create_b_array(&ring, root_of_unity_2n_pows, n, 1 << log2_m, memory_provider);
        let inv_root_of_unity_2n = memory_provider.get_new_init(n, |i| root_of_unity_2n_pows(-((i * i) as i64)));
        let root_of_unity_n = ring.pow(ring.clone_el(&root_of_unity_2n), 2);
        let m_fft_table = algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey::new_with_mem(ring, root_of_unity_m, log2_m, memory_provider);
        m_fft_table.unordered_fft(&mut b[..], m_fft_table.ring(), memory_provider);
        return FFTTableBluestein { 
            m_fft_table: m_fft_table, 
            b_bitreverse_fft: b, 
            inv_root_of_unity_2n: inv_root_of_unity_2n, 
            root_of_unity_n: root_of_unity_n,
            n: n
        };
    }
    
    fn create_b_array<F>(ring: &R, mut root_of_unity_2n_pows: F, n: usize, m: usize, memory_provider: &M) -> M::Object
        where F: FnMut(i64) -> El<R>
    {
        let mut b = memory_provider.get_new_init(m, |_| ring.zero());
        b[0] = ring.one();
        for i in 1..n {
            b[i] = root_of_unity_2n_pows((i * i) as i64 % (2 * n) as i64);
            b[m - i] = ring.clone_el(&b[i]);
        }
        return b;
    }

    pub fn new_with_mem_and_pows<F, G>(ring: R, mut root_of_unity_2n_pows: F, root_of_unity_m_pows: G, n: usize, log2_m: usize, memory_provider: &M) -> Self
        where F: FnMut(i64) -> El<R>,
            G: FnMut(i64) -> El<R>
    {
        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity(&ring, &root_of_unity_2n_pows(1), 2 * n));

        let mut b = Self::create_b_array(&ring, &mut root_of_unity_2n_pows, n, 1 << log2_m, memory_provider);
        let inv_root_of_unity_2n = memory_provider.get_new_init(n, |i| root_of_unity_2n_pows(-((i * i) as i64)));
        let root_of_unity_n = root_of_unity_2n_pows(2);
        let m_fft_table = algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey::new_with_mem_and_pows(ring, root_of_unity_m_pows, log2_m, memory_provider);
        m_fft_table.unordered_fft(&mut b[..], m_fft_table.ring(), memory_provider);
        return FFTTableBluestein { 
            m_fft_table: m_fft_table, 
            b_bitreverse_fft: b, 
            inv_root_of_unity_2n: inv_root_of_unity_2n, 
            root_of_unity_n: root_of_unity_n,
            n: n
        };
    }

    pub fn for_zn_with_mem(ring: R, n: usize, memory_provider: &M) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        let ZZ = StaticRing::<i64>::RING;
        let log2_m = ZZ.abs_log2_ceil(&(2 * n as i64 + 1)).unwrap();
        let root_of_unity_2n = algorithms::unity_root::get_prim_root_of_unity(&ring, 2 * n)?;
        let root_of_unity_m = algorithms::unity_root::get_prim_root_of_unity_pow2(&ring, log2_m)?;
        Some(Self::new_with_mem(ring, root_of_unity_2n, root_of_unity_m, n, log2_m, memory_provider))
    }

    pub fn for_complex_with_mem(ring: R, n: usize, memory_provider: &M) -> Self
        where R: RingStore<Type = Complex64>
    {
        let ZZ = StaticRing::<i64>::RING;
        let CC = Complex64::RING;
        let log2_m = ZZ.abs_log2_ceil(&(2 * n as i64 + 1)).unwrap();
        Self::new_with_mem_and_pows(ring, |x| CC.root_of_unity(x, 2 * n as i64), |x| CC.root_of_unity(x, 1 << log2_m), n, log2_m, memory_provider)
    }

    ///
    /// Computes the FFT of the given values using Bluestein's algorithm.
    /// 
    /// This supports any given ring, as long as the precomputed values stored in the table are
    /// also contained in the new ring. The result is wrong however if the canonical homomorphism
    /// `R -> S` does not map the N-th root of unity to a primitive N-th root of unity.
    /// 
    /// Basically, the idea is to write an FFT of any length (e.g. prime length) as a convolution,
    /// and compute the convolution efficiently using a power-of-two FFT (e.g. with the Cooley-Tuckey 
    /// algorithm).
    /// 
    pub fn fft_base<V, W, S, N, const INV: bool>(&self, mut values: V, ring: S, mut buffer: W, memory_provider: &N)
        where V: VectorViewMut<El<S>>, 
            W: VectorViewMut<El<S>>, 
            S: RingStore, 
            S::Type: CanonicalHom<R::Type>,
            N: MemoryProvider<El<S>>
    {
        assert_eq!(values.len(), self.n);
        assert_eq!(buffer.len(), self.m_fft_table.len());
        let hom = ring.can_hom(self.m_fft_table.ring()).unwrap();

        let base_ring = self.m_fft_table.ring();

        // set buffer to the zero-padded sequence values_i * z^(-i^2/2)
        for i in 0..self.n {
            let value = if INV {
                values.at((self.n - i) % self.n)
            } else {
                values.at(i)
            };
            *buffer.at_mut(i) = ring.clone_el(value);
            ring.get_ring().mul_assign_map_in_ref(base_ring.get_ring(), buffer.at_mut(i), &self.inv_root_of_unity_2n[i], hom.raw_hom());
        }
        for i in self.n..self.m_fft_table.len() {
            *buffer.at_mut(i) = ring.zero();
        }

        // perform convoluted product with b using a power-of-two fft
        self.m_fft_table.unordered_fft(&mut buffer, &ring, memory_provider);
        for i in 0..self.m_fft_table.len() {
            ring.get_ring().mul_assign_map_in_ref(base_ring.get_ring(), buffer.at_mut(i), &self.b_bitreverse_fft[i], hom.raw_hom());
        }
        self.m_fft_table.unordered_inv_fft(&mut buffer, &ring, memory_provider);

        // write values back, and multiply them with a twiddle factor
        for i in 0..self.n {
            *values.at_mut(i) = std::mem::replace(buffer.at_mut(i), ring.zero());
            ring.get_ring().mul_assign_map_in_ref(base_ring.get_ring(), values.at_mut(i), &self.inv_root_of_unity_2n[i], hom.raw_hom());
        }

        if INV {
            // finally, scale by 1/n
            let scale = ring.coerce(&base_ring, base_ring.checked_div(&base_ring.one(), &base_ring.from_int(self.n as i32)).unwrap());
            for i in 0..values.len() {
                ring.mul_assign_ref(values.at_mut(i), &scale);
            }
        }
    }
}

impl<R, M> PartialEq for FFTTableBluestein<R, M> 
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing,
        M: MemoryProvider<El<R>>
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.n == other.n &&
            self.ring().eq_el(self.root_of_unity(), other.root_of_unity())
    }
}

impl<R, M> FFTTable for FFTTableBluestein<R, M> 
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing,
        M: MemoryProvider<El<R>>
{
    type Ring = R;

    fn len(&self) -> usize {
        self.n
    }

    fn ring(&self) -> &R {
        self.m_fft_table.ring()
    }

    fn root_of_unity(&self) -> &El<R> {
        &self.root_of_unity_n
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        i
    }

    fn unordered_fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<R as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        let mut buffer = memory_provider.get_new_init(self.m_fft_table.len(), |_| ring.zero());
        self.fft_base::<_, _, _, N, false>(values, ring, buffer.deref_mut(), memory_provider);
    }

    fn unordered_inv_fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<R as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        let mut buffer = memory_provider.get_new_init(self.m_fft_table.len(), |_| ring.zero());
        self.fft_base::<_, _, _, _, true>(values, ring, buffer.deref_mut(), memory_provider);
    }

    fn fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N) 
        where V: VectorViewMut<El<S>>, 
            S: RingStore, 
            S::Type: CanonicalHom<R::Type>,
            N: MemoryProvider<El<S>>
    {
        self.unordered_fft(values, ring, memory_provider);
    }

    fn inv_fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N) 
        where V: VectorViewMut<El<S>>, 
            S: RingStore, 
            S::Type: CanonicalHom<R::Type>,
            N: MemoryProvider<El<S>>
    {
        self.unordered_inv_fft(values, ring, memory_provider);
    }
}

impl<R: RingStore<Type = Complex64>, M: MemoryProvider<El<R>>> ErrorEstimate for FFTTableBluestein<R, M> {
    
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
    let fft = FFTTableBluestein::new(ring, ring.from_int(36), ring.from_int(111), 5, 4);
    let mut values = [1, 3, 2, 0, 7];
    let mut buffer = [0; 16];
    fft.fft_base::<_, _, _, _, false>(&mut values, ring, &mut buffer, &AllocatingMemoryProvider);
    let expected = [13, 137, 202, 206, 170];
    assert_eq!(expected, values);
}

#[test]
fn test_inv_fft_base() {
    let ring = Zn::<241>::RING;
    // a 5-th root of unity is 91 
    let fft = FFTTableBluestein::new(ring, ring.from_int(36), ring.from_int(111), 5, 4);
    let values = [1, 3, 2, 0, 7];
    let mut work = values;
    let mut buffer = [0; 16];
    fft.fft_base::<_, _, _, _, false>(&mut work, ring, &mut buffer, &AllocatingMemoryProvider);
    fft.fft_base::<_, _, _, _, true>(&mut work, ring, &mut buffer, &AllocatingMemoryProvider);
    assert_eq!(values, work);
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for (p, _log2_m) in [(5, 4), (53, 7), (1009, 11)] {
        let fft = FFTTableBluestein::for_complex_with_mem(&CC, p, &AllocatingMemoryProvider);
        let mut array = AllocatingMemoryProvider.get_new_init(p as usize, |i| CC.root_of_unity(i as i64, p as i64));
        fft.fft(&mut array, CC, &AllocatingMemoryProvider);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}