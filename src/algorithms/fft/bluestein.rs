use std::ops::DerefMut;

use crate::algorithms::fft::FFTTable;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::mempool::AllocatingMemoryProvider;
use crate::mempool::MemoryProvider;
use crate::primitive_int::*;
use crate::ring::*;
use crate::algorithms;
use crate::rings::zn::*;
use crate::vector::VectorViewMut;

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
    /// contrary to expectations, this should be a 2n-th root of unity
    inv_root_of_unity_2n: El<R>,
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
        // checks on m and root_of_unity_m are done by the FFTTableCooleyTuckey
        assert!((1 << log2_m) >= 2 * n + 1);

        let m = 1 << log2_m;
        let mut b = memory_provider.get_new_init(m, |_| ring.zero());
        b[0] = ring.one();
        for i in 1..n {
            b[i] = ring.pow(ring.clone_el(&root_of_unity_2n), i * i);
            b[m - i] = ring.clone_el(&b[i]);
        }
        let inv_root_of_unity = ring.pow(ring.clone_el(&root_of_unity_2n), 2 * n - 1);
        let root_of_unity_n = ring.pow(root_of_unity_2n, 2);
        let m_fft_table = algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey::new_with_mem(ring, root_of_unity_m, log2_m, memory_provider);
        m_fft_table.unordered_fft(&mut b[..], m_fft_table.ring(), memory_provider);
        return FFTTableBluestein { 
            m_fft_table: m_fft_table, 
            b_bitreverse_fft: b, 
            inv_root_of_unity_2n: inv_root_of_unity, 
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
            ring.get_ring().mul_assign_map_in(base_ring.get_ring(), buffer.at_mut(i), base_ring.pow(base_ring.clone_el(&self.inv_root_of_unity_2n), i * i), hom.raw_hom());
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
            ring.get_ring().mul_assign_map_in(base_ring.get_ring(), values.at_mut(i), base_ring.pow(base_ring.clone_el(&self.inv_root_of_unity_2n), i * i), hom.raw_hom());
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