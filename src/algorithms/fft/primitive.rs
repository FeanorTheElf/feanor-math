use std::ops::DerefMut;

use crate::integer::IntegerRingStore;
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::{ring::*, vector::*, mempool::*, divisibility::*, primitive_int::*};
use crate::algorithms;
use crate::algorithms::fft::*;

pub trait CycloFFTTable {
    
    type Ring: RingStore;

    fn len(&self) -> usize;
    fn root_of_unity_order(&self) -> usize;
    fn ring(&self) -> &Self::Ring;

    ///
    /// Computes a variant of the FFT. In particular, it computes the evaluation of the polynomial
    /// `sum_i v_i x^i` of degree `phi(n) - 1` at all primitive `n`-th roots of unity. Note that in contrast,
    /// the standard FFT computes the evaluation of a polynomial of degree `n - 1` at all (including non-primitive)
    /// `n`-th roots of unity.
    /// 
    /// As opposed to the standard FFT, there is no natural order of the `n`-th primitive roots of unity,
    /// and hence this implementation does not specify the output to have any particular order. However,
    /// it is required that `unordered_fft` and `unordered_inv_fft` are inverse.
    /// 
    fn unordered_fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>;
        
    fn unordered_inv_fft<V, S, N>(&self, values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>;
}

pub struct CycloFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    base_table: cooley_tuckey::FFTTableCooleyTuckey<R, M>,
    additional_twiddles: M::Object,
    inv_additional_twiddles: M::Object
}

impl<R> CycloFFTTableCooleyTuckey<R, AllocatingMemoryProvider>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_mem(ring, root_of_unity, log2_n, &AllocatingMemoryProvider)
    }
}

impl<R, M> CycloFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(ring: R, root_of_unity: El<R>, log2_n: usize, memory_provider: &M) -> Self {
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(algorithms::unity_root::is_prim_root_of_unity_pow2(&ring, &root_of_unity, log2_n));
        let additional_twiddles = memory_provider.get_new_init(1 << (log2_n - 1), |i| ring.pow(ring.clone_el(&root_of_unity), i));
        let inv_root_of_unity = ring.invert(&root_of_unity).unwrap();
        let inv_additional_twiddles: <M as MemoryProvider<<<R as RingStore>::Type as RingBase>::Element>>::Object = memory_provider.get_new_init(1 << (log2_n - 1), |i| ring.pow(ring.clone_el(&inv_root_of_unity), i));
        let root_of_unity_n = ring.pow(root_of_unity, 2);
        let base_table = cooley_tuckey::FFTTableCooleyTuckey::new_with_mem(ring, root_of_unity_n, log2_n - 1, memory_provider);
        return CycloFFTTableCooleyTuckey { base_table, additional_twiddles, inv_additional_twiddles }
    }
}

impl<R, M> CycloFFTTableCooleyTuckey<R, M>
    where R: ZnRingStore,
        R::Type: ZnRing,
        M: MemoryProvider<El<R>>,
        <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
{
    pub fn for_zn_with_mem(ring: R, log2_n: usize, memory_provider: &M) -> Option<Self> 
    {
        let root_of_unity = algorithms::unity_root::get_prim_root_of_unity_pow2(&ring, log2_n)?;
        Some(Self::new_with_mem(ring, root_of_unity, log2_n, memory_provider))
    }
}


impl<R, M> CycloFFTTable for CycloFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    type Ring = R;

    fn len(&self) -> usize {
        self.base_table.len()
    }

    fn root_of_unity_order(&self) -> usize {
        2 * self.len()
    }

    fn ring(&self) -> &Self::Ring {
        self.base_table.ring()
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.additional_twiddles.at(i), hom.raw_hom());
        }
        self.base_table.unordered_fft(values, ring, memory_provider);
    }

    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        self.base_table.unordered_inv_fft(&mut values, &ring, memory_provider);
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.inv_additional_twiddles.at(i), hom.raw_hom());
        }
    }
}

pub struct CycloFFTTableBluestein<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    base_table: bluestein::FFTTableBluestein<R, M>
}

impl<R, M> CycloFFTTableBluestein<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(ring: R, root_of_unity_2n: El<R>, root_of_unity_m: El<R>, n: usize, log2_m: usize, memory_provider: &M) -> Self {
        assert!(ring.is_commutative());
        assert!(algorithms::miller_rabin::is_prime(&StaticRing::<i64>::RING, &(n as i64), 8));
        assert!(algorithms::unity_root::is_prim_root_of_unity(&ring, &root_of_unity_2n, 2 * n));
        assert!(algorithms::unity_root::is_prim_root_of_unity_pow2(&ring, &root_of_unity_m, log2_m));
        let base_table = bluestein::FFTTableBluestein::new_with_mem(ring, root_of_unity_2n, root_of_unity_m, n, log2_m, memory_provider);
        return CycloFFTTableBluestein { base_table }
    }
}

impl<R, M> CycloFFTTableBluestein<R, M>
    where R: ZnRingStore, 
        R::Type: ZnRing, 
        <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>,
        M: MemoryProvider<El<R>>
{
    pub fn for_zn_with_mem(ring: R, n: usize, memory_provider: &M) -> Option<Self> {
        let ZZ = StaticRing::<i64>::RING;
        let log2_m = ZZ.abs_log2_ceil(&(2 * n as i64 + 1)).unwrap();
        let root_of_unity_2n = algorithms::unity_root::get_prim_root_of_unity(&ring, 2 * n)?;
        let root_of_unity_m = algorithms::unity_root::get_prim_root_of_unity_pow2(&ring, log2_m)?;
        Some(Self::new_with_mem(ring, root_of_unity_2n, root_of_unity_m, n, log2_m, memory_provider))
    }
}

impl<R, M> CycloFFTTable for CycloFFTTableBluestein<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    type Ring = R;

    fn len(&self) -> usize {
        self.base_table.len() - 1
    }

    fn root_of_unity_order(&self) -> usize {
        self.base_table.len()
    }

    fn ring(&self) -> &Self::Ring {
        self.base_table.ring()
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore,
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        // basically, we zero-pad to p values, and perform the standard fft
        assert_eq!(values.len(), self.len());
        assert_eq!(self.len() + 1, self.root_of_unity_order());
        let mut additional = [ring.zero()];
        let buffer = (&mut values).chain(&mut additional);
        self.base_table.unordered_fft(buffer, ring, memory_provider);
        let mut current = additional.into_iter().next().unwrap();
        for i in (0..self.len()).rev() {
            current = std::mem::replace(values.at_mut(i), current);
        }
    }

    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        // basically, we zero-pad to p values, and perform the standard fft
        assert_eq!(values.len(), self.len());
        assert_eq!(self.len() + 1, self.root_of_unity_order());
        
        let mut additional = [ring.zero()];
        let mut buffer = (&mut additional).chain(&mut values);
        self.base_table.unordered_inv_fft(&mut buffer, &ring, memory_provider);
        // now buffer contains some polynomial mod `X^p - 1`
        // we have to take it modulo `Phi_p = 1 + ... + X^(p - 1)`
        let last = ring.clone_el(buffer.at(self.len()));
        let mut current = additional.into_iter().next().unwrap();
        for i in 0..self.len() {
            current = std::mem::replace(values.at_mut(i), ring.sub_ref_snd(current, &last));
        }
    }
}

pub struct CycloFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: CycloFFTTable<Ring = R>,
        T2: CycloFFTTable<Ring = R>
{
    table1: T1,
    table2: T2
}

impl<R, T1, T2> CycloFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: CycloFFTTable<Ring = R>,
        T2: CycloFFTTable<Ring = R>
{
    pub fn new(table1: T1, table2: T2) -> Self {
        assert!(table1.ring().get_ring() == table2.ring().get_ring());
        assert!(algorithms::eea::signed_gcd(table1.root_of_unity_order() as i64, table2.root_of_unity_order() as i64, &StaticRing::<i64>::RING) == 1);
        return CycloFFTTableTensorFactorization { table1, table2 }
    }
}

impl<R, T1, T2> CycloFFTTable for CycloFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: CycloFFTTable<Ring = R>,
        T2: CycloFFTTable<Ring = R>
{
    type Ring = T1::Ring;

    fn len(&self) -> usize {
        self.table1.len() * self.table2.len()
    }

    fn root_of_unity_order(&self) -> usize {
        self.table1.root_of_unity_order() * self.table2.root_of_unity_order()
    }

    fn ring(&self) -> &Self::Ring {
        self.table1.ring()
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        assert_eq!(values.len(), self.len());
        for i in 0..self.table2.len() {
            let v = Subvector::new(&mut values).subvector((i * self.table1.len())..((i + 1) * self.table1.len()));
            self.table1.unordered_fft(v, &ring, memory_provider);
        }
        for i in 0..self.table1.len() {
            let mut v = Subvector::new(&mut values).subvector(i..).stride(self.table1.len());
            self.table2.unordered_fft(&mut v, &ring, memory_provider);
            
        }
    }

    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>> 
    {
        assert_eq!(values.len(), self.len());
        for i in 0..self.table1.len() {
            self.table2.unordered_inv_fft(Subvector::new(&mut values).subvector(i..).stride(self.table1.len()), &ring, memory_provider);
        }
        for i in 0..self.table2.len() {
            self.table1.unordered_inv_fft(Subvector::new(&mut values).subvector((i * self.table1.len())..((i + 1) * self.table1.len())), &ring, memory_provider);
        }
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;

#[test]
fn test_cyclotomic_multiplication_bluestein_fft() {
    let ring = Zn::<113>::RING;
    let primitive_fft = CycloFFTTableBluestein::for_zn_with_mem(&ring, 7, &AllocatingMemoryProvider).unwrap();
    let mut values = [ring.from_int(1), ring.from_int(2), ring.from_int(1), ring.from_int(3), ring.from_int(2), ring.from_int(1)];
    let original = values;
    primitive_fft.unordered_fft(&mut values, &ring, &AllocatingMemoryProvider);
    primitive_fft.unordered_inv_fft(&mut values, &ring, &AllocatingMemoryProvider);
    assert_eq!(values, original);
    
    let expected = [ring.from_int(-2), ring.from_int(-3), ring.from_int(-7), ring.from_int(-6), ring.zero(), ring.from_int(-1)]; 
    primitive_fft.unordered_fft(&mut values, &ring, &AllocatingMemoryProvider);
    for i in 0..values.len() {
        ring.square(values.at_mut(i));
    }
    primitive_fft.unordered_inv_fft(&mut values, &ring, &AllocatingMemoryProvider);
    assert_eq!(values, expected);
}

#[test]
fn test_cyclotomic_multiplication_factor_fft() {
    let ring = Zn::<97>::RING;
    let primitive_fft = CycloFFTTableTensorFactorization::new(
        CycloFFTTableBluestein::for_zn_with_mem(&ring, 3, &AllocatingMemoryProvider).unwrap(),
        CycloFFTTableCooleyTuckey::for_zn_with_mem(&ring, 2, &AllocatingMemoryProvider).unwrap(),
    );
    let mut values = [ring.from_int(1), ring.zero(), ring.zero(), ring.from_int(2)];
    let original = values;
    primitive_fft.unordered_fft(&mut values, &ring, &AllocatingMemoryProvider);
    primitive_fft.unordered_inv_fft(&mut values, &ring, &AllocatingMemoryProvider);
    assert_eq!(values, original);
    
    primitive_fft.unordered_fft(&mut values, &ring, &AllocatingMemoryProvider);
    for i in 0..values.len() {
        ring.square(values.at_mut(i));
    }
    primitive_fft.unordered_inv_fft(&mut values, &ring, &AllocatingMemoryProvider);
    let expected = [ring.from_int(-3), ring.zero(), ring.zero(), ring.from_int(4)];
    assert_eq!(values, expected);
}