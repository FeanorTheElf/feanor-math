use std::ops::DerefMut;

use crate::integer::IntegerRingStore;
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::{ring::*, vector::*, mempool::*, divisibility::*, primitive_int::*};
use crate::algorithms;
use crate::algorithms::fft::*;

pub trait PrimitiveFFTTable {
    
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

pub struct PrimitiveFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    base_table: cooley_tuckey::FFTTableCooleyTuckey<R, M>,
    additional_twiddles: M::Object,
    inv_additional_twiddles: M::Object
}

impl<R> PrimitiveFFTTableCooleyTuckey<R, AllocatingMemoryProvider>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_mem(ring, root_of_unity, log2_n, &AllocatingMemoryProvider)
    }
}

impl<R, M> PrimitiveFFTTableCooleyTuckey<R, M>
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
        return PrimitiveFFTTableCooleyTuckey { base_table, additional_twiddles, inv_additional_twiddles }
    }
}

impl<R, M> PrimitiveFFTTableCooleyTuckey<R, M>
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


impl<R, M> PrimitiveFFTTable for PrimitiveFFTTableCooleyTuckey<R, M>
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

pub struct PrimitiveFFTTableBluestein<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    base_table: bluestein::FFTTableBluestein<R, M>,
    additional_twiddles: M::Object,
    inv_additional_twiddles: M::Object
}

impl<R, M> PrimitiveFFTTableBluestein<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    pub fn new_with_mem(ring: R, root_of_unity_2n: El<R>, root_of_unity_m: El<R>, n: usize, log2_m: usize, memory_provider: &M) -> Self {
        assert!(ring.is_commutative());
        assert!(algorithms::miller_rabin::is_prime(&StaticRing::<i64>::RING, &(n as i64), 8));
        assert!(algorithms::unity_root::is_prim_root_of_unity(&ring, &root_of_unity_2n, 2 * n));
        assert!(algorithms::unity_root::is_prim_root_of_unity_pow2(&ring, &root_of_unity_m, log2_m));
        let inv_root_of_unity_n = ring.pow(ring.invert(&root_of_unity_2n).unwrap(), 2);
        let additional_twiddles = memory_provider.get_new_init(n, |i| ring.pow(ring.clone_el(&inv_root_of_unity_n), i));
        let inv_additional_twiddles = memory_provider.get_new_init(n, |i| ring.pow(ring.clone_el(&root_of_unity_2n), 2 * i));
        let base_table = bluestein::FFTTableBluestein::new_with_mem(ring, root_of_unity_2n, root_of_unity_m, n, log2_m, memory_provider);
        return PrimitiveFFTTableBluestein { base_table, additional_twiddles, inv_additional_twiddles }
    }
}

impl<R, M> PrimitiveFFTTableBluestein<R, M>
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

impl<R, M> PrimitiveFFTTable for PrimitiveFFTTableBluestein<R, M>
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
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        let mut buffer = memory_provider.get_new_init(self.base_table.len(), |i| 
            if i < self.len() { 
                let mut result = ring.clone_el(values.at(i));
                ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), &mut result, self.additional_twiddles.at(i), hom.raw_hom());
                result
            } else {
                ring.zero()
            }
        );
        self.base_table.unordered_fft(&mut buffer.deref_mut(), ring, memory_provider);
        for i in 0..self.len() {
            std::mem::swap(values.at_mut(i), buffer.at_mut(i));
        }
    }

    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>,
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        let mut buffer = memory_provider.get_new_init(self.base_table.len(), |i| 
            if i < self.len() { 
                ring.clone_el(values.at(i))
            } else {
                ring.zero()
            }
        );
        self.base_table.unordered_inv_fft(&mut buffer.deref_mut(), &ring, memory_provider);
        for i in 0..self.len() {
            std::mem::swap(values.at_mut(i), buffer.at_mut(i));
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.inv_additional_twiddles.at(i), hom.raw_hom());
        }
    }
}

pub struct PrimitiveFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: PrimitiveFFTTable<Ring = R>,
        T2: PrimitiveFFTTable<Ring = R>
{
    table1: T1,
    table2: T2
}

impl<R, T1, T2> PrimitiveFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: PrimitiveFFTTable<Ring = R>,
        T2: PrimitiveFFTTable<Ring = R>
{
    pub fn new(table1: T1, table2: T2) -> Self {
        assert!(table1.ring().get_ring() == table2.ring().get_ring());
        assert!(algorithms::eea::signed_gcd(table1.root_of_unity_order() as i64, table2.root_of_unity_order() as i64, &StaticRing::<i64>::RING) == 1);
        return PrimitiveFFTTableTensorFactorization { table1, table2 }
    }
}

impl<R, T1, T2> PrimitiveFFTTable for PrimitiveFFTTableTensorFactorization<R, T1, T2>
    where R: RingStore,
        T1: PrimitiveFFTTable<Ring = R>,
        T2: PrimitiveFFTTable<Ring = R>
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
            self.table1.unordered_fft(Subvector::new(&mut values).subvector((i * self.table1.len())..((i + 1) * self.table1.len())), &ring, memory_provider);
        }
        for i in 0..self.table1.len() {
            self.table2.unordered_fft(Subvector::new(&mut values).subvector(i..).stride(self.table1.len()), &ring, memory_provider);
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
fn test_primitive_tensor_fft() {
    let ring = Zn::<97>::RING;
    let primitive_fft = PrimitiveFFTTableTensorFactorization::new(
        PrimitiveFFTTableBluestein::for_zn_with_mem(&ring, 3, &AllocatingMemoryProvider).unwrap(),
        PrimitiveFFTTableCooleyTuckey::for_zn_with_mem(&ring, 2, &AllocatingMemoryProvider).unwrap(),
    );
    let mut values = [ring.from_int(1), ring.zero(), ring.zero(), ring.from_int(2)];
    primitive_fft.unordered_fft(&mut values, &ring, &AllocatingMemoryProvider);
    println!("{:?}", values);
    for i in 0..4 {
        ring.square(values.at_mut(i));
    }
    primitive_fft.unordered_inv_fft(&mut values, &ring, &AllocatingMemoryProvider);
    let expected = [ring.from_int(-3), ring.zero(), ring.zero(), ring.from_int(4)];

    assert_eq!(values, expected);
}