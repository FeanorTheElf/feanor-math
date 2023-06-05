use crate::{ring::*, vector::*, mempool::{MemoryProvider, AllocatingMemoryProvider}, divisibility::{DivisibilityRingStore, DivisibilityRing}, algorithms::fft::FFTTable};

use super::cooley_tuckey::FFTTableCooleyTuckey;

pub trait PrimitiveFFTTable {
    
    type Ring: RingStore;

    fn len(&self) -> usize;
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
    fn unordered_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>>;
        
    fn unordered_inv_fft<V, S>(&self, values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>>;
}

pub struct PrimitiveFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    base_table: FFTTableCooleyTuckey<R, M>,
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
        assert!(ring.is_neg_one(&ring.pow(ring.clone_el(&root_of_unity), 1 << log2_n)));
        let additional_twiddles = memory_provider.get_new_init(1 << log2_n, |i| ring.pow(ring.clone_el(&root_of_unity), i));
        let inv_root_of_unity = ring.invert(&root_of_unity).unwrap();
        let inv_additional_twiddles = memory_provider.get_new_init(1 << log2_n, |i| ring.pow(ring.clone_el(&inv_root_of_unity), i));
        let root_of_unity_n = ring.pow(root_of_unity, 2);
        let base_table = FFTTableCooleyTuckey::new_with_mem(ring, root_of_unity_n, log2_n, memory_provider);
        return PrimitiveFFTTableCooleyTuckey { base_table, additional_twiddles, inv_additional_twiddles }
    }
}

impl<R, M> PrimitiveFFTTable for PrimitiveFFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, R::Type: DivisibilityRing, M: MemoryProvider<El<R>>
{
    type Ring = R;

    fn len(&self) -> usize {
        self.base_table.len()
    }

    fn ring(&self) -> &Self::Ring {
        self.base_table.ring()
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>> 
    {
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.additional_twiddles.at(i), hom.raw_hom());
        }
        self.base_table.unordered_fft(values, ring);
    }

    fn unordered_inv_fft<V, S>(&self, mut values: V, ring: S)
        where S: RingStore, S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, V: VectorViewMut<El<S>>
    {
        assert_eq!(values.len(), self.len());
        let hom = ring.can_hom(self.ring()).unwrap();
        self.base_table.unordered_inv_fft(&mut values, &ring);
        for i in 0..self.len() {
            ring.get_ring().mul_assign_map_in_ref(self.ring().get_ring(), values.at_mut(i), self.inv_additional_twiddles.at(i), hom.raw_hom());
        }
    }
}
