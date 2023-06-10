use crate::algorithms::unity_root::is_prim_root_of_unity_pow2;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::primitive_int::*;
use crate::mempool::{MemoryProvider, AllocatingMemoryProvider};
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::vector::SwappableVectorViewMut;
use crate::ring::*;
use crate::vector::VectorViewMut;
use crate::algorithms::fft::*;
use crate::algorithms;
use crate::rings::float_complex::*;
use super::complex_fft::*;

pub struct FFTTableCooleyTuckey<R, M: MemoryProvider<El<R>> = AllocatingMemoryProvider> 
    where R: RingStore
{
    ring: R,
    root_of_unity: El<R>,
    log2_n: usize,
    // stores the powers of root_of_unity in special bitreversed order
    root_of_unity_list: M::Object,
    // stores the powers of inv_root_of_unity in special bitreversed order
    inv_root_of_unity_list: M::Object
}

pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - bits)
}

impl<R> FFTTableCooleyTuckey<R>
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_mem(ring, root_of_unity, log2_n, &AllocatingMemoryProvider)
    }
    
    pub fn for_zn(ring: R, log2_n: usize) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        Self::for_zn_with_mem(ring, log2_n, &AllocatingMemoryProvider)
    }
}

impl<R, M: MemoryProvider<El<R>>> FFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    pub fn new_with_mem(ring: R, root_of_unity: El<R>, log2_n: usize, memory_provider: &M) -> Self {
        let mut root_of_unity_pow = |i: i64| if i >= 0 {
            ring.pow(ring.clone_el(&root_of_unity), i as usize)
        } else {
            ring.invert(&ring.pow(ring.clone_el(&root_of_unity), (-i) as usize)).unwrap()
        };
        // cannot call new_with_mem_and_pows() because of borrowing conflict
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        let root_of_unity_list = Self::create_root_of_unity_list(&ring, &mut root_of_unity_pow, log2_n, memory_provider);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(&ring, |i| root_of_unity_pow(-i), log2_n, memory_provider);
        let root_of_unity = root_of_unity_pow(1);
        FFTTableCooleyTuckey { ring, root_of_unity, log2_n, root_of_unity_list, inv_root_of_unity_list }
    }

    pub fn new_with_mem_and_pows<F>(ring: R, mut root_of_unity_pow: F, log2_n: usize, memory_provider: &M) -> Self 
        where F: FnMut(i64) -> El<R>
    {
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        let root_of_unity_list = Self::create_root_of_unity_list(&ring, &mut root_of_unity_pow, log2_n, memory_provider);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(&ring, |i| root_of_unity_pow(-i), log2_n, memory_provider);
        let root_of_unity = root_of_unity_pow(1);
        FFTTableCooleyTuckey { ring, root_of_unity, log2_n, root_of_unity_list, inv_root_of_unity_list }
    }

    fn create_root_of_unity_list<F>(ring: &R, mut root_of_unity_pow: F, log2_n: usize, memory_provider: &M) -> M::Object
        where F: FnMut(i64) -> El<R>
    {
        // in fact, we could choose this to have only length `(1 << log2_n) - 1`, but a power of two length is probably faster
        let mut root_of_unity_list = memory_provider.get_new_init(1 << log2_n, |_| ring.zero());
        let mut index = 0;
        for s in 0..log2_n {
            let m = 1 << s;
            let log2_group_size = log2_n - s;
            let twiddle_root = root_of_unity_pow(m);
            for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                let current_twiddle = ring.pow(ring.clone_el(&twiddle_root), bitreverse(i_bitreverse, log2_group_size));
                root_of_unity_list[index] = current_twiddle;
                index += 1;
            }
        }
        assert_eq!(index, (1 << log2_n) - 1);
        return root_of_unity_list;
    }

    ///
    /// Returns `inv_root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    pub(super) fn inv_root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.inv_root_of_unity_list[(1 << self.log2_n) - (1 << (self.log2_n - exp_2)) + (bitreverse_exp / 2)];
        return result;
    }

    ///
    /// Returns `root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    pub(super) fn root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.root_of_unity_list[(1 << self.log2_n) - (1 << (self.log2_n - exp_2)) + (bitreverse_exp / 2)];
        return result;
    }

    pub fn len(&self) -> usize {
        1 << self.log2_n
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn for_zn_with_mem(ring: R, log2_n: usize, memory_provider: &M) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        let root_of_unity = algorithms::unity_root::get_prim_root_of_unity_pow2(&ring, log2_n)?;
        Some(Self::new_with_mem(ring, root_of_unity, log2_n, memory_provider))
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

impl<R, M: MemoryProvider<El<R>>> FFTTable for FFTTableCooleyTuckey<R, M> 
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    type Ring = R;
    
    fn len(&self) -> usize {
        1 << self.log2_n
    }

    fn ring(&self) -> &R {
        &self.ring
    }

    fn root_of_unity(&self) -> &El<R> {
        &self.root_of_unity
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        self.unordered_fft(&mut values, ring, memory_provider);
        self.bitreverse_permute_inplace(&mut values);
    }
        
    fn inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        self.bitreverse_permute_inplace(&mut values);
        self.unordered_inv_fft(&mut values, ring, memory_provider);
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, _: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        assert!(values.len() == (1 << self.log2_n));
        let hom = ring.can_hom(&self.ring).unwrap();
        // check if the canonical hom `R -> S` maps `self.root_of_unity` to a primitive N-th root of unity
        debug_assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &hom.map_ref(&self.root_of_unity), self.log2_n));

        for s in (0..self.log2_n).rev() {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            
            for k in 0..m {
                // 
                // we want to compute a bitreverse_fft_inplace for `v_k, v_(k + m), v_(k + 2m), ..., v_(k + n - m)`;
                // call this sequence a
                //
                // we already have a bitreverse fft of `v_k, v_(k + 2m), v_(k + 4m), ..., v_(k + n - 2m) `
                // and `v_(k + m), v_(k + 3m), v_(k + 5m), ..., v_(k + n - m)` in the corresponding entries;
                // call these sequences a1 and a2
                //
                // Note that a1_i is stored in `(k + 2m * bitrev(i, n/m))` and a2_i in `(k + m + 2m * bitrev(i, n/m))`;
                // We want to store a_i in `(k + m + m * bitrev(i, 2n/m))`
                //
                for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                    //
                    // we want to compute `(a_i, a_(i + group_size/2)) = (a1_i + z^i a2_i, a1_i - z^i a2_i)`
                    //
                    // in bitreverse order, have
                    // `i_bitreverse     = bitrev(i, group_size) = 2 bitrev(i, group_size/2)` and
                    // `i_bitreverse + 1 = bitrev(i + group_size/2, group_size) = 2 bitrev(i, group_size/2) + 1`
                    //
                    let index1 = i_bitreverse * m + k;
                    let index2 = (i_bitreverse + 1) * m + k;

                    let current_twiddle = self.inv_root_of_unity_pow(s, i_bitreverse);

                    // `(values_i1, values_i2) = (values_i1 + twiddle * values_i2, values_i1 - twiddle * values_i2)`
                    ring.get_ring().mul_assign_map_in_ref(self.ring.get_ring(), values.at_mut(index2), current_twiddle, hom.raw_hom());
                    let new_a = ring.add_ref(values.at(index1), values.at(index2));
                    let a = std::mem::replace(values.at_mut(index1), new_a);
                    ring.sub_self_assign(values.at_mut(index2), a);
                }
            }
        }
    }
        
    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, _: &N)
        where S: RingStore,
            S::Type: CanonicalHom<R::Type>,
            V: VectorViewMut<El<S>>
    {
        // this is exactly `bitreverse_fft_inplace_base()` with all operations reversed
        assert!(values.len() == 1 << self.log2_n);
        let hom = ring.can_hom(&self.ring).unwrap();
        debug_assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &hom.map_ref(&self.root_of_unity), self.log2_n));

        for s in 0..self.log2_n {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            
            // these loops are independent, so we do not have to reverse them
            // this improves access to the stored roots of unity
            for k in 0..m {
                for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                    let index1 = i_bitreverse * m + k;
                    let index2 = (i_bitreverse + 1) * m + k;

                    let current_twiddle = self.root_of_unity_pow(s, i_bitreverse);

                    let new_a = ring.add_ref(values.at(index1), values.at(index2));
                    let a = std::mem::replace(values.at_mut(index1), new_a);
                    ring.sub_self_assign(values.at_mut(index2), a);
                    ring.get_ring().mul_assign_map_in_ref(self.ring.get_ring(), values.at_mut(index2), current_twiddle, hom.raw_hom());
                }
            }
        }

        // finally, scale by 1/n
        let scale = ring.coerce(&self.ring, self.ring.checked_div(&self.ring.one(), &self.ring.from_int(1 << self.log2_n)).unwrap());
        for i in 0..values.len() {
            ring.mul_assign_ref(values.at_mut(i), &scale);
        }
    }
}

impl<R: RingStore<Type = Complex64>> ErrorEstimate for Complex64FFTTable<FFTTableCooleyTuckey<R>> {
    
    fn expected_absolute_error(&self, input_bound: f64) -> f64 {
        unimplemented!()
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::rings::zn::zn_barett;
#[cfg(test)]
use crate::rings::zn::zn_42;
#[cfg(test)]
use crate::field::*;

#[test]
fn test_bitreverse_fft_inplace_basic() {
    let ring = Zn::<5>::RING;
    let z = ring.from_int(2);
    let fft = FFTTableCooleyTuckey::new(ring, ring.div(&1, &z), 2);
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.unordered_fft(&mut values, ring, &AllocatingMemoryProvider);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace_advanced() {
    let ring = Zn::<17>::RING;
    let z = ring.from_int(3);
    let fft = FFTTableCooleyTuckey::new(ring, z, 4);
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.unordered_fft(&mut values, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_inv_fft_inplace() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 4).unwrap();
    let values: [u64; 16] = [1, 2, 3, 2, 1, 0, 17 - 1, 17 - 2, 17 - 1, 0, 1, 2, 3, 4, 5, 6];
    let mut work = values;
    fft.unordered_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    fft.unordered_inv_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(&work, &values);
}

#[test]
fn test_for_zn() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));

    let ring = Zn::<97>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
}

#[cfg(test)]
fn run_fft_bench_round<R, S>(ring: S, fft: &FFTTableCooleyTuckey<R>, data: &Vec<El<S>>, copy: &mut Vec<El<S>>)
    where R: ZnRingStore, R::Type: ZnRing, S: ZnRingStore, S::Type: ZnRing + CanonicalHom<R::Type>
{
    copy.clear();
    copy.extend(data.iter().map(|x| ring.clone_el(x)));
    fft.unordered_fft(&mut copy[..], &ring, &AllocatingMemoryProvider);
    fft.unordered_inv_fft(&mut copy[..], &ring, &AllocatingMemoryProvider);
    assert_el_eq!(&ring, &copy[0], &data[0]);
}

#[bench]
fn bench_fft(bencher: &mut test::Bencher) {
    let ring = zn_barett::Zn::new(StaticRing::<i128>::RING, 1073872897);
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 15).unwrap();
    let data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << 15);
    bencher.iter(|| {
        run_fft_bench_round(&ring, &fft, &data, &mut copy)
    });
}

#[bench]
fn bench_fft_optimized(bencher: &mut test::Bencher) {
    let ring = zn_42::Zn::new(1073872897);
    let fastmul_ring = zn_42::ZnFastmul::new(ring);
    let fft = FFTTableCooleyTuckey::for_zn(&fastmul_ring, 15).unwrap();
    let data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << 15);
    bencher.iter(|| {
        run_fft_bench_round(&ring, &fft, &data, &mut copy)
    });
}