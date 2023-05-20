use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::integer::IntegerRingStore;
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::vector::SwappableVectorViewMut;
use crate::{ring::*, vector::VectorViewMut};

pub struct FFTTableCooleyTuckey<R> 
    where R: RingStore
{
    ring: R,
    root_of_unity: El<R>,
    inv_root_of_unity: El<R>,
    log2_n: usize,
    // stores the powers of root_of_unity in special bitreversed order
    root_of_unity_list: Vec<Vec<El<R>>>,
    // stores the powers of inv_root_of_unity in special bitreversed order
    inv_root_of_unity_list: Vec<Vec<El<R>>>
}

pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - bits)
}

impl<R> FFTTableCooleyTuckey<R>
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(ring.is_neg_one(&ring.pow(ring.clone_el(&root_of_unity), 1 << (log2_n - 1))));
        let root_of_unity_list = Self::create_root_of_unity_list(&ring, &root_of_unity, log2_n);
        let inv_root_of_unity = ring.pow(ring.clone_el(&root_of_unity), (1 << log2_n) - 1);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(&ring, &inv_root_of_unity, log2_n);
        FFTTableCooleyTuckey { ring, root_of_unity, inv_root_of_unity, log2_n, root_of_unity_list, inv_root_of_unity_list }
    }

    fn create_root_of_unity_list(ring: &R, root_of_unity: &El<R>, log2_n: usize) -> Vec<Vec<El<R>>> {
        let mut root_of_unity_list = Vec::new();
        for s in 0..log2_n {
            let m = 1 << s;
            let log2_group_size = log2_n - s;
            let twiddle_root = ring.pow(ring.clone_el(&root_of_unity), m);
            root_of_unity_list.push(Vec::new());
            for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                let current_twiddle = ring.pow(ring.clone_el(&twiddle_root), bitreverse(i_bitreverse, log2_group_size));
                root_of_unity_list.last_mut().unwrap().push(current_twiddle);
            }
        }
        return root_of_unity_list;
    }

    ///
    /// Returns `inv_root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    fn inv_root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.inv_root_of_unity_list[exp_2][bitreverse_exp / 2];
        debug_assert!(self.ring.eq_el(result, &self.ring.pow(self.ring.clone_el(&self.inv_root_of_unity), (1 << exp_2) * bitreverse(bitreverse_exp, self.log2_n - exp_2))));
        return result;
    }

    ///
    /// Returns `root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    fn root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.root_of_unity_list[exp_2][bitreverse_exp / 2];
        debug_assert!(self.ring.eq_el(result, &self.ring.pow(self.ring.clone_el(&self.root_of_unity), (1 << exp_2) * bitreverse(bitreverse_exp, self.log2_n - exp_2))));
        return result;
    }

    pub fn len(&self) -> usize {
        1 << self.log2_n
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn for_zn(ring: R, log2_n: usize) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing
    {
        assert!(log2_n > 0);
        assert!(ring.is_field());
        let ZZ = ring.integer_ring();
        let mut n = ZZ.one();
        ZZ.mul_pow_2(&mut n, log2_n);
        let order = ZZ.sub_ref_fst(ring.modulus(), ZZ.one());
        let power = ZZ.checked_div(&order, &n)?;
        
        let pow_n_half = |mut x: El<R>| {
            for _ in 1..log2_n {
                let x_copy = ring.clone_el(&x);
                ring.mul_assign(&mut x, x_copy);
            }
            return x;
        };

        let mut rng = oorandom::Rand64::new(ZZ.default_hash(ring.modulus()) as u128);
        let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
        while !ring.is_neg_one(&pow_n_half(ring.clone_el(&current))) {
            current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
        }
        return Some(Self::new(ring, current, log2_n));
    }

    ///
    /// Computes the bitreverse FFT of the given values, using the Cooley-Tuckey FFT algorithm.
    /// 
    /// This supports any given ring, as long as the precomputed values stored in the table are
    /// also contained in the new ring. The result is wrong however if the canonical homomorphism
    /// `R -> S` does not map the N-th root of unity to a primitive N-th root of unity.
    /// 
    /// Note that the FFT of a sequence `a_0, ..., a_(N - 1)` is defined as `Fa_k = sum_i a_i z^(-ik)`
    /// where `z` is an N-th root of unity. Since this is a bitreverse FFT, the output are not the `Fa_i`
    /// but instead the numbers `Fa_(bitrev(k))`, i.e. the same values in another order.
    /// 
    #[inline(never)]
    pub fn bitreverse_fft_inplace_base<V, S>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingStore, S::Type: CanonicalHom<R::Type>
    {
        assert!(values.len() == 1 << self.log2_n);
        let hom = ring.get_ring().has_canonical_hom(self.ring.get_ring()).unwrap();
        // check if the canonical hom `R -> S` maps `self.root_of_unity` to a primitive N-th root of unity
        debug_assert!(ring.is_neg_one(&ring.pow(ring.get_ring().map_in_ref(self.ring.get_ring(), &self.root_of_unity, &hom), 1 << (self.log2_n - 1))));

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

                    let current_twiddle = ring.get_ring().map_in_ref(self.ring.get_ring(), self.inv_root_of_unity_pow(s, i_bitreverse), &hom);

                    // `(values_i1, values_i2) = (values_i1 + twiddle * values_i2, values_i1 - twiddle * values_i2)`
                    ring.mul_assign(values.at_mut(index2), current_twiddle);
                    let new_a = ring.add_ref(values.at(index1), values.at(index2));
                    let a = std::mem::replace(values.at_mut(index1), new_a);
                    ring.sub_self_assign(values.at_mut(index2), a);
                }
            }
        }
    }

    ///
    /// this is exactly `bitreverse_fft_inplace_base()` with all operations reversed
    /// 
    #[inline(never)]
    pub fn bitreverse_inv_fft_inplace_base<V, S>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingStore, S::Type: CanonicalHom<R::Type>
    {
        assert!(values.len() == 1 << self.log2_n);
        let hom = ring.get_ring().has_canonical_hom(self.ring.get_ring()).unwrap();
        debug_assert!(ring.is_neg_one(&ring.pow(ring.get_ring().map_in_ref(self.ring.get_ring(), &self.root_of_unity, &hom), 1 << (self.log2_n - 1))));

        for s in 0..self.log2_n {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            
            // these loops are independent, so we do not have to reverse them
            // this improves access to the stored roots of unity
            for k in 0..m {
                for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                    let index1 = i_bitreverse * m + k;
                    let index2 = (i_bitreverse + 1) * m + k;

                    let current_twiddle = ring.get_ring().map_in_ref(self.ring.get_ring(), self.root_of_unity_pow(s, i_bitreverse), &hom);

                    let new_a = ring.add_ref(values.at(index1), values.at(index2));
                    let a = std::mem::replace(values.at_mut(index1), new_a);
                    ring.sub_self_assign(values.at_mut(index2), a);
                    ring.mul_assign(values.at_mut(index2), current_twiddle);
                }
            }
        }

        // finally, scale by 1/n
        let scale = ring.coerce(&self.ring, self.ring.checked_div(&self.ring.one(), &self.ring.from_int(1 << self.log2_n)).unwrap());
        for i in 0..values.len() {
            ring.mul_assign_ref(values.at_mut(i), &scale);
        }
    }

    pub fn bitreverse_permute_inplace<V>(&self, mut values: V) 
        where V: SwappableVectorViewMut<El<R>>
    {
        assert!(values.len() == 1 << self.log2_n);
        for i in 0..(1 << self.log2_n) {
            if bitreverse(i, self.log2_n) < i {
                values.swap(i, bitreverse(i, self.log2_n));
            }
        }
    }
    
    pub fn bitreverse_fft_inplace<V>(&self, values: V)
        where V: VectorViewMut<El<R>>
    {
        self.bitreverse_fft_inplace_base::<V, &R>(values, &self.ring);
    }

    pub fn bitreverse_inv_fft_inplace<V>(&self, mut values: V)
        where V: VectorViewMut<El<R>>
    {
        self.bitreverse_inv_fft_inplace_base::<&mut V, &R>(&mut values, &self.ring);
    }

    pub fn fft_inplace<V>(&self, mut values: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        self.bitreverse_fft_inplace(&mut values);
        self.bitreverse_permute_inplace(&mut values);
    }

    pub fn inv_fft_inplace<V>(&self, mut values: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        self.bitreverse_permute_inplace(&mut values);
        self.bitreverse_fft_inplace(&mut values);
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::rings::zn::zn_barett;
#[cfg(test)]
use crate::rings::zn::zn_42;
#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::field::*;

#[test]
fn test_bitreverse_fft_inplace_base() {
    let ring = Zn::<5>::RING;
    let z = ring.from_int(2);
    let fft = FFTTableCooleyTuckey::new(ring, ring.div(&1, &z), 2);
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.bitreverse_fft_inplace_base(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace() {
    let ring = Zn::<17>::RING;
    let z = ring.from_int(3);
    let fft = FFTTableCooleyTuckey::new(ring, z, 4);
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.bitreverse_fft_inplace(&mut values);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_inv_fft_inplace() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 4).unwrap();
    let values: [u64; 16] = [1, 2, 3, 2, 1, 0, 17 - 1, 17 - 2, 17 - 1, 0, 1, 2, 3, 4, 5, 6];
    let mut work = values;
    fft.bitreverse_fft_inplace(&mut work);
    fft.bitreverse_inv_fft_inplace(&mut work);
    assert_eq!(&work, &values);
}

#[test]
fn test_for_zn() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
    assert!(ring.is_neg_one(&ring.pow(fft.inv_root_of_unity, 8)));

    let ring = Zn::<97>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
    assert!(ring.is_neg_one(&ring.pow(fft.inv_root_of_unity, 8)));
}

#[cfg(test)]
fn run_fft_bench_round<R>(ring: R, fft: &FFTTableCooleyTuckey<R>, data: &Vec<El<R>>, copy: &mut Vec<El<R>>)
    where R: ZnRingStore, R::Type: ZnRing
{
    copy.clear();
    copy.extend(data.iter().map(|x| ring.clone_el(x)));
    fft.bitreverse_fft_inplace(&mut copy[..]);
    fft.bitreverse_inv_fft_inplace(&mut copy[..]);
    assert!(ring.eq_el(&copy[0], &data[0]));
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
fn bench_fft_lazy(bencher: &mut test::Bencher) {
    let ring = zn_42::Zn::new(1073872897);
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 15).unwrap();
    let data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << 15);
    bencher.iter(|| {
        run_fft_bench_round(&ring, &fft, &data, &mut copy)
    });
}