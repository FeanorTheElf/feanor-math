use crate::divisibility::DivisibilityRingWrapper;
use crate::vector::SwappableVectorViewMut;
use crate::{ring::*, vector::VectorViewMut};

pub struct FFTTableCooleyTuckey<R> 
    where R: RingWrapper
{
    ring: R,
    root_of_unity: El<R>,
    inv_root_of_unity: El<R>,
    log2_n: usize
}

pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - bits)
}

impl<R> FFTTableCooleyTuckey<R>
    where R: RingWrapper
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        assert!(log2_n > 0);
        assert!(ring.is_neg_one(&ring.pow(&root_of_unity, 1 << (log2_n - 1))));
        let inv_root_of_unity = ring.pow(&root_of_unity, (1 << log2_n) - 1);
        FFTTableCooleyTuckey { ring, root_of_unity, inv_root_of_unity, log2_n }
    }

    fn bitreverse_fft_inplace_base<V, S, const INV: bool>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingWrapper, S::Type: CanonicalHom<R::Type>
    {
        assert!(values.len() == 1 << self.log2_n);
        for s in (0..self.log2_n).rev() {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            let twiddle_root = if INV {
                ring.map_in(&self.ring, self.ring.pow(&self.root_of_unity, m))
            } else {
                ring.map_in(&self.ring, self.ring.pow(&self.inv_root_of_unity, m))
            };
            
            for k in 0..(1 << s) {
                let mut current_twiddle = ring.one();
                // 
                // we want to compute a bitreverse_fft_inplace for v_k, v_(k + m), v_(k + 2m), ..., v_(k + n - m);
                // call this sequence a
                //
                // we already have a bitreverse fft of v_k, v_(k + 2m), v_(k + 4m), ..., v_(k + n - 2m) 
                // and v_(k + m), v_(k + 3m), v_(k + 5m), ..., v_(k + n - m) in the corresponding entries;
                // call these sequences a1 and a2
                //
                for i in 0..(1 << (log2_group_size - 1)) {
                    let j = bitreverse(i, log2_group_size);
                    //
                    // we want to compute (a[i], a[i + group_size/2]) = (a1[i] + z a2[i], a1[i] - z^i a2[i])
                    //
                    // in bitreverse order, have
                    // local_index1 = bitrev(i, group_size) = 2 bitrev(i, group_size/2)
                    // local_index2 = bitrev(i + group_size/2, group_size) = 2 bitrev(i, group_size/2) + 1
                    //
                    let local_index1 = j;
                    let local_index2 = j + 1;
                    let global_index1 = local_index1 * m + k;
                    let global_index2 = local_index2 * m + k;
                    let twiddled_entry = ring.mul_ref(values.at(global_index2), &current_twiddle);
                    *values.at_mut(global_index2) = ring.sub_ref(values.at(global_index1), &twiddled_entry);
                    ring.add_assign(values.at_mut(global_index1), twiddled_entry);
                    ring.mul_assign_ref(&mut current_twiddle, &twiddle_root);
                }
            }
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
        self.bitreverse_fft_inplace_base::<V, &R, false>(values, &self.ring);
    }

    pub fn bitreverse_inv_fft_inplace<V>(&self, mut values: V)
        where V: VectorViewMut<El<R>>, R: DivisibilityRingWrapper
    {
        self.bitreverse_fft_inplace_base::<&mut V, &R, true>(&mut values, &self.ring);
        let scale = self.ring.checked_div(&self.ring.one(), &self.ring.from_z(1 << self.log2_n)).unwrap();
        for i in 0..values.len() {
            self.ring.mul_assign_ref(values.at_mut(i), &scale);
        }
    }
}

#[cfg(test)]
use crate::rings::zn_static::Zn;
#[cfg(test)]
use crate::field::*;

#[test]
fn test_bitreverse_fft_inplace_base() {
    let ring = Zn::<5>::RING;
    let z = ring.from_z(2);
    let fft = FFTTableCooleyTuckey {
        ring: ring,
        log2_n: 2,
        root_of_unity: ring.div(&1, &z),
        inv_root_of_unity: z
    };
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.bitreverse_fft_inplace_base::<_, _, false>(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace() {
    let ring = Zn::<17>::RING;
    let z = ring.from_z(3);
    let fft = FFTTableCooleyTuckey {
        ring: ring,
        log2_n: 4,
        root_of_unity: z,
        inv_root_of_unity: ring.div(&1, &z)
    };
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.bitreverse_fft_inplace(&mut values);
    assert_eq!(values, bitreverse_expected);
}