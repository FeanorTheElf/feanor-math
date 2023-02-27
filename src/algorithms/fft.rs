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
    pub fn bitreverse_fft_inplace<V, S>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingWrapper, S::Type: CanonicalHom<R::Type>, El<S>: std::fmt::Display
    {
        assert!(values.len() == 1 << self.log2_n);
        for s in (0..self.log2_n).rev() {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            let m_th_root = ring.map_in(&self.ring, self.ring.pow(&self.inv_root_of_unity, m));
            println!("{}", m_th_root);
            for k in 0..(1 << s) {
                // 
                // we want to compute a bitreverse_fft_inplace for v_k, v_(k + m), v_(k + 2m), ..., v_(k + n - m);
                // call this sequence a
                //
                // we already have a bitreverse fft of v_k, v_(k + 2m), v_(k + 4m), ..., v_(k + n - 2m) 
                // and v_(k + m), v_(k + 3m), v_(k + 5m), ..., v_(k + n - m) in the corresponding entries;
                // call these sequences a1 and a2
                //
                // i goes through 0, ..., group_size/2 - 1
                // j := bitrev(i, group_size)
                //
                for j in (0..(1 << log2_group_size)).step_by(2) {
                    //
                    // we want to compute (a[i], a[i + group_size/2]) = (a1[i] + z a2[i], a1[i] - z a2[i])
                    //
                    // in bitreverse order, have
                    // local_index1 = bitrev(i, group_size) = 2 bitrev(i, group_size/2)
                    // local_index2 = bitrev(i + group_size/2, group_size) = 2 bitrev(i, group_size/2) + 1
                    let local_index1 = j;
                    let local_index2 = j + 1;
                    let global_index1 = local_index1 * m + k;
                    let global_index2 = local_index2 * m + k;
                    println!("{}, {}", global_index1, global_index2);
                    let twiddled_entry = ring.mul_ref(values.at(global_index2), &m_th_root);
                    *values.at_mut(global_index2) = ring.sub_ref(values.at(global_index1), &twiddled_entry);
                    ring.add_assign(values.at_mut(global_index1), twiddled_entry);
                }
            }
            println!("{}, {}, {}, {}", values.at(0), values.at(1), values.at(2), values.at(3));
        }
    }
}

#[cfg(test)]
use crate::rings::zn_small::Zn;
#[cfg(test)]
use crate::field::*;

#[test]
fn test_bitreverse_fft_inplace_small() {
    let ring = Zn::<5>::RING;
    let z = ring.from_z(2);
    let fft = FFTTableCooleyTuckey {
        ring: ring,
        log2_n: 2,
        root_of_unity: z,
        inv_root_of_unity: ring.div(&1, &z)
    };
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 4];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.bitreverse_fft_inplace(&mut values, ring);
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

    fft.bitreverse_fft_inplace(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}