use crate::{ring::*, vector::VectorViewMut};

pub struct FFTTableHarvey<R> 
    where R: RingWrapper
{
    ring: R,
    root_of_unity: El<R>,
    log2_n: usize
}

impl<R> FFTTableHarvey<R>
    where R: RingWrapper
{
    pub fn bitreverse_fft_inplace<V, S>(&self, mut values: V, ring: S)
        where V: VectorViewMut<El<S>>, S: RingWrapper, S::Type: CanonicalHom<R::Type>
    {
        assert!(values.len() == 1 << self.log2_n);
        for s in (0..self.log2_n).rev() {
            let m = 1 << s;
            let log2_group_size = self.log2_n - s;
            let m_th_root = ring.map_in(&self.ring, self.ring.pow(&self.root_of_unity, m));
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
                    let twiddled_entry = ring.mul_ref(values.at(global_index2), &m_th_root);
                    *values.at_mut(global_index2) = ring.sub_ref(values.at(global_index1), &twiddled_entry);
                    ring.add_assign(values.at_mut(global_index1), twiddled_entry);
                }
            }

        }
    }
}

#[test]
fn test_bitreverse_fft_inplace() {
    
}