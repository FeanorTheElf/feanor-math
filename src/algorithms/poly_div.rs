use crate::{ring::*, vector::{VectorViewMut, VectorView}, mempool::*};

fn effective_length<R, V>(poly: V, ring: R) -> usize
    where R: RingStore, V: VectorView<El<R>>
{
    for i in (0..poly.len()).rev() {
        if !ring.is_zero(poly.at(i)) {
            return i + 1;
        }
    }
    return 0;
}

pub fn left_poly_div<R, V, W, M, F>(mut lhs: V, rhs: W, ring: R, mut left_div_lc: F, memory_provider: &M) -> Option<M::Object>
    where R: RingStore, 
        V: VectorViewMut<El<R>>, 
        W: VectorView<El<R>>,
        F: FnMut(El<R>) -> Option<El<R>>,
        M: MemoryProvider<El<R>>
{
    let rhs_len = effective_length(&rhs, &ring);
    assert!(rhs_len > 0);
    let lhs_len = effective_length(&lhs, &ring);
    if lhs_len < rhs_len {
        return Some(memory_provider.get_new_init(0, |_| ring.zero()));
    }
    let mut result = memory_provider.get_new_init(lhs_len + 1 - rhs_len, |_| ring.zero());
    for i in (0..result.len()).rev() {
        *result.at_mut(i) = left_div_lc(std::mem::replace(lhs.at_mut(i + rhs_len - 1), ring.zero()))?;
        for j in 0..(rhs_len - 1) {
            ring.sub_assign(lhs.at_mut(i + j), ring.mul_ref(rhs.at(j), result.at(i)));
        }
    }
    return Some(result);
}

#[cfg(test)]
use crate::primitive_int::*;

#[test]
pub fn test_poly_div() {
    let ring = StaticRing::<i64>::RING;
    let mut values = [0, 0, 0, 0, 1];
    let rhs = [1, 2, 1];
    let actual_quo = left_poly_div(&mut values, rhs, ring, |x| Some(x), &AllocatingMemoryProvider).unwrap();
    let expected_rem = [-3, -4, 0, 0, 0];
    let expected_quo = [3, -2, 1];
    assert_eq!(&expected_quo[..], &actual_quo[..]);
    assert_eq!(expected_rem, values);
    
    let ring = StaticRing::<i64>::RING;
    let mut values = [1, 2, 0, -3, 2, 1, 2, 2, 1, 8, -5, 0, 0, 0, 0, 1];
    let rhs = [-1, 1];
    left_poly_div(&mut values, rhs, ring, |x| Some(x), &AllocatingMemoryProvider).unwrap();
    let expected_rem = [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(expected_rem, values);
}