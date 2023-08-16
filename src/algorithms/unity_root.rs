use crate::ring::*;
use crate::primitive_int::{StaticRing, StaticRingBase};
use crate::rings::zn::{ZnRing, ZnRingStore};
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;

use super::int_factor::factor;

pub fn is_prim_root_of_unity_pow2<R: RingStore>(ring: R, el: &El<R>, log2_n: usize) -> bool {
    if log2_n == 0 {
        return ring.is_one(el);
    }
    ring.is_neg_one(&ring.pow(ring.clone_el(&el), 1 << (log2_n - 1)))
}

pub fn is_root_of_unity<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    assert!(n >= 1);
    ring.is_one(&ring.pow(ring.clone_el(&el), n))
}

pub fn is_prim_root_of_unity<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    assert!(n > 1);
    if !is_root_of_unity(&ring, el, n) {
        return false;
    }
    for (p, _) in factor(&StaticRing::<i128>::RING, n as i128) {
        if is_root_of_unity(&ring, el, n / p as usize) {
            return false;
        }
    }
    return true;
}

pub fn get_prim_root_of_unity<R>(ring: R, n: usize) -> Option<El<R>>
    where R: ZnRingStore, R::Type: ZnRing, <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
{
    assert!(ring.is_field());
    let ZZ = ring.integer_ring();
    let order = ZZ.sub_ref_fst(ring.modulus(), ZZ.one());
    let power = ZZ.checked_div(&order, &ZZ.coerce(&StaticRing::<i64>::RING, n as i64))?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(ring.modulus()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    while !is_prim_root_of_unity(&ring, &current, n) {
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    assert!(is_prim_root_of_unity(&ring, &current, n));
    return Some(current);
}

pub fn get_prim_root_of_unity_pow2<R>(ring: R, log2_n: usize) -> Option<El<R>>
    where R: ZnRingStore, R::Type: ZnRing, <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
{
    assert!(ring.is_field());
    let ZZ = ring.integer_ring();
    let order = ZZ.sub_ref_fst(ring.modulus(), ZZ.one());
    let power = ZZ.checked_div(&order, &ZZ.power_of_two(log2_n))?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(ring.modulus()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    while !is_prim_root_of_unity_pow2(&ring, &current, log2_n) {
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    assert!(is_prim_root_of_unity_pow2(&ring, &current, log2_n));
    return Some(current);
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_is_prim_root_of_unity() {
    let ring = Zn::<17>::RING;
    assert!(is_prim_root_of_unity_pow2(ring, &ring.from_int(2), 3));
    assert!(!is_prim_root_of_unity_pow2(ring, &ring.from_int(2), 4));
    assert!(is_prim_root_of_unity_pow2(ring, &ring.from_int(3), 4));

    let ring = Zn::<101>::RING;
    assert!(is_prim_root_of_unity(&ring, &ring.from_int(36), 5));
    assert!(is_prim_root_of_unity(&ring, &ring.from_int(3), 100));
    assert!(is_prim_root_of_unity(&ring, &ring.from_int(5), 25));
    assert!(!is_prim_root_of_unity(&ring, &ring.from_int(5), 50));
    assert!(is_prim_root_of_unity(&ring, &ring.from_int(6), 10));
    assert!(!is_prim_root_of_unity(&ring, &ring.from_int(6), 50));
}