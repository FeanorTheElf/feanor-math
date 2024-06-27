use crate::field::Field;
use crate::integer::BigIntRing;
use crate::ring::*;
use crate::primitive_int::*;
use crate::rings::finite::*;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;

use super::int_factor::factor;
use super::int_factor::is_prime_power;

#[stability::unstable(feature = "enable")]
pub fn is_prim_root_of_unity_pow2<R: RingStore>(ring: R, el: &El<R>, log2_n: usize) -> bool {
    if log2_n == 0 {
        return ring.is_one(el);
    }
    ring.is_neg_one(&ring.pow(ring.clone_el(&el), 1 << (log2_n - 1)))
}

#[stability::unstable(feature = "enable")]
pub fn is_root_of_unity<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    assert!(n >= 1);
    ring.is_one(&ring.pow(ring.clone_el(&el), n))
}

#[stability::unstable(feature = "enable")]
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

#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity<R>(ring: R, n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field
{
    const ZZ: BigIntRing = BigIntRing::RING;
    let (p, e) = is_prime_power(&ZZ, &ring.size(&ZZ).unwrap()).unwrap();
    let order = ZZ.mul(ZZ.sub_ref_fst(&p, ZZ.one()), ZZ.pow(p, e - 1));
    let power = ZZ.checked_div(&order, &ZZ.coerce(&StaticRing::<i64>::RING, n as i64))?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&ring.size(&ZZ).unwrap()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    while !is_prim_root_of_unity(&ring, &current, n) {
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    assert!(is_prim_root_of_unity(&ring, &current, n));
    return Some(current);
}

#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity_pow2<R>(ring: R, log2_n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field
{
    const ZZ: BigIntRing = BigIntRing::RING;
    let (p, e) = is_prime_power(&ZZ, &ring.size(&ZZ).unwrap()).unwrap();
    let order = ZZ.mul(ZZ.sub_ref_fst(&p, ZZ.one()), ZZ.pow(p, e - 1));
    let power = ZZ.checked_div(&order, &ZZ.power_of_two(log2_n))?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&ring.size(&ZZ).unwrap()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    while !is_prim_root_of_unity_pow2(&ring, &current, log2_n) {
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    assert!(is_prim_root_of_unity_pow2(&ring, &current, log2_n));
    return Some(current);
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::homomorphism::*;

#[test]
fn test_is_prim_root_of_unity() {
    let ring = Zn::<17>::RING;
    assert!(is_prim_root_of_unity_pow2(ring, &ring.int_hom().map(2), 3));
    assert!(!is_prim_root_of_unity_pow2(ring, &ring.int_hom().map(2), 4));
    assert!(is_prim_root_of_unity_pow2(ring, &ring.int_hom().map(3), 4));

    let ring = Zn::<101>::RING;
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(36), 5));
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(3), 100));
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(5), 25));
    assert!(!is_prim_root_of_unity(&ring, &ring.int_hom().map(5), 50));
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(6), 10));
    assert!(!is_prim_root_of_unity(&ring, &ring.int_hom().map(6), 50));
}