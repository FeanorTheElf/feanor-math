use crate::field::Field;
use crate::integer::int_cast;
use crate::integer::BigIntRing;
use crate::integer::IntegerRing;
use crate::ring::*;
use crate::primitive_int::*;
use crate::rings::finite::*;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::ordered::OrderedRingStore;

use super::int_factor::factor;

#[stability::unstable(feature = "enable")]
pub fn is_prim_root_of_unity_pow2<R: RingStore>(ring: R, el: &El<R>, log2_n: usize) -> bool {
    if log2_n == 0 {
        return ring.is_one(el);
    }
    ring.is_neg_one(&ring.pow(ring.clone_el(&el), 1 << (log2_n - 1)))
}

#[stability::unstable(feature = "enable")]
pub fn is_root_of_unity<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    is_root_of_unity_gen(ring, el, &n.try_into().unwrap(), StaticRing::<i64>::RING)
}

#[stability::unstable(feature = "enable")]
pub fn is_root_of_unity_gen<R: RingStore, I: RingStore>(ring: R, el: &El<R>, n: &El<I>, ZZ: I) -> bool
    where I::Type: IntegerRing
{
    assert!(ZZ.is_pos(n));
    ring.is_one(&ring.pow_gen(ring.clone_el(&el), n, ZZ))
}

#[stability::unstable(feature = "enable")]
pub fn is_prim_root_of_unity<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    is_prim_root_of_unity_gen(ring, el, &n.try_into().unwrap(), StaticRing::<i64>::RING)
}

#[stability::unstable(feature = "enable")]
pub fn is_prim_root_of_unity_gen<R: RingStore, I>(ring: R, el: &El<R>, n: &El<I>, ZZ: I) -> bool
    where I: RingStore + Copy,
        I::Type: IntegerRing
{
    if !is_root_of_unity_gen(&ring, el, n, ZZ) {
        return false;
    }
    for (p, _) in factor(&ZZ, ZZ.clone_el(n)) {
        if is_root_of_unity_gen(&ring, el, &ZZ.checked_div(n, &p).unwrap(), ZZ) {
            return false;
        }
    }
    return true;
}

#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity_gen<R, I>(ring: R, n: &El<I>, ZZ: I) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field,
        I: RingStore + Copy,
        I::Type: IntegerRing
{
    let order = ZZ.sub(ring.size(&ZZ).unwrap(), ZZ.one());
    let power = ZZ.checked_div(&order, n)?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&ring.size(&ZZ).unwrap()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    while !is_prim_root_of_unity_gen(&ring, &current, n, ZZ) {
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    debug_assert!(is_prim_root_of_unity_gen(&ring, &current, n, ZZ));
    return Some(current);
}

///
/// Returns a primitive `n`-th root of unity in the given finite field,
/// or `None`, if the order of the multiplicative group of the field is
/// not divisible by `n`.
/// 
pub fn get_prim_root_of_unity<R>(ring: R, n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field
{
    get_prim_root_of_unity_gen(ring, &int_cast(n.try_into().unwrap(), BigIntRing::RING, StaticRing::<i64>::RING), BigIntRing::RING)
}

///
/// Returns a primitive `2^log2_n`-th root of unity in the given finite field,
/// or `None`, if the order of the multiplicative group of the field is
/// not divisible by `2^log2_n`.
/// 
pub fn get_prim_root_of_unity_pow2<R>(ring: R, log2_n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field
{
    const ZZ: BigIntRing = BigIntRing::RING;
    let order = ZZ.sub(ring.size(&ZZ).unwrap(), ZZ.one());
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
use crate::rings::zn::zn_static::{Zn, Fp};
#[cfg(test)]
use crate::algorithms::poly_factor::FactorPolyField;
#[cfg(test)]
use crate::homomorphism::*;
#[cfg(test)]
use crate::algorithms::cyclotomic::cyclotomic_polynomial;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::poly::PolyRingStore;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;

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

    let ring = GaloisField::new(23, 2);
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(-1), 2));
    assert!(is_prim_root_of_unity(&ring, &ring.int_hom().map(2), 11));
    let poly_ring = DensePolyRing::new(&ring, "X");
    let (factorization, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &cyclotomic_polynomial(&poly_ring, 16));
    for (mut factor, _) in factorization {
        let normalization = poly_ring.base_ring().invert(poly_ring.lc(&factor).unwrap()).unwrap();
        poly_ring.inclusion().mul_assign_map(&mut factor, normalization);
        assert!(is_prim_root_of_unity(&ring, poly_ring.coefficient_at(&factor, 0), 16));
        assert!(is_prim_root_of_unity_pow2(&ring, poly_ring.coefficient_at(&factor, 0), 4));
    }
}

#[test]
fn test_get_prim_root_of_unity() {
    let ring = Fp::<17>::RING;
    assert!(is_prim_root_of_unity_pow2(&ring, &get_prim_root_of_unity_pow2(&ring, 4).unwrap(), 4));
    assert!(get_prim_root_of_unity_pow2(&ring, 5).is_none());

    let ring = Fp::<101>::RING;
    assert!(is_prim_root_of_unity_pow2(&ring, &get_prim_root_of_unity_pow2(&ring, 2).unwrap(), 2));
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity(&ring, 25).unwrap(), 25));
    assert!(get_prim_root_of_unity_pow2(&ring, 3).is_none());
    assert!(get_prim_root_of_unity(&ring, 125).is_none());
    
    let ring = GaloisField::new(23, 2);
    assert!(is_prim_root_of_unity_pow2(&ring, &get_prim_root_of_unity_pow2(&ring, 4).unwrap(), 4));
    assert!(get_prim_root_of_unity_pow2(&ring, 5).is_none());
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity(&ring, 3).unwrap(), 3));

    let ring = GaloisField::new(17, 16);
    assert!(is_prim_root_of_unity_pow2(&ring, &get_prim_root_of_unity_pow2(&ring, 4).unwrap(), 4));
}