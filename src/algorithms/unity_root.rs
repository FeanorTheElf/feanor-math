use tracing::instrument;

use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::field::Field;
use crate::integer::int_cast;
use crate::homomorphism::Homomorphism;
use crate::integer::BigIntRing;
use crate::integer::IntegerRing;
use crate::ring::*;
use crate::primitive_int::*;
use crate::rings::finite::*;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::ordered::OrderedRingStore;
use crate::rings::zn::ZnRing;

use super::int_factor::factor;

#[instrument(skip_all, level = "trace")]
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

#[instrument(skip_all, level = "trace")]
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
pub fn get_prim_root_of_unity_gen<R, I>(ring: R, n: &El<I>, ZZ: I, order: &El<I>) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing,
        I: RingStore + Copy,
        I::Type: IntegerRing
{
    let power = ZZ.checked_div(order, n)?;
    
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&ring.size(&ZZ).unwrap()) as u128);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        if is_prim_root_of_unity_gen(&ring, &current, n, ZZ) {
            return Some(current);
        }
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZ);
    }
    unreachable!()
}

///
/// Returns a primitive `n`-th root of unity in the given finite field,
/// or `None`, if the order of the multiplicative group of the field is
/// not divisible by `n`.
/// 
#[instrument(skip_all, level = "trace")]
pub fn get_prim_root_of_unity<R>(ring: R, n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: FiniteRing + Field
{
    let order = BigIntRing::RING.sub(ring.size(&BigIntRing::RING).unwrap(), BigIntRing::RING.one());
    get_prim_root_of_unity_gen(ring, &int_cast(n.try_into().unwrap(), BigIntRing::RING, StaticRing::<i64>::RING), BigIntRing::RING, &order)
}

#[instrument(skip_all, level = "trace")]
#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity_zn_gen<R, I>(ring: R, ZZ: &I, n: &El<I>) -> Option<El<R>>
    where R: RingStore, 
        R::Type: ZnRing,
        I: RingStore,
        I::Type: IntegerRing
{
    let order = factor(ZZ, ring.characteristic(ZZ).unwrap()).into_iter().map(|(p, e)| if ZZ.eq_el(&p, &ZZ.int_hom().map(2)) {
        match e {
            1 => ZZ.one(),
            2 => p,
            e => ZZ.pow(p, e - 2)
        }
    } else {
        ZZ.mul(ZZ.sub_ref_fst(&p, ZZ.one()), ZZ.pow(p, e - 1))
    }).fold(ZZ.one(), |current, next| if ZZ.is_lt(&current, &next) { next } else { current });
    get_prim_root_of_unity_gen(ring, n, ZZ, &order)
}

///
/// Returns a primitive `n`-th root of unity in the given ring `Z/kZ`,
/// or `None`, if the order of the multiplicative group of the field is
/// not divisible by `n`.
/// 
/// Note that if `Z/kZ` is not a prime, it may have more than `phi(k)`
/// primitive roots of unity, in particular there may be `a, b` with
/// `<a>, <b> <= (Z/kZ)*` both of order `n`, but `<a> n <b> = { 1 }`.
/// 
#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity_zn<R>(ring: R, n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: ZnRing
{
    get_prim_root_of_unity_zn_gen(ring, &BigIntRing::RING, &int_cast(n as i64, BigIntRing::RING, StaticRing::<i64>::RING))
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
    let order = BigIntRing::RING.sub(ring.size(&BigIntRing::RING).unwrap(), BigIntRing::RING.one());
    get_prim_root_of_unity_gen(ring, &BigIntRing::RING.power_of_two(log2_n), BigIntRing::RING, &order)
}

///
/// Returns a primitive `n`-th root of unity in the given ring `Z/kZ`,
/// or `None`, if the order of the multiplicative group of the field is
/// not divisible by `n`.
/// 
/// Note that if `Z/kZ` is not a prime, it may have more than `phi(k)`
/// primitive roots of unity, in particular there may be `a, b` with
/// `<a>, <b> <= (Z/kZ)*` both of order `n`, but `<a> n <b> = { 1 }`.
/// 
#[stability::unstable(feature = "enable")]
pub fn get_prim_root_of_unity_pow2_zn<R, I>(ring: R, log2_n: usize) -> Option<El<R>>
    where R: RingStore, 
        R::Type: ZnRing
{
    get_prim_root_of_unity_zn_gen(ring, &BigIntRing::RING, &BigIntRing::RING.power_of_two(log2_n))
}

#[cfg(test)]
use crate::rings::zn::zn_static::{Zn, Fp};
#[cfg(test)]
use crate::algorithms::poly_factor::FactorPolyField;
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
    feanor_tracing::DelayedLogger::init_test();
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
    feanor_tracing::DelayedLogger::init_test();
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

#[test]
fn test_get_prim_root_of_unity_zn() {
    let ring = Zn::<1>::RING;
    assert!(get_prim_root_of_unity_zn(&ring, 2).is_none());

    let ring = Fp::<2>::RING;
    assert!(get_prim_root_of_unity_zn(&ring, 2).is_none());

    let ring = Zn::<4>::RING;
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 2).unwrap(), 2));
    assert!(get_prim_root_of_unity_zn(&ring, 4).is_none());

    let ring = Zn::<8>::RING;
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 2).unwrap(), 2));
    assert!(get_prim_root_of_unity_zn(&ring, 4).is_none());

    let ring = Zn::<16>::RING;
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 4).unwrap(), 4));
    assert!(get_prim_root_of_unity_zn(&ring, 5).is_none());

    let ring = Zn::<15>::RING;
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 4).unwrap(), 4));
    assert!(get_prim_root_of_unity_zn(&ring, 5).is_none());

    let ring = Zn::<75>::RING;
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 5).unwrap(), 5));
    assert!(is_prim_root_of_unity(&ring, &get_prim_root_of_unity_zn(&ring, 4).unwrap(), 4));
    assert!(get_prim_root_of_unity_zn(&ring, 3).is_none());
    assert!(get_prim_root_of_unity_zn(&ring, 8).is_none());
}