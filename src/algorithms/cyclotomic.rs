use std::sync::{LazyLock, OnceLock};

use elsa::sync::FrozenMap;
use tracing::instrument;

use super::int_factor::factor;
use crate::divisibility::{DivisibilityRingStore, Domain, *};
use crate::field::Field;
use crate::integer::*;
use crate::pid::PrincipalIdealRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::finite::*;
use crate::rings::poly::sparse_poly::SparsePolyRing;
use crate::rings::poly::*;
use crate::rings::zn::{ZnRing, ZnRingStore};
use crate::{MAX_PROBABILISTIC_REPETITIONS, algorithms};

/// Computes the `n`-th cyclotomic polynomial.
///
/// The `n`-cyclotomic polynomial is the unique polynomial with integer
/// coefficients whose roots are the primitive `n`-th roots of unity.
/// In other words, we find
/// ```text
///   cyclotomic_polynomial(n) = prod_(i in (Z/nZ)*) X - exp(2 pi i / n)
/// ```
///
/// # Performance
///
/// The implementation uses an efficient algorithm based on the factorization of `n`.
/// However, since the degree of the result is `phi(n)` (about the same size as `n`),
/// it is recommended to pass a polynomial ring using a sparse representation, as this
/// will significantly improve performance (both within the algorithm, and the size of
/// the result).
///
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::sparse_poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::assert_el_eq;
///
/// let poly_ring = SparsePolyRing::new(StaticRing::<i64>::RING, "X");
/// let cyclo_poly = feanor_math::algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, 3);
/// assert_el_eq!(
///     poly_ring,
///     poly_ring.from_terms([(1, 0), (1, 1), (1, 2)].into_iter()),
///     &cyclo_poly
/// );
/// ```
#[instrument(skip_all, level = "trace")]
pub fn cyclotomic_polynomial<P>(P: P, n: usize) -> El<P>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
{
    let mut current = P.sub(P.indeterminate(), P.one());
    let ZZ = StaticRing::<i128>::RING;
    let mut power_of_x = 1;
    for (p, e) in algorithms::int_factor::factor(&ZZ, n as i128) {
        power_of_x *= ZZ.pow(p, e - 1) as usize;
        current = P
            .checked_div(
                &P.from_terms(
                    P.terms(&current)
                        .map(|(c, d)| (c.clone(), d * p as usize)),
                ),
                &current,
            )
            .unwrap();
    }
    return P.from_terms(
        P.terms(&current)
            .map(|(c, d)| (c.clone(), d * power_of_x)),
    );
}

/// Computes the `n`-th cyclotomic polynomial.
///
/// Results are cached globally. If you don't want this, use [`cyclotomic_polynomial()`].
#[stability::unstable(feature = "enable")]
pub fn cyclotomic_polynomial_cache(
    n: usize,
) -> (
    &'static SparsePolyRing<StaticRing<i32>>,
    &'static El<SparsePolyRing<StaticRing<i32>>>,
) {
    static POLY_RING_CACHE: OnceLock<SparsePolyRing<StaticRing<i32>>> = OnceLock::new();
    let ZZi32X = POLY_RING_CACHE.get_or_init(|| SparsePolyRing::new(StaticRing::<i32>::RING, "X"));
    static CYCLOTOMIC_POLY_CACHE: LazyLock<FrozenMap<usize, Box<El<SparsePolyRing<StaticRing<i32>>>>>> =
        LazyLock::new(|| FrozenMap::new());
    if let Some(res) = CYCLOTOMIC_POLY_CACHE.get(&n) {
        return (ZZi32X, res);
    } else {
        let ZZi64X = SparsePolyRing::new(StaticRing::<i32>::RING, "X");
        let poly = cyclotomic_polynomial(&ZZi64X, n);
        let poly = ZZi32X.from_terms(ZZi64X.terms(&poly).map(|(c, i)| ((*c).try_into().unwrap(), i)));
        let res = CYCLOTOMIC_POLY_CACHE.insert(n, Box::new(poly));
        return (ZZi32X, res);
    }
}

/// Checks if the given ring element is a primitive `2^log2_n`-th root of unity.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree. Note that if the underlying ring is not a field, there
/// may be more than `phi(n)` such roots.
///
/// Panics if the characteristic of the ring is not coprime to `n = 2^log2_n`, in which case
/// `Phi_n(X)` is ramified and there is no sensible notion of primitive roots of unity.
///
/// See also [`is_root_of_unity()`], [`is_prim_root_of_unity()`],
/// [`is_prim_root_of_unity_general()`].
#[instrument(skip_all, level = "trace")]
pub fn is_prim_root_of_unity_pow2<R: RingStore>(ring: R, el: &El<R>, log2_n: usize) -> bool {
    let characteristic = ring.characteristic(BigIntRing::RING).unwrap();
    assert!(BigIntRing::RING.is_zero(&characteristic) || BigIntRing::RING.is_odd(&characteristic));
    if log2_n == 0 {
        return ring.is_one(el);
    }
    ring.is_neg_one(&ring.pow(el.clone(), 1 << (log2_n - 1)))
}

/// Checks if the given ring element is a primitive `n`-th root of unity.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
///
/// See also [`is_prim_root_of_unity_pow2()`], [`is_root_of_unity()`],
/// [`is_prim_root_of_unity_general()`].
#[instrument(skip_all, level = "trace")]
pub fn is_prim_root_of_unity<R>(ring: R, el: &El<R>, n: &El<BigIntRing>) -> bool
where
    R: RingStore,
    R::Ring: Domain,
{
    let characteristic = ring.characteristic(BigIntRing::RING).unwrap();
    assert!(
        BigIntRing::RING.is_zero(&characteristic)
            || BigIntRing::RING.is_unit(&BigIntRing::RING.ideal_gen(n, &characteristic))
    );
    is_prim_root_of_unity_with_factorization(ring, el, n, &factor(BigIntRing::RING, n.clone()))
}

/// Checks if the given ring element is a primitive `n`-th root of unity.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree. Note that if the underlying ring is not a field, there
/// may be more than `phi(n)` such roots.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
///
/// See also [`is_prim_root_of_unity_pow2()`], [`is_root_of_unity()`], [`is_prim_root_of_unity()`].
#[instrument(skip_all, level = "trace")]
pub fn is_prim_root_of_unity_general<R: RingStore>(ring: R, el: &El<R>, n: usize) -> bool {
    let characteristic = ring.characteristic(BigIntRing::RING).unwrap();
    assert!(
        BigIntRing::RING.is_zero(&characteristic)
            || BigIntRing::RING.is_unit(&BigIntRing::RING.ideal_gen(
                &int_cast(n.try_into().unwrap(), BigIntRing::RING, StaticRing::<i64>::RING),
                &characteristic
            ))
    );
    if !ring.is_one(&ring.pow(el.clone(), n)) {
        return false;
    }
    let (ZZX, Phi_n) = cyclotomic_polynomial_cache(n);
    return ring.is_zero(&ZZX.evaluate(&Phi_n, el, ring.int_hom()));
}

/// Returns a primitive `n`-th root of unity in the given finite field, if one exists.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
#[instrument(skip_all, level = "trace")]
pub fn get_prim_root_of_unity<R>(ring: R, n: &El<BigIntRing>) -> Option<El<R>>
where
    R: RingStore,
    R::Ring: FiniteRing + Field,
{
    let ZZbig = BigIntRing::RING;
    let characteristic = ring.characteristic(ZZbig).unwrap();
    assert!(BigIntRing::RING.is_zero(&characteristic) || ZZbig.is_unit(&ZZbig.ideal_gen(n, &characteristic)));
    let order = ZZbig.sub(ring.size(&ZZbig).unwrap(), ZZbig.one());
    let power = ZZbig.checked_div(&order, n)?;
    let n_factorization = factor(ZZbig, n.clone());

    let mut rng = oorandom::Rand64::new(1);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZbig);
    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        if is_prim_root_of_unity_with_factorization(&ring, &current, n, &n_factorization) {
            return Some(current);
        }
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZbig);
    }
    unreachable!()
}

/// Returns a primitive `n`-th root of unity in the given ring `Z/qZ`, if one exists.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree. Note that if the underlying ring is not a field, there
/// may be more than `phi(n)` such roots.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
#[instrument(skip_all, level = "trace")]
pub fn get_prim_root_of_unity_zn<R>(ring: R, n: usize) -> Option<El<R>>
where
    R: RingStore,
    R::Ring: ZnRing,
{
    let ZZbig = BigIntRing::RING;
    let characteristic = ring.characteristic(ZZbig).unwrap();
    assert!(
        BigIntRing::RING.is_zero(&characteristic)
            || ZZbig.is_unit(&ZZbig.ideal_gen(
                &int_cast(n.try_into().unwrap(), ZZbig, StaticRing::<i64>::RING),
                &characteristic
            ))
    );

    let n_factorization = factor(StaticRing::<i64>::RING, n.try_into().unwrap());

    let mut accumulator = ring.zero();
    let ZZ = ring.integer_ring();
    'mod_pe: for (p, e) in factor(ZZ, ring.modulus().clone()) {
        let pe = ZZ.pow(p.clone(), e);
        let order = ZZ.sub_ref_fst(&pe, ZZ.pow(p, e - 1));
        let n = int_cast(n.try_into().unwrap(), ZZ, StaticRing::<i64>::RING);
        let power = ZZ.checked_div(&order, &n)?;
        let scale = ring.coerce(ZZ, ZZ.checked_div(ring.modulus(), &pe).unwrap());

        let mut rng = oorandom::Rand64::new(1);
        let base = ring.mul_ref_snd(ring.random_element(|| rng.rand_u64()), &scale);
        let mut current = ring.pow_gen(base, &power, ZZ);
        let one_mod_p = ring.pow_gen(current.clone(), &n, ZZ);
        for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
            if n_factorization.iter().all(|(factor_n, _)| {
                !ring.eq_el(
                    &one_mod_p,
                    &ring.pow_gen(
                        current.clone(),
                        &ZZ.checked_div(&n, &int_cast(*factor_n, ZZ, StaticRing::<i64>::RING))
                            .unwrap(),
                        ZZ,
                    ),
                )
            }) {
                ring.add_assign(&mut accumulator, current);
                continue 'mod_pe;
            }
            current = ring.pow_gen(
                ring.mul_ref_snd(ring.random_element(|| rng.rand_u64()), &scale),
                &power,
                ZZ,
            );
        }
        unreachable!()
    }
    return Some(accumulator);
}

/// Returns a primitive `2^log2_n`-th root of unity in the given finite field`, if one exists.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree. Note that if the underlying ring is not a field, there
/// may be more than `phi(n)` such roots.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
#[instrument(skip_all, level = "trace")]
pub fn get_prim_root_of_unity_pow2<R>(ring: R, log2_n: usize) -> Option<El<R>>
where
    R: RingStore,
    R::Ring: FiniteRing + Field,
{
    let ZZbig = BigIntRing::RING;
    let characteristic = ring.characteristic(ZZbig).unwrap();
    assert!(ZZbig.is_zero(&characteristic) || ZZbig.is_odd(&characteristic));
    let order = ZZbig.sub(ring.size(&ZZbig).unwrap(), ZZbig.one());
    let power = ZZbig.checked_div(&order, &ZZbig.power_of_two(log2_n))?;

    let mut rng = oorandom::Rand64::new(1);
    let mut current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZbig);
    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        if is_prim_root_of_unity_pow2(&ring, &current, log2_n) {
            return Some(current);
        }
        current = ring.pow_gen(ring.random_element(|| rng.rand_u64()), &power, ZZbig);
    }
    unreachable!()
}

/// Returns a primitive `2^log2_n`-th root of unity in the given ring `Z/qZ`, if one exists.
///
/// A primitive `n`-th root of unity is defined as any root of the `n`-th cyclotomic polynomial
/// `Phi_n(X)`, assuming it is squarefree. Note that if the underlying ring is not a field, there
/// may be more than `phi(n)` such roots.
///
/// Panics if the characteristic of the ring is not coprime to `n`, in which case `Phi_n(X)` is
/// ramified and there is no sensible notion of primitive roots of unity.
#[instrument(skip_all, level = "trace")]
pub fn get_prim_root_of_unity_pow2_zn<R>(ring: R, log2_n: usize) -> Option<El<R>>
where
    R: RingStore,
    R::Ring: ZnRing,
{
    if log2_n == 0 {
        return Some(ring.one());
    }
    let ZZbig = BigIntRing::RING;
    let characteristic = ring.characteristic(ZZbig).unwrap();
    assert!(ZZbig.is_zero(&characteristic) || ZZbig.is_odd(&characteristic));

    let mut accumulator = ring.zero();
    let ZZ = ring.integer_ring();
    'mod_pe: for (p, e) in factor(ZZ, ring.modulus().clone()) {
        let pe = ZZ.pow(p.clone(), e);
        let order = ZZ.sub_ref_fst(&pe, ZZ.pow(p, e - 1));
        let power = ZZ.checked_div(&order, &ZZ.power_of_two(log2_n))?;
        let scale = ring.coerce(ZZ, ZZ.checked_div(ring.modulus(), &pe).unwrap());

        let mut rng = oorandom::Rand64::new(1);
        let mut current = ring.pow_gen(
            ring.mul_ref_snd(ring.random_element(|| rng.rand_u64()), &scale),
            &power,
            ZZ,
        );
        let one_mod_p = ring.pow_gen(current.clone(), &ZZ.power_of_two(log2_n), ZZ);
        for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
            if !ring.eq_el(
                &one_mod_p,
                &ring.pow_gen(current.clone(), &ZZ.power_of_two(log2_n - 1), ZZ),
            ) {
                ring.add_assign(&mut accumulator, current);
                continue 'mod_pe;
            }
            current = ring.pow_gen(
                ring.mul_ref_snd(ring.random_element(|| rng.rand_u64()), &scale),
                &power,
                ZZ,
            );
        }
        unreachable!()
    }
    return Some(accumulator);
}

#[instrument(skip_all, level = "trace")]
#[stability::unstable(feature = "enable")]
pub fn is_prim_root_of_unity_with_factorization<R>(
    ring: R,
    el: &El<R>,
    n: &El<BigIntRing>,
    n_factorization: &[(El<BigIntRing>, usize)],
) -> bool
where
    R: RingStore,
    R::Ring: Domain,
{
    let characteristic = ring.characteristic(BigIntRing::RING).unwrap();
    debug_assert!(
        BigIntRing::RING.is_zero(&characteristic)
            || BigIntRing::RING.is_unit(&BigIntRing::RING.ideal_gen(n, &characteristic))
    );
    let ZZbig = BigIntRing::RING;
    debug_assert!(ZZbig.eq_el(
        n,
        &ZZbig.prod(n_factorization.iter().map(|(p, e)| ZZbig.pow(p.clone(), *e)))
    ));
    if !ring.is_one(&ring.pow_gen(el.clone(), n, ZZbig)) {
        return false;
    }
    for (p, _) in n_factorization {
        if ring.is_one(&ring.pow_gen(el.clone(), &ZZbig.checked_div(n, p).unwrap(), ZZbig)) {
            return false;
        }
    }
    return true;
}

#[cfg(test)]
use crate::homomorphism::*;
#[cfg(test)]
use crate::rings::extension::FreeAlgebraStore;
#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use crate::rings::poly::PolyRingStore;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_64b::Zn64B;
#[cfg(test)]
use crate::rings::zn::zn_static::{Fp, Zn};

#[test]
fn test_cyclotomic_polynomial() {
    feanor_tracing::DelayedLogger::init_test();
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 1), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 2)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 1), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 3)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 4)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 5)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (6, 1), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 6)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 7)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 6), (6, 3), (1, 0)]),
        &cyclotomic_polynomial(&poly_ring, 18)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([
            (1, 48),
            (1, 47),
            (1, 46),
            (6, 43),
            (6, 42),
            (5, 41),
            (6, 40),
            (6, 39),
            (1, 36),
            (1, 35),
            (1, 34),
            (1, 33),
            (1, 32),
            (1, 31),
            (6, 28),
            (6, 26),
            (6, 24),
            (6, 22),
            (6, 20),
            (1, 17),
            (1, 16),
            (1, 15),
            (1, 14),
            (1, 13),
            (1, 12),
            (6, 9),
            (6, 8),
            (5, 7),
            (6, 6),
            (6, 5),
            (1, 2),
            (1, 1),
            (1, 0)
        ]),
        &cyclotomic_polynomial(&poly_ring, 105)
    ));
}

#[test]
fn test_is_prim_root_of_unity_pow2() {
    assert_eq!(true, is_prim_root_of_unity_pow2(StaticRing::<i64>::RING, &1, 0));
    assert_eq!(false, is_prim_root_of_unity_pow2(StaticRing::<i64>::RING, &1, 1));
    assert_eq!(true, is_prim_root_of_unity_pow2(StaticRing::<i64>::RING, &-1, 1));

    assert_eq!(true, is_prim_root_of_unity_pow2(Fp::<3>::RING, &2, 1));
    assert_eq!(false, is_prim_root_of_unity_pow2(Fp::<3>::RING, &2, 2));
    assert_eq!(false, is_prim_root_of_unity_pow2(Fp::<3>::RING, &2, 0));

    assert_eq!(true, is_prim_root_of_unity_pow2(Fp::<19>::RING, &18, 1));
    assert_eq!(false, is_prim_root_of_unity_pow2(Fp::<19>::RING, &18, 2));

    let F = FreeAlgebraImpl::new(Fp::<5>::RING, 2, [3, 1]).as_field().unwrap();
    let FEl = |x: [u64; 2]| F.from_canonical_basis(x);
    assert_eq!(true, is_prim_root_of_unity_pow2(&F, &FEl([3, 4]), 3));
    assert_eq!(false, is_prim_root_of_unity_pow2(&F, &FEl([3, 4]), 4));
    assert_eq!(false, is_prim_root_of_unity_pow2(&F, &FEl([3, 4]), 2));

    assert_eq!(true, is_prim_root_of_unity_pow2(Zn::<{ 17 * 29 }>::RING, &302, 2));

    let F = Zn64B::new(65537);
    assert_eq!(true, is_prim_root_of_unity_pow2(&F, &F.int_hom().map(65281), 2));
}

#[test]
fn test_is_prim_root_of_unity() {
    assert_eq!(
        true,
        is_prim_root_of_unity(StaticRing::<i64>::RING, &1, &BigIntRing::RING.int_hom().map(1))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(StaticRing::<i64>::RING, &1, &BigIntRing::RING.int_hom().map(2))
    );
    assert_eq!(
        true,
        is_prim_root_of_unity(StaticRing::<i64>::RING, &-1, &BigIntRing::RING.int_hom().map(2))
    );

    assert_eq!(
        true,
        is_prim_root_of_unity(Fp::<3>::RING, &2, &BigIntRing::RING.int_hom().map(2))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(Fp::<3>::RING, &2, &BigIntRing::RING.int_hom().map(4))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(Fp::<3>::RING, &2, &BigIntRing::RING.int_hom().map(1))
    );

    assert_eq!(
        true,
        is_prim_root_of_unity(Fp::<19>::RING, &6, &BigIntRing::RING.int_hom().map(9))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(Fp::<19>::RING, &6, &BigIntRing::RING.int_hom().map(3))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(Fp::<19>::RING, &6, &BigIntRing::RING.int_hom().map(18))
    );

    let F = FreeAlgebraImpl::new(Fp::<5>::RING, 2, [3, 1]).as_field().unwrap();
    let FEl = |x: [u64; 2]| F.from_canonical_basis(x);
    assert_eq!(
        true,
        is_prim_root_of_unity(&F, &FEl([2, 2]), &BigIntRing::RING.int_hom().map(6))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(&F, &FEl([2, 2]), &BigIntRing::RING.int_hom().map(9))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(&F, &FEl([2, 2]), &BigIntRing::RING.int_hom().map(4))
    );
    assert_eq!(
        false,
        is_prim_root_of_unity(&F, &FEl([2, 2]), &BigIntRing::RING.int_hom().map(8))
    );
}

#[test]
fn test_is_prim_root_of_unity_general() {
    assert_eq!(true, is_prim_root_of_unity_general(StaticRing::<i64>::RING, &1, 1));
    assert_eq!(false, is_prim_root_of_unity_general(StaticRing::<i64>::RING, &1, 2));
    assert_eq!(true, is_prim_root_of_unity_general(StaticRing::<i64>::RING, &-1, 2));

    assert_eq!(true, is_prim_root_of_unity_general(Fp::<3>::RING, &2, 2));
    assert_eq!(false, is_prim_root_of_unity_general(Fp::<3>::RING, &2, 4));
    assert_eq!(false, is_prim_root_of_unity_general(Fp::<3>::RING, &2, 1));

    assert_eq!(true, is_prim_root_of_unity_general(Fp::<19>::RING, &6, 9));
    assert_eq!(false, is_prim_root_of_unity_general(Fp::<19>::RING, &6, 3));
    assert_eq!(false, is_prim_root_of_unity_general(Fp::<19>::RING, &6, 18));

    let F = FreeAlgebraImpl::new(Fp::<5>::RING, 2, [3, 1]);
    let FEl = |x: [u64; 2]| F.from_canonical_basis(x);
    assert_eq!(true, is_prim_root_of_unity_general(&F, &FEl([2, 2]), 6));
    assert_eq!(false, is_prim_root_of_unity_general(&F, &FEl([2, 2]), 9));
    assert_eq!(false, is_prim_root_of_unity_general(&F, &FEl([2, 2]), 4));
    assert_eq!(false, is_prim_root_of_unity_general(&F, &FEl([2, 2]), 8));

    assert_eq!(false, is_prim_root_of_unity_general(Zn::<85>::RING, &16, 2));
    assert_eq!(true, is_prim_root_of_unity_general(Zn::<85>::RING, &84, 2));
    assert_eq!(false, is_prim_root_of_unity_general(Zn::<85>::RING, &3, 16));
    assert_eq!(false, is_prim_root_of_unity_general(Zn::<85>::RING, &3, 4));
    assert_eq!(true, is_prim_root_of_unity_general(Zn::<85>::RING, &13, 4));

    assert_eq!(true, is_prim_root_of_unity_general(Zn::<{ 17 * 29 }>::RING, &302, 4));
}

#[test]
fn test_get_prim_root_of_unity() {
    assert!(is_prim_root_of_unity(
        Fp::<17>::RING,
        &get_prim_root_of_unity(Fp::<17>::RING, &BigIntRing::RING.int_hom().map(4)).unwrap(),
        &BigIntRing::RING.int_hom().map(4)
    ));
    assert!(is_prim_root_of_unity(
        Fp::<17>::RING,
        &get_prim_root_of_unity(Fp::<17>::RING, &BigIntRing::RING.int_hom().map(16)).unwrap(),
        &BigIntRing::RING.int_hom().map(16)
    ));
    assert!(get_prim_root_of_unity(Fp::<17>::RING, &BigIntRing::RING.int_hom().map(32)).is_none());

    assert!(is_prim_root_of_unity(
        Fp::<19>::RING,
        &get_prim_root_of_unity(Fp::<19>::RING, &BigIntRing::RING.int_hom().map(9)).unwrap(),
        &BigIntRing::RING.int_hom().map(9)
    ));
    assert!(is_prim_root_of_unity(
        Fp::<19>::RING,
        &get_prim_root_of_unity(Fp::<19>::RING, &BigIntRing::RING.int_hom().map(6)).unwrap(),
        &BigIntRing::RING.int_hom().map(6)
    ));
    assert!(get_prim_root_of_unity(Fp::<19>::RING, &BigIntRing::RING.int_hom().map(4)).is_none());

    let F = FreeAlgebraImpl::new(Fp::<5>::RING, 2, [3, 1]).as_field().unwrap();
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity(&F, &BigIntRing::RING.int_hom().map(6)).unwrap(),
        &BigIntRing::RING.int_hom().map(6)
    ));
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity(&F, &BigIntRing::RING.int_hom().map(4)).unwrap(),
        &BigIntRing::RING.int_hom().map(4)
    ));
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity(&F, &BigIntRing::RING.int_hom().map(8)).unwrap(),
        &BigIntRing::RING.int_hom().map(8)
    ));
    assert!(&get_prim_root_of_unity(&F, &BigIntRing::RING.int_hom().map(89)).is_none());
}

#[test]
#[should_panic]
fn test_get_prim_root_of_unity_ramified() {
    _ = get_prim_root_of_unity(GaloisField::new(17, 2), &BigIntRing::RING.int_hom().map(17));
}

#[test]
fn test_get_prim_root_of_unity_zn() {
    assert!(is_prim_root_of_unity_general(
        Zn::<17>::RING,
        &get_prim_root_of_unity_zn(Zn::<17>::RING, 4).unwrap(),
        4
    ));
    assert!(is_prim_root_of_unity_general(
        Zn::<17>::RING,
        &get_prim_root_of_unity_zn(Zn::<17>::RING, 16).unwrap(),
        16
    ));
    assert!(get_prim_root_of_unity_zn(Zn::<17>::RING, 32).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<19>::RING,
        &get_prim_root_of_unity_zn(Zn::<19>::RING, 9).unwrap(),
        9
    ));
    assert!(is_prim_root_of_unity_general(
        Zn::<19>::RING,
        &get_prim_root_of_unity_zn(Zn::<19>::RING, 6).unwrap(),
        6
    ));
    assert!(get_prim_root_of_unity_zn(Zn::<19>::RING, 4).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<{ 19 * 19 }>::RING,
        &get_prim_root_of_unity_zn(Zn::<{ 19 * 19 }>::RING, 9).unwrap(),
        9
    ));
    assert!(is_prim_root_of_unity_general(
        Zn::<{ 19 * 19 }>::RING,
        &get_prim_root_of_unity_zn(Zn::<{ 19 * 19 }>::RING, 6).unwrap(),
        6
    ));
    assert!(get_prim_root_of_unity_zn(Zn::<{ 19 * 19 }>::RING, 4).is_none());

    assert!(get_prim_root_of_unity_zn(Zn::<1024>::RING, 3).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<{ 29 * 17 }>::RING,
        &get_prim_root_of_unity_zn(Zn::<{ 29 * 17 }>::RING, 4).unwrap(),
        4
    ));
    assert!(get_prim_root_of_unity_zn(Zn::<{ 29 * 17 }>::RING, 7).is_none());
    assert!(get_prim_root_of_unity_zn(Zn::<{ 29 * 17 }>::RING, 8).is_none());
}

#[test]
#[should_panic]
fn test_get_prim_root_of_unity_zn_ramified() { _ = get_prim_root_of_unity_zn(Zn::<{ 17 * 17 }>::RING, 17); }

#[test]
fn test_get_prim_root_of_unity_pow2() {
    assert!(is_prim_root_of_unity(
        Fp::<17>::RING,
        &get_prim_root_of_unity_pow2(Fp::<17>::RING, 2).unwrap(),
        &BigIntRing::RING.int_hom().map(4)
    ));
    assert!(is_prim_root_of_unity(
        Fp::<17>::RING,
        &get_prim_root_of_unity_pow2(Fp::<17>::RING, 4).unwrap(),
        &BigIntRing::RING.int_hom().map(16)
    ));
    assert!(get_prim_root_of_unity(Fp::<17>::RING, &BigIntRing::RING.int_hom().map(32)).is_none());

    assert!(get_prim_root_of_unity_pow2(Fp::<19>::RING, 2).is_none());

    let F = FreeAlgebraImpl::new(Fp::<5>::RING, 2, [3, 1]).as_field().unwrap();
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity_pow2(&F, 1).unwrap(),
        &BigIntRing::RING.int_hom().map(2)
    ));
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity_pow2(&F, 2).unwrap(),
        &BigIntRing::RING.int_hom().map(4)
    ));
    assert!(is_prim_root_of_unity(
        &F,
        &get_prim_root_of_unity_pow2(&F, 3).unwrap(),
        &BigIntRing::RING.int_hom().map(8)
    ));
    assert!(&get_prim_root_of_unity_pow2(&F, 4).is_none());
}

#[test]
#[should_panic]
fn test_get_prim_root_of_unity_pow2_ramified() { _ = get_prim_root_of_unity_pow2(Fp::<2>::RING, 1).unwrap(); }

#[test]
fn test_get_prim_root_of_unity_pow2_zn() {
    assert!(is_prim_root_of_unity_general(
        Zn::<17>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<17>::RING, 0).unwrap(),
        1
    ));

    assert!(is_prim_root_of_unity_general(
        Zn::<17>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<17>::RING, 2).unwrap(),
        4
    ));
    assert!(is_prim_root_of_unity_general(
        Zn::<17>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<17>::RING, 4).unwrap(),
        16
    ));
    assert!(get_prim_root_of_unity_pow2_zn(Zn::<17>::RING, 32).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<{ 17 * 17 }>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<{ 17 * 17 }>::RING, 4).unwrap(),
        16
    ));
    assert!(get_prim_root_of_unity_pow2_zn(Zn::<17>::RING, 32).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<19>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<19>::RING, 1).unwrap(),
        2
    ));
    assert!(get_prim_root_of_unity_pow2_zn(Zn::<19>::RING, 2).is_none());

    assert!(is_prim_root_of_unity_general(
        Zn::<{ 29 * 17 }>::RING,
        &get_prim_root_of_unity_pow2_zn(Zn::<{ 29 * 17 }>::RING, 2).unwrap(),
        4
    ));
    assert!(get_prim_root_of_unity_pow2_zn(Zn::<{ 29 * 17 }>::RING, 3).is_none());
}

#[test]
#[should_panic]
fn test_get_prim_root_of_unity_pow2_zn_ramified() { _ = get_prim_root_of_unity_pow2_zn(Zn::<1024>::RING, 8).unwrap(); }
