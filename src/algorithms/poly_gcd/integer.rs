use std::cmp::Ordering;
use std::mem::swap;

use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::hensel::HenselLift;
use crate::algorithms::poly_gcd::gcd_lift::*;
use crate::algorithms::poly_gcd::power_decomposition_lift::*;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_root::poly_root;
use crate::algorithms::primelist::prime_fields_for_local_computation;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::poly::{PolyRing, PolyRingStore};
use crate::ring_impls::zn::zn_big::ZnGB;
use crate::ring_impls::zn::*;

const HOPE_FOR_SQUAREFREE_ATTEMPTS: usize = 1;
const BEST_EFFORT_SQUAREFREE_CHECKS: usize = 3;

#[instrument(skip_all, level = "trace")]
fn lift_poly<P>(
    ZZX: P,
    ZpeX: &DensePolyRing<ZnGB<BigIntRing>>,
    poly: &El<DensePolyRing<ZnGB<BigIntRing>>>,
) -> Result<El<P>, ()>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    let ZZ = ZZX.base_ring();
    let Zpe = ZpeX.base_ring();
    let size_bound = ZZbig.floor_div(Zpe.modulus().clone(), &int_cast(4, ZZbig, ZZi64));
    ZZX.try_from_terms(
        ZpeX.terms(poly)
            .map(|(c, i)| (Zpe.smallest_lift(c.clone()), i))
            .map(|(c, i)| {
                if ZZbig.abs_cmp(&c, &size_bound) == Ordering::Less {
                    Ok((c, i))
                } else {
                    Err(())
                }
            })
            .map(|x| x.map(|(c, i)| (int_cast(c, ZZ, ZZbig), i))),
    )
}

#[instrument(skip_all, level = "trace")]
fn lift_gcd_factorization<P>(
    ZZX: P,
    lifter: &mut HenselLift<DensePolyRing<ZnGB<BigIntRing>>>,
    target: &El<P>,
    prime: &El<BigIntRing>,
    lift_to_degree: usize,
) -> Result<El<P>, LiftUnsuccessful>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    let ZpeX = DensePolyRing::new(ZnGB::new(ZZbig, ZZbig.pow(prime.clone(), lift_to_degree)), "X");
    let Zpe = ZpeX.base_ring().clone();
    let target_mod_pe = ZpeX
        .lifted_hom(ZZX, ZpeX.base_ring().can_hom(ZZX.base_ring()).unwrap())
        .map_ref(target);
    let hom = Zpe.can_hom(&ZZbig).unwrap();
    take_mut::take(lifter, |lifter| {
        lifter.lift_to(lift_to_degree, ZpeX, &target_mod_pe, |old_base_ring, _, x| {
            hom.map(old_base_ring.smallest_lift(x.clone()))
        })
    });
    let lifted_factorization = lifter
        .factorization()
        .map(|f| lift_poly(ZZX, lifter.poly_ring(), f))
        .collect::<Result<Vec<_>, _>>();
    if let Ok(lifted_factorization) = lifted_factorization {
        let [gcd_lifted, target_over_gcd_lifted] = lifted_factorization.try_into().ok().unwrap();
        if ZZX.eq_el(&ZZX.mul_ref(&gcd_lifted, &target_over_gcd_lifted), target) {
            return Ok(gcd_lifted);
        } else {
            return Err(LiftUnsuccessful);
        }
    } else {
        return Err(LiftUnsuccessful);
    }
}

/// Tries to compute the gcd of monic polynomials `f, g in Z[X]`.
///
/// This will fail if `lhs/d, d` and `rhs/d, d` are both not pairwise coprime, where `d = gcd(lhs,
/// rhs)`. It can, in theory, also fail in other settings, but the probability is very low for
/// larger values of `attempts`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_gcd_integer_monic<P>(ZZX: P, lhs: &El<P>, rhs: &El<P>, attempts: usize) -> PolyGCDResult<El<P>>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    if ZZX.is_zero(lhs) {
        return PolyGCDResult::FoundGCD(rhs.clone());
    } else if ZZX.is_zero(rhs) {
        return PolyGCDResult::FoundGCD(lhs.clone());
    }
    let ZZ = ZZX.base_ring();
    assert!(ZZ.is_one(ZZX.lc(lhs).unwrap()));
    assert!(ZZ.is_one(ZZX.lc(rhs).unwrap()));

    let gcds_modulo_p = prime_fields_for_local_computation().take(attempts).map(|Fp| {
        let FpX = DensePolyRing::new(Fp, "X");
        let ZZX_to_FpX = FpX.lifted_hom(&ZZX, Fp.can_hom(ZZ).unwrap());
        let lhs = ZZX_to_FpX.map_ref(lhs);
        let rhs = ZZX_to_FpX.map_ref(rhs);
        let gcd = PolyTFracGCDRing::gcd(&FpX, &lhs, &rhs);
        return (PolyGCDSignature::new(FpX.degree(&gcd).unwrap()), (FpX, lhs, rhs, gcd));
    });
    poly_gcd_from_quotients(
        gcds_modulo_p,
        |(FpX, lhs, rhs, gcd)| {
            let prime = int_cast(*FpX.base_ring().modulus(), ZZbig, ZZi64);
            let new_base_ring = ZnGB::new(ZZbig, prime.clone());
            let new_poly_ring = DensePolyRing::new(new_base_ring.clone(), "X");
            let hom = ZnReductionMap::new(FpX.base_ring(), new_base_ring).unwrap();
            let lhs_over_gcd = FpX.checked_div(&lhs, &gcd).unwrap();
            if let Ok(lifter) = HenselLift::new(FpX.clone(), vec![gcd.clone(), lhs_over_gcd]) {
                return Ok((true, lifter.change_ring(new_poly_ring, hom), prime));
            }
            let rhs_over_gcd = FpX.checked_div(&rhs, &gcd).unwrap();
            if let Ok(lifter) = HenselLift::new(FpX.clone(), vec![gcd, rhs_over_gcd]) {
                return Ok((false, lifter.change_ring(new_poly_ring, hom), prime));
            }
            return Err(NotLiftable::NotSquarefree);
        },
        |(lift_to_lhs, lifter, prime), lift_to_degree| {
            if *lift_to_lhs {
                lift_gcd_factorization(&ZZX, lifter, lhs, prime, lift_to_degree)
            } else {
                lift_gcd_factorization(&ZZX, lifter, rhs, prime, lift_to_degree)
            }
        },
    )
}

/// Computes the gcd of monic polynomials `f, g in Z[X]`.
///
/// Use this when implementing [`PolyTFracGCDRing`] for integer
/// rings; Otherwise, compute power decompositions through [`PolyTFracGCDRing::gcd()`].
///
/// [`PolyTFracGCDRing`]: crate::algorithms::poly_gcd::PolyTFracGCDRing
/// [`PolyTFracGCDRing::gcd()`]: crate::algorithms::poly_gcd::PolyTFracGCDRing::gcd()
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_gcd_integer<P>(ZZX: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
where
    P: RingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingBase<P>: IntegerRing,
{
    if ZZX.is_zero(lhs) {
        return rhs.clone();
    } else if ZZX.is_zero(rhs) {
        return lhs.clone();
    }
    let lc_lcm = ZZX.base_ring().lcm(ZZX.lc(lhs).unwrap(), ZZX.lc(rhs).unwrap());
    let (mut lhs, _) = make_monic(ZZX, lhs, &lc_lcm);
    let (mut rhs, _) = make_monic(ZZX, rhs, &lc_lcm);
    match poly_gcd_integer_monic(ZZX, &lhs, &rhs, HOPE_FOR_SQUAREFREE_ATTEMPTS) {
        PolyGCDResult::TrivialGCD => return ZZX.one(),
        PolyGCDResult::FoundGCD(res) => return make_primitive(ZZX, &unmake_monic(ZZX, &res, &lc_lcm, &lc_lcm)).0,
        _ => {}
    }

    if ZZX.degree(&lhs).unwrap() > ZZX.degree(&rhs).unwrap() {
        swap(&mut lhs, &mut rhs);
    }
    let lhs_power_decomposition = poly_power_decomposition_integer_monic(ZZX, &lhs, PROBABILISTIC_REPETITIONS);
    let mut result = ZZX.one();
    for (fi, i) in &lhs_power_decomposition {
        for _ in 0..*i {
            match poly_gcd_integer_monic(ZZX, &fi, &rhs, PROBABILISTIC_REPETITIONS) {
                PolyGCDResult::TrivialGCD => break,
                PolyGCDResult::FoundGCD(gcd_i) => {
                    rhs = ZZX.checked_div(&rhs, &gcd_i).unwrap();
                    ZZX.mul_assign(&mut result, gcd_i);
                }
                _ => unreachable!(),
            }
        }
    }
    return make_primitive(ZZX, &unmake_monic(ZZX, &result, &lc_lcm, &lc_lcm)).0;
}

/// Checks whether the given integral polynomial is squarefree modulo a few suitable primes.
///  - If yes, it is for sure squarefree over the integers.
///  - If not, it is likely that it is not squarefree over the integers.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn best_effort_poly_is_squarefree_integer<P>(ZZX: P, poly: &El<P>) -> bool
where
    P: RingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingBase<P>: IntegerRing,
{
    if ZZX.is_zero(poly) {
        return false;
    }
    assert!(!ZZX.is_zero(poly));
    let ZZ = ZZX.base_ring();
    prime_fields_for_local_computation()
        .filter(|Fp| !ZZ.divides(ZZX.lc(poly).unwrap(), &int_cast(*Fp.modulus(), ZZ, ZZi64)))
        .take(BEST_EFFORT_SQUAREFREE_CHECKS)
        .all(|Fp| {
            let FpX = DensePolyRing::new(Fp, "X");
            let ZZX_to_FpX = FpX.lifted_hom(&ZZX, Fp.can_hom(ZZ).unwrap());
            let poly = ZZX_to_FpX.map_ref(poly);
            PolyTFracGCDRing::is_squarefree(&FpX, &poly)
        })
}

#[instrument(skip_all, level = "trace")]
fn lift_power_decomposition<P>(
    ZZX: P,
    lifter: &mut HenselLift<DensePolyRing<ZnGB<BigIntRing>>>,
    exponents: &[usize],
    target: &El<P>,
    prime: &El<BigIntRing>,
    lift_to_degree: usize,
) -> Result<Vec<(El<P>, usize)>, LiftUnsuccessful>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    let ZpeX = DensePolyRing::new(ZnGB::new(ZZbig, ZZbig.pow(prime.clone(), lift_to_degree)), "X");
    let Zpe = ZpeX.base_ring().clone();
    let target_mod_pe = ZpeX
        .lifted_hom(ZZX, ZpeX.base_ring().can_hom(ZZX.base_ring()).unwrap())
        .map_ref(target);
    let hom = Zpe.can_hom(&ZZbig).unwrap();
    take_mut::take(lifter, |lifter| {
        lifter.lift_to(lift_to_degree, ZpeX, &target_mod_pe, |old_base_ring, _, x| {
            hom.map(old_base_ring.smallest_lift(x.clone()))
        })
    });

    let lifted_factorization = lifter
        .factorization()
        .map(|f| lift_poly(ZZX, lifter.poly_ring(), f))
        .collect::<Result<Vec<_>, _>>();
    if let Ok(lifted_factorization) = lifted_factorization {
        assert_eq!(exponents.len(), lifted_factorization.len());
        if ZZX.eq_el(&ZZX.prod(lifted_factorization.iter().cloned()), target) {
            return lifted_factorization
                .into_iter()
                .zip(exponents.iter())
                .map(|(f, i)| poly_root(ZZX, &f, *i).map(|f| (f, *i)).ok_or(LiftUnsuccessful))
                .collect();
        } else {
            return Err(LiftUnsuccessful);
        }
    } else {
        return Err(LiftUnsuccessful);
    }
}

/// Computes the power decomposition of monic polynomials `f, g in Z[X]`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_power_decomposition_integer_monic<P>(ZZX: P, poly: &El<P>, attempts: usize) -> Vec<(El<P>, usize)>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    assert!(!ZZX.is_zero(poly));
    let ZZ = ZZX.base_ring();
    assert!(ZZ.is_one(ZZX.lc(poly).unwrap()));

    let power_decompositions_modulo_p = prime_fields_for_local_computation().take(attempts).map(|Fp| {
        let FpX = DensePolyRing::new(Fp, "X");
        let ZZX_to_FpX = FpX.lifted_hom(&ZZX, Fp.can_hom(ZZ).unwrap());
        let poly = ZZX_to_FpX.map_ref(poly);
        let power_decomposition = PolyTFracGCDRing::power_decomposition(&FpX, &poly);
        return (
            PolyPowerDecompositionSignature::from_decomposition(&FpX, &power_decomposition),
            (FpX, power_decomposition),
        );
    });
    poly_power_decomposition_from_quotients(
        power_decompositions_modulo_p,
        |(FpX, power_decomposition)| {
            let prime = int_cast(*FpX.base_ring().modulus(), ZZbig, ZZi64);
            let new_base_ring = ZnGB::new(ZZbig, prime.clone());
            let new_poly_ring = DensePolyRing::new(new_base_ring.clone(), "X");
            let hom = ZnReductionMap::new(FpX.base_ring(), new_base_ring).unwrap();
            let exponents = power_decomposition.iter().map(|(_, i)| *i).collect::<Vec<_>>();
            let factorization = power_decomposition
                .into_iter()
                .map(|(f, i)| FpX.pow(f, i))
                .collect::<Vec<_>>();
            Ok((
                exponents,
                HenselLift::new(FpX.clone(), factorization)
                    .unwrap()
                    .change_ring(new_poly_ring, hom),
                prime,
            ))
        },
        |(exponents, lifter, prime), lift_to_degree| {
            lift_power_decomposition(&ZZX, lifter, exponents, poly, prime, lift_to_degree)
        },
    )
    .unwrap()
}

/// Computes the power decomposition of polynomials `f, g in Z[X]` over the integers.
///
/// Use this when implementing [`PolyTFracGCDRing`] for integer
/// rings; Otherwise, compute power decompositions through
/// [`PolyTFracGCDRing::poly_power_decomposition()`].
///
/// [`PolyTFracGCDRing`]: crate::algorithms::poly_gcd::PolyTFracGCDRing
/// [`PolyTFracGCDRing::poly_power_decomposition()`]: crate::algorithms::poly_gcd::PolyTFracGCDRing::poly_power_decomposition()
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_power_decomposition_integer<P>(ZZX: P, poly: &El<P>) -> Vec<(El<P>, usize)>
where
    P: RingStore,
    P::Ring: PolyRing,
    BaseRingBase<P>: IntegerRing,
{
    assert!(!ZZX.is_zero(poly));
    let lc_poly = ZZX.lc(poly).unwrap().clone();
    let (monic_poly, lc_poly) = make_monic(&ZZX, poly, &lc_poly);
    let power_decomposition = poly_power_decomposition_integer_monic(&ZZX, &monic_poly, PROBABILISTIC_REPETITIONS);
    return power_decomposition
        .into_iter()
        .map(|(f, i)| (unmake_monic(&ZZX, &f, &lc_poly, &lc_poly), i))
        .collect();
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;

#[test]
fn test_poly_power_decomposition_integer() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = ZZbig;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let [f1, f2, f3, f4] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X - 1,
            X + 1,
            X.pow_ref(3) + X + 100,
            X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1,
        ]
    });
    let multiply_out =
        |list: &[(El<DensePolyRing<_>>, usize)]| poly_ring.prod(list.iter().map(|(g, k)| poly_ring.pow(g.clone(), *k)));
    let assert_eq = |expected: &[(El<DensePolyRing<_>>, usize)], actual: &[(El<DensePolyRing<_>>, usize)]| {
        assert!(expected.is_sorted_by_key(|(_, k)| *k));
        assert!(actual.is_sorted_by_key(|(_, k)| *k));
        assert_eq!(expected.len(), actual.len());
        for ((f_expected, k_expected), (f_actual, k_actual)) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(&poly_ring, f_expected, f_actual);
            assert_eq!(k_expected, k_actual);
        }
    };

    let expected = [(f1.clone(), 1)];
    let actual =
        poly_power_decomposition_integer_monic(&poly_ring, &multiply_out(&expected), PROBABILISTIC_REPETITIONS);
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.mul_ref(&f3, &f4), 3)];
    let actual =
        poly_power_decomposition_integer_monic(&poly_ring, &multiply_out(&expected), PROBABILISTIC_REPETITIONS);
    assert_eq(&expected, &actual);

    let expected = [(f2.clone(), 2), (poly_ring.mul_ref(&f3, &f4), 3)];
    let actual =
        poly_power_decomposition_integer_monic(&poly_ring, &multiply_out(&expected), PROBABILISTIC_REPETITIONS);
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.mul_ref(&f1, &f2), 1), (f4.clone(), 2), (f3.clone(), 3)];
    let actual =
        poly_power_decomposition_integer_monic(&poly_ring, &multiply_out(&expected), PROBABILISTIC_REPETITIONS);
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.mul_ref(&f1, &f2), 2), (f4.clone(), 4), (f3.clone(), 6)];
    let actual =
        poly_power_decomposition_integer_monic(&poly_ring, &multiply_out(&expected), PROBABILISTIC_REPETITIONS);
    assert_eq(&expected, &actual);
}

#[test]
fn random_test_poly_power_decomposition_integer() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = ZZbig;
    let poly_ring = DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(0);
    let bound = ring.int_hom().map(500);
    let mut random_poly_of_deg =
        |deg: usize| poly_ring.from_terms((0..=deg).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = random_poly_of_deg(5);
        let g = random_poly_of_deg(3);
        let h = random_poly_of_deg(2);
        let poly = make_primitive(
            &poly_ring,
            &poly_ring.prod([&f, &g, &g, &h, &h, &h].into_iter().map(|poly| poly.clone())),
        )
        .0;

        let mut power_decomp = poly_power_decomposition_integer(&poly_ring, &poly);
        for (f, _k) in &mut power_decomp {
            *f = make_primitive(&poly_ring, &f).0;
        }

        assert_el_eq!(
            &poly_ring,
            &poly,
            poly_ring.prod(power_decomp.iter().map(|(poly, k)| poly_ring.pow(poly.clone(), *k)))
        );
        assert!(
            poly_ring.divides(
                &poly_ring.prod(
                    power_decomp
                        .iter()
                        .filter(|(_, k)| k % 3 == 0)
                        .map(|(poly, k)| poly_ring.pow(poly.clone(), k / 3))
                ),
                &make_primitive(&poly_ring, &h).0
            )
        );
        assert!(
            poly_ring.divides(
                &poly_ring.prod(
                    power_decomp
                        .iter()
                        .filter(|(_, k)| k % 2 == 0)
                        .map(|(poly, k)| poly_ring.pow(poly.clone(), k / 2))
                ),
                &make_primitive(&poly_ring, &g).0
            )
        );
    }
}

#[test]
fn test_poly_gcd_integer() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = ZZbig;
    let poly_ring = DensePolyRing::new(ring, "X");
    let irred_polys = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X - 1,
            2 * X + 1,
            X.pow_ref(2) + X + 1,
            3 * X.pow_ref(3) + X + 100,
            6 * X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1,
        ]
    });
    let poly = |powers: [usize; 5]| {
        poly_ring.prod(
            powers
                .iter()
                .zip(irred_polys.iter())
                .map(|(e, f)| poly_ring.pow(f.clone(), *e)),
        )
    };

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 0, 0]),
        poly_gcd_integer(&poly_ring, &poly([1, 0, 1, 0, 0]), &poly([1, 0, 0, 1, 0]))
    );
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 1, 0, 1]),
        poly_gcd_integer(&poly_ring, &poly([1, 1, 1, 0, 1]), &poly([1, 0, 1, 1, 1]))
    );
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 2, 0, 1]),
        poly_gcd_integer(&poly_ring, &poly([1, 1, 3, 0, 1]), &poly([3, 0, 2, 0, 3]))
    );
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 5, 0],),
        poly_gcd_integer(&poly_ring, &poly([2, 1, 3, 5, 1]), &poly([1, 0, 0, 7, 0]))
    );
}

#[test]
fn random_test_poly_gcd_integer() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = ZZbig;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(0);
    let bound = ring.int_hom().map(500);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = poly_ring.from_terms((0..=16).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let g = poly_ring.from_terms((0..=14).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let h = poly_ring.from_terms((0..=12).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let lhs = poly_ring.mul_ref(&f, &h);
        let rhs = poly_ring.mul_ref(&g, &h);
        let gcd = make_primitive(&poly_ring, &poly_gcd_integer(&poly_ring, &lhs, &rhs)).0;

        assert!(poly_ring.divides(&lhs, &gcd));
        assert!(poly_ring.divides(&rhs, &gcd));
        assert!(poly_ring.divides(&gcd, &make_primitive(&poly_ring, &h).0));
    }
}
