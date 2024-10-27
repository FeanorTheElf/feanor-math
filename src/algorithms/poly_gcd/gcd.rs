use dense_poly::DensePolyRing;
use squarefree_part::poly_power_decomposition_monic_local;

use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_gcd::local::*;
use crate::algorithms::poly_gcd::hensel::*;
use crate::computation::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::unevaluate_aX;

const HOPE_FOR_SQUAREFREE_TRIES: usize = 3;

///
/// Tries to compute the gcd of monic polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`. This function will fail in two cases
///  - both `d, f/d` and `d, g/d` are not coprime
///  - the gcd of `f, g` cannot be read of from its reduction modulo `p^e`, where `e` is (as usual)
///    the result of the "heuristic exponent", scaled exponentially by the current attempt
/// If neither is the case, this function is likely to succeed
/// 
/// More precisely, computes some `d in R[X]` of maximal degree with the property that there exists 
/// `a in R \ {0}` such that `d | af, ag`.
///
fn poly_gcd_monic_coprime_local<P, F, Controller>(poly_ring: P, f: &El<P>, g: &El<P>, rng: F, current_attempt: usize, controller: Controller) -> Option<El<P>>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyGCDLocallyDomain,
        F: FnMut() -> u64,
        Controller: ComputationController
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(g).unwrap()));

    let ring = poly_ring.base_ring().get_ring();

    let prime = ring.random_maximal_ideal(rng);
    let heuristic_e = ring.heuristic_exponent(&prime, poly_ring.degree(f).unwrap(), poly_ring.terms(f).map(|(c, _)| c));
    assert!(heuristic_e >= 1);
    let e = (heuristic_e as f64 * INCREASE_EXPONENT_PER_ATTEMPT_CONSTANT.powi(current_attempt as i32)).floor() as usize;

    log_progress!(controller, "mod({}^{})", IdealDisplayWrapper::new(ring, &prime), e);

    let reduction_map = IntermediateReductionMap::new(ring, &prime, e, 1);

    let prime_field = ring.local_field_at(&prime);
    let prime_field_poly_ring = DensePolyRing::new(&prime_field, "X");
    let prime_ring = reduction_map.codomain();
    let iso = prime_field.can_iso(&prime_ring).unwrap();
    let reduce_prime_field = |h| prime_field_poly_ring.from_terms(poly_ring.terms(h).map(|(c, i)| (iso.inv().map(ring.reduce_ring_el(&prime, (&prime_ring, 1), ring.clone_el(c))), i)));

    let prime_field_f = reduce_prime_field(f);
    let prime_field_g = reduce_prime_field(g);
    let mut prime_field_d = prime_field_poly_ring.ideal_gen(&prime_field_f, &prime_field_g);
    prime_field_d = prime_field_poly_ring.normalize(prime_field_d);

    log_progress!(controller, "d({})", prime_field_poly_ring.degree(&prime_field_d).unwrap());

    let prime_field_f_over_d = prime_field_poly_ring.checked_div(&prime_field_f, &prime_field_d).unwrap();
    let prime_field_g_over_d = prime_field_poly_ring.checked_div(&prime_field_g, &prime_field_d).unwrap();
    let (poly, factor1, factor2) = if prime_field_poly_ring.is_unit(&prime_field_poly_ring.ideal_gen(&prime_field_d, &prime_field_f_over_d)) {
        (f, prime_field_d, prime_field_f_over_d)
    } else if prime_field_poly_ring.is_unit(&prime_field_poly_ring.ideal_gen(&prime_field_d, &prime_field_g_over_d)) {
        (g, prime_field_d, prime_field_g_over_d)
    } else {
        log_progress!(controller, "(not_coprime)");
        return None;
    };
    let target_poly_ring = DensePolyRing::new(reduction_map.domain(), "X");
    let reduced_poly = target_poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| (ring.reduce_ring_el(&prime, (reduction_map.domain(), reduction_map.from_e()), ring.clone_el(c)), i)));

    let (lifted_d, _lifted_other_factor) = hensel_lift(&reduction_map, &target_poly_ring, &prime_field_poly_ring, &reduced_poly, (&factor1, &factor2), controller.clone());
    let result = poly_ring.from_terms(target_poly_ring.terms(&lifted_d).map(|(c, i)| (ring.reconstruct_ring_el(&prime, (reduction_map.domain(), reduction_map.from_e()), reduction_map.domain().clone_el(c)), i)));

    if poly_ring.checked_div(&f, &result).is_none() || 
        poly_ring.checked_div(&g, &result).is_none()
    {
        log_progress!(controller, "(no_divisor)");
        return None;
    } else {
        return Some(balance_poly(poly_ring, result).0);
    }
}

///
/// Tries to compute the gcd of polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`. This function is likely to succeed if either `d, f/d` or `d, g/d`
/// are coprime (over `Frac(R)`), but will never succeed if neither is the case.
/// 
/// More precisely, computes some `d in R[X]` of maximal degree with the property that there exists 
/// `a in R \ {0}` such that `d | af, ag`.
///
fn poly_gcd_coprime_local<P, F, Controller>(poly_ring: P, f: El<P>, g: El<P>, rng: F, attempt: usize, controller: Controller) -> Option<El<P>>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyGCDLocallyDomain,
        F: FnMut() -> u64,
        Controller: ComputationController
{
    if poly_ring.is_zero(&f) {
        return Some(poly_ring.clone_el(&g));
    } else if poly_ring.is_zero(&g) {
        return Some(poly_ring.clone_el(&f));
    }
    let f = balance_poly(poly_ring, f).0;
    let g = balance_poly(poly_ring, g).0;
    let lcf = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lcg = poly_ring.base_ring().clone_el(poly_ring.lc(&g).unwrap());
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(&lcf, &lcg);
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(f, lcg), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(g, lcf), &a);
    let d_monic = poly_gcd_monic_coprime_local(poly_ring, &f_monic, &g_monic, rng, attempt, controller)?;

    return Some(balance_poly(poly_ring, unevaluate_aX(poly_ring, &d_monic, &a)).0);
}

///
/// Computes the gcd of monic polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`.
/// 
/// More precisely, computes some `d in R[X]` of maximal degree with the property that there exists 
/// `a in R \ {0}` such that `d | af, ag`.
/// 
/// The result can be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
///
#[stability::unstable(feature = "enable")]
pub fn poly_gcd_monic_local<'a, P, Controller>(poly_ring: P, mut f: &'a El<P>, mut g: &'a El<P>, controller: Controller) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyGCDLocallyDomain,
        Controller: ComputationController
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(g).unwrap()));

    start_computation!(controller, "gcd_local({}, {})", poly_ring.degree(f).unwrap(), poly_ring.degree(g).unwrap());

    let mut rng = oorandom::Rand64::new(1);
    for attempt in 0..HOPE_FOR_SQUAREFREE_TRIES {
        if let Some(result) = poly_gcd_monic_coprime_local(poly_ring, f, g, || rng.rand_u64(), attempt, controller.clone()) {
            return result;
        }
    }
    if poly_ring.degree(g).unwrap_or(0) <= poly_ring.degree(f).unwrap_or(0) {
        std::mem::swap(&mut f, &mut g);
    }
    let f_power_decomposition = poly_power_decomposition_monic_local(poly_ring, f, controller.clone());
    let mut g = poly_ring.clone_el(g);
    let mut d = poly_ring.one();
    'extract_part_i: for i in 1.. {
        let squarefree_part_i = poly_ring.prod(f_power_decomposition.iter().filter(|(_, j)| *j >= i).map(|(fj, _)| poly_ring.clone_el(fj)));
        if poly_ring.is_one(&squarefree_part_i) {

            finish_computation!(controller);
            return d;
        }
        for attempt in 0..MAX_PROBABILISTIC_REPETITIONS {
            if let Some(di) = poly_gcd_coprime_local(poly_ring, poly_ring.clone_el(&squarefree_part_i), poly_ring.clone_el(&g), || rng.rand_u64(), attempt, controller.clone()) {
                g = poly_ring.checked_div(&g, &di).unwrap();
                poly_ring.mul_assign(&mut d, di);
                continue 'extract_part_i;
            }
        }
        unreachable!()
    }
    unreachable!()
}

///
/// Computes the gcd of polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`.
/// 
/// More precisely, computes some `d in R[X]` of maximal degree with the property that there exists 
/// `a in R \ {0}` such that `d | af, ag`.
/// 
/// The result can be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
///
#[stability::unstable(feature = "enable")]
pub fn poly_gcd_local<P, Controller>(poly_ring: P, f: El<P>, g: El<P>, controller: Controller) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyGCDLocallyDomain,
        Controller: ComputationController
{
    if poly_ring.is_zero(&f) {
        return poly_ring.clone_el(&g);
    } else if poly_ring.is_zero(&g) {
        return poly_ring.clone_el(&f);
    }
    let f = balance_poly(poly_ring, f).0;
    let g = balance_poly(poly_ring, g).0;
    let lcf = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lcg = poly_ring.base_ring().clone_el(poly_ring.lc(&g).unwrap());
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(&lcf, &lcg);
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(f, lcg), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(g, lcf), &a);
    let d_monic = poly_gcd_monic_local(poly_ring, &f_monic, &g_monic, controller);

    return balance_poly(poly_ring, unevaluate_aX(poly_ring, &d_monic, &a)).0;
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::algorithms::poly_gcd::make_primitive;

#[test]
fn test_poly_gcd_local() {
    let ring = BigIntRing::RING;
    let poly_ring = DensePolyRing::new(ring, "X");
    let irred_polys = poly_ring.with_wrapped_indeterminate(|X| [
        X - 1,
        X + 1,
        X.pow_ref(2) + X + 1,
        X.pow_ref(3) + X + 100,
        X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1
    ]);
    let poly = |powers: [usize; 5], scale: i32| poly_ring.int_hom().mul_map(poly_ring.prod(powers.iter().zip(irred_polys.iter()).map(|(e, f)| poly_ring.pow(poly_ring.clone_el(f), *e))), scale);

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 0, 0], 1),
        poly_gcd_local(&poly_ring, poly([1, 0, 1, 0, 0], 1), poly([1, 0, 0, 1, 0], 2), LogProgress)
    );
    
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 1, 0, 1], 1),
        poly_gcd_local(&poly_ring, poly([1, 1, 1, 0, 1], 20), poly([1, 0, 1, 1, 1], 12), LogProgress)
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 2, 0, 1], 1),
        poly_gcd_local(&poly_ring, poly([1, 1, 3, 0, 1], 20), poly([3, 0, 2, 0, 3], 12), LogProgress)
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 5, 0], 1),
        poly_gcd_local(&poly_ring, poly([2, 1, 3, 5, 1], 20), poly([1, 0, 0, 7, 0], 12), LogProgress)
    );
}

#[test]
fn random_test_poly_gcd_local() {
    let ring = BigIntRing::RING;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = ring.int_hom().map(10000);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = poly_ring.from_terms((0..=20).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let g = poly_ring.from_terms((0..=20).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let h = poly_ring.from_terms((0..=10).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        // println!("Testing gcd on ({}) * ({}) and ({}) * ({})", poly_ring.format(&f), poly_ring.format(&h), poly_ring.format(&g), poly_ring.format(&h));
        let lhs = poly_ring.mul_ref(&f, &h);
        let rhs = poly_ring.mul_ref(&g, &h);
        let gcd = make_primitive(&poly_ring, &poly_gcd_local(&poly_ring, poly_ring.clone_el(&lhs), poly_ring.clone_el(&rhs), LogProgress)).0;
        // println!("Result {}", poly_ring.format(&gcd));

        assert!(poly_ring.checked_div(&lhs, &gcd).is_some());
        assert!(poly_ring.checked_div(&rhs, &gcd).is_some());
        assert!(poly_ring.checked_div(&gcd, &make_primitive(&poly_ring, &h).0).is_some());
    }
}