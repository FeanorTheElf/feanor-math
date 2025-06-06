use std::time::Instant;

use dense_poly::DensePolyRing;
use squarefree_part::poly_power_decomposition_monic_local;

use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_gcd::hensel::*;
use crate::seq::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::unevaluate_aX;

const HOPE_FOR_SQUAREFREE_TRIES: usize = 3;

///
/// Describes the relationship of `f, g, gcd(f, g)` modulo a single maximal ideal
/// 
#[derive(PartialEq, Eq)]
struct Signature {
    /// the degree of `gcd(f, g) mod m`
    gcd_deg: usize,
    /// whether `f/d` is coprime to `d`, where `d = gcd(f, g) mod m`
    coprime_to_f_over_d: bool
}

///
/// Tries to compute the gcd of monic polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`. This function will fail in two cases
///  - both `d, f/d` and `d, g/d` are not coprime
///  - the gcd of `f, g` cannot be reconstructed from its reduction modulo `p^e`, where `e` is (as usual)
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

    let ideal = ring.random_suitable_ideal(rng);
    let heuristic_e = ring.heuristic_exponent(&ideal, poly_ring.degree(f).unwrap(), poly_ring.terms(f).map(|(c, _)| c));
    assert!(heuristic_e >= 1);
    let e = (heuristic_e as f64 * INCREASE_EXPONENT_PER_ATTEMPT_CONSTANT.powi(current_attempt.try_into().unwrap())).floor() as usize;
    let reduction = ReductionContext::new(ring, &ideal, e);

    log_progress!(controller, "(mod={}^{})(parts={})", IdealDisplayWrapper::new(ring, &ideal), e, reduction.len());

    let mut signature: Option<Signature> = None;
    let mut poly_rings_mod_me = Vec::new();
    let mut gcds_mod_me = Vec::new();

    let mut setup_time = 0;
    let mut local_gcd_time = 0;
    let mut hensel_lift_time = 0;
    let mut prepare_lift_time = 0;
    let complete_start = Instant::now();

    for idx in 0..reduction.len() {

        let start = Instant::now();
        let S_to_F = reduction.intermediate_ring_to_field_reduction(idx);
        let F = ring.local_field_at(S_to_F.ideal(), S_to_F.max_ideal_idx());
        let F_iso = F.can_iso(*S_to_F.codomain()).unwrap();
        let R_to_F = F_iso.inv().compose(reduction.main_ring_to_field_reduction(idx));
        let FX = DensePolyRing::new(&F, "X");
        let RX_to_FX = FX.lifted_hom(poly_ring, &R_to_F);
        let end = Instant::now();
        setup_time += (end - start).as_millis();

        let start = Instant::now();
        let d = FX.normalize(FX.ideal_gen(&RX_to_FX.map_ref(f), &RX_to_FX.map_ref(g)));
        let end = Instant::now();
        local_gcd_time += (end - start).as_millis();

        let start = Instant::now();
        let f_over_d = FX.checked_div(&RX_to_FX.map_ref(f), &d).unwrap();
        let g_over_d = FX.checked_div(&RX_to_FX.map_ref(g), &d).unwrap();
        let deg_d = FX.degree(&d).unwrap();
        let (poly, factor1, factor2, new_signature) = if FX.is_unit(&FX.ideal_gen(&d, &f_over_d)) {
            (f, d, f_over_d, Signature { gcd_deg: deg_d, coprime_to_f_over_d: true })
        } else if FX.is_unit(&FX.ideal_gen(&d, &g_over_d)) {
            (g, d, g_over_d, Signature { gcd_deg: deg_d, coprime_to_f_over_d: false })
        } else {
            log_progress!(controller, "(not_coprime)");
            return None;
        };
        if signature.is_some() && signature.as_ref().unwrap() != &new_signature {
            log_progress!(controller, "(signature_mismatch)");
            return None;
        }
        signature = Some(new_signature);
        let SX = DensePolyRing::new(*S_to_F.domain(), "X");
        let RX_to_SX = SX.lifted_hom(poly_ring, reduction.main_ring_to_intermediate_ring_reduction(idx));
        let end = Instant::now();
        prepare_lift_time += (end - start).as_millis();

        let start = Instant::now();
        let (d, _) = hensel_lift_quadratic(&S_to_F, &SX, &FX, &RX_to_SX.map_ref(poly), (&factor1, &factor2), controller.clone());
        let end = Instant::now();
        hensel_lift_time += (end - start).as_millis();

        poly_rings_mod_me.push(SX);
        gcds_mod_me.push(d);
    }

    let start = Instant::now();
    let signature = signature.unwrap();
    let mut result = poly_ring.from_terms((0..=signature.gcd_deg).map(|i| (
        reduction.reconstruct_ring_el((0..reduction.len()).map_fn(|j| poly_rings_mod_me[j].coefficient_at(&gcds_mod_me[j], i))),
        i
    )));
    let end = Instant::now();
    let reconstruct_time = (end - start).as_millis();

    let start = Instant::now();
    let divides_f_and_g = poly_ring.divides(&f, &result) && poly_ring.divides(&g, &result);
    let end = Instant::now();
    let test_division_time = (end - start).as_millis();
    if !divides_f_and_g {
        let complete_end = Instant::now();
        log_progress!(controller, "(setup: {} ms, localgcd: {} ms, preparelift: {} ms, hensel: {} ms, reconstruct: {} ms, test: {} ms, all: {} ms)", setup_time, local_gcd_time, prepare_lift_time, hensel_lift_time, reconstruct_time, test_division_time, (complete_end - complete_start).as_millis());
        log_progress!(controller, "(no_divisor)");
        return None;
    } else {
        let complete_end = Instant::now();
        log_progress!(controller, "(setup: {} ms, localgcd: {} ms, preparelift: {} ms, hensel: {} ms, reconstruct: {} ms, test: {} ms, all: {} ms)", setup_time, local_gcd_time, prepare_lift_time, hensel_lift_time, reconstruct_time, test_division_time, (complete_end - complete_start).as_millis());
        _ = poly_ring.balance_poly(&mut result);
        return Some(result);
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
fn poly_gcd_coprime_local<P, F, Controller>(poly_ring: P, mut f: El<P>, mut g: El<P>, rng: F, attempt: usize, controller: Controller) -> Option<El<P>>
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
    _ = poly_ring.balance_poly(&mut f);
    _ = poly_ring.balance_poly(&mut g);
    let lcf = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lcg = poly_ring.base_ring().clone_el(poly_ring.lc(&g).unwrap());
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(&lcf, &lcg);
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(f, lcg), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(g, lcf), &a);

    let d_monic = poly_gcd_monic_coprime_local(poly_ring, &f_monic, &g_monic, rng, attempt, controller)?;

    let mut result = unevaluate_aX(poly_ring, &d_monic, &a);
    _ = poly_ring.balance_poly(&mut result);
    return Some(result);
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

    controller.run_computation(format_args!("gcd_local(deg_l={}, deg_r={})", poly_ring.degree(f).unwrap(), poly_ring.degree(g).unwrap()), |controller| {

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
    })
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
pub fn poly_gcd_local<P, Controller>(poly_ring: P, mut f: El<P>, mut g: El<P>, controller: Controller) -> El<P>
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
    _ = poly_ring.balance_poly(&mut f);
    _ = poly_ring.balance_poly(&mut g);
    let lcf = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lcg = poly_ring.base_ring().clone_el(poly_ring.lc(&g).unwrap());
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(&lcf, &lcg);
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(f, lcg), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_map(g, lcf), &a);

    let d_monic = poly_gcd_monic_local(poly_ring, &f_monic, &g_monic, controller);

    let mut result = unevaluate_aX(poly_ring, &d_monic, &a);
    _ = poly_ring.balance_poly(&mut result);
    return result;
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
        poly_gcd_local(&poly_ring, poly([1, 0, 1, 0, 0], 1), poly([1, 0, 0, 1, 0], 2), LOG_PROGRESS)
    );
    
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 1, 0, 1], 1),
        poly_gcd_local(&poly_ring, poly([1, 1, 1, 0, 1], 20), poly([1, 0, 1, 1, 1], 12), LOG_PROGRESS)
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 2, 0, 1], 1),
        poly_gcd_local(&poly_ring, poly([1, 1, 3, 0, 1], 20), poly([3, 0, 2, 0, 3], 12), LOG_PROGRESS)
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 5, 0], 1),
        poly_gcd_local(&poly_ring, poly([2, 1, 3, 5, 1], 20), poly([1, 0, 0, 7, 0], 12), LOG_PROGRESS)
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
        let gcd = make_primitive(&poly_ring, &poly_gcd_local(&poly_ring, poly_ring.clone_el(&lhs), poly_ring.clone_el(&rhs), LOG_PROGRESS)).0;
        // println!("Result {}", poly_ring.format(&gcd));

        assert!(poly_ring.divides(&lhs, &gcd));
        assert!(poly_ring.divides(&rhs, &gcd));
        assert!(poly_ring.divides(&gcd, &make_primitive(&poly_ring, &h).0));
    }
}