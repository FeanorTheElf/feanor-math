use dense_poly::DensePolyRing;

use crate::algorithms::poly_factor_local::balance_poly;
use crate::algorithms::poly_factor_local::hensel::hensel_lift;
use crate::algorithms::poly_factor_local::squarefree_part::poly_power_decomposition_monic_local;
use crate::algorithms::poly_factor_local::IntermediateReductionMap;
use crate::homomorphism::*;
use crate::ring::*;
use crate::rings::poly::*;
use crate::pid::*;
use crate::divisibility::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::unevaluate_aX;
use super::FactorPolyLocallyDomain;

const HOPE_FOR_SQUAREFREE_TRIES: usize = 3;

///
/// Tries to compute the gcd of monic polynomials `f, g in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`. This function is likely to succeed if either `d, f/d` or `d, g/d`
/// are coprime (over `Frac(R)`), but will never succeed if neither is the case.
/// 
/// More precisely, computes some `d in R[X]` of maximal degree with the property that there exists 
/// `a in R \ {0}` such that `d | af, ag`.
///
fn poly_gcd_monic_coprime_local<P, F>(poly_ring: P, f: &El<P>, g: &El<P>, rng: F) -> Option<El<P>>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain,
        F: FnMut() -> u64
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(g).unwrap()));

    let ring = poly_ring.base_ring().get_ring();
    let scale_to_ring_factor = ring.factor_scaling();
    let bound = ring.factor_coeff_bound(poly_ring.terms(f).map(|(c, _)| ring.pseudo_norm(c).abs().powi(2)).sum::<f64>().sqrt(), poly_ring.degree(f).unwrap()) * ring.pseudo_norm(&scale_to_ring_factor);

    let prime = ring.random_maximal_ideal(rng);
    let e = ring.required_power(&prime, bound);
    let reduction_map = IntermediateReductionMap::new(ring, &prime, e, 1);

    let prime_field = ring.local_field_at(&prime);
    let prime_field_poly_ring = DensePolyRing::new(&prime_field, "X");
    let prime_ring = reduction_map.codomain();
    let iso = prime_field.can_iso(&prime_ring).unwrap();
    let reduce_prime_field = |h| prime_field_poly_ring.from_terms(poly_ring.terms(h).map(|(c, i)| (iso.inv().map(ring.reduce_full(&prime, (&prime_ring, 1), ring.clone_el(c))), i)));

    let prime_field_f = reduce_prime_field(f);
    let prime_field_g = reduce_prime_field(g);
    let mut prime_field_d = prime_field_poly_ring.ideal_gen(&prime_field_f, &prime_field_g);
    prime_field_d = prime_field_poly_ring.normalize(prime_field_d);

    let prime_field_f_over_d = prime_field_poly_ring.checked_div(&prime_field_f, &prime_field_d).unwrap();
    let prime_field_g_over_d = prime_field_poly_ring.checked_div(&prime_field_g, &prime_field_d).unwrap();
    let (poly, factor1, factor2) = if prime_field_poly_ring.is_unit(&prime_field_poly_ring.ideal_gen(&prime_field_d, &prime_field_f_over_d)) {
        (f, prime_field_d, prime_field_f_over_d)
    } else if prime_field_poly_ring.is_unit(&prime_field_poly_ring.ideal_gen(&prime_field_d, &prime_field_g_over_d)) {
        (g, prime_field_d, prime_field_g_over_d)
    } else {
        return None;
    };
    let target_poly_ring = DensePolyRing::new(reduction_map.domain(), "X");
    let reduced_poly = target_poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| (ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), ring.clone_el(c)), i)));

    let (lifted_d, _lifted_other_factor) = hensel_lift(&reduction_map, &target_poly_ring, &prime_field_poly_ring, &reduced_poly, (&factor1, &factor2));
    let target_ring_scale_to_ring_factor = ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(&scale_to_ring_factor));

    let result = poly_ring.from_terms(target_poly_ring.terms(&lifted_d).map(|(c, i)| (
        ring.lift_full(&prime, (reduction_map.domain(), reduction_map.from_e()), reduction_map.domain().mul_ref(c, &target_ring_scale_to_ring_factor)), 
        i
    )));

    if poly_ring.checked_div(&poly_ring.inclusion().mul_ref_map(&f, &scale_to_ring_factor), &result).is_none() || 
        poly_ring.checked_div(&poly_ring.inclusion().mul_ref_map(&g, &scale_to_ring_factor), &result).is_none()
    {
        return None;
    } else {
        return Some(balance_poly(poly_ring, &result).0);
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
fn poly_gcd_coprime_local<P, F>(poly_ring: P, f: &El<P>, g: &El<P>, rng: F) -> Option<El<P>>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain,
        F: FnMut() -> u64
{
    if poly_ring.is_zero(f) {
        return Some(poly_ring.clone_el(g));
    } else if poly_ring.is_zero(g) {
        return Some(poly_ring.clone_el(f));
    }
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(poly_ring.lc(f).unwrap(), poly_ring.lc(g).unwrap());
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_ref_map(f, poly_ring.lc(g).unwrap()), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_ref_map(g, poly_ring.lc(f).unwrap()), &a);
    let d_monic = poly_gcd_monic_coprime_local(poly_ring, &f_monic, &g_monic, rng)?;

    return Some(balance_poly(poly_ring, &unevaluate_aX(poly_ring, &d_monic, &a)).0);
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
pub fn poly_gcd_monic_local<'a, P>(poly_ring: P, mut f: &'a El<P>, mut g: &'a El<P>) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(g).unwrap()));

    let mut rng = oorandom::Rand64::new(1);
    for _ in 0..HOPE_FOR_SQUAREFREE_TRIES {
        if let Some(result) = poly_gcd_monic_coprime_local(poly_ring, f, g, || rng.rand_u64()) {
            return result;
        }
    }
    if poly_ring.degree(g).unwrap_or(0) <= poly_ring.degree(f).unwrap_or(0) {
        std::mem::swap(&mut f, &mut g);
    }
    let f_power_decomposition = poly_power_decomposition_monic_local(poly_ring, f);
    let mut g = poly_ring.clone_el(g);
    let mut d = poly_ring.one();
    'extract_part_i: for i in 1.. {
        let squarefree_part_i = poly_ring.prod(f_power_decomposition.iter().filter(|(_, j)| *j >= i).map(|(fj, _)| poly_ring.clone_el(fj)));
        if poly_ring.is_one(&squarefree_part_i) {
            return d;
        }
        for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
            if let Some(di) = poly_gcd_coprime_local(poly_ring, &squarefree_part_i, &g, || rng.rand_u64()) {
                g = poly_ring.checked_div(&g, &di).unwrap();
                poly_ring.mul_assign(&mut d, di);
                continue 'extract_part_i;
            }
        }
        unreachable!()
    }
    return balance_poly(poly_ring, &d).0;
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
pub fn poly_gcd_local< P>(poly_ring: P, f: &El<P>, g: &El<P>) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain
{
    if poly_ring.is_zero(f) {
        return poly_ring.clone_el(g);
    } else if poly_ring.is_zero(g) {
        return poly_ring.clone_el(f);
    }
    let ring = poly_ring.base_ring();
    let a = ring.mul_ref(poly_ring.lc(f).unwrap(), poly_ring.lc(g).unwrap());
    let f_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_ref_map(f, poly_ring.lc(g).unwrap()), &a);
    let g_monic = evaluate_aX(poly_ring, &poly_ring.inclusion().mul_ref_map(g, poly_ring.lc(f).unwrap()), &a);
    let d_monic = poly_gcd_monic_local(poly_ring, &f_monic, &g_monic);

    return balance_poly(poly_ring, &unevaluate_aX(poly_ring, &d_monic, &a)).0;
}

#[cfg(test)]
use crate::integer::*;

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
        poly_gcd_local(&poly_ring, &poly([1, 0, 1, 0, 0], 1), &poly([1, 0, 0, 1, 0], 2))
    );
    
    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 1, 0, 1], 1),
        poly_gcd_local(&poly_ring, &poly([1, 1, 1, 0, 1], 20), &poly([1, 0, 1, 1, 1], 12))
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 2, 0, 1], 1),
        poly_gcd_local(&poly_ring, &poly([1, 1, 3, 0, 1], 20), &poly([3, 0, 2, 0, 3], 12))
    );

    assert_el_eq!(
        &poly_ring,
        poly([1, 0, 0, 5, 0], 1),
        poly_gcd_local(&poly_ring, &poly([2, 1, 3, 5, 1], 20), &poly([1, 0, 0, 7, 0], 12))
    );
}