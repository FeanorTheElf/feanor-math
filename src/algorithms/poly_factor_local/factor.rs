use crate::algorithms::poly_factor_local::balance_poly;
use crate::algorithms::poly_factor_local::hensel::hensel_lift_factorization;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::rings::poly::dense_poly::*;
use crate::algorithms::poly_factor_local::poly_root;
use crate::divisibility::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::unevaluate_aX;
use super::{FactorPolyLocallyDomain, IntermediateReductionMap};

fn factor_poly_monic_local<P>(poly_ring: P, f: &El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    let mut rng = oorandom::Rand64::new(1);
    let ring = poly_ring.base_ring().get_ring();
    let scale_to_ring_factor = ring.factor_scaling();
    let bound = ring.factor_coeff_bound(poly_ring.terms(f).map(|(c, _)| ring.pseudo_norm(c).abs().powi(2)).sum::<f64>().sqrt(), poly_ring.degree(f).unwrap()) * ring.pseudo_norm(&scale_to_ring_factor);

    'try_random_prime: for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let prime = ring.random_maximal_ideal(|| rng.rand_u64());
        let e = ring.required_power(&prime, bound);
        let reduction_map = IntermediateReductionMap::new(ring, &prime, e, 1);

        let prime_field = ring.local_field_at(&prime);
        let prime_field_poly_ring = DensePolyRing::new(&prime_field, "X");
        let prime_ring = reduction_map.codomain();
        let iso = prime_field.can_iso(&prime_ring).unwrap();

        let prime_field_f = prime_field_poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (iso.inv().map(ring.reduce_full(&prime, (&prime_ring, 1), ring.clone_el(c))), i)));
        let mut powers = Vec::new();
        let mut factors = Vec::new();
        for (f, k) in <_ as FactorPolyField>::factor_poly(&prime_field_poly_ring, &prime_field_f).0 {
            powers.push(k);
            factors.push(prime_field_poly_ring.pow(f, k));
        }
    
        let target_poly_ring = DensePolyRing::new(reduction_map.domain(), "X");
        let target_ring_scale_to_ring_factor = ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(&scale_to_ring_factor));
        let local_ring_f = target_poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(c)), i)));
        
        let mut lifted_factorization = Vec::new();
        for (factor, k) in hensel_lift_factorization(&reduction_map, &target_poly_ring, &prime_field_poly_ring, &local_ring_f, &factors[..]).into_iter().zip(powers.iter()) {
            lifted_factorization.push(poly_ring.from_terms(target_poly_ring.terms(&factor).map(|(c, i)| (
                ring.mul(
                    ring.lift_full(&prime, (reduction_map.domain(), reduction_map.from_e()), reduction_map.domain().mul_ref(c, &target_ring_scale_to_ring_factor)), 
                    // to map the factor into the ring, it would be sufficient to multiply by `scale_to_ring_factor`;
                    // however, in order to take the root later, we need it to be a perfect `i`-th power, thus `scale_to_ring_factor^i`
                    poly_ring.base_ring().pow(ring.clone_el(&scale_to_ring_factor), k - 1)
                ),
                i
            ))));
        }
    
        let mut result = Vec::new();
        for (k, factor) in powers.into_iter().zip(lifted_factorization.into_iter()) {
            if let Some(root_of_factor) = poly_root(poly_ring, &factor, k) {
                result.push((balance_poly(poly_ring, &root_of_factor).0, k));
            } else {
                continue 'try_random_prime;
            }
        }
        debug_assert!(poly_ring.eq_el(&f, &poly_ring.prod(result.iter().map(|(f, k)| poly_ring.pow(poly_ring.clone_el(f), *k)))));
        result.sort_unstable_by_key(|(_, k)| *k);
        return result;
    }
    unreachable!()
}

///
/// Computes the factorization of a polynomial `f in R[X]` over `Frac(R)` locally, without ever
/// explicitly working in `Frac(R)`.
/// 
/// In other words, returns tuples `(fi, ei)` where the `fi` are irreducible polynomials, both over
/// `R` and over `Frac(R)`, such that there exists `a in R \ {0}` with `af = f1^e1 ... fr^er`.
/// 
/// The results can all be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_poly_local<P>(poly_ring: P, f: &El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain + DivisibilityRing
{
    assert!(!poly_ring.is_zero(f));
    let lcf = poly_ring.lc(f).unwrap();
    let f_monic = evaluate_aX(poly_ring, f, lcf);
    let factorization = factor_poly_monic_local(poly_ring, &f_monic);
    let result = factorization.into_iter().map(|(fi, i)| {
        (balance_poly(poly_ring, &unevaluate_aX(poly_ring, &fi, &lcf)).0, i)
    }).collect::<Vec<_>>();
    debug_assert!(poly_ring.checked_div(&poly_ring.prod(result.iter().map(|(fi, i)| poly_ring.pow(poly_ring.clone_el(fi), *i))), f).is_some());
    debug_assert_eq!(poly_ring.degree(f).unwrap(), result.iter().map(|(fi, i)| *i * poly_ring.degree(fi).unwrap()).sum::<usize>());
    return result;
}

#[cfg(test)]
use crate::integer::*;

#[test]
fn test_factor_poly_local() {
    let ring = BigIntRing::RING;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let [f1, f2, f3, f4] = poly_ring.with_wrapped_indeterminate(|X| [
        X - 1,
        X + 1,
        X.pow_ref(3) + X + 100,
        X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1
    ]);
    let multiply_out = |list: &[(El<DensePolyRing<_>>, usize)]| poly_ring.prod(list.iter().map(|(g, k)| poly_ring.pow(poly_ring.clone_el(g), *k)));
    let assert_eq = |expected: &[(El<DensePolyRing<_>>, usize)], mut actual: Vec<(El<DensePolyRing<_>>, usize)>| {
        assert_eq!(expected.len(), actual.len());
        for (f_expected, k_expected) in expected.iter() {
            let idx = actual.iter().enumerate().filter(|(_, (f, _k))| poly_ring.eq_el(f_expected, f)).next();
            assert!(idx.is_some(), "Did not find factor ({})^{} in computed factorization", poly_ring.format(f_expected), *k_expected);
            let idx = idx.unwrap().0;
            let (_f, k) = actual.swap_remove(idx);
            assert_eq!(*k_expected, k);
        }
        assert!(actual.len() == 0, "Computed factorization contained unexpected factor ({})^{}", poly_ring.format(&actual[0].0), actual[0].1);
    };
    
    let expected = [(poly_ring.clone_el(&f1), 1)];
    let actual = factor_poly_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f3), 3), (poly_ring.clone_el(&f4), 3)];
    let actual = factor_poly_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f2), 2), (poly_ring.clone_el(&f3), 3), (poly_ring.clone_el(&f4), 3)];
    let actual = factor_poly_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, actual);
    
    let expected = [(poly_ring.clone_el(&f1), 1), (poly_ring.clone_el(&f2), 1), (poly_ring.clone_el(&f4), 2), (poly_ring.clone_el(&f3), 3)];
    let actual = factor_poly_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, actual);
    
    let expected = [(poly_ring.clone_el(&f1), 2), (poly_ring.clone_el(&f3), 2), (poly_ring.clone_el(&f4), 4), (poly_ring.clone_el(&f2), 5)];
    let actual = factor_poly_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, actual);
}