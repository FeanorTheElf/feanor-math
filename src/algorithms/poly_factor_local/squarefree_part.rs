use dense_poly::DensePolyRing;

use crate::algorithms::poly_factor::integer::poly_power_decomposition_global;
use crate::algorithms::poly_factor_local::hensel::hensel_lift_factorization;
use crate::algorithms::poly_factor_local::poly_root;
use crate::algorithms::poly_factor_local::IntermediateReductionMap;
use crate::homomorphism::*;
use crate::ring::*;
use crate::rings::poly::*;
use crate::divisibility::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::balance_poly;
use super::evaluate_aX;
use super::unevaluate_aX;
use super::FactorPolyLocallyDomain;

///
/// For a polynomial `f in R[X]`, computes squarefree monic polynomials `fi` such that `a f = f1 f2^2 f3^3 ...`
/// for some nonzero element `a in R \ {0}`. These polynomials are returned as tuples `(fi, i)` with `fi != 0`.
/// 
/// The results can all be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_monic_local<P>(poly_ring: P, f: &El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(&f).unwrap()));
    let ring = poly_ring.base_ring().get_ring();
    let mut rng = oorandom::Rand64::new(1);
    let scale_to_ring_factor = ring.factor_scaling();
    let poly_l2_pseudo_norm_ln = poly_ring.terms(&f).map(|(c, _)| ring.ln_pseudo_norm(c)).max_by(f64::total_cmp).unwrap() + (poly_ring.degree(&f).unwrap() as f64).ln();
    let ln_bound = ring.ln_factor_coeff_bound(poly_l2_pseudo_norm_ln, poly_ring.degree(&f).unwrap()) + ring.ln_pseudo_norm(&scale_to_ring_factor);

    'try_random_prime: for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let prime = ring.random_maximal_ideal(|| rng.rand_u64());
        let e = ring.required_power(&prime, ln_bound);
        let reduction_map = IntermediateReductionMap::new(ring, &prime, e, 1);

        let prime_field = ring.local_field_at(&prime);
        let prime_field_poly_ring = DensePolyRing::new(&prime_field, "X");
        let prime_ring = reduction_map.codomain();
        let iso = prime_field.can_iso(&prime_ring).unwrap();

        let prime_field_f = prime_field_poly_ring.from_terms(poly_ring.terms(&f).map(|(c, i)| (iso.inv().map(ring.reduce_full(&prime, (&prime_ring, 1), ring.clone_el(c))), i)));
        let mut powers = Vec::new();
        let mut factors = Vec::new();
        for (f, k) in poly_power_decomposition_global(&prime_field_poly_ring, &prime_field_f) {
            powers.push(k);
            factors.push(prime_field_poly_ring.pow(f, k));
        }
    
        let target_poly_ring = DensePolyRing::new(reduction_map.domain(), "X");
        let target_ring_scale_to_ring_factor = ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(&scale_to_ring_factor));
        let local_ring_f = target_poly_ring.from_terms(poly_ring.terms(&f).map(|(c, i)| (ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(c)), i)));
        
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
                result.push((balance_poly(poly_ring, root_of_factor).0, k));
            } else {
                continue 'try_random_prime;
            }
        }
        debug_assert!(poly_ring.eq_el(&f, &poly_ring.prod(result.iter().map(|(factor, k)| poly_ring.pow(poly_ring.clone_el(factor), *k)))));
        result.sort_unstable_by_key(|(_, k)| *k);
        return result;
    }
    unreachable!()
}

///
/// For a polynomial `f in R[X]`, computes squarefree polynomials `fi` such that `a f = f1 f2^2 f3^3 ...`
/// for some nonzero ring element `a in R \ {0}`. These polynomials are returned as tuples `(fi, i)` with `fi != 0`.
/// 
/// The results can all be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_local<P>(poly_ring: P, f: El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain + DivisibilityRing
{
    assert!(!poly_ring.is_zero(&f));
    let f = balance_poly(poly_ring, f).0;
    let lcf = poly_ring.lc(&f).unwrap();
    let f_monic = evaluate_aX(poly_ring, &f, lcf);
    let power_decomposition = poly_power_decomposition_monic_local(poly_ring, &f_monic);
    let result = power_decomposition.into_iter().map(|(fi, i)| {
        (balance_poly(poly_ring, unevaluate_aX(poly_ring, &fi, &lcf)).0, i)
    }).collect::<Vec<_>>();
    debug_assert!(poly_ring.checked_div(&poly_ring.prod(result.iter().map(|(fi, i)| poly_ring.pow(poly_ring.clone_el(fi), *i))), &f).is_some());
    debug_assert_eq!(poly_ring.degree(&f).unwrap(), result.iter().map(|(fi, i)| *i * poly_ring.degree(fi).unwrap()).sum::<usize>());
    return result;
}

///
/// Computes the square-free part of a polynomial `f in R[X]`, up to multiplication by `R \ {0}`.
/// 
/// More concretely, returns the largest square-free polynomial `g` for which there is `a in R \ {0}`
/// with `g | af`.
/// 
/// The result can be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_local<P>(poly_ring: P, f: El<P>) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain + DivisibilityRing
{
    assert!(!poly_ring.is_zero(&f));
    balance_poly(poly_ring, poly_ring.prod(poly_power_decomposition_local(poly_ring, f).into_iter().map(|(fi, _i)| fi))).0
}

#[cfg(test)]
use crate::integer::*;

#[test]
fn test_squarefree_part_local() {
    let ring = BigIntRing::RING;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let [f1, f2, f3, f4] = poly_ring.with_wrapped_indeterminate(|X| [
        X - 1,
        X + 1,
        X.pow_ref(3) + X + 100,
        X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1
    ]);
    let multiply_out = |list: &[(El<DensePolyRing<_>>, usize)]| poly_ring.prod(list.iter().map(|(g, k)| poly_ring.pow(poly_ring.clone_el(g), *k)));
    let assert_eq = |expected: &[(El<DensePolyRing<_>>, usize)], actual: &[(El<DensePolyRing<_>>, usize)]| {
        assert!(expected.is_sorted_by_key(|(_, k)| *k));
        assert!(actual.is_sorted_by_key(|(_, k)| *k));
        assert_eq!(expected.len(), actual.len());
        for ((f_expected, k_expected), (f_actual, k_actual)) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(&poly_ring, f_expected, f_actual);
            assert_eq!(k_expected, k_actual);
        }
    };

    let expected = [(poly_ring.clone_el(&f1), 1)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.mul_ref(&f3, &f4), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.clone_el(&f2), 2), (poly_ring.mul_ref(&f3, &f4), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);
    
    let expected = [(poly_ring.mul_ref(&f1, &f2), 1), (poly_ring.clone_el(&f4), 2), (poly_ring.clone_el(&f3), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);
    
    let expected = [(poly_ring.mul_ref(&f1, &f2), 2), (poly_ring.clone_el(&f4), 4), (poly_ring.clone_el(&f3), 6)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected));
    assert_eq(&expected, &actual);
}