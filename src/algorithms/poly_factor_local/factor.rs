use std::convert::identity;

use crate::algorithms::poly_factor_local::balance_poly;
use crate::algorithms::poly_factor_local::hensel::hensel_lift_factorization;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::iters::clone_slice;
use crate::iters::powerset;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::rings::poly::dense_poly::*;
use crate::divisibility::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::squarefree_part::poly_power_decomposition_local;
use super::unevaluate_aX;
use super::{FactorPolyLocallyDomain, IntermediateReductionMap};

fn combine_local_factors_local<'ring, 'a, R, P, Q>(ring: &R, scale_to_ring_factor: &R::Element, maximal_ideal: &R::MaximalIdeal<'ring>, poly_ring: P, f: &El<P>, local_poly_ring: Q, local_e: usize, local_factors: Vec<El<Q>>) -> Vec<El<P>>
    where R: ?Sized + FactorPolyLocallyDomain,
        P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = R>,
        Q: RingStore + Copy,
        Q::Type: PolyRing<BaseRing = &'a R::LocalRing<'ring>> + DivisibilityRing,
        R::LocalRing<'ring>: 'a
{
    debug_assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    debug_assert!(local_factors.iter().all(|local_factor| local_poly_ring.base_ring().is_one(local_poly_ring.lc(local_factor).unwrap())));

    let local_ring_scale_to_ring_factor = ring.reduce_full(&maximal_ideal, (local_poly_ring.base_ring(), local_e), poly_ring.base_ring().clone_el(&scale_to_ring_factor));
    let lift_full = |factor| balance_poly(poly_ring, poly_ring.from_terms(local_poly_ring.terms(&factor).map(|(c, i)| (ring.lift_full(maximal_ideal, (local_poly_ring.base_ring(), local_e), local_poly_ring.base_ring().clone_el(c)), i)))).0;

    let mut ungrouped_factors = (0..local_factors.len()).collect::<Vec<_>>();
    let mut current = poly_ring.inclusion().mul_ref_map(f, scale_to_ring_factor);
    let mut result = Vec::new();
    while ungrouped_factors.len() > 0 {
        // Here we use the naive approach to group the factors such that the product of each group
        // is integral - just try all combinations. It might be worth using LLL for this instead;
        // note that powerset yields smaller subsets first
        let (factor, new, factor_group) = powerset(ungrouped_factors.iter().copied(), |indices| {
            if indices.len() == 0 {
                return None;
            }
            let factor = local_poly_ring.prod(indices.iter().copied().map(|i| local_poly_ring.clone_el(&local_factors[i])));
            let liftable_factor = local_poly_ring.inclusion().mul_ref_snd_map(factor, &local_ring_scale_to_ring_factor);
            let lifted_factor = lift_full(liftable_factor);
            if let Some(quo) = poly_ring.checked_div(&current, &lifted_factor) {
                return Some((lifted_factor, quo, clone_slice(indices)));
            } else {
                return None;
            }
        }).filter_map(identity).next().unwrap();
        current = poly_ring.inclusion().mul_ref_snd_map(new, scale_to_ring_factor);
        result.push(factor);
        ungrouped_factors.retain(|j| !factor_group.contains(j));
    }
    return result;
}

///
/// Given squarefree and monic `f in R[X]`, computes irreducible polynomials `fi in R[X]` (irreducible both over `R` and
/// `Frac(R)`) such that `af = f1 ... fr` for some `a in R \ {0}`.
/// 
fn factor_poly_squarefree_monic_local<P>(poly_ring: P, f: &El<P>) -> Vec<El<P>>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
    let mut rng = oorandom::Rand64::new(1);
    let ring = poly_ring.base_ring().get_ring();
    let scale_to_ring_factor = ring.factor_scaling();
    let poly_l2_pseudo_norm_ln = poly_ring.terms(f).map(|(c, _)| ring.ln_pseudo_norm(c).abs()).max_by(f64::total_cmp).unwrap() + (poly_ring.degree(f).unwrap() as f64).ln();
    let ln_bound = ring.ln_factor_coeff_bound(poly_l2_pseudo_norm_ln, poly_ring.degree(f).unwrap()) + ring.ln_pseudo_norm(&scale_to_ring_factor);

    'try_random_prime: for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let prime = ring.random_maximal_ideal(|| rng.rand_u64());
        let e = ring.required_power(&prime, ln_bound);
        let reduction_map = IntermediateReductionMap::new(ring, &prime, e, 1);

        let prime_field = ring.local_field_at(&prime);
        let prime_field_poly_ring = DensePolyRing::new(&prime_field, "X");
        let prime_ring = reduction_map.codomain();
        let iso = prime_field.can_iso(&prime_ring).unwrap();

        let prime_field_f = prime_field_poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (iso.inv().map(ring.reduce_full(&prime, (&prime_ring, 1), ring.clone_el(c))), i)));
        let mut factors = Vec::new();
        for (f, k) in <_ as FactorPolyField>::factor_poly(&prime_field_poly_ring, &prime_field_f).0 {
            if k > 1 {
                continue 'try_random_prime;
            }
            factors.push(f);
        }
    
        let target_poly_ring = DensePolyRing::new(reduction_map.domain(), "X");
        let local_ring_f = target_poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (ring.reduce_full(&prime, (reduction_map.domain(), reduction_map.from_e()), poly_ring.base_ring().clone_el(c)), i)));
        
        let local_ring_factorization = hensel_lift_factorization(&reduction_map, &target_poly_ring, &prime_field_poly_ring, &local_ring_f, &factors[..]);
        return combine_local_factors_local(ring, &scale_to_ring_factor, &prime, poly_ring, f, &target_poly_ring, reduction_map.from_e(), local_ring_factorization);
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
pub fn factor_poly_local<P>(poly_ring: P, f: El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyLocallyDomain + DivisibilityRing
{
    assert!(!poly_ring.is_zero(&f));
    let power_decomposition = poly_power_decomposition_local(poly_ring, poly_ring.clone_el(&f));
    let mut result = Vec::new();
    let mut current = poly_ring.clone_el(&f);
    for (factor, _k) in power_decomposition {
        let lc_factor = poly_ring.lc(&factor).unwrap();
        let factor_monic = evaluate_aX(poly_ring, &factor, lc_factor);
        let factorization = factor_poly_squarefree_monic_local(poly_ring, &factor_monic);
        for irred_factor in factorization.into_iter().map(|fi| {
            balance_poly(poly_ring, unevaluate_aX(poly_ring, &fi, &lc_factor)).0
        }) {
            let irred_factor_lc = poly_ring.lc(&irred_factor).unwrap();
            let mut power = 0;
            while let Some(quo) = poly_ring.checked_div(&poly_ring.inclusion().mul_ref_map(&current, &poly_ring.base_ring().pow(poly_ring.base_ring().clone_el(&irred_factor_lc), poly_ring.degree(&f).unwrap())), &irred_factor) {
                current = balance_poly(poly_ring, quo).0;
                power += 1;
            }
            assert!(power >= 1);
            result.push((irred_factor, power));
        }
    }
    debug_assert_eq!(poly_ring.degree(&f).unwrap(), result.iter().map(|(fi, i)| *i * poly_ring.degree(fi).unwrap()).sum::<usize>());
    return result;
}

#[cfg(test)]
use crate::integer::*;
#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::algorithms::poly_div::poly_div_domain;
#[cfg(test)]
use crate::algorithms::poly_factor_local::make_primitive;

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
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f3), 3), (poly_ring.clone_el(&f4), 3)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f2), 2), (poly_ring.clone_el(&f3), 3), (poly_ring.clone_el(&f4), 3)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);
    
    let expected = [(poly_ring.clone_el(&f1), 1), (poly_ring.clone_el(&f2), 1), (poly_ring.clone_el(&f4), 2), (poly_ring.clone_el(&f3), 3)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);

    // this is a tricky case, since for every prime `p`, at least one `fi` splits - however they are all irreducible over ZZ
    let [f1, f2, f3] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(2) + 1,
        X.pow_ref(2) + 2,
        X.pow_ref(2) - 2
    ]);

    let expected = [(poly_ring.clone_el(&f1), 1), (poly_ring.clone_el(&f2), 1), (poly_ring.clone_el(&f3), 1)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f1), 2), (poly_ring.clone_el(&f2), 1), (poly_ring.clone_el(&f3), 1)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);

    let expected = [(poly_ring.clone_el(&f1), 2), (poly_ring.clone_el(&f2), 2), (poly_ring.clone_el(&f3), 2)];
    let actual = factor_poly_local(&poly_ring, multiply_out(&expected));
    assert_eq(&expected, actual);
}

#[test]
fn random_test_factor_poly_local() {
    let ring = BigIntRing::RING;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = ring.int_hom().map(10000);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = poly_ring.from_terms((0..=20).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let g = poly_ring.from_terms((0..=10).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        println!("Testing factorization on ({}) * ({})^2", poly_ring.format(&f), poly_ring.format(&g));
        let product = poly_ring.mul_ref_fst(&f, poly_ring.mul_ref(&g, &g));
        let factorization = factor_poly_local(&poly_ring, poly_ring.clone_el(&product));
        assert!(factorization.len() >= 2);
        assert!(factorization.iter().any(|(_, k)| *k >= 2));
        for (factor, _) in &factorization {
            let (_, rem1, _) = poly_div_domain(&poly_ring, poly_ring.clone_el(&f), factor);
            let (_, rem2, _) = poly_div_domain(&poly_ring, poly_ring.clone_el(&g), factor);
            assert!(poly_ring.is_zero(&rem1) || poly_ring.is_zero(&rem2));
        }
        assert_el_eq!(
            &poly_ring,
            make_primitive(&poly_ring, &product).0,
            make_primitive(&poly_ring, &poly_ring.prod(factorization.iter().map(|(f, e)| poly_ring.pow(poly_ring.clone_el(f), *e)))).0
        )
    }
}