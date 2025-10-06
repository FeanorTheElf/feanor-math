use std::convert::identity;

use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_gcd::hensel::*;
use crate::algorithms::poly_gcd::squarefree_part::poly_power_decomposition_local;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::computation::ComputationController;
use crate::reduce_lift::poly_factor_gcd::*;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::rings::poly::dense_poly::*;
use crate::divisibility::*;
use crate::iters::clone_slice;
use crate::iters::powerset;
use crate::seq::VectorView;
use crate::MAX_PROBABILISTIC_REPETITIONS;

fn combine_local_factors_local<'ring, 'data, 'local, R, P1, P2>(reduction: &'local ReductionContext<'ring, 'data, R>, poly_ring: P1, poly: &El<P1>, local_poly_ring: P2, local_e: usize, local_factors: Vec<El<P2>>) -> Vec<El<P1>>
    where R: ?Sized + PolyGCDLocallyDomain,
        P1: RingStore + Copy,
        P1::Type: PolyRing + DivisibilityRing,
        <P1::Type as RingExtension>::BaseRing: RingStore<Type = R>,
        P2: RingStore + Copy,
        P2::Type: PolyRing<BaseRing = &'local R::LocalRing<'ring>>,
        R::LocalRing<'ring>: 'local
{
    debug_assert!(poly_ring.base_ring().is_one(poly_ring.lc(poly).unwrap()));
    debug_assert!(local_factors.iter().all(|local_factor| local_poly_ring.base_ring().is_one(local_poly_ring.lc(local_factor).unwrap())));

    let ring = poly_ring.base_ring().get_ring();
    let reconstruct_poly = |factor| {
        let mut result = poly_ring.from_terms(local_poly_ring.terms(&factor).map(|(c, i)| (ring.reconstruct_ring_el(reduction.ideal(), std::slice::from_ref(*local_poly_ring.base_ring()).as_fn(), local_e, std::slice::from_ref(c).as_fn()), i)));
        _ = poly_ring.balance_poly(&mut result);
        return result;
    };

    let mut ungrouped_factors = (0..local_factors.len()).collect::<Vec<_>>();
    let mut current = poly_ring.clone_el(poly);
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
            let lifted_factor = reconstruct_poly(factor);
            if let Some(quo) = poly_ring.checked_div(&current, &lifted_factor) {
                return Some((lifted_factor, quo, clone_slice(indices)));
            } else {
                return None;
            }
        }).filter_map(identity).next().unwrap();
        current = new;
        result.push(factor);
        ungrouped_factors.retain(|j| !factor_group.contains(j));
    }
    return result;
}

#[stability::unstable(feature = "enable")]
pub enum FactorAndLiftModpeResult<P>
    where P: RingStore,
        P::Type: PolyRing
{
    PartialFactorization(Vec<El<P>>),
    Irreducible,
    Unknown,
    NotSquarefreeModpe
}

///
/// Factors the given monic polynomial modulo `p^e` and searches for the smallest groups of factors whose product
/// lifts and gives a factor of `f` globally. If all factors of `f` are shortest lifts of polynomials modulo `p^e`,
/// this means that the result is the factorization of `f`.
/// 
/// For smaller `e`, this might still compute a nontrivial, partial factorization. However, clearly, it might
/// return non-irreducible factors of `f`.
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_and_lift_mod_pe<'ring, R, P, Controller>(poly_ring: P, prime: &R::SuitableIdeal<'ring>, e: usize, poly: &El<P>, controller: Controller) -> FactorAndLiftModpeResult<P>
    where R: ?Sized + PolyGCDLocallyDomain,
        P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = R>,
        Controller: ComputationController
{
    let ring = poly_ring.base_ring().get_ring();
    assert_eq!(1, ring.maximal_ideal_factor_count(&prime), "currently only maximal ideals are supported, got {}", IdealDisplayWrapper::new(ring, prime));

    let reduction = ReductionContext::new(ring, prime, e);
    let red_map = reduction.intermediate_ring_to_field_reduction(0);

    let iso = reduction.base_ring_to_field_iso(0);
    let F = iso.codomain();
    let FX = DensePolyRing::new(&F, "X");
    let R_to_F = (&iso).compose(reduction.main_ring_to_field_reduction(0));

    let poly_mod_m = FX.lifted_hom(poly_ring, R_to_F).map_ref(poly);
    let mut factors = Vec::new();
    for (f, k) in <_ as FactorPolyField>::factor_poly_with_controller(&FX, &poly_mod_m, controller.clone()).0 {
        if k > 1 {
            return FactorAndLiftModpeResult::NotSquarefreeModpe;
        }
        factors.push(f);
    }
    if factors.len() == 1 {
        return FactorAndLiftModpeResult::Irreducible;
    }

    let SX = DensePolyRing::new(*red_map.domain(), "X");
    let poly_mod_me = SX.lifted_hom(poly_ring, reduction.main_ring_to_intermediate_ring_reduction(0)).map_ref(poly);
    let factorization_mod_me = hensel_lift_factorization(&red_map, &SX, &FX, &poly_mod_me, &factors[..], controller);
    let combined_factorization = combine_local_factors_local(&reduction, poly_ring, poly, &SX, e, factorization_mod_me);
    if combined_factorization.len() == 1 {
        return FactorAndLiftModpeResult::Unknown;
    } else {
        return FactorAndLiftModpeResult::PartialFactorization(combined_factorization);
    }
}

fn ln_factor_max_coeff<P>(ZZX: P, f: &El<P>) -> f64
    where P: RingStore,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    assert!(!ZZX.is_zero(f));
    let ZZ = ZZX.base_ring();
    let d = ZZX.degree(f).unwrap();

    // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen,
    // or equivalently Ex. 20 from Chapter 4.6.2 in Knuth's Art
    let log2_poly_norm = ZZX.terms(f).map(|(c, _)| ZZ.abs_log2_ceil(c).unwrap()).max().unwrap() as f64 + (d as f64).log2();
    return (log2_poly_norm + d as f64) * 2f64.ln();
}

fn factor_squarefree_monic_integer_poly_local<'a, P, Controller>(ZZX: P, f: &El<P>, controller: Controller) -> Vec<El<P>>
    where P: 'a + RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Controller: ComputationController
{
    let ZZ = ZZX.base_ring();
    assert!(ZZ.is_one(ZZX.lc(f).unwrap()));
    let mut rng = oorandom::Rand64::new(1);
    let bound = ln_factor_max_coeff(ZZX, f);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {

        let prime = ZZ.get_ring().random_suitable_ideal(|| rng.rand_u64());
        assert_eq!(1, ZZ.get_ring().maximal_ideal_factor_count(&prime));
        let prime_f64 = BigIntRing::RING.to_float_approx(&ZZ.get_ring().principal_ideal_generator(&prime));
        let e = (bound / prime_f64.ln()).ceil() as usize + 1;
        log_progress!(controller, "(mod={}^{})", IdealDisplayWrapper::new(ZZ.get_ring(), &prime), e);
        match factor_and_lift_mod_pe(ZZX, &prime, e, f, controller.clone()) {
            FactorAndLiftModpeResult::Irreducible => return vec![ZZX.clone_el(f)],
            FactorAndLiftModpeResult::PartialFactorization(result) => return result,
            // unknown means irreducible, since we chose `e` large enough
            FactorAndLiftModpeResult::Unknown => return vec![ZZX.clone_el(f)],
            FactorAndLiftModpeResult::NotSquarefreeModpe => {}
        }
    }
    unreachable!()
}

///
/// Factors the given polynomial over the integers.
/// 
/// Its factors are returned as primitive polynomials, thus their
/// product is `f` only up to multiplication by a nonzero integer. 
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_factor_integer<P, Controller>(ZZX: P, f: El<P>, controller: Controller) -> Vec<(El<P>, usize)>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Controller: ComputationController
{
    assert!(!ZZX.is_zero(&f));
    let power_decomposition = poly_power_decomposition_local(ZZX, ZZX.clone_el(&f), controller.clone());

    controller.run_computation(format_args!("factor_int_poly(deg={})", ZZX.degree(&f).unwrap()), |controller| {

        let mut result = Vec::new();
        let mut current = ZZX.clone_el(&f);
        for (factor, _k) in power_decomposition {
            log_progress!(controller, "(deg={})", ZZX.degree(&factor).unwrap());
            let lc_factor = ZZX.lc(&factor).unwrap();
            let factor_monic = evaluate_aX(ZZX, &factor, lc_factor);
            let factorization = factor_squarefree_monic_integer_poly_local(&ZZX, &factor_monic, controller.clone());
            for irred_factor in factorization.into_iter().map(|fi| unevaluate_aX(ZZX, &fi, &lc_factor)) {
                let irred_factor_lc = ZZX.lc(&irred_factor).unwrap();
                let mut power = 0;
                let irred_factor = make_primitive(ZZX, &irred_factor).0;
                while let Some(quo) = ZZX.checked_div(&ZZX.inclusion().mul_ref_map(&current, &ZZX.base_ring().pow(ZZX.base_ring().clone_el(&irred_factor_lc), ZZX.degree(&f).unwrap())), &irred_factor) {
                    current = quo;
                    _ = ZZX.balance_poly(&mut current);
                    power += 1;
                }
                assert!(power >= 1);
                result.push((irred_factor, power));
            }
        }
        debug_assert_eq!(ZZX.degree(&f).unwrap(), result.iter().map(|(fi, i)| *i * ZZX.degree(fi).unwrap()).sum::<usize>());
        return result;
    })
}

#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::algorithms::poly_gcd::make_primitive;
#[cfg(test)]
use crate::computation::TEST_LOG_PROGRESS;

#[test]
fn test_factor_int_poly() {
    let ZZX = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1, X + 1]);
    let input = ZZX.mul_ref(&f, &g);
    let actual = poly_factor_integer(&ZZX, input, TEST_LOG_PROGRESS);
    assert_eq!(2, actual.len());
    for (factor, e) in &actual {
        assert_eq!(1, *e);
        assert!(ZZX.eq_el(&f, factor) || ZZX.eq_el(&g, factor), "Got unexpected factor {}", ZZX.formatted_el(&factor));
    }

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [5 * X.pow_ref(2) + 1, 3 * X.pow_ref(2) + 2]);
    let input = ZZX.mul_ref(&f, &g);
    let actual = poly_factor_integer(&ZZX, input, TEST_LOG_PROGRESS);
    assert_eq!(2, actual.len());
    for (factor, e) in &actual {
        assert_eq!(1, *e);
        assert!(ZZX.eq_el(&f, factor) || ZZX.eq_el(&g, factor), "Got unexpected factor {}", ZZX.formatted_el(&factor));
    }

    let [f] = ZZX.with_wrapped_indeterminate(|X| [5 * X.pow_ref(2) + 1]);
    let input = ZZX.mul_ref(&f, &f);
    let actual = poly_factor_integer(&ZZX, input, TEST_LOG_PROGRESS);
    assert_eq!(1, actual.len());
    assert_eq!(2, actual[0].1);
    assert_el_eq!(&ZZX, &f, &actual[0].0);
}
