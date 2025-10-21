use std::fmt::Debug;

use crate::rings::poly::dense_poly::DensePolyRing;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::poly_gcd::hensel::*;
use crate::seq::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

use super::evaluate_aX;
use super::unevaluate_aX;
use super::INCREASE_EXPONENT_PER_ATTEMPT_CONSTANT;

///
/// For the power-decomposition `f = f1^e1 ... fr^er`, stores a tuple (ei, deg(fi))
/// 
#[derive(PartialEq, Eq, Debug)]
struct Signature {
    perfect_power: usize,
    degree: usize
}

///
/// Lifts the power decomposition modulo `I^e` to a potential power decomposition in the ring `R`.
/// If this indeed gives the correct power decomposition, it is returned, otherwise `None` is returned.
/// 
/// We use the notation
///  - `R` is the main ring
///  - `F` is `R/m` where `m` is a maximal ideal containing the currently considered ideal `I`
///  - `S` is `R/m^e`
/// 
fn power_decomposition_from_local_power_decomposition<'ring, 'data, 'local, R, P>(
    reduction: &'local ReductionContext<'ring, 'data, R>, 
    RX: P, 
    poly: &El<P>, 
    signature: &[Signature], 
    SXs: &[DensePolyRing<&'local R::LocalRing<'ring>>], 
    local_power_decompositions: &[Vec<El<DensePolyRing<&'local R::LocalRing<'ring>>>>]
) -> Option<Vec<(El<P>, usize)>>
    where R: ?Sized + PolyLiftFactorsDomain,
        P: RingStore + Copy,
        P::Type: PolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = R>
{
    assert_eq!(reduction.len(), local_power_decompositions.len());
    assert_eq!(reduction.len(), SXs.len());

    let mut result = Vec::new();
    for (k, sig) in signature.iter().enumerate() {
        let power_factor = RX.from_terms((0..=(sig.degree * sig.perfect_power)).map(|i| (
            reduction.reconstruct_ring_el((0..reduction.len()).map_fn(|j| SXs[j].coefficient_at(&local_power_decompositions[j][k], i))),
            i
        )));
        if let Some(mut root_of_factor) = poly_root(RX, &power_factor, sig.perfect_power) {
            _ = RX.balance_poly(&mut root_of_factor);
            let lc_inv = RX.base_ring().invert(RX.lc(&root_of_factor).unwrap()).unwrap();
            RX.inclusion().mul_assign_map(&mut root_of_factor, lc_inv);
            result.push((root_of_factor, sig.perfect_power));
        } else {
            return None;
        }
    }
    // at first, I thought this could not happen, but actually it can. If we do a faulty lift, the polynomials might after all still 
    // turn out to be perfect powers; the alternative to this check here would be to check previously if all "factors" really divide f;
    // I believe this is faster
    if !RX.eq_el(&poly, &RX.prod(result.iter().map(|(f, k)| RX.pow(RX.clone_el(f), *k)))) {
        return None;
    }
    return Some(result);
}

///
/// Computes the power decomposition modulo `m` and lifts it to `m^e`, for the given maximal ideal `m`
/// as specified by `S_to_F`.
/// 
/// We use the notation
///  - `R` is the main ring
///  - `F` is `R/m` where `m` is a maximal ideal containing the currently considered ideal `I`
///  - `S` is `R/m^e`
/// 
fn compute_local_power_decomposition<'ring, 'data, 'local, R, P1, P2, Controller>(
    RX: P1, 
    f: &El<P1>, 
    S_to_F: &PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>, 
    SX: P2,
    controller: Controller
) -> Option<(Vec<Signature>, Vec<El<P2>>)>
    where R: ?Sized + PolyLiftFactorsDomain,
        P1: RingStore + Copy,
        P1::Type: PolyRing,
        <P1::Type as RingExtension>::BaseRing: RingStore<Type = R>,
        P2: RingStore + Copy,
        P2::Type: PolyRing<BaseRing = &'local R::LocalRing<'ring>>,
        R::LocalRing<'ring>: 'local,
        Controller: ComputationController
{
    assert!(SX.base_ring().get_ring() == S_to_F.domain().get_ring());
    let R = RX.base_ring().get_ring();
    let F = R.local_field_at(S_to_F.ideal(), S_to_F.max_ideal_idx());
    let FX = DensePolyRing::new(&F, "X");
    let iso = PolyGCDLocallyBaseRingToFieldIso::new(R, S_to_F.ideal(), S_to_F.codomain().get_ring(), F.get_ring(), S_to_F.max_ideal_idx());

    let f_mod_m = FX.from_terms(RX.terms(f).map(|(c, i)| (
        iso.map(R.reduce_ring_el(S_to_F.ideal(), (S_to_F.codomain().get_ring(), 1), S_to_F.max_ideal_idx(), R.clone_el(c))),
        i
    )));
    let f_mod_me = SX.from_terms(RX.terms(f).map(|(c, i)| (
        R.reduce_ring_el(S_to_F.ideal(), (S_to_F.domain().get_ring(), S_to_F.from_e()), S_to_F.max_ideal_idx(), R.clone_el(c)),
        i
    )));

    let mut power_decomposition_mod_m = Vec::new();
    let mut signature = Vec::new();
    for (f, k) in <_ as PolyTFracGCDRing>::power_decomposition(&FX, &f_mod_m).into_iter() {
        signature.push(Signature {
            perfect_power: k,
            degree: FX.degree(&f).unwrap()
        });
        power_decomposition_mod_m.push(FX.pow(f, k));
    }

    let power_decomposition_mod_me = hensel_lift_factorization(
        S_to_F,
        SX,
        &FX,
        &f_mod_me,
        &power_decomposition_mod_m[..],
        controller
    );

    return Some((
        signature,
        power_decomposition_mod_me
    ));
}

///
/// For a monic polynomial `f in R[X]`, computes squarefree polynomials `fi` such that `a f = f1 f2^2 f3^3 ...`
/// for some nonzero element `a in R \ {0}`. These polynomials are returned as tuples `(fi, i)` with `fi != 0`.
/// 
/// The results can all be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_monic_local<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyLiftFactorsDomain,
        Controller: ComputationController
{
    assert!(poly_ring.base_ring().is_one(poly_ring.lc(poly).unwrap()));

    controller.run_computation(format_args!("power_decomp_local(deg={})", poly_ring.degree(poly).unwrap()), |controller| {

        let ring = poly_ring.base_ring().get_ring();
        let mut rng = oorandom::Rand64::new(1);

        'try_random_ideal: for current_attempt in 0..MAX_PROBABILISTIC_REPETITIONS {

            let ideal = ring.random_suitable_ideal(|| rng.rand_u64(), current_attempt);
            let heuristic_e = ring.heuristic_exponent(&ideal, poly_ring.degree(poly).unwrap(), poly_ring.terms(poly).map(|(c, _)| c));
            assert!(heuristic_e >= 1);
            let e = (heuristic_e as f64 * INCREASE_EXPONENT_PER_ATTEMPT_CONSTANT.powi(current_attempt.try_into().unwrap())).floor() as usize;
            let reduction = ReductionContext::new(ring, &ideal, e);

            log_progress!(controller, "(mod={}^{})(parts={})", IdealDisplayWrapper::new(ring, &ideal), e, reduction.len());

            let mut signature: Option<Vec<_>> = None;
            let mut poly_rings_mod_me = Vec::new();
            let mut power_decompositions_mod_me = Vec::new();

            for idx in 0..reduction.len() {
                let SX = DensePolyRing::new(*reduction.intermediate_ring_to_field_reduction(idx).domain(), "X");
                match compute_local_power_decomposition(poly_ring, poly, &reduction.intermediate_ring_to_field_reduction(idx), &SX, controller.clone()) {
                    None => {
                        unreachable!("`compute_local_power_decomposition()` currently cannot fail");
                    },
                    Some((new_signature, local_power_decomposition)) => if new_signature == &[Signature { degree: poly_ring.degree(poly).unwrap(), perfect_power: 1 }] {
                        return vec![(poly_ring.clone_el(poly), 1)];
                    } else if signature.is_some() && &signature.as_ref().unwrap()[..] != &new_signature[..] {
                        log_progress!(controller, "(signature_mismatch)");
                        continue 'try_random_ideal;
                    } else {
                        signature = Some(new_signature);
                        power_decompositions_mod_me.push(local_power_decomposition);
                        poly_rings_mod_me.push(SX);
                    }
                }
            }

            if let Some(result) = power_decomposition_from_local_power_decomposition(&reduction, poly_ring, poly, &signature.as_ref().unwrap()[..], &poly_rings_mod_me[..], &power_decompositions_mod_me[..]) {
                return result;
            } else {
                log_progress!(controller, "(invalid_lift)");
            }
        }
        unreachable!()
    })
}

///
/// For a polynomial `f in R[X]`, computes squarefree polynomials `fi` such that `a f = f1 f2^2 f3^3 ...`
/// for some nonzero ring element `a in R \ {0}`. These polynomials are returned as tuples `(fi, i)` with `fi != 0`.
/// 
/// The results can all be assumed to be "balanced", according to the contract of [`DivisibilityRing::balance_factor()`]
/// of the underlying ring.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_local<P, Controller>(poly_ring: P, mut f: El<P>, controller: Controller) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyLiftFactorsDomain + DivisibilityRing,
        Controller: ComputationController
{
    assert!(!poly_ring.is_zero(&f));
    _ = poly_ring.balance_poly(&mut f);
    let lcf = poly_ring.lc(&f).unwrap();
    let f_monic = evaluate_aX(poly_ring, &f, lcf);
    let power_decomposition = poly_power_decomposition_monic_local(poly_ring, &f_monic, controller);
    let result = power_decomposition.into_iter().map(|(fi, i)| {
        let mut result = unevaluate_aX(poly_ring, &fi, &lcf);
        _ = poly_ring.balance_poly(&mut result);
        return (result, i);
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
pub fn poly_squarefree_part_local<P, Controller>(poly_ring: P, f: El<P>, controller: Controller) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PolyLiftFactorsDomain + DivisibilityRing,
        Controller: ComputationController
{
    assert!(!poly_ring.is_zero(&f));
    let mut result = poly_ring.prod(poly_power_decomposition_local(poly_ring, f, controller).into_iter().map(|(fi, _i)| fi));
    _ = poly_ring.balance_poly(&mut result);
    return result;
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use super::make_primitive;

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
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected), TEST_LOG_PROGRESS);
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.mul_ref(&f3, &f4), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected), TEST_LOG_PROGRESS);
    assert_eq(&expected, &actual);

    let expected = [(poly_ring.clone_el(&f2), 2), (poly_ring.mul_ref(&f3, &f4), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected), TEST_LOG_PROGRESS);
    assert_eq(&expected, &actual);
    
    let expected = [(poly_ring.mul_ref(&f1, &f2), 1), (poly_ring.clone_el(&f4), 2), (poly_ring.clone_el(&f3), 3)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected), TEST_LOG_PROGRESS);
    assert_eq(&expected, &actual);
    
    let expected = [(poly_ring.mul_ref(&f1, &f2), 2), (poly_ring.clone_el(&f4), 4), (poly_ring.clone_el(&f3), 6)];
    let actual = poly_power_decomposition_monic_local(&poly_ring, &multiply_out(&expected), TEST_LOG_PROGRESS);
    assert_eq(&expected, &actual);
}

#[test]
#[ignore]
fn random_test_poly_power_decomposition_local() {
    let ring = BigIntRing::RING;
    let poly_ring = dense_poly::DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = ring.int_hom().map(1000);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = poly_ring.from_terms((0..=7).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let g = poly_ring.from_terms((0..=4).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let h = poly_ring.from_terms((0..=2).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let poly = make_primitive(&poly_ring, &poly_ring.prod([&f, &g, &g, &h, &h, &h, &h, &h].into_iter().map(|poly| poly_ring.clone_el(poly)))).0;
        
        let mut power_decomp = poly_power_decomposition_local(&poly_ring, poly_ring.clone_el(&poly), TEST_LOG_PROGRESS);
        for (f, _k) in &mut power_decomp {
            *f = make_primitive(&poly_ring, &f).0;
        }

        assert_el_eq!(&poly_ring, &poly, poly_ring.prod(power_decomp.iter().map(|(poly, k)| poly_ring.pow(poly_ring.clone_el(poly), *k))));
        assert!(poly_ring.divides(&poly_ring.prod(power_decomp.iter().filter(|(_, k)| k % 5 == 0).map(|(poly, k)| poly_ring.pow(poly_ring.clone_el(poly), k / 5))), &make_primitive(&poly_ring, &h).0));
        assert!(poly_ring.divides(&poly_ring.prod(power_decomp.iter().filter(|(_, k)| k % 2 == 0).map(|(poly, k)| poly_ring.pow(poly_ring.clone_el(poly), k / 2))), &make_primitive(&poly_ring, &g).0));
    }
}