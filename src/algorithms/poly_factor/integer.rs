use std::cmp::{Ordering, min};
use std::collections::BinaryHeap;
use std::convert::identity;

use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::hensel::{HenselLift, create_power_p_poly_ring};
use crate::algorithms::poly_factor::FactorPolyField;
use crate::algorithms::poly_gcd::gcd_lift::*;
use crate::algorithms::poly_gcd::integer::poly_power_decomposition_integer_monic;
use crate::algorithms::poly_gcd::{make_monic, make_primitive, unmake_monic};
use crate::algorithms::primelist::prime_fields_for_local_computation;
use crate::iters::{clone_slice, powerset};
use crate::prelude::*;
use crate::ring_impls::as_field::AsField;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::poly::{PolyRing, PolyRingStore};
use crate::ring_impls::zn::zn_64b::Zn64B;
use crate::ring_impls::zn::zn_big::ZnGB;
use crate::ring_impls::zn::{ZnReductionMap, ZnRing, ZnRingStore};

/// The polynomial was found to be not squarefree modulo multiple prime ideals. This usually means
/// that it it not squarefree globally.
#[stability::unstable(feature = "enable")]
pub struct ProbablyNotSquarefree;

/// Current Hensel lifting didn't yield the correct result. Since for factorizations, factors modulo
/// `p^e` are combined until one retrieves a factor of the original polynomial, this can only happen
/// when the the polynomial itself is not retrieved from its value modulo `p^e`, in other words if
/// `p^e` is of similar size as some coefficients of the polynomial.
#[stability::unstable(feature = "enable")]
pub struct LiftUnsuccessful;

#[instrument(skip_all, level = "trace")]
fn combine_local_factors_local<P>(
    ZZX: P,
    ZpeX: &DensePolyRing<ZnGB<BigIntRing>>,
    poly: &El<P>,
    local_factors: &[&El<DensePolyRing<ZnGB<BigIntRing>>>],
) -> Result<Vec<El<P>>, LiftUnsuccessful>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingBase<P>: IntegerRing,
{
    let ZZ = ZZX.base_ring();
    let Zpe = ZpeX.base_ring();
    debug_assert!(ZZ.is_one(ZZX.lc(poly).unwrap()));
    debug_assert!(local_factors.iter().all(|f| Zpe.is_one(ZpeX.lc(f).unwrap())));

    let reconstruct_poly = |factor| {
        ZZX.from_terms(
            ZpeX.terms(&factor)
                .map(|(c, i)| (int_cast(Zpe.smallest_lift(c.clone()), ZZ, Zpe.integer_ring()), i)),
        )
    };

    let mut ungrouped_factors = (0..local_factors.len()).collect::<Vec<_>>();
    let mut current = poly.clone();
    let mut result = Vec::new();
    while ungrouped_factors.len() > 0 {
        // Here we use the naive approach to group the factors such that the product of each group
        // is integral - just try all combinations. It might be worth using LLL for this instead;
        // note that powerset yields smaller subsets first
        let (factor, new, factor_group) = powerset(ungrouped_factors.iter().copied(), |indices| {
            if indices.len() == 0 {
                return None;
            }
            let factor = ZpeX.prod(indices.iter().copied().map(|i| local_factors[i].clone()));
            let lifted_factor: El<P> = reconstruct_poly(factor);
            if let Some(quo) = ZZX.checked_div(&current, &lifted_factor) {
                return Some((lifted_factor, quo, clone_slice(indices)));
            } else {
                return None;
            }
        })
        .filter_map(identity)
        .next()
        .ok_or(LiftUnsuccessful)?;
        current = new;
        result.push(factor);
        ungrouped_factors.retain(|j| !factor_group.contains(j));
    }
    return Ok(result);
}

struct PolyFactorizationLift {
    factorization_lifter: HenselLift<DensePolyRing<ZnGB<BigIntRing>>>,
    prime: El<BigIntRing>,
}

impl PolyFactorizationLift {
    #[instrument(skip_all, level = "trace")]
    fn new(
        FpX: DensePolyRing<AsField<Zn64B>>,
        factorization: Vec<El<DensePolyRing<AsField<Zn64B>>>>,
    ) -> Result<Self, ProbablyNotSquarefree> {
        let lifter = HenselLift::new(&FpX, factorization).map_err(|_| ProbablyNotSquarefree)?;
        let ZpX = DensePolyRing::new(
            ZnGB::new(ZZbig, int_cast(*FpX.base_ring().modulus(), ZZbig, ZZi64)),
            "X",
        );
        let hom = ZnReductionMap::new(FpX.base_ring(), ZpX.base_ring().clone()).unwrap();
        return Ok(Self {
            prime: ZpX.base_ring().modulus().clone(),
            factorization_lifter: lifter.change_ring(ZpX, hom),
        });
    }

    #[instrument(skip_all, level = "trace")]
    fn lift_factorization_to<P>(
        &mut self,
        poly_ring: P,
        target: &El<P>,
        lift_to_degree: usize,
    ) -> Result<Vec<El<P>>, LiftUnsuccessful>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingBase<P>: IntegerRing,
    {
        let ZpeX = create_power_p_poly_ring(self.prime.clone(), lift_to_degree);
        let target_mod_pe = ZpeX
            .lifted_hom(&poly_ring, ZpeX.base_ring().can_hom(poly_ring.base_ring()).unwrap())
            .map_ref(target);
        take_mut::take(&mut self.factorization_lifter, |lifter| {
            lifter.lift_to(lift_to_degree, ZpeX, &target_mod_pe, |old_Zpe, Zpe, x| {
                Zpe.get_ring()
                    .from_int_promise_reduced(old_Zpe.smallest_positive_lift(x.clone()))
            })
        });
        let result = combine_local_factors_local(
            poly_ring,
            self.factorization_lifter.poly_ring(),
            target,
            &self.factorization_lifter.factorization().collect::<Vec<_>>(),
        );
        return result;
    }
}

fn ln_factor_max_coeff<P>(ZZX: P, f: &El<P>) -> f64
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    <BaseRingStore<P> as RingStore>::Ring: IntegerRing,
{
    assert!(!ZZX.is_zero(f));
    let ZZ = ZZX.base_ring();
    let d = ZZX.degree(f).unwrap();

    // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen,
    // or equivalently Ex. 20 from Chapter 4.6.2 in Knuth's Art
    let log2_poly_norm =
        ZZX.terms(f).map(|(c, _)| ZZ.abs_log2_ceil(c).unwrap()).max().unwrap() as f64 + (d as f64).log2();
    return (log2_poly_norm + d as f64) * 2.0f64.ln();
}

#[stability::unstable(feature = "enable")]
pub enum PolyFactorizationResult<T> {
    /// The factorization was found
    FoundFactorization(T),
    /// The polynomial was found to be not squarefree modulo multiple prime ideals. This usually
    /// means that it it not squarefree globally.
    ProbablyNotSquarefree,
}

#[stability::unstable(feature = "enable")]
pub struct PolyFactorizationInQuotient<State> {
    /// The degree such that lifting the local factorization to modulus `p^max_lift_degree` is
    /// provably enough to derive the complete factorization of the polynomial
    pub max_lift_degree: usize,
    /// Number of factors of `poly mod p`
    pub factor_count: usize,
    /// State that can later be used to lift the factorization of `poly mod p` to higher powers of
    /// `p`
    pub state: State,
}

const MIN_QUOTIENT_FACTORIZATIONS_BEFORE_LIFT: usize = 4;
const NON_SQUAREFREE_COUNT_ABORT: usize = 3;
const RAMP_UP_LIFT_TO: f64 = 2.0;
const BEGIN_LIFT_UP_TO_LOGFRACTION: f64 = 4.0;

/// High-level approach of deriving the factorization of a polynomial by
/// computing it in quotients, and trying to lift the result.
///
/// This function doesn't handle any arithmetic, but encodes the
/// high-level strategy:
///  - `create_factorizations` is given a polynomial and produces an iterator over different primes
///    `p` together with information about the factorization of `poly mod p`.
///  - this function will consider multiple elements in the output of `create_factorization` and
///    choose one (or more) for lifting. In that case, it will first call `start_lift` with the
///    corresponding state.
///  - `start_lift` may fail if the factorization of the polynomial modulo the prime is not suitable
///    for lifting (e.g. the factors are not squarefree).
///  - if it doesn't fail, `start_lift` produces `lift` object. The function will then call
///    `proceed_with_lift(lift, e)`, which will actually lift the factorization of `poly mod p` to
///    `R/p^e`. The output of `proceed_with_lift(lift, e)` should then be a (potentially incomplete)
///    factorization of `poly`. Note that if `e = max_lift_degree` as returned by
///    `create_factorization`, the output of `proceed_with_lift(lift, e)` is assumed to be the
///    complete factorization of `poly`.
///  - Note that `proceed_with_lift(lift, e)` may be called multiple times with increasing `e`. It
///    may be advantageous if state from intermediate lifts is preserved.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_factorization_from_quotients<T, I, F_quotients, F_start, F_lift, State, OngoingLift>(
    poly: T,
    mut create_factorizations: F_quotients,
    mut start_lift: F_start,
    mut proceed_with_lift: F_lift,
) -> PolyFactorizationResult<Vec<T>>
where
    F_quotients: FnMut(&T) -> I,
    I: Iterator<Item = PolyFactorizationInQuotient<State>>,
    F_start: FnMut(State) -> Result<OngoingLift, NotLiftable>,
    F_lift: FnMut(&mut OngoingLift, &T, usize) -> Result<Vec<T>, LiftUnsuccessful>,
{
    struct CmpByKey<T>(i64, T);

    impl<T> PartialEq for CmpByKey<T> {
        fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<T> PartialOrd for CmpByKey<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }

    impl<T> Eq for CmpByKey<T> {}

    impl<T> Ord for CmpByKey<T> {
        fn cmp(&self, other: &Self) -> Ordering { self.0.cmp(&other.0) }
    }

    let mut result = Vec::new();
    let mut open = Vec::new();
    open.push(poly);

    'factor_open: while let Some(poly) = open.pop() {
        let mut candidates_for_lift = BinaryHeap::new();
        let mut ongoing_lifts = Vec::new();
        let mut found_non_squarefree = 0;
        for factorization_in_quotient in create_factorizations(&poly) {
            if factorization_in_quotient.factor_count == 1 {
                result.push(poly);
                continue 'factor_open;
            }
            candidates_for_lift.push(CmpByKey(
                -(factorization_in_quotient.factor_count as i64),
                factorization_in_quotient,
            ));
            if candidates_for_lift.len() >= MIN_QUOTIENT_FACTORIZATIONS_BEFORE_LIFT {
                let candidate = candidates_for_lift.pop().unwrap().1;
                let start_with_lift_attempt =
                    ((candidate.max_lift_degree as f64).ln() / RAMP_UP_LIFT_TO.ln() / BEGIN_LIFT_UP_TO_LOGFRACTION)
                        .ceil() as i32;
                match start_lift(candidate.state) {
                    Ok(lift) => ongoing_lifts.push((candidate.max_lift_degree, start_with_lift_attempt, lift)),
                    Err(NotLiftable::NotSquarefree) => {
                        if found_non_squarefree + 1 >= NON_SQUAREFREE_COUNT_ABORT {
                            return PolyFactorizationResult::ProbablyNotSquarefree;
                        } else {
                            found_non_squarefree += 1;
                        }
                    }
                    Err(NotLiftable::BadPrime) => {}
                }
                for (max_deg, lift_attempt, ongoing_lift) in &mut ongoing_lifts {
                    *lift_attempt += 1;
                    let lift_to_deg = min(*max_deg, RAMP_UP_LIFT_TO.powi(*lift_attempt).ceil() as usize);
                    let lift_result = proceed_with_lift(ongoing_lift, &poly, lift_to_deg);
                    if let Ok(lift_result) = lift_result {
                        if lift_to_deg == *max_deg {
                            result.extend(lift_result);
                            continue 'factor_open;
                        } else if lift_result.len() > 1 {
                            open.extend(lift_result);
                            continue 'factor_open;
                        }
                    }
                }
            }
        }
        unreachable!()
    }
    return PolyFactorizationResult::FoundFactorization(result);
}

#[instrument(skip_all, level = "trace")]
fn poly_factor_integer_squarefree_monic<P>(ZZX: P, f: El<P>) -> Vec<El<P>>
where
    P: RingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    <BaseRingStore<P> as RingStore>::Ring: IntegerRing,
{
    let ZZ = ZZX.base_ring();
    assert!(ZZ.is_one(ZZX.lc(&f).unwrap()));

    let result = poly_factorization_from_quotients(
        f,
        |poly| {
            let poly = poly.clone();
            let bound = ln_factor_max_coeff(ZZX, &poly);
            prime_fields_for_local_computation()
                .take(PROBABILISTIC_REPETITIONS)
                .map(move |Fp| {
                    let FpX = DensePolyRing::new(Fp, "X");
                    let f_mod_p = FpX
                        .lifted_hom(ZZX, FpX.base_ring().can_hom(ZZX.base_ring()).unwrap())
                        .map_ref(&poly);
                    let factorization = FactorPolyField::factor_poly(&FpX, &f_mod_p).0;
                    let factor_count = factorization.iter().map(|(_, e)| *e).sum::<usize>();
                    let prime_f64 = *FpX.base_ring().modulus() as f64;
                    let max_lift_degree = (bound / prime_f64.ln()).ceil() as usize + 1;
                    return PolyFactorizationInQuotient {
                        factor_count,
                        max_lift_degree,
                        state: (FpX, factorization),
                    };
                })
        },
        |(FpX, factorization)| {
            if factorization.iter().any(|(_, e)| *e > 1) {
                Err(NotLiftable::NotSquarefree)
            } else {
                PolyFactorizationLift::new(FpX, factorization.into_iter().map(|(f, _)| f).collect())
                    .map_err(|_| NotLiftable::NotSquarefree)
            }
        },
        |lift, target, lift_to_degree| lift.lift_factorization_to(ZZX, target, lift_to_degree),
    );
    match result {
        PolyFactorizationResult::FoundFactorization(result) => result,
        PolyFactorizationResult::ProbablyNotSquarefree => unreachable!(),
    }
}

/// Factors the given polynomial over the integers.
///
/// Its factors are returned as primitive polynomials, thus their
/// product is `f` only up to multiplication by a nonzero integer.
///
/// Use this when implementing [`FactorPolyField`] for integer
/// rings; Otherwise, compute power decompositions through
/// [`FactorPolyField::factor_poly()`].
///
/// [`FactorPolyField`]: crate::algorithms::poly_factor::FactorPolyField
/// [`FactorPolyField::poly_power_decomposition()`]: crate::algorithms::poly_factor::FactorPolyField::factor_poly()
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_factor_integer<P>(ZZX: P, f: &El<P>) -> Vec<(El<P>, usize)>
where
    P: PolyRingStore + Copy,
    P::Ring: PolyRing + DivisibilityRing,
    <BaseRingStore<P> as RingStore>::Ring: IntegerRing,
{
    assert!(!ZZX.is_zero(&f));
    let f = make_primitive(ZZX, f).0;
    let lc_f = ZZX.lc(&f).unwrap().clone();
    let (f, _) = make_monic(ZZX, &f, &lc_f);
    let power_decomposition = poly_power_decomposition_integer_monic(ZZX, &f, PROBABILISTIC_REPETITIONS);

    let mut result = Vec::new();
    let mut current = f.clone();
    for (factor, _k) in power_decomposition {
        let factorization = poly_factor_integer_squarefree_monic(&ZZX, factor);
        for irred_factor in factorization.into_iter() {
            let mut power = 0;
            while let Some(quo) = ZZX.checked_div(&current, &irred_factor) {
                current = quo;
                power += 1;
            }
            assert!(power >= 1);
            result.push((irred_factor, power));
        }
    }
    debug_assert_eq!(
        ZZX.degree(&f).unwrap(),
        result.iter().map(|(fi, i)| *i * ZZX.degree(fi).unwrap()).sum::<usize>()
    );
    return result
        .into_iter()
        .map(|(f, e)| (make_primitive(ZZX, &unmake_monic(ZZX, &f, &lc_f, &lc_f)).0, e))
        .collect();
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;

#[test]
fn test_poly_factor_integer() {
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
    let assert_correct_factorization = |powers: [usize; 5]| {
        let input = poly_ring.prod(
            powers
                .iter()
                .enumerate()
                .map(|(i, e)| poly_ring.pow(irred_polys[i].clone(), *e)),
        );
        let actual = poly_factor_integer(&poly_ring, &input);
        assert_eq!(powers.iter().filter(|e| **e != 0).count(), actual.len());
        for (f, e) in actual {
            if let Some((idx, _)) = irred_polys
                .iter()
                .enumerate()
                .filter(|(_, g)| poly_ring.eq_el(&f, *g))
                .next()
            {
                assert_eq!(powers[idx], e);
            } else {
                panic!(
                    "factorization yielded {}, which is not an expected irreducible factor",
                    poly_ring.formatted_el(&f)
                )
            }
        }
    };

    assert_correct_factorization([1, 0, 0, 0, 0]);
    assert_correct_factorization([1, 0, 1, 0, 0]);
    assert_correct_factorization([1, 1, 0, 1, 1]);
    assert_correct_factorization([1, 0, 2, 0, 2]);
    assert_correct_factorization([0, 4, 0, 0, 0]);
}

#[test]
fn random_test_poly_factor_integer() {
    // feanor_tracing::DelayedLogger::init_test();

    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    let filtered_chrome_layer = tracing_subscriber::Layer::with_filter(chrome_layer, tracing_subscriber::filter::filter_fn(|_metadata| true));
    tracing_subscriber::util::SubscriberInitExt::init(tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt::with(tracing_subscriber::registry(), filtered_chrome_layer));

    let ring = ZZbig;
    let poly_ring = DensePolyRing::new(ring, "X");
    let mut rng = oorandom::Rand64::new(0);
    let bound = ring.int_hom().map(500);
    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f = poly_ring.from_terms((0..=12).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let g = poly_ring.from_terms((0..=11).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let h = poly_ring.from_terms((0..=10).map(|i| (ring.get_uniformly_random(&bound, || rng.rand_u64()), i)));
        let input = poly_ring.prod([f, g, h]);
        let actual = poly_factor_integer(&poly_ring, &input);
        assert!(actual.iter().map(|(_, e)| *e).sum::<usize>() >= 3);
        assert_el_eq!(
            &poly_ring,
            input,
            poly_ring.prod(actual.into_iter().map(|(f, e)| poly_ring.pow(f, e)))
        );
    }
}
