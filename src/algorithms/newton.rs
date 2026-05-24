use std::cmp::Ordering;

use oorandom::Rand64;
use tracing::instrument;

use crate::PROBABILISTIC_REPETITIONS;
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::homomorphism::*;
use crate::prelude::*;
use crate::ring_impls::float_complex::{Complex64, Complex64Base, Complex64El};
use crate::ring_impls::poly::dense_poly::*;
use crate::ring_impls::poly::*;

const NEWTON_MAX_SCALE: u32 = 10;
const NEWTON_ITERATIONS: usize = 32;

#[derive(Debug)]
#[stability::unstable(feature = "enable")]
pub struct PrecisionError;

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn absolute_error_of_poly_eval<P>(
    poly_ring: P,
    f: &El<P>,
    poly_deg: usize,
    point: Complex64El,
    relative_error_point: f64,
) -> f64
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingStore<P>: RingStore<Ring = Complex64Base>,
{
    let CC = Complex64::RING;
    let mut current = point;
    let mut current_relative_error = relative_error_point;
    let mut total_error = 0.0;
    for i in 1..=poly_deg {
        total_error += CC.abs(*poly_ring.coefficient_at(f, i)) * CC.abs(current) * current_relative_error;
        // technically, we would have `(1 + current_relative_error)(1 + f64::EPSILON) - 1`, but we ignore
        // `O(f64::EPSILON^2)` terms here
        current_relative_error += f64::EPSILON;
        CC.mul_assign(&mut current, point);
    }
    return total_error;
}

/// Bounds the distance of `approx_root` to the real root. If we cannot establish with reasonable
/// confidence that there is a root close to `approx_root`, a [`PrecisionError`] will be returned.
///
/// # Implementation
///
/// Denote the polynomial by `f` and its Taylor series coefficients (at `x`) by `c_j = f^(j)(x)/j!`.
/// Assume `c_1 != 0`.
/// Consider the map `T(t) = t - f(x + t)/f'(x)`.
/// For `|t| < R`, we find that
/// ```text
///   |T(t)| = | t - \sum_i c_i/c_1 t^i | <= | c_0/c_1 | + | \sum_{i >= 2} c_i/c_1 t^i |
///         <= | c_0/c_1 | + \sum_{i >= 2} | c_i/c_1 | R^i
/// ```
/// Thus, for any `R` where `|c_0| + \sum_{i >= 2} |c_i| R^i <= R |c_1|`, we know that `T` maps the
/// disk of radius `R` to within the disk of radius `R`, and thus must have a fixed point.
/// In other words, there is a root of `f` within distance `R` to `x`.
#[instrument(skip_all, level = "trace")]
fn bound_distance_to_root<P>(
    approx_root: Complex64El,
    CCX: P,
    poly: &El<P>,
    poly_deg: usize,
) -> Result<f64, PrecisionError>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingStore<P>: RingStore<Ring = Complex64Base>,
{
    let CC = Complex64::RING;
    let f = poly;
    let f_prime = derive_poly(&CCX, &f);

    // Computes `\sum_{i >= 2} |c_i| R^i`
    let taylor_series_term = |within_radius: f64| {
        let abs_taylor_series_coeffs = (0..=poly_deg).scan(poly.clone(), |current, i| {
            let result = CC.abs(CCX.evaluate(current, &approx_root, CC.identity()));
            CCX.inclusion()
                .mul_assign_map(current, CC.from_f64(1.0 / (i as f64 + 1.0)));
            *current = derive_poly(&CCX, &current);
            return Some(result);
        });
        return abs_taylor_series_coeffs
            .enumerate()
            .skip(2)
            .map(|(i, c)| c * within_radius.powi(i as i32))
            .sum::<f64>();
    };

    let f_eval_upper_bound = CC.abs(CCX.evaluate(&f, &approx_root, CC.identity()))
        + absolute_error_of_poly_eval(&CCX, &f, poly_deg, approx_root, 0.0);
    let f_prime_eval_norm = CC.abs(CCX.evaluate(&f_prime, &approx_root, CC.identity()));
    let estimated_radius = f_eval_upper_bound / f_prime_eval_norm;
    if !estimated_radius.is_finite() {
        return Err(PrecisionError);
    }
    debug_assert!(estimated_radius >= 0.0);

    let try_radiuses = (-5..=10).map(|i| estimated_radius * 1.5f64.powi(i));

    let result = try_radiuses
        .filter(|R| taylor_series_term(*R) + f_eval_upper_bound <= R * f_prime_eval_norm)
        .min_by(f64::total_cmp)
        .ok_or(PrecisionError)?;
    return Ok(result);
}

#[instrument(skip_all, level = "trace")]
fn newton_with_initial<P>(
    poly_ring: P,
    f: &El<P>,
    poly_deg: usize,
    initial: El<Complex64>,
) -> Result<(El<Complex64>, f64), PrecisionError>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingStore<P>: RingStore<Ring = Complex64Base>,
{
    let CC = Complex64::RING;
    let f_prime = derive_poly(&poly_ring, f);
    let mut current = initial;
    for _ in 0..NEWTON_ITERATIONS {
        current = CC.sub(
            current,
            CC.div(
                &poly_ring.evaluate(f, &current, CC.identity()),
                &poly_ring.evaluate(&f_prime, &current, CC.identity()),
            ),
        );
    }
    return Ok((current, bound_distance_to_root(current, poly_ring, f, poly_deg)?));
}

/// Finds an approximation to a complex root of the given squarefree integer polynomial.
///
/// Prefer [`find_approximate_complex_root()`], since that function checks that the polynomial is
/// indeed squarefree.
#[instrument(skip_all, level = "trace")]
fn find_approximate_complex_root_promise_is_squarefree<P>(
    poly_ring: P,
    f: &El<P>,
    poly_deg: usize,
) -> Result<(El<Complex64>, f64), PrecisionError>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    BaseRingStore<P>: RingStore<Ring = Complex64Base>,
{
    let mut rng = Rand64::new(0);
    let CC = Complex64::RING;
    assert!(
        poly_ring
            .terms(f)
            .all(|(c, _): (&Complex64El, _)| CC.re(*c).is_finite() && CC.im(*c).is_finite())
    );
    let (approx_root, distance) = (0..PROBABILISTIC_REPETITIONS)
        .filter_map(|_| {
            let starting_point_unscaled = CC.add(
                // this cast might wrap around i64::MAX, so also produces negative values
                CC.from_f64((rng.rand_u64() as i64) as f64 / i64::MAX as f64),
                CC.mul(
                    Complex64::I,
                    CC.from_f64((rng.rand_u64() as i64) as f64 / i64::MAX as f64),
                ),
            );
            let scale = (rng.rand_u64() % (2 * NEWTON_MAX_SCALE as u64)) as i32 - NEWTON_MAX_SCALE as i32;
            let starting_point = CC.mul(starting_point_unscaled, CC.from_f64(2.0f64.powi(scale)));
            let (point, distance) = newton_with_initial(&poly_ring, f, poly_deg, starting_point).ok()?;
            return Some((point, distance));
        })
        .min_by(|(_, r1), (_, r2)| match (r1.is_finite(), r2.is_finite()) {
            (false, false) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => f64::total_cmp(r1, r2),
        })
        .ok_or(PrecisionError)?;
    return Ok((approx_root, distance));
}

/// Finds an approximation to a complex root of the given squarefree integer polynomial.
///
/// This function does not try to be as efficient as possible, but instead tries
/// to avoid (or at least detect) as many numerical problems as possible.
///
/// The first return value is an approximation to the root of the polynomial, and the
/// second return value is an upper bound to the distance to the exact root.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn find_approximate_complex_root<P>(ZZX: P, f: &El<P>) -> Result<(El<Complex64>, f64), PrecisionError>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    <BaseRingStore<P> as RingStore>::Ring: IntegerRing,
{
    assert!(ZZX.degree(f).unwrap_or(0) > 0);
    let CC = Complex64::RING;
    assert!(PolyTFracGCDRing::is_squarefree(&ZZX, f));
    let CCX = DensePolyRing::new(CC, "X");
    return find_approximate_complex_root_promise_is_squarefree(
        &CCX,
        &CCX.lifted_hom(&ZZX, CC.can_hom(ZZX.base_ring()).unwrap()).map_ref(f),
        ZZX.degree(f).unwrap(),
    );
}

/// Finds an approximation to all complex roots of the given integer polynomial.
///
/// This function does not try to be as efficient as possible, but instead tries
/// to avoid (or at least detect) as many numerical problems as possible.
/// However, the task of finding all roots has quite bad numerical stability, especially
/// if some of the roots are close together. Hence, this is likely to fail for polynomials
/// with large degrees (say > 100) or very large coefficients.
///
/// The first component of each returned tuple is an approximation to a root of the
/// polynomial, and the second component is an upper bound to the distance to exact root.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn find_all_approximate_complex_roots<P>(ZZX: P, poly: &El<P>) -> Result<Vec<(El<Complex64>, f64)>, PrecisionError>
where
    P: RingStore,
    P::Ring: PolyRing + DivisibilityRing,
    <BaseRingStore<P> as RingStore>::Ring: IntegerRing,
{
    assert!(ZZX.degree(poly).unwrap_or(0) > 0);
    let CC = Complex64::RING;
    let ZZ_to_CC = CC.can_hom(ZZX.base_ring()).unwrap();
    assert!(PolyTFracGCDRing::is_squarefree(&ZZX, poly));
    let CCX = DensePolyRing::new(CC, "X");
    let ZZX_to_CCX = CCX.lifted_hom(&ZZX, ZZ_to_CC);

    let d = ZZX.degree(poly).unwrap();
    let f = ZZX_to_CCX.map_ref(poly);
    let mut remaining_poly = f.clone();
    let mut result = Vec::new();
    for i in 0..ZZX.degree(&poly).unwrap() {
        let (next_root_initial, _) = find_approximate_complex_root_promise_is_squarefree(&CCX, &remaining_poly, d - i)?;
        let (next_root, distance) = newton_with_initial(&CCX, &f, d, next_root_initial)?;
        if result.iter().any(|(prev_root, prev_distance)| {
            if CC.abs(CC.sub(*prev_root, next_root)) <= distance + prev_distance {
                true
            } else {
                false
            }
        }) {
            return Err(PrecisionError);
        }
        result.push((next_root, distance));
        let mut new_remaining_poly = CCX.zero();
        for j in (1..=(d - i)).rev() {
            let lc = *CCX.coefficient_at(&remaining_poly, j);
            CCX.get_ring()
                .add_assign_from_terms(&mut new_remaining_poly, [(lc, j - 1)]);
            CCX.get_ring()
                .add_assign_from_terms(&mut remaining_poly, [(CC.mul(lc, next_root), j - 1)]);
        }
        remaining_poly = new_remaining_poly;
    }
    // just some canonical order to make tests and debugging easier
    result.sort_unstable_by(|(l, _), (r, _)| {
        f64::total_cmp(&(CC.re(*l) + CC.im(*l) * 0.000001), &(CC.re(*r) + CC.im(*r) * 0.000001))
    });
    return Ok(result);
}

#[cfg(test)]
use std::f64::consts::PI;

#[cfg(test)]
use crate::algorithms::cyclotomic::cyclotomic_polynomial;
#[cfg(test)]
use crate::ring_properties::pid::PrincipalIdealRingStore;

#[test]
fn test_find_approximate_complex_root() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let ZZX = DensePolyRing::new(&ZZ, "X");
    let CC = Complex64::RING;

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f).unwrap();
    assert!(radius <= 0.000000001);
    assert!(CC.abs(CC.sub(Complex64::I, root)) <= radius || CC.abs(CC.add(Complex64::I, root)) <= radius);

    let [f] = ZZX.with_wrapped_indeterminate(|X| [100000000 * X.pow_ref(2) + 1]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f).unwrap();
    assert!(radius <= 0.000000001);
    assert!(
        CC.abs(CC.sub(CC.mul(Complex64::I, CC.from_f64(0.0001)), root)) <= radius
            || CC.abs(CC.add(CC.mul(Complex64::I, CC.from_f64(0.0001)), root)) <= radius
    );

    let f = cyclotomic_polynomial(&ZZX, 105);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f).unwrap();
    assert!(radius <= 0.000000001);
    let root_of_unity = |k, n| CC.exp(CC.mul(CC.from_f64(2.0 * PI * k as f64 / n as f64), Complex64::I));
    assert!(
        (0..105)
            .filter(|k| ZZi64.ideal_gen(k, &105) == 1)
            .any(|k| CC.abs(CC.sub(root_of_unity(k, 105), root)) <= radius)
    );

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 6 * X.pow_ref(2) - 11]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f).unwrap();
    assert!(radius <= 0.000000001);
    let expected = [
        CC.from_f64(-2.7335207983477241859817441249894389670263693372196541057936155022),
        CC.from_f64(2.7335207983477241859817441249894389670263693372196541057936155022),
        CC.mul(
            CC.from_f64(-1.2133160985495821701841076107103126272177604441592580488679327999),
            Complex64::I,
        ),
        CC.mul(
            CC.from_f64(1.2133160985495821701841076107103126272177604441592580488679327999),
            Complex64::I,
        ),
    ];
    assert!(expected.iter().any(|x| CC.abs(CC.sub(*x, root)) <= radius));
}

#[test]
fn test_find_all_approximate_complex_roots() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let ZZX = DensePolyRing::new(&ZZ, "X");
    let CC = Complex64::RING;

    // let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 7 * X + 100]);
    // let expected = [
    //     CC.from_f64(-5.1425347350362689414102462079341122827700721701123021518405977272),
    //     CC.add(
    //         CC.from_f64(2.5712673675181344707051231039670561413850360850561510759202988636),
    //         CC.mul(
    //             CC.from_f64(-3.5824918179656616775077885147170618458138064112538190024361795659),
    //             Complex64::I,
    //         ),
    //     ),
    //     CC.add(
    //         CC.from_f64(2.5712673675181344707051231039670561413850360850561510759202988636),
    //         CC.mul(
    //             CC.from_f64(3.5824918179656616775077885147170618458138064112538190024361795659),
    //             Complex64::I,
    //         ),
    //     ),
    // ];
    // let actual = find_all_approximate_complex_roots(&ZZX, &f).unwrap();
    // for (expected, (actual, dist)) in expected.iter().copied().zip(actual.iter().copied()) {
    //     assert!(dist < 0.000000001);
    //     assert!(CC.abs(CC.sub(actual, expected)) <= dist);
    // }

    let root_of_unity = |k, n| CC.exp(CC.mul(CC.from_f64(2.0 * PI * k as f64 / n as f64), Complex64::I));
    let f = cyclotomic_polynomial(&ZZX, 105);
    let expected = (1..=52)
        .rev()
        .filter(|i| ZZi64.ideal_gen(i, &105) == 1)
        .flat_map(|i| [root_of_unity(105 - i, 105), root_of_unity(i, 105)])
        .chain([CC.one()]);
    let actual = find_all_approximate_complex_roots(&ZZX, &f).unwrap();
    assert_eq!(48, actual.len());
    for (expected, (actual, dist)) in expected.zip(actual.iter().copied()) {
        assert!(dist < 0.000000001);
        assert!(CC.abs(CC.sub(actual, expected)) <= dist);
    }
}
