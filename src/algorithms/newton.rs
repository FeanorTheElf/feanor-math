use std::cmp::Ordering;

use oorandom::Rand64;

use crate::algorithms::poly_gcd::squarefree_part::poly_squarefree_part_local;
use crate::computation::DontObserve;
use crate::field::FieldStore;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::integer::*;
use crate::ring::*;
use crate::primitive_int::StaticRing;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::rings::float_complex::Complex64;
use crate::rings::poly::*;

const NEWTON_MAX_SCALE: u32 = 10;
const ASSUME_RADIUS_TO_APPROX_RADIUS_FACTOR: f64 = 2.;

///
/// Finds an approximation to a complex root of the given integer polynomial.
/// 
/// This function does not try to be as efficient as possible, but instead tries
/// to avoid (or at least detect) as many numerical problems as possible.
/// 
/// The first return value is an approximation to the root of the polynomial, and the
/// second return value is an upper bound to the distance of the actual root.
/// 
#[stability::unstable(feature = "enable")]
pub fn find_approximate_complex_root<P>(poly_ring: P, el: &El<P>) -> (El<Complex64>, f64)
    where P: RingStore,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    let mut rng = Rand64::new(1);
    let CC = Complex64::RING;
    let hom = CC.can_hom(poly_ring.base_ring()).unwrap();
    assert!(poly_ring.checked_div(&poly_squarefree_part_local(&poly_ring, poly_ring.clone_el(el), DontObserve), el).is_some(), "polynomial must be square-free");
    let derivate = derive_poly(&poly_ring, el);
    let f_coeffs_f64 = (0..=poly_ring.degree(el).unwrap()).map(|i| poly_ring.base_ring().to_float_approx(poly_ring.coefficient_at(el, i))).collect::<Vec<_>>();
    let f_prime_coeffs_f64 = (0..=poly_ring.degree(&derivate).unwrap()).map(|i| poly_ring.base_ring().to_float_approx(poly_ring.coefficient_at(&derivate, i))).collect::<Vec<_>>();
    let eval = |x: El<Complex64>, coeffs: &[f64]| {
        let mut current = CC.zero();
        for c in coeffs.iter().rev() {
            current = CC.add(CC.from_f64(*c), CC.mul(x, current));
        }
        return current;
    };
    let eval_f = |x: El<Complex64>| eval(x, &f_coeffs_f64[..]);
    let eval_f_prime = |x: El<Complex64>| eval(x, &f_prime_coeffs_f64[..]);

    let (result, approx_radius) = (0..MAX_PROBABILISTIC_REPETITIONS).map(|_| {
        let starting_point_unscaled = CC.add(
            // this cast might wrap around i64::MAX, so also produces negative values
            CC.from_f64((rng.rand_u64() as i64) as f64 / i64::MAX as f64),
            CC.mul(Complex64::I, CC.from_f64((rng.rand_u64() as i64) as f64 / i64::MAX as f64))
        );
        let scale = (rng.rand_u64() % (2 * NEWTON_MAX_SCALE as u64)) as i32 - NEWTON_MAX_SCALE as i32;
        let starting_point = CC.mul(starting_point_unscaled, CC.from_f64(2.0f64.powi(scale)));

        let mut current = starting_point;
        for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
            current = CC.sub(current, CC.div(&eval_f(current), &eval_f_prime(current)));
        }

        // we expect the root to lie within this radius of current
        let approx_radius = CC.abs(eval_f(current)) / CC.abs(eval_f_prime(current));

        return (current, approx_radius);
    }).min_by(|(_, r1), (_, r2)| match (r1.is_finite(), r2.is_finite()) {
        (false, false) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => f64::total_cmp(r1, r2)
    }).unwrap();
    assert!(approx_radius.is_finite());
    assert!(approx_radius <= 1.);
    assert!(approx_radius >= 0.);

    // to describe how the polynomial behaves close to the root, use taylor series expansion
    let mut higher_derivates_at_point = Vec::new();
    let mut current = derive_poly(&poly_ring, &derivate);
    while !poly_ring.is_zero(&current) {
        higher_derivates_at_point.push(CC.abs(poly_ring.evaluate(&current, &result, &hom)));
        current = derive_poly(&poly_ring, &current);
    }
    // the question is: how much can the first derivative change within assume_radius to result?
    // this should be bounded by something sufficiently smaller than |f'(result)|, to guarantee that the
    // root is indeed within this area 
    let assume_radius = approx_radius * ASSUME_RADIUS_TO_APPROX_RADIUS_FACTOR;
    let max_change = higher_derivates_at_point.iter().enumerate()
        .map(|(i, c)| c * assume_radius.powi(i as i32 + 2) / factorial(&(i as i64 + 2), StaticRing::<i64>::RING) as f64)
        .sum::<f64>();
    assert!(max_change <= CC.abs(eval_f_prime(result)) * (ASSUME_RADIUS_TO_APPROX_RADIUS_FACTOR - 1.0) / ASSUME_RADIUS_TO_APPROX_RADIUS_FACTOR);

    return (result, assume_radius);
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::algorithms::cyclotomic::cyclotomic_polynomial;
#[cfg(test)]
use std::f64::consts::PI;
#[cfg(test)]
use crate::algorithms::eea::signed_gcd;

#[test]
fn test_newton() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");
    let CC = Complex64::RING;

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f);
    assert!(radius <= 0.000000001);
    assert!(
        CC.abs(CC.sub(Complex64::I, root)) <= radius ||
        CC.abs(CC.add(Complex64::I, root)) <= radius
    );

    let [f] = ZZX.with_wrapped_indeterminate(|X| [100000000 * X.pow_ref(2) + 1]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f);
    assert!(radius <= 0.000000001);
    assert!(
        CC.abs(CC.sub(CC.mul(Complex64::I, CC.from_f64(0.0001)), root)) <= radius ||
        CC.abs(CC.add(CC.mul(Complex64::I, CC.from_f64(0.0001)), root)) <= radius
    );
    
    let f = cyclotomic_polynomial(&ZZX, 105);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f);
    assert!(radius <= 0.000000001);
    assert!((0..105).filter(|k| signed_gcd(*k, 105, StaticRing::<i64>::RING) == 1).any(|k| CC.abs(CC.sub(CC.exp(CC.mul(CC.from_f64(2.0 * PI * k as f64 / 105.0), Complex64::I)), root)) <= radius));
    
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 6 * X.pow_ref(2) - 11]);
    let (root, radius) = find_approximate_complex_root(&ZZX, &f);
    assert!(radius <= 0.000000001);
    let expected = [
        CC.from_f64(-2.733520798347724185981744124989438967026369337219654105793615502203052155806604412202635382085483901190),
        CC.from_f64(2.7335207983477241859817441249894389670263693372196541057936155022030521558066044122026353820854839011906),
        CC.mul(CC.from_f64(-1.2133160985495821701841076107103126272177604441592580488679327999854151424818126554107447540704308999966020120078580), Complex64::I),
        CC.mul(CC.from_f64(1.21331609854958217018410761071031262721776044415925804886793279998541514248181265541074475407043089999660201200785805), Complex64::I)
    ];
    assert!(expected.iter().any(|x| CC.abs(CC.sub(*x, root)) <= radius));
}