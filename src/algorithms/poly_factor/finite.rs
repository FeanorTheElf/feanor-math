use tracing::{Level, event, instrument};

use crate::algorithms::poly_factor::cantor_zassenhaus::{
    cantor_zassenhaus, cantor_zassenhaus_even, distinct_degree_factorization,
};
use crate::algorithms::poly_gcd::finite::poly_squarefree_part_finite_field;
use crate::prelude::*;
use crate::ring_impls::poly::*;

/// Factors a polynomial with coefficients in a finite field.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_factor_finite_field<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<BaseRingStore<P>>)
where
    P: RingStore,
    P::Ring: PolyRing + EuclideanRing,
    <BaseRingStore<P> as RingStore>::Ring: FiniteRing + Field,
{
    assert!(!poly_ring.is_zero(&f));
    let even_char = ZZbig.is_even(&poly_ring.base_ring().characteristic(&ZZbig).unwrap());

    event!(
        Level::TRACE,
        poly_deg = poly_ring.degree(f).unwrap(),
        field_size_bits = ZZbig
            .abs_log2_ceil(&poly_ring.base_ring().size(ZZbig).unwrap())
            .unwrap()
    );

    let mut result = Vec::new();
    let mut unit = poly_ring.base_ring().one();
    let mut el = f.clone();

    // we repeatedly remove the square-free part
    while !poly_ring.is_unit(&el) {
        let sqrfree_part = poly_squarefree_part_finite_field(&poly_ring, &el);
        assert!(!poly_ring.is_unit(&sqrfree_part));

        // factor the square-free part into distinct-degree factors
        let distinct_degree_factors = distinct_degree_factorization(&poly_ring, sqrfree_part.clone());
        for (d, factor_d) in distinct_degree_factors.into_iter().enumerate() {
            let mut stack = Vec::new();
            stack.push(factor_d);

            // and finally extract each individual factor
            while let Some(mut current) = stack.pop() {
                current = poly_ring.normalize(current).0;

                if poly_ring.is_one(&current) {
                    continue;
                } else if poly_ring.degree(&current) == Some(d) {
                    // add to result
                    let mut found = false;
                    for (factor, power) in &mut result {
                        if poly_ring.eq_el(factor, &current) {
                            *power += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        result.push((current, 1));
                    }
                } else if even_char {
                    let factor = cantor_zassenhaus_even(&poly_ring, current.clone(), d);
                    stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                    stack.push(factor);
                } else {
                    let factor = cantor_zassenhaus(&poly_ring, current.clone(), d);
                    stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                    stack.push(factor);
                }
            }
        }
        el = poly_ring.checked_div(&el, &sqrfree_part).unwrap();
    }
    poly_ring
        .base_ring()
        .mul_assign_ref(&mut unit, poly_ring.coefficient_at(&el, 0));
    debug_assert!(poly_ring.base_ring().is_unit(&unit));
    return (result, unit);
}
