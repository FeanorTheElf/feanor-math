use crate::algorithms::poly_gcd::finite::poly_squarefree_part_finite_field;
use crate::divisibility::*;
use crate::field::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::finite::*;
use crate::rings::poly::*;
use crate::homomorphism::SelfIso;
use super::cantor_zassenhaus;

///
/// Factors a polynomial with coefficients in a finite field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_factor_finite_field<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field + SelfIso
{
    assert!(!poly_ring.is_zero(&f));
    let even_char = BigIntRing::RING.is_even(&poly_ring.base_ring().characteristic(&BigIntRing::RING).unwrap());

    let mut result = Vec::new();
    let mut unit = poly_ring.base_ring().one();
    let mut el = poly_ring.clone_el(&f);

    // we repeatedly remove the square-free part
    while !poly_ring.is_unit(&el) {

        let sqrfree_part = poly_squarefree_part_finite_field(&poly_ring, &el);
        assert!(!poly_ring.is_unit(&sqrfree_part));

        // factor the square-free part into distinct-degree factors
        let distinct_degree_factors = cantor_zassenhaus::distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part));
        for (d, factor_d) in distinct_degree_factors.into_iter().enumerate() {
            let mut stack = Vec::new();
            stack.push(factor_d);
            
            // and finally extract each individual factor
            while let Some(mut current) = stack.pop() {
                current = poly_ring.normalize(current);

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
                    let factor = cantor_zassenhaus::cantor_zassenhaus_even(&poly_ring, poly_ring.clone_el(&current), d);
                    stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                    stack.push(factor);
                } else {
                    let factor = cantor_zassenhaus::cantor_zassenhaus(&poly_ring, poly_ring.clone_el(&current), d);
                    stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                    stack.push(factor);
                }
            }
        }
        el = poly_ring.checked_div(&el, &sqrfree_part).unwrap();
    }
    poly_ring.base_ring().mul_assign_ref(&mut unit, poly_ring.coefficient_at(&el, 0));
    debug_assert!(poly_ring.base_ring().is_unit(&unit));

    return (result, unit);
}
