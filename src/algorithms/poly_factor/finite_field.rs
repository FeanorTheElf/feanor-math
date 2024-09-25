
use crate::algorithms::poly_squarefree::finite_field::finite_field_poly_squarefree_part;
use crate::divisibility::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::finite::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::specialization::*;
use super::cantor_zassenhaus;

///
/// Factors a polynomial with coefficients in a finite field.
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_over_finite_field<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(!poly_ring.is_zero(&f));
    let even_char = BigIntRing::RING.is_even(&poly_ring.base_ring().characteristic(&BigIntRing::RING).unwrap());

    let mut result = Vec::new();
    let mut unit = poly_ring.base_ring().one();
    let mut el = poly_ring.clone_el(&f);

    // we repeatedly remove the square-free part
    while !poly_ring.is_unit(&el) {

        let sqrfree_part = finite_field_poly_squarefree_part(&poly_ring, poly_ring.clone_el(&el));
        assert!(!poly_ring.is_unit(&sqrfree_part));

        // factor the square-free part into distinct-degree factors
        let squarefree_factorization = cantor_zassenhaus::distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part));
        for (d, factor_d) in squarefree_factorization.into_iter().enumerate() {
            let mut stack = Vec::new();
            stack.push(factor_d);
            
            // and finally extract each individual factor
            while let Some(mut current) = stack.pop() {
                // normalize current
                let lc = poly_ring.lc(&current).unwrap();
                poly_ring.base_ring().mul_assign_ref(&mut unit, lc);
                let lc_inv = poly_ring.base_ring().div(&poly_ring.base_ring().one(), lc);
                poly_ring.inclusion().mul_assign_ref_map(&mut current, &lc_inv);

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

///
/// Factors the given polynomial, if the base field is a finite field.
/// Otherwise, `None` is returned.
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_if_finite_field<P>(poly_ring: P, f: &El<P>) -> Option<(Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + SpecializeToFiniteField
{
    poly_ring.base_ring().get_ring().specialize_finite_field(FactorPolyFiniteField { poly_ring: poly_ring.get_ring(), poly: poly_ring.clone_el(f) }).ok()
}

struct FactorPolyFiniteField<'a, P>
    where P: ?Sized + PolyRing + EuclideanRing,
        <P::BaseRing as RingStore>::Type: Field
{
    poly_ring: &'a P,
    poly: P::Element
}

impl<'a, P> FiniteFieldOperation<<P::BaseRing as RingStore>::Type> for FactorPolyFiniteField<'a, P>
    where P: ?Sized + PolyRing + EuclideanRing,
        <P::BaseRing as RingStore>::Type: Field
{
    type Output<'d> = (Vec<(P::Element, usize)>, El<P::BaseRing>)
        where Self: 'd;

    fn execute<'d, F>(self, field: F) -> Self::Output<'d>
        where Self: 'd,
            F: 'd + RingStore,
            F::Type: FiniteRing + Field + CanIsoFromTo<<P::BaseRing as RingStore>::Type> + PerfectField + SpecializeToFiniteField
    {
        let poly_ring = DensePolyRing::new(&field, "X");
        let base_iso = field.can_iso(self.poly_ring.base_ring()).unwrap();
        let iso = poly_ring.lifted_hom(RingRef::new(self.poly_ring), base_iso.inv());
        let poly = iso.map(self.poly);

        let (result, unit) = factor_over_finite_field(&poly_ring, &poly);

        let map_back = RingRef::new(self.poly_ring).into_lifted_hom(&poly_ring, &base_iso);
        return (result.into_iter().map(|(f, e)| (map_back.map(f), e)).collect::<Vec<_>>(), base_iso.map(unit));
    }
}
