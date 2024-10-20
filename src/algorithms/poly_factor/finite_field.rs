use crate::algorithms::int_factor::is_prime_power;
use crate::divisibility::*;
use crate::field::*;
use crate::integer::*;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::finite::*;
use crate::rings::poly::*;
use crate::homomorphism::Homomorphism;
use crate::specialization::*;
use super::cantor_zassenhaus;

///
/// Factors a polynomial with coefficients in a finite field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_factor_finite_field<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
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

        let sqrfree_part = poly_squarefree_part_finite_field(&poly_ring, &el);
        assert!(!poly_ring.is_unit(&sqrfree_part));

        // factor the square-free part into distinct-degree factors
        let squarefree_factorization = cantor_zassenhaus::distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part));
        for (d, factor_d) in squarefree_factorization.into_iter().enumerate() {
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

///
/// Factors the given polynomial, if the base field is a finite field.
/// Otherwise, `None` is returned.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_factor_if_finite_field<P>(poly_ring: P, f: &El<P>) -> Option<(Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FiniteRingSpecializable
{
    <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as FiniteRingSpecializable>::specialize(FactorPolyFiniteField { poly_ring: poly_ring.get_ring(), poly: poly_ring.clone_el(f) }).ok()
}

struct FactorPolyFiniteField<'a, P>
    where P: ?Sized + PolyRing + EuclideanRing,
        <P::BaseRing as RingStore>::Type: Field
{
    poly_ring: &'a P,
    poly: P::Element
}

impl<'a, P> FiniteRingOperation<<P::BaseRing as RingStore>::Type> for FactorPolyFiniteField<'a, P>
    where P: ?Sized + PolyRing + EuclideanRing,
        <P::BaseRing as RingStore>::Type: Field
{
    type Output = (Vec<(P::Element, usize)>, El<P::BaseRing>);

    fn execute(self) -> Self::Output
        where <P::BaseRing as RingStore>::Type: FiniteRing
    {
        poly_factor_finite_field(RingRef::new(self.poly_ring), &self.poly)
    }
}

///
/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
/// 
/// The returned polynomial is always monic, and with this restriction, it
/// is unique.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_finite_field<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(!poly_ring.is_zero(&poly));
    if poly_ring.degree(poly).unwrap() == 0 {
        return poly_ring.one();
    }
    let derivate = derive_poly(&poly_ring, poly);
    if poly_ring.is_zero(&derivate) {
        let q = poly_ring.base_ring().size(&BigIntRing::RING).unwrap();
        let (p, e) = is_prime_power(BigIntRing::RING, &q).unwrap();
        let p = int_cast(p, StaticRing::<i64>::RING, BigIntRing::RING) as usize;
        assert!(p > 0);
        let undo_frobenius = Frobenius::new(poly_ring.base_ring(), e - 1);
        let base_poly = poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| {
            debug_assert!(i % p == 0);
            (undo_frobenius.map_ref(c), i / p)
        }));
        return poly_squarefree_part_finite_field(poly_ring, &base_poly);
    } else {
        let square_part = poly_ring.ideal_gen(poly, &derivate);
        let result = poly_ring.checked_div(poly, &square_part).unwrap();
        return poly_ring.normalize(result);
    }
}

///
/// If the given polynomial ring is over a finite field, returns the result of [`finite_field_poly_squarefree_part()`].
/// Otherwise, `None` is returned.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_if_finite_field<P>(poly_ring: P, poly: &El<P>) -> Option<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FiniteRingSpecializable
{
    <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as FiniteRingSpecializable>::specialize(FiniteFieldPolySquarefreePart { poly_ring: poly_ring.get_ring(), poly: poly }).ok()
}

struct FiniteFieldPolySquarefreePart<'a, P>
    where P: ?Sized + PolyRing + PrincipalIdealRing,
        <P::BaseRing as RingStore>::Type: PerfectField + FiniteRingSpecializable
{
    poly_ring: &'a P,
    poly: &'a P::Element
}

impl<'a, P> FiniteRingOperation<<P::BaseRing as RingStore>::Type> for FiniteFieldPolySquarefreePart<'a, P>
    where P: ?Sized + PolyRing + PrincipalIdealRing,
        <P::BaseRing as RingStore>::Type: PerfectField + FiniteRingSpecializable
{
    type Output = P::Element;

    fn execute(self) -> Self::Output
        where <P::BaseRing as RingStore>::Type: FiniteRing
    {
        poly_squarefree_part_finite_field(RingRef::new(self.poly_ring), &self.poly)
    }
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use crate::rings::extension::FreeAlgebraStore;
#[cfg(test)]
use crate::rings::zn::zn_64;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;
#[cfg(test)]

#[test]
fn test_poly_squarefree_part_finite_field_multiplicity_p() {
    let ring = DensePolyRing::new(zn_64::Zn::new(5).as_field().ok().unwrap(), "X");
    let [f] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(10)]);
    let [g] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(2)]);
    let actual = poly_squarefree_part_finite_field(&ring, &f);
    assert_el_eq!(ring, g, actual);
}

#[test]
fn test_poly_squarefree_part_finite_field_galois_field() {
    let ring = DensePolyRing::new(GaloisField::new(2, 3), "X");
    let f = ring.from_terms([(ring.base_ring().pow(ring.base_ring().canonical_gen(), 2), 0), (ring.base_ring().one(), 2)]);
    let g = ring.from_terms([(ring.base_ring().canonical_gen(), 0), (ring.base_ring().one(), 1)]);
    let actual = poly_squarefree_part_finite_field(&ring, &f);
    assert_el_eq!(ring, g, actual);
}
