use dense_poly::DensePolyRing;

use crate::algorithms::int_factor::is_prime_power;
use crate::integer::*;
use crate::primitive_int::StaticRing;
use crate::rings::finite::*;
use crate::field::*;
use crate::ring::*;
use crate::rings::finite_field::*;
use crate::rings::poly::*;
use crate::pid::*;
use crate::divisibility::*;
use crate::homomorphism::*;

///
/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
/// 
/// The returned polynomial is always monic, and with this restriction, it
/// is unique.
/// 
#[stability::unstable(feature = "enable")]
pub fn finite_field_poly_squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
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
        return finite_field_poly_squarefree_part(poly_ring, &base_poly);
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
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FiniteFieldSpecializable
{
    if let Ok(field) = poly_ring.base_ring().as_finite_field() {
        let new_poly_ring = DensePolyRing::new(&field, "X");
        let base_iso = field.can_iso(poly_ring.base_ring()).unwrap();
        let iso = new_poly_ring.into_lifted_hom(&poly_ring, base_iso.inv());
        let poly = iso.map_ref(poly);

        let result = finite_field_poly_squarefree_part(&new_poly_ring, &poly);

        let map_back = poly_ring.lifted_hom(&poly_ring, &base_iso);
        return Some(map_back.map(result));
    } else {
        return None;
    }
}
