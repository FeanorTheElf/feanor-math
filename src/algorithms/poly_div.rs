use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::poly::*;

pub fn sparse_poly_div<P, S, F, E, H>(mut lhs: El<P>, rhs: &El<S>, lhs_ring: P, rhs_ring: S, mut left_div_lc: F, hom: &H) -> Result<(El<P>, El<P>), E>
    where S: PolyRingStore,
        S::Type: PolyRing,
        P: PolyRingStore,
        P::Type: PolyRing,
        H: Homomorphism<<<S::Type as RingExtension>::BaseRing as RingStore>::Type, <<P::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: FnMut(&El<<P::Type as RingExtension>::BaseRing>) -> Result<El<<P::Type as RingExtension>::BaseRing>, E>
{
    assert!(rhs_ring.degree(rhs).is_some());
    assert!(lhs_ring.base_ring().get_ring() == hom.codomain().get_ring());
    assert!(rhs_ring.base_ring().get_ring() == hom.domain().get_ring());

    let rhs_deg = rhs_ring.degree(rhs).unwrap();
    if lhs_ring.degree(&lhs).is_none() {
        return Ok((lhs_ring.zero(), lhs));
    }
    let lhs_deg = lhs_ring.degree(&lhs).unwrap();
    if lhs_deg < rhs_deg {
        return Ok((lhs_ring.zero(), lhs));
    }
    let mut result = lhs_ring.zero();
    for i in (0..(lhs_deg + 1 - rhs_deg)).rev() {
        let quo = left_div_lc(lhs_ring.coefficient_at(&lhs, i +  rhs_deg))?;
        if !lhs_ring.base_ring().is_zero(&quo) {
            lhs_ring.get_ring().add_assign_from_terms(
                &mut lhs, 
                rhs_ring.terms(rhs)
                    .map(|(c, j)| {
                        let mut subtract = lhs_ring.base_ring().clone_el(&quo);
                        hom.mul_assign_map_ref(&mut subtract, c);
                        return (lhs_ring.base_ring().negate(subtract), i + j);
                    })
            );
        }
        lhs_ring.get_ring().add_assign_from_terms(&mut result, std::iter::once((quo, i)));
    }
    return Ok((result, lhs));
}