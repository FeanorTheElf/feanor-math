use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::divisibility::Domain;
use crate::pid::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::poly::*;

///
/// Computes the polynomial division of `lhs` by `rhs`, i.e. `lhs = q * rhs + r` with
/// `deg(r) < deg(rhs)`. 
/// 
/// This requires a function `left_div_lc` that computes the division of an element of the 
/// base ring by the leading coefficient of `rhs`. If the base ring is a field, this can
/// just be standard division. In other cases, this depends on the exact situation you are
/// in - e.g. `rhs` might be monic or in in a specific context, it might be guaranteed that the 
/// division always works. If this is not the case, look also at [`poly_div_rem_domain()`], which
/// implicitly performs the polynomial division over the field of fractions.
/// 
pub fn poly_div_rem<P, S, F, E, H>(mut lhs: El<P>, rhs: &El<S>, lhs_ring: P, rhs_ring: S, mut left_div_lc: F, hom: H) -> Result<(El<P>, El<P>), E>
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
                        hom.mul_assign_ref_map(&mut subtract, c);
                        return (lhs_ring.base_ring().negate(subtract), i + j);
                    })
            );
        }
        lhs_ring.get_ring().add_assign_from_terms(&mut result, std::iter::once((quo, i)));
    }
    return Ok((result, lhs));
}

///
/// Computes the remainder of the polynomial division of `lhs` by `rhs`, i.e. `r` of 
/// degree `deg(r) < deg(rhs)` such that there exists `q` with `lhs = q * rhs + r`.
/// If you also require `q`, consider using [`poly_div_rem()`].
/// 
/// Since we don't have to compute `q`, this might be faster than [`poly_div_rem()`].
/// 
/// This requires a function `left_div_lc` that computes the division of an element of the 
/// base ring by the leading coefficient of `rhs`. If the base ring is a field, this can
/// just be standard division. In other cases, this depends on the exact situation you are
/// in - e.g. `rhs` might be monic or in in a specific context, it might be guaranteed that the 
/// division always works. If this is not the case, look also at [`poly_div_domain()`], which
/// implicitly performs the polynomial division over the field of fractions.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_rem<P, S, F, E, H>(mut lhs: El<P>, rhs: &El<S>, lhs_ring: P, rhs_ring: S, mut left_div_lc: F, hom: H) -> Result<El<P>, E>
    where S: PolyRingStore,
        S::Type: PolyRing,
        P: PolyRingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        H: Homomorphism<<<S::Type as RingExtension>::BaseRing as RingStore>::Type, <<P::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: FnMut(&El<<P::Type as RingExtension>::BaseRing>) -> Result<El<<P::Type as RingExtension>::BaseRing>, E>
{
    assert!(rhs_ring.degree(rhs).is_some());
    assert!(lhs_ring.base_ring().get_ring() == hom.codomain().get_ring());
    assert!(rhs_ring.base_ring().get_ring() == hom.domain().get_ring());

    let rhs_deg = rhs_ring.degree(rhs).unwrap();
    if lhs_ring.degree(&lhs).is_none() {
        return Ok(lhs_ring.zero());
    }
    let lhs_deg = lhs_ring.degree(&lhs).unwrap();
    if lhs_deg < rhs_deg {
        return Ok(lhs_ring.zero());
    }
    for i in (0..(lhs_deg + 1 - rhs_deg)).rev() {
        let quo = left_div_lc(lhs_ring.coefficient_at(&lhs, i +  rhs_deg))?;
        if !lhs_ring.base_ring().is_zero(&quo) {
            lhs_ring.get_ring().add_assign_from_terms(
                &mut lhs, 
                rhs_ring.terms(rhs)
                    .map(|(c, j)| {
                        let mut subtract = lhs_ring.base_ring().clone_el(&quo);
                        hom.mul_assign_ref_map(&mut subtract, c);
                        return (lhs_ring.base_ring().negate(subtract), i + j);
                    })
            );
        }
        lhs_ring.balance_poly(&mut lhs);
    }
    return Ok(lhs);
}

///
/// Computes `(q, r, a)` such that `a * lhs = q * rhs + r` and `deg(r) < deg(rhs)`.
/// The chosen factor `a` is in the base ring and is the smallest possible w.r.t.
/// divisibility.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_div_rem_domain<P>(ring: P, mut lhs: El<P>, rhs: &El<P>) -> (El<P>, El<P>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Domain + PrincipalIdealRing
{
    assert!(!ring.is_zero(rhs));
    let d = ring.degree(rhs).unwrap();
    let base_ring = ring.base_ring();
    let rhs_lc = ring.lc(rhs).unwrap();

    let mut current_scale = base_ring.one();
    let mut terms = Vec::new();
    while let Some(lhs_deg) = ring.degree(&lhs) {
        if lhs_deg < d {
            break;
        }
        let lhs_lc = base_ring.clone_el(ring.lc(&lhs).unwrap());
        let gcd = base_ring.ideal_gen(&lhs_lc, &rhs_lc);
        let additional_scale = base_ring.checked_div(&rhs_lc, &gcd).unwrap();

        base_ring.mul_assign_ref(&mut current_scale, &additional_scale);
        terms.iter_mut().for_each(|(c, _)| base_ring.mul_assign_ref(c, &additional_scale));
        ring.inclusion().mul_assign_map(&mut lhs, additional_scale);

        let factor = base_ring.checked_div(ring.lc(&lhs).unwrap(), rhs_lc).unwrap();
        ring.get_ring().add_assign_from_terms(&mut lhs,
            ring.terms(rhs).map(|(c, i)| (base_ring.negate(base_ring.mul_ref(c, &factor)), i + lhs_deg - d))
        );
        terms.push((factor, lhs_deg - d));
    }
    return (ring.from_terms(terms.into_iter()), lhs, current_scale);
}