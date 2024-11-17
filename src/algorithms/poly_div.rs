use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::divisibility::Domain;
use crate::pid::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::finite::FiniteRing;
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
    where S: RingStore,
        S::Type: PolyRing,
        P: RingStore,
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
    where S: RingStore,
        S::Type: PolyRing,
        P: RingStore,
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
    where P: RingStore,
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

#[stability::unstable(feature = "enable")]
pub enum PolyDivRemReducedError<R>
    where R: ?Sized + RingBase
{
    NotReduced(R::Element),
    NotDivisibleByContent(R::Element)
}

///
/// Given polynomials `f, g` over a finite and reduced ring `R`, tries to compute the
/// polynomial division of `f` by `g`, i.e. values `q, r in R[X]` with `f = q g + r`
/// and `deg(r) < deg(g)`.
/// 
/// As opposed to the case when `R` is a field, it is possible that this fails if `cont(g)`
/// does not divide `cont(f)`. Hence, the possible results of this function are
///  - success, i.e. `q, r` such that `f = q g + r` and `deg(r) < deg(g)`
///  - [`PolyDivRemReducedError::NotDivisibleByContent`] with the content `c in R`, if `c` does not divide `cont(f)`
///  - [`PolyDivRemReducedError::NotReduced`] with a nilpotent element `x != 0` if `x^2 = 0`, which means that 
///    the ring is not reduced
/// 
/// Since a finite reduced ring is always a product of finite fields, one could (in theory)
/// compute the polynomial division in each field and reconstruct the result. However, this
/// function finds the result without computing the ring factors, which can be very difficult
/// in some situations (e.g. it might require factoring the modulus `n`).
/// 
/// Note that sometimes, even if `R` is not reduced, or `cont(g)` does not divide `cont(f)`, there
/// might still exist suitable `q, r`. In these cases, it is unspecified whether the function aborts
/// or returns suitable `q, r`.
/// 
/// # Why reduced?
/// 
/// Polynomial division as above makes sense over finite reduced rings (since they are always
/// a product of finite fields), but cannot be properly defined over unreduced rings anymore.
/// The reason is that the divisors of some polynomial `f` don't need to have degree `<= deg(f)`,
/// e.g.
/// ```text
///   1 = (5 X^2 + 5 X + 1) (-5 X^2 - 5 X + 1) mod 5^2
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_div_rem_finite_reduced<P>(ring: P, mut lhs: El<P>, rhs: &El<P>) -> Result<(El<P>, El<P>), PolyDivRemReducedError<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + PrincipalIdealRing
{
    assert!(!ring.is_zero(rhs));
    let rhs_deg = ring.degree(rhs).unwrap();
    let mut result = ring.zero();
    while ring.degree(&lhs).is_some() && ring.degree(&lhs).unwrap() >= rhs_deg {
        let lhs_deg = ring.degree(&lhs).unwrap();
        let lcf = ring.lc(&lhs).unwrap();
        let mut h = ring.zero();
        let mut annihilator = ring.base_ring().one();
        let mut i: i64 = rhs_deg as i64;
        let mut d = ring.base_ring().zero();
        while ring.base_ring().checked_div(lcf, &d).is_none() {
            if i == -1 {
                return Err(PolyDivRemReducedError::NotDivisibleByContent(d));
            }
            let (s, t, new_d) = ring.base_ring().extended_ideal_gen(&d, &ring.base_ring().mul_ref(&annihilator, ring.coefficient_at(&rhs, i as usize)));
            ring.inclusion().mul_assign_map(&mut h, s);
            ring.add_assign(&mut h, ring.from_terms([(ring.base_ring().mul_ref(&annihilator, &t), lhs_deg - i as usize)]));
            annihilator = ring.base_ring().annihilator(&new_d);
            d = new_d;
            i = i - 1;
            if !ring.base_ring().is_unit(&ring.base_ring().ideal_gen(&annihilator, &d)) {
                let nilpotent = ring.base_ring().annihilator(&ring.base_ring().ideal_gen(&annihilator, &d));
                debug_assert!(!ring.base_ring().is_zero(&nilpotent));
                debug_assert!(ring.base_ring().is_zero(&ring.base_ring().mul_ref(&nilpotent, &nilpotent)));
                return Err(PolyDivRemReducedError::NotReduced(nilpotent));
            }
        }
        ring.sub_assign(&mut lhs, ring.mul_ref(&h, rhs));
        ring.add_assign(&mut result, h);
    }
    return Ok((result, lhs));
}

#[cfg(test)]
use crate::rings::zn::zn_64::*;
#[cfg(test)]
use dense_poly::DensePolyRing;

#[test]
fn test_poly_div_rem_finite_reduced() {
    let base_ring = Zn::new(5 * 7 * 11);
    let ring = DensePolyRing::new(base_ring, "X");

    let [f, g, _q, _r] = ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(2),
        X.pow_ref(2) * 5 + X * 7 + 11,
        X * (-77) + 108,
        91 * X - 33
    ]);
    let (q, r) = poly_div_rem_finite_reduced(&ring, ring.clone_el(&f), &g).ok().unwrap();
    assert_eq!(1, ring.degree(&r).unwrap());
    assert_el_eq!(&ring, &f, ring.add(ring.mul(q, g), r));

    let [f, g] = ring.with_wrapped_indeterminate(|X| [
        5 * X.pow_ref(2),
        X * 5 * 11 + X * 7 * 11,
    ]);
    if let Err(PolyDivRemReducedError::NotDivisibleByContent(content)) = poly_div_rem_finite_reduced(&ring, f, &g) {
        assert!(base_ring.checked_div(&content, &base_ring.int_hom().map(11)).is_some());
        assert!(base_ring.checked_div(&base_ring.int_hom().map(11), &content).is_some());
    } else {
        assert!(false);
    }

    let base_ring = Zn::new(5 * 5 * 11);
    let ring = DensePolyRing::new(base_ring, "X");

    // note that `11 = (1 - 5 X - 5 X^2)(1 + 55 X + 55 X^2) mod 5^2 * 11`
    let [g] = ring.with_wrapped_indeterminate(|X| [
        1 - 5 * X - 5 * X.pow_ref(2)
    ]);
    let f = ring.from_terms([(base_ring.int_hom().map(11), 2)]);
    if let Err(PolyDivRemReducedError::NotReduced(nilpotent)) = poly_div_rem_finite_reduced(&ring, ring.clone_el(&f), &g) {
        assert!(!base_ring.is_zero(&nilpotent));
        assert!(base_ring.is_zero(&base_ring.pow(nilpotent, 2)));
    } else {
        assert!(false);
    }
}