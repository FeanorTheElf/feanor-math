use std::mem::swap;

use crate::algorithms::eea::eea;
use crate::algorithms::int_factor::is_prime_power;
use crate::algorithms::poly_div::poly_div_rem_finite_reduced;
use crate::algorithms::poly_div::PolyDivRemReducedError;
use crate::computation::ComputationController;
use crate::primitive_int::*;
use crate::ring::*;
use crate::rings::poly::*;
use crate::field::*;
use crate::pid::*;
use crate::divisibility::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::rings::finite::*;

///
/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_finite_field<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field,
        Controller: ComputationController
{
    assert!(!poly_ring.is_zero(&poly));
    let squarefree_part = poly_squarefree_part_finite_field(poly_ring, poly, controller.clone());
    if poly_ring.degree(&squarefree_part).unwrap() == poly_ring.degree(&poly).unwrap() {
        return vec![(squarefree_part, 1)];
    } else {
        let square_part = poly_ring.checked_div(&poly, &squarefree_part).unwrap();
        let square_part_decomposition = poly_power_decomposition_finite_field(poly_ring, &square_part, controller);
        let mut result = square_part_decomposition;
        let mut degree = 0;
        for (g, k) in &mut result {
            *k += 1;
            degree += poly_ring.degree(g).unwrap() * *k;
        }
        if degree != poly_ring.degree(&poly).unwrap() {
            let remaining_part = poly_ring.checked_div(&poly, &poly_ring.prod(result.iter().map(|(g, e)| poly_ring.pow(poly_ring.clone_el(g), *e)))).unwrap();
            result.insert(0, (poly_ring.normalize(remaining_part), 1));
        }
        return result;
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
pub fn poly_squarefree_part_finite_field<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> El<P>
    where P: RingStore,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field,
        Controller: ComputationController
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
        return poly_squarefree_part_finite_field(poly_ring, &base_poly, controller);
    } else {
        let square_part = poly_ring.ideal_gen(poly, &derivate);
        let result = poly_ring.checked_div(poly, &square_part).unwrap();
        return poly_ring.normalize(result);
    }
}

const FAST_POLY_EEA_THRESHOLD: usize = 32;

///
/// Computes linearly independent vectors `(s, t)` and `(s', t')` such that
/// `s * lhs + t * rhs` and `s' * lhs + t' * rhs` are both of degree at most
/// `target_deg`.
/// 
/// The degrees of `s, t, s', t'` are bounded as
/// ```text
///   deg(s) < deg(rhs) - deg(s * lhs + t * rhs)
///   deg(t) < deg(lhs) - deg(s * lhs + t * rhs)
///   deg(s') < deg(rhs) - deg(s' * lhs + t' * rhs)
///   deg(t') < deg(lhs) - deg(s' * lhs + t' * rhs)
/// ```
///
fn partial_eea<P>(ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize) -> [El<P>; 4]
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    if ring.is_zero(&lhs) || ring.is_zero(&rhs) {
        return [ring.one(), ring.zero(), ring.zero(), ring.one()];
    }
    let (mut a, mut b) = (ring.clone_el(&lhs), ring.clone_el(&rhs));
    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());
    
    if ring.degree(&a).unwrap() < ring.degree(&b).unwrap() {
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }

    while ring.degree(&a).unwrap() > target_deg && !ring.is_zero(&b) {
        debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &lhs), ring.mul_ref(&ta, &rhs))));
        debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &lhs), ring.mul_ref(&tb, &rhs))));

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quo, &tb));
        sa = ring.sub(sa, ring.mul_ref_snd(quo, &sb));
        a = rem;

        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
        
        debug_assert_eq!(ring.degree(&sb).unwrap(), ring.degree(&rhs).unwrap() - ring.degree(&a).unwrap());
        debug_assert_eq!(ring.degree(&tb).unwrap(), ring.degree(&lhs).unwrap() - ring.degree(&a).unwrap());
    }
    return [sa, ta, sb, tb];
}

#[stability::unstable(feature = "enable")]
pub fn fast_poly_eea<P, Controller>(poly_ring: P, lhs: &El<P>, rhs: &El<P>, controller: Controller) -> (El<P>, El<P>, El<P>)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        Controller: ComputationController
{
    fn fast_poly_eea_rec<P, Controller>(poly_ring: P, lhs: El<P>, rhs: El<P>, controller: Controller) -> (El<P>, El<P>, El<P>, El<P>, El<P>)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
            Controller: ComputationController
    {
        unimplemented!()
    }

    return unimplemented!();
}

///
/// Computes the gcd of two polynomials over a finite and reduced ring.
/// 
/// This is well-defined, since a finite reduced ring is always a product of
/// finite fields.
/// 
/// If the ring is not reduced, this function may fail and return `Err(nil)`, where
/// `nil` is a nilpotent element of the ring. However, as long as the gcd of `lhs` and
/// `rhs` exists, this function may alternatively return it, even in cases where the ring is 
/// not reduced (note however that over a non-reduced ring, the gcd does not always exist).
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_gcd_finite_reduced<P>(poly_ring: P, mut lhs: El<P>, mut rhs: El<P>) -> Result<El<P>, El<<P::Type as RingExtension>::BaseRing>>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + PrincipalIdealRing
{
    while !poly_ring.is_zero(&rhs) {
        match poly_div_rem_finite_reduced(poly_ring, poly_ring.clone_el(&lhs), &rhs) {
            Ok((_q, r)) => {
                lhs = r;
                std::mem::swap(&mut lhs, &mut rhs);
            },
            Err(PolyDivRemReducedError::NotReduced(nilpotent)) => return Err(nilpotent),
            Err(PolyDivRemReducedError::NotDivisibleByContent(content_rhs)) => {
                // we find a decomposition `R ~ R/c x R/Ann(c)` for the content `c` of `rhs`.
                // clearly the gcd must be `lhs` modulo `c` (since `rhs = 0 mod c`); furthermore,
                // modulo `Ann(c)` the content `c` is a unit, so `gcd(lhs, rhs) = gcd(c * lhs, rhs) mod Ann(c)`
                let content_ann = poly_ring.base_ring().annihilator(&content_rhs);
                if !poly_ring.base_ring().is_unit(&poly_ring.base_ring().ideal_gen(&content_rhs, &content_ann)) {
                    return Err(poly_ring.base_ring().annihilator(&poly_ring.base_ring().ideal_gen(&content_rhs, &content_ann)));
                }
                let mod_content_gcd = poly_gcd_finite_reduced(poly_ring, poly_ring.inclusion().mul_ref_map(&lhs, &content_rhs), rhs)?;
                debug_assert!(poly_ring.terms(&mod_content_gcd).all(|(c, _)| poly_ring.base_ring().divides(c, &content_rhs)));
                return Ok(poly_ring.add(
                    poly_ring.inclusion().mul_ref_map(&lhs, &content_ann),
                    mod_content_gcd
                ));
            }
        }
    }
    return Ok(lhs);
}

#[cfg(test)]
use crate::rings::zn::zn_64;
#[cfg(test)]
use crate::rings::zn::zn_rns::*;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::seq::VectorView;
#[cfg(test)]
use crate::algorithms::poly_div::poly_checked_div_finite_reduced;

#[test]
fn test_poly_gcd_finite_reduced() {
    let base_ring = Zn::new([5, 7, 11, 13].into_iter().map(|p| zn_64::Zn::new(p).as_field().ok().unwrap()).collect(), StaticRing::<i64>::RING);
    let poly_ring = DensePolyRing::new(&base_ring, "X");
    let component_poly_rings: [_; 4] = std::array::from_fn(|i| DensePolyRing::new(base_ring.at(i), "X"));

    let [f0, g0, expected0] = component_poly_rings[0].with_wrapped_indeterminate(|X| [
        (X.pow_ref(2) + 2) * (X.pow_ref(3) + X + 1),
        (X + 1) * (X + 2) * (X.pow_ref(3) + X + 1),
        X.pow_ref(3) + X + 1
    ]);

    let f1 = component_poly_rings[1].zero();
    let [g1, expected1] = component_poly_rings[1].with_wrapped_indeterminate(|X| [
        X.pow_ref(3) + X + 1,
        X.pow_ref(3) + X + 1
    ]);

    let f2 = component_poly_rings[2].int_hom().map(1);
    let g2 = component_poly_rings[2].zero();
    let expected2 = component_poly_rings[2].one();
    
    let f3 = component_poly_rings[3].zero();
    let g3 = component_poly_rings[3].zero();
    let expected3 = component_poly_rings[3].zero();

    fn reconstruct<'a, 'b, R>(polys: [El<DensePolyRing<&'a R>>; 4], poly_rings: &[DensePolyRing<&'a R>; 4], poly_ring: &DensePolyRing<&'b Zn<R, StaticRing<i64>>>) -> El<DensePolyRing<&'b Zn<R, StaticRing<i64>>>>
        where R: RingStore,
            R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
    {
        poly_ring.from_terms((0..10).map(|i| (poly_ring.base_ring().from_congruence(polys.iter().zip(poly_rings.iter()).map(|(f, P)| P.base_ring().clone_el(P.coefficient_at(f, i)))), i)))
    }
    
    let f = reconstruct([f0, f1, f2, f3], &component_poly_rings, &poly_ring);
    let g = reconstruct([g0, g1, g2, g3], &component_poly_rings, &poly_ring);
    let expected = reconstruct([expected0, expected1, expected2, expected3], &component_poly_rings, &poly_ring);
    let actual = poly_gcd_finite_reduced(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g)).ok().unwrap();

    assert!(poly_checked_div_finite_reduced(&poly_ring, poly_ring.clone_el(&actual), poly_ring.clone_el(&expected)).ok().unwrap().is_some());
    assert!(poly_checked_div_finite_reduced(&poly_ring, poly_ring.clone_el(&expected), poly_ring.clone_el(&actual)).ok().unwrap().is_some());
}

#[test]
fn test_partial_eea() {
    let field = zn_64::Zn::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
        X.pow_ref(10) + X.pow_ref(6) + 1,
    ]);

    for k in (1..10).rev() {
        let [s1, t1, s2, t2] = partial_eea(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g), k);
        let a = poly_ring.add(poly_ring.mul_ref(&s1, &f), poly_ring.mul_ref(&t1, &g));
        let b = poly_ring.add(poly_ring.mul_ref(&s2, &f), poly_ring.mul_ref(&t2, &g));
        assert_eq!(k, poly_ring.degree(&a).unwrap());
        assert_eq!(k - 1, poly_ring.degree(&b).unwrap());
        assert!(poly_ring.degree(&s1).is_none() || poly_ring.degree(&s1).unwrap() < 10 - poly_ring.degree(&a).unwrap());
        assert!(poly_ring.degree(&t1).is_none() || poly_ring.degree(&t1).unwrap() < 9 - poly_ring.degree(&a).unwrap());
        assert!(poly_ring.degree(&s2).is_none() || poly_ring.degree(&s2).unwrap() < 10 - poly_ring.degree(&b).unwrap());
        assert!(poly_ring.degree(&t2).is_none() || poly_ring.degree(&t2).unwrap() < 9 - poly_ring.degree(&b).unwrap());
    }
}