use crate::divisibility::DivisibilityRingStore;
use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::integer::int_cast;
use crate::pid::{EuclideanRing, PrincipalIdealRingStore};
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::zn::{ReductionMap, ZnRing, ZnRingStore};
use crate::ring::*;

use super::int_factor::is_prime_power;

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h`
/// modulo `p^e`.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift<P, R, S>(target_ring: &P, source_ring: &R, prime_ring: &S, f: &El<P>, factors: (&El<R>, &El<R>)) -> (El<P>, El<P>)
    where P: PolyRingStore, P::Type: PolyRing,
        R: PolyRingStore, R::Type: PolyRing,
        S: PolyRingStore, S::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        <<R as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<R as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        <<S as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<S as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field
{
    let ZZ = prime_ring.base_ring().integer_ring();
    let ZZ_source = source_ring.base_ring().integer_ring();
    let ZZ_target = target_ring.base_ring().integer_ring();

    let p = prime_ring.base_ring().modulus();
    let (p_source, e) = is_prime_power(ZZ_source, source_ring.base_ring().modulus()).unwrap();
    let (p_target, r) = is_prime_power(ZZ_target, target_ring.base_ring().modulus()).unwrap();
    assert_el_eq!(ZZ, p, int_cast(p_source, ZZ, ZZ_source));
    assert_el_eq!(ZZ, p, int_cast(ZZ_target.clone_el(&p_target), ZZ, ZZ_target));

    assert!(r > e);
    assert!(target_ring.base_ring().is_one(target_ring.lc(f).unwrap()));
    assert!(source_ring.base_ring().is_one(source_ring.lc(factors.0).unwrap()));
    assert!(source_ring.base_ring().is_one(source_ring.lc(factors.1).unwrap()));
    

    let pe_to_p = ReductionMap::new(source_ring.base_ring(), prime_ring.base_ring()).unwrap();
    let pr_to_pe = ReductionMap::new(target_ring.base_ring(), source_ring.base_ring()).unwrap();
    let pr_to_p = ReductionMap::new(target_ring.base_ring(), prime_ring.base_ring()).unwrap();

    assert_el_eq!(source_ring, source_ring.mul_ref(factors.0, factors.1), &source_ring.from_terms(target_ring.terms(f).map(|(c, i)| (pr_to_pe.map_ref(c), i))));

    let (g, h) = factors;
    let reduced_g = prime_ring.from_terms(source_ring.terms(g).map(|(c, i)| (pe_to_p.map_ref(c), i)));
    let reduced_h = prime_ring.from_terms(source_ring.terms(h).map(|(c, i)| (pe_to_p.map_ref(c), i)));
    let (s, t, d) = prime_ring.extended_ideal_gen(&reduced_g, &reduced_h);
    assert!(prime_ring.is_unit(&d));
    
    let lifted_s = target_ring.from_terms(prime_ring.terms(&prime_ring.checked_div(&s, &d).unwrap()).map(|(c, i)| (pr_to_p.smallest_lift_ref(c), i)));
    let lifted_t = target_ring.from_terms(prime_ring.terms(&prime_ring.checked_div(&t, &d).unwrap()).map(|(c, i)| (pr_to_p.smallest_lift_ref(c), i)));

    let mut current_g = target_ring.from_terms(source_ring.terms(factors.0).map(|(c, i)| (pr_to_pe.smallest_lift_ref(c), i)));
    let mut current_h = target_ring.from_terms(source_ring.terms(factors.1).map(|(c, i)| (pr_to_pe.smallest_lift_ref(c), i)));
    
    for _ in e..r {
        let delta = target_ring.sub_ref_fst(f, target_ring.mul_ref(&current_g, &current_h));
        let mut delta_g = target_ring.mul_ref(&lifted_t, &delta);
        let mut delta_h = target_ring.mul_ref(&lifted_s, &delta);
        delta_g = target_ring.div_rem_monic(delta_g, &current_g).1;
        delta_h = target_ring.div_rem_monic(delta_h, &current_h).1;
        target_ring.add_assign(&mut current_g, delta_g);
        target_ring.add_assign(&mut current_h, delta_h);
        debug_assert!(target_ring.degree(&current_g).unwrap() == source_ring.degree(&g).unwrap());
        debug_assert!(target_ring.degree(&current_h).unwrap() == source_ring.degree(&h).unwrap());
    }
    assert_el_eq!(target_ring, f, target_ring.mul_ref(&current_g, &current_h));
    return (current_g, current_h);
}

///
/// Like [`hensel_lift()`] but for an arbitrary number of factors.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift_factorization<P, R, S>(target_ring: &P, source_ring: &R, prime_ring: &S, f: &El<P>, factors: &[El<R>]) -> Vec<El<P>>
    where P: PolyRingStore, P::Type: PolyRing,
        R: PolyRingStore, R::Type: PolyRing,
        S: PolyRingStore, S::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        <<R as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<R as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        <<S as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<S as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field
{
    if factors.len() == 1 {
        return vec![target_ring.clone_el(f)];
    }
    let (g, h) = (&factors[0], source_ring.prod(factors[1..].iter().map(|h| source_ring.clone_el(h))));
    let (g_lifted, h_lifted) = hensel_lift(target_ring, source_ring, prime_ring, &f, (g, &h));
    let mut result = hensel_lift_factorization(target_ring, source_ring, prime_ring, &h_lifted, &factors[1..]);
    result.insert(0, g_lifted);
    return result;
}

#[cfg(test)]
use crate::rings::zn::zn_static::{Zn, Fp};
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;

#[test]
fn test_lift_factorization() {
    let source_ring = Fp::<3>::RING;
    let target_ring = Zn::<27>::RING;

    let P = DensePolyRing::new(target_ring, "X");
    let R = DensePolyRing::new(source_ring, "X");

    // we lift the factorization `X^4 + 1 = (X^2 + 2 X + 2)(X^2 + X + 2)`
    let f = P.from_terms([(10, 0), (21, 1), (15, 2), (3, 3), (1, 4)].into_iter());
    let g = R.from_terms([(2, 0), (2, 1), (1, 2)].into_iter());
    let h = R.from_terms([(2, 0), (1, 1), (1, 2)].into_iter());

    let (lifted_g, lifted_h) = hensel_lift(&P, &R, &R, &f, (&g, &h));

    assert_el_eq!(P, f, P.mul_ref(&lifted_g, &lifted_h));
}

#[test]
fn test_lift_factorization_nonfield_base() {
    let prime_ring = Fp::<5>::RING;
    let source_ring = Zn::<25>::RING;
    let target_ring = Zn::<125>::RING;

    let P = DensePolyRing::new(target_ring, "X");
    let R = DensePolyRing::new(source_ring, "X");
    let S = DensePolyRing::new(prime_ring, "X");

    // we lift the factorization `X^5 + 6 X^4 + 8 X^3 + 12 X^2 + 3 X + 2 = (X^2 + X + 2)(X^3 + X + 1)`
    let f = P.from_terms([(2, 0), (3, 1), (12, 2), (8, 3), (6, 4), (1, 5)].into_iter());
    let g = R.from_terms([(2, 0), (1, 1), (1, 2)].into_iter());
    let h = R.from_terms([(1, 0), (1, 1), (5, 2), (1, 3)].into_iter());

    let (lifted_g, lifted_h) = hensel_lift(&P, &R, &S, &f, (&g, &h));

    assert_el_eq!(P, f, P.mul_ref(&lifted_g, &lifted_h));
}
