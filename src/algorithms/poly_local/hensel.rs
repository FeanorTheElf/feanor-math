use crate::computation::ComputationController;
use crate::pid::*;
use crate::ring::*;
use crate::divisibility::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::seq::*;
use super::{PolyGCDLocallyDomain, IntermediateReductionMap};

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h`
/// modulo `p^e`.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift<'ring, 'b, R, P1, P2, Controller>(
    reduction_map: &IntermediateReductionMap<'ring, 'b, R>, 
    target_poly_ring: P1, 
    base_poly_ring: P2, 
    f: &El<P1>, 
    factors: (&El<P2>, &El<P2>),
    controller: Controller
) -> (El<P1>, El<P1>)
    where R: ?Sized + PolyGCDLocallyDomain,
        P1: RingStore, P1::Type: PolyRing,
        <P1::Type as RingExtension>::BaseRing: RingStore<Type = R::LocalRingBase<'ring>>,
        P2: RingStore, P2::Type: PolyRing + PrincipalIdealRing,
        <P2::Type as RingExtension>::BaseRing: RingStore<Type = R::LocalFieldBase<'ring>>,
        Controller: ComputationController
{
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(f).unwrap()));
    assert!(base_poly_ring.base_ring().is_one(base_poly_ring.lc(factors.0).unwrap()));
    assert!(base_poly_ring.base_ring().is_one(base_poly_ring.lc(factors.1).unwrap()));
    assert!(target_poly_ring.base_ring().get_ring() == reduction_map.domain().get_ring());

    let prime_field = base_poly_ring.base_ring();
    let prime_ring = reduction_map.codomain();
    let prime_ring_iso = prime_field.can_iso(&prime_ring).unwrap();

    let (g, h) = factors;
    let (mut s, mut t, d) = base_poly_ring.extended_ideal_gen(g, h);
    assert!(base_poly_ring.degree(&d).unwrap() == 0);
    let d_inv = prime_field.invert(base_poly_ring.coefficient_at(&d, 0)).unwrap();
    base_poly_ring.inclusion().mul_assign_ref_map(&mut s, &d_inv);
    base_poly_ring.inclusion().mul_assign_map(&mut t, d_inv);

    let lift = |f| {
        target_poly_ring.from_terms(base_poly_ring.terms(f).map(|(c, i)| (reduction_map.parent_ring().get_ring().lift_partial(reduction_map.maximal_ideal(), (reduction_map.codomain(), reduction_map.to_e()), (reduction_map.domain(), reduction_map.from_e()), prime_ring_iso.map_ref(c)), i)))
    };

    let lifted_s = lift(&s);
    let lifted_t = lift(&t);
    let mut current_g = lift(g);
    let mut current_h = lift(h);

    let P = target_poly_ring;
    for _ in reduction_map.to_e()..reduction_map.from_e() {
        log_progress!(controller, ".");
        let delta = P.sub_ref_fst(f, P.mul_ref(&current_g, &current_h));
        let mut delta_g = P.mul_ref(&lifted_t, &delta);
        let mut delta_h = P.mul_ref(&lifted_s, &delta);
        delta_g = P.div_rem_monic(delta_g, &current_g).1;
        delta_h = P.div_rem_monic(delta_h, &current_h).1;
        P.add_assign(&mut current_g, delta_g);
        P.add_assign(&mut current_h, delta_h);
        debug_assert!(P.degree(&current_g).unwrap() == base_poly_ring.degree(&g).unwrap());
        debug_assert!(P.degree(&current_h).unwrap() == base_poly_ring.degree(&h).unwrap());
    }
    assert_el_eq!(P, f, P.mul_ref(&current_g, &current_h));
    return (current_g, current_h);
}

///
/// Like [`hensel_lift()`] but for an arbitrary number of factors.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift_factorization<'ring, 'b, R, P1, P2, V, Controller>(
    reduction_map: &IntermediateReductionMap<'ring, 'b, R>, 
    target_poly_ring: P1, 
    base_poly_ring: P2, 
    f: &El<P1>, 
    factors: V,
    controller: Controller
) -> Vec<El<P1>>
    where R: ?Sized + PolyGCDLocallyDomain,
        P1: RingStore + Copy, P1::Type: PolyRing,
        <P1::Type as RingExtension>::BaseRing: RingStore<Type = R::LocalRingBase<'ring>>,
        P2: RingStore + Copy, P2::Type: PolyRing + PrincipalIdealRing,
        <P2::Type as RingExtension>::BaseRing: RingStore<Type = R::LocalFieldBase<'ring>>,
        V: SelfSubvectorView<El<P2>>,
        Controller: ComputationController
{
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(f).unwrap()));
    assert!(factors.as_iter().all(|f| base_poly_ring.base_ring().is_one(base_poly_ring.lc(f).unwrap())));
    assert!(target_poly_ring.base_ring().get_ring() == reduction_map.domain().get_ring());

    if factors.len() == 1 {
        return vec![target_poly_ring.clone_el(f)];
    }
    let (g, h) = (factors.at(0), base_poly_ring.prod(factors.as_iter().skip(1).map(|h| base_poly_ring.clone_el(h))));
    let (g_lifted, h_lifted) = hensel_lift(reduction_map, target_poly_ring, base_poly_ring, &f, (g, &h), controller.clone());
    let mut result = hensel_lift_factorization(reduction_map, target_poly_ring, base_poly_ring, &h_lifted, factors.restrict(1..), controller);
    result.insert(0, g_lifted);
    return result;
}
