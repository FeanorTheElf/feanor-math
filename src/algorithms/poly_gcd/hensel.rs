use crate::computation::ComputationController;
use crate::pid::*;
use crate::ring::*;
use crate::divisibility::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::seq::*;
use crate::algorithms::poly_gcd::local::*;

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h`
/// modulo `p^e`.
/// 
/// This uses linear Hensel lifting, thus will be slower than [`hensel_lift_quadratic()`]
/// if `r >> e`.
/// 
fn hensel_lift_linear<'ring, 'data, 'local, R, P1, P2, Controller>(
    reduction_map: &PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>, 
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

    let lift_to_target_poly_ring = |f| {
        target_poly_ring.from_terms(base_poly_ring.terms(f).map(|(c, i)| (
            reduction_map.parent_ring().get_ring().lift_partial(reduction_map.ideal(), (reduction_map.codomain(), reduction_map.to_e()), (reduction_map.domain(), reduction_map.from_e()), reduction_map.max_ideal_idx(), prime_ring_iso.map_ref(c)), 
            i
        )))
    };

    let lifted_s = lift_to_target_poly_ring(&s);
    let lifted_t = lift_to_target_poly_ring(&t);
    let mut current_g = lift_to_target_poly_ring(g);
    let mut current_h = lift_to_target_poly_ring(h);

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
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h` modulo `p^e`.
/// 
/// This uses quadratic Hensel lifting, thus will be faster than [`hensel_lift_linear()`]
/// if `r >> e`.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift_quadratic<'ring, 'data, 'local, R, P1, P2, Controller>(
    reduction_map: &PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>, 
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

    let lift_to_target_poly_ring = |f| {
        target_poly_ring.from_terms(base_poly_ring.terms(f).map(|(c, i)| (
            reduction_map.parent_ring().get_ring().lift_partial(reduction_map.ideal(), (reduction_map.codomain(), reduction_map.to_e()), (reduction_map.domain(), reduction_map.from_e()), reduction_map.max_ideal_idx(), prime_ring_iso.map_ref(c)), 
            i
        )))
    };

    let mut current_s = lift_to_target_poly_ring(&s);
    let mut current_t = lift_to_target_poly_ring(&t);
    let mut current_g = lift_to_target_poly_ring(g);
    let mut current_h = lift_to_target_poly_ring(h);
    // we have to lift the Bezout identity starting from `e = 1`, so for simplicity,
    // start lifting everything from `e = 1` on
    let mut current_e = 1;

    let P = target_poly_ring;
    while current_e < reduction_map.from_e() {
        log_progress!(controller, ".");

        // first, lift the polynomials
        // the formula is `g' = g - delta * t`, `h' = h - delta * s` where `delta = gh - f`
        let delta = P.sub_ref_fst(f, P.mul_ref(&current_g, &current_h));
        let mut delta_g = P.mul_ref(&current_t, &delta);
        let mut delta_h = P.mul_ref(&current_s, &delta);
        delta_g = P.div_rem_monic(delta_g, &current_g).1;
        delta_h = P.div_rem_monic(delta_h, &current_h).1;
        P.add_assign(&mut current_g, delta_g);
        P.add_assign(&mut current_h, delta_h);
        debug_assert!(P.degree(&current_g).unwrap() == base_poly_ring.degree(&g).unwrap());
        debug_assert!(P.degree(&current_h).unwrap() == base_poly_ring.degree(&h).unwrap());

        // now lift the bezout identity
        // the formula is `s' = s(2 - (sg + th))`, `t' = t(2 - (sg + th))`
        let bezout_value = P.add(P.mul_ref(&current_s, &current_g), P.mul_ref(&current_t, &current_h));
        P.mul_assign(&mut current_s, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
        P.mul_assign(&mut current_t, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
        current_s = P.div_rem_monic(current_s, &current_h).1;
        current_t = P.div_rem_monic(current_t, &current_g).1;

        current_e = 2 * current_e;
    }
    assert_el_eq!(P, f, P.mul_ref(&current_g, &current_h));
    return (current_g, current_h);
}

///
/// Given monic coprime polynomials `f, g` modulo `p^r` and a Bezout identity `sf + tg = 1 mod p^e`
/// for `e < r`, this computes a Bezout identity `s' f + t' g = 1` with `s', t'` polynomials modulo 
/// `p^r` that reduce to `s, t` modulo `p^e`.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift_bezout_identity_quadratic<'ring, 'data, 'local, R, P1, P2, Controller>(
    reduction_map: &PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>, 
    target_poly_ring: P1, 
    base_poly_ring: P2, 
    f: &El<P1>,
    g: &El<P1>, 
    (s, t): (&El<P2>, &El<P2>),
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
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(g).unwrap()));
    assert!(target_poly_ring.base_ring().get_ring() == reduction_map.domain().get_ring());

    let prime_field = base_poly_ring.base_ring();
    let prime_ring = reduction_map.codomain();
    let prime_ring_iso = prime_field.can_iso(&prime_ring).unwrap();

    let f_base = base_poly_ring.lifted_hom(&target_poly_ring, prime_ring_iso.inv().compose(reduction_map)).map_ref(&f);
    let g_base = base_poly_ring.lifted_hom(&target_poly_ring, prime_ring_iso.inv().compose(reduction_map)).map_ref(&g);
    assert!(base_poly_ring.is_one(&base_poly_ring.add(base_poly_ring.mul_ref(&f_base, s), base_poly_ring.mul_ref(&g_base, t))));

    let lift_to_target_poly_ring = |f| {
        target_poly_ring.from_terms(base_poly_ring.terms(f).map(|(c, i)| (
            reduction_map.parent_ring().get_ring().lift_partial(reduction_map.ideal(), (reduction_map.codomain(), reduction_map.to_e()), (reduction_map.domain(), reduction_map.from_e()), reduction_map.max_ideal_idx(), prime_ring_iso.map_ref(c)), 
            i
        )))
    };

    let mut current_s = lift_to_target_poly_ring(&s);
    let mut current_t = lift_to_target_poly_ring(&t);
    let mut current_e = 1;

    let P = target_poly_ring;
    while current_e < reduction_map.from_e() {
        log_progress!(controller, ".");

        // lift the bezout identity
        // the formula is `s' = s(2 - (sg + th))`, `t' = t(2 - (sg + th))`
        let bezout_value = P.add(P.mul_ref(&current_s, f), P.mul_ref(&current_t, g));
        P.println(&bezout_value);
        P.mul_assign(&mut current_s, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
        P.mul_assign(&mut current_t, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
        current_s = P.div_rem_monic(current_s, g).1;
        current_t = P.div_rem_monic(current_t, f).1;

        current_e = 2 * current_e;
    }
    debug_assert!(P.is_one(&P.add(P.mul_ref(f, &current_s), P.mul_ref(g, &current_t))));
    return (current_s, current_t);
}

///
/// Like [`hensel_lift()`] but for an arbitrary number of factors.
/// 
#[stability::unstable(feature = "enable")]
pub fn hensel_lift_factorization<'ring, 'data, 'local, R, P1, P2, V, Controller>(
    reduction_map: &PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>, 
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
    let (g_lifted, h_lifted) = hensel_lift_quadratic(reduction_map, target_poly_ring, base_poly_ring, &f, (g, &h), controller.clone());
    let mut result = hensel_lift_factorization(reduction_map, target_poly_ring, base_poly_ring, &h_lifted, factors.restrict(1..), controller);
    result.insert(0, g_lifted);
    return result;
}

#[cfg(test)]
use crate::computation::DontObserve;
#[cfg(test)]
use super::BigIntRing;
#[cfg(test)]
use super::DensePolyRing;

#[test]
fn test_hensel_lift() {
    let ZZ = BigIntRing::RING;
    let prime = 5;
    let Zp = ZZ.get_ring().local_ring_at(&prime, 1, 0);
    let Fp = ZZ.get_ring().local_field_at(&prime, 0);
    let Zpe = ZZ.get_ring().local_ring_at(&prime, 6, 0);
    let Zpe_to_Zp = PolyGCDLocallyIntermediateReductionMap::new(ZZ.get_ring(), &prime, &Zpe, 6, &Zp, 1, 0);
    let ZpeX = DensePolyRing::new(&Zpe, "X");
    let FpX = DensePolyRing::new(&Fp, "X");
    let ZpeX_to_ZpX = FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp));

    let [f, g] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 3, X + 1]);
    let h = ZpeX.mul_ref(&f, &g);
    let (actual_f, actual_g) = hensel_lift_linear(&Zpe_to_Zp, &ZpeX, &FpX, &h, (&ZpeX_to_ZpX.map_ref(&f), &ZpeX_to_ZpX.map_ref(&g)), DontObserve);
    assert_el_eq!(&ZpeX, &f, &actual_f);
    assert_el_eq!(&ZpeX, &g, &actual_g);
    let (actual_f, actual_g) = hensel_lift_quadratic(&Zpe_to_Zp, &ZpeX, &FpX, &h, (&ZpeX_to_ZpX.map_ref(&f), &ZpeX_to_ZpX.map_ref(&g)), DontObserve);
    assert_el_eq!(&ZpeX, &f, &actual_f);
    assert_el_eq!(&ZpeX, &g, &actual_g);

    let [f, g] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 25 * X + 3 + 625, X + 1 + 125]);
    let h = ZpeX.mul_ref(&f, &g);
    let (actual_f, actual_g) = hensel_lift_linear(&Zpe_to_Zp, &ZpeX, &FpX, &h, (&ZpeX_to_ZpX.map_ref(&f), &ZpeX_to_ZpX.map_ref(&g)), DontObserve);
    assert_el_eq!(&ZpeX, &f, &actual_f);
    assert_el_eq!(&ZpeX, &g, &actual_g);
    let (actual_f, actual_g) = hensel_lift_quadratic(&Zpe_to_Zp, &ZpeX, &FpX, &h, (&ZpeX_to_ZpX.map_ref(&f), &ZpeX_to_ZpX.map_ref(&g)), DontObserve);
    assert_el_eq!(&ZpeX, &f, &actual_f);
    assert_el_eq!(&ZpeX, &g, &actual_g);
}

#[test]
fn test_hensel_lift_bezout_identity() {
    let ZZ = BigIntRing::RING;
    let prime = 5;
    let Zp = ZZ.get_ring().local_ring_at(&prime, 1, 0);
    let Fp = ZZ.get_ring().local_field_at(&prime, 0);
    let Zpe = ZZ.get_ring().local_ring_at(&prime, 6, 0);
    let Zpe_to_Zp = PolyGCDLocallyIntermediateReductionMap::new(ZZ.get_ring(), &prime, &Zpe, 6, &Zp, 1, 0);
    let ZpeX = DensePolyRing::new(&Zpe, "X");
    let FpX = DensePolyRing::new(&Fp, "X");

    let [f, g] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 6 * X + 2, X.pow_ref(2) + 11]);
    let [s_base, t_base] = FpX.with_wrapped_indeterminate(|X| [3 * X + 3, 2 * X]);
    let (s, t) = hensel_lift_bezout_identity_quadratic(&Zpe_to_Zp, &ZpeX, &FpX, &f, &g, (&s_base, &t_base), DontObserve);
    assert_eq!(1, ZpeX.degree(&s).unwrap());
    assert_eq!(1, ZpeX.degree(&t).unwrap());
    assert_el_eq!(&ZpeX, ZpeX.one(), ZpeX.add(ZpeX.mul_ref(&f, &s), ZpeX.mul_ref(&g, &t)));
    assert_el_eq!(&FpX, &s_base, FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp)).map_ref(&s));
    assert_el_eq!(&FpX, &t_base, FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp)).map_ref(&t));
}