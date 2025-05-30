use crate::algorithms::int_factor::is_prime_power;
use crate::computation::ComputationController;
use crate::local::PrincipalLocalRing;
use crate::pid::*;
use crate::ring::*;
use crate::divisibility::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::rings::zn::FromModulusCreateableZnRing;
use crate::rings::zn::*;
use crate::seq::*;
use crate::reduce_lift::poly_factor_gcd::*;

use crate::computation::DontObserve;
use super::DensePolyRing;
use super::AsFieldBase;

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h`
/// modulo `p^e`.
/// 
/// This uses linear Hensel lifting, thus will be slower than [`hensel_lift_quadratic()`]
/// if `r >> e`.
/// 
#[cfg(test)]
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
fn hensel_lift_bezout_identity_quadratic<'ring, 'data, 'local, R, P1, P2, Controller>(
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
/// Computes a "Bezout identity" `sf + tg = 1` in the ring `Z/p^eZ` for a
/// prime `p` and an exponent `e`, for monic "coprime" polynomials `f, g`.
/// If `f` and `g` are not "coprime", `None` is returned.
/// 
/// Note that `(Z/p^eZ)[X]` is not a gcd domain, thus the notion of "Bezout identity"
/// and "coprime" must be taken with a grain of salt. What we mean is that the polynomials
/// `f mod p` and `g mod p` are coprime over `Fp`, in which case this function find
/// polynomial `s, t` over `Z/p^eZ` of degree `deg(s) < deg(g)` and `deg(t) < deg(f)`
/// such that `sf + tg = 1`.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::local::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::algorithms::poly_gcd::hensel::*;
/// # use feanor_math::assert_el_eq;
/// let ring = AsLocalPIR::from_zn(Zn::new(81)).unwrap();
/// let poly_ring = DensePolyRing::new(&ring, "X");
/// let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 1, X.pow_ref(2) + X + 2]);
/// let (s, t) = local_zn_ring_bezout_identity(&poly_ring, &f, &g).unwrap();
/// assert_el_eq!(&poly_ring, poly_ring.one(), poly_ring.add(poly_ring.mul_ref(&f, &s), poly_ring.mul_ref(&g, &t)));
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn local_zn_ring_bezout_identity<P>(poly_ring: P, f: &El<P>, g: &El<P>) -> Option<(El<P>, El<P>)>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing,
        AsFieldBase<RingValue<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>>: CanIsoFromTo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> + SelfIso
{
    if poly_ring.is_zero(f) {
        if poly_ring.is_one(g) {
            return Some((poly_ring.zero(), poly_ring.one()));
        } else {
            return None;
        }
    } else if poly_ring.is_zero(g) {
        if poly_ring.is_one(f) {
            return Some((poly_ring.one(), poly_ring.zero()));
        } else {
            return None;
        }
    }
    let Zpe = poly_ring.base_ring();
    let ZZ = Zpe.integer_ring();
    let (p, e) = is_prime_power(ZZ, Zpe.modulus()).unwrap();
    let wrapped_ring: IntegersWithLocalZnQuotient<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> = IntegersWithLocalZnQuotient::new(ZZ, p);
    let reduction_context = wrapped_ring.reduction_context(e, 1);

    let Zpe_to_Zp = reduction_context.intermediate_ring_to_field_reduction(0);
    let Fp = wrapped_ring.local_field_at(Zpe_to_Zp.ideal(), 0);
    let Zp = Zpe_to_Zp.codomain();
    let FpX = DensePolyRing::new(&Fp, "X");
    let Zpe_to_Fp = Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp);
    let ZpeX_to_FpX = FpX.lifted_hom(&poly_ring, &Zpe_to_Fp);

    let (mut s_base, mut t_base, d_base) = FpX.extended_ideal_gen(&ZpeX_to_FpX.map_ref(f), &ZpeX_to_FpX.map_ref(g));
    if FpX.degree(&d_base).unwrap() > 0 {
        return None;
    }
    let scale = Fp.invert(FpX.coefficient_at(&d_base, 0)).unwrap();
    FpX.inclusion().mul_assign_ref_map(&mut s_base, &scale);
    FpX.inclusion().mul_assign_ref_map(&mut t_base, &scale);
    let (s, t) = hensel_lift_bezout_identity_quadratic(&Zpe_to_Zp, &poly_ring, &FpX, f, g, (&s_base, &t_base), DontObserve);

    return Some((s, t));
}

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization of `f mod p^e` into monic and
/// pairwise coprime factors (with `e < r`), computes a monic lift of each factor, such that their
/// product is `f mod p^r`.
/// 
fn hensel_lift_factorization_internal<'ring, 'data, 'local, R, P1, P2, V, Controller>(
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
    if factors.len() == 1 {
        return vec![target_poly_ring.clone_el(f)];
    }
    let (g, h) = (factors.at(0), base_poly_ring.prod(factors.as_iter().skip(1).map(|h| base_poly_ring.clone_el(h))));
    let (g_lifted, h_lifted) = hensel_lift_quadratic(reduction_map, target_poly_ring, base_poly_ring, &f, (g, &h), controller.clone());
    let mut result = hensel_lift_factorization_internal(reduction_map, target_poly_ring, base_poly_ring, &h_lifted, factors.restrict(1..), controller);
    result.insert(0, g_lifted);
    return result;
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

    let result = controller.run_computation(format_args!("hensel_lift(deg={}, to={})", target_poly_ring.degree(f).unwrap(), reduction_map.from_e()), |controller| hensel_lift_factorization_internal(reduction_map, target_poly_ring, base_poly_ring, f, factors, controller));
    return result;
}

#[cfg(test)]
use super::BigIntRing;

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