use std::marker::PhantomData;

use tracing::instrument;

use crate::algorithms::int_factor::is_prime_power;
use crate::divisibility::*;
use crate::homomorphism::*;
use crate::pid::*;
use crate::reduce_lift::lift_poly_factors::*;
use crate::ring::*;
use crate::rings::poly::dense_poly::*;
use crate::rings::poly::*;
use crate::rings::zn::{FromModulusCreateableZnRing, *};
use crate::seq::*;

#[derive(Clone)]
pub struct HenselLift<P_current>
where
    P_current: RingStore,
    P_current::Ring: PolyRing,
{
    current_poly_ring: P_current,
    current_e: usize,
    /// The list of factors to be lifted, except for the last factor, thus of length n - 1
    current_factors: Vec<HenselLiftableBarrettReducer<P_current::Ring>>,
    /// the i-th element contains the product of factors [(i + 1)..]; of length n - 1, thus we
    /// always lift current_factors[i] and current_partial_prods[i] with each other
    current_partial_prods: Vec<HenselLiftableBarrettReducer<P_current::Ring>>,
    current_factor_bezout: Vec<El<P_current>>,
    current_partial_prods_bezout: Vec<El<P_current>>,
}

impl<P_current> HenselLift<P_current>
where
    P_current: RingStore,
    P_current::Ring: PolyRing + PrincipalIdealRing,
{
    #[instrument(skip_all, level = "trace")]
    pub fn new(poly_ring: P_current, mut factors: Vec<El<P_current>>) -> Self {
        assert!(factors.len() >= 1);
        for f in &factors {
            assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
        }
        let n = factors.len();
        let last = factors.pop().unwrap();
        let partial_prods = {
            let mut result = factors
                .iter()
                .rev()
                .scan(last, |current, next| {
                    let result = current.clone();
                    poly_ring.mul_assign_ref(current, next);
                    return Some(result);
                })
                .collect::<Vec<_>>();
            result.reverse();
            result
        };
        debug_assert_eq!(n - 1, partial_prods.len());
        debug_assert_eq!(n - 1, factors.len());

        let mut factor_bezout = Vec::with_capacity(n - 1);
        let mut partial_prods_bezout = Vec::with_capacity(n - 1);
        let mut factor_reducers = Vec::with_capacity(n - 1);
        let mut partial_prod_reducers = Vec::with_capacity(n - 1);
        for (g, h) in factors.into_iter().zip(partial_prods.into_iter()) {
            let (s, t, d) = poly_ring.extended_ideal_gen(&g, &h);
            if let Some(d_inv) = poly_ring.invert(&d) {
                factor_bezout.push(poly_ring.mul_ref_snd(s, &d_inv));
                partial_prods_bezout.push(poly_ring.mul_ref_snd(t, &d_inv));
                let deg_g = poly_ring.degree(&g).unwrap();
                let deg_h = poly_ring.degree(&h).unwrap();
                factor_reducers.push(HenselLiftableBarrettReducer::new(&poly_ring, g, deg_h, 1));
                partial_prod_reducers.push(HenselLiftableBarrettReducer::new(&poly_ring, h, deg_g, 1));
            } else {
                panic!("given polynomials are not pairwise coprime")
            }
        }
        return Self {
            current_e: 1,
            current_factor_bezout: factor_bezout,
            current_partial_prods_bezout: partial_prods_bezout,
            current_factors: factor_reducers,
            current_partial_prods: partial_prod_reducers,
            current_poly_ring: poly_ring,
        };
    }
}

impl<P_current> HenselLift<P_current>
where
    P_current: RingStore,
    P_current::Ring: PolyRing,
{
    ///
    /// Lifts the current factorization to a factorization of `target` in `new_ring`.
    /// 
    /// This function requires that `target` is congruent to the product of the current
    /// factors modulo `m^prev_e`, and unfortunately cannot check this with the current information.
    /// 
    #[instrument(skip_all, level = "trace")]
    pub fn lift_to<P_new, L>(
        self,
        new_e: usize,
        new_ring: P_new,
        target: &El<P_new>,
        mut lift: L,
    ) -> HenselLift<P_new>
    where
        P_new: RingStore,
        P_new::Ring: PolyRing,
        L: FnMut(&El<BaseRingStore<P_current>>) -> El<BaseRingStore<P_new>>,
    {
        let P = new_ring;
        let n = self.current_factors.len() + 1;
        
        let mut factor_bezout: Vec<El<P_new>> = Vec::with_capacity(n - 1);
        let mut partial_prods_bezout: Vec<El<P_new>> = Vec::with_capacity(n - 1);
        let mut factor_reducers: Vec<HenselLiftableBarrettReducer<P_new::Ring>> = Vec::with_capacity(n - 1);
        let mut partial_prod_reducers: Vec<HenselLiftableBarrettReducer<P_new::Ring>> = Vec::with_capacity(n - 1);

        for (i, ((g, h), (s, t))) in self
            .current_factors
            .into_iter()
            .zip(self.current_partial_prods.into_iter())
            .zip(
                self.current_factor_bezout
                    .into_iter()
                    .zip(self.current_partial_prods_bezout.into_iter()),
            ).enumerate()
        {
            let f = if i == 0 { target } else { &partial_prod_reducers.last().unwrap().poly };
            let deg_g = self.current_poly_ring.degree(&g.poly).unwrap();
            let deg_h = self.current_poly_ring.degree(&h.poly).unwrap();
            let deg_delta_bound = deg_g + deg_h;
            let mut g = g.change_ring(&P, |poly| P.from_terms(self.current_poly_ring.terms(&poly).map(|(c, i)| (lift(c), i))));
            let mut h = h.change_ring(&P, |poly| P.from_terms(self.current_poly_ring.terms(&poly).map(|(c, i)| (lift(c), i))));
            let mut s = P.from_terms(self.current_poly_ring.terms(&s).map(|(c, i)| (lift(c), i)));
            let mut t = P.from_terms(self.current_poly_ring.terms(&t).map(|(c, i)| (lift(c), i)));
            let mut current_e = self.current_e;
            while current_e < new_e {
                // the formula is `g' = g - delta * t`, `h' = h - delta * s` where `delta = gh - f`
                let delta = P.sub_ref_fst(f, P.mul_ref(&g.poly, &h.poly));
                debug_assert!(P.degree(&delta).is_none() || P.degree(&delta).unwrap() < deg_delta_bound);
                let mut delta_g = P.mul_ref(&t, &delta);
                let mut delta_h = P.mul_ref(&s, &delta);
                delta_g = g.div_rem_poly(&P, delta_g).1;
                delta_h = h.div_rem_poly(&P, delta_h).1;
                g.lift(&P, delta_g, 2 * current_e);
                h.lift(&P, delta_h, 2 * current_e);
                debug_assert!(P.degree(&g.poly).unwrap() == deg_g);
                debug_assert!(P.degree(&h.poly).unwrap() == deg_h);

                // now lift the bezout identity
                // the formula is `s' = s(2 - (sg + th))`, `t' = t(2 - (sg + th))`
                let bezout_value = P.add(P.mul_ref(&s, &g.poly), P.mul_ref(&t, &h.poly));
                debug_assert!(P.degree(&bezout_value).is_none() || P.degree(&bezout_value).unwrap() < deg_delta_bound);
                P.mul_assign(&mut s, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
                P.mul_assign(&mut t, P.sub_ref_snd(P.int_hom().map(2), &bezout_value));
                assert!(P.degree(&s).is_none() || P.degree(&s).unwrap() < deg_h + deg_delta_bound);
                assert!(P.degree(&t).is_none() || P.degree(&t).unwrap() < deg_g + deg_delta_bound);
                s = h.div_rem_poly(&P, s).1;
                t = g.div_rem_poly(&P, t).1;
                debug_assert!(P.degree(&s).is_none() || P.degree(&s).unwrap() < deg_h);
                debug_assert!(P.degree(&t).is_none() || P.degree(&t).unwrap() < deg_g);

                current_e = 2 * current_e;
            }
            g.e = new_e;
            h.e = new_e;
            factor_reducers.push(g);
            partial_prod_reducers.push(h);
            factor_bezout.push(s);
            partial_prods_bezout.push(t);
        }

        return HenselLift {
            current_e: new_e,
            current_factor_bezout: factor_bezout,
            current_factors: factor_reducers,
            current_partial_prods: partial_prod_reducers,
            current_partial_prods_bezout: partial_prods_bezout,
            current_poly_ring: P,
        };
    }

    pub fn poly_ring(&self) -> &P_current {
        &self.current_poly_ring
    }

    pub fn factorization<'a>(&'a self) -> impl Iterator<Item = &'a El<P_current>> {
        self.current_factors.iter().chain([self.current_partial_prods.last().unwrap()]).map(|r| &r.poly)
    }

    pub fn current_e(&self) -> usize {
        self.current_e
    }
}

struct HenselLiftableBarrettReducer<P: ?Sized + PolyRing> {
    ring: PhantomData<Box<P>>,
    n: usize,
    neg_Xn_div_poly: P::Element,
    poly: P::Element,
    poly_deg: usize,
    e: usize,
}

impl<P: ?Sized + PolyRing> HenselLiftableBarrettReducer<P> {
    #[instrument(skip_all, level = "trace")]
    fn div_rem_poly<S>(&self, poly_ring: S, poly: El<S>) -> (El<S>, El<S>)
    where
        S: RingStore<Ring = P>,
    {
        assert!(poly_ring.degree(&poly).unwrap_or(0) <= self.n);
        let scaled_quotient = poly_ring.mul_ref(&poly, &self.neg_Xn_div_poly);
        let quotient = poly_ring.from_terms(
            poly_ring
                .terms(&scaled_quotient)
                .filter(|(_, i)| *i >= self.n)
                .map(|(c, i)| (c.clone(), i - self.n)),
        );
        let remainder = poly_ring.add(poly, poly_ring.mul_ref(&quotient, &self.poly));
        let truncated_remainder = poly_ring.from_terms(
            poly_ring
                .terms(&remainder)
                .filter(|(_, i)| *i < self.poly_deg)
                .map(|(c, i)| (c.clone(), i)),
        );
        return (poly_ring.negate(quotient), truncated_remainder);
    }

    #[instrument(skip_all, level = "trace")]
    fn new<S>(poly_ring: S, poly: El<S>, other_d: usize, start_e: usize) -> Self
    where
        S: RingStore<Ring = P> + Copy,
    {
        let poly_deg = poly_ring.degree(&poly).unwrap();
        let n = poly_deg + other_d;
        assert!(poly_ring.base_ring().is_one(poly_ring.lc(&poly).unwrap()));
        let neg_Xn_div_poly = poly_ring
            .div_rem_monic(poly_ring.from_terms([(poly_ring.base_ring().neg_one(), n)]), &poly)
            .0;
        return Self {
            ring: PhantomData,
            n,
            e: start_e,
            poly,
            poly_deg,
            neg_Xn_div_poly,
        };
    }

    fn change_ring<S, F>(self, _new_poly_ring: S, mut lift: F) -> HenselLiftableBarrettReducer<S::Ring>
    where
        S: RingStore + Copy,
        S::Ring: PolyRing,
        F: FnMut(P::Element) -> El<S>,
    {
        HenselLiftableBarrettReducer {
            ring: PhantomData,
            n: self.n,
            neg_Xn_div_poly: lift(self.neg_Xn_div_poly),
            poly: lift(self.poly),
            poly_deg: self.poly_deg,
            e: self.e,
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn lift<S>(&mut self, poly_ring: S, delta_poly: El<S>, new_e: usize)
    where
        S: RingStore<Ring = P> + Copy,
    {
        assert!(new_e <= 2 * self.e);
        self.e = new_e;
        let new_f = poly_ring.add_ref(&self.poly, &delta_poly);
        let delta_quo = self
            .div_rem_poly(
                poly_ring,
                poly_ring.add(
                    poly_ring.from_terms([(poly_ring.base_ring().one(), self.n)]),
                    poly_ring.mul_ref(&self.neg_Xn_div_poly, &new_f),
                ),
            )
            .0;
        self.poly = new_f;
        poly_ring.sub_assign(&mut self.neg_Xn_div_poly, delta_quo);
    }
}

impl<P: ?Sized + PolyRing> Clone for HenselLiftableBarrettReducer<P> {

    fn clone(&self) -> Self {
        Self {
            e: self.e,
            n: self.n,
            neg_Xn_div_poly: self.neg_Xn_div_poly.clone(),
            poly: self.poly.clone(),
            poly_deg: self.poly_deg,
            ring: self.ring
        }
    }
}

/// Given monic coprime polynomials `f, g` modulo `p^r` and a Bezout identity `sf + tg = 1 mod p^e`
/// for `e < r`, this computes a Bezout identity `s' f + t' g = 1` with `s', t'` polynomials modulo
/// `p^r` that reduce to `s, t` modulo `p^e`.
#[instrument(skip_all, level = "trace")]
fn hensel_lift_bezout_identity_quadratic<'ring, 'data, 'local, R, P1, P2>(
    reduction_map: &PolyLiftFactorsDomainIntermediateReductionMap<'ring, 'data, 'local, R>,
    target_poly_ring: P1,
    base_poly_ring: P2,
    f: &El<P1>,
    g: &El<P1>,
    (s, t): (&El<P2>, &El<P2>),
) -> (El<P1>, El<P1>)
where
    R: ?Sized + PolyLiftFactorsDomain,
    P1: RingStore,
    P1::Ring: PolyRing,
    BaseRingStore<P1>: RingStore<Ring = R::LocalRingBase<'ring>>,
    P2: RingStore,
    P2::Ring: PolyRing + PrincipalIdealRing,
    BaseRingStore<P2>: RingStore<Ring = R::LocalFieldBase<'ring>>,
{
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(f).unwrap()));
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(g).unwrap()));
    assert!(target_poly_ring.base_ring().get_ring() == reduction_map.domain().get_ring());

    let prime_field = base_poly_ring.base_ring();
    let prime_ring = reduction_map.codomain();
    let prime_ring_iso = PolyLiftFactorsDomainBaseRingToFieldIso::new(
        reduction_map.parent_ring().into(),
        reduction_map.ideal(),
        prime_ring.get_ring(),
        prime_field.get_ring(),
        reduction_map.max_ideal_idx(),
    );
    let poly_hom = base_poly_ring.lifted_hom(&target_poly_ring, (&prime_ring_iso).compose(&reduction_map));
    assert_el_eq!(
        base_poly_ring,
        base_poly_ring.one(),
        base_poly_ring.add(poly_hom.mul_ref_map(s, f), poly_hom.mul_ref_map(t, g))
    );

    let f_base = base_poly_ring
        .lifted_hom(&target_poly_ring, (&prime_ring_iso).compose(reduction_map))
        .map_ref(&f);
    let g_base = base_poly_ring
        .lifted_hom(&target_poly_ring, (&prime_ring_iso).compose(reduction_map))
        .map_ref(&g);
    assert!(
        base_poly_ring
            .is_one(&base_poly_ring.add(base_poly_ring.mul_ref(&f_base, s), base_poly_ring.mul_ref(&g_base, t)))
    );

    let lift_to_target_poly_ring = |f| {
        target_poly_ring.from_terms(base_poly_ring.terms(f).map(|(c, i)| {
            (
                reduction_map.parent_ring().get_ring().lift_partial(
                    reduction_map.ideal(),
                    (reduction_map.codomain().get_ring(), reduction_map.to_e()),
                    (reduction_map.domain().get_ring(), reduction_map.from_e()),
                    reduction_map.max_ideal_idx(),
                    prime_ring_iso.inv().map_ref(c),
                ),
                i,
            )
        }))
    };

    let mut current_s = lift_to_target_poly_ring(&s);
    let mut current_t = lift_to_target_poly_ring(&t);
    let mut current_e = 1;

    let P = target_poly_ring;
    while current_e < reduction_map.from_e() {
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
/// # use feanor_math::rings::zn::zn_64b::*;
/// # use feanor_math::rings::local::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::algorithms::poly_gcd::hensel::*;
/// # use feanor_math::assert_el_eq;
/// let ring = AsLocalPIR::from_zn(Zn64B::new(81)).unwrap();
/// let poly_ring = DensePolyRing::new(&ring, "X");
/// let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 1, X.pow_ref(2) + X + 2]);
/// let (s, t) = local_zn_ring_bezout_identity(&poly_ring, &f, &g).unwrap();
/// assert_el_eq!(
///     &poly_ring,
///     poly_ring.one(),
///     poly_ring.add(poly_ring.mul_ref(&f, &s), poly_ring.mul_ref(&g, &t))
/// );
/// ```
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn local_zn_ring_bezout_identity<P>(poly_ring: P, f: &El<P>, g: &El<P>) -> Option<(El<P>, El<P>)>
where
    P: RingStore,
    P::Ring: PolyRing,
    <BaseRingStore<P> as RingStore>::Ring: SelfIso + ZnRing + FromModulusCreateableZnRing + Clone,
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
    let wrapped_ring: IntegersWithZnQuotient<<BaseRingStore<P> as RingStore>::Ring> =
        IntegersWithZnQuotient::new(ZZ, p);
    let reduction_context = wrapped_ring.reduction_context(e);

    let Zpe_to_Zp = reduction_context.intermediate_ring_to_field_reduction(0);
    let Fp = wrapped_ring.quotient_field_at(Zpe_to_Zp.ideal(), 0);
    let FpX = DensePolyRing::new(&Fp, "X");
    let Zpe_to_Fp = reduction_context.base_ring_to_field_iso(0).compose(&Zpe_to_Zp);
    let ZpeX_to_FpX = FpX.lifted_hom(&poly_ring, &Zpe_to_Fp);

    let (mut s_base, mut t_base, d_base) = FpX.extended_ideal_gen(&ZpeX_to_FpX.map_ref(f), &ZpeX_to_FpX.map_ref(g));
    if FpX.degree(&d_base).unwrap() > 0 {
        return None;
    }
    let scale = Fp.invert(FpX.coefficient_at(&d_base, 0)).unwrap();
    FpX.inclusion().mul_assign_ref_map(&mut s_base, &scale);
    FpX.inclusion().mul_assign_ref_map(&mut t_base, &scale);
    let (s, t) = hensel_lift_bezout_identity_quadratic(&Zpe_to_Zp, &poly_ring, &FpX, f, g, (&s_base, &t_base));

    return Some((s, t));
}

/// Given the factorization of `f` into pairwise coprime factors modulo a maximal
/// ideal `m` of the ring `R` (given by `reduction_map`), lifts each factor to `R/m^e`
/// (with `e` given implicitly by `reduction_map`) so that their product is `f`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn hensel_lift_factorization<'ring, 'data, 'local, R, P1, P2>(
    reduction_map: &PolyLiftFactorsDomainIntermediateReductionMap<'ring, 'data, 'local, R>,
    target_poly_ring: P1,
    base_poly_ring: P2,
    f: &El<P1>,
    factors: &[El<P2>],
) -> Vec<El<P1>>
where
    R: ?Sized + PolyLiftFactorsDomain,
    P1: RingStore + Copy,
    P1::Ring: PolyRing,
    BaseRingStore<P1>: RingStore<Ring = R::LocalRingBase<'ring>>,
    P2: RingStore + Copy,
    P2::Ring: PolyRing + PrincipalIdealRing,
    BaseRingStore<P2>: RingStore<Ring = R::LocalFieldBase<'ring>>,
{
    assert!(target_poly_ring.base_ring().get_ring() == reduction_map.domain().get_ring());
    assert!(target_poly_ring.base_ring().is_one(target_poly_ring.lc(f).unwrap()));
    assert!(
        factors
            .as_iter()
            .all(|f| base_poly_ring.base_ring().is_one(base_poly_ring.lc(f).unwrap()))
    );
    let prime_field = base_poly_ring.base_ring();
    let prime_ring = reduction_map.codomain();
    let prime_ring_iso = PolyLiftFactorsDomainBaseRingToFieldIso::new(
        reduction_map.parent_ring().into(),
        reduction_map.ideal(),
        prime_ring.get_ring(),
        prime_field.get_ring(),
        reduction_map.max_ideal_idx(),
    );
    assert_el_eq!(&base_poly_ring, base_poly_ring.lifted_hom(target_poly_ring, (&prime_ring_iso).compose(&reduction_map)).map_ref(f), base_poly_ring.prod(factors.iter().cloned()));

    let lifter = HenselLift::new(base_poly_ring, factors.to_owned());
    let lifted = lifter.lift_to(reduction_map.from_e(), target_poly_ring, f, |x| reduction_map.parent_ring().get_ring().lift_partial(
        reduction_map.ideal(),
        (reduction_map.codomain().get_ring(), 1),
        (reduction_map.domain().get_ring(), reduction_map.from_e()),
        reduction_map.max_ideal_idx(),
        prime_ring_iso.inv().map_ref(x)
    ));
    return lifted.factorization().cloned().collect();
}

#[cfg(test)]
use crate::integer::*;
#[cfg(test)]
use crate::rings::zn::zn_64b::Zn64B;

#[test]
fn test_hensel_lift() {
    feanor_tracing::DelayedLogger::init_test();

    let FpX = DensePolyRing::new(Zn64B::new(5).as_field().unwrap(), "X");
    let ZpeX = DensePolyRing::new(Zn64B::new(5 * 5 * 5 * 5 * 5 * 5), "X");
    let [g, h] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 3, X + 1]);
    let target = ZpeX.mul_ref(&g, &h);
    let mod_p = ZnReductionMap::new(ZpeX.base_ring(), FpX.base_ring()).unwrap();
    let poly_mod_p = FpX.lifted_hom(&ZpeX, &mod_p);
    let lifter = HenselLift::new(&FpX, vec![poly_mod_p.map_ref(&g), poly_mod_p.map_ref(&h)]);
    {
        let lifted = lifter.clone().lift_to(6, &ZpeX, &target, |x| mod_p.any_preimage(*x));
        let [actual_g, actual_h] = lifted.factorization().collect::<Vec<_>>().try_into().unwrap();
        assert_el_eq!(&ZpeX, &g, &actual_g);
        assert_el_eq!(&ZpeX, &h, &actual_h);
    }
    {
        let ZpfX = DensePolyRing::new(Zn64B::new(5 * 5 * 5), "X");
        let Zpf_to_Zp = ZnReductionMap::new(ZpfX.base_ring(), FpX.base_ring()).unwrap();
        let Zpe_to_Zpf = ZnReductionMap::new(ZpeX.base_ring(), ZpfX.base_ring()).unwrap();
        let ZpeX_to_ZpfX = ZpfX.lifted_hom(&ZpeX, &Zpe_to_Zpf);
        let lifted = lifter
            .lift_to(3, &ZpfX, &ZpeX_to_ZpfX.map_ref(&target), |x| Zpf_to_Zp.any_preimage(*x))
            .lift_to(6, &ZpeX, &target, |x| Zpe_to_Zpf.any_preimage(*x));
        let [actual_g, actual_h] = lifted.factorization().collect::<Vec<_>>().try_into().unwrap();
        assert_el_eq!(&ZpeX, &g, &actual_g);
        assert_el_eq!(&ZpeX, &h, &actual_h);
    }

    let [g, h] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 25 * X + 3 + 625, X + 1 + 125]);
    let target = ZpeX.mul_ref(&g, &h);
    let lifter = HenselLift::new(&FpX, vec![poly_mod_p.map_ref(&g), poly_mod_p.map_ref(&h)]);
    {
        let lifted = lifter.clone().lift_to(6, &ZpeX, &target, |x| mod_p.any_preimage(*x));
        let [actual_g, actual_h] = lifted.factorization().collect::<Vec<_>>().try_into().unwrap();
        assert_el_eq!(&ZpeX, &g, &actual_g);
        assert_el_eq!(&ZpeX, &h, &actual_h);
    }
}

#[test]
fn test_hensel_lift_bezout_identity() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = BigIntRing::RING;
    let prime = 5;
    let Zp = ZZ.get_ring().quotient_ring_at(&prime, 1, 0);
    let Fp = ZZ.get_ring().quotient_field_at(&prime, 0);
    let Zpe = ZZ.get_ring().quotient_ring_at(&prime, 6, 0);
    let Zpe_to_Zp = PolyLiftFactorsDomainIntermediateReductionMap::new(ZZ.get_ring(), &prime, &Zpe, 6, &Zp, 1, 0);
    let ZpeX = DensePolyRing::new(&Zpe, "X");
    let FpX = DensePolyRing::new(&Fp, "X");

    let [f, g] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 6 * X + 2, X.pow_ref(2) + 11]);
    let [s_base, t_base] = FpX.with_wrapped_indeterminate(|X| [3 * X + 3, 2 * X]);
    let (s, t) = hensel_lift_bezout_identity_quadratic(&Zpe_to_Zp, &ZpeX, &FpX, &f, &g, (&s_base, &t_base));
    assert_eq!(1, ZpeX.degree(&s).unwrap());
    assert_eq!(1, ZpeX.degree(&t).unwrap());
    assert_el_eq!(&ZpeX, ZpeX.one(), ZpeX.add(ZpeX.mul_ref(&f, &s), ZpeX.mul_ref(&g, &t)));
    assert_el_eq!(
        &FpX,
        &s_base,
        FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp))
            .map_ref(&s)
    );
    assert_el_eq!(
        &FpX,
        &t_base,
        FpX.lifted_hom(&ZpeX, Fp.can_hom(&Zp).unwrap().compose(&Zpe_to_Zp))
            .map_ref(&t)
    );
}
