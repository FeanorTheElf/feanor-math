use std::alloc::Global;
use std::marker::PhantomData;
use std::sync::Arc;

use tracing::instrument;

use crate::algorithms::convolution::ntt::NTTConvolution;
use crate::algorithms::convolution::{DynConvolution, TypeErasedConvolution};
use crate::algorithms::cyclotomic::get_prim_root_of_unity_pow2;
use crate::homomorphism::*;
use crate::prelude::*;
use crate::ring_impls::poly::dense_poly::{DensePolyRing, DensePolyRingBase};
use crate::ring_impls::poly::*;
use crate::ring_impls::zn::zn_64b::Zn64B;
use crate::ring_impls::zn::zn_big::ZnGB;
use crate::ring_impls::zn::*;

#[derive(Debug)]
pub struct FactorsNotCoprimeError;

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
    single_factor: Option<El<P_current>>,
    current_factor_bezout: Vec<El<P_current>>,
    current_partial_prods_bezout: Vec<El<P_current>>,
}

impl<P_current> HenselLift<P_current>
where
    P_current: RingStore,
    P_current::Ring: PolyRing + PrincipalIdealRing,
{
    #[instrument(skip_all, level = "trace")]
    pub fn new(poly_ring: P_current, mut factors: Vec<El<P_current>>) -> Result<Self, FactorsNotCoprimeError> {
        assert!(factors.len() >= 1);
        for f in &factors {
            assert!(poly_ring.base_ring().is_one(poly_ring.lc(f).unwrap()));
        }
        if factors.len() == 1 {
            return Ok(Self {
                current_e: 1,
                current_factor_bezout: Vec::new(),
                current_partial_prods_bezout: Vec::new(),
                current_factors: Vec::new(),
                current_partial_prods: Vec::new(),
                single_factor: Some(factors.into_iter().next().unwrap()),
                current_poly_ring: poly_ring,
            });
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
                let deg_delta_bound = deg_g + deg_h;
                factor_reducers.push(HenselLiftableBarrettReducer::new(&poly_ring, g, deg_delta_bound, 1));
                partial_prod_reducers.push(HenselLiftableBarrettReducer::new(&poly_ring, h, deg_delta_bound, 1));
            } else {
                return Err(FactorsNotCoprimeError);
            }
        }
        return Ok(Self {
            current_e: 1,
            current_factor_bezout: factor_bezout,
            current_partial_prods_bezout: partial_prods_bezout,
            current_factors: factor_reducers,
            current_partial_prods: partial_prod_reducers,
            single_factor: None,
            current_poly_ring: poly_ring,
        });
    }
}

impl<P_current> HenselLift<P_current>
where
    P_current: RingStore,
    P_current::Ring: PolyRing,
{
    pub fn change_ring<P_new, H>(self, new_ring: P_new, hom: H) -> HenselLift<P_new>
    where
        P_new: RingStore,
        P_new::Ring: PolyRing,
        H: Homomorphism<BaseRingBase<P_current>, BaseRingBase<P_new>>,
    {
        let poly_hom = new_ring.lifted_hom(&self.current_poly_ring, &hom);
        HenselLift {
            current_e: self.current_e,
            current_factor_bezout: self
                .current_factor_bezout
                .into_iter()
                .map(|f| poly_hom.map(f))
                .collect(),
            current_factors: self
                .current_factors
                .into_iter()
                .map(|f| HenselLiftableBarrettReducer {
                    ring: PhantomData,
                    n: f.n,
                    neg_Xn_div_poly: poly_hom.map(f.neg_Xn_div_poly),
                    poly: poly_hom.map(f.poly),
                    poly_deg: f.poly_deg,
                    e: f.e,
                })
                .collect(),
            current_partial_prods: self
                .current_partial_prods
                .into_iter()
                .map(|f| HenselLiftableBarrettReducer {
                    ring: PhantomData,
                    n: f.n,
                    neg_Xn_div_poly: poly_hom.map(f.neg_Xn_div_poly),
                    poly: poly_hom.map(f.poly),
                    poly_deg: f.poly_deg,
                    e: f.e,
                })
                .collect(),
            single_factor: self.single_factor.map(|f| poly_hom.map(f)),
            current_partial_prods_bezout: self
                .current_partial_prods_bezout
                .into_iter()
                .map(|f| poly_hom.map(f))
                .collect(),
            current_poly_ring: new_ring,
        }
    }

    /// Lifts the current factorization to a factorization of `target` in `new_ring`.
    ///
    /// This function requires that `target` is congruent to the product of the current
    /// factors modulo `m^prev_e`, and unfortunately cannot check this with the current information.
    #[instrument(skip_all, level = "trace", fields(e = %new_e))]
    pub fn lift_to<P_new, L>(self, new_e: usize, new_ring: P_new, target: &El<P_new>, mut lift: L) -> HenselLift<P_new>
    where
        P_new: RingStore,
        P_new::Ring: PolyRing,
        L: FnMut(
            &BaseRingStore<P_current>,
            &BaseRingStore<P_new>,
            &El<BaseRingStore<P_current>>,
        ) -> El<BaseRingStore<P_new>>,
    {
        let old_P = &self.current_poly_ring;
        let P = new_ring;
        let n = self.current_factors.len() + 1;

        if n == 1 {
            return HenselLift {
                current_e: new_e,
                current_factor_bezout: Vec::new(),
                current_factors: Vec::new(),
                current_partial_prods: Vec::new(),
                current_partial_prods_bezout: Vec::new(),
                single_factor: Some(target.clone()),
                current_poly_ring: P,
            };
        }

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
            )
            .enumerate()
        {
            let f = if i == 0 {
                target
            } else {
                &partial_prod_reducers.last().unwrap().poly
            };
            let deg_g = old_P.degree(&g.poly).unwrap();
            let deg_h = old_P.degree(&h.poly).unwrap();
            let deg_delta_bound = deg_g + deg_h;
            let mut g = g.change_ring(&P, |poly| {
                P.from_terms(
                    old_P
                        .terms(&poly)
                        .map(|(c, i)| (lift(old_P.base_ring(), P.base_ring(), c), i)),
                )
            });
            let mut h = h.change_ring(&P, |poly| {
                P.from_terms(
                    old_P
                        .terms(&poly)
                        .map(|(c, i)| (lift(old_P.base_ring(), P.base_ring(), c), i)),
                )
            });
            let mut s = P.from_terms(
                old_P
                    .terms(&s)
                    .map(|(c, i)| (lift(old_P.base_ring(), P.base_ring(), c), i)),
            );
            let mut t = P.from_terms(
                old_P
                    .terms(&t)
                    .map(|(c, i)| (lift(old_P.base_ring(), P.base_ring(), c), i)),
            );
            let mut current_e = self.current_e;
            while current_e < new_e {
                // the formula is `g' = g - delta * t`, `h' = h - delta * s` where `delta = gh - f`
                let delta = P.sub_ref_fst(f, P.mul_ref(&g.poly, &h.poly));
                debug_assert!(P.degree(&delta).is_none() || P.degree(&delta).unwrap() < deg_delta_bound);
                let mut delta_g = P.mul_ref(&t, &delta);
                let mut delta_h = P.mul_ref(&s, &delta);
                delta_g = g.poly_div_barett(&P, delta_g).1;
                delta_h = h.poly_div_barett(&P, delta_h).1;
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
                s = h.poly_div_barett(&P, s).1;
                t = g.poly_div_barett(&P, t).1;
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
            single_factor: None,
            current_poly_ring: P,
        };
    }

    pub fn poly_ring(&self) -> &P_current { &self.current_poly_ring }

    pub fn factorization<'a>(&'a self) -> impl Iterator<Item = &'a El<P_current>> {
        self.current_factors.iter().map(|r| &r.poly).chain([self
            .current_partial_prods
            .last()
            .map(|r| &r.poly)
            .or(self.single_factor.as_ref())
            .unwrap()])
    }

    pub fn current_e(&self) -> usize { self.current_e }
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
    fn poly_div_barett<S>(&self, poly_ring: S, poly: El<S>) -> (El<S>, El<S>)
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
            .poly_div_barett(
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
            ring: self.ring,
        }
    }
}

/// Creates the ring `Z/p^eZ` for a potentially large prime p and large power.
/// Furthermore, chooses a suitable convolution with the polynomial ring.
///
/// This function assumes that `prime` is indeed a prime, and may panic or produce
/// wrong results otherwise.
#[instrument(skip_all, level = "trace")]
#[stability::unstable(feature = "enable")]
pub fn create_power_p_poly_ring(prime: El<BigIntRing>, power: usize) -> DensePolyRing<ZnGB<BigIntRing>> {
    let dividing_power_of_two = ZZbig
        .abs_lowest_set_bit(&ZZbig.sub_ref_fst(&prime, ZZbig.one()))
        .unwrap();
    let Zpe = ZnGB::new(ZZbig, ZZbig.pow(prime.clone(), power));
    if dividing_power_of_two >= 10 {
        let n = 1 << (dividing_power_of_two - 1);
        let mut rou = if ZZbig.abs_log2_ceil(&prime).unwrap() <= 57 {
            let Fp = Zn64B::new(int_cast(prime, ZZi64, ZZbig) as u64).as_field().unwrap();
            let rou = get_prim_root_of_unity_pow2(&Fp, dividing_power_of_two).unwrap();
            debug_assert!(Fp.is_neg_one(&Fp.pow(rou, n)));
            Zpe.get_ring()
                .from_int_promise_reduced(int_cast(Fp.smallest_positive_lift(rou), ZZbig, ZZi64))
        } else {
            let Fp = ZnGB::new(ZZbig, prime).as_field().unwrap();
            let rou = get_prim_root_of_unity_pow2(&Fp, dividing_power_of_two).unwrap();
            debug_assert!(Fp.is_neg_one(&Fp.pow(rou.clone(), n)));
            Zpe.get_ring().from_int_promise_reduced(Fp.smallest_positive_lift(rou))
        };
        let one_over_n = Zpe.invert(&Zpe.coerce(&ZZi64, n as i64)).unwrap();
        let one_minus_one_over_n = Zpe.sub_ref_snd(Zpe.one(), &one_over_n);
        let mut e = 1;
        while e < power {
            rou = Zpe.sub(
                Zpe.mul_ref(&rou, &one_minus_one_over_n),
                Zpe.mul_ref_snd(Zpe.invert(&Zpe.pow(rou, n - 1)).unwrap(), &one_over_n),
            );
            e *= 2;
        }
        let convolution: DynConvolution<_> = Arc::new(TypeErasedConvolution::new(NTTConvolution::new(
            Zpe.clone(),
            rou,
            dividing_power_of_two,
        )));
        return DensePolyRing::from(DensePolyRingBase::new_with_convolution(Zpe, "X", Global, convolution));
    } else {
        return DensePolyRing::new(Zpe, "X");
    }
}

#[test]
fn test_hensel_lift() {
    feanor_tracing::DelayedLogger::init_test();

    let FpX = DensePolyRing::new(Zn64B::new(5).as_field().unwrap(), "X");
    let ZpeX = DensePolyRing::new(Zn64B::new(5 * 5 * 5 * 5 * 5 * 5), "X");
    let [g, h] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 3, X + 1]);
    let target = ZpeX.mul_ref(&g, &h);
    let mod_p = ZnReductionMap::new(ZpeX.base_ring(), FpX.base_ring()).unwrap();
    let poly_mod_p = FpX.lifted_hom(&ZpeX, &mod_p);
    let lifter = HenselLift::new(&FpX, vec![poly_mod_p.map_ref(&g), poly_mod_p.map_ref(&h)]).unwrap();
    {
        let lifted = lifter
            .clone()
            .lift_to(6, &ZpeX, &target, |_, _, x| mod_p.any_preimage(*x));
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
            .lift_to(3, &ZpfX, &ZpeX_to_ZpfX.map_ref(&target), |_, _, x| {
                Zpf_to_Zp.any_preimage(*x)
            })
            .lift_to(6, &ZpeX, &target, |_, _, x| Zpe_to_Zpf.any_preimage(*x));
        let [actual_g, actual_h] = lifted.factorization().collect::<Vec<_>>().try_into().unwrap();
        assert_el_eq!(&ZpeX, &g, &actual_g);
        assert_el_eq!(&ZpeX, &h, &actual_h);
    }

    let [g, h] = ZpeX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 25 * X + 3 + 625, X + 1 + 125]);
    let target = ZpeX.mul_ref(&g, &h);
    let lifter = HenselLift::new(&FpX, vec![poly_mod_p.map_ref(&g), poly_mod_p.map_ref(&h)]).unwrap();
    {
        let lifted = lifter
            .clone()
            .lift_to(6, &ZpeX, &target, |_, _, x| mod_p.any_preimage(*x));
        let [actual_g, actual_h] = lifted.factorization().collect::<Vec<_>>().try_into().unwrap();
        assert_el_eq!(&ZpeX, &g, &actual_g);
        assert_el_eq!(&ZpeX, &h, &actual_h);
    }
}
