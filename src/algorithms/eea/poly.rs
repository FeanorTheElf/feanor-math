use std::mem::swap;
use std::cmp::max;

use crate::algorithms::erathostenes::enumerate_primes;
use crate::algorithms::hensel::hensel_lift;
use crate::algorithms::poly_factor::integer::max_coeff_of_factor;
use crate::field::Field;
use crate::homomorphism::CanHomFrom;
use crate::homomorphism::Homomorphism;
use crate::divisibility::*;
use crate::integer::IntegerRing;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::rings::poly::*;
use crate::ring::*;
use crate::integer::*;
use crate::pid::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::choose_zn_impl;
use crate::rings::zn::zn_64::{Zn, ZnBase};
use crate::rings::zn::*;
use crate::field::FieldStore;

use super::lcm;

struct IntegerPolynomialGCDUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Q: RingStore,
        Q::Type: PolyRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>

{
    ZZX: P,
    FpX: Q,
    f: &'a El<P>,
    g: &'a El<P>,
    leading_coeff: &'a El<<P::Type as RingExtension>::BaseRing>
}

impl<'a, P, Q> ZnOperation<Option<El<P>>> for IntegerPolynomialGCDUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Q: RingStore,
        Q::Type: PolyRing + EuclideanRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    fn call<R: ZnRingStore>(self, Zpe: R) -> Option<El<P>>
        where R::Type: ZnRing
    {
        let Fp = self.FpX.base_ring();
        let ZZ = self.ZZX.base_ring();
        let ZZ_to_Fp = Fp.can_hom(ZZ).unwrap();
        let ZZX_to_FpX = self.FpX.lifted_hom(&self.ZZX, &ZZ_to_Fp);

        let f_mod_p = ZZX_to_FpX.map_ref(&self.f);
        let g_mod_p = ZZX_to_FpX.map_ref(&self.g);
        let mut d_mod_p = self.FpX.ideal_gen(&f_mod_p, &g_mod_p);
        let scale = Fp.invert(self.FpX.lc(&d_mod_p).unwrap()).unwrap();
        self.FpX.inclusion().mul_assign_map(&mut d_mod_p, scale);
        let mut f_over_d_mod_p = self.FpX.checked_div(&f_mod_p, &d_mod_p).unwrap();
        let scale = Fp.invert(self.FpX.lc(&f_over_d_mod_p).unwrap()).unwrap();
        self.FpX.inclusion().mul_assign_map(&mut f_over_d_mod_p, scale);

        let ZpeX = DensePolyRing::new(&Zpe, "X");
        let ZZ_to_Zpe = Zpe.can_hom(Zpe.integer_ring()).unwrap();
        let reduce_pe = |c: &El<<P::Type as RingExtension>::BaseRing>| ZZ_to_Zpe.map(int_cast(ZZ.clone_el(c), Zpe.integer_ring(), ZZ));
        let lc_f_mod_pe = Zpe.invert(&reduce_pe(self.ZZX.lc(&self.f).unwrap())).unwrap();
        let f_monic_mod_pe = ZpeX.from_terms(self.ZZX.terms(&self.f).map(|(c, i)| (Zpe.mul_ref_snd(reduce_pe(c), &lc_f_mod_pe), i)));

        let (d_lifted, _f_over_d_lifted) = hensel_lift(
            &ZpeX, 
            &self.FpX, 
            &self.FpX, 
            &f_monic_mod_pe, 
            (&d_mod_p, &f_over_d_mod_p)
        );

        // if the prime `p` is good, then `d_lifted` is now the reduction
        // of `d_frac = gcd(f, g) / lc(gcd(f, g))` modulo `p^e` (since `d_lifted` is monic).
        // unfortunately, `d_frac` is not integral, thus we cannot take the shortest lift of
        // `d_lifted`. However, we can take the shortest lift of `a * d_lifted` for `a = gcd(lc(f), lc(g))`,
        // since `lc(gcd(f, g)) | a`, thus `a * d_frac` is integral. Note that this means the size bound
        // has to be for the factors of `a f`, not for the factors of `f`.

        let leading_coeff_mod_pe = reduce_pe(&self.leading_coeff);
        let scaled_d_lifted = self.ZZX.from_terms(ZpeX.terms(&d_lifted).map(|(c, i)| (int_cast(Zpe.smallest_lift(Zpe.mul_ref(c, &leading_coeff_mod_pe)), ZZ, Zpe.integer_ring()), i)));
        let d = make_primitive(scaled_d_lifted, &self.ZZX);

        if self.ZZX.checked_div(&self.f, &d).is_some() {
            return Some(d);
        } else {
            return None;
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn poly_eea_global<R>(fst: El<R>, snd: El<R>, ring: R) -> (El<R>, El<R>, El<R>) 
    where R: RingStore,
        R::Type: PolyRing + EuclideanRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    let (mut a, mut b) = (ring.clone_el(&fst), ring.clone_el(&snd));

    let a_balance_factor = ring.get_ring().balance_element(&mut a);
    let b_balance_factor = ring.get_ring().balance_element(&mut b);

    let (mut sa, mut ta) = (ring.inclusion().map(ring.base_ring().invert(&a_balance_factor).unwrap()), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.inclusion().map(ring.base_ring().invert(&b_balance_factor).unwrap()));

    while !ring.is_zero(&b) {
        let balance_factor = ring.get_ring().balance_element(&mut b);
        let inv_balance_factor = ring.base_ring().invert(&balance_factor).unwrap();
        ring.inclusion().mul_assign_ref_map(&mut tb, &inv_balance_factor);
        ring.inclusion().mul_assign_ref_map(&mut sb, &inv_balance_factor);

        let scale_factor = ring.base_ring().pow(ring.base_ring().clone_el(ring.lc(&b).unwrap()), ring.degree(&b).unwrap());
        ring.inclusion().mul_assign_ref_map(&mut a, &scale_factor);
        ring.inclusion().mul_assign_ref_map(&mut ta, &scale_factor);
        ring.inclusion().mul_assign_ref_map(&mut sa, &scale_factor);

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quo, &tb));
        sa = ring.sub(sa, ring.mul_ref(&quo, &sb));
        a = rem;
        
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }

    if ring.is_zero(&a) {
        return (sa, ta, a);
    }

    let lc_inv = ring.base_ring().invert(ring.lc(&a).unwrap()).unwrap();
    ring.inclusion().mul_assign_ref_map(&mut a, &lc_inv);
    ring.inclusion().mul_assign_ref_map(&mut sa, &lc_inv);
    ring.inclusion().mul_assign_ref_map(&mut ta, &lc_inv);

    return (sa, ta, a);
}

fn make_primitive<R>(f: El<R>, ZZX: R) -> El<R>
    where R: RingStore,
        R::Type: PolyRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
{
    let content = ZZX.terms(&f).map(|(c, _)| c).fold(ZZX.base_ring().zero(), |a, b| ZZX.base_ring().ideal_gen(&a, b));
    return ZZX.from_terms(ZZX.terms(&f).map(|(c, i)| (ZZX.base_ring().checked_div(c, &content).unwrap(), i)));
}

#[stability::unstable(feature = "enable")]
pub fn integer_poly_gcd_local<R>(mut fst: El<R>, mut snd: El<R>, ZZX: R) -> El<R>
    where R: RingStore,
        R::Type: PolyRing + DivisibilityRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        ZnBase: CanHomFrom<<<R::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    if ZZX.is_zero(&fst) {
        return snd;
    } else if ZZX.is_zero(&snd) {
        return fst;
    } else if ZZX.degree(&fst).unwrap() == 0 || ZZX.degree(&snd).unwrap() == 0 {
        return ZZX.one();
    }

    let ZZ = ZZX.base_ring();

    if ZZX.degree(&fst).unwrap() > ZZX.degree(&snd).unwrap() {
        std::mem::swap(&mut fst, &mut snd);
    }

    let f = make_primitive(fst, &ZZX);
    let g = make_primitive(snd, &ZZX);
    let leading_coeff = ZZ.ideal_gen(ZZX.lc(&f).unwrap(), ZZX.lc(&g).unwrap());

    let ZZbig = BigIntRing::RING;
    let ZZi64 = StaticRing::<i64>::RING;
    let bound = ZZbig.clone_el(<_ as OrderedRingStore>::max(&ZZbig, &max_coeff_of_factor(&ZZX, &f, &leading_coeff), &max_coeff_of_factor(&ZZX, &g, &leading_coeff)));

    // small primes have lower probability of working
    for p in enumerate_primes(&StaticRing::<i64>::RING, &10000).into_iter().skip(100) {

        let p_big = int_cast(p as i64, ZZ, StaticRing::<i64>::RING);
        if ZZ.checked_div(ZZX.lc(&f).unwrap(), &p_big).is_some() || ZZ.checked_div(ZZX.lc(&g).unwrap(), &p_big).is_some() {
            continue;
        }

        // the gcd in `Fp[X]` now cannot be "smaller" (i.e. lower degree) than the gcd in `ZZ`, since `lc(gcd(f, g))` is not divisible by `p`;
        // it can still be "larger", however in this case, we will fail and try a new prime; this is quite unlikely

        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let FpX = DensePolyRing::new(Fp, "X");

        let exponent = max(2, ZZbig.abs_log2_ceil(&bound).unwrap() / ZZi64.abs_log2_floor(&p).unwrap() + 1);
        let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZi64), exponent);
        debug_assert!(ZZbig.is_gt(&modulus, &ZZbig.mul_ref_fst(&bound, ZZbig.int_hom().map(2))));

        if let Some(result) = choose_zn_impl(ZZbig, modulus, IntegerPolynomialGCDUsingHenselLifting {
            f: &f, g: &g, ZZX: &ZZX, FpX: FpX, leading_coeff: &leading_coeff
        }) {
            return result;
        }
    }
    unreachable!()
}

#[stability::unstable(feature = "enable")]
pub fn rational_poly_gcd_local<R, I>(fst: El<R>, snd: El<R>, QQX: R) -> El<R>
    where R: RingStore,
        R::Type: PolyRing + DivisibilityRing,
        <R::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing,
        ZnBase: CanHomFrom<I::Type>
{
    if QQX.is_zero(&fst) {
        return snd;
    } else if QQX.is_zero(&snd) {
        return fst;
    }

    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();
    let fst_den_lcm = QQX.terms(&fst).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| lcm(a, ZZ.clone_el(b), ZZ));
    let snd_den_lcm = QQX.terms(&fst).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| lcm(a, ZZ.clone_el(b), ZZ));

    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(&fst).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(QQ.get_ring().num(c), &fst_den_lcm), QQ.get_ring().den(c)).unwrap(), i)));
    let g = ZZX.from_terms(QQX.terms(&snd).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(QQ.get_ring().num(c), &snd_den_lcm), QQ.get_ring().den(c)).unwrap(), i)));
    let d = integer_poly_gcd_local(f, g, &ZZX);
    let d_lc = QQ.inclusion().map_ref(ZZX.lc(&d).unwrap());

    return QQX.from_terms(ZZX.terms(&d).map(|(c, i)| (QQ.div(&QQ.inclusion().map_ref(c), &d_lc), i)));
}

#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::rational::RationalField;

#[test]
fn test_polynomial_eea_global() {
    let ring = DensePolyRing::new(RationalField::new(BigIntRing::RING), "X");
    let [f, g, expected_gcd] = ring.with_wrapped_indeterminate(|X| [
        (X.pow_ref(2) + 1) * (X.pow_ref(3) + 2),
        (X.pow_ref(2) + 1) * (2 * X + 1),
        X.pow_ref(2) + 1
    ]);

    let (s, t, actual_gcd) = poly_eea_global(ring.clone_el(&f), ring.clone_el(&g), &ring);
    assert_el_eq!(ring, &expected_gcd, actual_gcd);
    assert_el_eq!(ring, &expected_gcd, ring.add(ring.mul_ref(&s, &f), ring.mul_ref(&t, &g)));
}

#[test]
fn test_polynomial_eea_local() {
    let ring = DensePolyRing::new(BigIntRing::RING, "X");
    let [f, g, expected_gcd] = ring.with_wrapped_indeterminate(|X| [
        (X.pow_ref(2) + 1) * (X.pow_ref(3) + 2),
        (X.pow_ref(2) + 1) * (2 * X + 1),
        X.pow_ref(2) + 1
    ]);

    let actual_gcd = integer_poly_gcd_local(ring.clone_el(&f), ring.clone_el(&g), &ring);
    assert_el_eq!(ring, &expected_gcd, actual_gcd);

    let ring = DensePolyRing::new(BigIntRing::RING, "X");
    let [f_over_d, g_over_d, d] = ring.with_wrapped_indeterminate(|X| [
        (X.pow_ref(2) + 1) * (3 * X.pow_ref(3) + 2),
        (2 * X.pow_ref(2) + 1) * (2 * X + 1),
        X.pow_ref(3) * 7 + X - 1
    ]);

    let actual_gcd = integer_poly_gcd_local(ring.mul_ref(&f_over_d, &d), ring.mul_ref(&g_over_d, &d), &ring);
    assert_el_eq!(ring, &d, actual_gcd);
}

#[test]
fn test_int_poly_gcd_random() {
    let mut rng = oorandom::Rand64::new(1);
    let ZZbig = BigIntRing::RING;
    let QQ = RationalField::new(ZZbig);
    let ring = DensePolyRing::new(&QQ, "X");
    let coeff_bound = ZZbig.int_hom().map(10);

    for _ in 0..20 {
        let d = ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f = ring.mul_ref_snd(ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i))), &d);
        let g = ring.mul_ref_snd(ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i))), &d);

        let (_, _, d1) = poly_eea_global(ring.clone_el(&f), ring.clone_el(&g), &ring);
        let d2 = rational_poly_gcd_local(ring.clone_el(&f), ring.clone_el(&g), &ring);

        assert_el_eq!(&ring, &d1, &d2);
        assert!(ring.checked_div(&d1, &d).is_some());
        assert!(ring.checked_div(&f, &d1).is_some());
        assert!(ring.checked_div(&g, &d1).is_some());
    }
}