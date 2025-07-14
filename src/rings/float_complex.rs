use std::f64::EPSILON;
use std::f64::consts::PI;

use crate::homomorphism::CanHomFrom;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::field::Field;
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::impl_eq_based_self_iso;
use crate::ring::*;
use crate::divisibility::{DivisibilityRing, Domain};
use crate::rings::approx_real::float::Real64;
use crate::rings::rational::RationalFieldBase;

///
/// An approximate implementation of the complex numbers `C`, using 64 bit floating
/// point numbers.
/// 
/// # Warning
/// 
/// Since floating point numbers do not exactly represent the complex numbers, and this crate follows
/// a mathematically precise approach, we cannot provide any function related to equality.
/// In particular, `Complex64Base.eq_el(a, b)` is not supported, and will panic. 
/// Hence, this ring has only limited use within this crate, and is currently only used for
/// floating-point FFTs. 
/// 
#[derive(Clone, Copy, PartialEq)]
pub struct Complex64Base;

#[derive(Clone, Copy)]
pub struct Complex64El(f64, f64);

///
/// [`RingStore`] corresponding to [`Complex64Base`]
/// 
pub type Complex64 = RingValue<Complex64Base>;

impl Complex64 {

    pub const RING: Self = RingValue::from(Complex64Base);
    pub const I: Complex64El = Complex64El(0., 1.);
}

impl Complex64Base {

    pub fn abs(&self, Complex64El(re, im): Complex64El) -> f64 {
        (re * re + im * im).sqrt()
    }

    pub fn conjugate(&self, Complex64El(re, im): Complex64El) -> Complex64El {
        Complex64El(re, -im)
    }

    pub fn exp(&self, Complex64El(exp_re, exp_im): Complex64El) -> Complex64El {
        let angle = exp_im;
        let abs = exp_re.exp();
        Complex64El(abs * angle.cos(), abs * angle.sin())
    }

    pub fn closest_gaussian_int(&self, Complex64El(re, im): Complex64El) -> (i64, i64) {
        (re.round() as i64, im.round() as i64)
    }

    pub fn ln_main_branch(&self, Complex64El(re, im): Complex64El) -> Complex64El {
        Complex64El(self.abs(Complex64El(re, im)).ln(), im.atan2(re))
    }

    pub fn is_absolute_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, absolute_threshold: f64) -> bool {
        self.abs(self.sub(lhs, rhs)) < absolute_threshold
    }

    pub fn is_relative_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, relative_limit: f64) -> bool {
        self.is_absolute_approx_eq(lhs, rhs, self.abs(lhs) * relative_limit)
    }

    pub fn is_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, precision: u64) -> bool {
        let scaled_precision = precision as f64 * EPSILON;
        if self.is_absolute_approx_eq(lhs, self.zero(), scaled_precision) {
            self.is_absolute_approx_eq(rhs, self.zero(), scaled_precision)
        } else {
            self.is_relative_approx_eq(lhs, rhs, scaled_precision)
        }
    }

    pub fn from_f64(&self, x: f64) -> Complex64El {
        Complex64El(x, 0.)
    }

    pub fn root_of_unity(&self, i: i64, n: i64) -> Complex64El {
        self.exp(self.mul(self.from_f64((i as f64 / n as f64) * (2. * PI)), Complex64::I))
    }

    pub fn re(&self, Complex64El(re, _im): Complex64El) -> f64 {
        re
    }

    pub fn im(&self, Complex64El(_re, im): Complex64El) -> f64 {
        im
    }
}

impl Complex64 {
    
    pub fn abs(&self, val: Complex64El) -> f64 { self.get_ring().abs(val) }

    pub fn conjugate(&self, val: Complex64El) -> Complex64El { self.get_ring().conjugate(val) }

    pub fn exp(&self, exp: Complex64El) -> Complex64El { self.get_ring().exp(exp) }

    pub fn closest_gaussian_int(&self, val: Complex64El) -> (i64, i64) { self.get_ring().closest_gaussian_int(val) }

    pub fn ln_main_branch(&self, val: Complex64El) -> Complex64El { self.get_ring().ln_main_branch(val) }

    pub fn is_absolute_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, absolute_threshold: f64) -> bool { self.get_ring().is_absolute_approx_eq(lhs, rhs, absolute_threshold) }

    pub fn is_relative_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, relative_limit: f64) -> bool { self.get_ring().is_relative_approx_eq(lhs, rhs, relative_limit) }

    pub fn is_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, precision: u64) -> bool { self.get_ring().is_approx_eq(lhs, rhs, precision) }

    pub fn from_f64(&self, x: f64) -> Complex64El { self.get_ring().from_f64(x) }

    pub fn root_of_unity(&self, i: i64, n: i64) -> Complex64El { self.get_ring().root_of_unity(i, n) }

    pub fn re(&self, x: Complex64El) -> f64 { self.get_ring().re(x) }

    pub fn im(&self, x: Complex64El) -> f64 { self.get_ring().im(x) }
}

impl RingBase for Complex64Base {
 
    type Element = Complex64El;
    
    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, Complex64El(lhs_re, lhs_im): &mut Self::Element, Complex64El(rhs_re, rhs_im): Self::Element) {
        *lhs_re += rhs_re;
        *lhs_im += rhs_im;
    }

    fn negate_inplace(&self, Complex64El(re, im): &mut Self::Element) {
        *re = -*re;
        *im = -*im;
    }

    fn mul_assign(&self, Complex64El(lhs_re, lhs_im): &mut Self::Element, Complex64El(rhs_re, rhs_im): Self::Element) {
        let new_im = *lhs_re * rhs_im + *lhs_im * rhs_re;
        *lhs_re = *lhs_re * rhs_re - *lhs_im * rhs_im;
        *lhs_im = new_im;
    }

    fn from_int(&self, value: i32) -> Self::Element {
        Complex64El(value as f64, 0.)
    }
    
    fn eq_el(&self, _: &Self::Element, _: &Self::Element) -> bool {
        panic!("Cannot provide equality on approximate rings")
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        self.exp(self.mul(self.ln_main_branch(x), Complex64El(integers.to_float_approx(power), 0.)))
    }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn is_approximate(&self) -> bool { true }

    fn dbg_within<'a>(&self, Complex64El(re, im): &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if env >= EnvBindingStrength::Product {
            write!(out, "({} + {}i)", re, im)
        } else {
            write!(out, "{} + {}i", re, im)
        }
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }
    
    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(ZZ.zero())
    }
}

impl_eq_based_self_iso!{ Complex64Base }

impl Domain for Complex64Base {}

impl DivisibilityRing for Complex64Base {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let abs_sqr = self.abs(*rhs) * self.abs(*rhs);
        let Complex64El(res_re, res_im) =  self.mul(*lhs, self.conjugate(*rhs));
        return Some(Complex64El(res_re / abs_sqr, res_im / abs_sqr));
    }
}

impl PrincipalIdealRing for Complex64Base {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div(lhs, rhs)
    }
    
    fn extended_ideal_gen(&self, _lhs: &Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }
}

impl EuclideanRing for Complex64Base {

    fn euclidean_div_rem(&self, _lhs: Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }

    fn euclidean_deg(&self, _: &Self::Element) -> Option<usize> {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }
}

impl Field for Complex64Base {
    
    fn div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.checked_left_div(lhs, rhs).unwrap()
    }
}

impl RingExtension for Complex64Base {

    type BaseRing = Real64;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &Real64::RING
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.from_f64(x)
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        lhs.0 *= *rhs;
        lhs.1 *= *rhs;
    }
}

impl<I: ?Sized + IntegerRing> CanHomFrom<I> for Complex64Base {
    
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &I) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &I, el: <I as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &I, el: &<I as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        self.from_f64(from.to_float_approx(el))
    }
}

impl<I> CanHomFrom<RationalFieldBase<I>> for Complex64Base
    where I: IntegerRingStore,
        I::Type: IntegerRing
{    
    type Homomorphism = <Self as CanHomFrom<I::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &RationalFieldBase<I>) -> Option<Self::Homomorphism> {
        self.has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &RationalFieldBase<I>, el: <RationalFieldBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &RationalFieldBase<I>, el: &<RationalFieldBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.div(&self.map_in_ref(from.base_ring().get_ring(), from.num(el), hom), &self.map_in_ref(from.base_ring().get_ring(), from.den(el), hom))
    }
}

#[test]
fn test_pow() {
    let CC = Complex64::RING;
    let i = Complex64::I;
    assert!(CC.is_approx_eq(CC.negate(i), CC.pow(i, 3), 1));
    assert!(!CC.is_approx_eq(CC.negate(i), CC.pow(i, 1024 + 3), 1));
    assert!(CC.is_approx_eq(CC.negate(i), CC.pow(i, 1024 + 3), 100));
    assert!(CC.is_approx_eq(CC.exp(CC.mul(CC.from_f64(PI / 4.), i)), CC.mul(CC.add(CC.one(), i), CC.from_f64(2f64.powf(-0.5))), 1));

    let seventh_root_of_unity = CC.exp(CC.mul(i, CC.from_f64(2. * PI / 7.)));
    assert!(CC.is_approx_eq(CC.pow(seventh_root_of_unity, 7 * 100 + 1), seventh_root_of_unity, 1000));
}

#[test]
fn test_mul() {
    let CC = Complex64::RING;
    let i = Complex64::I;
    assert!(CC.is_approx_eq(CC.mul(i, i), CC.from_f64(-1.), 1));
    assert!(CC.is_approx_eq(CC.mul(i, CC.negate(i)), CC.from_f64(1.), 1));
    assert!(CC.is_approx_eq(CC.mul(CC.add(i, CC.one()), i), CC.sub(i, CC.one()), 1));
}