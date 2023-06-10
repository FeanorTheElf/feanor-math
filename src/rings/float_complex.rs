use std::f64::EPSILON;

use crate::euclidean::EuclideanRing;
use crate::field::Field;
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::ring::*;
use crate::divisibility::DivisibilityRing;

#[derive(Clone, Copy, PartialEq)]
pub struct Complex64;

#[derive(Clone, Copy)]
pub struct Complex64El(f64, f64);

impl Complex64 {

    pub const RING: RingValue<Complex64> = RingValue::from(Complex64);

    pub const I: Complex64El = Complex64El(0., 1.);

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
        RingRef::new(self).println(&self.sub(lhs, rhs));
        println!("{}, {}", self.abs(self.sub(lhs, rhs)), absolute_threshold);
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
}

impl RingValue<Complex64> {
    
    pub fn abs(&self, val: Complex64El) -> f64 { self.get_ring().abs(val) }

    pub fn conjugate(&self, val: Complex64El) -> Complex64El { self.get_ring().conjugate(val) }

    pub fn exp(&self, exp: Complex64El) -> Complex64El { self.get_ring().exp(exp) }

    pub fn closest_gaussian_int(&self, val: Complex64El) -> (i64, i64) { self.get_ring().closest_gaussian_int(val) }

    pub fn ln_main_branch(&self, val: Complex64El) -> Complex64El { self.get_ring().ln_main_branch(val) }

    pub fn is_absolute_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, absolute_threshold: f64) -> bool { self.get_ring().is_absolute_approx_eq(lhs, rhs, absolute_threshold) }

    pub fn is_relative_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, relative_limit: f64) -> bool { self.get_ring().is_relative_approx_eq(lhs, rhs, relative_limit) }

    pub fn is_approx_eq(&self, lhs: Complex64El, rhs: Complex64El, precision: u64) -> bool { self.get_ring().is_approx_eq(lhs, rhs, precision) }

    pub fn from_f64(&self, x: f64) -> Complex64El { self.get_ring().from_f64(x) }
}

impl RingBase for Complex64 {
 
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
        where R::Type: IntegerRing,
            Self: SelfIso 
    {
        self.exp(self.mul(self.ln_main_branch(x), Complex64El(integers.to_float_approx(power), 0.)))
    }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn is_approximate(&self) -> bool { true }

    fn dbg<'a>(&self, Complex64El(re, im): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{} + {}i", re, im)
    }
}

impl CanonicalHom<Complex64> for Complex64 {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &Complex64) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _: &Complex64, el: <Complex64 as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<Complex64> for Complex64 {

    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &Complex64) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, _: &Complex64, el: Self::Element, _: &Self::Isomorphism) -> <Complex64 as RingBase>::Element {
        el
    }
}

impl DivisibilityRing for Complex64 {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let abs = self.abs(*rhs);
        let Complex64El(res_re, res_im) =  self.mul(*lhs, self.conjugate(*rhs));
        return Some(Complex64El(res_re / abs, res_im / abs));
    }
}

impl EuclideanRing for Complex64 {

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, _: &Self::Element) -> Option<usize> {
        // since we do not provide equality checking, we cannot really implement this correctly;
        // anyway, this should do in almost all situations
        None
    }
}

impl Field for Complex64 {}

#[test]
fn test_pow() {
    let CC = Complex64::RING;
    let i = Complex64::I;
    assert!(CC.is_approx_eq(CC.negate(i), CC.pow(i, 3), 1));
    assert!(!CC.is_approx_eq(CC.negate(i), CC.pow(i, 1024 + 3), 1));
    assert!(CC.is_approx_eq(CC.negate(i), CC.pow(i, 1024 + 3), 100));
    assert!(CC.is_approx_eq(CC.exp(CC.mul(CC.from_f64(std::f64::consts::PI / 4.), i)), CC.mul(CC.add(CC.one(), i), CC.from_f64(2f64.powf(-0.5))), 1));

    let seventh_root_of_unity = CC.exp(CC.mul(i, CC.from_f64(2. * std::f64::consts::PI / 7.)));
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