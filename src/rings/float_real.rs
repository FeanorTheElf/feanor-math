use std::f64::EPSILON;
use std::f64::consts::PI;

use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::field::Field;
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::ring::*;
use crate::homomorphism::*;
use crate::divisibility::{DivisibilityRing, Domain};

#[derive(Clone, Copy, PartialEq)]
pub struct Real64Base;

pub type Real64 = RingValue<Real64Base>;

impl Real64 {

    pub const RING: RingValue<Real64Base> = RingValue::from(Real64Base);
}

impl Real64Base {

    pub fn is_absolute_approx_eq(&self, lhs: <Self as RingBase>::Element, rhs: <Self as RingBase>::Element, absolute_threshold: f64) -> bool {
        self.abs(self.sub(lhs, rhs)) < absolute_threshold
    }

    pub fn is_relative_approx_eq(&self, lhs: <Self as RingBase>::Element, rhs: <Self as RingBase>::Element, relative_threshold: f64) -> bool {
        self.is_absolute_approx_eq(lhs, rhs, self.abs(lhs) * relative_threshold)
    }

    pub fn is_approx_eq(&self, lhs: <Self as RingBase>::Element, rhs: <Self as RingBase>::Element, precision: u64) -> bool {
        let scaled_precision = precision as f64 * EPSILON;
        if self.is_absolute_approx_eq(lhs, self.zero(), scaled_precision) {
            self.is_absolute_approx_eq(rhs, self.zero(), scaled_precision)
        } else {
            self.is_relative_approx_eq(lhs, rhs, scaled_precision)
        }
    }
}

impl RingBase for Real64Base {
 
    type Element = f64;
    
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

    fn dbg<'a>(&self, Complex64El(re, im): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{} + {}i", re, im)
    }
    
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(ZZ.zero())
    }
}

impl_eq_based_self_iso!{ Complex64 }

impl Domain for Complex64 {}

impl DivisibilityRing for Complex64 {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let abs_sqr = self.abs(*rhs) * self.abs(*rhs);
        let Complex64El(res_re, res_im) =  self.mul(*lhs, self.conjugate(*rhs));
        return Some(Complex64El(res_re / abs_sqr, res_im / abs_sqr));
    }
}

impl PrincipalIdealRing for Complex64 {

    fn ideal_gen(&self, _lhs: &Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }
}

impl EuclideanRing for Complex64 {

    fn euclidean_div_rem(&self, _lhs: Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }

    fn euclidean_deg(&self, _: &Self::Element) -> Option<usize> {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
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