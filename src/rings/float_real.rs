use std::f64::EPSILON;

use crate::ordered::OrderedRing;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::field::Field;
use crate::integer::{int_cast, IntegerRing, IntegerRingStore};
use crate::primitive_int::StaticRing;
use crate::{impl_eq_based_self_iso, ring::*};
use crate::homomorphism::*;
use crate::divisibility::{DivisibilityRing, Domain};

use super::rational::{RationalField, RationalFieldBase};

///
/// An approximate implementation of the real numbers `R`, using 64 bit floating
/// point numbers.
/// 
/// # Warning
/// 
/// Since floating point numbers do not exactly represent the real numbers, and this crate follows
/// a mathematically precise approach, we cannot provide any function related to equality.
/// In particular, `Real64Base.eq_el(a, b)` is not supported, and will panic. 
/// Hence, this ring has only limited use within this crate, and is currently only used for
/// floating-point FFTs. 
/// 
#[derive(Clone, Copy, PartialEq)]
pub struct Real64Base;

///
/// [`RingStore`] corresponding to [`Real64Base`]
/// 
pub type Real64 = RingValue<Real64Base>;

impl Real64 {

    pub const RING: RingValue<Real64Base> = RingValue::from(Real64Base);
}

impl Real64Base {

    pub fn is_absolute_approx_eq(&self, lhs: <Self as RingBase>::Element, rhs: <Self as RingBase>::Element, absolute_threshold: f64) -> bool {
        (lhs - rhs).abs() < absolute_threshold
    }

    pub fn is_relative_approx_eq(&self, lhs: <Self as RingBase>::Element, rhs: <Self as RingBase>::Element, relative_threshold: f64) -> bool {
        self.is_absolute_approx_eq(lhs, rhs, (lhs.abs() + rhs.abs()) * relative_threshold)
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

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs += rhs;
    }

    fn negate_inplace(&self, x: &mut Self::Element) {
        *x = -*x;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs *= rhs;
    }

    fn from_int(&self, value: i32) -> Self::Element {
        value as f64
    }
    
    fn eq_el(&self, _: &Self::Element, _: &Self::Element) -> bool {
        panic!("Cannot provide equality on approximate rings")
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        if integers.get_ring().representable_bits().is_some() && integers.get_ring().representable_bits().unwrap() < i32::BITS as usize {
            x.powi(int_cast(integers.clone_el(power), &StaticRing::<i32>::RING, integers))
        } else {
            x.powf(integers.to_float_approx(power))
        }
    }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn is_approximate(&self) -> bool { true }

    fn dbg<'a>(&self, x: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", x)
    }
    
    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(ZZ.zero())
    }
}

impl_eq_based_self_iso!{ Real64Base }

impl Domain for Real64Base {}

impl DivisibilityRing for Real64Base {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        assert!(*rhs != 0.);
        return Some(*lhs / *rhs);
    }
}

impl PrincipalIdealRing for Real64Base {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div(lhs, rhs)
    }
    
    fn extended_ideal_gen(&self, _lhs: &Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }
}

impl EuclideanRing for Real64Base {

    fn euclidean_div_rem(&self, _lhs: Self::Element, _rhs: &Self::Element) -> (Self::Element, Self::Element) {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }

    fn euclidean_deg(&self, _: &Self::Element) -> Option<usize> {
        panic!("Since Complex64 is only approximate, this cannot be implemented properly")
    }
}

impl Field for Real64Base {

    fn div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.checked_left_div(lhs, rhs).unwrap()
    }
}

impl OrderedRing for Real64Base {

    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        f64::partial_cmp(lhs, rhs).unwrap()
    }
}

impl<I> CanHomFrom<I> for Real64Base 
    where I: ?Sized + IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &I) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &I, el: <I as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        from.to_float_approx(&el)
    }

    fn map_in_ref(&self, from: &I, el: &<I as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        from.to_float_approx(el)
    }
}

impl<I> CanHomFrom<RationalFieldBase<I>> for Real64Base 
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &RationalFieldBase<I>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &RationalFieldBase<I>, el: El<RationalField<I>>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &RationalFieldBase<I>, el: &El<RationalField<I>>, _hom: &Self::Homomorphism) -> Self::Element {
        from.base_ring().to_float_approx(from.num(el)) / from.base_ring().to_float_approx(from.den(el))
    }
}