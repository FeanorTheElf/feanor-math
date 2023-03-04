use std::{ops::{AddAssign, SubAssign, MulAssign, Neg, Div, Rem}, marker::PhantomData};
use crate::{ring::*, euclidean::EuclideanRing, divisibility::DivisibilityRing, ordered::OrderedRing};
use crate::integer::*;
use crate::algorithms::multiply::KaratsubaHint;

pub trait PrimitiveInt: AddAssign + SubAssign + MulAssign + Neg<Output = Self> + Eq + From<i8> + TryFrom<i32> + TryFrom<i128> + Into<i128> + Copy + Div<Self, Output = Self> + Rem<Self, Output = Self> {}

impl PrimitiveInt for i8 {}
impl PrimitiveInt for i16 {}
impl PrimitiveInt for i32 {}
impl PrimitiveInt for i64 {}
impl PrimitiveInt for i128 {}

impl<T: PrimitiveInt, S: PrimitiveInt> CanonicalHom<StaticRingBase<T>> for StaticRingBase<S> {

    fn has_canonical_hom(&self, _: &StaticRingBase<T>) -> bool {
        true
    }

    fn map_in(&self, _: &StaticRingBase<T>, el: T) -> S {
        S::try_from(el.into()).map_err(|_| ()).unwrap()
    }
}

impl<T: PrimitiveInt, S: PrimitiveInt> CanonicalIso<StaticRingBase<T>> for StaticRingBase<S> {
    
    fn has_canonical_iso(&self, _: &StaticRingBase<T>) -> bool {
        true
    }

    fn map_out(&self, _: &StaticRingBase<T>, el: S) -> T {
        T::try_from(el.into()).map_err(|_| ()).unwrap()
    }
}

impl<T: PrimitiveInt> DivisibilityRing for StaticRingBase<T> {
    
    fn checked_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let (div, rem) = self.euclidean_div_rem(*lhs, rhs);
        if self.is_zero(&rem) {
            return Some(div);
        } else {
            return None;
        }
    }
}

impl<T: PrimitiveInt> EuclideanRing for StaticRingBase<T> {
    
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        (lhs / *rhs, lhs % *rhs)
    }

    fn euclidean_deg(&self, val: &Self::Element) -> usize {
        self.map_out(StaticRing::<i128>::RING.get_ring(), *val).abs() as usize
    }
}

impl<T: PrimitiveInt> OrderedRing for StaticRingBase<T> {
    
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        self.map_out(StaticRing::<i128>::RING.get_ring(), *lhs).cmp(
            &self.map_out(StaticRing::<i128>::RING.get_ring(), *rhs)
        )
    }
}

impl<T: PrimitiveInt> IntegerRing for StaticRingBase<T> {

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool {
        match self.map_out(StaticRing::<i128>::RING.get_ring(), *value) {
            i128::MIN => i == i128::BITS as usize - 1,
            x => (x.abs() >> i) & 1 == 1
        }
    }

    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        match self.map_out(StaticRing::<i128>::RING.get_ring(), *value) {
            0 => None,
            i128::MIN => Some(i128::BITS as usize - 1),
            x => Some(i128::BITS as usize - x.abs().leading_zeros() as usize - 1)
        }
    }

    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = self.map_in(&StaticRing::<i128>::RING.get_ring(), self.map_out(&StaticRing::<i128>::RING.get_ring(), *value) >> power);
    }

    fn mul_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = self.map_in(&StaticRing::<i128>::RING.get_ring(), self.map_out(&StaticRing::<i128>::RING.get_ring(), *value) << power);
    }
}

pub struct StaticRingBase<T> {
    element: PhantomData<T>
}

impl<T: PrimitiveInt> RingValue<StaticRingBase<T>> {
    pub const RING: StaticRing<T> = RingValue::new(StaticRingBase { element: PhantomData });
}

impl<T> Copy for StaticRingBase<T> {}

impl<T> Clone for StaticRingBase<T> {

    fn clone(&self) -> Self {
        *self
    }    
}

impl<T: PrimitiveInt> RingBase for StaticRingBase<T> {
    
    type Element = T;

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs += rhs;
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        *lhs = -*lhs;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs *= rhs;
    }

    fn from_z(&self, value: i32) -> Self::Element { T::try_from(value).map_err(|_| ()).unwrap() }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        *lhs == *rhs
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
}

impl<T: PrimitiveInt> KaratsubaHint for StaticRingBase<T> {
    
}

pub type StaticRing<T> = RingValue<StaticRingBase<T>>;

#[test]
fn test_ixx_bit_op() {
    let ring_i16 = StaticRing::<i16>::RING;
    let ring_i128 = StaticRing::<i128>::RING;
    assert_eq!(Some(2), ring_i16.abs_highest_set_bit(&0x5));
    assert_eq!(Some(15), ring_i16.abs_highest_set_bit(&i16::MIN));
    assert_eq!(Some(1), ring_i16.abs_highest_set_bit(&-2));
    assert_eq!(Some(2), ring_i128.abs_highest_set_bit(&0x5));
    assert_eq!(Some(127), ring_i128.abs_highest_set_bit(&i128::MIN));
    assert_eq!(Some(126), ring_i128.abs_highest_set_bit(&(i128::MIN + 1)));
    assert_eq!(Some(126), ring_i128.abs_highest_set_bit(&(-1 - i128::MIN)));
    assert_eq!(Some(1), ring_i128.abs_highest_set_bit(&-2));
    assert_eq!(true, ring_i128.abs_is_bit_set(&-12, 2));
    assert_eq!(false, ring_i128.abs_is_bit_set(&-12, 1));
     assert_eq!(true, ring_i128.abs_is_bit_set(&i128::MIN, 127));
    assert_eq!(false, ring_i128.abs_is_bit_set(&i128::MIN, 126));
}