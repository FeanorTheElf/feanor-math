use std::any::TypeId;
use std::ops::{AddAssign, Div, MulAssign, Neg, Rem, Shr, SubAssign};
use std::marker::PhantomData;
use std::fmt::{Debug, Display};

use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer}; 
use serde::de::{DeserializeOwned, DeserializeSeed};

use crate::{impl_interpolation_base_ring_char_zero, impl_poly_gcd_locally_for_ZZ, impl_eval_poly_locally_for_ZZ};
use crate::ring::*;
use crate::algorithms;
use crate::homomorphism::*;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::divisibility::*;
use crate::ordered::*;
use crate::integer::*;
use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::matmul::StrassenHint;
use crate::specialization::*;
use crate::serialization::SerializableElementRing;

///
/// Trait for `i8` to `i128`.
/// 
pub trait PrimitiveInt: 'static + Send + Sync + Serialize + DeserializeOwned + AddAssign + SubAssign + MulAssign + Neg<Output = Self> + Shr<usize, Output = Self> + Eq + Into<Self::Larger> + TryFrom<Self::Larger> + From<i8> + TryFrom<i32> + TryFrom<i128> + Into<i128> + Copy + Div<Self, Output = Self> + Rem<Self, Output = Self> + Display {

    ///
    /// The primitive integer that is "twice as large" as this one.
    /// The only exception is `i128`, for which there is no larger primitive integer.
    /// 
    type Larger: PrimitiveInt;

    ///
    /// Returns the number of bits of this integer type.
    /// 
    fn bits() -> usize;

    ///
    /// The functions [`i8::overflowing_mul()`] to [`i128::overflowing_mul()`].
    /// 
    fn overflowing_mul(self, rhs: Self) -> Self;
    
    ///
    /// The functions [`i8::overflowing_sub()`] to [`i128::overflowing_sub()`].
    /// 
    fn overflowing_sub(self, rhs: Self) -> Self;
}

impl PrimitiveInt for i8 {

    type Larger = i16;

    fn bits() -> usize { Self::BITS as usize }
    fn overflowing_mul(self, rhs: Self) -> Self { i8::overflowing_mul(self, rhs).0 }
    fn overflowing_sub(self, rhs: Self) -> Self { i8::overflowing_sub(self, rhs).0 }
}

impl PrimitiveInt for i16 {

    type Larger = i32;

    fn bits() -> usize { Self::BITS as usize }
    fn overflowing_mul(self, rhs: Self) -> Self { i16::overflowing_mul(self, rhs).0 }
    fn overflowing_sub(self, rhs: Self) -> Self { i16::overflowing_sub(self, rhs).0 }
}

impl PrimitiveInt for i32 {

    type Larger = i64;

    fn bits() -> usize { Self::BITS as usize }
    fn overflowing_mul(self, rhs: Self) -> Self { i32::overflowing_mul(self, rhs).0 }
    fn overflowing_sub(self, rhs: Self) -> Self { i32::overflowing_sub(self, rhs).0 }
}

impl PrimitiveInt for i64 {

    type Larger = i128;

    fn bits() -> usize { Self::BITS as usize }
    fn overflowing_mul(self, rhs: Self) -> Self { i64::overflowing_mul(self, rhs).0 }
    fn overflowing_sub(self, rhs: Self) -> Self { i64::overflowing_sub(self, rhs).0 }
}

impl PrimitiveInt for i128 {

    type Larger = i128;

    fn bits() -> usize { Self::BITS as usize }
    fn overflowing_mul(self, rhs: Self) -> Self { i128::overflowing_mul(self, rhs).0 }
    fn overflowing_sub(self, rhs: Self) -> Self { i128::overflowing_sub(self, rhs).0 }
}

macro_rules! specialize_int_cast {
    ($(($int_from:ty, $int_to:ty)),*) => {
        $(
            impl IntCast<StaticRingBase<$int_from>> for StaticRingBase<$int_to> {
                
                fn cast(&self, _: &StaticRingBase<$int_from>, value: $int_from) -> Self::Element {
                    <$int_to>::try_from(<_ as Into<i128>>::into(value)).map_err(|_| ()).unwrap()
                }
            }
        )*
    };
}

specialize_int_cast!{
    (i8, i8), (i8, i16), (i8, i32), (i8, i64), (i8, i128),
    (i16, i8), (i16, i16), (i16, i32), (i16, i64), (i16, i128),
    (i32, i8), (i32, i16), (i32, i32), (i32, i64), (i32, i128),
    (i64, i8), (i64, i16), (i64, i32), (i64, i64), (i64, i128),
    (i128, i8), (i128, i16), (i128, i32), (i128, i64), (i128, i128)
}

impl<T: PrimitiveInt> DivisibilityRing for StaticRingBase<T> {
    
    type PreparedDivisorData = PrimitiveIntPreparedDivisorData<T>;

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.zero());
        } else if self.is_zero(rhs) {
            return None;
        }
        let (div, rem) = self.euclidean_div_rem(*lhs, rhs);
        if self.is_zero(&rem) {
            return Some(div);
        } else {
            return None;
        }
    }
    
    fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        Some(elements.fold(self.zero(), |a, b| self.ideal_gen(&a, b)))
    }

    fn prepare_divisor(&self, x: &Self::Element) -> Self::PreparedDivisorData {
        // currently prepared division is not implemented for i128, as using Barett-reduction here
        // requires 256-bit arithmetic, and I saw no need to make that effort
        if TypeId::of::<T>() == TypeId::of::<i128>() {
            return PrimitiveIntPreparedDivisorData(T::from(0));
        }
        return match <T as Into<i128>>::into(*x) {
            0 => PrimitiveIntPreparedDivisorData(T::from(0)),
            1 => PrimitiveIntPreparedDivisorData(T::try_from((1i128 << (T::bits() - 1)) - 1).ok().unwrap()),
            -1 => PrimitiveIntPreparedDivisorData(T::try_from((-1i128 << (T::bits() - 1)) + 1).ok().unwrap()),
            val => PrimitiveIntPreparedDivisorData(<T as TryFrom<i128>>::try_from((1i128 << (T::bits() - 1)) / val).ok().unwrap())
        };
    }
    
    fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> Option<Self::Element> {
        // currently prepared division is not implemented for i128, as using Barett-reduction here
        // requires 256-bit arithmetic, and I saw no need to make that effort
        if TypeId::of::<T>() == TypeId::of::<i128>() {
            return self.checked_left_div(lhs, &rhs);
        }
        if *rhs == T::from(0) {
            if *lhs == T::from(0) { Some(T::from(0)) } else { None }
        } else {
            let mut prod = <T as Into<T::Larger>>::into(*lhs);
            prod *=  <T as Into<T::Larger>>::into(rhs_prep.0);
            let mut result = <T as TryFrom<T::Larger>>::try_from(prod >> (T::bits() - 1)).ok().unwrap();
            let remainder = T::overflowing_sub(*lhs, T::overflowing_mul(result, *rhs));
            if remainder == T::from(0) {
                Some(result)
            } else if remainder == *rhs {
                result += T::from(1);
                Some(result)
            } else if -remainder == *rhs {
                result -= T::from(1);
                Some(result)
            } else {
                None
            }
        }
    }
}

///
/// Data associated to an element of [`StaticRing`] that allows for faster division. 
/// 
/// See also [`DivisibilityRing::prepare_divisor()`].
/// 
#[derive(Clone, Copy, Debug)]
pub struct PrimitiveIntPreparedDivisorData<T: PrimitiveInt>(T);

impl<T: PrimitiveInt> Domain for StaticRingBase<T> {}

impl<T: PrimitiveInt> PrincipalIdealRing for StaticRingBase<T> {
    
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        algorithms::eea::eea(*lhs, *rhs, StaticRing::<T>::RING)
    }
}

impl<T: PrimitiveInt> EuclideanRing for StaticRingBase<T> {
    
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        (lhs / *rhs, lhs % *rhs)
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*val).checked_abs().and_then(|x| usize::try_from(x).ok())
    }
}

impl<T: PrimitiveInt> OrderedRing for StaticRingBase<T> {
    
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*lhs).cmp(
            &RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*rhs)
        )
    }
}

impl_interpolation_base_ring_char_zero!{ <{T}> InterpolationBaseRing for StaticRingBase<T> where T: PrimitiveInt }

impl_poly_gcd_locally_for_ZZ!{ <{T}> IntegerPolyGCDRing for StaticRingBase<T> where T: PrimitiveInt }

impl_eval_poly_locally_for_ZZ!{ <{T}> EvalPolyLocallyRing for StaticRingBase<T> where T: PrimitiveInt }

impl<T> FiniteRingSpecializable for StaticRingBase<T>
    where T: PrimitiveInt
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<T: PrimitiveInt> IntegerRing for StaticRingBase<T> {

    fn to_float_approx(&self, value: &Self::Element) -> f64 { 
        RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) as f64
    }

    fn from_float_approx(&self, value: f64) -> Option<Self::Element> {
        Some(RingRef::new(self).coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, value as i128))
    }

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool {
        match RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) {
            i128::MIN => i == i128::BITS as usize - 1,
            x => (x.abs() >> i) & 1 == 1
        }
    }

    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        match RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) {
            0 => None,
            i128::MIN => Some(i128::BITS as usize - 1),
            x => Some(i128::BITS as usize - x.abs().leading_zeros() as usize - 1)
        }
    }

    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        match RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) {
            0 => None,
            i128::MIN => Some(i128::BITS as usize - 1),
            x => Some(x.abs().trailing_zeros() as usize)
        }
    }

    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = RingRef::new(self).coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, 
            RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) / (1 << power));
    }

    fn mul_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = RingRef::new(self).coerce::<StaticRing<i128>>(&StaticRing::<i128>::RING, 
            RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*value) << power);
    }

    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, mut rng: G) -> Self::Element {
        assert!(log2_bound_exclusive <= T::bits() - 1);
        RingRef::new(self).coerce::<StaticRing<i128>>(
            &StaticRing::<i128>::RING, 
            ((((rng() as u128) << u64::BITS) | (rng() as u128)) & ((1 << log2_bound_exclusive) - 1)) as i128
        )
    }

    fn representable_bits(&self) -> Option<usize> {
        Some(T::bits() - 1)
    }
}

impl<T: PrimitiveInt> HashableElRing for StaticRingBase<T> {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        h.write_i128(RingRef::new(self).can_iso::<StaticRing<i128>>(&StaticRing::<i128>::RING).unwrap().map(*el))
    }
}

///
/// The ring of integers `Z`, using the arithmetic of the primitive integer type `T`.
/// 
/// For the difference to [`StaticRing`], see the documentation of [`crate::ring::RingStore`].
/// 
pub struct StaticRingBase<T> {
    element: PhantomData<T>
}

impl<T> Debug for StaticRingBase<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z")
    }
}

impl<T> PartialEq for StaticRingBase<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T: PrimitiveInt> RingValue<StaticRingBase<T>> {

    ///
    /// The singleton ring instance of [`StaticRing`].
    /// 
    pub const RING: StaticRing<T> = RingValue::from(StaticRingBase { element: PhantomData });
}

impl<T> Copy for StaticRingBase<T> {}

impl<T> Clone for StaticRingBase<T> {

    fn clone(&self) -> Self {
        *self
    }    
}

impl<T: PrimitiveInt> RingBase for StaticRingBase<T> {
    
    type Element = T;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs += rhs;
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        *lhs = -*lhs;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs *= rhs;
    }

    fn from_int(&self, value: i32) -> Self::Element { T::try_from(value).map_err(|_| ()).unwrap() }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        *lhs == *rhs
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _: EnvBindingStrength) -> std::fmt::Result {
        write!(out, "{}", *value)
    }
    
    fn characteristic<I: RingStore>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(ZZ.zero())
    }

    fn pow_gen<R: RingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        assert!(!integers.is_neg(power));
        algorithms::sqr_mul::generic_abs_square_and_multiply(
            x, 
            power, 
            &integers,
            |mut a| {
                self.square(&mut a);
                a
            },
            |a, b| self.mul_ref_fst(a, b),
            self.one()
        )
    }
    
    fn is_approximate(&self) -> bool { false }
}

impl KaratsubaHint for StaticRingBase<i8> {
    fn karatsuba_threshold(&self) -> usize { 4 }
}

impl KaratsubaHint for StaticRingBase<i16> {
    fn karatsuba_threshold(&self) -> usize { 4 }
}

impl KaratsubaHint for StaticRingBase<i32> {
    fn karatsuba_threshold(&self) -> usize { 4 }
}

impl KaratsubaHint for StaticRingBase<i64> {
    fn karatsuba_threshold(&self) -> usize { 4 }
}

impl KaratsubaHint for StaticRingBase<i128> {
    fn karatsuba_threshold(&self) -> usize { 3 }
}

impl StrassenHint for StaticRingBase<i8> {
    fn strassen_threshold(&self) -> usize { 6 }
}

impl StrassenHint for StaticRingBase<i16> {
    fn strassen_threshold(&self) -> usize { 6 }
}

impl StrassenHint for StaticRingBase<i32> {
    fn strassen_threshold(&self) -> usize { 6 }
}

impl StrassenHint for StaticRingBase<i64> {
    fn strassen_threshold(&self) -> usize { 6 }
}

impl StrassenHint for StaticRingBase<i128> {
    fn strassen_threshold(&self) -> usize { 5 }
}

impl<T: PrimitiveInt> SerializableElementRing for StaticRingBase<T> {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        T::deserialize(deserializer)
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        T::serialize(el, serializer)
    }
}

impl<T: PrimitiveInt> Serialize for StaticRingBase<T> {

    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("IntegerRing(primitive int)", ()).serialize(serializer)
    }
}

impl<'de, T: PrimitiveInt> Deserialize<'de> for StaticRingBase<T> {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("IntegerRing(primitive int)", PhantomData::<()>).deserialize(deserializer).map(|()| StaticRing::<T>::RING.into())
    }
}

///
/// The ring of integers `Z`, using the arithmetic of the primitive integer type `T`.
/// 
pub type StaticRing<T> = RingValue<StaticRingBase<T>>;

impl<T: PrimitiveInt> Default for StaticRingBase<T> {
    fn default() -> Self {
        StaticRing::RING.into()
    }
}

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

#[test]
fn test_get_uniformly_random() {
    crate::integer::generic_tests::test_integer_get_uniformly_random(StaticRing::<i8>::RING);
    crate::integer::generic_tests::test_integer_get_uniformly_random(StaticRing::<i16>::RING);
    crate::integer::generic_tests::test_integer_get_uniformly_random(StaticRing::<i32>::RING);
    crate::integer::generic_tests::test_integer_get_uniformly_random(StaticRing::<i64>::RING);
    crate::integer::generic_tests::test_integer_get_uniformly_random(StaticRing::<i128>::RING);
}

#[test]
fn test_integer_axioms() {
    crate::integer::generic_tests::test_integer_axioms(StaticRing::<i8>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::integer::generic_tests::test_integer_axioms(StaticRing::<i16>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::integer::generic_tests::test_integer_axioms(StaticRing::<i32>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::integer::generic_tests::test_integer_axioms(StaticRing::<i64>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::integer::generic_tests::test_integer_axioms(StaticRing::<i128>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
}

#[test]
fn test_euclidean_ring_axioms() {
    crate::pid::generic_tests::test_euclidean_ring_axioms(StaticRing::<i8>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_euclidean_ring_axioms(StaticRing::<i16>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_euclidean_ring_axioms(StaticRing::<i32>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_euclidean_ring_axioms(StaticRing::<i64>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_euclidean_ring_axioms(StaticRing::<i128>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
}

#[test]
fn test_principal_ideal_ring_ring_axioms() {
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(StaticRing::<i8>::RING, [-2, -1, 0, 1, 2].into_iter());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(StaticRing::<i16>::RING, [-2, -1, 0, 1, 2, 3, 4].into_iter());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(StaticRing::<i32>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(StaticRing::<i64>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(StaticRing::<i128>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());

}

#[test]
fn test_lowest_set_bit() {
    assert_eq!(None, StaticRing::<i32>::RING.abs_lowest_set_bit(&0));
    assert_eq!(Some(0), StaticRing::<i32>::RING.abs_lowest_set_bit(&3));
    assert_eq!(Some(0), StaticRing::<i32>::RING.abs_lowest_set_bit(&-3));
    assert_eq!(None, StaticRing::<i128>::RING.abs_lowest_set_bit(&0));
    assert_eq!(Some(127), StaticRing::<i128>::RING.abs_lowest_set_bit(&i128::MIN));
    assert_eq!(Some(0), StaticRing::<i128>::RING.abs_lowest_set_bit(&i128::MAX));
}

#[test]
fn test_prepared_div() {
    type PrimInt = i8;
    for x in PrimInt::MIN..PrimInt::MAX {
        let div_x = PreparedDivisor::new(StaticRing::<PrimInt>::RING.get_ring(), x);
        for y in PrimInt::MIN..PrimInt::MAX {
            if x == 0 {
                if y == 0 {
                    assert!(div_x.checked_left_div_by(&y, StaticRing::<PrimInt>::RING.get_ring()).is_some());
                } else {
                    assert!(div_x.checked_left_div_by(&y, StaticRing::<PrimInt>::RING.get_ring()).is_none());
                }
            } else if y == PrimInt::MIN && x == -1 {
                // this cannot be evaluated without overflow
            } else if y % x == 0 {
                assert_eq!(y / x, div_x.checked_left_div_by(&y, StaticRing::<PrimInt>::RING.get_ring()).unwrap());
            } else {
                assert!(div_x.checked_left_div_by(&y, StaticRing::<PrimInt>::RING.get_ring()).is_none());
            }
        }
    }
}

#[test]
fn test_serialization() {
    crate::serialization::generic_tests::test_serialize_deserialize(StaticRing::<i8>::RING.into());
    crate::serialization::generic_tests::test_serialize_deserialize(StaticRing::<i16>::RING.into());
    crate::serialization::generic_tests::test_serialize_deserialize(StaticRing::<i32>::RING.into());
    crate::serialization::generic_tests::test_serialize_deserialize(StaticRing::<i64>::RING.into());
    crate::serialization::generic_tests::test_serialize_deserialize(StaticRing::<i128>::RING.into());
}
