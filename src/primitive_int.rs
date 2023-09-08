use std::ops::{AddAssign, SubAssign, MulAssign, Neg, Div, Rem};
use std::marker::PhantomData;
use std::fmt::Display;

use crate::ring::*;
use crate::euclidean::EuclideanRing;
use crate::divisibility::DivisibilityRing;
use crate::ordered::OrderedRing;
use crate::rings::rust_bigint::{RustBigint, RustBigintRingBase};
use crate::integer::*;
use crate::algorithms::conv_mul::KaratsubaHint;

///
/// Trait for `i8` to `i128`.
/// 
pub trait PrimitiveInt: AddAssign + SubAssign + MulAssign + Neg<Output = Self> + Eq + From<i8> + TryFrom<i32> + TryFrom<i128> + Into<i128> + Copy + Div<Self, Output = Self> + Rem<Self, Output = Self> + Display {

    fn bits() -> usize;
}

impl PrimitiveInt for i8 {
    fn bits() -> usize { Self::BITS as usize }
}

impl PrimitiveInt for i16 {
    fn bits() -> usize { Self::BITS as usize }
}

impl PrimitiveInt for i32 {
    fn bits() -> usize { Self::BITS as usize }
}

impl PrimitiveInt for i64 {
    fn bits() -> usize { Self::BITS as usize }
}

impl PrimitiveInt for i128 {
    fn bits() -> usize { Self::BITS as usize }
}

impl<I: ?Sized + IntegerRing, S: PrimitiveInt> CanonicalHom<I> for StaticRingBase<S> {

    type Homomorphism = ();

    default fn has_canonical_hom(&self, _: &I) -> Option<()> {
        Some(())
    }

    default fn map_in(&self, from: &I, el: I::Element, _: &()) -> S {
        generic_impls::generic_map_in(from, self, &el)
    }
}

impl<I: ?Sized + IntegerRing, S: PrimitiveInt> CanonicalIso<I> for StaticRingBase<S> {

    type Isomorphism = ();

    default fn has_canonical_iso(&self, _: &I) -> Option<()> {
        Some(())
    }

    default fn map_out(&self, from: &I, el: S, _: &()) -> I::Element {
        generic_impls::generic_map_in(self, from, &el)
    }
}

macro_rules! specialize_map_from_primitive_int {
    ($($int:ty),*) => {
        $(            
            impl<S: PrimitiveInt> CanonicalHom<StaticRingBase<$int>> for StaticRingBase<S> {

                fn has_canonical_hom(&self, _: &StaticRingBase<$int>) -> Option<()> {
                    Some(())
                }

                fn map_in(&self, _: &StaticRingBase<$int>, el: $int, _: &()) -> S {
                    S::try_from(<_ as Into<i128>>::into(el)).map_err(|_| ()).unwrap()
                }
            }
                        
            impl<S: PrimitiveInt> CanonicalIso<StaticRingBase<$int>> for StaticRingBase<S> {

                fn has_canonical_iso(&self, _: &StaticRingBase<$int>) -> Option<()> {
                    Some(())
                }

                fn map_out(&self, from: &StaticRingBase<$int>, el: S, _: &()) -> $int {
                    from.map_in(self, el, &())
                }
            }

        )*
    };
}

specialize_map_from_primitive_int!{i8, i16, i32, i64, i128}

impl<T: PrimitiveInt> CanonicalHom<RustBigintRingBase> for StaticRingBase<T> {

    fn has_canonical_hom(&self, _: &RustBigintRingBase) -> Option<()> {
        Some(())
    }

    fn map_in(&self, from: &RustBigintRingBase, el: RustBigint, _: &()) -> T {
        from.map_out(self, el, &())
    }
}

impl<T: PrimitiveInt> CanonicalIso<RustBigintRingBase> for StaticRingBase<T> {

    fn has_canonical_iso(&self, _: &RustBigintRingBase) -> Option<()> {
        Some(())
    }

    fn map_out(&self, from: &RustBigintRingBase, el: T, _: &()) -> RustBigint {
        from.map_in(self, el, &())
    }
}

#[cfg(feature = "mpir")]
impl<T: PrimitiveInt> CanonicalHom<crate::rings::mpir::MPZBase> for StaticRingBase<T> {

    fn has_canonical_hom(&self, _: &crate::rings::mpir::MPZBase) -> Option<()> {
        Some(())
    }

    fn map_in(&self, from: &crate::rings::mpir::MPZBase, el: crate::rings::mpir::MPZEl, _: &()) -> T {
        from.map_out(self, el, &())
    }
}

#[cfg(feature = "mpir")]
impl<T: PrimitiveInt> CanonicalIso<crate::rings::mpir::MPZBase> for StaticRingBase<T> {

    fn has_canonical_iso(&self, _: &crate::rings::mpir::MPZBase) -> Option<()> {
        Some(())
    }

    fn map_out(&self, from: &crate::rings::mpir::MPZBase, el: T, _: &()) -> crate::rings::mpir::MPZEl {
        from.map_in(self, el, &())
    }
}

impl<T: PrimitiveInt> DivisibilityRing for StaticRingBase<T> {
    
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
}

impl<T: PrimitiveInt> EuclideanRing for StaticRingBase<T> {
    
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        (lhs / *rhs, lhs % *rhs)
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        RingRef::new(self).cast(&StaticRing::<i128>::RING, *val).checked_abs().and_then(|x| usize::try_from(x).ok())
    }
}

impl<T: PrimitiveInt> OrderedRing for StaticRingBase<T> {
    
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        RingRef::new(self).cast(&StaticRing::<i128>::RING, *lhs).cmp(
            &RingRef::new(self).cast(&StaticRing::<i128>::RING, *rhs)
        )
    }
}

impl<T: PrimitiveInt> IntegerRing for StaticRingBase<T> {

    fn to_float_approx(&self, value: &Self::Element) -> f64 { 
        RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) as f64
    }

    fn from_float_approx(&self, value: f64) -> Option<Self::Element> {
        Some(RingRef::new(self).coerce(&StaticRing::<i128>::RING, value as i128))
    }

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool {
        match RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) {
            i128::MIN => i == i128::BITS as usize - 1,
            x => (x.abs() >> i) & 1 == 1
        }
    }

    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        match RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) {
            0 => None,
            i128::MIN => Some(i128::BITS as usize - 1),
            x => Some(i128::BITS as usize - x.abs().leading_zeros() as usize - 1)
        }
    }

    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        match RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) {
            0 => None,
            i128::MIN => Some(i128::BITS as usize - 1),
            x => Some(x.abs().trailing_zeros() as usize)
        }
    }

    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = RingRef::new(self).coerce(&StaticRing::<i128>::RING, 
            RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) / (1 << power));
    }

    fn mul_pow_2(&self, value: &mut Self::Element, power: usize) {
        *value = RingRef::new(self).coerce(&StaticRing::<i128>::RING, 
            RingRef::new(self).cast(&StaticRing::<i128>::RING, *value) << power);
    }

    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, mut rng: G) -> Self::Element {
        assert!(log2_bound_exclusive <= T::bits() - 1);
        RingRef::new(self).coerce(
            &StaticRing::<i128>::RING, 
            ((((rng() as u128) << u64::BITS as u32) | (rng() as u128)) & ((1 << log2_bound_exclusive) - 1)) as i128
        )
    }
}

impl<T: PrimitiveInt> HashableElRing for StaticRingBase<T> {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        h.write_i128(RingRef::new(self).cast(&StaticRing::<i128>::RING, *el))
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

impl<T> PartialEq for StaticRingBase<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T: PrimitiveInt> RingValue<StaticRingBase<T>> {
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
    
    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", *value)
    }
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

///
/// The ring of integers `Z`, using the arithmetic of the primitive integer type `T`.
/// 
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

#[test]
fn test_get_uniformly_random() {
    generic_test_integer_uniformly_random(StaticRing::<i8>::RING);
    generic_test_integer_uniformly_random(StaticRing::<i16>::RING);
    generic_test_integer_uniformly_random(StaticRing::<i32>::RING);
    generic_test_integer_uniformly_random(StaticRing::<i64>::RING);
    generic_test_integer_uniformly_random(StaticRing::<i128>::RING);
}

#[test]
fn test_integer_axioms() {
    generic_test_integer_axioms(StaticRing::<i8>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    generic_test_integer_axioms(StaticRing::<i16>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    generic_test_integer_axioms(StaticRing::<i32>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    generic_test_integer_axioms(StaticRing::<i64>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
    generic_test_integer_axioms(StaticRing::<i128>::RING, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8].into_iter());
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