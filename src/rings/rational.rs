use crate::algorithms::convolution::{DefaultConvolutionRing, DynConvolution, NaiveConvolution, TypeErasableConvolution};
use crate::serialization::*;
use crate::algorithms::matmul::{ComputeInnerProduct, StrassenHint};
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::algorithms::poly_gcd::gcd_locally::poly_gcd_local;
use crate::algorithms::poly_gcd::squarefree_part::poly_power_decomposition_local;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::field::*;
use crate::homomorphism::*;
use crate::algorithms::resultant::ComputeResultantRing;
use crate::integer::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::impl_interpolation_base_ring_char_zero;
use crate::rings::fraction::*;
use crate::rings::poly::PolyRing;
use crate::pid::{EuclideanRing, EuclideanRingStore, PrincipalIdealRing, PrincipalIdealRingStore};
use crate::specialization::*;
use crate::ring::*;

use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use feanor_serde::seq::{DeserializeSeedSeq, SerializableSeq};
use serde::{Serialize, Deserialize, Deserializer, Serializer};
use serde::de::DeserializeSeed;

use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::replace;
use std::ops::Deref;
use std::sync::Arc;

///
/// An implementation of the rational number `Q`, based on representing them
/// as a tuple `(numerator, denominator)`.
/// 
/// Be careful when instantiating it with finite-precision integers, like `StaticRing<i64>`,
/// since by nature of the rational numbers, both numerator and denominator can increase
/// dramatically, even when the numbers itself are of moderate size.
/// 
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::field::FieldStore;
/// let ZZ = StaticRing::<i64>::RING;
/// let QQ = RationalField::new(ZZ);
/// let hom = QQ.can_hom(&ZZ).unwrap();
/// let one_half = QQ.div(&QQ.one(), &hom.map(2));
/// assert_el_eq!(QQ, QQ.div(&QQ.one(), &hom.map(4)), QQ.pow(one_half, 2));
/// ```
/// You can also retrieve numerator and denominator.
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::field::FieldStore;
/// # let ZZ = StaticRing::<i64>::RING;
/// # let QQ = RationalField::new(ZZ);
/// # let hom = QQ.can_hom(&ZZ).unwrap();
/// # let one_half = QQ.div(&QQ.one(), &hom.map(2));
/// assert_el_eq!(ZZ, ZZ.int_hom().map(1), QQ.num(&one_half));
/// assert_el_eq!(ZZ, ZZ.int_hom().map(2), QQ.den(&one_half));
/// ```
/// 
pub struct RationalFieldBase<I: RingStore>
    where I::Type: IntegerRing
{
    integers: I
}

impl<I> Clone for RationalFieldBase<I>
    where I: RingStore + Clone,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        Self {
            integers: self.integers.clone()
        }
    }
}

impl<I> Copy for RationalFieldBase<I>
    where I: RingStore + Copy,
        I::Type: IntegerRing,
        El<I>: Copy
{}

impl<I> Debug for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q")
    }
}
///
/// [`RingStore`] corresponding to [`RationalFieldBase`]
/// 
pub type RationalField<I> = RingValue<RationalFieldBase<I>>;

///
/// An element of [`RationalField`], i.e. a fraction of two integers.
/// 
pub struct RationalFieldEl<I>(El<I>, El<I>)
    where I: RingStore,
        I::Type: IntegerRing;

impl<I> Debug for RationalFieldEl<I>
    where I: RingStore,
        I::Type: IntegerRing,
        El<I>: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RationalFieldEl")
            .field("num", &self.0)
            .field("den", &self.1)
            .finish()
    }
}

impl<I> Clone for RationalFieldEl<I>
    where I: RingStore,
        I::Type: IntegerRing,
        El<I>: Clone
{
    fn clone(&self) -> Self {
        RationalFieldEl(self.0.clone(), self.1.clone())
    }
}

impl<I> Copy for RationalFieldEl<I>
    where I: RingStore,
        I::Type: IntegerRing,
        El<I>: Copy
{}

impl<I> PartialEq for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.integers.get_ring() == other.integers.get_ring()
    }
}

impl<I> RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// The numerator of the fully reduced fraction.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::rings::rational::*;
    /// # use feanor_math::homomorphism::Homomorphism;
    /// # use feanor_math::field::FieldStore;
    /// # let ZZ = StaticRing::<i64>::RING;
    /// # let QQ = RationalField::new(ZZ);
    /// assert_el_eq!(ZZ, 2, QQ.num(&QQ.div(&QQ.inclusion().map(6), &QQ.inclusion().map(3))));
    /// ```
    /// 
    pub fn num<'a>(&'a self, el: &'a <Self as RingBase>::Element) -> &'a El<I> {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        &el.0
    }

    ///
    /// The denominator of the fully reduced fraction.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::rings::rational::*;
    /// # use feanor_math::homomorphism::Homomorphism;
    /// # use feanor_math::field::FieldStore;
    /// # let ZZ = StaticRing::<i64>::RING;
    /// # let QQ = RationalField::new(ZZ);
    /// assert_el_eq!(ZZ, 3, QQ.den(&QQ.div(&QQ.inclusion().map(3), &QQ.inclusion().map(9))));
    /// ```
    /// 
    pub fn den<'a>(&'a self, el: &'a <Self as RingBase>::Element) -> &'a El<I> {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        &el.1
    }

    fn inner_product_impl<J, T1, T2>(&self, els: J) -> RationalFieldEl<I>
        where J: Iterator<Item = (T1, T2)>,
            T1: Deref<Target = RationalFieldEl<I>>,
            T2: Deref<Target = RationalFieldEl<I>>
    {
        let mut current_den = self.integers.one();
        // rationale: If we take an inner product, there is a significant chance that all summands
        // have a denominator that divides an unknown "maximal denominator"; It seems likely that
        // after we have processed some elements, the lcm of all denominators is this maximal denominator,
        // and thus it makes sense to optimize for the case that the next denominator divides the
        // maximal denominator
        let mut current_num = self.integers.zero();
        for (lhs, rhs) in els {
            let new_den = self.integers.mul_ref(&lhs.1, &rhs.1);
            let (quo, rem) = self.integers.euclidean_div_rem(self.integers.clone_el(&current_den), &new_den);
            if self.integers.is_zero(&rem) {
                current_num = self.integers.fma(&lhs.0, &self.integers.mul_ref_fst(&rhs.0, quo), current_num);
            } else {
                // we already did the first euclidean division for finding the gcd, so use rem here
                let gcd = self.integers.ideal_gen(&rem, &new_den);
                let scale_by = self.integers.checked_div(&new_den, &gcd).unwrap();
                let quo = self.integers.checked_div(&current_den, &gcd).unwrap();
                self.integers.mul_assign_ref(&mut current_num, &scale_by);
                self.integers.mul_assign_ref(&mut current_den, &scale_by);
                current_num = self.integers.fma(&lhs.0, &self.integers.mul_ref_fst(&rhs.0, quo), current_num);
            }
        }
        self.reduce((&mut current_num, &mut current_den));
        return RationalFieldEl(current_num, current_den);
    }
}

impl<I> RationalField<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// Returns the fraction field of the given integer ring.
    /// 
    pub const fn new(integers: I) -> Self {
        RingValue::from(RationalFieldBase { integers })
    }

    ///
    /// See [`RationalFieldBase::num()`].
    /// 
    pub fn num<'a>(&'a self, el: &'a El<Self>) -> &'a El<I> {
        self.get_ring().num(el)
    }

    ///
    /// See [`RationalFieldBase::den()`].
    /// 
    pub fn den<'a>(&'a self, el: &'a El<Self>) -> &'a El<I> {
        self.get_ring().den(el)
    }
}

impl<I> RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn reduce(&self, value: (&mut El<I>, &mut El<I>)) {
        let mut gcd = self.integers.ideal_gen(&*value.0, &*value.1);
        if self.integers.is_neg(&gcd) != self.integers.is_neg(&*value.1) {
            self.integers.negate_inplace(&mut gcd);
        }
        *value.0 = self.integers.checked_div(&*value.0, &gcd).unwrap();
        *value.1 = self.integers.checked_div(&*value.1, &gcd).unwrap();
    }

    fn mul_assign_raw(&self, lhs: &mut <Self as RingBase>::Element, rhs: (&El<I>, &El<I>)) {
        self.integers.mul_assign_ref(&mut lhs.0, rhs.0);
        self.integers.mul_assign_ref(&mut lhs.1, rhs.1);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }
}

impl<I> RingBase for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    type Element = RationalFieldEl<I>;

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        if self.integers.is_zero(&rhs.0) {
            // do nothing
        } else if self.integers.is_zero(&lhs.0) {
            *lhs = rhs;
        } else if self.integers.is_one(&lhs.1) && self.integers.is_one(&rhs.1) {
            self.integers.add_assign(&mut lhs.0, rhs.0);
        } else {
            self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
            lhs.0 = self.integers.fma(&rhs.0, &lhs.1, replace(&mut lhs.0, self.integers.zero()));
            self.integers.mul_assign(&mut lhs.1, rhs.1);
            self.reduce((&mut lhs.0, &mut lhs.1));
        }
    }

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        RationalFieldEl(self.integers.clone_el(&val.0), self.integers.clone_el(&val.1))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        if self.integers.is_zero(&rhs.0) {
            // do nothing
        } else if self.integers.is_zero(&lhs.0) {
            *lhs = self.clone_el(rhs);
        } else if self.integers.is_one(&lhs.1) && self.integers.is_one(&rhs.1) {
            self.integers.add_assign_ref(&mut lhs.0, &rhs.0);
        } else {
            self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
            lhs.0 = self.integers.fma(&rhs.0, &lhs.1, replace(&mut lhs.0, self.integers.zero()));
            self.integers.mul_assign_ref(&mut lhs.1, &rhs.1);
            self.reduce((&mut lhs.0, &mut lhs.1));
        }
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.mul_assign_raw(lhs, (&rhs.0, &rhs.1))
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.integers.mul_assign(&mut lhs.0, rhs.0);
        self.integers.mul_assign(&mut lhs.1, rhs.1);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.integers.negate_inplace(&mut lhs.0);
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.integers.eq_el(&self.integers.mul_ref(&lhs.0, &rhs.1), &self.integers.mul_ref(&lhs.1, &rhs.0))
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.integers.is_zero(&value.0)
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        self.integers.eq_el(&value.0, &value.1)
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        self.integers.eq_el(&value.0, &self.integers.negate(self.integers.clone_el(&value.1)))
    }

    fn is_approximate(&self) -> bool {
        false
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn characteristic<J: RingStore + Copy>(&self, ZZ: J) -> Option<El<J>>
        where J::Type: IntegerRing
    {
        Some(ZZ.zero())
    }

    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if self.base_ring().is_one(&value.1) {
            write!(out, "{}", self.integers.formatted_el(&value.0))
        } else {
            if env > EnvBindingStrength::Product {
                write!(out, "({}/{})", self.integers.formatted_el(&value.0), self.integers.formatted_el(&value.1))
            } else {
                write!(out, "{}/{}", self.integers.formatted_el(&value.0), self.integers.formatted_el(&value.1))
            }
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RationalFieldEl(self.integers.get_ring().from_int(value), self.integers.one())
    }
}

impl<I: RingStore> HashableElRing for RationalFieldBase<I>
    where I::Type: IntegerRing + HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        debug_assert!(!self.base_ring().is_neg(&el.1));
        self.integers.get_ring().hash(&el.0, h);
        self.integers.get_ring().hash(&el.1, h);
    }
}

impl<I: RingStore> StrassenHint for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    default fn strassen_threshold(&self) -> usize {
        usize::MAX
    }
}

impl<I: RingStore> DefaultConvolutionRing for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    fn create_default_convolution<'conv>(&self, _max_len_hint: Option<usize>) -> DynConvolution<'conv, Self>
        where Self: 'conv
    {
        Arc::new(TypeErasableConvolution::new(NaiveConvolution))
    }
}

impl<I> RingExtension for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    type BaseRing = I;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.integers
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        RationalFieldEl(x, self.integers.one())
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.integers.mul_assign_ref(&mut lhs.0, rhs);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }
}

impl<I> Serialize for RationalFieldBase<I>
    where I: RingStore + Serialize,
        I::Type: IntegerRing
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("RationalField", self.base_ring()).serialize(serializer)
    }
}

impl<'de, I> Deserialize<'de> for RationalFieldBase<I>
    where I: RingStore + Deserialize<'de>,
        I::Type: IntegerRing
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("RationalField", PhantomData::<I>).deserialize(deserializer).map(|base_ring| RationalFieldBase { integers: base_ring })
    }
}

impl<I> SerializableElementRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing + SerializableElementRing
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("Rational", DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeWithRing::new(self.base_ring())).take(3),
            (None, None),
            |mut current, next| {
                if current.0.is_none() {
                    current.0 = Some(next);
                } else if current.1.is_none() {
                    current.1 = Some(next);
                } else {
                    unreachable!();
                }
                return current;
            }
        )).deserialize(deserializer).map(|res| self.from_fraction(res.0.unwrap(), res.1.unwrap()))
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("Rational", SerializableSeq::new_with_len(
            [SerializeWithRing::new(&el.0, self.base_ring()), SerializeWithRing::new(&el.1, self.base_ring())].iter(), 2
        )).serialize(serializer)
    }
}

impl<I, J> CanHomFrom<RationalFieldBase<J>> for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing,
        J: RingStore,
        J::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &RationalFieldBase<J>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &RationalFieldBase<J>, el: <RationalFieldBase<J> as RingBase>::Element, (): &Self::Homomorphism) -> Self::Element {
        RationalFieldEl(int_cast(el.0, self.base_ring(), from.base_ring()), int_cast(el.1, self.base_ring(), from.base_ring()))
    }
}

impl<I, J> CanIsoFromTo<RationalFieldBase<J>> for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing,
        J: RingStore,
        J::Type: IntegerRing
{
    type Isomorphism = ();

    fn has_canonical_iso(&self, _from: &RationalFieldBase<J>) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, from: &RationalFieldBase<J>, el: Self::Element, (): &Self::Homomorphism) -> <RationalFieldBase<J> as RingBase>::Element {
        RationalFieldEl(int_cast(el.0, from.base_ring(), self.base_ring()), int_cast(el.1, from.base_ring(), self.base_ring()))
    }
}

impl<I, J> CanHomFrom<J> for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing,
        J: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &J) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &J, el: <J as RingBase>::Element, (): &Self::Homomorphism) -> Self::Element {
        RationalFieldEl(int_cast(el, self.base_ring(), &RingRef::new(from)), self.integers.one())
    }
}

impl<I> DivisibilityRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            Some(self.zero())
        } else if self.is_zero(rhs) {
            None
        } else {
            let mut result = self.clone_el(lhs);
            self.mul_assign_raw(&mut result, (&rhs.1, &rhs.0));
            Some(result)
        }
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        !self.is_zero(x)
    }

    fn balance_factor<'a, J>(&self, elements: J) -> Option<Self::Element>
        where J: Iterator<Item = &'a Self::Element>,
            Self:'a
    {
        let (num, den) = elements.fold(
            (self.integers.zero(), self.integers.one()), 
            |x, y| {
                let num_gcd = self.base_ring().ideal_gen(&x.0, self.num(y));
                let den_lcm = self.base_ring().checked_div(&self.base_ring().mul_ref(&x.1, self.den(y)), &self.base_ring().ideal_gen(&x.1, self.den(y))).unwrap();
                (num_gcd, den_lcm)
            });
        return Some(RationalFieldEl(num, den));
    }

    fn prepare_divisor(&self, _: &Self::Element) -> Self::PreparedDivisorData {
        ()
    }
}

impl_interpolation_base_ring_char_zero!{ <{I}> InterpolationBaseRing for RationalFieldBase<I> where I: RingStore, I::Type: IntegerRing + ComputeResultantRing }

impl<I> PrincipalIdealRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return (self.zero(), self.zero(), self.zero());
        } else if self.is_zero(lhs) {
            return (self.zero(), self.one(), self.clone_el(rhs));
        } else {
            return (self.one(), self.zero(), self.clone_el(lhs));
        }
    }
}

impl<I> EuclideanRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) { Some(0) } else { Some(1) }
    }

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }
}

impl<I> Domain for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{}

impl<I> PerfectField for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{}

impl<I> Field for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{}

impl<I> FiniteRingSpecializable for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<I> FractionField for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn as_fraction(&self, el: Self::Element) -> (El<Self::BaseRing>, El<Self::BaseRing>) {
        (el.0, el.1)
    }
}

impl<I> OrderedRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        assert!(self.integers.is_pos(&lhs.1) && self.integers.is_pos(&rhs.1));
        self.integers.cmp(&self.integers.mul_ref(&lhs.0, &rhs.1), &self.integers.mul_ref(&rhs.0, &lhs.1))
    }
}

struct DerefT<T>(T);

impl<T> Deref for DerefT<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<I> ComputeInnerProduct for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn inner_product_ref<'a, J: IntoIterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: J) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.inner_product_impl(els.into_iter())
    }

    fn inner_product_ref_fst<'a, J: IntoIterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: J) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.inner_product_impl(els.into_iter().map(|(lhs, rhs)| (lhs, DerefT(rhs))))
    }

    fn inner_product<J: IntoIterator<Item = (Self::Element, Self::Element)>>(&self, els: J) -> Self::Element {
        self.inner_product_impl(els.into_iter().map(|(lhs, rhs)| (DerefT(lhs), DerefT(rhs))))
    }
}

impl<I> PolyTFracGCDRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            BaseRing<P>: RingStore<Type = Self>
    {
        assert!(!poly_ring.is_zero(poly));
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| ZZ.checked_div(&ZZ.mul_ref(&a, b), &ZZ.ideal_gen(&a, b)).unwrap());
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let power_decomp = poly_power_decomposition_local(&ZZX, f);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return power_decomp.into_iter().map(|(f, k)| (QQX.normalize(ZZX_to_QQX.map(f)), k)).collect();
    }

    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            BaseRing<P>: RingStore<Type = Self>
    {
        if poly_ring.is_zero(lhs) {
            return poly_ring.clone_el(rhs);
        } else if poly_ring.is_zero(rhs) {
            return poly_ring.clone_el(lhs);
        }
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm_lhs = QQX.terms(lhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| ZZ.checked_div(&ZZ.mul_ref(&a, b), &ZZ.ideal_gen(&a, b)).unwrap());
        let den_lcm_rhs = QQX.terms(rhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| ZZ.checked_div(&ZZ.mul_ref(&a, b), &ZZ.ideal_gen(&a, b)).unwrap());
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let lhs = ZZX.from_terms(QQX.terms(lhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_lhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let rhs = ZZX.from_terms(QQX.terms(rhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_rhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let result = poly_gcd_local(&ZZX, lhs, rhs);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return QQX.normalize(ZZX_to_QQX.map(result));
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = El<RationalField<StaticRing<i64>>>> {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    let incl = ring.into_int_hom();
    (-6..8).flat_map(move |x| (-2..5).filter(|y| *y != 0).map(move |y| ring.checked_div(&incl.map(x), &incl.map(y)).unwrap()))
}

#[test]
fn test_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);

    let half = ring.checked_div(&ring.int_hom().map(1), &ring.int_hom().map(2)).unwrap();
    assert!(!ring.is_one(&half));
    assert!(!ring.is_zero(&half));
    assert_el_eq!(ring, ring.one(), ring.add_ref(&half, &half));
    crate::ring::generic_tests::test_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_inner_product() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);

    assert_el_eq!(ring, ring.from_fraction(5, 3), <_ as ComputeInnerProduct>::inner_product(ring.get_ring(), [
        (ring.from_fraction(1, 1), ring.from_fraction(1, 1)),
        (ring.from_fraction(2, 1), ring.from_fraction(1, 3))
    ]));

    assert_el_eq!(ring, ring.from_fraction(11, 6), <_ as ComputeInnerProduct>::inner_product(ring.get_ring(), [
        (ring.from_fraction(1, 3), ring.from_fraction(2, 1)),
        (ring.from_fraction(2, 3), ring.from_fraction(1, 1)),
        (ring.from_fraction(2, 3), ring.from_fraction(1, 2)),
        (ring.from_fraction(1, 3), ring.from_fraction(1, 2))
    ]));
}

#[test]
fn test_divisibility_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::divisibility::generic_tests::test_divisibility_axioms(ring, edge_case_elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::pid::generic_tests::test_euclidean_ring_axioms(ring, edge_case_elements());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_int_hom_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hom_axioms(&StaticRing::<i64>::RING, ring, -16..15);
}

#[test]
fn test_serialization() {
    LogAlgorithmSubscriber::init_test();
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::serialization::generic_tests::test_serialization(ring, edge_case_elements());
}

#[test]
fn test_serialize_deserialize() {
    LogAlgorithmSubscriber::init_test();
    crate::serialization::generic_tests::test_serialize_deserialize(RationalField::new(StaticRing::<i64>::RING).into());
    crate::serialization::generic_tests::test_serialize_deserialize(RationalField::new(BigIntRing::RING).into());
}

#[test]
fn test_serialize_postcard() {
    LogAlgorithmSubscriber::init_test();
    let ring: RingValue<RationalFieldBase<RingValue<crate::primitive_int::StaticRingBase<i64>>>> = RationalField::new(StaticRing::<i64>::RING);
    let serialized = postcard::to_allocvec(&SerializeWithRing::new(&ring.int_hom().map(42), &ring)).unwrap();
    let result = DeserializeWithRing::new(&ring).deserialize(
        &mut postcard::Deserializer::from_flavor(postcard::de_flavors::Slice::new(&serialized))
    ).unwrap();

    assert_el_eq!(&ring, ring.int_hom().map(42), result);
}