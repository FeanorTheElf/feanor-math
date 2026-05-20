use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::replace;
use std::ops::Deref;
use std::sync::Arc;

use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use feanor_serde::seq::{DeserializeSeedSeq, SerializableSeq};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::algorithms::convolution::{
    DefaultConvolutionRing, DynConvolution, SchoolbookConvolution, TypeErasedConvolution,
};
use crate::algorithms::matmul::{ComputeInnerProduct, StrassenHint};
use crate::algorithms::poly_factor::FactorPolyField;
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::algorithms::resultant::ComputeResultantRing;
use crate::homomorphism::*;
use crate::impl_interpolation_base_ring_char_zero;
use crate::prelude::*;
use crate::ring_impls::fraction::*;
use crate::ring_impls::poly::PolyRing;
use crate::ring_properties::field::PerfectField;
use crate::ring_properties::serialization::*;
use crate::ring_properties::specialization::*;

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
/// let ZZ = ZZi64;
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
/// # let ZZ = ZZi64;
/// # let QQ = RationalField::new(ZZ);
/// # let hom = QQ.can_hom(&ZZ).unwrap();
/// # let one_half = QQ.div(&QQ.one(), &hom.map(2));
/// assert_el_eq!(ZZ, ZZ.int_hom().map(1), QQ.num(&one_half));
/// assert_el_eq!(ZZ, ZZ.int_hom().map(2), QQ.den(&one_half));
/// ```
pub struct RationalFieldBase<I: RingStore>
where
    I::Ring: IntegerRing,
{
    integers: I,
}

impl<I> Clone for RationalFieldBase<I>
where
    I: RingStore + Clone,
    I::Ring: IntegerRing,
{
    fn clone(&self) -> Self {
        Self {
            integers: self.integers.clone(),
        }
    }
}

impl<I> Copy for RationalFieldBase<I>
where
    I: RingStore + Copy,
    I::Ring: IntegerRing,
    El<I>: Copy,
{
}

impl<I> Debug for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Q") }
}
/// [`RingStore`] corresponding to [`RationalFieldBase`]
pub type RationalField<I> = RingValue<RationalFieldBase<I>>;

/// An element of [`RationalField`], i.e. a fraction of two integers.
pub struct RationalFieldEl<I>(El<I>, El<I>)
where
    I: RingStore,
    I::Ring: IntegerRing;

impl<I> Debug for RationalFieldEl<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    El<I>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RationalFieldEl")
            .field("num", &self.0)
            .field("den", &self.1)
            .finish()
    }
}

impl<I> Clone for RationalFieldEl<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    El<I>: Clone,
{
    fn clone(&self) -> Self { RationalFieldEl(self.0.clone(), self.1.clone()) }
}

impl<I> Copy for RationalFieldEl<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    El<I>: Copy,
{
}

impl<I> PartialEq for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn eq(&self, other: &Self) -> bool { self.integers.get_ring() == other.integers.get_ring() }
}

impl<I> RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
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
    /// # let ZZ = ZZi64;
    /// # let QQ = RationalField::new(ZZ);
    /// assert_el_eq!(
    ///     ZZ,
    ///     2,
    ///     QQ.num(&QQ.div(&QQ.inclusion().map(6), &QQ.inclusion().map(3)))
    /// );
    /// ```
    pub fn num<'a>(&'a self, el: &'a <Self as RingBase>::Element) -> &'a El<I> {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        &el.0
    }

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
    /// # let ZZ = ZZi64;
    /// # let QQ = RationalField::new(ZZ);
    /// assert_el_eq!(
    ///     ZZ,
    ///     3,
    ///     QQ.den(&QQ.div(&QQ.inclusion().map(3), &QQ.inclusion().map(9)))
    /// );
    /// ```
    pub fn den<'a>(&'a self, el: &'a <Self as RingBase>::Element) -> &'a El<I> {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        &el.1
    }

    fn inner_product_impl<J, T1, T2>(&self, els: J) -> RationalFieldEl<I>
    where
        J: Iterator<Item = (T1, T2)>,
        T1: Deref<Target = RationalFieldEl<I>>,
        T2: Deref<Target = RationalFieldEl<I>>,
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
            let (quo, rem) = self.integers.euclidean_div_rem(current_den.clone(), &new_den);
            if self.integers.is_zero(&rem) {
                current_num = self
                    .integers
                    .fma(&lhs.0, &self.integers.mul_ref_fst(&rhs.0, quo), current_num);
            } else {
                // we already did the first euclidean division for finding the gcd, so use rem here
                let gcd = self.integers.ideal_gen(&rem, &new_den);
                let scale_by = self.integers.checked_div(&new_den, &gcd).unwrap();
                let quo = self.integers.checked_div(&current_den, &gcd).unwrap();
                self.integers.mul_assign_ref(&mut current_num, &scale_by);
                self.integers.mul_assign_ref(&mut current_den, &scale_by);
                current_num = self
                    .integers
                    .fma(&lhs.0, &self.integers.mul_ref_fst(&rhs.0, quo), current_num);
            }
        }
        self.reduce((&mut current_num, &mut current_den));
        return RationalFieldEl(current_num, current_den);
    }
}

impl<I> RationalField<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    /// Returns the fraction field of the given integer ring.
    pub const fn new(integers: I) -> Self { RingValue::from(RationalFieldBase { integers }) }

    /// See [`RationalFieldBase::num()`].
    pub fn num<'a>(&'a self, el: &'a El<Self>) -> &'a El<I> { self.get_ring().num(el) }

    /// See [`RationalFieldBase::den()`].
    pub fn den<'a>(&'a self, el: &'a El<Self>) -> &'a El<I> { self.get_ring().den(el) }
}

impl<I> RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
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
where
    I: RingStore,
    I::Ring: IntegerRing,
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
            lhs.0 = self
                .integers
                .fma(&rhs.0, &lhs.1, replace(&mut lhs.0, self.integers.zero()));
            self.integers.mul_assign(&mut lhs.1, rhs.1);
            self.reduce((&mut lhs.0, &mut lhs.1));
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        if self.integers.is_zero(&rhs.0) {
            // do nothing
        } else if self.integers.is_zero(&lhs.0) {
            *lhs = rhs.clone();
        } else if self.integers.is_one(&lhs.1) && self.integers.is_one(&rhs.1) {
            self.integers.add_assign_ref(&mut lhs.0, &rhs.0);
        } else {
            self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
            lhs.0 = self
                .integers
                .fma(&rhs.0, &lhs.1, replace(&mut lhs.0, self.integers.zero()));
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

    fn negate_inplace(&self, lhs: &mut Self::Element) { self.integers.negate_inplace(&mut lhs.0); }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.integers.eq_el(
            &self.integers.mul_ref(&lhs.0, &rhs.1),
            &self.integers.mul_ref(&lhs.1, &rhs.0),
        )
    }

    fn is_zero(&self, value: &Self::Element) -> bool { self.integers.is_zero(&value.0) }

    fn is_one(&self, value: &Self::Element) -> bool { self.integers.eq_el(&value.0, &value.1) }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        self.integers.eq_el(&value.0, &self.integers.negate(value.1.clone()))
    }

    fn is_approximate(&self) -> bool { false }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn characteristic<J: RingStore + Copy>(&self, ZZ: J) -> Option<El<J>>
    where
        J::Ring: IntegerRing,
    {
        Some(ZZ.zero())
    }

    fn fmt_el_within<'a>(
        &self,
        value: &Self::Element,
        out: &mut std::fmt::Formatter<'a>,
        env: EnvBindingStrength,
    ) -> std::fmt::Result {
        if self.base_ring().is_one(&value.1) {
            write!(out, "{}", self.integers.formatted_el(&value.0))
        } else {
            if env > EnvBindingStrength::Product {
                write!(
                    out,
                    "({}/{})",
                    self.integers.formatted_el(&value.0),
                    self.integers.formatted_el(&value.1)
                )
            } else {
                write!(
                    out,
                    "{}/{}",
                    self.integers.formatted_el(&value.0),
                    self.integers.formatted_el(&value.1)
                )
            }
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RationalFieldEl(self.integers.get_ring().from_int(value), self.integers.one())
    }
}

impl<I: RingStore> HashableElRing for RationalFieldBase<I>
where
    I::Ring: IntegerRing + HashableElRing,
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        debug_assert!(self.base_ring().is_unit(&self.base_ring().ideal_gen(&el.0, &el.1)));
        debug_assert!(!self.base_ring().is_neg(&el.1));
        self.integers.get_ring().hash(&el.0, h);
        self.integers.get_ring().hash(&el.1, h);
    }
}

impl<I: RingStore> StrassenHint for RationalFieldBase<I>
where
    I::Ring: IntegerRing,
{
    default fn strassen_threshold(&self) -> usize { usize::MAX }
}

impl<I: RingStore> DefaultConvolutionRing for RationalFieldBase<I>
where
    I::Ring: IntegerRing,
{
    default fn create_default_convolution<'conv, S>(_self_: S, _max_len: Option<usize>) -> DynConvolution<'conv, Self>
    where
        S: RingStore<Ring = Self> + 'conv,
        Self: 'conv,
    {
        Arc::new(TypeErasedConvolution::new(SchoolbookConvolution))
    }
}

impl<I> RingExtension for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    type BaseRing = I;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing { &self.integers }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element { RationalFieldEl(x, self.integers.one()) }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.integers.mul_assign_ref(&mut lhs.0, rhs);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }
}

impl<I> Serialize for RationalFieldBase<I>
where
    I: RingStore + Serialize,
    I::Ring: IntegerRing,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableNewtypeStruct::new("RationalField", self.base_ring()).serialize(serializer)
    }
}

impl<'de, I> Deserialize<'de> for RationalFieldBase<I>
where
    I: RingStore + Deserialize<'de>,
    I::Ring: IntegerRing,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        DeserializeSeedNewtypeStruct::new("RationalField", PhantomData::<I>)
            .deserialize(deserializer)
            .map(|base_ring| RationalFieldBase { integers: base_ring })
    }
}

impl<I> SerializableElementRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing + SerializableElementRing,
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
    where
        D: Deserializer<'de>,
    {
        DeserializeSeedNewtypeStruct::new(
            "Rational",
            DeserializeSeedSeq::new(
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
                },
            ),
        )
        .deserialize(deserializer)
        .map(|res| self.from_fraction(res.0.unwrap(), res.1.unwrap()))
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableNewtypeStruct::new(
            "Rational",
            SerializableSeq::new_with_len(
                [
                    SerializeWithRing::new(&el.0, self.base_ring()),
                    SerializeWithRing::new(&el.1, self.base_ring()),
                ]
                .iter(),
                2,
            ),
        )
        .serialize(serializer)
    }
}

impl<I, J> CanHomFrom<RationalFieldBase<J>> for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    J: RingStore,
    J::Ring: IntegerRing,
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &RationalFieldBase<J>) -> Option<Self::Homomorphism> { Some(()) }

    fn map_in(
        &self,
        from: &RationalFieldBase<J>,
        el: <RationalFieldBase<J> as RingBase>::Element,
        (): &Self::Homomorphism,
    ) -> Self::Element {
        RationalFieldEl(
            int_cast(el.0, self.base_ring(), from.base_ring()),
            int_cast(el.1, self.base_ring(), from.base_ring()),
        )
    }
}

impl<I, J> CanIsoFromTo<RationalFieldBase<J>> for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    J: RingStore,
    J::Ring: IntegerRing,
{
    type Isomorphism = ();

    fn has_canonical_iso(&self, _from: &RationalFieldBase<J>) -> Option<Self::Isomorphism> { Some(()) }

    fn map_out(
        &self,
        from: &RationalFieldBase<J>,
        el: Self::Element,
        (): &Self::Homomorphism,
    ) -> <RationalFieldBase<J> as RingBase>::Element {
        RationalFieldEl(
            int_cast(el.0, from.base_ring(), self.base_ring()),
            int_cast(el.1, from.base_ring(), self.base_ring()),
        )
    }
}

impl<I, J> CanHomFrom<J> for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
    J: IntegerRing,
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &J) -> Option<Self::Homomorphism> { Some(()) }

    fn map_in(&self, from: &J, el: <J as RingBase>::Element, (): &Self::Homomorphism) -> Self::Element {
        RationalFieldEl(
            int_cast(el, self.base_ring(), &RingRef::from(from)),
            self.integers.one(),
        )
    }
}

impl<I> DivisibilityRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            Some(self.zero())
        } else if self.is_zero(rhs) {
            None
        } else {
            let mut result = lhs.clone();
            self.mul_assign_raw(&mut result, (&rhs.1, &rhs.0));
            Some(result)
        }
    }

    fn is_unit(&self, x: &Self::Element) -> bool { !self.is_zero(x) }

    fn balance_factor<'a, J>(&self, elements: J) -> Option<Self::Element>
    where
        J: Iterator<Item = &'a Self::Element>,
        Self: 'a,
    {
        let (num, den) = elements.fold((self.integers.zero(), self.integers.one()), |x, y| {
            let num_gcd = self.base_ring().ideal_gen(&x.0, self.num(y));
            let den_lcm = self
                .base_ring()
                .checked_div(
                    &self.base_ring().mul_ref(&x.1, self.den(y)),
                    &self.base_ring().ideal_gen(&x.1, self.den(y)),
                )
                .unwrap();
            (num_gcd, den_lcm)
        });
        return Some(RationalFieldEl(num, den));
    }

    fn prepare_divisor(&self, _: &Self::Element) -> Self::PreparedDivisorData { () }
}

impl_interpolation_base_ring_char_zero! { <{I}> InterpolationBaseRing for RationalFieldBase<I> where I: RingStore, I::Ring: IntegerRing + ComputeResultantRing }

impl<I> PrincipalIdealRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }

    fn extended_ideal_gen(
        &self,
        lhs: &Self::Element,
        rhs: &Self::Element,
    ) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return (self.zero(), self.zero(), self.zero());
        } else if self.is_zero(lhs) {
            return (self.zero(), self.one(), rhs.clone());
        } else {
            return (self.one(), self.zero(), lhs.clone());
        }
    }
}

impl<I> EuclideanRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> { if self.is_zero(val) { Some(0) } else { Some(1) } }

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }
}

impl<I> Domain for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<I> PerfectField for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<I> Field for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<I> FiniteRingSpecializable for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output { op.fallback() }
}

impl<I> FractionField for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn as_fraction(&self, el: Self::Element) -> (El<Self::BaseRing>, El<Self::BaseRing>) { (el.0, el.1) }
}

impl<I> OrderedRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        assert!(self.integers.is_pos(&lhs.1) && self.integers.is_pos(&rhs.1));
        self.integers.cmp(
            &self.integers.mul_ref(&lhs.0, &rhs.1),
            &self.integers.mul_ref(&rhs.0, &lhs.1),
        )
    }
}

struct DerefT<T>(T);

impl<T> Deref for DerefT<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<I> ComputeInnerProduct for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn inner_product_ref<'a, J: IntoIterator<Item = (&'a Self::Element, &'a Self::Element)>>(
        &self,
        els: J,
    ) -> Self::Element
    where
        Self::Element: 'a,
        Self: 'a,
    {
        self.inner_product_impl(els.into_iter())
    }

    fn inner_product_ref_fst<'a, J: IntoIterator<Item = (&'a Self::Element, Self::Element)>>(
        &self,
        els: J,
    ) -> Self::Element
    where
        Self::Element: 'a,
        Self: 'a,
    {
        self.inner_product_impl(els.into_iter().map(|(lhs, rhs)| (lhs, DerefT(rhs))))
    }

    fn inner_product<J: IntoIterator<Item = (Self::Element, Self::Element)>>(&self, els: J) -> Self::Element {
        self.inner_product_impl(els.into_iter().map(|(lhs, rhs)| (DerefT(lhs), DerefT(rhs))))
    }
}

impl<I> PolyTFracGCDRing for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn power_decomposition<P>(_poly_ring: P, _poly: &El<P>) -> Vec<(El<P>, usize)>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn gcd<P>(_poly_ring: P, _lhs: &El<P>, _rhs: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn is_squarefree<P>(_poly_ring: P, _poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn squarefree_part<P>(_poly_ring: P, _poly: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }
}

impl<I> FactorPolyField for RationalFieldBase<I>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn factor_poly<P>(_poly_ring: P, _poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn is_irred<P>(_poly_ring: P, _poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }
}

#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::ring_impls::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::ring_impls::poly::*;
#[cfg(test)]
use crate::ring_impls::primitive_int::StaticRing;

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = El<RationalField<StaticRing<i64>>>> {
    let ring = RationalField::new(ZZi64);
    let incl = ring.into_int_hom();
    (-6..8).flat_map(move |x| {
        (-2..5)
            .filter(|y| *y != 0)
            .map(move |y| ring.checked_div(&incl.map(x), &incl.map(y)).unwrap())
    })
}

#[test]
fn test_ring_axioms() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);

    let half = ring
        .checked_div(&ring.int_hom().map(1), &ring.int_hom().map(2))
        .unwrap();
    assert!(!ring.is_one(&half));
    assert!(!ring.is_zero(&half));
    assert_el_eq!(ring, ring.one(), ring.add_ref(&half, &half));
    crate::ring::generic_tests::test_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_inner_product() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);

    assert_el_eq!(
        ring,
        ring.from_fraction(5, 3),
        <_ as ComputeInnerProduct>::inner_product(
            ring.get_ring(),
            [
                (ring.from_fraction(1, 1), ring.from_fraction(1, 1)),
                (ring.from_fraction(2, 1), ring.from_fraction(1, 3))
            ]
        )
    );

    assert_el_eq!(
        ring,
        ring.from_fraction(11, 6),
        <_ as ComputeInnerProduct>::inner_product(
            ring.get_ring(),
            [
                (ring.from_fraction(1, 3), ring.from_fraction(2, 1)),
                (ring.from_fraction(2, 3), ring.from_fraction(1, 1)),
                (ring.from_fraction(2, 3), ring.from_fraction(1, 2)),
                (ring.from_fraction(1, 3), ring.from_fraction(1, 2))
            ]
        )
    );
}

#[test]
fn test_divisibility_axioms() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);
    crate::ring_properties::divisibility::generic_tests::test_divisibility_axioms(ring, edge_case_elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);
    crate::ring_properties::pid::generic_tests::test_euclidean_ring_axioms(ring, edge_case_elements());
    crate::ring_properties::pid::generic_tests::test_principal_ideal_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_int_hom_axioms() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);
    crate::ring::generic_tests::test_hom_axioms(&ZZi64, ring, -16..15);
}

#[test]
fn test_serialization() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = RationalField::new(ZZi64);
    crate::ring_properties::serialization::generic_tests::test_serialization(ring, edge_case_elements());
}

#[test]
fn test_serialize_deserialize() {
    feanor_tracing::DelayedLogger::init_test();
    crate::ring_properties::serialization::generic_tests::test_serialize_deserialize(RationalField::new(ZZi64).into());
    crate::ring_properties::serialization::generic_tests::test_serialize_deserialize(RationalField::new(ZZbig).into());
}

#[test]
fn test_serialize_postcard() {
    feanor_tracing::DelayedLogger::init_test();
    let ring: RingValue<RationalFieldBase<RingValue<crate::ring_impls::primitive_int::StaticRingBase<i64>>>> =
        RationalField::new(ZZi64);
    let serialized = postcard::to_allocvec(&SerializeWithRing::new(&ring.int_hom().map(42), &ring)).unwrap();
    let result = DeserializeWithRing::new(&ring)
        .deserialize(&mut postcard::Deserializer::from_flavor(
            postcard::de_flavors::Slice::new(&serialized),
        ))
        .unwrap();

    assert_el_eq!(&ring, ring.int_hom().map(42), result);
}

#[test]
fn test_factor_rational_poly() {
    feanor_tracing::DelayedLogger::init_test();
    let QQ = RationalField::new(ZZbig);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f = poly_ring.from_terms([(incl.map(2), 0), (incl.map(1), 3)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(
        &poly_ring,
        &poly_ring.prod([f.clone(), f.clone(), g.clone()].into_iter()),
    );
    assert_eq!(2, actual.len());
    assert_el_eq!(poly_ring, f, actual[0].0);
    assert_eq!(2, actual[0].1);
    assert_el_eq!(poly_ring, g, actual[1].0);
    assert_eq!(1, actual[1].1);
    assert_el_eq!(QQ, QQ.one(), unit);

    let f = poly_ring.from_terms([(incl.map(3), 0), (incl.map(1), 1)]);
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
    assert_eq!(1, actual.len());
    assert_eq!(1, actual[0].1);
    assert_el_eq!(&poly_ring, f, &actual[0].0);
    assert_el_eq!(QQ, QQ.one(), unit);

    let [mut f] = poly_ring.with_wrapped_indeterminate(|X| {
        [16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4)]
    });
    poly_ring
        .inclusion()
        .mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
    assert_eq!(1, actual.len());
    assert_eq!(2, actual[0].1);
    assert_el_eq!(QQ, QQ.one(), unit);
}

#[test]
fn test_factor_nonmonic_poly() {
    feanor_tracing::DelayedLogger::init_test();
    let QQ = RationalField::new(ZZbig);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f = poly_ring.from_terms([(QQ.div(&incl.map(3), &incl.map(5)), 0), (incl.map(1), 4)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(
        &poly_ring,
        &poly_ring.prod([f.clone(), f.clone(), g.clone(), poly_ring.int_hom().map(100)].into_iter()),
    );
    assert_eq!(2, actual.len());

    assert_el_eq!(poly_ring, g, actual[0].0);
    assert_eq!(1, actual[0].1);
    assert_el_eq!(poly_ring, f, actual[1].0);
    assert_eq!(2, actual[1].1);
    assert_el_eq!(QQ, incl.map(100), unit);
}
