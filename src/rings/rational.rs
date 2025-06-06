use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::eea::{signed_gcd, signed_lcm};
use crate::serialization::*;
use crate::algorithms::matmul::StrassenHint;
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::algorithms::poly_gcd::gcd::poly_gcd_local;
use crate::algorithms::poly_gcd::squarefree_part::poly_power_decomposition_local;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::field::*;
use crate::homomorphism::*;
use crate::computation::DontObserve;
use crate::algorithms::resultant::ComputeResultantRing;
use crate::integer::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::impl_interpolation_base_ring_char_zero;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::specialization::*;
use crate::ring::*;
use crate::seq::*;

use serde::{Serialize, Deserialize, Deserializer, Serializer};
use serde::de::DeserializeSeed;

use std::fmt::Debug;
use std::marker::PhantomData;

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
#[derive(Debug)]
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

///
/// [`RingStore`] corresponding to [`RationalFieldBase`]
/// 
pub type RationalField<I> = RingValue<RationalFieldBase<I>>;

pub struct RationalFieldEl<I>(El<I>, El<I>)
    where I: RingStore,
        I::Type: IntegerRing;

impl<I> Debug for RationalFieldEl<I>
    where I: RingStore,
        I::Type: IntegerRing,
        El<I>: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RationalFieldEl({:?}, {:?})", &self.0, &self.1)
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
        debug_assert!(self.base_ring().is_one(&signed_gcd(self.base_ring().clone_el(&el.1), self.base_ring().clone_el(&el.0), self.base_ring())));
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
        debug_assert!(self.base_ring().is_one(&signed_gcd(self.base_ring().clone_el(&el.1), self.base_ring().clone_el(&el.0), self.base_ring())));
        &el.1
    }
}

impl<I> RationalField<I>
    where I: RingStore,
        I::Type: IntegerRing
{
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
        // take the denominator first, as in this case gcd will have the same sign, and the final denominator will be positive
        let gcd = signed_gcd(self.integers.clone_el(&*value.1), self.integers.clone_el(&*value.0), &self.integers);
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

    fn add_assign(&self, lhs: &mut Self::Element, mut rhs: Self::Element) {
        if self.integers.is_one(&lhs.1) && self.integers.is_one(&rhs.1) {
            self.integers.add_assign(&mut lhs.0, rhs.0);
        } else {
            self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
            self.integers.mul_assign_ref(&mut rhs.0, &lhs.1);
            self.integers.mul_assign(&mut lhs.1, rhs.1);
            self.integers.add_assign(&mut lhs.0, rhs.0);
            self.reduce((&mut lhs.0, &mut lhs.1));
        }
    }

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        RationalFieldEl(self.integers.clone_el(&val.0), self.integers.clone_el(&val.1))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        if self.integers.is_one(&lhs.1) && self.integers.is_one(&rhs.1) {
            self.integers.add_assign_ref(&mut lhs.0, &rhs.0);
        } else {
            self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
            self.integers.add_assign(&mut lhs.0, self.integers.mul_ref(&lhs.1, &rhs.0));
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

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if self.base_ring().is_one(&value.1) {
            write!(out, "{}", self.integers.format(&value.0))
        } else {
            if env > EnvBindingStrength::Product {
                write!(out, "({}/{})", self.integers.format(&value.0), self.integers.format(&value.1))
            } else {
                write!(out, "{}/{}", self.integers.format(&value.0), self.integers.format(&value.1))
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
        let gcd = signed_gcd(self.integers.clone_el(&el.1), self.integers.clone_el(&el.0), &self.integers);
        self.integers.get_ring().hash(&self.integers.checked_div(&el.0, &gcd).unwrap(), h);
        self.integers.get_ring().hash(&self.integers.checked_div(&el.1, &gcd).unwrap(), h);
    }
}

impl<I: RingStore> StrassenHint for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    default fn strassen_threshold(&self) -> usize {
        usize::MAX
    }
}

impl<I: RingStore> KaratsubaHint for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    default fn karatsuba_threshold(&self) -> usize {
        usize::MAX
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
        SerializableNewtype::new("RationalField", self.base_ring()).serialize(serializer)
    }
}

impl<'de, I> Deserialize<'de> for RationalFieldBase<I>
    where I: RingStore + Deserialize<'de>,
        I::Type: IntegerRing
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtype::new("RationalField", PhantomData::<I>).deserialize(deserializer).map(|base_ring| RationalFieldBase { integers: base_ring })
    }
}

impl<I> SerializableElementRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing + SerializableElementRing
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtype::new("Rational", DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeWithRing::new(self.base_ring())).take(2),
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
        SerializableNewtype::new("Rational", SerializableSeq::new(
            [SerializeWithRing::new(&el.0, self.base_ring()), SerializeWithRing::new(&el.1, self.base_ring())].as_fn()
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
            |x, y| (signed_gcd(x.0, self.base_ring().clone_el(self.num(y)), self.base_ring()), signed_lcm(x.1, self.base_ring().clone_el(self.den(y)), self.base_ring())));
        return Some(RationalFieldEl(num, den));
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

impl<I> PolyTFracGCDRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        assert!(!poly_ring.is_zero(poly));
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let power_decomp = poly_power_decomposition_local(&ZZX, f, DontObserve);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return power_decomp.into_iter().map(|(f, k)| (QQX.normalize(ZZX_to_QQX.map(f)), k)).collect();
    }
    
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        if poly_ring.is_zero(lhs) {
            return poly_ring.clone_el(rhs);
        } else if poly_ring.is_zero(rhs) {
            return poly_ring.clone_el(lhs);
        }
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm_lhs = QQX.terms(lhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        let den_lcm_rhs = QQX.terms(rhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let lhs = ZZX.from_terms(QQX.terms(lhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_lhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let rhs = ZZX.from_terms(QQX.terms(rhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_rhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let result = poly_gcd_local(&ZZX, lhs, rhs, DontObserve);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return QQX.normalize(ZZX_to_QQX.map(result));
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::homomorphism::Homomorphism;

use super::fraction::FractionField;
use super::poly::PolyRing;

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = El<RationalField<StaticRing<i64>>>> {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    let incl = ring.into_int_hom();
    (-6..8).flat_map(move |x| (-2..5).filter(|y| *y != 0).map(move |y| ring.checked_div(&incl.map(x), &incl.map(y)).unwrap()))
}

#[test]
fn test_ring_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);

    let half = ring.checked_div(&ring.int_hom().map(1), &ring.int_hom().map(2)).unwrap();
    assert!(!ring.is_one(&half));
    assert!(!ring.is_zero(&half));
    assert_el_eq!(ring, ring.one(), ring.add_ref(&half, &half));
    crate::ring::generic_tests::test_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_divisibility_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::divisibility::generic_tests::test_divisibility_axioms(ring, edge_case_elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::pid::generic_tests::test_euclidean_ring_axioms(ring, edge_case_elements());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_int_hom_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hom_axioms(&StaticRing::<i64>::RING, ring, -16..15);
}

#[test]
fn test_serialization() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::serialization::generic_tests::test_serialization(ring, edge_case_elements());
}

#[test]
fn test_serialize_deserialize() {
    crate::serialization::generic_tests::test_serialize_deserialize(RationalField::new(StaticRing::<i64>::RING).into());
    crate::serialization::generic_tests::test_serialize_deserialize(RationalField::new(BigIntRing::RING).into());
}