use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::algorithms::sqr_mul::generic_abs_square_and_multiply;
use crate::prelude::*;
use crate::ring::HashableElRing;
use crate::ring_properties::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::ring_properties::integer::BigIntRing;
use crate::ring_properties::ordered::OrderedRingStore;
use crate::ring_properties::serialization::{DeserializeWithRing, SerializableElementRing, SerializeWithRing};

/// Trait for implementations of generic abelian groups, for which only
/// the group operation, equality testing and computing hash values is supported.
///
/// These groups from the model for which most dlog algorithms have been developed.
/// Note that if your group is actually the additive group of a ring, it is very
/// likely that you can solve dlog much more efficiently by using [`crate::algorithms::linsolve`].
///
/// The design mirrors [`RingBase`] and [`RingStore`], with [`AbelianGroupStore`] being
/// the counterpart to [`RingStore`].
pub trait AbelianGroupBase: PartialEq + Debug + Send + Sync {
    /// Type used to represent elements of this group.
    type Element: Sized + Send + Sync + Clone;

    /// Checks whether two group elements are equal.
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool;

    /// Applies the group operation to two elements.
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element;

    /// Applies the group operation to two elements.
    ///
    /// As opposed to [`AbelianGroupBase::op()`], this takes both arguments by reference.
    fn op_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element { self.op(lhs.clone(), rhs.clone()) }

    /// Applies the group operation to two elements.
    ///
    /// As opposed to [`AbelianGroupBase::op()`], this takes the second argument by reference.
    fn op_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element { self.op(lhs, rhs.clone()) }

    /// Computes the inverse of the give element, i.e. the unique group element `x^-1` such that
    /// `x * x^-1` is the identity element.
    fn inv(&self, x: &Self::Element) -> Self::Element;

    /// Returns the identity element of the group, i.e. the unique element `1` such that
    /// `x * 1 = x` for all group elements `x`.
    fn identity(&self) -> Self::Element;

    /// Hashes the group element.
    ///
    /// This should satisfy all the standard properties usually satisfied by hashing,
    /// in particular it should be compatible with [`AbelianGroupBase::eq_el()`].
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H);

    /// Raises a group element to the given power, i.e. computes `x * x * ... * x`,
    /// in total `power` times. Works also for negative values of `power`.
    fn pow_gen<R>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element
    where
        R: RingStore,
        R::Ring: IntegerRing,
    {
        let res = generic_abs_square_and_multiply(
            x,
            power,
            &integers,
            |a| self.op_ref(&a, &a),
            |a, b| self.op_ref_snd(b, &a),
            self.identity(),
        );
        if !integers.is_neg(power) { res } else { self.inv(&res) }
    }

    /// Checks whether the given element is the identity element of the group.
    ///
    /// Equivalent to `group.eq_el(x, &group.identity())`, but may be faster.
    fn is_identity(&self, x: &Self::Element) -> bool { self.eq_el(x, &self.identity()) }

    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result;
}

/// Alias for the type of elements of a group underlying an `AbelianGroupStore`.
///
/// Analogue of [`El`] for rings.
pub type GEl<G> = <<G as AbelianGroupStore>::Group as AbelianGroupBase>::Element;

/// Analogue of [`crate::delegate!`] for groups.
#[macro_export]
macro_rules! delegate_group {
    ($base_trait:ty, fn $name:ident (&self, $($pname:ident: $ptype:ty),*) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self, $($pname: $ptype),*) -> $rtype {
            <Self::Group as $base_trait>::$name(self.get_group(), $($pname),*)
        }
    };
    ($base_trait:ty, fn $name:ident (&self) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self) -> $rtype {
            <Self::Group as $base_trait>::$name(self.get_group())
        }
    };
}

/// Object provides access to a generic abelian group, as modelled by [`AbelianGroupBase`].
///
/// The design of [`AbelianGroupBase`] and [`AbelianGroupStore`] mirrors
/// the design of [`RingBase`] and [`RingStore`]. See there for details.
pub trait AbelianGroupStore: Send + Sync {
    type Group: AbelianGroupBase;

    fn get_group(&self) -> &Self::Group;

    delegate_group! { AbelianGroupBase, fn eq_el(&self, lhs: &GEl<Self>, rhs: &GEl<Self>) -> bool }
    delegate_group! { AbelianGroupBase, fn op(&self, lhs: GEl<Self>, rhs: GEl<Self>) -> GEl<Self> }
    delegate_group! { AbelianGroupBase, fn op_ref(&self, lhs: &GEl<Self>, rhs: &GEl<Self>) -> GEl<Self> }
    delegate_group! { AbelianGroupBase, fn op_ref_snd(&self, lhs: GEl<Self>, rhs: &GEl<Self>) -> GEl<Self> }
    delegate_group! { AbelianGroupBase, fn inv(&self, x: &GEl<Self>) -> GEl<Self> }
    delegate_group! { AbelianGroupBase, fn identity(&self) -> GEl<Self> }
    delegate_group! { AbelianGroupBase, fn is_identity(&self, x: &GEl<Self>) -> bool }

    fn hash<H: Hasher>(&self, x: &GEl<Self>, hasher: &mut H) { self.get_group().hash(x, hasher) }

    /// Raises the given element to the given power.
    ///
    /// See also [`RingBase::pow_gen()`] and [`RingStore::pow_gen()`].
    fn pow(&self, x: GEl<Self>, power: i64) -> GEl<Self> { self.pow_gen(x, &power, ZZi64) }

    /// Raises the given element to the given power.
    ///
    /// See also [`RingBase::pow_gen()`] and [`RingStore::pow_gen()`].
    fn pow_bigint(&self, x: GEl<Self>, power: &El<BigIntRing>) -> GEl<Self> { self.pow_gen(x, power, ZZbig) }

    /// Raises the given element to the given power, which should be a positive integer
    /// belonging to an arbitrary [`IntegerRing`].
    ///
    /// See also [`RingBase::pow_gen()`].
    fn pow_gen<R: RingStore>(&self, x: GEl<Self>, power: &El<R>, integers: R) -> GEl<Self>
    where
        R::Ring: IntegerRing,
    {
        self.get_group().pow_gen(x, power, integers)
    }

    fn formatted_el<'a>(&'a self, x: &'a GEl<Self>) -> GroupElementDisplayWrapper<'a, Self::Group> {
        GroupElementDisplayWrapper {
            group: self.get_group(),
            element: x,
        }
    }
}

impl<G> AbelianGroupStore for G
where
    G: Deref + Send + Sync,
    G::Target: AbelianGroupStore,
{
    type Group = <G::Target as AbelianGroupStore>::Group;

    fn get_group(&self) -> &Self::Group { (**self).get_group() }
}

/// Variant of `assert_eq!` for group elements; analogue of [`assert_el_eq!`] for groups.
#[macro_export]
macro_rules! assert_gel_eq {
    ($group:expr, $lhs:expr, $rhs:expr) => {
        match (&$group, &$lhs, &$rhs) {
            (group_val, lhs_val, rhs_val) => {
                assert!(
                    <_ as $crate::group::AbelianGroupStore>::eq_el(group_val, lhs_val, rhs_val),
                    "Assertion failed: {} != {}",
                    <_ as $crate::group::AbelianGroupStore>::formatted_el(group_val, lhs_val),
                    <_ as $crate::group::AbelianGroupStore>::formatted_el(group_val, rhs_val)
                );
            }
        }
    };
}

/// Like [`assert_gel_eq!`], but only active when debug assertions are enabled.
#[macro_export]
macro_rules! debug_assert_gel_eq {
    ($group:expr, $lhs:expr, $rhs:expr) => {
        #[cfg(debug_assertions)]
        {
            assert_gel_eq!($group, $lhs, $rhs)
        }
    };
}

/// Analogue of [`RingValue`] for groups.
#[repr(transparent)]
#[derive(Serialize, Deserialize)]
pub struct GroupValue<G: AbelianGroupBase> {
    group: G,
}

impl<G: AbelianGroupBase> From<G> for GroupValue<G> {
    fn from(value: G) -> Self { Self { group: value } }
}

impl<G: AbelianGroupBase + Sized> GroupValue<G> {
    pub fn into(self) -> G { self.group }

    pub fn from_ref<'a>(group: &'a G) -> &'a Self { unsafe { std::mem::transmute(group) } }
}

impl<G: AbelianGroupBase> AbelianGroupStore for GroupValue<G> {
    type Group = G;

    fn get_group(&self) -> &Self::Group { &self.group }
}

impl<G: AbelianGroupBase + Clone> Clone for GroupValue<G> {
    fn clone(&self) -> Self {
        Self {
            group: self.group.clone(),
        }
    }
}

impl<G: AbelianGroupBase + Copy> Copy for GroupValue<G> {}

/// The additive group of a ring, implements [`AbelianGroupBase`].
///
/// # Attention
///
/// It is unlikely that you want to use this, except for testing
/// group-related algorithms.
///
/// In most cases, it does not make much sense to compute dlogs in the additive
/// group of a ring using generic methods, since algorithms as in
/// [`crate::algorithms::linsolve`] will be much faster.
pub struct AddGroupBase<R: RingStore>(pub R);

/// [`AbelianGroupStore`] corresponding to [`AddGroupBase`].
pub type AddGroup<R> = GroupValue<AddGroupBase<R>>;

/// The multiplicative group of a ring, implements [`AbelianGroupBase`].
#[derive(Serialize, Deserialize)]
pub struct MultGroupBase<R: RingStore>(R);

/// [`AbelianGroupStore`] corresponding to [`MultGroupBase`].
pub type MultGroup<R> = GroupValue<MultGroupBase<R>>;

/// Elements from the multiplicative group of `R`.
pub struct MultGroupEl<R: RingStore>(El<R>);

impl<R> PartialEq for AddGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing,
{
    fn eq(&self, other: &Self) -> bool { self.0.get_ring() == other.0.get_ring() }
}

impl<R> Debug for AddGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self.0.get_ring()) }
}

impl<R> AbelianGroupBase for AddGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing,
{
    type Element = El<R>;

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(lhs, rhs) }
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element { self.0.add(lhs, rhs) }
    fn inv(&self, x: &Self::Element) -> Self::Element { self.0.negate(x.clone()) }
    fn identity(&self) -> Self::Element { self.0.zero() }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(x, hasher) }
    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.0.get_ring().fmt_el(value, out)
    }
}

impl<R> AddGroup<R>
where
    R: RingStore,
    R::Ring: HashableElRing,
{
    pub fn new(ring: R) -> Self { Self::from(AddGroupBase(ring)) }
}

impl<R> Clone for AddGroupBase<R>
where
    R: RingStore + Clone,
    R::Ring: HashableElRing,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<R> Copy for AddGroupBase<R>
where
    R: RingStore + Clone + Copy,
    R::Ring: HashableElRing,
    El<R>: Copy,
{
}

impl<R> PartialEq for MultGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
{
    fn eq(&self, other: &Self) -> bool { self.0.get_ring() == other.0.get_ring() }
}

impl<R> Debug for MultGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "({:?})*", self.0.get_ring()) }
}

impl<R> AbelianGroupBase for MultGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
{
    type Element = MultGroupEl<R>;

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(&lhs.0, &rhs.0) }
    fn inv(&self, x: &Self::Element) -> Self::Element { MultGroupEl(self.0.invert(&x.0).unwrap()) }
    fn identity(&self) -> Self::Element { MultGroupEl(self.0.one()) }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(&x.0, hasher) }
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element { MultGroupEl(self.0.mul(lhs.0, rhs.0)) }
    fn fmt_el<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.0.get_ring().fmt_el(&value.0, out)
    }
}

impl<R> SerializableElementGroup for MultGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing + SerializableElementRing,
{
    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableNewtypeStruct::new("MultGroupEl", SerializeWithRing::new(&el.0, &self.0)).serialize(serializer)
    }

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
    where
        D: Deserializer<'de>,
    {
        DeserializeSeedNewtypeStruct::new("MultGroupEl", DeserializeWithRing::new(&self.0))
            .deserialize(deserializer)
            .map(|x| MultGroupEl(x))
    }
}

impl<R> Clone for MultGroupBase<R>
where
    R: RingStore + Clone,
    R::Ring: HashableElRing + DivisibilityRing,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<R> Copy for MultGroupBase<R>
where
    R: RingStore + Copy,
    R::Ring: HashableElRing + DivisibilityRing,
{
}

impl<R> Clone for MultGroupEl<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
    El<R>: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<R> Debug for MultGroupEl<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
    El<R>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self.0) }
}

impl<R> MultGroupBase<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
{
    pub fn new(ring: R) -> Self { return Self(ring); }

    pub fn underlying_ring(&self) -> &R { &self.0 }

    /// If `x` is contained in `R*`, returns a [`MultGroupEl`] representing
    /// `x`. Otherwise, `None` is returned.
    pub fn from_ring_el(&self, x: El<R>) -> Option<MultGroupEl<R>> {
        if self.0.is_unit(&x) { Some(MultGroupEl(x)) } else { None }
    }

    /// Returns the ring element represented by the given group element.
    pub fn as_ring_el<'a>(&self, x: &'a MultGroupEl<R>) -> &'a El<R> { &x.0 }
}

impl<R> MultGroup<R>
where
    R: RingStore,
    R::Ring: HashableElRing + DivisibilityRing,
{
    pub fn new(ring: R) -> Self { Self::from(MultGroupBase::new(ring)) }

    pub fn underlying_ring(&self) -> &R { self.get_group().underlying_ring() }

    /// If `x` is contained in `R*`, returns a [`MultGroupEl`] representing
    /// `x`. Otherwise, `None` is returned.
    pub fn from_ring_el(&self, x: El<R>) -> Option<MultGroupEl<R>> { self.get_group().from_ring_el(x) }

    /// Returns the ring element represented by the given group element.
    pub fn as_ring_el<'a>(&self, x: &'a MultGroupEl<R>) -> &'a El<R> { self.get_group().as_ring_el(x) }
}

pub struct HashableGroupEl<G: AbelianGroupStore> {
    group: G,
    el: GEl<G>,
}

impl<G: AbelianGroupStore> HashableGroupEl<G> {
    pub fn new(group: G, el: GEl<G>) -> Self { Self { group, el } }
}

impl<G: AbelianGroupStore> PartialEq for HashableGroupEl<G> {
    fn eq(&self, other: &Self) -> bool { self.group.eq_el(&self.el, &other.el) }
}

impl<G: AbelianGroupStore> Eq for HashableGroupEl<G> {}

impl<G: AbelianGroupStore> Hash for HashableGroupEl<G> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.group.hash(&self.el, state) }
}

/// Trait for rings whose elements can be serialized.
///
/// Serialization and deserialization mostly follow the principles of the `serde` crate, with
/// the main difference that ring elements cannot be serialized/deserialized on their own, but
/// only w.r.t. a specific ring.
pub trait SerializableElementGroup: AbelianGroupBase {
    /// Deserializes an element of this ring from the given deserializer.
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
    where
        D: Deserializer<'de>;

    /// Serializes an element of this ring to the given serializer.
    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer;
}

/// Wrapper of a group that implements [`serde::DeserializationSeed`] by trying to deserialize
/// an element w.r.t. the wrapped group.
#[derive(Clone)]
pub struct DeserializeWithGroup<G: AbelianGroupStore>
where
    G::Group: SerializableElementGroup,
{
    group: G,
}

impl<G> DeserializeWithGroup<G>
where
    G: AbelianGroupStore,
    G::Group: SerializableElementGroup,
{
    pub fn new(group: G) -> Self { Self { group } }
}

impl<'de, G> DeserializeSeed<'de> for DeserializeWithGroup<G>
where
    G: AbelianGroupStore,
    G::Group: SerializableElementGroup,
{
    type Value = GEl<G>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.group.get_group().deserialize(deserializer)
    }
}

/// Wraps a group and a reference to one of its elements. Implements [`serde::Serialize`]
/// and will serialize the element w.r.t. the group.
pub struct SerializeWithGroup<'a, G: AbelianGroupStore>
where
    G::Group: SerializableElementGroup,
{
    group: G,
    el: &'a GEl<G>,
}

impl<'a, G: AbelianGroupStore> SerializeWithGroup<'a, G>
where
    G::Group: SerializableElementGroup,
{
    pub fn new(el: &'a GEl<G>, group: G) -> Self { Self { el, group } }
}

impl<'a, G: AbelianGroupStore> Serialize for SerializeWithGroup<'a, G>
where
    G::Group: SerializableElementGroup,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.group.get_group().serialize(self.el, serializer)
    }
}

/// Wraps a ring and a one of its elements. Implements [`serde::Serialize`] and
/// will serialize the element w.r.t. the ring.
pub struct SerializeOwnedWithGroup<G: AbelianGroupStore>
where
    G::Group: SerializableElementGroup,
{
    group: G,
    el: GEl<G>,
}

impl<G: AbelianGroupStore> SerializeOwnedWithGroup<G>
where
    G::Group: SerializableElementGroup,
{
    pub fn new(el: GEl<G>, group: G) -> Self { Self { el, group } }
}

impl<G: AbelianGroupStore> Serialize for SerializeOwnedWithGroup<G>
where
    G::Group: SerializableElementGroup,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.group.get_group().serialize(&self.el, serializer)
    }
}

pub struct GroupElementDisplayWrapper<'a, G: AbelianGroupBase + ?Sized> {
    group: &'a G,
    element: &'a G::Element,
}

impl<'a, G: AbelianGroupBase + ?Sized> std::fmt::Display for GroupElementDisplayWrapper<'a, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.group.fmt_el(self.element, f) }
}

impl<'a, G: AbelianGroupBase + ?Sized> std::fmt::Debug for GroupElementDisplayWrapper<'a, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.group.fmt_el(self.element, f) }
}
