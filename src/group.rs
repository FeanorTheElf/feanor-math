use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::algorithms::sqr_mul::generic_abs_square_and_multiply;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::integer::BigIntRing;
use crate::ring::*;
use crate::ordered::OrderedRingStore;

///
/// Trait for implementations of generic abelian groups, for which only
/// the group operation, equality testing and computing hash values is supported.
/// 
/// These groups from the model for which most dlog algorithms have been developed.
/// Note that if your group is actually the additive group of a ring, it is very
/// likely that you can solve dlog much more efficiently by using [`crate::algorithms::linsolve`].
/// 
/// The design mirrors [`RingBase`] and [`RingStore`], with [`AbelianGroupStore`] being
/// the counterpart to [`RingStore`].
/// 
/// 
#[stability::unstable(feature = "enable")]
pub trait AbelianGroupBase: PartialEq {

    ///
    /// Type used to represent elements of this group.
    /// 
    type Element;

    ///
    /// Clones an element of the group.
    /// 
    fn clone_el(&self, x: &Self::Element) -> Self::Element;

    ///
    /// Checks whether two group elements are equal.
    /// 
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool;

    ///
    /// Applies the group operation to two elements.
    /// 
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element;

    ///
    /// Applies the group operation to two elements.
    /// 
    /// As opposed to [`AbelianGroupBase::op()`], this takes both arguments by reference.
    /// 
    fn op_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.op(self.clone_el(lhs), self.clone_el(rhs))
    }

    ///
    /// Applies the group operation to two elements.
    /// 
    /// As opposed to [`AbelianGroupBase::op()`], this takes the second argument by reference.
    /// 
    fn op_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.op(lhs, self.clone_el(rhs))
    }

    ///
    /// Computes the inverse of the give element, i.e. the unique group element `x^-1` such that
    /// `x * x^-1` is the identity element.
    /// 
    fn inv(&self, x: &Self::Element) -> Self::Element;

    ///
    /// Returns the identity element of the group, i.e. the unique element `1` such that
    /// `x * 1 = x` for all group elements `x`.
    /// 
    fn identity(&self) -> Self::Element;

    ///
    /// Hashes the group element.
    /// 
    /// This should satisfy all the standard properties usually satisfied by hashing,
    /// in particular it should be compatible with [`AbelianGroupBase::eq_el()`].
    /// 
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H);

    ///
    /// Raises a group element to the given power, i.e. computes `x * x * ... * x`,
    /// in total `e` times.
    /// 
    fn pow(&self, x: &Self::Element, e: &El<BigIntRing>) -> Self::Element {
        let res = generic_abs_square_and_multiply(
            self.clone_el(x), 
            e, 
            BigIntRing::RING, 
            |a| self.op_ref(&a, &a), 
            |a, b| self.op_ref_snd(b, &a), 
            self.identity()
        );
        if !BigIntRing::RING.is_neg(e) { res } else { self.inv(&res) }
    }

    ///
    /// Checks whether the given element is the identity element of the group.
    /// 
    /// Equivalent to `group.eq_el(x, &group.identity())`, but may be faster.
    /// 
    fn is_identity(&self, x: &Self::Element) -> bool {
        self.eq_el(x, &self.identity())
    }
}

///
/// Alias for the type of elements of a group underlying an `AbelianGroupStore`.
/// 
#[stability::unstable(feature = "enable")]
#[allow(type_alias_bounds)]
pub type GroupEl<G: AbelianGroupStore> = <G::Type as AbelianGroupBase>::Element;

///
/// Analogue of [`crate::delegate!`] for groups.
/// 
#[macro_export]
macro_rules! delegate_group {
    ($base_trait:ty, fn $name:ident (&self, $($pname:ident: $ptype:ty),*) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self, $($pname: $ptype),*) -> $rtype {
            <Self::Type as $base_trait>::$name(self.get_group(), $($pname),*)
        }
    };
    ($base_trait:ty, fn $name:ident (&self) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self) -> $rtype {
            <Self::Type as $base_trait>::$name(self.get_group())
        }
    };
}

///
/// Object provides access to a generic abelian group, as modelled by [`AbelianGroupBase`].
/// 
/// The design of [`AbelianGroupBase`] and [`AbelianGroupStore`] mirrors
/// the design of [`RingBase`] and [`RingStore`]. See there for details.
/// 
#[stability::unstable(feature = "enable")]
pub trait AbelianGroupStore {
    type Type: AbelianGroupBase;

    fn get_group(&self) -> &Self::Type;

    delegate_group!{ AbelianGroupBase, fn clone_el(&self, el: &GroupEl<Self>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn eq_el(&self, lhs: &GroupEl<Self>, rhs: &GroupEl<Self>) -> bool }
    delegate_group!{ AbelianGroupBase, fn op(&self, lhs: GroupEl<Self>, rhs: GroupEl<Self>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn op_ref(&self, lhs: &GroupEl<Self>, rhs: &GroupEl<Self>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn op_ref_snd(&self, lhs: GroupEl<Self>, rhs: &GroupEl<Self>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn inv(&self, x: &GroupEl<Self>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn identity(&self) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn pow(&self, x: &GroupEl<Self>, e: &El<BigIntRing>) -> GroupEl<Self> }
    delegate_group!{ AbelianGroupBase, fn is_identity(&self, x: &GroupEl<Self>) -> bool }

    fn hash<H: Hasher>(&self, x: &GroupEl<Self>, hasher: &mut H) {
        self.get_group().hash(x, hasher)
    }
}

impl<G> AbelianGroupStore for G
    where G: Deref,
        G::Target: AbelianGroupStore
{
    type Type = <G::Target as AbelianGroupStore>::Type;

    fn get_group(&self) ->  &Self::Type {
        (**self).get_group()
    }
}

///
/// Analogue of [`RingValue`] for groups.
/// 
#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct GroupValue<G: AbelianGroupBase> {
    group: G
}

impl<G: AbelianGroupBase> From<G> for GroupValue<G> {
    
    fn from(value: G) -> Self {
        Self { group: value }
    }
}

impl<G: AbelianGroupBase + Sized> GroupValue<G> {
    
    #[stability::unstable(feature = "enable")]
    pub fn from_ref<'a>(group: &'a G) -> &'a Self {
        unsafe { std::mem::transmute(group) }
    }
}

impl<G: AbelianGroupBase> AbelianGroupStore for GroupValue<G> {
    type Type = G;

    fn get_group(&self) ->  &Self::Type {
        &self.group
    }
}

impl<G: AbelianGroupBase + Clone> Clone for GroupValue<G> {
    fn clone(&self) -> Self {
        Self { group: self.group.clone() }
    }
}

impl<G: AbelianGroupBase + Copy> Copy for GroupValue<G> {}

///
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
/// 
#[stability::unstable(feature = "enable")]
pub struct AddGroupBase<R: RingStore>(pub R);

///
/// [`AbelianGroupStore`] corresponding to [`AddGroupBase`].
/// 
#[stability::unstable(feature = "enable")]
#[allow(type_alias_bounds)]
pub type AddGroup<R: RingStore> = GroupValue<AddGroupBase<R>>;

///
/// The multiplicative group of a ring, implements [`AbelianGroupBase`].
/// 
#[stability::unstable(feature = "enable")]
pub struct MultGroupBase<R: RingStore>(R);

///
/// [`AbelianGroupStore`] corresponding to [`MultGroupBase`].
/// 
#[stability::unstable(feature = "enable")]
#[allow(type_alias_bounds)]
pub type MultGroup<R: RingStore> = GroupValue<MultGroupBase<R>>;

///
/// Elements from the multiplicative group of `R`.
/// 
#[stability::unstable(feature = "enable")]
pub struct MultGroupEl<R: RingStore>(El<R>);

impl<R: RingStore> PartialEq for AddGroupBase<R>
    where R::Type: HashableElRing
{
    fn eq(&self, other: &Self) -> bool {
        self.0.get_ring() == other.0.get_ring()
    }
}

impl<R: RingStore> AbelianGroupBase for AddGroupBase<R>
    where R::Type: HashableElRing
{
    type Element = El<R>;

    fn clone_el(&self, x: &Self::Element) -> Self::Element { self.0.clone_el(x) }
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(lhs, rhs) }
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element { self.0.add(lhs, rhs)}
    fn inv(&self, x: &Self::Element) -> Self::Element { self.0.negate(self.0.clone_el(x)) }
    fn identity(&self) -> Self::Element { self.0.zero() }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(x, hasher) }
}

impl<R: RingStore> AddGroup<R>
    where R::Type: HashableElRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        Self::from(AddGroupBase(ring))
    }
}

impl<R: RingStore> PartialEq for MultGroupBase<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    fn eq(&self, other: &Self) -> bool {
        self.0.get_ring() == other.0.get_ring()
    }
}

impl<R: RingStore> AbelianGroupBase for MultGroupBase<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    type Element = MultGroupEl<R>;

    fn clone_el(&self, x: &Self::Element) -> Self::Element { MultGroupEl(self.0.clone_el(&x.0)) }
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { self.0.eq_el(&lhs.0, &rhs.0) }
    fn inv(&self, x: &Self::Element) -> Self::Element { MultGroupEl(self.0.invert(&x.0).unwrap()) }
    fn identity(&self) -> Self::Element { MultGroupEl(self.0.one()) }
    fn hash<H: Hasher>(&self, x: &Self::Element, hasher: &mut H) { self.0.hash(&x.0, hasher) }
    fn op(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element { MultGroupEl(self.0.mul(lhs.0, rhs.0)) }
}

impl<R: RingStore> Clone for MultGroupBase<R> 
    where R: Clone, R::Type: HashableElRing + DivisibilityRing
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: RingStore> Copy for MultGroupBase<R> 
    where R: Copy, R::Type: HashableElRing + DivisibilityRing
{}

impl<R: RingStore> Debug for MultGroupBase<R> 
    where R::Type: Debug + HashableElRing + DivisibilityRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?})*", self.0.get_ring())
    }
}

impl<R: RingStore> Clone for MultGroupEl<R> 
    where R::Type: HashableElRing + DivisibilityRing,
        El<R>: Clone
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: RingStore> Debug for MultGroupEl<R> 
    where R::Type: HashableElRing + DivisibilityRing,
        El<R>: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<R: RingStore> MultGroupBase<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        assert!(ring.is_commutative());
        return Self(ring);
    }

    #[stability::unstable(feature = "enable")]
    pub fn underlying_ring(&self) -> &R {
        &self.0
    }

    ///
    /// If `x` is contained in `R*`, returns a [`MultGroupEl`] representing
    /// `x`. Otherwise, `None` is returned.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn from_ring_el(&self, x: El<R>) -> Option<MultGroupEl<R>> {
        if self.0.is_unit(&x) {
            Some(MultGroupEl(x))
        } else {
            None
        }
    }

    ///
    /// Returns the ring element represented by the given group element.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn as_ring_el<'a>(&self, x: &'a MultGroupEl<R>) -> &'a El<R> {
        &x.0
    }
}

impl<R: RingStore> MultGroup<R> 
    where R::Type: HashableElRing + DivisibilityRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        Self::from(MultGroupBase::new(ring))
    }

    #[stability::unstable(feature = "enable")]
    pub fn underlying_ring(&self) -> &R {
        self.get_group().underlying_ring()
    }

    ///
    /// If `x` is contained in `R*`, returns a [`MultGroupEl`] representing
    /// `x`. Otherwise, `None` is returned.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn from_ring_el(&self, x: El<R>) -> Option<MultGroupEl<R>> {
        self.get_group().from_ring_el(x)
    }

    ///
    /// Returns the ring element represented by the given group element.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn as_ring_el<'a>(&self, x: &'a MultGroupEl<R>) -> &'a El<R> {
        self.get_group().as_ring_el(x)
    }
}

#[stability::unstable(feature = "enable")]
pub struct HashableGroupEl<G: AbelianGroupStore> {
    group: G,
    el: GroupEl<G>
}

impl<G: AbelianGroupStore> HashableGroupEl<G> {
    
    #[stability::unstable(feature = "enable")]
    pub fn new(group: G, el: GroupEl<G>) -> Self {
        Self { group, el }
    }
}

impl<G: AbelianGroupStore> PartialEq for HashableGroupEl<G> {
    fn eq(&self, other: &Self) -> bool {
        self.group.eq_el(&self.el, &other.el)
    }
}

impl<G: AbelianGroupStore> Eq for HashableGroupEl<G> {}

impl<G: AbelianGroupStore> Hash for HashableGroupEl<G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.group.hash(&self.el, state)
    }
}
