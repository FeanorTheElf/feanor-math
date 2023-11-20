use std::ops::Deref;

use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::algorithms;

///
/// Basic trait for objects that have a ring structure.
/// 
/// Implementors of this trait should provide the basic ring operations,
/// and additionally operators for displaying and equality testing. If
/// a performance advantage can be achieved by acceFpting some arguments by
/// reference instead of by value, the default-implemented functions for
/// ring operations on references should be overwritten.
/// 
/// # Relationship with [`RingStore`]
/// 
/// Note that usually, this trait will not be used directly, but always
/// through a [`RingStore`]. In more detail, while this trait
/// defines the functionality, [`RingStore`] allows abstracting
/// the storage - everything that allows access to a ring then is a 
/// [`RingStore`], as for example, references or shared pointers
/// to rings. If you want to use rings directly by value, some technical
/// details make it necessary to use the no-op container [`RingStore`].
/// For more detail, see the documentation of [`RingStore`].
///  
/// # Example
/// 
/// An example implementation of a new, very useless ring type that represents
/// 32-bit integers stored on the heap.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// 
/// #[derive(PartialEq)]
/// struct MyRingBase;
/// 
/// impl RingBase for MyRingBase {
///     
///     type Element = Box<i32>;
/// 
///     fn clone_el(&self, val: &Self::Element) -> Self::Element { val.clone() }
///
///     fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { **lhs += *rhs; }
/// 
///     fn negate_inplace(&self, lhs: &mut Self::Element) { **lhs = -**lhs; }
/// 
///     fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { **lhs *= *rhs; }
/// 
///     fn from_int(&self, value: i32) -> Self::Element { Box::new(value) }
/// 
///     fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { **lhs == **rhs }
/// 
///     fn is_commutative(&self) -> bool { true }
/// 
///     fn is_noetherian(&self) -> bool { true }
/// 
///     fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
///         write!(out, "{}", **value)
///     }
/// }
/// 
/// // To use the ring through a RingStore, it is also required to implement CanHomFrom<Self>
/// // and CanonicalIso<Self>.
/// 
/// impl CanHomFrom<MyRingBase> for MyRingBase {
/// 
///     type Homomorphism = ();
/// 
///     fn has_canonical_hom(&self, _from: &MyRingBase) -> Option<()> { Some(()) }
/// 
///     fn map_in(&self, _from: &MyRingBase, el: Self::Element, _: &()) -> Self::Element { el }
/// }
/// 
/// impl CanonicalIso<MyRingBase> for MyRingBase {
/// 
///     type Isomorphism = ();
/// 
///     fn has_canonical_iso(&self, _from: &MyRingBase) -> Option<()> { Some(()) }
/// 
///     fn map_out(&self, _from: &MyRingBase, el: Self::Element, _: &()) -> Self::Element { el }
/// }
/// 
/// // A type alias for the simple, by-value RingStore.
/// pub type MyRing = RingValue<MyRingBase>;
/// 
/// impl MyRingBase {
/// 
///     pub const RING: MyRing = RingValue::from(MyRingBase);
/// }
/// 
/// let ring = MyRingBase::RING;
/// assert!(ring.eq_el(
///     &ring.int_hom().map(6), 
///     &ring.mul(ring.int_hom().map(3), ring.int_hom().map(2))
/// ));
/// ```
/// 
/// # A note on equality
/// 
/// Generally speaking, the notion of being canonically isomorphic 
/// (given by [`CanonicalIso`] is often more useful for rings than 
/// equality (defined by [`PartialEq`]).
/// 
/// In particular, being canonically isomorphic means that that there
/// is a bidirectional mapping of elements `a in Ring1 <-> b in Ring2`
/// such that `a` and `b` behave exactly the same. This mapping is provided
/// by the functions of [`CanonicalIso`]. Note that every ring is supposed
/// to be canonically isomorphic to itself, via the identiy mapping.
/// 
/// The notion of equality is stronger than that. In particular, implementors
/// of [`PartialEq`] must ensure that if rings `R` and `S` are equal, then
/// they are canonically isomorphic and the canonical isomorphism is given
/// by bitwise identity map. In particular, elements of `R` and `S` must have
/// the same type.
/// 
/// Hence, be careful to not mix up elements of different rings, even if they
/// have the same type. This can easily lead to nasty errors. For example, 
/// consider the following code
/// ```
/// # use feanor_math::ring::*;
    /// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::zn::*;
/// 
/// let Z7 = zn_barett::Zn::new(StaticRing::<i64>::RING, 7);
/// let Z11 = zn_barett::Zn::new(StaticRing::<i64>::RING, 11);
/// let neg_one = Z7.int_hom().map(-1);
/// assert!(!Z11.is_neg_one(&neg_one));
/// ```
/// 
pub trait RingBase: PartialEq {

    type Element;

    fn clone_el(&self, val: &Self::Element) -> Self::Element;
    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.add_assign(lhs, self.clone_el(rhs)) }
    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element);
    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.sub_assign(lhs, self.clone_el(rhs)) }
    fn negate_inplace(&self, lhs: &mut Self::Element);
    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element);
    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.mul_assign(lhs, self.clone_el(rhs)) }
    fn zero(&self) -> Self::Element { self.from_int(0) }
    fn one(&self) -> Self::Element { self.from_int(1) }
    fn neg_one(&self) -> Self::Element { self.from_int(-1) }
    fn from_int(&self, value: i32) -> Self::Element;
    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool;
    fn is_zero(&self, value: &Self::Element) -> bool { self.eq_el(value, &self.zero()) }
    fn is_one(&self, value: &Self::Element) -> bool { self.eq_el(value, &self.one()) }
    fn is_neg_one(&self, value: &Self::Element) -> bool { self.eq_el(value, &self.neg_one()) }
    fn is_commutative(&self) -> bool;
    fn is_noetherian(&self) -> bool;

    ///
    /// Returns whether this ring computes with approximations to elements.
    /// This would usually be the case for rings that are based on `f32` or
    /// `f64`, to represent real or complex numbers.
    /// 
    /// Note that these rings cannot provide implementations for [`Self::eq_el()`], 
    /// [`Self::is_zero()`] etc, and hence are of limited use in this crate.
    /// Currently, the only way how approximate rings are used is a complex-valued
    /// fast Fourier transform, via [`crate::rings::float_complex::Complex64`].
    /// 
    fn is_approximate(&self) -> bool { false }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result;

    fn square(&self, value: &mut Self::Element) {
        self.mul_assign(value, self.clone_el(value));
    }

    fn negate(&self, mut value: Self::Element) -> Self::Element {
        self.negate_inplace(&mut value);
        return value;
    }
    
    fn sub_assign(&self, lhs: &mut Self::Element, mut rhs: Self::Element) {
        self.negate_inplace(&mut rhs);
        self.add_assign(lhs, rhs);
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        self.mul_assign(lhs, self.from_int(rhs));
    }

    fn mul_int(&self, mut lhs: Self::Element, rhs: i32) -> Self::Element {
        self.mul_assign_int(&mut lhs, rhs);
        return lhs;
    }

    fn mul_int_ref(&self, lhs: &Self::Element, rhs: i32) -> Self::Element {
        self.mul_int(self.clone_el(lhs), rhs)
    }

    ///
    /// Computes `lhs := rhs - lhs`.
    /// 
    fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.negate_inplace(lhs);
        self.add_assign(lhs, rhs);
    }

    ///
    /// Computes `lhs := rhs - lhs`.
    /// 
    fn sub_self_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.negate_inplace(lhs);
        self.add_assign_ref(lhs, rhs);
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = self.clone_el(lhs);
        self.add_assign_ref(&mut result, rhs);
        return result;
    }

    fn add_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        self.add_assign_ref(&mut rhs, lhs);
        return rhs;
    }

    fn add_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.add_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn add(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.add_assign(&mut lhs, rhs);
        return lhs;
    }

    fn sub_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = self.clone_el(lhs);
        self.sub_assign_ref(&mut result, rhs);
        return result;
    }

    fn sub_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        self.sub_assign_ref(&mut rhs, lhs);
        self.negate_inplace(&mut rhs);
        return rhs;
    }

    fn sub_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.sub_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn sub(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.sub_assign(&mut lhs, rhs);
        return lhs;
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = self.clone_el(lhs);
        self.mul_assign_ref(&mut result, rhs);
        return result;
    }

    fn mul_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        if self.is_commutative() {
            self.mul_assign_ref(&mut rhs, lhs);
            return rhs;
        } else {
            let mut result = self.clone_el(lhs);
            self.mul_assign(&mut result, rhs);
            return result;
        }
    }

    fn mul_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.mul_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn mul(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.mul_assign(&mut lhs, rhs);
        return lhs;
    }
    
    ///
    /// Raises `x` to the power of an arbitrary, nonnegative integer given by
    /// a custom integer ring implementation.
    /// 
    /// Unless overriden by implementors, this uses a square-and-multiply approach
    /// to achieve running time O(log(power)).
    /// 
    /// # Panic
    /// 
    /// This may panic if `power` is negative.
    /// 
    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing,
            Self: SelfIso
    {
        assert!(!integers.is_neg(power));
        algorithms::sqr_mul::generic_pow(
            x, 
            power, 
            &RingRef::new(self),
            &RingRef::new(self),
            &integers
        )
    }

    fn sum<I>(&self, els: I) -> Self::Element 
        where I: Iterator<Item = Self::Element>
    {
        els.fold(self.zero(), |a, b| self.add(a, b))
    }

    fn prod<I>(&self, els: I) -> Self::Element 
        where I: Iterator<Item = Self::Element>
    {
        els.fold(self.one(), |a, b| self.mul(a, b))
    }
}

///
/// Used to easily implement functions in the trait definition of
/// [`RingStore`] and its subtraits to delegate the call to the same
/// function of the underlying [`RingBase`].
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # #[macro_use] use feanor_math::delegate;
/// 
/// trait WeirdRingBase: RingBase {
///     fn foo(&self) -> Self::Element;
/// }
/// 
/// trait WeirdRingStore: RingStore
///     where Self::Type: WeirdRingBase
/// {
///     delegate!{ fn foo(&self) -> El<Self> }
/// }
/// ```
/// 
#[macro_export]
macro_rules! delegate {
    (fn $name:ident (&self, $($pname:ident: $ptype:ty),*) -> $rtype:ty) => {
        fn $name (&self, $($pname: $ptype),*) -> $rtype {
            self.get_ring().$name($($pname),*)
        }
    };
    (fn $name:ident (&self) -> $rtype:ty) => {
        fn $name (&self) -> $rtype {
            self.get_ring().$name()
        }
    };
}

///
/// Implements the trivial canonical isomorphism `Self: CanonicalIso<Self>` for the
/// given type. 
/// 
/// Note that this does not support generic types, as for those, it is
/// usually better to implement
/// ```ignore
/// RingConstructor<R>: CanonicalIso<RingConstructor<S>>
///     where R: CanonicalIso<S>
/// ```
/// or something similar.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// 
/// #[derive(PartialEq, Clone, Copy)]
/// struct MyI32Ring;
/// 
/// impl DelegateRing for MyI32Ring {
/// 
///     type Base = StaticRingBase<i32>;
///     type Element = i32;
/// 
///     fn get_delegate(&self) -> &Self::Base {
///         StaticRing::<i32>::RING.get_ring()
///     }
/// 
///     fn delegate_ref<'a>(&self, el: &'a i32) -> &'a i32 {
///         el
///     }
/// 
///     fn delegate_mut<'a>(&self, el: &'a mut i32) -> &'a mut i32 {
///         el
///     }
/// 
///     fn delegate(&self, el: i32) -> i32 {
///         el
///     }
/// 
///     fn postprocess_delegate_mut(&self, _: &mut i32) {
///         // sometimes it might be necessary to fix some data of `Self::Element`
///         // if the underlying `Self::Base::Element` was modified via `delegate_mut()`;
///         // this is not the case here, so leave empty
///     }
/// 
///     fn rev_delegate(&self, el: i32) -> i32 {
///         el
///     }
/// }
/// 
/// // since we provide `PartialEq`, the trait `CanonicalIso<Self>` is trivial
/// // to implement
/// impl_eq_based_self_iso!{ MyI32Ring }
/// 
/// let ring = RingValue::from(MyI32Ring);
/// assert_el_eq!(&ring, &ring.int_hom().map(1), &ring.one());
/// ```
/// 
#[macro_export]
macro_rules! impl_eq_based_self_iso {
    ($type:ty) => {
        impl CanHomFrom<Self> for $type {

            type Homomorphism = ();

            fn has_canonical_hom(&self, from: &Self) -> Option<()> {
                if self == from {
                    Some(())
                } else {
                    None
                }
            }

            fn map_in(&self, _from: &Self, el: <Self as RingBase>::Element, _: &Self::Homomorphism) -> <Self as RingBase>::Element {
                el
            }
        }
        
        impl CanonicalIso<Self> for $type {

            type Isomorphism = ();

            fn has_canonical_iso(&self, from: &Self) -> Option<()> {
                if self == from {
                    Some(())
                } else {
                    None
                }
            }

            fn map_out(&self, _from: &Self, el: <Self as RingBase>::Element, _: &Self::Homomorphism) -> <Self as RingBase>::Element {
                el
            }
        }
    };
}

///
/// Equivalent to `assert_eq!` to assert that two ring elements are equal.
/// Frequently used in tests
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::assert_el_eq;
/// 
/// assert_el_eq!(&StaticRing::<i32>::RING, &3, &3);
/// // is equivalent to
/// assert_eq!(3, 3);
/// ```
/// If the ring elements are not comparable on their own, this is really useful
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::assert_el_eq;
/// 
/// // this does not have an equivalent representation with assert_eq!
/// assert_el_eq!(&BigIntRing::RING, &BigIntRing::RING.int_hom().map(3), &BigIntRing::RING.int_hom().map(3));
/// ```
/// 
#[macro_export]
macro_rules! assert_el_eq {
    ($ring:expr, $lhs:expr, $rhs:expr) => {
        match ($ring, $lhs, $rhs) {
            (ring_val, lhs_val, rhs_val) => {
                assert!(ring_val.eq_el(lhs_val, rhs_val), "Assertion failed: {} != {}", <_ as $crate::ring::RingStore>::format(ring_val, lhs_val), <_ as $crate::ring::RingStore>::format(ring_val, rhs_val));
            }
        }
    }
}

///
/// Basic trait for objects that store (in some sense) a ring. This can
/// be a ring-by-value, a reference to a ring, or really any object that
/// provides access to a [`RingBase`] object.
/// 
/// As opposed to [`RingBase`], which is responsible for the
/// functionality and ring operations, this trait is solely responsible for
/// the storage. The two basic implementors are [`RingValue`] and [`RingRef`],
/// which just wrap a value resp. reference to a [`RingBase`] object.
/// Building on that, every object that wraps a [`RingStore`] object can implement
/// again [`RingStore`]. This applies in particular to implementors of
/// `Deref<Target: RingStore>`, for whom there is a blanket implementation.
/// 
/// # Example
/// 
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// fn add_in_ring<R: RingStore>(ring: R, a: El<R>, b: El<R>) -> El<R> {
///     ring.add(a, b)
/// }
/// 
/// let ring: RingValue<StaticRingBase<i64>> = StaticRing::<i64>::RING;
/// assert_el_eq!(&ring, &7, &add_in_ring(ring, 3, 4));
/// ```
/// 
/// # What does this do?
/// 
/// We need a framework that allows nesting rings, e.g. to provide a polynomial ring
/// over a finite field - say `PolyRing<FiniteField>`. However, the simplest
/// implementation
/// ```ignore
/// struct PolyRing<BaseRing: Ring> { /* omitted */ }
/// ```
/// would have the effect that `PolyRing<FiniteField>` and `PolyRing<&FiniteField>`
/// are entirely different types. While implementing relationships between them
/// is possible, the approach does not scale well when we consider many rings and
/// multiple layers of nesting.
/// 
/// # Note for implementors
/// 
/// Generally speaking it is not recommended to overwrite any of the default-implementations
/// of ring functionality, as this is against the spirit of this trait. Instead,
/// just provide an implementation of `get_ring()` and put ring functionality in
/// a custom implementation of [`RingBase`].
/// 
pub trait RingStore: Sized {
    
    type Type: RingBase + SelfIso + ?Sized;

    fn get_ring<'a>(&'a self) -> &'a Self::Type;

    delegate!{ fn clone_el(&self, val: &El<Self>) -> El<Self> }
    delegate!{ fn add_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn add_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn sub_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn sub_self_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn sub_self_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn negate_inplace(&self, lhs: &mut El<Self>) -> () }
    delegate!{ fn mul_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn mul_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn zero(&self) -> El<Self> }
    delegate!{ fn one(&self) -> El<Self> }
    delegate!{ fn neg_one(&self) -> El<Self> }
    delegate!{ fn eq_el(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_zero(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_one(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_neg_one(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_commutative(&self) -> bool }
    delegate!{ fn is_noetherian(&self) -> bool }
    delegate!{ fn negate(&self, value: El<Self>) -> El<Self> }
    delegate!{ fn sub_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn add_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn add_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn add_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn add(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn sub_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn sub_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn sub_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn sub(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn mul_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn mul_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn mul_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn mul(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn square(&self, value: &mut El<Self>) -> () }
    
    fn coerce<S>(&self, from: &S, el: El<S>) -> El<Self>
        where S: RingStore, Self::Type: CanHomFrom<S::Type> 
    {
        self.get_ring().map_in(from.get_ring(), el, &self.get_ring().has_canonical_hom(from.get_ring()).unwrap())
    }

    fn coerce_ref<S>(&self, from: &S, el: &El<S>) -> El<Self>
        where S: RingStore, Self::Type: CanHomFrom<S::Type> 
    {
        self.get_ring().map_in_ref(from.get_ring(), el, &self.get_ring().has_canonical_hom(from.get_ring()).unwrap())
    }

    fn cast<S>(&self, to: &S, el: El<Self>) -> El<S>
        where S: RingStore, Self::Type: CanonicalIso<S::Type> 
    {
        self.get_ring().map_out(to.get_ring(), el, &self.get_ring().has_canonical_iso(to.get_ring()).unwrap())
    }

    ///
    /// Returns the canonical homomorphism `from -> self`, if it exists,
    /// moving both rings into the [`CanHom`] object.
    /// 
    fn into_can_hom<S>(self, from: S) -> Result<CanHom<S, Self>, (S, Self)>
        where Self: Sized, S: RingStore, Self::Type: CanHomFrom<S::Type>
    {
        CanHom::new(from, self)
    }

    ///
    /// Returns the canonical isomorphism `from -> self`, if it exists,
    /// moving both rings into the [`CanHom`] object.
    /// 
    fn into_can_iso<S>(self, from: S) -> Result<CanIso<S, Self>, (S, Self)>
        where Self: Sized, S: RingStore, Self::Type: CanonicalIso<S::Type>
    {
        CanIso::new(from, self)
    }

    ///
    /// Returns the canonical homomorphism `from -> self`, if it exists.
    /// 
    fn can_hom<'a, S>(&'a self, from: &'a S) -> Option<CanHom<&'a S, &'a Self>>
        where S: RingStore, Self::Type: CanHomFrom<S::Type>
    {
        self.into_can_hom(from).ok()
    }

    ///
    /// Returns the canonical isomorphism `from -> self`, if it exists.
    /// 
    fn can_iso<'a, S>(&'a self, from: &'a S) -> Option<CanIso<&'a S, &'a Self>>
        where S: RingStore, Self::Type: CanonicalIso<S::Type>
    {
        self.into_can_iso(from).ok()
    }

    fn into_int_hom(self) -> IntHom<Self> {
        IntHom::new(self)
    }

    fn int_hom<'a>(&'a self) -> IntHom<&'a Self> {
        self.into_int_hom()
    }

    fn sum<I>(&self, els: I) -> El<Self> 
        where I: Iterator<Item = El<Self>>
    {
        self.get_ring().sum(els)
    }

    fn prod<I>(&self, els: I) -> El<Self> 
        where I: Iterator<Item = El<Self>>
    {
        self.get_ring().prod(els)
    }

    fn pow(&self, mut x: El<Self>, power: usize) -> El<Self> {
        // special cases to increase performance
        if power == 0 {
            return self.one();
        } else if power == 1 {
            return x;
        } else if power == 2 {
            self.square(&mut x);
            return x;
        }
        self.pow_gen(x, &(power as i64), StaticRing::<i64>::RING)
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: El<Self>, power: &El<R>, integers: R) -> El<Self> 
        where R::Type: IntegerRing
    {
        self.get_ring().pow_gen(x, power, integers)
    }

    ///
    /// Returns an object that represents the given ring element and implements
    /// [`std::fmt::Display`], to use as formatting parameter.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::integer::*;
    /// let ring = BigIntRing::RING;
    /// let element = ring.int_hom().map(3);
    /// println!("{}", ring.format(&element));
    /// 
    fn format<'a>(&'a self, value: &'a El<Self>) -> RingElementDisplayWrapper<'a, Self> {
        RingElementDisplayWrapper { ring: self, element: value }
    }

    fn println(&self, value: &El<Self>) {
        println!("{}", self.format(value));
    }
}

pub trait RingExtensionStore: RingStore
    where Self::Type: RingExtension
{
    fn base_ring<'a>(&'a self) -> &'a <Self::Type as RingExtension>::BaseRing {
        self.get_ring().base_ring()
    }

    fn into_inclusion(self) -> Inclusion<Self> {
        Inclusion::new(self)
    }

    fn inclusion<'a>(&'a self) -> Inclusion<&'a Self> {
        self.into_inclusion()
    }
}

impl<R: RingStore> RingExtensionStore for R
    where R::Type: RingExtension
{}

pub struct RingElementDisplayWrapper<'a, R: RingStore + ?Sized> {
    ring: &'a R,
    element: &'a El<R>
}

impl<'a, R: RingStore + ?Sized> std::fmt::Display for RingElementDisplayWrapper<'a, R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.get_ring().dbg(self.element, f)
    }
}

///
/// Trait for rings that are an extension ring of a base ring.
/// This does not have to be a proper extension in the mathematical
/// sense, but is in some cases implemented for a wrapper of a ring
/// object that represents the same ring.
/// 
/// Hence, this is technically just a ring `R` with an injective homomorphism
/// `BaseRing -> R`, but unlike [`CanHomFrom`], implementors must provide
/// a reference to `BaseRing` via [`RingExtension::base_ring()`].
/// 
/// # Overlap with [`CanHomFrom`]
/// 
/// There is a certain amount of functionality overlap with [`CanHomFrom`], and
/// in a perfect world, this trait would also be a subtrait of `CanHomFrom<<Self::BaseRing as RingStore>::Type>`.
/// However, due to the issue with multiple blanket implementations for [`CanHomFrom`] (see
/// the docs), this is not the case and in fact there are ring extensions that do not implement
/// `CanHomFrom<<Self::BaseRing as RingStore>::Type>`.
/// 
pub trait RingExtension: RingBase {
    
    type BaseRing: RingStore;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing;
    fn from(&self, x: El<Self::BaseRing>) -> Self::Element;
    
    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        self.from(self.base_ring().get_ring().clone_el(x))
    }

    ///
    /// Computes `lhs := lhs * rhs`, where `rhs` is mapped into this
    /// ring via [`RingExtension::from_ref()`]. Note that this may be
    /// faster than `self.mul_assign(lhs, self.from_ref(rhs))`.
    /// 
    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.mul_assign(lhs, self.from_ref(rhs));
    }
}

///
/// Trait for rings that can compute hashes for their elements.
/// This should be compatible with [`RingBase::eq_el`] in the usual way.
/// 
pub trait HashableElRing: RingBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H);
}

pub trait HashableElRingStore: RingStore
    where Self::Type: HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &El<Self>, h: &mut H) {
        self.get_ring().hash(el, h)
    }

    fn default_hash(&self, el: &El<Self>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(el, &mut hasher);
        return <std::collections::hash_map::DefaultHasher as std::hash::Hasher>::finish(&hasher);
    }
}

impl<R> HashableElRingStore for R
    where R: RingStore,
        R::Type: HashableElRing
{}

///
/// Alias for `<<Self as RingStore>::Type as RingBase>::Element`.
/// 
pub type El<R> = <<R as RingStore>::Type as RingBase>::Element;

///
/// The most fundamental [`crate::ring::RingStore`]. It is basically
/// a no-op container, i.e. stores a [`crate::ring::RingBase`] object
/// by value, and allows accessing it.
/// 
/// # Why is this necessary?
/// 
/// In fact, that we need this trait is just the result of a technical
/// detail. We cannot implement
/// ```ignore
/// impl<R: RingBase> RingStore for R {}
/// impl<'a, R: RingStore> RingStore for &;a R {}
/// ```
/// since this might cause conflicting implementations.
/// Instead, we implement
/// ```ignore
/// impl<R: RingBase> RingStore for RingValue<R> {}
/// impl<'a, R: RingStore> RingStore for &;a R {}
/// ```
/// This causes some inconvenience, as now we cannot chain
/// [`crate::ring::RingStore`] in the case of [`crate::ring::RingValue`].
/// Furthermore, this trait will be necessary everywhere - 
/// to define a reference to a ring of type `A`, we now have to
/// write `&RingValue<A>`.
/// 
/// To simplify this, we propose to use the following simple pattern:
/// Create your ring type as
/// ```ignore
/// struct ABase { ... }
/// impl RingBase for ABase { ... } 
/// ```
/// and then provide a type alias
/// ```ignore
/// type A = RingValue<ABase>;
/// ```
/// 
#[derive(Copy, Clone)]
pub struct RingValue<R: RingBase> {
    ring: R
}

impl<R: RingBase> RingValue<R> {

    pub const fn from(value: R) -> Self {
        RingValue { ring: value }
    }
}

impl<R: RingBase + CanonicalIso<R>> RingStore for RingValue<R> {

    type Type = R;
    
    fn get_ring(&self) -> &R {
        &self.ring
    }
}

///
/// The second most basic [`crate::ring::RingStore`]. Similarly to 
/// [`crate::ring::RingValue`] it is just a no-op container.
/// 
/// # Why do we need this in addition to [`crate::ring::RingValue`]?
/// 
/// The role of `RingRef` is much more niche than the role of [`crate::ring::RingValue`].
/// However, it might happen that we want to implement [`crate::ring::RingBase`]-functions (or traits on the
/// same level, e.g. [`crate::ring::CanHomFrom`], [`crate::divisibility::DivisibilityRing`]),
/// and use more high-level techniques for that (e.g. complex algorithms, for example [`crate::algorithms::eea`]
/// or [`crate::algorithms::sqr_mul`]). In this case, we only have a reference to a [`crate::ring::RingBase`]
/// object, but require a [`crate::ring::RingStore`] object to use the algorithm.
/// 
pub struct RingRef<'a, R: RingBase + CanonicalIso<R> + ?Sized> {
    ring: &'a R
}

impl<'a, R: RingBase + CanonicalIso<R> + ?Sized> Clone for RingRef<'a, R> {

    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: RingBase + CanonicalIso<R> + ?Sized> Copy for RingRef<'a, R> {}

impl<'a, R: RingBase + CanonicalIso<R> + ?Sized> RingRef<'a, R> {

    pub const fn new(value: &'a R) -> Self {
        RingRef { ring: value }
    }
}

impl<'a, R: RingBase + CanonicalIso<R> + ?Sized> RingStore for RingRef<'a, R> {

    type Type = R;
    
    fn get_ring(&self) -> &R {
        self.ring
    }
}

impl<'a, S: Deref> RingStore for S
    where S::Target: RingStore
{    
    type Type = <<S as Deref>::Target as RingStore>::Type;
    
    fn get_ring<'b>(&'b self) -> &'b Self::Type {
        (**self).get_ring()
    }
}

#[cfg(test)]
use std::rc::Rc;

#[test]
fn test_ring_rc_lifetimes() {
    let ring = Rc::new(StaticRing::<i32>::RING);
    let mut ring_ref = None;
    assert!(ring_ref.is_none());
    {
        ring_ref = Some(ring.get_ring());
    }
    assert!(ring.get_ring().is_commutative());
    assert!(ring_ref.is_some());
}

#[test]
fn test_internal_wrappings_dont_matter() {
    
    #[derive(Copy, Clone, PartialEq)]
    pub struct ABase;

    #[allow(unused)]
    #[derive(Copy, Clone)]
    pub struct BBase<R: RingStore> {
        base: R
    }

    impl<R: RingStore> PartialEq for BBase<R> {
        fn eq(&self, other: &Self) -> bool {
            self.base.get_ring() == other.base.get_ring()
        }
    }

    impl RingBase for ABase {
        type Element = i32;

        fn clone_el(&self, val: &Self::Element) -> Self::Element {
            *val
        }

        fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs += rhs;
        }

        fn negate_inplace(&self, lhs: &mut Self::Element) {
            *lhs = -*lhs;
        }

        fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
            *lhs == *rhs
        }

        fn is_commutative(&self) -> bool {
            true
        }

        fn is_noetherian(&self) -> bool {
            true
        }

        fn from_int(&self, value: i32) -> Self::Element {
            value
        }

        fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs *= rhs;
        }

        fn dbg<'a>(&self, _: &Self::Element, _: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
            Ok(())
        }
    }

    impl_eq_based_self_iso!{ ABase }

    impl<R: RingStore> RingBase for BBase<R> {
        type Element = i32;

        fn clone_el(&self, val: &Self::Element) -> Self::Element {
            *val
        }

        fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs += rhs;
        }
        fn negate_inplace(&self, lhs: &mut Self::Element) {
            *lhs = -*lhs;
        }

        fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
            *lhs == *rhs
        }

        fn is_commutative(&self) -> bool {
            true
        }

        fn is_noetherian(&self) -> bool {
            true
        }

        fn from_int(&self, value: i32) -> Self::Element {
            value
        }

        fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs *= rhs;
        }

        fn dbg<'a>(&self, _: &Self::Element, _: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
            Ok(())
        }
    }

    impl<R: RingStore> CanHomFrom<ABase> for BBase<R> {

        type Homomorphism = ();

        fn has_canonical_hom(&self, _: &ABase) -> Option<()> {
            Some(())
        }

        fn map_in(&self, _: &ABase, el: <ABase as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

    impl<R: RingStore, S: RingStore> CanHomFrom<BBase<S>> for BBase<R> 
        where R::Type: CanHomFrom<S::Type>
    {
        type Homomorphism = ();

        fn has_canonical_hom(&self, _: &BBase<S>) -> Option<()> {
            Some(())
        }

        fn map_in(&self, _: &BBase<S>, el: <BBase<S> as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

    impl<R: RingStore> CanonicalIso<BBase<R>> for BBase<R> {

        type Isomorphism = ();

        fn has_canonical_iso(&self, _: &BBase<R>) -> Option<()> {
            Some(())
        }

        fn map_out(&self, _: &BBase<R>, el: <BBase<R> as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

    type A = RingValue<ABase>;
    type B<R> = RingValue<BBase<R>>;

    let a: A = RingValue { ring: ABase };
    let b1: B<A> = RingValue { ring: BBase { base: a } };
    let b2: B<&B<A>> = RingValue { ring: BBase { base: &b1 } };
    let b3: B<&A> = RingValue { ring: BBase { base: &a } };
    b1.coerce(&a, 0);
    b2.coerce(&a, 0);
    b2.coerce(&b1, 0);
    b2.coerce(&b3, 0);
    (&b2).coerce(&b3, 0);
    (&b2).coerce(&&&b3, 0);
}



#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use super::*;

    pub fn test_hom_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
        where S::Type: CanHomFrom<R::Type>
    {
        let hom = to.get_ring().has_canonical_hom(from.get_ring()).unwrap();
        let elements = edge_case_elements.collect::<Vec<_>>();

        for a in &elements {
            for b in &elements {
                {
                    let map_a = to.get_ring().map_in_ref(from.get_ring(), a, &hom);
                    let map_b = to.get_ring().map_in_ref(from.get_ring(), b, &hom);
                    let map_add = to.add_ref(&map_a, &map_b);
                    let add_map = to.get_ring().map_in(from.get_ring(), from.add_ref(a, b), &hom);
                    assert!(to.eq_el(&map_add, &add_map), "Additive homomorphic property failed: hom({} + {}) = {} != {} = {} + {}", from.format(a), from.format(b), to.format(&add_map), to.format(&map_add), to.format(&map_a), to.format(&map_b));
                }
                {
                    let map_a = to.get_ring().map_in_ref(from.get_ring(), a, &hom);
                    let map_b = to.get_ring().map_in_ref(from.get_ring(), b, &hom);
                    let map_mul = to.mul_ref(&map_a, &map_b);
                    let mul_map = to.get_ring().map_in(from.get_ring(), from.mul_ref(a, b), &hom);
                    assert!(to.eq_el(&map_mul, &mul_map), "Multiplicative homomorphic property failed: hom({} * {}) = {} != {} = {} * {}", from.format(a), from.format(b), to.format(&mul_map), to.format(&map_mul), to.format(&map_a), to.format(&map_b));
                }
                {
                    let map_a = to.get_ring().map_in_ref(from.get_ring(), a, &hom);
                    let mul_map = to.get_ring().map_in(from.get_ring(), from.mul_ref(a, b), &hom);
                    let mut mul_assign = to.clone_el(&map_a);
                    to.get_ring().mul_assign_map_in_ref(from.get_ring(), &mut mul_assign, b, &hom);
                    assert!(to.eq_el(&mul_assign, &mul_map), "mul_assign_map_in_ref() failed: hom({} * {}) = {} != {} = mul_map_in(hom({}), {})", from.format(a), from.format(b), to.format(&mul_map), to.format(&mul_assign), to.format(&map_a), from.format(b));
                }
            }
        }
    }

    pub fn test_iso_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
        where S::Type: CanonicalIso<R::Type>
    {
        let hom = to.get_ring().has_canonical_hom(from.get_ring()).unwrap();
        let iso = to.get_ring().has_canonical_iso(from.get_ring()).unwrap();
        let elements = edge_case_elements.collect::<Vec<_>>();

        for a in &elements {
            let map_in = to.get_ring().map_in_ref(from.get_ring(), a, &hom);
            let map_in_out = to.get_ring().map_out(from.get_ring(), to.clone_el(&map_in), &iso);
            assert!(from.eq_el(&map_in_out, &a), "Bijectivity failed: {} != {} = hom^-1({}) = hom^-1(hom({}))", from.format(a), from.format(&map_in_out), to.format(&map_in), from.format(a));
        }
    }

    pub fn test_self_iso<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: SelfIso
    {
        let hom = ring.get_ring().has_canonical_hom(ring.get_ring()).unwrap();
        let iso = ring.get_ring().has_canonical_iso(ring.get_ring()).unwrap();
        let elements = edge_case_elements.collect::<Vec<_>>();

        test_hom_axioms(&ring, &ring, elements.iter().map(|x| ring.clone_el(x)));
        test_iso_axioms(&ring, &ring, elements.iter().map(|x| ring.clone_el(x)));

        for a in &elements {
            assert_el_eq!(&ring, a, &ring.get_ring().map_in_ref(ring.get_ring(), a, &hom));
            assert_el_eq!(&ring, a, &ring.get_ring().map_out(ring.get_ring(), ring.clone_el(a), &iso));
        }
    }

    pub fn test_ring_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: SelfIso
    {
        let elements = edge_case_elements.collect::<Vec<_>>();
        let zero = ring.zero();
        let one = ring.one();

        // check self-subtraction
        for a in &elements {
            let a_minus_a = ring.sub(ring.clone_el(a), ring.clone_el(a));
            assert!(ring.eq_el(&zero, &a_minus_a), "Additive inverse failed: {} - {} = {} != {}", ring.format(a), ring.format(a), ring.format(&a_minus_a), ring.format(&zero));
        }

        // check identity elements
        for a in &elements {
            let a_plus_zero = ring.add(ring.clone_el(a), ring.clone_el(&zero));
            assert!(ring.eq_el(a, &a_plus_zero), "Additive neutral element failed: {} + {} = {} != {}", ring.format(a), ring.format(&zero), ring.format(&a_plus_zero), ring.format(a));
            
            let a_times_one = ring.mul(ring.clone_el(a), ring.clone_el(&one));
            assert!(ring.eq_el(a, &a_times_one), "Multiplicative neutral element failed: {} * {} = {} != {}", ring.format(a), ring.format(&one), ring.format(&a_times_one), ring.format(a));
        }

        // check commutativity
        for a in &elements {
            for b in &elements {
                {
                    let ab = ring.add_ref(a, b);
                    let ba = ring.add_ref(b, a);
                    assert!(ring.eq_el(&ab, &ba), "Additive commutativity failed: {} + {} = {} != {} = {} + {}", ring.format(a), ring.format(b), ring.format(&ab), ring.format(&ba), ring.format(b), ring.format(a));
                }
                    
                if ring.is_commutative() {
                    let ab = ring.mul_ref(a, b);
                    let ba = ring.mul_ref(b, a);
                    assert!(ring.eq_el(&ab, &ba), "Multiplicative commutativity failed: {} * {} = {} != {} = {} * {}", ring.format(a), ring.format(b), ring.format(&ab), ring.format(&ba), ring.format(b), ring.format(a));
                }
            }
        }

        // check associativity
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    {
                        let ab_c = ring.add_ref_snd(ring.add_ref(a, b), c);
                        let a_bc = ring.add_ref_fst(c, ring.add_ref(b, a));
                        assert!(ring.eq_el(&ab_c, &a_bc), "Additive associativity failed: ({} + {}) + {} = {} != {} = {} + ({} + {})", ring.format(a), ring.format(b), ring.format(c), ring.format(&ab_c), ring.format(&a_bc), ring.format(a), ring.format(b), ring.format(c));
                    }
                    {
                        let ab_c = ring.mul_ref_snd(ring.mul_ref(a, b), c);
                        let a_bc = ring.mul_ref_fst(c, ring.mul_ref(b, a));
                        assert!(ring.eq_el(&ab_c, &a_bc), "Multiplicative associativity failed: ({} * {}) * {} = {} != {} = {} * ({} * {})", ring.format(a), ring.format(b), ring.format(c), ring.format(&ab_c), ring.format(&a_bc), ring.format(a), ring.format(b), ring.format(c));
                    }
                }
            }
        }
        
        // check distributivity
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    let ab_c = ring.mul_ref_snd(ring.add_ref(a, b), c);
                    let ac_bc = ring.add(ring.mul_ref(a, c), ring.mul_ref(b, c));
                    assert!(ring.eq_el(&ab_c, &ac_bc), "Distributivity failed: ({} + {}) * {} = {} != {} = {} * {} + {} * {}", ring.format(a), ring.format(b), ring.format(c), ring.format(&ab_c), ring.format(&ac_bc), ring.format(a), ring.format(c), ring.format(b), ring.format(c));

                    let a_bc = ring.mul_ref_fst(a, ring.add_ref(b, c));
                    let ab_ac = ring.add(ring.mul_ref(a, b), ring.mul_ref(a, c));
                    assert!(ring.eq_el(&a_bc, &ab_ac), "Distributivity failed: {} * ({} + {}) = {} != {} = {} * {} + {} * {}", ring.format(a), ring.format(b), ring.format(c), ring.format(&a_bc), ring.format(&ab_ac), ring.format(a), ring.format(b), ring.format(a), ring.format(c));                }
            }
        }

        test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    }
}