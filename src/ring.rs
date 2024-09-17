use std::ops::Deref;

use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::algorithms;

///
/// Describes the context in which to print an algebraic expression.
/// It is usually used to determine when to use parenthesis during printing.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// struct CustomDisplay<'a>(&'a DensePolyRing<StaticRing<i64>>, &'a El<DensePolyRing<StaticRing<i64>>>, EnvBindingStrength);
/// impl<'a> std::fmt::Display for CustomDisplay<'a> {
///     fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
///         self.0.get_ring().dbg_within(self.1, formatter, self.2)
///     }
/// }
/// let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let f = poly_ring.add(poly_ring.one(), poly_ring.indeterminate());
/// assert_eq!("1 + 1X", format!("{}", CustomDisplay(&poly_ring, &f, EnvBindingStrength::Weakest)));
/// assert_eq!("1 + 1X", format!("{}", CustomDisplay(&poly_ring, &f, EnvBindingStrength::Sum)));
/// assert_eq!("(1 + 1X)", format!("{}", CustomDisplay(&poly_ring, &f, EnvBindingStrength::Product)));
/// assert_eq!("(1 + 1X)", format!("{}", CustomDisplay(&poly_ring, &f, EnvBindingStrength::Power)));
/// assert_eq!("(1 + 1X)", format!("{}", CustomDisplay(&poly_ring, &f, EnvBindingStrength::Strongest)));
/// ```
/// 
#[derive(PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub enum EnvBindingStrength {
    Weakest, Sum, Product, Power, Strongest
}

///
/// Basic trait for objects that have a ring structure. This trait is 
/// implementor-facing, so designed to be used for implementing new
/// rings.
/// 
/// Implementors of this trait should provide the basic ring operations,
/// and additionally operators for displaying and equality testing. If
/// a performance advantage can be achieved by accepting some arguments by
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
/// # A note on equality
/// 
/// Generally speaking, the notion of being canonically isomorphic 
/// (given by [`CanIsoFromTo`] is often more useful for rings than 
/// equality (defined by [`PartialEq`]).
/// 
/// In particular, being canonically isomorphic means that that there
/// is a bidirectional mapping of elements `a in Ring1 <-> b in Ring2`
/// such that `a` and `b` behave exactly the same. This mapping is provided
/// by the functions of [`CanIsoFromTo`]. Note that every ring is supposed
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
/// let Z7 = zn_big::Zn::new(StaticRing::<i64>::RING, 7);
/// let Z11 = zn_big::Zn::new(StaticRing::<i64>::RING, 11);
/// assert!(Z11.get_ring() != Z7.get_ring());
/// let neg_one = Z7.int_hom().map(-1);
/// assert!(!Z11.is_neg_one(&neg_one));
/// ```
/// It can even make problems if both rings are isomorphic, and might be expected
/// to be intuitively "the same".
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::multivariate::*;
/// # use feanor_math::rings::multivariate::multivariate_impl::*;
/// let fst = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 2);
/// let snd = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 2);
/// // we might think they are the same, but they are not!
/// assert!(fst.get_ring() != snd.get_ring());
/// // f = X * Y
/// let f = fst.create_term(1, fst.create_monomial([1, 1].into_iter()));
/// // g = Y
/// let g = snd.create_term(1, snd.create_monomial([0, 1].into_iter()));
/// // thus, we may not swap elements between them (or get a nasty surprise, like `f = g`)
/// assert!(fst.eq_el(&f, &g));
/// // note that when debug assertions are enabled, this will panic, since the
/// // ring will detect that g is not one of its elements; however, the equality
/// // check will pass in builds without debug assertions.
/// ```
/// However, swapping elements between rings is well-defined and correct if they are "equal"
/// as given by `PartialEq` (not just canonically isomorphic)
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::zn::*;
/// let Z11_fst = zn_big::Zn::new(StaticRing::<i64>::RING, 7);
/// let Z11_snd = Z11_fst.clone();
/// assert!(Z11_fst.get_ring() == Z11_snd.get_ring());
/// let neg_one = Z11_fst.int_hom().map(-1);
/// assert!(Z11_fst.is_neg_one(&neg_one));
/// ```
/// 
/// # Example
/// 
/// An example implementation of a new, very useless ring type that represents
/// 32-bit integers stored on the heap.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::integer::*;
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
/// 
///     fn characteristic<I>(&self, ZZ: &I) -> Option<El<I>>
///         where I: IntegerRingStore, I::Type: IntegerRing
///     {
///         Some(ZZ.zero())
///     }
/// }
/// 
/// // To use the ring through a RingStore, it is also required to implement CanHomFrom<Self>
/// // and CanIsoFromTo<Self>.
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
/// impl CanIsoFromTo<MyRingBase> for MyRingBase {
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
    /// Note that these rings cannot provide implementations for [`RingBase::eq_el()`], 
    /// [`RingBase::is_zero()`] etc, and hence are of limited use in this crate.
    /// Currently, the only way how approximate rings are used is a complex-valued
    /// fast Fourier transform, via [`crate::rings::float_complex::Complex64`].
    /// 
    fn is_approximate(&self) -> bool { false }

    ///
    /// Writes a human-readable representation of `value` to `out`.
    /// 
    /// Used by [`RingStore::format()`], [`RingStore::println()`] and the implementations of [`std::fmt::Debug`] 
    /// and [`std::fmt::Display`] of [`crate::wrapper::RingElementWrapper`].
    /// 
    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result;

    ///
    /// Writes a human-readable representation of `value` to `out`, taking into account the possible context
    /// to place parenthesis as needed.
    /// 
    /// See also [`RingBase::dbg()`] and [`EnvBindingStrength`].
    /// 
    /// Used by [`RingStore::format()`], [`RingStore::println()`] and the implementations of [`std::fmt::Debug`] 
    /// and [`std::fmt::Display`] of [`crate::wrapper::RingElementWrapper`].
    /// 
    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        self.dbg(value, out)
    }

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
        where R::Type: IntegerRing
    {
        assert!(!integers.is_neg(power));
        algorithms::sqr_mul::generic_pow_shortest_chain_table(
            x, 
            power, 
            &integers,
            |a| {
                let mut a_copy = self.clone_el(a);
                self.square(&mut a_copy);
                Ok(a_copy)
            },
            |a, b| Ok(self.mul_ref(a, b)),
            |a| self.clone_el(a),
            self.one()
        ).unwrap_or_else(|x| x)
    }

    ///
    /// Sums the elements given by the iterator.
    /// 
    /// The implementation might be as simple as `els.fold(self.zero(), |a, b| self.add(a, b))`, but
    /// can be more efficient than that in some cases.
    /// 
    fn sum<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        els.into_iter().fold(self.zero(), |a, b| self.add(a, b))
    }

    ///
    /// Computes the product of the elements given by the iterator.
    /// 
    /// The implementation might be as simple as `els.fold(self.one(), |a, b| self.mul(a, b))`, but
    /// can be more efficient than that in some cases.
    /// 
    fn prod<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        els.into_iter().fold(self.one(), |a, b| self.mul(a, b))
    }
    
    ///
    /// Returns the characteristic of this ring as an element of the given
    /// implementation of `ZZ`. 
    /// 
    /// If `None` is returned, this means the given integer ring might not be able
    /// to represent the characteristic. This must never happen if the given implementation
    /// of `ZZ` allows for unbounded integers (like [`crate::integer::BigIntRing`]).
    /// In other cases however, we allow to perform the size check heuristically only,
    /// so this might return `None` even in some cases where the integer ring would in
    /// fact be able to represent the characteristic.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::rings::zn::*;
    /// let ZZ = StaticRing::<i16>::RING;
    /// assert_eq!(Some(0), StaticRing::<i64>::RING.characteristic(&ZZ));
    /// assert_eq!(None, zn_64::Zn::new(i16::MAX as u64 + 1).characteristic(&ZZ));
    /// assert_eq!(Some(i16::MAX), zn_64::Zn::new(i16::MAX as u64).characteristic(&ZZ));
    /// ```
    /// 
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing;
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
///     delegate!{ WeirdRingBase, fn foo(&self) -> El<Self> }
/// }
/// ```
/// 
/// # Limitations
/// 
/// This macro does not work if the function takes generic parameters.
/// In this case, write the delegation manually.
/// 
#[macro_export]
macro_rules! delegate {
    ($base_trait:ty, fn $name:ident (&self, $($pname:ident: $ptype:ty),*) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self, $($pname: $ptype),*) -> $rtype {
            <Self::Type as $base_trait>::$name(self.get_ring(), $($pname),*)
        }
    };
    ($base_trait:ty, fn $name:ident (&self) -> $rtype:ty) => {
        #[doc = concat!(" See [`", stringify!($base_trait), "::", stringify!($name), "()`]")]
        fn $name (&self) -> $rtype {
            <Self::Type as $base_trait>::$name(self.get_ring())
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
/// assert_el_eq!(StaticRing::<i32>::RING, 3, 3);
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
/// assert_el_eq!(BigIntRing::RING, BigIntRing::RING.int_hom().map(3), BigIntRing::RING.int_hom().map(3));
/// ```
/// 
#[macro_export]
macro_rules! assert_el_eq {
    ($ring:expr, $lhs:expr, $rhs:expr) => {
        match (&$ring, &$lhs, &$rhs) {
            (ring_val, lhs_val, rhs_val) => {
                assert!(<_ as $crate::ring::RingStore>::eq_el(ring_val, lhs_val, rhs_val), "Assertion failed: {} != {}", <_ as $crate::ring::RingStore>::format(ring_val, lhs_val), <_ as $crate::ring::RingStore>::format(ring_val, rhs_val));
            }
        }
    }
}

///
/// Basic trait for objects that store (in some sense) a ring. It can also
/// be considered the user-facing trait for rings, so rings are always supposed
/// to be used through a `RingStore`-object.
/// 
/// This can be a ring-by-value, a reference to a ring, or really any object that
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
/// # use std::rc::Rc;
/// fn add_in_ring<R: RingStore>(ring: R, a: El<R>, b: El<R>) -> El<R> {
///     ring.add(a, b)
/// }
/// 
/// let ring: RingValue<StaticRingBase<i64>> = StaticRing::<i64>::RING;
/// assert_el_eq!(ring, 7, add_in_ring(ring, 3, 4));
/// assert_el_eq!(ring, 7, add_in_ring(&ring, 3, 4));
/// assert_el_eq!(ring, 7, add_in_ring(Rc::new(ring), 3, 4));
/// ```
/// 
/// # What does this do?
/// 
/// We need a framework that allows nesting rings, e.g. to provide a polynomial ring
/// over a finite field - say `PolyRing<FiniteField>`. However, the simplest
/// implementation
/// ```rust,ignore
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
    
    type Type: RingBase + ?Sized;

    fn get_ring<'a>(&'a self) -> &'a Self::Type;

    delegate!{ RingBase, fn clone_el(&self, val: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn add_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ RingBase, fn add_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ RingBase, fn sub_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ RingBase, fn sub_self_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ RingBase, fn sub_self_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ RingBase, fn negate_inplace(&self, lhs: &mut El<Self>) -> () }
    delegate!{ RingBase, fn mul_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ RingBase, fn mul_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ RingBase, fn zero(&self) -> El<Self> }
    delegate!{ RingBase, fn one(&self) -> El<Self> }
    delegate!{ RingBase, fn neg_one(&self) -> El<Self> }
    delegate!{ RingBase, fn eq_el(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ RingBase, fn is_zero(&self, value: &El<Self>) -> bool }
    delegate!{ RingBase, fn is_one(&self, value: &El<Self>) -> bool }
    delegate!{ RingBase, fn is_neg_one(&self, value: &El<Self>) -> bool }
    delegate!{ RingBase, fn is_commutative(&self) -> bool }
    delegate!{ RingBase, fn is_noetherian(&self) -> bool }
    delegate!{ RingBase, fn negate(&self, value: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn sub_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ RingBase, fn add_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn add_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn add_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn add(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn sub_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn sub_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn sub_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn sub(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn mul_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn mul_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn mul_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ RingBase, fn mul(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ RingBase, fn square(&self, value: &mut El<Self>) -> () }

    fn coerce<S>(&self, from: &S, el: El<S>) -> El<Self>
        where S: RingStore, Self::Type: CanHomFrom<S::Type> 
    {
        self.get_ring().map_in(from.get_ring(), el, &self.get_ring().has_canonical_hom(from.get_ring()).unwrap())
    }

    ///
    /// Returns the identity map `self -> self`.
    /// 
    fn into_identity(self) -> Identity<Self> {
        Identity::new(self)
    }

    ///
    /// Returns the identity map `self -> self`.
    /// 
    fn identity<'a>(&'a self) -> Identity<&'a Self> {
        self.into_identity()
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
        where Self: Sized, S: RingStore, Self::Type: CanIsoFromTo<S::Type>
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
        where S: RingStore, Self::Type: CanIsoFromTo<S::Type>
    {
        self.into_can_iso(from).ok()
    }

    ///
    /// Returns the homomorphism `Z -> self` that exists for any ring.
    /// 
    fn into_int_hom(self) -> IntHom<Self> {
        IntHom::new(self)
    }

    ///
    /// Returns the homomorphism `Z -> self` that exists for any ring.
    /// 
    fn int_hom<'a>(&'a self) -> IntHom<&'a Self> {
        self.into_int_hom()
    }

    fn sum<I>(&self, els: I) -> El<Self> 
        where I: IntoIterator<Item = El<Self>>
    {
        self.get_ring().sum(els)
    }

    #[stability::unstable(feature = "enable")]
    fn try_sum<I, E>(&self, els: I) -> Result<El<Self>, E>
        where I: IntoIterator<Item = Result<El<Self>, E>>
    {
        let mut error = None;
        let result = self.get_ring().sum(els.into_iter().map_while(|el| match el {
            Ok(el) => Some(el),
            Err(err) => {
                error = Some(err);
                None
            }
        }));
        if let Some(err) = error {
            return Err(err);
        } else {
            return Ok(result);
        }
    }

    fn prod<I>(&self, els: I) -> El<Self> 
        where I: IntoIterator<Item = El<Self>>
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
    
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_ring().characteristic(ZZ)
    }
}

///
/// [`RingStore`] for [`RingExtension`]s
/// 
pub trait RingExtensionStore: RingStore
    where Self::Type: RingExtension
{
    fn base_ring<'a>(&'a self) -> &'a <Self::Type as RingExtension>::BaseRing {
        self.get_ring().base_ring()
    }

    ///
    /// Returns the inclusion map of the base ring `R -> self`.
    /// 
    fn into_inclusion(self) -> Inclusion<Self> {
        Inclusion::new(self)
    }

    ///
    /// Returns the inclusion map of the base ring `R -> self`.
    /// 
    fn inclusion<'a>(&'a self) -> Inclusion<&'a Self> {
        self.into_inclusion()
    }
}

impl<R: RingStore> RingExtensionStore for R
    where R::Type: RingExtension
{}

///
/// Wrapper around a ring and one of its elements that implements [`std::fmt::Display`]
/// and will print the element. Used by [`RingStore::format()`].
/// 
pub struct RingElementDisplayWrapper<'a, R: RingStore + ?Sized> {
    ring: &'a R,
    element: &'a El<R>
}

impl<'a, R: RingStore + ?Sized> std::fmt::Display for RingElementDisplayWrapper<'a, R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.get_ring().dbg(self.element, f)
    }
}

impl<'a, R: RingStore + ?Sized> std::fmt::Debug for RingElementDisplayWrapper<'a, R> {

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

///
/// [`RingStore`] for [`HashableElRing`]s
/// 
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
/// ```rust,ignore
/// impl<R: RingBase> RingStore for R {}
/// impl<'a, R: RingStore> RingStore for &'a R {}
/// ```
/// since this might cause conflicting implementations.
/// Instead, we implement
/// ```rust,ignore
/// impl<R: RingBase> RingStore for RingValue<R> {}
/// impl<'a, R: RingStore> RingStore for &'a R {}
/// ```
/// This causes some inconvenience, as now we cannot chain
/// [`crate::ring::RingStore`] in the case of [`crate::ring::RingValue`].
/// Furthermore, this trait will be necessary everywhere - 
/// to define a reference to a ring of type `A`, we now have to
/// write `&RingValue<A>`.
/// 
/// To simplify this, we propose to use the following simple pattern:
/// Create your ring type as
/// ```rust,ignore
/// struct ABase { ... }
/// impl RingBase for ABase { ... } 
/// ```
/// and then provide a type alias
/// ```rust,ignore
/// type A = RingValue<ABase>;
/// ```
/// 
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RingValue<R: RingBase> {
    ring: R
}

impl<R: RingBase> RingValue<R> {

    pub const fn from(value: R) -> Self {
        RingValue { ring: value }
    }

    pub fn from_ref<'a>(value: &'a R) -> &'a Self {
        unsafe { std::mem::transmute(value) }
    }

    pub fn into(self) -> R {
        self.ring
    }
}

impl<R: RingBase> RingStore for RingValue<R> {

    type Type = R;
    
    fn get_ring(&self) -> &R {
        &self.ring
    }
}

impl<R: RingBase + Default> Default for RingValue<R> {
    
    fn default() -> Self {
        Self::from(R::default())
    }
}

///
/// The second most basic [`crate::ring::RingStore`]. Similarly to 
/// [`crate::ring::RingValue`] it is just a no-op container.
/// 
/// # Why do we need this in addition to [`crate::ring::RingValue`]?
/// 
/// Before [`RingStore::from_ref()`] was added, this was important to
/// allow using a reference to a [`RingBase`] as [`RingStore`]. Since then,
/// it indeed has only a marginal importance, but note that it is currently
/// the only way of working with unsized rings (an admittedly pretty exotic
/// case).
/// 
#[repr(transparent)]
pub struct RingRef<'a, R: RingBase + ?Sized> {
    ring: &'a R
}

impl<'a, R: RingBase + ?Sized> Clone for RingRef<'a, R> {

    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: RingBase + ?Sized> Copy for RingRef<'a, R> {}

impl<'a, R: RingBase + ?Sized> RingRef<'a, R> {

    pub const fn new(value: &'a R) -> Self {
        RingRef { ring: value }
    }
}

impl<'a, R: RingBase + ?Sized> RingStore for RingRef<'a, R> {

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
#[cfg(test)]
use crate::impl_eq_based_self_iso;

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
    
    #[derive(Clone, PartialEq)]
    pub struct ABase;

    #[allow(unused)]
    #[derive(Clone)]
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

        fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
                where I::Type: IntegerRing
        {
            Some(ZZ.zero())
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
        
        fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
                where I::Type: IntegerRing
        {
            Some(ZZ.zero())
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

    impl<R: RingStore> CanIsoFromTo<BBase<R>> for BBase<R> 
        where R::Type: CanHomFrom<R::Type>
    {
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
    let b1: B<A> = RingValue { ring: BBase { base: a.clone() } };
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

    use crate::integer::{int_cast, BigIntRing};

    use super::*;

    pub fn test_hom_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
        where S::Type: CanHomFrom<R::Type>
    {
        let hom = to.can_hom(&from).unwrap();
        crate::homomorphism::generic_tests::test_homomorphism_axioms(hom, edge_case_elements);
    }

    pub fn test_iso_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
        where S::Type: CanIsoFromTo<R::Type>
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
            assert_el_eq!(ring, a, ring.get_ring().map_in_ref(ring.get_ring(), a, &hom));
            assert_el_eq!(ring, a, ring.get_ring().map_out(ring.get_ring(), ring.clone_el(a), &iso));
        }
    }

    pub fn test_hash_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: HashableElRing
    {
        let elements = edge_case_elements.collect::<Vec<_>>();

        // technically not required, but we should test hash inequality and this really should be true
        assert_ne!(ring.default_hash(&ring.one()), ring.default_hash(&ring.zero()));

        for a in &elements {
            for b in &elements {
                assert!(!ring.eq_el(a, b) || ring.default_hash(a) == ring.default_hash(b));
            }
        }

        for a in &elements {
            for b in &elements {
                let expr = ring.sub(ring.mul_ref_fst(a, ring.add_ref_fst(b, ring.one())), ring.mul_ref(a, b));
                assert!(ring.default_hash(a) == ring.default_hash(&expr));
            }
        }
    }

    pub fn test_ring_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) {
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

        // check characteristic
        let ZZbig = BigIntRing::RING;
        let char = ring.characteristic(&ZZbig).unwrap();
        
        if ZZbig.is_geq(&char, &ZZbig.power_of_two(7)) {
            assert_eq!(None, ring.characteristic(&StaticRing::<i8>::RING));
        }
        if ZZbig.is_geq(&char, &ZZbig.power_of_two(15)) {
            assert_eq!(None, ring.characteristic(&StaticRing::<i16>::RING));
        }
        if ZZbig.is_geq(&char, &ZZbig.power_of_two(31)) {
            assert_eq!(None, ring.characteristic(&StaticRing::<i32>::RING));
        }
        if ZZbig.is_geq(&char, &ZZbig.power_of_two(63)) {
            assert_eq!(None, ring.characteristic(&StaticRing::<i64>::RING));
        }
        if ZZbig.is_geq(&char, &ZZbig.power_of_two(127)) {
            assert_eq!(None, ring.characteristic(&StaticRing::<i128>::RING));
        }
        if ZZbig.is_lt(&char, &ZZbig.power_of_two(31)) {
            let char = int_cast(char, &StaticRing::<i32>::RING, &ZZbig);

            assert_el_eq!(ring, ring.zero(), ring.get_ring().from_int(char));
            
            if char == 0 {
                for i in 1..(1 << 10) {
                    assert!(!ring.is_zero(&ring.get_ring().from_int(i)));
                }
            } else {
                for i in 1..char {
                    assert!(!ring.is_zero(&ring.get_ring().from_int(i)));
                }
            }
        }

    }
}

#[test]
fn test_environment_binding() {
    assert!(EnvBindingStrength::Strongest > EnvBindingStrength::Power);
    assert!(EnvBindingStrength::Power > EnvBindingStrength::Product);
    assert!(EnvBindingStrength::Product > EnvBindingStrength::Sum);
    assert!(EnvBindingStrength::Sum > EnvBindingStrength::Weakest);
}