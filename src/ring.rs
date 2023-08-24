use std::ops::Deref;

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
/// // To use the ring through a RingStore, it is also required to implement CanonicalHom<Self>
/// // and CanonicalIso<Self>.
/// 
/// impl CanonicalHom<MyRingBase> for MyRingBase {
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
/// 
/// pub type MyRing = RingValue<MyRingBase>;
/// 
/// impl MyRingBase {
/// 
///     pub const RING: MyRing = RingValue::from(MyRingBase);
/// }
/// 
/// let ring = MyRingBase::RING;
/// assert!(ring.eq_el(
///     &ring.from_int(6), 
///     &ring.mul(ring.from_int(3), ring.from_int(2))
/// ));
/// ```
/// And here is the example from the Readme, for the finite binary field F2
/// ```
/// use feanor_math::assert_el_eq;
/// use feanor_math::ring::*;
/// 
/// #[derive(PartialEq)]
/// struct F2Base;
/// 
/// impl RingBase for F2Base {
///    
///     type Element = u8;
/// 
///     fn clone_el(&self, val: &Self::Element) -> Self::Element {
///         *val
///     }
/// 
///     fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
///         *lhs = (*lhs + rhs) % 2;
///     }
///     
///     fn negate_inplace(&self, lhs: &mut Self::Element) {
///         *lhs = (2 - *lhs) % 2;
///     }
/// 
///     fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
///         *lhs = (*lhs * rhs) % 2;
///     }
///     
///     fn from_int(&self, value: i32) -> Self::Element {
///         // make sure that we handle negative numbers correctly
///         (((value % 2) + 2) % 2) as u8
///     }
/// 
///     fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
///         // elements are always represented by 0 or 1
///         *lhs == *rhs
///     }
///     
///     fn is_commutative(&self) -> bool { true }
///     fn is_noetherian(&self) -> bool { true }
/// 
///     fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
///         write!(out, "{}", *value)
///     }
/// }
/// 
/// // To properly use a ring, in addition to RingBase we have to implement CanonicalHom<Self> and
/// // CanonicalIso<Self>. This ensures that the ring works well with the canonical ring mapping
/// // framework, that later allows us to use functions like `cast()` or `coerce()`.
/// // In practice, we might also want to add implementations like `CanonicalHom<I> where I: IntegerRing`
/// // or CanonicalIso<feanor_math::rings::zn::zn_static::ZnBase<2, true>>.
/// 
/// impl CanonicalHom<F2Base> for F2Base {
///     
///     type Homomorphism = ();
/// 
///     fn has_canonical_hom(&self, from: &Self) -> Option<Self::Homomorphism> {
///         // a canonical homomorphism F -> F exists for all rings F of type F2Base, as
///         // there is only one possible instance of F2Base
///         Some(())
///     }
/// 
///     fn map_in(&self, from: &Self, el: Self::Element, hom: &Self::Homomorphism) -> Self::Element {
///         el
///     }
/// }
/// 
/// impl CanonicalIso<F2Base> for F2Base {
///     
///     type Isomorphism = ();
/// 
///     fn has_canonical_iso(&self, from: &Self) -> Option<Self::Isomorphism> {
///         // a canonical isomorphism F -> F exists for all rings F of type F2Base, as
///         // there is only one possible instance of F2Base
///         Some(())
///     }
/// 
///     fn map_out(&self, from: &Self, el: Self::Element, hom: &Self::Homomorphism) -> Self::Element {
///         el
///     }
/// }
/// 
/// pub const F2: RingValue<F2Base> = RingValue::from(F2Base);
/// 
/// assert_el_eq!(&F2, &F2.from_int(1), &F2.add(F2.one(), F2.zero()));
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
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::zn::*;
/// 
/// let Z7 = zn_barett::Zn::new(StaticRing::<i64>::RING, 7);
/// let Z11 = zn_barett::Zn::new(StaticRing::<i64>::RING, 11);
/// let neg_one = Z7.from_int(-1);
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
    /// Computes `lhs = rhs - lhs`
    /// 
    fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.negate_inplace(lhs);
        self.add_assign(lhs, rhs);
    }

    ///
    /// Computes `lhs = rhs - lhs`
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
    
    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing,
            Self: SelfIso
    {
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

pub struct CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalHom<R::Type>
{
    from: R,
    to: S,
    data: <S::Type as CanonicalHom<R::Type>>::Homomorphism
}

impl<R, S> CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalHom<R::Type>
{
    pub fn map(&self, el: El<R>) -> El<S> {
        self.to.get_ring().map_in(self.from.get_ring(), el, &self.data)
    }

    pub fn map_ref(&self, el: &El<R>) -> El<S> {
        self.to.get_ring().map_in_ref(self.from.get_ring(), el, &self.data)
    }

    pub fn raw_hom(&self) -> &<S::Type as CanonicalHom<R::Type>>::Homomorphism {
        &self.data
    }

    pub fn domain(&self) -> &R {
        &self.from
    }

    pub fn codomain(&self) -> &S {
        &self.to
    }
}

pub struct CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalIso<R::Type>
{
    from: R,
    to: S,
    data: <S::Type as CanonicalHom<R::Type>>::Homomorphism,
    data_back: <S::Type as CanonicalIso<R::Type>>::Isomorphism
}

impl<R, S> CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalIso<R::Type>
{
    pub fn map(&self, el: El<R>) -> El<S> {
        self.to.get_ring().map_in(self.from.get_ring(), el, &self.data)
    }

    pub fn map_ref(&self, el: &El<R>) -> El<S> {
        self.to.get_ring().map_in_ref(self.from.get_ring(), el, &self.data)
    }

    pub fn map_back(&self, el: El<S>) -> El<R> {
        self.to.get_ring().map_out(self.from.get_ring(), el, &self.data_back)
    }

    pub fn domain(&self) -> &R {
        &self.from
    }

    pub fn raw_hom(&self) -> &<S::Type as CanonicalHom<R::Type>>::Homomorphism {
        &self.data
    }

    pub fn raw_iso(&self) -> &<S::Type as CanonicalIso<R::Type>>::Isomorphism {
        &self.data_back
    }

    pub fn codomain(&self) -> &S {
        &self.to
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
/// The next example is the one from the Readme
/// ```
/// use feanor_math::assert_el_eq;
/// use feanor_math::ring::*;
/// use feanor_math::primitive_int::*;
/// use feanor_math::rings::zn::zn_barett::*;
/// use feanor_math::rings::zn::*;
/// use feanor_math::rings::finite::*;
/// use feanor_math::algorithms;
///
/// use oorandom;
///
/// fn fermat_is_prime(n: i64) -> bool {
///     // the Fermat primality test is based on the observation that a^n == a mod n if n
///     // is a prime; On the other hand, if n is not prime, we hope that there are many
///     // a such that this is not the case. 
///     // Note that this is not always the case, and so more advanced primality tests should 
///     // be used in practice. This is just a proof of concept.
/// 
///     let ZZ = StaticRing::<i64>::RING;
///     let Zn = Zn::new(ZZ, n); // the ring Z/nZ
/// 
///     // check for 6 random a whether a^n == a mod n
///     let mut rng = oorandom::Rand64::new(0);
///     for _ in 0..6 {
///         let a = Zn.random_element(|| rng.rand_u64());
///         let a_n = Zn.pow(Zn.clone_el(&a), n as usize);
///         if !Zn.eq_el(&a, &a_n) {
///             return false;
///         }
///     }
///     return true;
/// }
/// 
/// assert!(algorithms::miller_rabin::is_prime(StaticRing::<i64>::RING, &91, 6) == fermat_is_prime(91));
/// ```
/// And here the generic version
/// ```
/// use feanor_math::ring::*;
/// use feanor_math::integer::*;
/// use feanor_math::rings::bigint::*;
/// use feanor_math::rings::zn::zn_barett::*;
/// use feanor_math::rings::zn::*;
/// use feanor_math::rings::finite::*;
/// use feanor_math::algorithms;
/// 
/// use oorandom;
/// 
/// fn fermat_is_prime<R>(ZZ: R, n: El<R>) -> bool 
///     where R: RingStore, R::Type: IntegerRing
/// {
///     // the Fermat primality test is based on the observation that a^n == a mod n if n
///     // is a prime; On the other hand, if n is not prime, we hope that there are many
///     // a such that this is not the case. 
///     // Note that this is not always the case, and so more advanced primality tests should 
///     // be used in practice. This is just a proof of concept.
/// 
///     // ZZ is not guaranteed to be Copy anymore, so use reference instead
///     let Zn = Zn::new(&ZZ, ZZ.clone_el(&n)); // the ring Z/nZ
/// 
///     // check for 6 random a whether a^n == a mod n
///     let mut rng = oorandom::Rand64::new(0);
///     for _ in 0..6 {
///         let a = Zn.random_element(|| rng.rand_u64());
///         // use a generic square-and-multiply powering function that works with any implementation
///         // of integers
///         let a_n = Zn.pow_gen(Zn.clone_el(&a), &n, &ZZ);
///         if !Zn.eq_el(&a, &a_n) {
///             return false;
///         }
///     }
///     return true;
/// }
/// 
/// // the miller-rabin primality test is implemented in feanor_math::algorithms, so we can
/// // check our implementation
/// let n = DefaultBigIntRing::RING.from_int(91);
/// assert!(algorithms::miller_rabin::is_prime(DefaultBigIntRing::RING, &n, 6) == fermat_is_prime(DefaultBigIntRing::RING, n));
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
pub trait RingStore {
    
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
    delegate!{ fn from_int(&self, value: i32) -> El<Self> }
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
    delegate!{ fn mul_assign_int(&self, lhs: &mut El<Self>, rhs: i32) -> () }
    delegate!{ fn mul_int(&self, lhs: El<Self>, rhs: i32) -> El<Self> }
    delegate!{ fn mul_int_ref(&self, lhs: &El<Self>, rhs: i32) -> El<Self> }
    
    fn coerce<S>(&self, from: &S, el: El<S>) -> El<Self>
        where S: RingStore, Self::Type: CanonicalHom<S::Type> 
    {
        self.get_ring().map_in(from.get_ring(), el, &self.get_ring().has_canonical_hom(from.get_ring()).unwrap())
    }

    fn coerce_ref<S>(&self, from: &S, el: &El<S>) -> El<Self>
        where S: RingStore, Self::Type: CanonicalHom<S::Type> 
    {
        self.get_ring().map_in_ref(from.get_ring(), el, &self.get_ring().has_canonical_hom(from.get_ring()).unwrap())
    }

    fn cast<S>(&self, to: &S, el: El<Self>) -> El<S>
        where S: RingStore, Self::Type: CanonicalIso<S::Type> 
    {
        self.get_ring().map_out(to.get_ring(), el, &self.get_ring().has_canonical_iso(to.get_ring()).unwrap())
    }

    fn into_can_hom<S>(self, from: S) -> Result<CanHom<S, Self>, (Self, S)>
        where Self: Sized, S: RingStore, Self::Type: CanonicalHom<S::Type>
    {
        match self.get_ring().has_canonical_hom(from.get_ring()) {
            Some(hom) => Ok(CanHom { from: from, to: self, data: hom }),
            None => Err((self, from))
        }
    }

    fn into_can_iso<S>(self, from: S) -> Result<CanIso<S, Self>, (Self, S)>
        where Self: Sized, S: RingStore, Self::Type: CanonicalIso<S::Type>
    {
        match (self.get_ring().has_canonical_hom(from.get_ring()), self.get_ring().has_canonical_iso(from.get_ring())) {
            (Some(hom), Some(iso)) => Ok(CanIso { from: from, to: self, data: hom, data_back: iso }),
            _ => Err((self, from))
        }
    }

    fn can_hom<'a, S>(&'a self, from: &'a S) -> Option<CanHom<&'a S, &'a Self>>
        where S: RingStore, Self::Type: CanonicalHom<S::Type>
    {
        self.into_can_hom(from).ok()
    }

    fn can_iso<'a, S>(&'a self, from: &'a S) -> Option<CanIso<&'a S, &'a Self>>
        where S: RingStore, Self::Type: CanonicalIso<S::Type>
    {
        self.into_can_iso(from).ok()
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

    delegate!{ fn from(&self, x: El<<Self::Type as RingExtension>::BaseRing>) -> El<Self> }
    delegate!{ fn from_ref(&self, x: &El<<Self::Type as RingExtension>::BaseRing>) -> El<Self> }
    delegate!{ fn mul_assign_base(&self, lhs: &mut El<Self>, rhs: &El<<Self::Type as RingExtension>::BaseRing>) -> () }
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
/// Trait for rings R that have a canonical homomorphism `S -> R`.
/// A ring homomorphism is expected to be unital.
/// 
/// # Exact requirements
/// 
/// Which homomorphisms are considered canonical is up to implementors,
/// as long as any diagram of canonical homomorphisms commutes. In
/// other words, if there are rings `R, S` and "intermediate rings"
/// `R1, ..., Rn` resp. `R1', ..., Rm'` such that there are canonical
/// homomorphisms
/// ```text
/// S -> R1 -> R2 -> ... -> Rn -> R
/// ```
/// and
/// ```text
/// S -> R1' -> R2' -> ... -> Rm' -> R
/// ```
/// then both homomorphism chains should yield same results on same
/// inputs.
/// 
/// If the canonical homomorphism might be an isomorphism, consider also
/// implementing [`CanonicalIso`].
/// 
/// # Example
/// 
/// Most integer rings support canonical homomorphisms between them.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// let R = StaticRing::<i64>::RING;
/// let S = DefaultBigIntRing::RING;
/// // on RingBase level
/// let hom = R.get_ring().has_canonical_hom(S.get_ring()).unwrap();
/// assert_eq!(8, R.get_ring().map_in(S.get_ring(), S.from_int(8), &hom));
/// // on RingStore level
/// assert_eq!(8, R.coerce(&S, S.from_int(8)));
/// ```
/// 
/// # Limitations
/// 
/// The rust constraints regarding conflicting impl make it, in some cases,
/// impossible to implement all the canonical homomorphisms that we would like.
/// This is true in particular, if the rings are highly generic, and build
/// on base rings. In this case, it should always be preferred to implement
/// `CanonicalIso` for rings that are "the same", and on the other hand not
/// to implement classical homomorphisms, like `ZZ -> R` which exists for any
/// ring R. In applicable cases, consider also implementing [`RingExtension`].
/// 
/// Because of this reason, implementing [`RingExtension`] also does not require
/// an implementation of `CanonicalHom<Self::BaseRing>`. Hence, if you as a user
/// miss a certain implementation of `CanonicalHom`, check whether there maybe
/// is a corresponding implementation of [`RingExtension`], or a member function.
/// 
/// # More examples
/// 
/// ## Integer rings
/// 
/// Basically, all given integer rings have canonical isomorphisms between each other.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// let Z_i8 = StaticRing::<i8>::RING;
/// let Z_i32 = StaticRing::<i32>::RING;
/// let Z_i128 = StaticRing::<i128>::RING;
/// let Z_big = DefaultBigIntRing::RING;
/// 
/// assert!(Z_i8.can_iso(&Z_i8).is_some());
/// assert!(Z_i8.can_iso(&Z_i32).is_some());
/// assert!(Z_i8.can_iso(&Z_i128).is_some());
/// assert!(Z_i8.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_i32.can_iso(&Z_i8).is_some());
/// assert!(Z_i32.can_iso(&Z_i32).is_some());
/// assert!(Z_i32.can_iso(&Z_i128).is_some());
/// assert!(Z_i32.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_i128.can_iso(&Z_i8).is_some());
/// assert!(Z_i128.can_iso(&Z_i32).is_some());
/// assert!(Z_i128.can_iso(&Z_i128).is_some());
/// assert!(Z_i128.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_big.can_iso(&Z_i8).is_some());
/// assert!(Z_big.can_iso(&Z_i32).is_some());
/// assert!(Z_big.can_iso(&Z_i128).is_some());
/// assert!(Z_big.can_iso(&Z_big).is_some());
/// 
/// // there are also some blanket implementations/trait bounds
/// fn from_i32<I: IntegerRingStore>(to: &I) where I::Type: IntegerRing {
///     to.can_hom(&StaticRing::<i32>::RING);
/// }
/// 
/// fn to_i32<I: IntegerRingStore>(from: &I) where I::Type: IntegerRing {
///     StaticRing::<i32>::RING.can_hom(from);
/// }
/// ```
/// Notably, the only blanket implementations are currently
/// 
/// ## Integer quotient rings `Z/nZ`
/// 
/// Due to conflicting implementations, only the most useful conversions
/// are implemented for `Z/nZ`.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_42;
/// # use feanor_math::rings::zn::zn_barett;
/// # use feanor_math::rings::zn::zn_rns;
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = DefaultBigIntRing::RING;
/// 
/// let Zn_barett_i128 = zn_barett::Zn::new(ZZ, 17 * 257);
/// let Zn_barett_big = zn_barett::Zn::new(ZZ_big, ZZ_big.from_int(17 * 257));
/// let Zn_std = zn_42::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::from_primes(ZZ_big, vec![17, 257]);
/// 
/// assert!(Zn_barett_i128.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_barett_i128.can_iso(&Zn_barett_big).is_some());
/// 
/// assert!(Zn_barett_big.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_barett_big.can_iso(&Zn_barett_big).is_some());
/// 
/// assert!(Zn_std.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_std.can_iso(&Zn_std).is_some());
/// 
/// assert!(Zn_rns.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_rns.can_iso(&Zn_barett_big).is_some());
/// assert!(Zn_rns.can_iso(&Zn_rns).is_some());
/// ```
/// Additionally, there are the projections `Z -> Z/nZ`.
/// They are all implemented, even though [`crate::rings::zn::ZnRing`] currently
/// only requires the projection from the "associated" integer ring.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::rings::zn::*;
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = DefaultBigIntRing::RING;
/// 
/// let Zn_barett_i128 = zn_barett::Zn::new(ZZ, 17 * 257);
/// let Zn_barett_big = zn_barett::Zn::new(ZZ_big, ZZ_big.from_int(17 * 257));
/// let Zn_std = zn_42::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::from_primes(ZZ_big, vec![17, 257]);
/// 
/// assert!(Zn_barett_i128.can_hom(&ZZ).is_some());
/// assert!(Zn_barett_i128.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_barett_big.can_hom(&ZZ).is_some());
/// assert!(Zn_barett_big.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_std.can_hom(&ZZ).is_some());
/// assert!(Zn_std.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_rns.can_hom(&ZZ).is_some());
/// assert!(Zn_rns.can_hom(&ZZ_big).is_some());
/// ```
/// 
/// ## Polynomial Rings
/// 
/// For the two provided univariate polynomial ring implementations, we have the isomorphisms
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::rings::poly::*;
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let P_dense = dense_poly::DensePolyRing::new(ZZ, "X");
/// let P_sparse = sparse_poly::SparsePolyRing::new(ZZ, "X");
/// 
/// assert!(P_dense.can_iso(&P_dense).is_some());
/// assert!(P_dense.can_iso(&P_sparse).is_some());
/// assert!(P_sparse.can_iso(&P_dense).is_some());
/// assert!(P_sparse.can_iso(&P_sparse).is_some());
/// ```
/// Unfortunately, the inclusions `R -> R[X]` are not implemented as canonical homomorphisms,
/// however provided by the functions of [`RingExtension`].
/// 
pub trait CanonicalHom<S>: RingBase
    where S: RingBase + ?Sized
{
    type Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism>;
    fn map_in(&self, from: &S, el: S::Element, hom: &Self::Homomorphism) -> Self::Element;

    fn map_in_ref(&self, from: &S, el: &S::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(from, from.clone_el(el), hom)
    }

    fn mul_assign_map_in(&self, from: &S, lhs: &mut Self::Element, rhs: S::Element, hom: &Self::Homomorphism) {
        self.mul_assign(lhs, self.map_in(from, rhs, hom));
    }

    fn mul_assign_map_in_ref(&self, from: &S, lhs: &mut Self::Element, rhs: &S::Element, hom: &Self::Homomorphism) {
        self.mul_assign(lhs, self.map_in_ref(from, rhs, hom));
    }
}

///
/// Trait for rings R that have a canonical isomorphism `S -> R`.
/// A ring homomorphism is expected to be unital.
/// 
/// # Exact requirements
/// 
/// Same as for [`CanonicalHom`], it is up to implementors to decide which
/// isomorphisms are canonical, as long as each diagram that contains
/// only canonical homomorphisms, canonical isomorphisms and their inverses
/// commutes.
/// In other words, if there are rings `R, S` and "intermediate rings"
/// `R1, ..., Rn` resp. `R1', ..., Rm'` such that there are canonical
/// homomorphisms `->` or isomorphisms `<~>` connecting them - e.g. like
/// ```text
/// S -> R1 -> R2 <~> R3 <~> R4 -> ... -> Rn -> R
/// ```
/// and
/// ```text
/// S <~> R1' -> R2' -> ... -> Rm' -> R
/// ```
/// then both chains should yield same results on same inputs.
/// 
/// Hence, it would be natural if the trait were symmetrical, i.e.
///  for any implementation `R: CanonicalIso<S>` there is also an
/// implementation `S: CanonicalIso<R>`. However, because of the trait
/// impl constraints of Rust, this is unpracticable and so we only
/// require the implementation `R: CanonicalHom<S>`.
/// 
pub trait CanonicalIso<S>: CanonicalHom<S>
    where S: RingBase + ?Sized
{
    type Isomorphism;

    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism>;
    fn map_out(&self, from: &S, el: Self::Element, iso: &Self::Isomorphism) -> S::Element;
}

pub trait SelfIso: CanonicalIso<Self> {}

impl<R: ?Sized + CanonicalIso<R>> SelfIso for R {}

pub trait RingExtension: RingBase {
    
    type BaseRing: RingStore;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing;
    fn from(&self, x: El<Self::BaseRing>) -> Self::Element;
    
    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        self.from(self.base_ring().get_ring().clone_el(x))
    }

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
/// same level, e.g. [`crate::ring::CanonicalHom`], [`crate::divisibility::DivisibilityRing`]),
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

    impl CanonicalHom<ABase> for ABase {

        type Homomorphism = ();
        
        fn has_canonical_hom(&self, _: &ABase) -> Option<()> {
            Some(())
        }

        fn map_in(&self, _: &ABase, el: <ABase as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

    impl CanonicalIso<ABase> for ABase {
        
        type Isomorphism = ();

        fn has_canonical_iso(&self, _: &ABase) -> Option<()> {
            Some(())
        }

        fn map_out(&self, _: &ABase, el: <ABase as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

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

    impl<R: RingStore> CanonicalHom<ABase> for BBase<R> {

        type Homomorphism = ();

        fn has_canonical_hom(&self, _: &ABase) -> Option<()> {
            Some(())
        }

        fn map_in(&self, _: &ABase, el: <ABase as RingBase>::Element, _: &()) -> Self::Element {
            el
        }
    }

    impl<R: RingStore, S: RingStore> CanonicalHom<BBase<S>> for BBase<R> 
        where R::Type: CanonicalHom<S::Type>
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
pub fn generic_test_canonical_hom_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
    where S::Type: CanonicalHom<R::Type>
{
    let hom = to.get_ring().has_canonical_hom(from.get_ring()).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();

    for a in &elements {
        for b in &elements {
            assert_el_eq!(&to,
                &to.add(to.get_ring().map_in_ref(from.get_ring(), a, &hom), to.get_ring().map_in_ref(from.get_ring(), b, &hom)),
                &to.get_ring().map_in(from.get_ring(), from.add_ref(a, b), &hom)
            );
            assert_el_eq!(&to,
                &to.mul(to.get_ring().map_in_ref(from.get_ring(), a, &hom), to.get_ring().map_in_ref(from.get_ring(), b, &hom)),
                &to.get_ring().map_in(from.get_ring(), from.mul_ref(a, b), &hom)
            );
        }
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_canonical_iso_axioms<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(from: R, to: S, edge_case_elements: I)
    where S::Type: CanonicalIso<R::Type>
{
    let hom = to.get_ring().has_canonical_hom(from.get_ring()).unwrap();
    let iso = to.get_ring().has_canonical_iso(from.get_ring()).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();

    for a in &elements {
        to.println(&to.get_ring().map_in_ref(from.get_ring(), a, &hom));
        assert_el_eq!(&from,
            a, 
            &to.get_ring().map_out(from.get_ring(), to.get_ring().map_in_ref(from.get_ring(), a, &hom), &iso)
        );
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_self_iso<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
    where R::Type: SelfIso
{
    let hom = ring.get_ring().has_canonical_hom(ring.get_ring()).unwrap();
    let iso = ring.get_ring().has_canonical_iso(ring.get_ring()).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();

    generic_test_canonical_hom_axioms(&ring, &ring, elements.iter().map(|x| ring.clone_el(x)));
    generic_test_canonical_iso_axioms(&ring, &ring, elements.iter().map(|x| ring.clone_el(x)));

    for a in &elements {
        assert_el_eq!(&ring, a, &ring.get_ring().map_in_ref(ring.get_ring(), a, &hom));
        assert_el_eq!(&ring, a, &ring.get_ring().map_out(ring.get_ring(), ring.clone_el(a), &iso));
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_ring_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
    where R::Type: SelfIso
{
    let elements = edge_case_elements.collect::<Vec<_>>();
    let zero = ring.zero();
    let one = ring.one();

    // check self-subtraction
    for a in &elements {
        assert_el_eq!(&ring, &zero, &ring.sub(ring.clone_el(a), ring.clone_el(a)));
    }

    // check identity elements
    for a in &elements {
        assert_el_eq!(&ring, &a, &ring.add(ring.clone_el(a), ring.clone_el(&zero)));
        assert_el_eq!(&ring, &a, &ring.mul(ring.clone_el(a), ring.clone_el(&one)));
    }

    // check commutativity
    for a in &elements {
        for b in &elements {
            assert_el_eq!(&ring,
                &ring.add(ring.clone_el(a), ring.clone_el(b)), 
                &ring.add(ring.clone_el(b), ring.clone_el(a))
            );
                
            if ring.is_commutative() {
                assert_el_eq!(&ring,
                    &ring.mul(ring.clone_el(a), ring.clone_el(b)), 
                    &ring.mul(ring.clone_el(b), ring.clone_el(a))
                );
            }
        }
    }

    // check associativity
    for a in &elements {
        for b in &elements {
            for c in &elements {
                assert_el_eq!(&ring,
                    &ring.add(ring.clone_el(a), ring.add(ring.clone_el(b), ring.clone_el(c))), 
                    &ring.add(ring.add(ring.clone_el(a), ring.clone_el(b)), ring.clone_el(c))
                );
                assert_el_eq!(&ring,
                    &ring.mul(ring.clone_el(a), ring.mul(ring.clone_el(b), ring.clone_el(c))), 
                    &ring.mul(ring.mul(ring.clone_el(a), ring.clone_el(b)), ring.clone_el(c))
                );
            }
        }
    }
    
    // check distributivity
    for a in &elements {
        for b in &elements {
            for c in &elements {
                assert_el_eq!(
                    &ring,
                    &ring.mul(ring.clone_el(a), ring.add(ring.clone_el(b), ring.clone_el(c))), 
                    &ring.add(ring.mul(ring.clone_el(a), ring.clone_el(b)), ring.mul(ring.clone_el(a), ring.clone_el(c)))
                );
                assert_el_eq!(&ring,
                    &ring.mul(ring.add(ring.clone_el(a), ring.clone_el(b)), ring.clone_el(c)), 
                    &ring.add(ring.mul(ring.clone_el(a), ring.clone_el(c)), ring.mul(ring.clone_el(b), ring.clone_el(c)))
                );
            }
        }
    }

    generic_test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
}