use std::marker::PhantomData;
use std::fmt::Debug;

use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::homomorphism::*;
use crate::divisibility::DivisibilityRing;
use crate::rings::extension::FreeAlgebra;
use crate::rings::{zn::ZnRing, finite::FiniteRing};
use crate::integer::{IntegerRingStore, IntegerRing};
use crate::serialization::SerializableElementRing;
use crate::specialization::*;

///
/// Trait to simplify implementing newtype-pattern for rings.
/// When you want to create a ring that just wraps another ring,
/// possibly adding some functionality, you can implement `DelegateRing`
/// instead of `RingBase`, and just provide how to map elements in the new
/// ring to the wrapped ring and vice versa.
/// 
/// # Conditional Implementations
/// 
/// Some special ring traits (e.g. [`DivisibilityRing`]) are immediately implemented 
/// for `R: DelegateRing` as soon as `R::Base: SpecialRingTrait`. However, this 
/// of course prevents any implementation fo [`DelegateRing`] to provide a custom 
/// implementation (except for specialization in some cases), due to conflicting 
/// trait impls. Hence, for other traits, we use marker traits to mark an implementation 
/// `R` of [`DelegateRing`] to also automatically implement a special ring trait as soon 
/// as `R::Base: SpecialRingTrait`. These cases are currently
///  - [`DelegateRingImplFiniteRing`] for automatic implementations of [`FiniteRing`] and
///    [`FiniteRingSpecializable`].
///  - [`DelegateRingImplEuclideanRing`] for automatic implementations of [`PrincipalIdealRing`]
///    and [`EuclideanRing`]
/// 
/// # Example
/// 
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// 
/// #[derive(PartialEq, Clone)]
/// struct MyI32Ring;
/// struct MyI32RingEl(i32);
/// 
/// impl DelegateRing for MyI32Ring {
/// 
///     type Base = StaticRingBase<i32>;
///     type Element = MyI32RingEl;
/// 
///     fn get_delegate(&self) -> &Self::Base {
///         StaticRing::<i32>::RING.get_ring()
///     }
/// 
///     fn delegate_ref<'a>(&self, MyI32RingEl(el): &'a MyI32RingEl) -> &'a i32 {
///         el
///     }
/// 
///     fn delegate_mut<'a>(&self, MyI32RingEl(el): &'a mut MyI32RingEl) -> &'a mut i32 {
///         el
///     }
/// 
///     fn delegate(&self, MyI32RingEl(el): MyI32RingEl) -> i32 {
///         el
///     }
/// 
///     fn postprocess_delegate_mut(&self, _: &mut MyI32RingEl) {
///         // sometimes it might be necessary to fix some data of `Self::Element`
///         // if the underlying `Self::Base::Element` was modified via `delegate_mut()`;
///         // this is not the case here, so leave empty
///     }
/// 
///     fn rev_delegate(&self, el: i32) -> MyI32RingEl {
///         MyI32RingEl(el)
///     }
/// }
/// 
/// // you will have to implement `CanIsoFromTo<Self>`
/// impl_eq_based_self_iso!{ MyI32Ring }
/// 
/// let ring = RingValue::from(MyI32Ring);
/// assert_el_eq!(ring, ring.int_hom().map(1), ring.one());
/// ```
/// An example when special ring traits are automatically implemented is 
/// given by the following.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// 
/// struct BoringRingWrapper<R>(R);
/// 
/// impl<R> PartialEq for BoringRingWrapper<R>
///     where R: RingStore
/// {
///     fn eq(&self, other: &Self) -> bool {
///         self.0.get_ring() == other.0.get_ring()
///     }
/// }
/// 
/// impl<R> DelegateRing for BoringRingWrapper<R>
///     where R: RingStore
/// {
///     type Base = R::Type;
///     type Element = El<R>;
/// 
///     fn get_delegate(&self) -> &Self::Base {
///         self.0.get_ring()
///     }
/// 
///     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
///     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
///     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
///     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// }
/// ```
/// [`DivisibilityRing`] is automatically implemented (but can be specialized):
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// #
/// # struct BoringRingWrapper<R>(R);
/// # 
/// # impl<R> PartialEq for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     fn eq(&self, other: &Self) -> bool {
/// #         self.0.get_ring() == other.0.get_ring()
/// #     }
/// # }
/// # 
/// # impl<R> DelegateRing for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     type Base = R::Type;
/// #     type Element = El<R>;
/// # 
/// #     fn get_delegate(&self) -> &Self::Base {
/// #         self.0.get_ring()
/// #     }
/// # 
/// #     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
/// #     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
/// #     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
/// #     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// # }
/// fn divide_in_wrapped_ring<R>(base_ring: R)
///     where R: RingStore,
///         R::Type: DivisibilityRing
/// {
///     let wrapped_ring = BoringRingWrapper(base_ring);
///     assert!(wrapped_ring.checked_div(&wrapped_ring.one(), &wrapped_ring.one()).is_some());
/// }
/// ```
/// [`FiniteRing`] for example is not automatically implemented:
/// ```rust,compile_fail
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// #
/// # impl<R> PartialEq for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     fn eq(&self, other: &Self) -> bool {
/// #         self.0.get_ring() == other.0.get_ring()
/// #     }
/// # }
/// #
/// # struct BoringRingWrapper<R>(R);
/// # 
/// # impl<R> DelegateRing for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     type Base = R::Type;
/// #     type Element = El<R>;
/// # 
/// #     fn get_delegate(&self) -> &Self::Base {
/// #         self.0.get_ring()
/// #     }
/// # 
/// #     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
/// #     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
/// #     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
/// #     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// # }
/// fn size_of_wrapped_ring<R>(base_ring: R)
///     where R: RingStore,
///         R::Type: FiniteRing
/// {
///     let wrapped_ring = BoringRingWrapper(base_ring);
///     assert!(wrapped_ring.size(BigIntRing::RING).is_some());
/// }
/// ```
/// But we can add a delegate-implementation of [`FiniteRing`] by adding the marker trait [`DelegateRingImplFiniteRing`]:
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// #
/// # impl<R> PartialEq for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     fn eq(&self, other: &Self) -> bool {
/// #         self.0.get_ring() == other.0.get_ring()
/// #     }
/// # }
/// # 
/// # struct BoringRingWrapper<R>(R);
/// # 
/// # impl<R> DelegateRing for BoringRingWrapper<R>
/// #     where R: RingStore
/// # {
/// #     type Base = R::Type;
/// #     type Element = El<R>;
/// # 
/// #     fn get_delegate(&self) -> &Self::Base {
/// #         self.0.get_ring()
/// #     }
/// # 
/// #     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
/// #     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
/// #     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
/// #     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// # }
/// impl<R> DelegateRingImplFiniteRing for BoringRingWrapper<R>
///     where R: RingStore
/// {}
/// 
/// fn size_of_wrapped_ring<R>(base_ring: R)
///     where R: RingStore,
///         R::Type: FiniteRing
/// {
///     let wrapped_ring = BoringRingWrapper(base_ring);
///     assert!(wrapped_ring.size(BigIntRing::RING).is_some());
/// }
/// ```
/// 
pub trait DelegateRing: PartialEq + Debug + Send + Sync {

    ///
    /// Type of the delegated-to ring.
    /// 
    type Base: ?Sized + RingBase;

    ///
    /// Type of elements of this ring. These should always wrap elements from the delegated-to ring,
    /// but may store additional data.
    /// 
    type Element: Send + Sync;

    ///
    /// Returns a reference to the delegated-to ring, which is used by all other default
    /// implementations to actually implement arithmetic operations.
    /// 
    fn get_delegate(&self) -> &Self::Base;
    
    ///
    /// Provides a reference to the delegated-to ring element stored in the given element from this ring.
    /// 
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element;
    
    ///
    /// Provides a mutable reference to the delegated-to ring element stored in the given element from this ring.
    /// 
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element;
    
    ///
    /// Creates an element of the delegated-to ring, representing the given element from this ring.
    /// 
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element;

    ///
    /// Creates an element of this ring, representing the given element from the delegated-to ring.
    /// 
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element;
    
    ///
    /// Called after every operation of the delegated-to ring that accepts a mutable reference
    /// (which is acquired using [`DelegateRing::delegate_mut()`]). 
    /// 
    /// This can be used to update additional data, that is stored for every element in 
    /// addition to the delegated-to ring element. In many cases, this can be empty.
    /// 
    fn postprocess_delegate_mut(&self, el: &mut Self::Element) {
        *el = self.rev_delegate(self.get_delegate().clone_el(self.delegate_ref(el)));
    }

    ///
    /// Necessary in some locations to satisfy the type system
    /// 
    fn element_cast(&self, el: Self::Element) -> <Self as RingBase>::Element {
        el
    }

    ///
    /// Necessary in some locations to satisfy the type system
    /// 
    fn rev_element_cast(&self, el: <Self as RingBase>::Element) -> Self::Element {
        el
    }

    ///
    /// Necessary in some locations to satisfy the type system
    /// 
    fn rev_element_cast_ref<'a>(&self, el: &'a <Self as RingBase>::Element) -> &'a Self::Element {
        el
    }

    ///
    /// Necessary in some locations to satisfy the type system
    /// 
    fn rev_element_cast_mut<'a>(&self, el: &'a mut <Self as RingBase>::Element) -> &'a mut Self::Element {
        el
    }
}

impl<R: DelegateRing + PartialEq + ?Sized> RingBase for R {

    type Element = <Self as DelegateRing>::Element;

    default fn clone_el(&self, val: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().clone_el(self.delegate_ref(val)))
    }
    
    default fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().add_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().add_assign(self.delegate_mut(lhs), self.delegate(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().sub_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().sub_self_assign(self.delegate_mut(lhs), self.delegate(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn sub_self_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().sub_self_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.get_delegate().negate_inplace(self.delegate_mut(lhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().mul_assign(self.delegate_mut(lhs), self.delegate(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().mul_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn square(&self, value: &mut Self::Element) {
        self.get_delegate().square(self.delegate_mut(value));
        self.postprocess_delegate_mut(value);
    }

    default fn zero(&self) -> Self::Element {
        self.rev_delegate(self.get_delegate().zero())
    }

    default fn one(&self) -> Self::Element {
        self.rev_delegate(self.get_delegate().one())
    }

    default fn neg_one(&self) -> Self::Element {
        self.rev_delegate(self.get_delegate().neg_one())
    }

    default fn from_int(&self, value: i32) -> Self::Element {
        self.rev_delegate(self.get_delegate().from_int(value))
    }

    default fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.get_delegate().eq_el(self.delegate_ref(lhs), self.delegate_ref(rhs))
    }

    default fn is_zero(&self, value: &Self::Element) -> bool {
        self.get_delegate().is_zero(self.delegate_ref(value))
    }

    default fn is_one(&self, value: &Self::Element) -> bool {
        self.get_delegate().is_one(self.delegate_ref(value))
    }

    default fn is_neg_one(&self, value: &Self::Element) -> bool {
        self.get_delegate().is_neg_one(self.delegate_ref(value))
    }

    default fn is_commutative(&self) -> bool {
        self.get_delegate().is_commutative()
    }

    default fn is_noetherian(&self) -> bool {
        self.get_delegate().is_noetherian()
    }

    default fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.get_delegate().dbg(self.delegate_ref(value), out)
    }

    default fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        self.get_delegate().dbg_within(self.delegate_ref(value), out, env)
    }

    default fn negate(&self, value: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().negate(self.delegate(value)))
    }
    
    default fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().sub_assign(self.delegate_mut(lhs), self.delegate(rhs));
        self.postprocess_delegate_mut(lhs);
    }

    default fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().add_ref(self.delegate_ref(lhs), self.delegate_ref(rhs)))
    }

    default fn add_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().add_ref_fst(self.delegate_ref(lhs), self.delegate(rhs)))
    }

    default fn add_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().add_ref_snd(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    default fn add(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().add(self.delegate(lhs), self.delegate(rhs)))
    }

    default fn sub_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().sub_ref(self.delegate_ref(lhs), self.delegate_ref(rhs)))
    }

    default fn sub_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().sub_ref_fst(self.delegate_ref(lhs), self.delegate(rhs)))
    }

    default fn sub_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().sub_ref_snd(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    default fn sub(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().sub(self.delegate(lhs), self.delegate(rhs)))
    }

    default fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_ref(self.delegate_ref(lhs), self.delegate_ref(rhs)))
    }

    default fn mul_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_ref_fst(self.delegate_ref(lhs), self.delegate(rhs)))
    }

    default fn mul_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_ref_snd(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    default fn mul(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul(self.delegate(lhs), self.delegate(rhs)))
    }
    
    default fn is_approximate(&self) -> bool { self.get_delegate().is_approximate() }

    default fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        self.get_delegate().mul_assign_int(self.delegate_mut(lhs), rhs);
        self.postprocess_delegate_mut(lhs);
    }

    default fn mul_int(&self, lhs: Self::Element, rhs: i32) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_int(self.delegate(lhs), rhs))
    }

    default fn mul_int_ref(&self, lhs: &Self::Element, rhs: i32) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_int_ref(self.delegate_ref(lhs), rhs))
    }
    
    default fn pow_gen<S: IntegerRingStore>(&self, x: Self::Element, power: &El<S>, integers: S) -> Self::Element 
        where S::Type: IntegerRing
    {
        self.rev_delegate(self.get_delegate().pow_gen(self.delegate(x), power, integers))
    }

    default fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_delegate().characteristic(ZZ)
    }
}

impl<R: DelegateRing + ?Sized> DivisibilityRing for R
    where R::Base: DivisibilityRing
{
    type PreparedDivisorData = <R::Base as DivisibilityRing>::PreparedDivisorData;

    default fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(self.delegate_ref(lhs), self.delegate_ref(rhs))
            .map(|x| self.rev_delegate(x))
    }

    default fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        self.get_delegate().balance_factor(elements.map(|x| self.delegate_ref(x))).map(|c| self.rev_delegate(c))
    }

    fn prepare_divisor(&self, x: &Self::Element) -> Self::PreparedDivisorData {
        self.get_delegate().prepare_divisor(self.delegate_ref(x))
    }

    default fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> Option<Self::Element> {
        self.get_delegate().checked_left_div_prepared(self.delegate_ref(lhs), self.delegate_ref(rhs), rhs_prep)
            .map(|res| self.rev_delegate(res))
    }

    default fn divides_left_prepared(&self, lhs: &Self::Element, rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> bool {
        self.get_delegate().divides_left_prepared(self.delegate_ref(lhs), self.delegate_ref(rhs), rhs_prep)
    }
}

impl<R: DelegateRing + ?Sized> SerializableElementRing for R
    where R::Base: DivisibilityRing + SerializableElementRing
{
    default fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        self.get_delegate().serialize(self.delegate_ref(el), serializer)
    }

    default fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: serde::Deserializer<'de>
    {
        self.get_delegate().deserialize(deserializer).map(|x| self.rev_delegate(x))
    }
}

///
/// Iterator over all elements of a finite [`DelegateRing`].
/// 
pub struct DelegateFiniteRingElementsIter<'a, R: ?Sized>
    where R: DelegateRing, R::Base: FiniteRing
{
    ring: &'a R,
    base: <R::Base as FiniteRing>::ElementsIter<'a>
}

impl<'a, R: ?Sized> Clone for DelegateFiniteRingElementsIter<'a, R>
    where R: DelegateRing, R::Base: FiniteRing
{
    fn clone(&self) -> Self {
        Self { ring: self.ring, base: self.base.clone() }
    }
}

impl<'a, R: ?Sized> Iterator for DelegateFiniteRingElementsIter<'a, R>
    where R: DelegateRing, R::Base: FiniteRing
{
    type Item = <R as RingBase>::Element;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|x| self.ring.rev_delegate(x))
    }
}

///
/// Marks a [`DelegateRing`] `R` to be considered in the blanket implementation
/// `R: FiniteRing where R::Base: FiniteRing`.
/// 
/// We don't want to implement `R: FiniteRing` for any `DelegateRing` `R`, since
/// some ring newtypes want to have control of when the ring is [`FiniteRing`].
/// 
pub trait DelegateRingImplFiniteRing: DelegateRing {}

impl<R: DelegateRingImplFiniteRing + ?Sized> FiniteRing for R
    where R::Base: FiniteRing
{
    type ElementsIter<'a> = DelegateFiniteRingElementsIter<'a, R>
        where R: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        DelegateFiniteRingElementsIter {
            ring: self,
            base: self.get_delegate().elements()
        }
    }
    
    default fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <R as RingBase>::Element {
        self.element_cast(self.rev_delegate(self.get_delegate().random_element(rng)))
    }

    default fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_delegate().size(ZZ)
    }
}

impl<R: DelegateRing + ?Sized> HashableElRing for R 
    where R::Base: HashableElRing
{
    default fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.get_delegate().hash(self.delegate_ref(el), h)
    }
}

///
/// Marks a [`DelegateRing`] `R` to be considered in the blanket implementation
/// `R: EuclideanRing where R::Base: EuclideanRing` and 
/// `R: PrincipalIdealRing where R::Base: PrincipalIdealRing`.
/// 
/// We don't want to implement `R: EuclideanRing` for any `DelegateRing` `R`, since
/// some ring newtypes want to have control of when the ring is [`EuclideanRing`].
/// 
pub trait DelegateRingImplEuclideanRing: DelegateRing {}

impl<R: DelegateRingImplEuclideanRing + ?Sized> PrincipalIdealRing for R
    where R::Base: PrincipalIdealRing
{
    default fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_div_min(self.delegate_ref(lhs), self.delegate_ref(rhs)).map(|res| self.rev_delegate(res))
    }

    default fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self.get_delegate().extended_ideal_gen(self.delegate_ref(self.rev_element_cast_ref(lhs)), self.delegate_ref(self.rev_element_cast_ref(rhs)));
        return (self.element_cast(self.rev_delegate(s)), self.element_cast(self.rev_delegate(t)), self.element_cast(self.rev_delegate(d)));
    }

    default fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.element_cast(self.rev_delegate(self.get_delegate().ideal_gen(self.delegate_ref(self.rev_element_cast_ref(lhs)), self.delegate_ref(self.rev_element_cast_ref(rhs)))))
    }
}

impl<R: DelegateRingImplEuclideanRing + ?Sized> EuclideanRing for R
    where R::Base: EuclideanRing
{
    default fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        let (q, r) = self.get_delegate().euclidean_div_rem(self.delegate(lhs), self.delegate_ref(rhs));
        return (self.rev_delegate(q), self.rev_delegate(r));
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        self.get_delegate().euclidean_deg(self.delegate_ref(val))
    }

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().euclidean_div(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().euclidean_rem(self.delegate(lhs), self.delegate_ref(rhs)))
    }
}

impl<R> FiniteRingSpecializable for R
    where R: DelegateRingImplFiniteRing + ?Sized,
        R::Base: FiniteRingSpecializable
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        struct OpWrapper<R, O>
            where R: DelegateRingImplFiniteRing + ?Sized,
                R::Base: FiniteRingSpecializable,
                O: FiniteRingOperation<R>
        {
            operation: O,
            ring: PhantomData<Box<R>>
        }

        impl<R, O> FiniteRingOperation<R::Base> for OpWrapper<R, O>
            where R: DelegateRingImplFiniteRing + ?Sized,
                R::Base: FiniteRingSpecializable,
                O: FiniteRingOperation<R>
        {
            type Output = O::Output;
            fn execute(self) -> Self::Output where R::Base: FiniteRing {
                self.operation.execute()
            }
            fn fallback(self) -> Self::Output {
                self.operation.fallback()
            }
        }

        <R::Base as FiniteRingSpecializable>::specialize(OpWrapper { operation: op, ring: PhantomData })
    }
}

impl<R: DelegateRingImplFiniteRing + ?Sized> ZnRing for R
    where R::Base: ZnRing, 
        Self: PrincipalIdealRing,
        R: CanHomFrom<<R::Base as ZnRing>::IntegerRingBase>
{
    type IntegerRingBase = <R::Base as ZnRing>::IntegerRingBase;
    type IntegerRing = <R::Base as ZnRing>::IntegerRing;

    default fn integer_ring(&self) -> &Self::IntegerRing {
        self.get_delegate().integer_ring()
    }

    default fn modulus(&self) -> &El<Self::IntegerRing> {
        self.get_delegate().modulus()
    }

    default fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.get_delegate().smallest_positive_lift(self.delegate(self.rev_element_cast(el)))
    }

    default fn smallest_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.get_delegate().smallest_lift(self.delegate(self.rev_element_cast(el)))
    }

    default fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        self.element_cast(self.rev_delegate(self.get_delegate().from_int_promise_reduced(x)))
    }
}

impl<R> RingExtension for R 
    where R: DelegateRing,
        R::Base: RingExtension
{
    type BaseRing = <R::Base as RingExtension>::BaseRing;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.get_delegate().base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.rev_delegate(self.get_delegate().from(x))
    }
}

impl<R> FreeAlgebra for R 
    where R: DelegateRing,
        <R as DelegateRing>::Base: FreeAlgebra
{
    type VectorRepresentation<'a> = <<R as DelegateRing>::Base as FreeAlgebra>::VectorRepresentation<'a>
        where Self: 'a;

    default fn canonical_gen(&self) -> Self::Element {
        self.rev_delegate(self.get_delegate().canonical_gen())
    }

    default fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>,
            V::IntoIter: DoubleEndedIterator
    {
        self.rev_delegate(self.get_delegate().from_canonical_basis(vec.into_iter().map(|x| x)))
    }

    default fn rank(&self) -> usize {
        self.get_delegate().rank()
    }

    default fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        self.get_delegate().wrt_canonical_basis(self.delegate_ref(el))
    }
}

#[stability::unstable(feature = "enable")]
pub struct WrapHom<'a, R>
    where R: DelegateRing
{
    to: RingRef<'a, R>,
    from: RingRef<'a, R::Base>
}

impl<'a, R> WrapHom<'a, R>
    where R: DelegateRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'a R) -> Self {
        Self {
            to: RingRef::new(ring),
            from: RingRef::new(ring.get_delegate())
        }
    }
}

impl<'a, R> Homomorphism<<R as DelegateRing>::Base, R> for WrapHom<'a, R>
    where R: DelegateRing
{
    type DomainStore = RingRef<'a, <R as DelegateRing>::Base>;
    type CodomainStore = RingRef<'a, R>;

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.from
    }

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to
    }

    fn map(&self, x: <<R as DelegateRing>::Base as RingBase>::Element) -> <R as RingBase>::Element {
        self.to.get_ring().element_cast(self.to.get_ring().rev_delegate(x))
    }
}

#[stability::unstable(feature = "enable")]
pub struct UnwrapHom<'a, R>
    where R: DelegateRing
{
    from: RingRef<'a, R>,
    to: RingRef<'a, R::Base>
}

impl<'a, R> UnwrapHom<'a, R>
    where R: DelegateRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'a R) -> Self {
        Self {
            from: RingRef::new(ring),
            to: RingRef::new(ring.get_delegate())
        }
    }
}

impl<'a, R> Homomorphism<R, <R as DelegateRing>::Base> for UnwrapHom<'a, R>
    where R: DelegateRing
{
    type DomainStore = RingRef<'a, R>;
    type CodomainStore = RingRef<'a, <R as DelegateRing>::Base>;

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.from
    }

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to
    }

    fn map(&self, x: <R as RingBase>::Element) -> <<R as DelegateRing>::Base as RingBase>::Element {
        self.from.get_ring().delegate(self.from.get_ring().rev_element_cast(x))
    }
}

