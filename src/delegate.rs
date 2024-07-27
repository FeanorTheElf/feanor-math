use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::homomorphism::*;
use crate::divisibility::DivisibilityRing;
use crate::rings::extension::FreeAlgebra;
use crate::rings::{zn::ZnRing, finite::FiniteRing};
use crate::integer::{IntegerRingStore, IntegerRing};

///
/// Trait to simplify implementing newtype-pattern for rings.
/// When you want to create a ring that just wraps another ring,
/// possibly adding some functionality, you can implement `DelegateRing`
/// instead of `RingBase`, and just provide how to map elements in the new
/// ring to the wrapped ring and vice versa.
/// 
/// # Example
/// ```
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
/// 
pub trait DelegateRing: PartialEq {

    type Base: ?Sized + RingBase;
    type Element;

    fn get_delegate(&self) -> &Self::Base;
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element;
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element;
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element;
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element;
    
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

    default fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_delegate().characteristic(ZZ)
    }
}

impl<R: DelegateRing + ?Sized> DivisibilityRing for R
    where R::Base: DivisibilityRing
{
    default fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(self.delegate_ref(lhs), self.delegate_ref(rhs))
            .map(|x| self.rev_delegate(x))
    }
}

pub struct DelegateZnRingElementsIter<'a, R: ?Sized>
    where R: DelegateRing, R::Base: FiniteRing
{
    ring: &'a R,
    base: <R::Base as FiniteRing>::ElementsIter<'a>
}

impl<'a, R: ?Sized> Clone for DelegateZnRingElementsIter<'a, R>
    where R: DelegateRing, R::Base: FiniteRing
{
    fn clone(&self) -> Self {
        Self { ring: self.ring, base: self.base.clone() }
    }
}

impl<'a, R: ?Sized> Iterator for DelegateZnRingElementsIter<'a, R>
    where R: DelegateRing, R::Base: FiniteRing
{
    type Item = <R as RingBase>::Element;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|x| self.ring.rev_delegate(x))
    }
}

impl<R: DelegateRing + ?Sized> FiniteRing for R
    where R::Base: FiniteRing
{
    type ElementsIter<'a> = DelegateZnRingElementsIter<'a, R>
        where R: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        DelegateZnRingElementsIter {
            ring: self,
            base: self.get_delegate().elements()
        }
    }
    
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <R as RingBase>::Element {
        self.element_cast(self.rev_delegate(self.get_delegate().random_element(rng)))
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_delegate().size(ZZ)
    }
}

impl<R: DelegateRing + ?Sized> ZnRing for R
    where R::Base: ZnRing, 
        Self: PrincipalIdealRing,
        R: CanHomFrom<<R::Base as ZnRing>::IntegerRingBase>
{
    type IntegerRingBase = <R::Base as ZnRing>::IntegerRingBase;
    type Integers = <R::Base as ZnRing>::Integers;

    fn integer_ring(&self) -> &Self::Integers {
        self.get_delegate().integer_ring()
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.get_delegate().modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.get_delegate().smallest_positive_lift(self.delegate(self.rev_element_cast(el)))
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.get_delegate().smallest_lift(self.delegate(self.rev_element_cast(el)))
    }

    fn from_int_promise_reduced(&self, x: El<Self::Integers>) -> Self::Element {
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

    fn canonical_gen(&self) -> Self::Element {
        self.rev_delegate(self.get_delegate().canonical_gen())
    }

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = El<Self::BaseRing>>
    {
        self.rev_delegate(self.get_delegate().from_canonical_basis(vec.map(|x| x)))
    }

    fn rank(&self) -> usize {
        self.get_delegate().rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        self.get_delegate().wrt_canonical_basis(self.delegate_ref(el))
    }
}
