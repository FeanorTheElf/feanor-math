use crate::{ring::*, divisibility::DivisibilityRing, rings::zn::ZnRing, integer::{IntegerRingStore, IntegerRing}};

///
/// Trait to simplify implementing newtype-pattern for rings.
/// When you want to create a ring that just wraps another ring,
/// possibly adding some functionality, you can implement `DelegateRing`
/// instead of `RingBase`, and just provide how to map elements in the new
/// ring to the wrapped ring and vice versa.
/// 
pub trait DelegateRing: PartialEq {

    type Base: ?Sized + RingBase + SelfIso;
    type Element;

    fn get_delegate(&self) -> &Self::Base;
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element;
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element;
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element;
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element;
    
    fn postprocess_delegate_mut(&self, el: &mut Self::Element) {
        *el = self.rev_delegate(self.get_delegate().clone_el(self.delegate_ref(el)));
    }

    // necessary in some locations to satisfy the type system
    fn element_cast(&self, el: Self::Element) -> <Self as RingBase>::Element {
        el
    }

    // necessary in some locations to satisfy the type system
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

    fn mul_int(&self, lhs: Self::Element, rhs: i32) -> Self::Element {
        self.rev_delegate(self.get_delegate().mul_int(self.delegate(lhs), rhs))
    }
    
    fn pow_gen<S: IntegerRingStore>(&self, x: Self::Element, power: &El<S>, integers: S) -> Self::Element 
        where S::Type: IntegerRing
    {
        self.rev_delegate(self.get_delegate().pow_gen(self.delegate(x), power, integers))
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
    where R: DelegateRing, R::Base: ZnRing
{
    ring: &'a R,
    base: <R::Base as ZnRing>::ElementsIter<'a>
}

impl<'a, R: ?Sized> Iterator for DelegateZnRingElementsIter<'a, R>
    where R: DelegateRing, R::Base: ZnRing
{
    type Item = <R as RingBase>::Element;

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|x| self.ring.rev_delegate(x))
    }
}

impl<R: DelegateRing + ?Sized> ZnRing for R
    where R::Base: ZnRing, R: CanonicalHom<<R::Base as ZnRing>::IntegerRingBase> + SelfIso
{
    type IntegerRingBase = <R::Base as ZnRing>::IntegerRingBase;
    type Integers = <R::Base as ZnRing>::Integers;
    type ElementsIter<'a> = DelegateZnRingElementsIter<'a, R>
        where R: 'a;

    fn integer_ring(&self) -> &Self::Integers {
        self.get_delegate().integer_ring()
    }

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        DelegateZnRingElementsIter {
            ring: self,
            base: self.get_delegate().elements()
        }
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.get_delegate().modulus()
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <R as RingBase>::Element {
        self.element_cast(self.rev_delegate(self.get_delegate().random_element(rng)))
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.get_delegate().smallest_positive_lift(self.delegate(self.rev_element_cast(el)))
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.get_delegate().smallest_lift(self.delegate(self.rev_element_cast(el)))
    }
}

