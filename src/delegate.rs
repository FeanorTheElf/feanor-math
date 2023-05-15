use crate::ring::*;

///
/// Trait to simplify implementing newtype-pattern for rings.
/// When you want to create a ring that just wraps another ring,
/// possibly adding some functionality, you can implement `DelegateRing`
/// instead of `RingBase`, and just provide how to map elements in the new
/// ring to the wrapped ring and vice versa.
/// 
pub trait DelegateRing {

    type Base: ?Sized + RingBase;
    type Element;

    fn get_delegate(&self) -> &Self::Base;
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element;
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element;
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element;
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element;
}

impl<R: DelegateRing> RingBase for R {

    type Element = <Self as DelegateRing>::Element;

    default fn clone_el(&self, val: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().clone_el(self.delegate_ref(val)))
    }
    
    default fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().add_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs))
    }

    default fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().add_assign(self.delegate_mut(lhs), self.delegate(rhs))
    }

    default fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().sub_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs))
    }

    default fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().sub_self_assign(self.delegate_mut(lhs), self.delegate(rhs))
    }

    default fn sub_self_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().sub_self_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs))
    }

    default fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.get_delegate().negate_inplace(self.delegate_mut(lhs))
    }

    default fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.get_delegate().mul_assign(self.delegate_mut(lhs), self.delegate(rhs))
    }

    default fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.get_delegate().mul_assign_ref(self.delegate_mut(lhs), self.delegate_ref(rhs))
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
        self.get_delegate().sub_assign(self.delegate_mut(lhs), self.delegate(rhs))
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
}