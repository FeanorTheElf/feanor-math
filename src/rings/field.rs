use crate::delegate::DelegateRing;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::field::Field;
use crate::integer::IntegerRing;
use crate::ring::*;

use super::extension::FreeAlgebra;
use crate::homomorphism::*;

#[derive(Clone, Copy)]
pub struct AsFieldBase<R: DivisibilityRingStore> 
    where R::Type: DivisibilityRing
{
    base: R
}

impl<R> PartialEq for AsFieldBase<R>
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

#[allow(type_alias_bounds)]
pub type AsField<R: DivisibilityRingStore> = RingValue<AsFieldBase<R>>;

pub struct FieldEl<R: DivisibilityRingStore>(El<R>)
    where R::Type: DivisibilityRing;

impl<R: DivisibilityRingStore> Clone for FieldEl<R> 
    where El<R>: Clone,
        R::Type: DivisibilityRing
{
    fn clone(&self) -> Self {
        FieldEl(self.0.clone())
    }
}

impl<R: DivisibilityRingStore> Copy for FieldEl<R> 
    where El<R>: Copy,
        R::Type: DivisibilityRing
{}

impl<R: DivisibilityRingStore> AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    ///
    /// This function is not really unsafe, but users should be careful to only use
    /// it with rings that are fields. This cannot be checked in here, so must be checked
    /// by the caller.
    /// 
    pub fn promise_is_field(base: R) -> Self {
        Self { base }
    }

    pub fn unwrap_element(&self, el: <Self as RingBase>::Element) -> El<R> {
        el.0
    }
}

impl<R: DivisibilityRingStore> DelegateRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    type Element = FieldEl<R>;
    type Base = R::Type;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.0
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.0
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.0
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        FieldEl(el)
    }
}

impl<R: DivisibilityRingStore, S: DivisibilityRingStore> CanHomFrom<AsFieldBase<S>> for AsFieldBase<R> 
    where R::Type: DivisibilityRing + CanHomFrom<S::Type>,
        S::Type: DivisibilityRing
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFieldBase<S>) -> Option<Self::Homomorphism> {
        <R::Type as CanHomFrom<S::Type>>::has_canonical_hom(self.get_delegate(), from.get_delegate())
    }

    fn map_in(&self, from: &AsFieldBase<S>, el: FieldEl<S>, hom: &Self::Homomorphism) -> Self::Element {
        FieldEl(<R::Type as CanHomFrom<S::Type>>::map_in(self.get_delegate(), from.get_delegate(), el.0, hom))
    }
}

impl<R: DivisibilityRingStore, S: DivisibilityRingStore> CanonicalIso<AsFieldBase<S>> for AsFieldBase<R> 
    where R::Type: DivisibilityRing + CanonicalIso<S::Type>,
        S::Type: DivisibilityRing
{
    type Isomorphism = <R::Type as CanonicalIso<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<S>) -> Option<Self::Isomorphism> {
        <R::Type as CanonicalIso<S::Type>>::has_canonical_iso(self.get_delegate(), from.get_delegate())
    }

    fn map_out(&self, from: &AsFieldBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> FieldEl<S> {
        FieldEl(<R::Type as CanonicalIso<S::Type>>::map_out(self.get_delegate(), from.get_delegate(), el.0, iso))
    }
}

impl<R: DivisibilityRingStore> RingExtension for AsFieldBase<R> 
    where R::Type: DivisibilityRing + RingExtension
{
    type BaseRing = <R::Type as RingExtension>::BaseRing;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.get_delegate().base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.rev_delegate(self.get_delegate().from(x))
    }
}

impl<R: DivisibilityRingStore, S: IntegerRing + ?Sized> CanHomFrom<S> for AsFieldBase<R> 
    where R::Type: DivisibilityRing + CanHomFrom<S>
{
    type Homomorphism = <R::Type as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: S::Element, hom: &Self::Homomorphism) -> Self::Element {
        FieldEl(<R::Type as CanHomFrom<S>>::map_in(self.get_delegate(), from, el, hom))
    }
}

impl<R: DivisibilityRingStore> DivisibilityRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(&lhs.0, &rhs.0).map(FieldEl)
    }
}

impl<R: DivisibilityRingStore> PrincipalIdealRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }
}

impl<R: DivisibilityRingStore> EuclideanRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }

    fn euclidean_rem(&self, _: Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        self.zero()
    }
}

impl<R: DivisibilityRingStore> HashableElRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing + HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.get_delegate().hash(&el.0, h)
    }
}

impl<R: DivisibilityRingStore> Domain for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{}

impl<R: DivisibilityRingStore> Field for AsFieldBase<R>
    where R::Type: DivisibilityRing
{
    fn div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        FieldEl(self.get_delegate().checked_left_div(&lhs.0, &rhs.0).unwrap())
    }
}

impl<R: DivisibilityRingStore> FreeAlgebra for AsFieldBase<R> 
    where R::Type: DivisibilityRing + FreeAlgebra
{
    type VectorRepresentation<'a> = <R::Type as FreeAlgebra>::VectorRepresentation<'a>
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

#[cfg(test)]
use crate::rings::zn::zn_barett::Zn;
#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_canonical_hom_axioms_static_int() {
    let R = Zn::new(StaticRing::<i64>::RING, 17).as_field().ok().unwrap();
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, &R, 0..17);
}

#[test]
fn test_divisibility_axioms() {
    let R = Zn::new(StaticRing::<i64>::RING, 17).as_field().ok().unwrap();
    crate::divisibility::generic_tests::test_divisibility_axioms(&R, R.elements());
}

#[test]
fn test_canonical_hom_axioms_zn_barett() {
    let R = Zn::new(StaticRing::<i64>::RING, 17).as_field().ok().unwrap();
    crate::ring::generic_tests::test_hom_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
    crate::ring::generic_tests::test_iso_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
}