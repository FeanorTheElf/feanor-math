use std::fmt::Debug;

use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::int_factor::is_prime_power;
use crate::algorithms::matmul::*;
use crate::reduce_lift::poly_eval::InterpolationBaseRing;
use crate::delegate::*;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::field::Field;
use crate::local::{PrincipalLocalRing, PrincipalLocalRingStore};
use crate::pid::*;
use crate::integer::IntegerRing;
use crate::ring::*;
use crate::homomorphism::*;
use super::field::{AsField, AsFieldBase};
use crate::rings::zn::*;

///
/// A wrapper around a ring that marks this ring to be a local principal ideal ring. 
/// 
/// The design is analogous to [`crate::rings::field::AsFieldBase`].
/// 
#[stability::unstable(feature = "enable")]
pub struct AsLocalPIRBase<R: DivisibilityRingStore> 
    where R::Type: DivisibilityRing
{
    base: R,
    max_ideal_gen: LocalPIREl<R>,
    nilpotent_power: Option<usize>
}

impl<R> Debug for AsLocalPIRBase<R>
    where R: RingStore,
        R::Type: DivisibilityRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AsLocalPIR({:?})", self.get_delegate())
    }
}

impl<R> Clone for AsLocalPIRBase<R>
    where R: DivisibilityRingStore + Clone,
        R::Type: DivisibilityRing
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            max_ideal_gen: self.clone_el(&self.max_ideal_gen),
            nilpotent_power: self.nilpotent_power
        }
    }
}

impl<R> Copy for AsLocalPIRBase<R>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        El<R>: Copy
{}

impl<R> PartialEq for AsLocalPIRBase<R>
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

///
/// [`RingStore`] for [`AsLocalPIRBase`].
/// 
#[stability::unstable(feature = "enable")]
pub type AsLocalPIR<R> = RingValue<AsLocalPIRBase<R>>;

#[stability::unstable(feature = "enable")]
pub struct LocalPIREl<R: DivisibilityRingStore>(El<R>)
    where R::Type: DivisibilityRing;

impl<R: DivisibilityRingStore> Clone for LocalPIREl<R> 
    where El<R>: Clone,
        R::Type: DivisibilityRing
{
    fn clone(&self) -> Self {
        LocalPIREl(self.0.clone())
    }
}

impl<R: DivisibilityRingStore> Copy for LocalPIREl<R> 
    where El<R>: Copy,
        R::Type: DivisibilityRing
{}

impl<R> AsLocalPIR<R> 
    where R: RingStore, 
        R::Type: ZnRing
{
    #[stability::unstable(feature = "enable")]
    pub fn from_zn(ring: R) -> Option<Self> {
        let (p, e) = is_prime_power(ring.integer_ring(), ring.modulus())?;
        let g = ring.can_hom(ring.integer_ring()).unwrap().map(p);
        Some(Self::from(AsLocalPIRBase::promise_is_local_pir(ring, g, Some(e))))
    }
}

impl<R> AsLocalPIR<R> 
    where R: RingStore, 
        R::Type: Field
{
    #[stability::unstable(feature = "enable")]
    pub fn from_field(ring: R) -> Self {
        let zero = ring.zero();
        Self::from(AsLocalPIRBase::promise_is_local_pir(ring, zero, Some(1)))
    }
}

impl<R> AsLocalPIR<R> 
    where R: RingStore, 
        R::Type: PrincipalLocalRing
{
    #[stability::unstable(feature = "enable")]
    pub fn from_localpir(ring: R) -> Self {
        let max_ideal_gen = ring.clone_el(ring.max_ideal_gen());
        let nilpotent_power = ring.nilpotent_power();
        Self::from(AsLocalPIRBase::promise_is_local_pir(ring, max_ideal_gen, nilpotent_power))
    }
}

impl<R> AsLocalPIR<R> 
    where R: RingStore, 
        R::Type: DivisibilityRing
{
    #[stability::unstable(feature = "enable")]
    pub fn from_as_field(ring: AsField<R>) -> Self {
        let ring = ring.into().unwrap_self();
        let zero = ring.zero();
        Self::from(AsLocalPIRBase::promise_is_local_pir(ring, zero, Some(1)))
    }
}

impl<R: DivisibilityRingStore> AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    #[stability::unstable(feature = "enable")]
    pub fn promise_is_local_pir(base: R, max_ideal_gen: El<R>, nilpotent_power: Option<usize>) -> Self {
        assert!(base.is_commutative());
        let max_ideal_gen = LocalPIREl(max_ideal_gen);
        Self { base, max_ideal_gen, nilpotent_power }
    }

    #[stability::unstable(feature = "enable")]
    pub fn unwrap_element(&self, el: <Self as RingBase>::Element) -> El<R> {
        el.0
    }

    #[stability::unstable(feature = "enable")]
    pub fn unwrap_self(self) -> R {
        self.base
    }
}

impl<R: DivisibilityRingStore> DelegateRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    type Element = LocalPIREl<R>;
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
        LocalPIREl(el)
    }
}

impl<R: DivisibilityRingStore> DelegateRingImplFiniteRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{}

impl<R: DivisibilityRingStore, S: DivisibilityRingStore> CanHomFrom<AsLocalPIRBase<S>> for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing + CanHomFrom<S::Type>,
        S::Type: DivisibilityRing
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsLocalPIRBase<S>) -> Option<Self::Homomorphism> {
        <R::Type as CanHomFrom<S::Type>>::has_canonical_hom(self.get_delegate(), from.get_delegate())
    }

    fn map_in(&self, from: &AsLocalPIRBase<S>, el: LocalPIREl<S>, hom: &Self::Homomorphism) -> Self::Element {
        LocalPIREl(<R::Type as CanHomFrom<S::Type>>::map_in(self.get_delegate(), from.get_delegate(), el.0, hom))
    }
}

impl<R: DivisibilityRingStore, S: DivisibilityRingStore> CanIsoFromTo<AsLocalPIRBase<S>> for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing + CanIsoFromTo<S::Type>,
        S::Type: DivisibilityRing
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsLocalPIRBase<S>) -> Option<Self::Isomorphism> {
        <R::Type as CanIsoFromTo<S::Type>>::has_canonical_iso(self.get_delegate(), from.get_delegate())
    }

    fn map_out(&self, from: &AsLocalPIRBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> LocalPIREl<S> {
        LocalPIREl(<R::Type as CanIsoFromTo<S::Type>>::map_out(self.get_delegate(), from.get_delegate(), el.0, iso))
    }
}

///
/// Necessary to potentially implement [`crate::rings::zn::ZnRing`].
/// 
impl<R: DivisibilityRingStore, S: IntegerRing + ?Sized> CanHomFrom<S> for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing + CanHomFrom<S>
{
    type Homomorphism = <R::Type as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: S::Element, hom: &Self::Homomorphism) -> Self::Element {
        LocalPIREl(<R::Type as CanHomFrom<S>>::map_in(self.get_delegate(), from, el, hom))
    }
}

impl<R: DivisibilityRingStore> DivisibilityRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(&lhs.0, &rhs.0).map(LocalPIREl)
    }
}

impl<R: DivisibilityRingStore> PrincipalIdealRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if let Some(e) = self.nilpotent_power {
            if self.is_zero(lhs) && self.is_zero(rhs) {
                return Some(self.one());
            } else if self.is_zero(lhs) {
                return Some(RingRef::new(self).pow(self.clone_el(self.max_ideal_gen()), e - self.valuation(rhs).unwrap()));
            } else {
                // the constraint `rhs * result = lhs` already fixes the evaluation of `result` uniquely
                return self.checked_left_div(lhs, rhs);
            }
        } else {
            // we are in a domain
            self.checked_left_div(lhs, rhs)
        }
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.checked_left_div(lhs, rhs).is_some() {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }

    fn create_elimination_matrix(&self, a: &Self::Element, b: &Self::Element) -> ([Self::Element; 4], Self::Element) {
        if let Some(quo) = self.checked_left_div(b, a) {
            (
                [self.one(), self.zero(), self.negate(quo), self.one()],
                self.clone_el(a)
            )
        } else {
            let quo = self.checked_left_div(a, b).unwrap();
            (
                [self.zero(), self.one(), self.one(), self.negate(quo)],
                self.clone_el(b)
            )
        }
    }
}

impl<R: DivisibilityRingStore> KaratsubaHint for AsLocalPIRBase<R>
    where R::Type: DivisibilityRing
{
    fn karatsuba_threshold(&self) -> usize {
        self.get_delegate().karatsuba_threshold()
    }
}

impl<R: DivisibilityRingStore> StrassenHint for AsLocalPIRBase<R>
    where R::Type: DivisibilityRing
{
    fn strassen_threshold(&self) -> usize {
        self.get_delegate().strassen_threshold()
    }
}

impl<R: DivisibilityRingStore> ComputeInnerProduct for AsLocalPIRBase<R>
    where R::Type: DivisibilityRing
{
    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        self.rev_delegate(self.get_delegate().inner_product(els.map(|(a, b)| (self.delegate(a), self.delegate(b)))))
    }

    fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.rev_delegate(self.get_delegate().inner_product_ref(els.map(|(a, b)| (self.delegate_ref(a), self.delegate_ref(b)))))
    }

    fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.rev_delegate(self.get_delegate().inner_product_ref_fst(els.map(|(a, b)| (self.delegate_ref(a), self.delegate(b)))))
    }
}

impl<R: DivisibilityRingStore> EuclideanRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        if let Some(quo) = self.checked_left_div(&lhs, rhs) {
            (quo, self.zero())
        } else {
            (self.zero(), lhs)
        }
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        self.valuation(val)
    }
}

impl<R: DivisibilityRingStore> PrincipalLocalRing for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing
{
    fn max_ideal_gen(&self) ->  &Self::Element {
        &self.max_ideal_gen
    }

    fn nilpotent_power(&self) -> Option<usize> {
        self.nilpotent_power
    }
}

impl<R: DivisibilityRingStore> Domain for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing + Domain
{}

impl<R> FromModulusCreateableZnRing for AsLocalPIRBase<RingValue<R>> 
    where R: DivisibilityRing + ZnRing + FromModulusCreateableZnRing
{
    fn from_modulus<F, E>(create_modulus: F) -> Result<Self, E>
        where F:FnOnce(&Self::IntegerRingBase) -> Result<El<Self::IntegerRing>, E> 
    {
        <R as FromModulusCreateableZnRing>::from_modulus(create_modulus).map(|ring| AsLocalPIR::from_zn(RingValue::from(ring)).unwrap().into())
    }
}

impl<R: DivisibilityRingStore> Field for AsLocalPIRBase<R> 
    where R::Type: DivisibilityRing + Field
{}

impl<R> InterpolationBaseRing for AsLocalPIRBase<R>
    where R: RingStore,
        R::Type: InterpolationBaseRing
{
    type ExtendedRingBase<'a> = <R::Type as InterpolationBaseRing>::ExtendedRingBase<'a>
        where Self: 'a;

    type ExtendedRing<'a> = <R::Type as InterpolationBaseRing>::ExtendedRing<'a>
        where Self: 'a;

    fn in_base<'a, S>(&self, ext_ring: S, el: El<S>) -> Option<Self::Element>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        self.get_delegate().in_base(ext_ring, el).map(|x| self.rev_delegate(x))
    }

    fn in_extension<'a, S>(&self, ext_ring: S, el: Self::Element) -> El<S>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        self.get_delegate().in_extension(ext_ring, self.delegate(el))
    }

    fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>) {
        self.get_delegate().interpolation_points(count)
    }
}

impl<R1, R2> CanHomFrom<AsFieldBase<R1>> for AsLocalPIRBase<R2>
    where R1: RingStore, R2: RingStore,
        R2::Type: CanHomFrom<R1::Type>,
        R1::Type: DivisibilityRing,
        R2::Type: DivisibilityRing
{
    type Homomorphism = <R2::Type as CanHomFrom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFieldBase<R1>) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from.get_delegate())
    }

    fn map_in(&self, from: &AsFieldBase<R1>, el: <AsFieldBase<R1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.get_delegate().map_in(from.get_delegate(), from.delegate(el), hom))
    }
}

impl<R1, R2> CanIsoFromTo<AsFieldBase<R1>> for AsLocalPIRBase<R2>
    where R1: RingStore, R2: RingStore,
        R2::Type: CanIsoFromTo<R1::Type>,
        R1::Type: DivisibilityRing,
        R2::Type: DivisibilityRing
{
    type Isomorphism = <R2::Type as CanIsoFromTo<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<R1>) -> Option<Self::Isomorphism> {
        self.get_delegate().has_canonical_iso(from.get_delegate())
    }

    fn map_out(&self, from: &AsFieldBase<R1>, el: Self::Element, iso: &Self::Isomorphism) -> <AsFieldBase<R1> as RingBase>::Element {
        from.rev_delegate(self.get_delegate().map_out(from.get_delegate(), self.delegate(el), iso))
    }
}

///
/// Implements the isomorphisms `S: CanHomFrom<AsFieldBase<RingStore<Type = R>>>` and 
/// `AsFieldBase<RingStore<Type = S>>: CanHomFrom<R>`.
/// 
/// For details, see [`crate::impl_field_wrap_unwrap_homs!`]
/// 
#[macro_export]
macro_rules! impl_localpir_wrap_unwrap_homs {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsLocalPIRStore, $($gen_args)*> CanHomFrom<$self_type_from> for $crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>
            where AsLocalPIRStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$self_type_from) -> Option<Self::Homomorphism> {
                use $crate::delegate::*;
                self.get_delegate().has_canonical_hom(from)
            }

            fn map_in(&self, from: &$self_type_from, el: <$self_type_from as $crate::ring::RingBase>::Element, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                use $crate::delegate::*;
                self.rev_delegate(self.get_delegate().map_in(from, el, hom))
            }
        }
        
        impl<AsLocalPIRStore, $($gen_args)*> CanHomFrom<$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>> for $self_type_to
            where AsLocalPIRStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>) -> Option<Self::Homomorphism> {
                use $crate::delegate::*;
                self.has_canonical_hom(from.get_delegate())
            }

            fn map_in(&self, from: &$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>, el: $crate::rings::local::LocalPIREl<AsLocalPIRStore>, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                use $crate::delegate::*;
                self.map_in(from.get_delegate(), from.delegate(el), hom)
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_localpir_wrap_unwrap_homs!{ <{}> $self_type_from, $self_type_to where }
    };
}

///
/// Implements the isomorphisms `S: CanIsoFromTo<AsLocalPIRBase<RingStore<Type = R>>>` and `AsLocalPIRBase<RingStore<Type = S>>: CanIsoFromTo<R>`.
/// 
/// For details, see [`crate::impl_field_wrap_unwrap_isos!`]
/// 
#[macro_export]
macro_rules! impl_localpir_wrap_unwrap_isos {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsLocalPIRStore, $($gen_args)*> CanIsoFromTo<$self_type_from> for $crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>
            where AsLocalPIRStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$self_type_from) -> Option<Self::Isomorphism> {
                use $crate::delegate::*;
                self.get_delegate().has_canonical_iso(from)
            }

            fn map_out(&self, from: &$self_type_from, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <$self_type_from as RingBase>::Element {
                use $crate::delegate::*;
                self.get_delegate().map_out(from, self.delegate(el), iso)
            }
        }
        
        impl<AsLocalPIRStore, $($gen_args)*> CanIsoFromTo<$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>> for $self_type_to
            where AsLocalPIRStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>) -> Option<Self::Isomorphism> {
                use $crate::delegate::*;
                self.has_canonical_iso(from.get_delegate())
            }

            fn map_out(&self, from: &$crate::rings::local::AsLocalPIRBase<AsLocalPIRStore>, el: <Self as RingBase>::Element, hom: &Self::Isomorphism) -> $crate::rings::local::LocalPIREl<AsLocalPIRStore> {
                use $crate::delegate::*;
                from.rev_delegate(self.map_out(from.get_delegate(), el, hom))
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_localpir_wrap_unwrap_isos!{ <{}> $self_type_from, $self_type_to where }
    };
}

#[cfg(test)]
use crate::rings::zn::zn_big::ZnGB;
#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use std::time::Instant;
#[cfg(test)]
use crate::algorithms::linsolve::LinSolveRing;
#[cfg(test)]
use crate::matrix::{OwnedMatrix, TransposableSubmatrix, TransposableSubmatrixMut};
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use super::extension::galois_field::GaloisField;

#[test]
fn test_canonical_hom_axioms_static_int() {
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 8)).unwrap();
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, &R, 0..8);
}

#[test]
fn test_divisibility_axioms() {
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 8)).unwrap();
    crate::divisibility::generic_tests::test_divisibility_axioms(&R, R.elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 8)).unwrap();
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 9)).unwrap();
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 17)).unwrap();
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
}

#[test]
fn test_canonical_hom_axioms_wrap_unwrap() {
    let R = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 8)).unwrap();
    crate::ring::generic_tests::test_hom_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
    crate::ring::generic_tests::test_iso_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
}

#[test]
fn test_checked_div_min() {
    let ring = AsLocalPIR::from_zn(ZnGB::new(StaticRing::<i64>::RING, 27)).unwrap();
    assert_el_eq!(&ring, ring.zero(), ring.checked_div_min(&ring.zero(), &ring.one()).unwrap());
    assert_el_eq!(&ring, ring.int_hom().map(9), ring.checked_div_min(&ring.zero(), &ring.int_hom().map(3)).unwrap());
    assert_el_eq!(&ring, ring.int_hom().map(3), ring.checked_div_min(&ring.zero(), &ring.int_hom().map(9)).unwrap());
    assert_el_eq!(&ring, ring.one(), ring.checked_div_min(&ring.zero(), &ring.zero()).unwrap());
}

#[test]
#[ignore]
fn test_solve_large_galois_ring() {
    let ring: AsLocalPIR<_> = GaloisField::new(17, 2048).get_ring().galois_ring(5);
    let mut matrix: OwnedMatrix<_> = OwnedMatrix::zero(2, 2, &ring);
    let mut rng = oorandom::Rand64::new(1);

    *matrix.at_mut(0, 0) = ring.random_element(|| rng.rand_u64());
    *matrix.at_mut(0, 1) = ring.random_element(|| rng.rand_u64());
    *matrix.at_mut(1, 1) = ring.random_element(|| rng.rand_u64());
    assert!(ring.is_unit(&ring.sub_ref(matrix.at(0, 1), matrix.at(1, 1))), "matrix generation failed, pick another seed");
    *matrix.at_mut(1, 0) = ring.clone_el(matrix.at(0, 0));

    let mut rhs: OwnedMatrix<_> = OwnedMatrix::zero(2, 1, &ring);
    *rhs.at_mut(0, 0) = ring.random_element(|| rng.rand_u64());
    *rhs.at_mut(0, 1) = ring.random_element(|| rng.rand_u64());

    let rhs = rhs;
    let matrix = matrix;
    let mut result: OwnedMatrix<_> = OwnedMatrix::zero(2, 1, &ring);

    let start = Instant::now();
    ring.get_ring().solve_right(matrix.clone_matrix(&ring).data_mut(), rhs.clone_matrix(&ring).data_mut(), result.data_mut(), Global).assert_solved();
    let end = Instant::now();
    println!("Solved over GR(17, 5, 2048) in {} ms", (end - start).as_millis());

    let mut product: OwnedMatrix<_> = OwnedMatrix::zero(2, 1, &ring);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(matrix.data()), TransposableSubmatrix::from(result.data()), TransposableSubmatrixMut::from(product.data_mut()), &ring);

    assert_matrix_eq!(ring, rhs, product);
}
