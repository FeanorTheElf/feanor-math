use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::matmul::{ComputeInnerProduct, StrassenHint};
use crate::compute_locally::{InterpolationBaseRing, InterpolationBaseRingStore};
use crate::delegate::*;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::local::PrincipalLocalRing;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::field::*;
use crate::integer::IntegerRing;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::zn::FromModulusCreateableZnRing;
use crate::rings::zn::*;
use crate::specialization::FiniteRingSpecializable;
use super::local::AsLocalPIRBase;
use crate::primitive_int::*;

///
/// A wrapper around a ring that marks this ring to be a perfect field. In particular,
/// the functions provided by [`DivisibilityRing`] will be used to provide
/// field-like division for the wrapped ring.
///
pub struct AsFieldBase<R: DivisibilityRingStore> 
    where R::Type: DivisibilityRing
{
    base: R,
    zero: FieldEl<R>
}

impl<R> PerfectField for AsFieldBase<R>
    where R: RingStore,
        R::Type: DivisibilityRing
{}

impl<R> Clone for AsFieldBase<R>
    where R: DivisibilityRingStore + Clone,
        R::Type: DivisibilityRing
{
    fn clone(&self) -> Self {
        Self { zero: FieldEl(self.base.zero()), base: self.base.clone() }
    }
}

impl<R> Copy for AsFieldBase<R>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        El<R>: Copy
{}

impl<R> PartialEq for AsFieldBase<R>
    where R: DivisibilityRingStore,
        R::Type: DivisibilityRing
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

///
/// [`RingStore`] for [`AsFieldBase`].
/// 
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
    /// Assumes that the given ring is a perfect field, and wraps it in [`AsFieldBase`].
    /// 
    /// This function is not really unsafe, but users should be careful to only use
    /// it with rings that are perfect fields. This cannot be checked in here, so must 
    /// be checked by the caller.
    /// 
    pub fn promise_is_perfect_field(base: R) -> Self {
        Self { zero: FieldEl(base.zero()), base }
    }

    ///
    /// Assumes that the given ring is a field and checks whether it is perfect. If this is the case,
    /// the function wraps it in [`AsFieldBase`]. 
    ///
    /// This function is not really unsafe, but users should be careful to only use
    /// it with rings that are fields. This cannot be checked in here, so must be checked
    /// by the caller.
    /// 
    /// Note that this function does check that the passed field is perfect. However,
    /// that check is too conservative, and might fail on some perfect fields.
    /// If you are sure that the field is also perfect, consider using [`AsFieldBase::promise_is_perfect_field()`]
    /// instead.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn promise_is_field(base: R) -> Result<Self, R>
        where R::Type: FiniteRingSpecializable
    {
        let characteristic = base.characteristic(&StaticRing::<i64>::RING);
        if characteristic.is_some() && characteristic.unwrap() == 0 {
            return Ok(Self::promise_is_perfect_field(base));
        } else if <R::Type as FiniteRingSpecializable>::is_finite_ring() {
            return Ok(Self::promise_is_perfect_field(base));
        } else {
            return Err(base);
        }
    }

    pub fn unwrap_element(&self, el: <Self as RingBase>::Element) -> El<R> {
        el.0
    }

    ///
    /// Removes the wrapper, returning the underlying base field.
    /// 
    pub fn unwrap_self(self) -> R {
        self.base
    }
}

impl<R: DivisibilityRingStore> AsFieldBase<R> 
    where R::Type: PerfectField
{
    ///
    /// Creates a new [`AsFieldBase`] from a ring that is already known to be a
    /// perfect field.
    /// 
    pub fn from_field(base: R) -> Self {
        Self::promise_is_perfect_field(base)
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

impl<R: DivisibilityRingStore> DelegateRingImplFiniteRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{}

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

impl<R: DivisibilityRingStore, S: DivisibilityRingStore> CanIsoFromTo<AsFieldBase<S>> for AsFieldBase<R> 
    where R::Type: DivisibilityRing + CanIsoFromTo<S::Type>,
        S::Type: DivisibilityRing
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<S>) -> Option<Self::Isomorphism> {
        <R::Type as CanIsoFromTo<S::Type>>::has_canonical_iso(self.get_delegate(), from.get_delegate())
    }

    fn map_out(&self, from: &AsFieldBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> FieldEl<S> {
        FieldEl(<R::Type as CanIsoFromTo<S::Type>>::map_out(self.get_delegate(), from.get_delegate(), el.0, iso))
    }
}

///
/// Necessary to potentially implement [`crate::rings::zn::ZnRing`].
/// 
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

impl<R1, R2> CanHomFrom<AsLocalPIRBase<R1>> for AsFieldBase<R2>
    where R1: RingStore, R2: RingStore,
        R2::Type: CanHomFrom<R1::Type>,
        R1::Type: DivisibilityRing,
        R2::Type: DivisibilityRing
{
    type Homomorphism = <R2::Type as CanHomFrom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsLocalPIRBase<R1>) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from.get_delegate())
    }

    fn map_in(&self, from: &AsLocalPIRBase<R1>, el: <AsLocalPIRBase<R1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.get_delegate().map_in(from.get_delegate(), from.delegate(el), hom))
    }
}

impl<R1, R2> CanIsoFromTo<AsLocalPIRBase<R1>> for AsFieldBase<R2>
    where R1: RingStore, R2: RingStore,
        R2::Type: CanIsoFromTo<R1::Type>,
        R1::Type: DivisibilityRing,
        R2::Type: DivisibilityRing
{
    type Isomorphism = <R2::Type as CanIsoFromTo<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsLocalPIRBase<R1>) -> Option<Self::Isomorphism> {
        self.get_delegate().has_canonical_iso(from.get_delegate())
    }

    fn map_out(&self, from: &AsLocalPIRBase<R1>, el: Self::Element, iso: &Self::Isomorphism) -> <AsLocalPIRBase<R1> as RingBase>::Element {
        from.rev_delegate(self.get_delegate().map_out(from.get_delegate(), self.delegate(el), iso))
    }
}

impl<R: DivisibilityRingStore> DivisibilityRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(&lhs.0, &rhs.0).map(FieldEl)
    }
}

impl<R: RingStore> PrincipalIdealRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div(lhs, rhs)
    }
    
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }
}

impl<R: DivisibilityRingStore> PrincipalLocalRing for AsFieldBase<R> 
    where R::Type: DivisibilityRing
{
    fn max_ideal_gen(&self) ->  &Self::Element {
        &self.zero
    }

    fn nilpotent_power(&self) -> Option<usize> {
        Some(1)
    }

    fn valuation(&self, x: &Self::Element) -> Option<usize> {
        if self.is_zero(x) {
            return None;
        } else {
            return Some(0);
        }
    }
}

impl<R> InterpolationBaseRing for AsFieldBase<R>
    where R: InterpolationBaseRingStore,
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

impl<R: DivisibilityRingStore> KaratsubaHint for AsFieldBase<R>
    where R::Type: DivisibilityRing
{
    fn karatsuba_threshold(&self) -> usize {
        self.get_delegate().karatsuba_threshold()
    }
}

impl<R: DivisibilityRingStore> ComputeInnerProduct for AsFieldBase<R>
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

impl<R: DivisibilityRingStore> StrassenHint for AsFieldBase<R>
    where R::Type: DivisibilityRing
{
    fn strassen_threshold(&self) -> usize {
        self.get_delegate().strassen_threshold()
    }
}

impl<R> FromModulusCreateableZnRing for AsFieldBase<RingValue<R>> 
    where R: DivisibilityRing + ZnRing + FromModulusCreateableZnRing
{
    fn create<F, E>(create_modulus: F) -> Result<Self, E>
        where F:FnOnce(&Self::IntegerRingBase) -> Result<El<Self::IntegerRing>, E> 
    {
        <R as FromModulusCreateableZnRing>::create(create_modulus).map(|ring| RingValue::from(ring).as_field().ok().unwrap().into())
    }
}

///
/// Implements the isomorphisms `S: CanHomFrom<AsFieldBase<RingStore<Type = R>>>` and 
/// `AsFieldBase<RingStore<Type = S>>: CanHomFrom<R>`.
/// 
/// This has to be a macro, as a blanket implementation would unfortunately cause conflicting impls.
/// Usually, whenever a ring naturally might be a ring (e.g. like [`crate::rings::zn::zn_64::Zn`] or
/// [`crate::rings::extension::extension_impl::FreeAlgebraImpl`], which even provide a function like 
/// [`crate::rings::zn::ZnRingStore::as_field()`]), you might use this macro to implement [`CanHomFrom`]
/// that simplify conversion from and to the field wrapper.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::field::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::{impl_eq_based_self_iso, impl_field_wrap_unwrap_homs, impl_field_wrap_unwrap_isos};
/// // A no-op wrapper around Zn
/// #[derive(Copy, Clone)]
/// struct MyPossibleField {
///     base_zn: Zn
/// }
/// 
/// impl PartialEq for MyPossibleField {
/// 
///     fn eq(&self, other: &Self) -> bool {
///         self.base_zn.get_ring() == other.base_zn.get_ring()
///     }
/// }
/// 
/// // impl_wrap_unwrap_homs! relies on the homs/isos of the wrapped ring, so provide those
/// impl_eq_based_self_iso!{ MyPossibleField }
/// 
/// impl DelegateRing for MyPossibleField {
///     
///     type Base = ZnBase;
///     type Element = ZnEl;
///
///     fn get_delegate(&self) -> &Self::Base { self.base_zn.get_ring() }
///     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
///     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
///     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
///     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// }
/// 
/// impl_field_wrap_unwrap_homs!{ MyPossibleField, MyPossibleField }
/// 
/// // there is also a generic verision, which looks like
/// impl_field_wrap_unwrap_isos!{ <{ /* type params here */ }> MyPossibleField, MyPossibleField where /* constraints here */ }
/// 
/// let R = RingValue::from(MyPossibleField { base_zn: Zn::new(5) });
/// let R_field = RingValue::from(AsFieldBase::promise_is_perfect_field(R));
/// let _ = R.can_hom(&R_field).unwrap();
/// let _ = R_field.can_hom(&R).unwrap();
/// let _ = R.can_iso(&R_field).unwrap();
/// let _ = R_field.can_iso(&R).unwrap();
/// ```
/// 
#[macro_export]
macro_rules! impl_field_wrap_unwrap_homs {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsFieldRingStore, $($gen_args)*> CanHomFrom<$self_type_from> for $crate::rings::field::AsFieldBase<AsFieldRingStore>
            where AsFieldRingStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$self_type_from) -> Option<Self::Homomorphism> {
                self.get_delegate().has_canonical_hom(from)
            }

            fn map_in(&self, from: &$self_type_from, el: <$self_type_from as $crate::ring::RingBase>::Element, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                self.rev_delegate(self.get_delegate().map_in(from, el, hom))
            }
        }
        
        impl<AsFieldRingStore, $($gen_args)*> CanHomFrom<$crate::rings::field::AsFieldBase<AsFieldRingStore>> for $self_type_to
            where AsFieldRingStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$crate::rings::field::AsFieldBase<AsFieldRingStore>) -> Option<Self::Homomorphism> {
                self.has_canonical_hom(from.get_delegate())
            }

            fn map_in(&self, from: &$crate::rings::field::AsFieldBase<AsFieldRingStore>, el: $crate::rings::field::FieldEl<AsFieldRingStore>, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                self.map_in(from.get_delegate(), from.delegate(el), hom)
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_field_wrap_unwrap_homs!{ <{}> $self_type_from, $self_type_to where }
    };
}

///
/// Implements the isomorphisms `S: CanIsoFromTo<AsFieldBase<RingStore<Type = R>>>` and `AsFieldBase<RingStore<Type = S>>: CanIsoFromTo<R>`.
/// 
/// This has to be a macro, as a blanket implementation would unfortunately cause conflicting impls.
/// For an example and more detailed explanation, see [`impl_field_wrap_unwrap_homs!`]; 
/// 
#[macro_export]
macro_rules! impl_field_wrap_unwrap_isos {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsFieldRingStore, $($gen_args)*> CanIsoFromTo<$self_type_from> for $crate::rings::field::AsFieldBase<AsFieldRingStore>
            where AsFieldRingStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$self_type_from) -> Option<Self::Isomorphism> {
                self.get_delegate().has_canonical_iso(from)
            }

            fn map_out(&self, from: &$self_type_from, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <$self_type_from as RingBase>::Element {
                self.get_delegate().map_out(from, self.delegate(el), iso)
            }
        }
        
        impl<AsFieldRingStore, $($gen_args)*> CanIsoFromTo<$crate::rings::field::AsFieldBase<AsFieldRingStore>> for $self_type_to
            where AsFieldRingStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$crate::rings::field::AsFieldBase<AsFieldRingStore>) -> Option<Self::Isomorphism> {
                self.has_canonical_iso(from.get_delegate())
            }

            fn map_out(&self, from: &$crate::rings::field::AsFieldBase<AsFieldRingStore>, el: <Self as RingBase>::Element, hom: &Self::Isomorphism) -> $crate::rings::field::FieldEl<AsFieldRingStore> {
                from.rev_delegate(self.map_out(from.get_delegate(), el, hom))
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_field_wrap_unwrap_isos!{ <{}> $self_type_from, $self_type_to where }
    };
}

#[cfg(test)]
use crate::rings::zn::zn_big::Zn;
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
fn test_canonical_hom_axioms_wrap_unwrap() {
    let R = Zn::new(StaticRing::<i64>::RING, 17).as_field().ok().unwrap();
    crate::ring::generic_tests::test_hom_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
    crate::ring::generic_tests::test_iso_axioms(RingRef::new(R.get_ring().get_delegate()), &R, RingRef::new(R.get_ring().get_delegate()).elements());
}