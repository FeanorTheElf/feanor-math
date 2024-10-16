use std::marker::PhantomData;
use std::ops::Range;

use crate::algorithms::int_factor::is_prime_power;
use crate::delegate::DelegateRing;
use crate::divisibility::{DivisibilityRing, Domain};
use crate::field::{Field, PerfectField};
use crate::homomorphism::{CanHomFrom, CanIsoFromTo, Homomorphism, LargeIntHom};
use crate::integer::*;
use crate::iters::{multi_cartesian_product, MultiProduct};
use crate::local::PrincipalLocalRing;
use crate::ordered::OrderedRingStore;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::seq::CloneClonable;
use crate::unsafe_any::UnsafeAny;
use super::extension::*;
use super::field::AsFieldBase;
use super::finite::*;
use super::local::AsLocalPIRBase;
use super::zn::*;

///
/// A wrapper around a ring that marks this ring to be a a finite field.
///
pub struct AsFiniteFieldBase<R> 
    where R: RingStore, R::Type: Field
{
    base: R,
    canonical_basis: Vec<El<R>>,
    zero: FiniteFieldEl<R>
}

pub trait FiniteFieldSpecializable: Field {

    fn as_finite_field<R: RingStore<Type = Self>>(self_store: R) -> Result<AsFiniteField<R>, R>;
}

pub trait FiniteFieldSpecializableStore: RingStore
    where Self::Type: FiniteFieldSpecializable
{
    fn as_finite_field(self) -> Result<AsFiniteField<Self>, Self> {
        <Self::Type as FiniteFieldSpecializable>::as_finite_field(self)
    }
}

impl<R> FiniteFieldSpecializableStore for R
    where R: RingStore,
        R::Type: FiniteFieldSpecializable
{}

impl<R> Clone for AsFiniteFieldBase<R>
    where R: RingStore + Clone,
        R::Type: Field
{
    fn clone(&self) -> Self {
        Self {
            zero: self.clone_el(&self.zero),
            canonical_basis: self.canonical_basis.iter().map(|x| self.base.clone_el(x)).collect(),
            base: self.base.clone()
        }
    }
}

impl<R> PartialEq for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    fn eq(&self, other: &Self) -> bool {
        if self.base.get_ring() == other.base.get_ring() {
            assert!(self.canonical_basis.len() == other.canonical_basis.len());
            true
        } else {
            false
        }
    }
}

///
/// [`RingStore`] for [`AsFieldBase`].
/// 
#[allow(type_alias_bounds)]
pub type AsFiniteField<R> = RingValue<AsFiniteFieldBase<R>>;

#[repr(transparent)]
pub struct FiniteFieldEl<R>(El<R>)
    where R: RingStore,
        R::Type: Field;

impl<R> Clone for FiniteFieldEl<R>
    where R: RingStore,
        R::Type: Field,
        El<R>: Clone
{
    fn clone(&self) -> Self {
        FiniteFieldEl(self.0.clone())
    }
}

impl<R> Copy for FiniteFieldEl<R>
    where R: RingStore,
        R::Type: Field,
        El<R>: Copy
{}

impl<R> AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    pub fn promise_is_finite(base: R, some_basis: Vec<El<R>>) -> Self {
        Self {
            canonical_basis: some_basis,
            zero: FiniteFieldEl(base.zero()),
            base: base
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

impl<R> AsFiniteField<R>
    where R: RingStore,
        R::Type: Field + ZnRing
{
    ///
    /// Creates a new [`AsFieldBase`] from a ring that is known to be `Fp` for a
    /// prime `p`.
    /// 
    pub fn from_fp(base: R) -> Self {
        let one = base.one();
        return AsFiniteField::from(AsFiniteFieldBase::promise_is_finite(base, vec![one]));
    }
}

impl<R> AsFiniteField<R>
    where R: RingStore,
        R::Type: FreeAlgebra + Field,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{
    ///
    /// Creates a new [`AsFieldBase`] from a ring that is known to be a galois
    /// field
    /// 
    pub fn from_galois_field(base: R) -> Self {
        let rank = base.rank();
        let basis = (0..rank).map(|i| base.pow(base.canonical_gen(), i)).collect();
        return AsFiniteField::from(AsFiniteFieldBase::promise_is_finite(base, basis));
    }
}

impl<R> AsFiniteField<R>
    where R: RingStore,
        R::Type: Field + FreeAlgebra,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteFieldSpecializable
{
    pub fn from_extension(base: R) -> Result<Self, R> {
        unimplemented!()
    }
}

impl<R> DelegateRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    type Element = FiniteFieldEl<R>;
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
        FiniteFieldEl(el)
    }
}

impl<R, S> CanHomFrom<AsFiniteFieldBase<S>> for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field,
        S: RingStore,
        S::Type: Field,
        R::Type: CanHomFrom<S::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFiniteFieldBase<S>) -> Option<Self::Homomorphism> {
        <R::Type as CanHomFrom<S::Type>>::has_canonical_hom(self.get_delegate(), from.get_delegate())
    }

    fn map_in(&self, from: &AsFiniteFieldBase<S>, el: FiniteFieldEl<S>, hom: &Self::Homomorphism) -> Self::Element {
        FiniteFieldEl(<R::Type as CanHomFrom<S::Type>>::map_in(self.get_delegate(), from.get_delegate(), el.0, hom))
    }
}

impl<R, S> CanIsoFromTo<AsFiniteFieldBase<S>> for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field,
        S: RingStore,
        S::Type: Field,
        R::Type: CanIsoFromTo<S::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFiniteFieldBase<S>) -> Option<Self::Isomorphism> {
        <R::Type as CanIsoFromTo<S::Type>>::has_canonical_iso(self.get_delegate(), from.get_delegate())
    }

    fn map_out(&self, from: &AsFiniteFieldBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> FiniteFieldEl<S> {
        FiniteFieldEl(<R::Type as CanIsoFromTo<S::Type>>::map_out(self.get_delegate(), from.get_delegate(), el.0, iso))
    }
}

///
/// Necessary to potentially implement [`crate::rings::zn::ZnRing`].
/// 
impl<R, S: IntegerRing + ?Sized> CanHomFrom<S> for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field,
        R::Type: CanHomFrom<S>
{
    type Homomorphism = <R::Type as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: S::Element, hom: &Self::Homomorphism) -> Self::Element {
        FiniteFieldEl(<R::Type as CanHomFrom<S>>::map_in(self.get_delegate(), from, el, hom))
    }
}

impl<R, S> CanHomFrom<AsLocalPIRBase<S>> for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field,
        S: RingStore,
        S::Type: Field,
        R::Type: CanHomFrom<S::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsLocalPIRBase<S>) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from.get_delegate())
    }

    fn map_in(&self, from: &AsLocalPIRBase<S>, el: <AsLocalPIRBase<S> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.get_delegate().map_in(from.get_delegate(), from.delegate(el), hom))
    }
}

impl<R, S> CanIsoFromTo<AsLocalPIRBase<S>> for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field,
        S: RingStore,
        S::Type: Field,
        R::Type: CanIsoFromTo<S::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsLocalPIRBase<S>) -> Option<Self::Isomorphism> {
        self.get_delegate().has_canonical_iso(from.get_delegate())
    }

    fn map_out(&self, from: &AsLocalPIRBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> <AsLocalPIRBase<S> as RingBase>::Element {
        from.rev_delegate(self.get_delegate().map_out(from.get_delegate(), self.delegate(el), iso))
    }
}

impl<R> DivisibilityRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.get_delegate().checked_left_div(&lhs.0, &rhs.0).map(FiniteFieldEl)
    }
}

impl<R> PrincipalIdealRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
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

impl<R> PrincipalLocalRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
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

impl<R> EuclideanRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        let (q, r) = self.get_delegate().euclidean_div_rem(self.delegate(lhs), self.delegate_ref(rhs));
        return (self.rev_delegate(q), self.rev_delegate(r));
    }

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().euclidean_div(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.rev_delegate(self.get_delegate().euclidean_rem(self.delegate(lhs), self.delegate_ref(rhs)))
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        self.get_delegate().euclidean_deg(self.delegate_ref(val))
    }
}

impl<R> Domain for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{}

impl<R> Field for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{}

impl<R> PerfectField for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{}

pub struct AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{
    ring: &'a AsFiniteFieldBase<R>
}

impl<'a, 'b, R> Clone for AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, R> Copy for AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{}

impl<'a, 'b, R> FnOnce<(&'b [i32],)> for AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{
    type Output = El<AsFiniteField<R>>;

    extern "rust-call" fn call_once(self, args: (&'b [i32],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R> FnMut<(&'b [i32],)> for AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [i32],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R> Fn<(&'b [i32],)> for AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>
    where R: RingStore,
        R::Type: Field
{
    extern "rust-call" fn call(&self, args: (&'b [i32],)) -> Self::Output {
        if args.0[0] == i32::MAX - 1 {
            panic!("AsFiniteFieldBase supports only enumerating i32::MAX elements of the ring");
        }
        let int_hom = self.ring.base.int_hom();
        FiniteFieldEl(<_ as RingBase>::sum(self.ring.base.get_ring(), self.ring.canonical_basis.iter().zip(args.0.iter().copied()).map(|(a, b)| {
            int_hom.mul_ref_map(a, &b)
        })))
    }
}

impl<R> FiniteRing for AsFiniteFieldBase<R>
    where R: RingStore,
        R::Type: Field
{
    type ElementsIter<'a> = MultiProduct<Range<i32>, AsFiniteFieldBaseWrtBasisElementsCreator<'a, R>, CloneClonable, FiniteFieldEl<R>>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        let ZZbig = BigIntRing::RING;
        let i32_MAX = ZZbig.int_hom().map(i32::MAX);
        let coeff_bound_exclusive = if ZZbig.is_leq(&self.base.characteristic(ZZbig).unwrap(), &i32_MAX) {
            int_cast(self.base.characteristic(ZZbig).unwrap(), StaticRing::<i32>::RING, ZZbig)
        } else {
            i32::MAX
        };
        multi_cartesian_product((0..self.canonical_basis.len()).map(|_| (0..coeff_bound_exclusive)), AsFiniteFieldBaseWrtBasisElementsCreator { ring: self }, CloneClonable)
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let char = self.base.characteristic(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.canonical_basis.len() * ZZ.abs_log2_ceil(&char).unwrap() {
            return Some(ZZ.pow(char, self.canonical_basis.len()));
        } else {
            return None;
        }
    }

    default fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> <Self as RingBase>::Element {
        if let Some(char) = self.base.characteristic(StaticRing::<i32>::RING) {
            let int_hom = self.base.int_hom();
            FiniteFieldEl(<_ as RingBase>::sum(self.base.get_ring(), self.canonical_basis.iter().map(|a| {
                int_hom.mul_ref_map(a, &StaticRing::<i32>::RING.get_uniformly_random(&char, || rng()))
            })))
        } else {
            let char = self.base.characteristic(BigIntRing::RING).unwrap();
            let int_hom = LargeIntHom::new(BigIntRing::RING, &self.base);
            FiniteFieldEl(<_ as RingBase>::sum(self.base.get_ring(), self.canonical_basis.iter().map(|a| {
                int_hom.mul_ref_map(a, &BigIntRing::RING.get_uniformly_random(&char, || rng()))
            })))
        }
    }
}

impl<R1, R2, S> CanHomFrom<AsFieldBase<R1>> for AsFiniteFieldBase<S>
    where S: RingStore<Type = AsFieldBase<R2>>,
        R1: RingStore,
        R1::Type: DivisibilityRing,
        R2: RingStore,
        R2::Type: DivisibilityRing,
        R2::Type: CanHomFrom<R1::Type>
{
    type Homomorphism = <AsFieldBase<R2> as CanHomFrom<AsFieldBase<R1>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFieldBase<R1>) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &AsFieldBase<R1>, el: <AsFieldBase<R1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.get_ring().map_in(from, el, hom))
    }

    fn map_in_ref(&self, from: &AsFieldBase<R1>, el: &<AsFieldBase<R1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.get_ring().map_in_ref(from, el, hom))
    }
}

impl<R1, R2, S> CanIsoFromTo<AsFieldBase<R1>> for AsFiniteFieldBase<S>
    where S: RingStore<Type = AsFieldBase<R2>>,
        R1: RingStore,
        R1::Type: DivisibilityRing,
        R2: RingStore,
        R2::Type: DivisibilityRing,
        R2::Type: CanIsoFromTo<R1::Type>
{
    type Isomorphism = <AsFieldBase<R2> as CanIsoFromTo<AsFieldBase<R1>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<R1>) -> Option<Self::Isomorphism> {
        self.base.get_ring().has_canonical_iso(from)
    }

    fn map_out(&self, from: &AsFieldBase<R1>, el: Self::Element, iso: &Self::Isomorphism) -> <AsFieldBase<R1> as RingBase>::Element {
        self.base.get_ring().map_out(from, self.delegate(el), iso)
    }
}

///
/// Implements the homomorphisms `S: CanHomFrom<AsFiniteFieldBase<RingStore<Type = R>>>` and `AsFiniteFieldBase<RingStore<Type = S>>: CanHomFrom<R>`.
/// 
/// This has to be a macro, as a blanket implementation would unfortunately cause conflicting impls.
/// For an example and more detailed explanation, see [`feanor_math::impl_field_wrap_unwrap_homs!`]; 
/// 
#[macro_export]
macro_rules! impl_finitefield_wrap_unwrap_homs {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsFiniteFieldRingStore, $($gen_args)*> CanHomFrom<$self_type_from> for $crate::rings::field::AsFiniteFieldBase<AsFiniteFieldRingStore>
            where AsFiniteFieldRingStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$self_type_from) -> Option<Self::Homomorphism> {
                self.get_delegate().has_canonical_hom(from)
            }

            fn map_in(&self, from: &$self_type_from, el: <$self_type_from as $crate::ring::RingBase>::Element, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                self.rev_delegate(self.get_delegate().map_in(from, el, hom))
            }
        }
        
        impl<AsFiniteFieldRingStore, $($gen_args)*> CanHomFrom<$crate::rings::field::AsFiniteFieldBase<AsFiniteFieldRingStore>> for $self_type_to
            where AsFiniteFieldRingStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Homomorphism = <$self_type_to as CanHomFrom<$self_type_from>>::Homomorphism;

            fn has_canonical_hom(&self, from: &$crate::rings::field::AsFiniteFieldBase<AsFiniteFieldRingStore>) -> Option<Self::Homomorphism> {
                self.has_canonical_hom(from.get_delegate())
            }

            fn map_in(&self, from: &$crate::rings::field::AsFiniteFieldBase<AsFiniteFieldRingStore>, el: $crate::rings::field::FieldEl<AsFiniteFieldRingStore>, hom: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                self.map_in(from.get_delegate(), from.delegate(el), hom)
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_finitefield_wrap_unwrap_homs!{ <{}> $self_type_from, $self_type_to where }
    };
}

///
/// Implements the isomorphisms `S: CanIsoFromTo<AsFiniteFieldBase<RingStore<Type = R>>>` and `AsFiniteFieldBase<RingStore<Type = S>>: CanIsoFromTo<R>`.
/// 
/// This has to be a macro, as a blanket implementation would unfortunately cause conflicting impls.
/// For an example and more detailed explanation, see [`feanor_math::impl_field_wrap_unwrap_homs!`]; 
/// 
#[macro_export]
macro_rules! impl_finitefield_wrap_unwrap_isos {
    (<{$($gen_args:tt)*}> $self_type_from:ty, $self_type_to:ty where $($constraints:tt)*) => {
        
        impl<AsFiniteFieldRingStore, $($gen_args)*> CanIsoFromTo<$self_type_from> for $crate::rings::finite_field::AsFiniteFieldBase<AsFiniteFieldRingStore>
            where AsFiniteFieldRingStore: RingStore<Type = $self_type_to>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$self_type_from) -> Option<Self::Isomorphism> {
                self.get_delegate().has_canonical_iso(from)
            }

            fn map_out(&self, from: &$self_type_from, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <$self_type_from as RingBase>::Element {
                self.get_delegate().map_out(from, self.delegate(el), iso)
            }
        }
        
        impl<AsFiniteFieldRingStore, $($gen_args)*> CanIsoFromTo<$crate::rings::finite_field::AsFiniteFieldBase<AsFiniteFieldRingStore>> for $self_type_to
            where AsFiniteFieldRingStore: RingStore<Type = $self_type_from>, $($constraints)*
        {
            type Isomorphism = <$self_type_to as CanIsoFromTo<$self_type_from>>::Isomorphism;

            fn has_canonical_iso(&self, from: &$crate::rings::finite_field::AsFiniteFieldBase<AsFiniteFieldRingStore>) -> Option<Self::Isomorphism> {
                self.has_canonical_iso(from.get_delegate())
            }

            fn map_out(&self, from: &$crate::rings::finite_field::AsFiniteFieldBase<AsFiniteFieldRingStore>, el: <Self as RingBase>::Element, hom: &Self::Isomorphism) -> $crate::rings::finite_field::FiniteFieldEl<AsFieldRingStore> {
                from.rev_delegate(self.map_out(from.get_delegate(), el, hom))
            }
        }
    };
    ($self_type_from:ty, $self_type_to:ty) => {
        impl_finitefield_wrap_unwrap_isos!{ <{}> $self_type_from, $self_type_to where }
    };
}

#[stability::unstable(feature = "enable")]
pub trait FiniteField: Field + FiniteRing {
    
    type FrobeniusData;

    fn create_frobenius(&self, exponent_of_p: usize) -> (Self::FrobeniusData, usize);

    fn apply_frobenius(&self, _frobenius_data: &Self::FrobeniusData, exponent_of_p: usize, x: Self::Element) -> Self::Element;
}

#[stability::unstable(feature = "enable")]
pub struct UnsafeAnyFrobeniusDataGuarded<GuardType: ?Sized> {
    content: UnsafeAny,
    guard: PhantomData<GuardType>
}

impl<GuardType: ?Sized> UnsafeAnyFrobeniusDataGuarded<GuardType> {
    
    #[stability::unstable(feature = "enable")]
    pub fn uninit() -> UnsafeAnyFrobeniusDataGuarded<GuardType> {
        Self {
            content: UnsafeAny::uninit(),
            guard: PhantomData
        }
    }
    
    #[stability::unstable(feature = "enable")]
    pub unsafe fn from<T>(value: T) -> UnsafeAnyFrobeniusDataGuarded<GuardType> {
        Self {
            content: UnsafeAny::from(value),
            guard: PhantomData
        }
    }
    
    
    #[stability::unstable(feature = "enable")]
    pub unsafe fn get<'a, T>(&'a self) -> &'a T {
        self.content.get()
    }
}

impl<R> FiniteField for R
    where R: ?Sized + Field + FiniteRing
{
    type FrobeniusData = UnsafeAnyFrobeniusDataGuarded<R>;

    default fn create_frobenius(&self, exponent_of_p: usize) -> (Self::FrobeniusData, usize) {
        (UnsafeAnyFrobeniusDataGuarded::uninit(), exponent_of_p)
    }

    default fn apply_frobenius(&self, _frobenius_data: &Self::FrobeniusData, exponent_of_p: usize, x: Self::Element) -> Self::Element {
        let q = self.size(&BigIntRing::RING).unwrap();
        let (p, e) = is_prime_power(BigIntRing::RING, &q).unwrap();
        return RingRef::new(self).pow_gen(x, &BigIntRing::RING.pow(p, exponent_of_p % e), BigIntRing::RING);
    }
}

#[stability::unstable(feature = "enable")]
pub struct Frobenius<R>
    where R: RingStore,
        R::Type: Field + FiniteRing
{
    field: R,
    data: <R::Type as FiniteField>::FrobeniusData,
    exponent_of_p: usize
}

impl<R> Frobenius<R>
    where R: RingStore,
        R::Type: Field + FiniteRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(field: R, exponent_of_p: usize) -> Self {
        let (data, exponent_of_p2) = field.get_ring().create_frobenius(exponent_of_p);
        assert_eq!(exponent_of_p, exponent_of_p2);
        return Self { field: field, data: data, exponent_of_p: exponent_of_p };
    }
}

impl<R> Homomorphism<R::Type, R::Type> for Frobenius<R>
    where R: RingStore,
        R::Type: Field + FiniteRing
{
    type CodomainStore = R;
    type DomainStore = R;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.field
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.field
    }

    fn map(&self, x: <R::Type as RingBase>::Element) -> <R::Type as RingBase>::Element {
        self.field.get_ring().apply_frobenius(&self.data, self.exponent_of_p, x)
    }
}

#[cfg(test)]
use galois_field::GaloisField;

#[test]
fn test_finite_ring_axioms() {
    let base = zn_64::Zn::new(17).as_field().ok().unwrap();
    let ring = AsFiniteField::from(AsFiniteFieldBase::from_fp(base));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&ring);

    let base = GaloisField::new(5, 3);
    let ring = AsFiniteField::from(AsFiniteFieldBase::from_galois_field(&base));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&ring);
}