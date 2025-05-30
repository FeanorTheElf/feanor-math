use std::marker::PhantomData;

use crate::algorithms::int_factor::is_prime_power;
use crate::field::*;
use crate::homomorphism::Homomorphism;
use crate::ring::*;
use crate::integer::{BigIntRing, IntegerRing, IntegerRingStore};
use crate::specialization::FiniteRingSpecializable;
use crate::unsafe_any::UnsafeAny;

///
/// Trait for rings that are finite.
/// 
/// Currently [`FiniteRing`] is a subtrait of the unstable trait [`FiniteRingSpecializable`],
/// so it is at the moment impossible to implement [`FiniteRing`] for a custom ring type
/// without enabling unstable features. Sorry.
/// 
pub trait FiniteRing: RingBase + FiniteRingSpecializable {

    type ElementsIter<'a>: Sized + Clone + Iterator<Item = <Self as RingBase>::Element>
        where Self: 'a;

    ///
    /// Returns an iterator over all elements of this ring.
    /// The order is not specified.
    /// 
    fn elements<'a>(&'a self) -> Self::ElementsIter<'a>;

    ///
    /// Returns a uniformly random element from this ring, using the randomness
    /// provided by `rng`.
    /// 
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element;

    ///
    /// Returns the number of elements in this ring, if it fits within
    /// the given integer ring.
    /// 
    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing;
}

///
/// [`RingStore`] for [`FiniteRing`]
/// 
pub trait FiniteRingStore: RingStore
    where Self::Type: FiniteRing
{
    ///
    /// See [`FiniteRing::elements()`].
    /// 
    fn elements<'a>(&'a self) -> <Self::Type as FiniteRing>::ElementsIter<'a> {
        self.get_ring().elements()
    }

    ///
    /// See [`FiniteRing::random_element()`].
    /// 
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> {
        self.get_ring().random_element(rng)
    }

    ///
    /// See [`FiniteRing::size()`].
    /// 
    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_ring().size(ZZ)
    }
}

impl<R: RingStore> FiniteRingStore for R
    where R::Type: FiniteRing
{}

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

#[stability::unstable(feature = "enable")]
pub trait ComputeFrobeniusRing: Field + FiniteRing {
    
    type FrobeniusData;

    fn create_frobenius(&self, exponent_of_p: usize) -> (Self::FrobeniusData, usize);

    fn apply_frobenius(&self, _frobenius_data: &Self::FrobeniusData, exponent_of_p: usize, x: Self::Element) -> Self::Element;
}

impl<R> ComputeFrobeniusRing for R
    where R: ?Sized + Field + FiniteRing
{
    type FrobeniusData = UnsafeAnyFrobeniusDataGuarded<R>;

    default fn create_frobenius(&self, exponent_of_p: usize) -> (Self::FrobeniusData, usize) {
        (UnsafeAnyFrobeniusDataGuarded::uninit(), exponent_of_p)
    }

    default fn apply_frobenius(&self, _frobenius_data: &Self::FrobeniusData, exponent_of_p: usize, x: Self::Element) -> Self::Element {
        if exponent_of_p == 0 {
            return x
        } else {
            let (p, e) = is_prime_power(BigIntRing::RING, &self.size(&BigIntRing::RING).unwrap()).unwrap();
            return RingRef::new(self).pow_gen(x, &BigIntRing::RING.pow(p, exponent_of_p % e), BigIntRing::RING);
        }
    }
}

#[stability::unstable(feature = "enable")]
pub struct Frobenius<R>
    where R: RingStore,
        R::Type: FiniteRing + Field
{
    field: R,
    data: <R::Type as ComputeFrobeniusRing>::FrobeniusData,
    exponent_of_p: usize
}

impl<R> Frobenius<R>
    where R: RingStore,
        R::Type: FiniteRing + Field
{
    #[stability::unstable(feature = "enable")]
    pub fn new(field: R, exponent_of_p: usize) -> Self {
        let (_p, field_dimension) = is_prime_power(BigIntRing::RING, &field.size(BigIntRing::RING).unwrap()).unwrap();
        let exponent_of_p = exponent_of_p % field_dimension;
        let (data, exponent_of_p2) = field.get_ring().create_frobenius(exponent_of_p);
        assert_eq!(exponent_of_p, exponent_of_p2);
        return Self { field: field, data: data, exponent_of_p: exponent_of_p };
    }
}

impl<R> Homomorphism<R::Type, R::Type> for Frobenius<R>
    where R: RingStore,
        R::Type: FiniteRing + Field
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

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::divisibility::DivisibilityRingStore;
    use crate::integer::{int_cast, BigIntRing, IntegerRingStore};
    use crate::ordered::OrderedRingStore;
    use crate::primitive_int::StaticRing;

    use super::{FiniteRing, FiniteRingStore, RingStore};

    pub fn test_finite_ring_axioms<R>(ring: &R)
        where R: RingStore,
            R::Type: FiniteRing
    {
        let ZZ = BigIntRing::RING;
        let size = ring.size(&ZZ).unwrap();
        let char = ring.characteristic(&ZZ).unwrap();
        assert!(ZZ.divides(&size, &char));

        if ZZ.is_geq(&size, &ZZ.power_of_two(7)) {
            assert_eq!(None, ring.size(&StaticRing::<i8>::RING));
        }

        if ZZ.is_leq(&size, &ZZ.power_of_two(30)) {
            assert_eq!(int_cast(size, &StaticRing::<i64>::RING, &ZZ) as usize, ring.elements().count());
        }
    }
}
