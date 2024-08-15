use crate::ring::*;
use crate::integer::{IntegerRingStore, IntegerRing};

///
/// Trait for rings that are finite.
/// 
pub trait FiniteRing: RingBase {

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
    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
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
    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.get_ring().size(ZZ)
    }
}

impl<R: RingStore> FiniteRingStore for R
    where R::Type: FiniteRing
{}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::divisibility::DivisibilityRingStore;
    use crate::integer::{int_cast, BigIntRing, IntegerRingStore};
    use crate::ordered::OrderedRingStore;
    use crate::primitive_int::StaticRing;

    use super::{FiniteRing, FiniteRingStore};

    pub fn test_finite_ring_axioms<R>(ring: &R)
        where R: FiniteRingStore,
            R::Type: FiniteRing
    {
        let ZZ = BigIntRing::RING;
        let size = ring.size(&ZZ).unwrap();
        let char = ring.characteristic(&ZZ).unwrap();
        assert!(ZZ.checked_div(&size, &char).is_some());

        if ZZ.is_geq(&size, &ZZ.power_of_two(7)) {
            assert_eq!(None, ring.size(&StaticRing::<i8>::RING));
        }

        if ZZ.is_leq(&size, &ZZ.power_of_two(30)) {
            assert_eq!(int_cast(size, &StaticRing::<i64>::RING, &ZZ) as usize, ring.elements().count());
        }
    }
}