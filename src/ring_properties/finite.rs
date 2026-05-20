use crate::prelude::*;
use crate::ring_properties::specialization::FiniteRingSpecializable;

/// Trait for rings that are finite.
///
/// Currently [`FiniteRing`] is a subtrait of the unstable trait [`FiniteRingSpecializable`],
/// so it is at the moment impossible to implement [`FiniteRing`] for a custom ring type
/// without enabling unstable features. Sorry.
pub trait FiniteRing: RingBase + FiniteRingSpecializable {
    /// Returns a uniformly random element from this ring, using the randomness
    /// provided by `rng`.
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element;

    /// Returns the number of elements in this ring, if it fits within
    /// the given integer ring.
    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
    where
        I::Ring: IntegerRing;
}

/// [`RingStore`] for [`FiniteRing`]
pub trait FiniteRingStore: RingStore
where
    Self::Ring: FiniteRing,
{
    /// See [`FiniteRing::random_element()`].
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> { self.get_ring().random_element(rng) }

    /// See [`FiniteRing::size()`].
    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
    where
        I::Ring: IntegerRing,
    {
        self.get_ring().size(ZZ)
    }
}

impl<R: RingStore> FiniteRingStore for R where R::Ring: FiniteRing {}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use super::*;

    pub fn test_finite_ring_axioms<R>(ring: &R)
    where
        R: RingStore,
        R::Ring: FiniteRing,
    {
        let size = ring.size(&ZZbig).unwrap();
        let char = ring.characteristic(&ZZbig).unwrap();
        assert!(ZZbig.divides(&size, &char));

        if ZZbig.is_geq(&size, &ZZbig.power_of_two(7)) {
            assert_eq!(None, ring.size(&StaticRing::<i8>::RING));
        }
    }
}
