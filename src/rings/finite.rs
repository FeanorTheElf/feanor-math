use crate::ring::*;
use crate::integer::*;
use crate::specialization::FiniteRingSpecializable;

///
/// Trait for rings that are finite.
/// 
/// Currently [`FiniteRing`] is a subtrait of the unstable trait [`FiniteRingSpecializable`],
/// so it is at the moment impossible to implement [`FiniteRing`] for a custom ring type
/// without enabling unstable features. Sorry.
/// 
pub trait FiniteRing: RingBase + FiniteRingSpecializable {

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

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::divisibility::DivisibilityRingStore;
    use crate::integer::{BigIntRing, IntegerRingStore};
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
    }
}
