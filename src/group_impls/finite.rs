use crate::group::*;
use crate::integer::IntegerRing;
use crate::ring::*;

#[stability::unstable(feature = "enable")]
pub trait FiniteGroupBase: AbelianGroupBase {
    
    /// Returns the order of this group as an element of the given
    /// implementation of `ZZ`.
    ///
    /// If `None` is returned, this means the given integer ring might not be able
    /// to represent the group order. This must never happen if the given implementation
    /// of `ZZ` allows for unbounded integers (like [`crate::integer::BigIntRing`]).
    /// In other cases however, we allow to perform the size check heuristically only,
    /// so this might return `None` even in some cases where the integer ring would in
    /// fact be able to represent the group order.
    fn group_order<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
    where
        I::Type: IntegerRing;
    
    /// Returns a uniformly random element from this ring, using the randomness
    /// provided by `rng`.
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as AbelianGroupBase>::Element;
}

#[stability::unstable(feature = "enable")]
pub trait FiniteGroupStore: AbelianGroupStore
    where Self::Type: FiniteGroupBase
{    
    /// See [`FiniteGroupBase::group_order()`].
    fn group_order<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
    where
        I::Type: IntegerRing
    {
        self.get_group().group_order(ZZ)
    }

    /// See [`FiniteGroupBase::random_element()`].
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> GroupEl<Self> {
        self.get_group().random_element(rng)
    }
}

impl<G: AbelianGroupStore> FiniteGroupStore for G
    where G::Type: FiniteGroupBase
{}