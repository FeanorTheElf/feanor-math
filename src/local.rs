use crate::algorithms::int_bisect;
use crate::pid::{EuclideanRing, EuclideanRingStore};
use crate::divisibility::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;

#[stability::unstable(feature = "enable")]
pub trait PrincipalLocalRing: EuclideanRing {

    fn max_ideal_gen(&self) -> &Self::Element;

    fn valuation(&self, x: &Self::Element) -> Option<usize> {
        assert!(self.is_noetherian());
        if self.is_zero(x) {
            return None;
        }
        let ring = RingRef::new(self);
        return Some(int_bisect::find_root_floor(&StaticRing::<i64>::RING, 0, |e| {
            if *e < 0 || ring.checked_div(x, &ring.pow(ring.clone_el(ring.max_ideal_gen()), *e as usize)).is_some() {
                -1
            } else {
                1
            }
        }) as usize)
    }
}

#[stability::unstable(feature = "enable")]
pub trait PrincipalLocalRingStore: EuclideanRingStore
    where Self::Type: PrincipalLocalRing
{
    delegate!{ PrincipalLocalRing, fn max_ideal_gen(&self) -> &El<Self> }
    delegate!{ PrincipalLocalRing, fn valuation(&self, x: &El<Self>) -> Option<usize> }
}

impl<R> PrincipalLocalRingStore for R
    where R: RingStore,
        R::Type: PrincipalLocalRing {}