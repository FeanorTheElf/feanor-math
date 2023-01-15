use crate::ring::*;

pub trait DivisibilityRing: RingBase {

    fn checked_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element>;
}

pub trait DivisibilityRingWrapper: RingWrapper<Type: DivisibilityRing> {

    delegate!{ fn checked_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
}

impl<R> DivisibilityRingWrapper for R
    where R: RingWrapper<Type: DivisibilityRing>
{}