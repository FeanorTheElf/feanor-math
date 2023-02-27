use crate::ring::*;
use crate::divisibility::*;

pub trait EuclideanRing: DivisibilityRing {

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);
    fn euclidean_deg(&self, val: &Self::Element) -> usize;

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).0
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).1
    }
}

pub trait EuclideanRingWrapper: RingWrapper<Type: EuclideanRing> + DivisibilityRingWrapper {

    delegate!{ fn euclidean_div_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }
    delegate!{ fn euclidean_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_deg(&self, val: &El<Self>) -> usize }
}

impl<R> EuclideanRingWrapper for R
    where R: RingWrapper, R::Type: EuclideanRing
{}