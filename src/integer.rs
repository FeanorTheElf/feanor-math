use crate::primitive::StaticRingBase;
use crate::ring::*;
use crate::euclidean::*;
use crate::ordered::*;

pub trait IntegerRing: EuclideanRing + CanonicalIso<StaticRingBase<i128>> + OrderedRing {

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool;
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize>;
}

pub trait IntegerRingWrapper: RingWrapper<Type: IntegerRing> {

    delegate!{ fn abs_is_bit_set(&self, value: &El<Self>, i: usize) -> bool }
    delegate!{ fn abs_highest_set_bit(&self, value: &El<Self>) -> Option<usize> }
}

impl<R> IntegerRingWrapper for R
    where R: RingWrapper<Type: IntegerRing>
{}