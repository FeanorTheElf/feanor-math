use crate::ring::*;

pub trait IntegerRing: RingBase {

    fn to_i128(&self, value: &Self::Element) -> Result<i128, ()>;
    fn from_i128(&self, value: i128) -> Result<Self::Element, ()>;

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