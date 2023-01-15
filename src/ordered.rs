use crate::ring::*;
use std::cmp::*;

pub trait OrderedRing: RingBase {

    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> Ordering;

    fn is_leq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) != Ordering::Greater
    }
    
    fn is_geq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) != Ordering::Less
    }

    fn is_lt(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) == Ordering::Less
    }
    
    fn is_gt(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) == Ordering::Greater
    }

    fn is_neg(&self, value: &Self::Element) -> bool {
        self.is_lt(value, &self.zero())
    }

    fn is_pos(&self, value: &Self::Element) -> bool {
        self.is_gt(value, &self.zero())
    }

    fn abs(&self, value: Self::Element) -> Self::Element {
        if self.is_neg(&value) {
            self.negate(value)
        } else {
            value
        }
    }
}

pub trait OrderedRingWrapper: RingWrapper<Type: OrderedRing> {

    delegate!{ fn cmp(&self, lhs: &El<Self>, rhs: &El<Self>) -> Ordering }
    delegate!{ fn is_leq(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_geq(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_lt(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_gt(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_neg(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_pos(&self, value: &El<Self>) -> bool }
    delegate!{ fn abs(&self, value: El<Self>) -> El<Self> }
}

impl<R> OrderedRingWrapper for R
    where R: RingWrapper<Type: OrderedRing>
{}