use crate::ring::*;
use std::cmp::*;

///
/// Trait for rings that have a total ordering on their elements.
/// The ordering must be compatible with addition and multiplication
/// in the usual sense.
/// 
/// In particular, this should only be implemented for rings that are
/// subrings of the real numbers.
/// 
pub trait OrderedRing: RingBase {

    ///
    /// Returns whether `lhs` is [`Ordering::Less`], [`Ordering::Equal`] or [`Ordering::Greater`]
    /// than `rhs`.
    /// 
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> Ordering;

    ///
    /// Returns whether `abs(lhs)` is [`Ordering::Less`], [`Ordering::Equal`] or [`Ordering::Greater`]
    /// than `abs(rhs)`.
    /// 
    fn abs_cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> Ordering {
        self.cmp(&self.abs(self.clone_el(lhs)), &self.abs(self.clone_el(rhs)))
    }

    ///
    /// Returns whether `lhs <= rhs`.
    /// 
    fn is_leq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) != Ordering::Greater
    }
    
    ///
    /// Returns whether `lhs >= rhs`.
    /// 
    fn is_geq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) != Ordering::Less
    }

    ///
    /// Returns whether `lhs < rhs`.
    /// 
    fn is_lt(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) == Ordering::Less
    }
    
    ///
    /// Returns whether `lhs > rhs`.
    /// 
    fn is_gt(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.cmp(lhs, rhs) == Ordering::Greater
    }

    ///
    /// Returns whether `value < 0`.
    /// 
    fn is_neg(&self, value: &Self::Element) -> bool {
        self.is_lt(value, &self.zero())
    }

    ///
    /// Returns whether `value > 0`.
    /// 
    fn is_pos(&self, value: &Self::Element) -> bool {
        self.is_gt(value, &self.zero())
    }

    ///
    /// Returns the absolute value of `value`, i.e. `value` if `value >= 0` and `-value` otherwise.
    /// 
    fn abs(&self, value: Self::Element) -> Self::Element {
        if self.is_neg(&value) {
            self.negate(value)
        } else {
            value
        }
    }

    ///
    /// Returns the larger one of `fst` and `snd`.
    /// 
    fn max<'a>(&self, fst: &'a Self::Element, snd: &'a Self::Element) -> &'a Self::Element {
        if self.is_geq(fst, snd) {
            return fst;
        } else {
            return snd;
        }
    }
}

///
/// Trait for [`RingStore`]s that store [`OrderedRing`]s. Mainly used
/// to provide a convenient interface to the `OrderedRing`-functions.
/// 
pub trait OrderedRingStore: RingStore
    where Self::Type: OrderedRing
{
    delegate!{ OrderedRing, fn cmp(&self, lhs: &El<Self>, rhs: &El<Self>) -> Ordering }
    delegate!{ OrderedRing, fn abs_cmp(&self, lhs: &El<Self>, rhs: &El<Self>) -> Ordering }
    delegate!{ OrderedRing, fn is_leq(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn is_geq(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn is_lt(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn is_gt(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn is_neg(&self, value: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn is_pos(&self, value: &El<Self>) -> bool }
    delegate!{ OrderedRing, fn abs(&self, value: El<Self>) -> El<Self> }
    
    ///
    /// See [`OrderedRing::max()`].
    /// 
    fn max<'a>(&self, fst: &'a El<Self>, snd: &'a El<Self>) -> &'a El<Self> {
        self.get_ring().max(fst, snd)
    }
}

impl<R: ?Sized> OrderedRingStore for R
    where R: RingStore,
        R::Type: OrderedRing
{}