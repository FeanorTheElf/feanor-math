use crate::ring::*;
use crate::euclidean::*;

///
/// Trait for rings that are fields, i.e. where every
/// nonzero element has an inverse.
/// 
/// Note that fields must be commutative.
/// 
pub trait Field: EuclideanRing {

    fn div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        return self.checked_left_div(lhs, rhs).unwrap();
    }
}

pub trait FieldWrapper: RingStore + EuclideanRingStore
    where Self::Type: Field
{
    delegate!{ fn div(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
}

impl<R> FieldWrapper for R
    where R: RingStore, R::Type: Field
{}