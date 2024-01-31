use crate::divisibility::Domain;
use crate::ring::*;
use crate::pid::*;

///
/// Trait for rings that are fields, i.e. where every
/// nonzero element has an inverse.
/// 
/// Note that fields must be commutative.
/// 
pub trait Field: Domain + EuclideanRing {

    fn div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        return self.checked_left_div(lhs, rhs).unwrap();
    }
}

///
/// Trait for [`RingStore`]s that store [`Field`]s. Mainly used
/// to provide a convenient interface to the `Field`-functions.
/// 
pub trait FieldStore: RingStore + EuclideanRingStore
    where Self::Type: Field
{
    delegate!{ Field, fn div(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
}

impl<R> FieldStore for R
    where R: RingStore, R::Type: Field
{}