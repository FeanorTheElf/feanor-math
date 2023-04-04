use crate::ring::*;
use crate::euclidean::*;

///
/// Trait for rings that are fields, i.e. where every
/// nonzero element has an inverse.
/// 
/// Note that fields must be commutative.
/// 
pub trait Field: EuclideanRing {}

pub trait FieldWrapper: RingStore<Type: Field> + EuclideanRingStore {

    fn div(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> {
        assert!(!self.is_zero(rhs));
        self.checked_div(lhs, rhs).unwrap()
    }
}

impl<R> FieldWrapper for R
    where R: RingStore, R::Type: Field
{}