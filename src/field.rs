use crate::ring::*;
use crate::euclidean::*;

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