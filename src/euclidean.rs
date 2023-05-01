use crate::ring::*;
use crate::divisibility::*;

///
/// Trait for rings that support euclidean division.
/// 
/// In other words, there is a degree function d(.) 
/// returning positive integers such that for every `x, y` 
/// with `y != 0` there are `q, r` with `x = qy + r` and 
/// `d(r) < d(y)`. Note that `q, r` do not have to be unique, 
/// and implementations are free to use any choice.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::euclidean::*;
/// # use feanor_math::primitive_int::*;
/// let ring = StaticRing::<i64>::RING;
/// let (q, r) = ring.euclidean_div_rem(14, &6);
/// assert!(ring.eq(&14, &ring.add(ring.mul(q, 6), r)));
/// assert!(ring.euclidean_deg(&r) < ring.euclidean_deg(&6));
/// ```
/// 
pub trait EuclideanRing: DivisibilityRing {

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize>;

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).0
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).1
    }
}

pub trait EuclideanRingStore: RingStore<Type: EuclideanRing> + DivisibilityRingStore {

    delegate!{ fn euclidean_div_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }
    delegate!{ fn euclidean_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_deg(&self, val: &El<Self>) -> Option<usize> }
}

impl<R> EuclideanRingStore for R
    where R: RingStore, R::Type: EuclideanRing
{}

#[cfg(test)]
pub fn generic_test_euclidean_axioms<R: EuclideanRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) {
    assert!(ring.is_commutative());
    assert!(ring.is_noetherian());
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            if ring.is_zero(b) {
                continue;
            }
            let (q, r) = ring.euclidean_div_rem(ring.clone(a), b);
            assert!(ring.euclidean_deg(b).is_none() || ring.euclidean_deg(&r).unwrap_or(usize::MAX) < ring.euclidean_deg(b).unwrap());
            assert!(ring.eq(a, &ring.add(ring.mul(q, ring.clone(b)), r)));
        }
    }
}