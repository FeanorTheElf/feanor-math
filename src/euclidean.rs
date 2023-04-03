use crate::ring::*;
use crate::divisibility::*;

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
pub fn test_euclidean_axioms<R: EuclideanRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) {
    assert!(ring.is_commutative());
    assert!(ring.is_noetherian());
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            if ring.is_zero(b) {
                continue;
            }
            let (q, r) = ring.euclidean_div_rem(a.clone(), b);
            println!("{:?}, {:?}", ring.euclidean_deg(&r), ring.euclidean_deg(&b));
            assert!(ring.euclidean_deg(b).is_none() || ring.euclidean_deg(&r).unwrap_or(usize::MAX) < ring.euclidean_deg(b).unwrap());
            assert!(ring.eq(a, &ring.add(ring.mul(q, b.clone()), r)));
        }
    }
}