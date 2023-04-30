use crate::ring::*;

///
/// Trait for rings that support checking divisibility, i.e.
/// whether for `x, y` there is `k` such that `x = ky`.
/// 
pub trait DivisibilityRing: RingBase {

    ///
    /// Checks whether there is an element x such that `rhs * x = lhs`, and
    /// returns it if it exists. Note that this does not have to be unique, if
    /// rhs is a left zero-divisor. In particular, this function will return any
    /// element in the ring if `lhs = rhs = 0`.
    /// 
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element>;

    fn is_unit(&self, x: &Self::Element) -> bool {
        self.checked_left_div(&self.one(), x).is_some()
    }
}

pub trait DivisibilityRingStore: RingStore<Type: DivisibilityRing> {

    delegate!{ fn checked_left_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
    delegate!{ fn is_unit(&self, x: &El<Self>) -> bool }

    fn checked_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> {
        assert!(self.is_commutative());
        self.checked_left_div(lhs, rhs)
    }
}

impl<R> DivisibilityRingStore for R
    where R: RingStore, R::Type: DivisibilityRing
{}

#[cfg(test)]
pub fn generic_test_divisibility_axioms<R: DivisibilityRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) {
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            let ab = ring.mul(a.clone(), b.clone());
            let c = ring.checked_left_div(&ab, &a);
            assert!(c.is_some());
            assert!(ring.eq(&ab, &ring.mul(a.clone(), c.unwrap())));

            if !ring.is_unit(b) {
                assert!(ring.checked_left_div(&ring.add(ab.clone(), ring.one()), &b).is_none());
                assert!(ring.checked_left_div(&ring.sub(ab.clone(), ring.one()), &b).is_none());
            }
        }
    }
}