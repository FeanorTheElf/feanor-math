use crate::ring::*;

///
/// Trait for rings that support checking divisibility, i.e.
/// whether for `x, y` there is `k` such that `x = ky`.
/// 
pub trait DivisibilityRing: RingBase {

    ///
    /// Checks whether there is an element `x` such that `rhs * x = lhs`, and
    /// returns it if it exists. Note that this does not have to be unique, if
    /// rhs is a left zero-divisor. In particular, this function will return any
    /// element in the ring if `lhs = rhs = 0`.
    /// 
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element>;

    ///
    /// Returns whether the given element is a unit, i.e. has an inverse.
    /// 
    fn is_unit(&self, x: &Self::Element) -> bool {
        self.checked_left_div(&self.one(), x).is_some()
    }
}

///
/// Trait for rings that support "preparing" division by `x`, i.e. compute some
/// additional data for `x` that can later be used to speed up division by `x`.
/// 
/// The semantics are the same as for [`DivisibilityRing`], just the performance 
/// behavior is different.
/// 
#[stability::unstable(feature = "enable")]
pub trait PreparedDivisibilityRing: DivisibilityRing {

    type PreparedDivisor;

    fn prepare_divisor(&self, x: &Self::Element) -> Self::PreparedDivisor;

    fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &Self::PreparedDivisor) -> Option<Self::Element>;

    fn is_unit_prepared(&self, x: &Self::PreparedDivisor) -> bool {
        self.checked_left_div_prepared(&self.one(), x).is_some()
    }
}

///
/// Trait for rings that are integral, i.e. have no zero divisors.
/// 
/// A zero divisor is a nonzero element `a` such that there is a nonzero
/// element `b` with `ab = 0`.
/// 
pub trait Domain: DivisibilityRing {}

///
/// Trait for [`RingStore`]s that store [`DivisibilityRing`]s. Mainly used
/// to provide a convenient interface to the `DivisibilityRing`-functions.
/// 
pub trait DivisibilityRingStore: RingStore
    where Self::Type: DivisibilityRing
{
    delegate!{ DivisibilityRing, fn checked_left_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
    delegate!{ DivisibilityRing, fn is_unit(&self, x: &El<Self>) -> bool }

    fn checked_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> {
        assert!(self.is_commutative());
        self.checked_left_div(lhs, rhs)
    }

    fn invert(&self, el: &El<Self>) -> Option<El<Self>> {
        self.checked_div(&self.one(), el)
    }
}

impl<R> DivisibilityRingStore for R
    where R: RingStore, R::Type: DivisibilityRing
{}

///
/// Trait for [`RingStore`]s that store [`PreparedDivisibilityRing`]s. Mainly used
/// to provide a convenient interface to the `PreparedDivisibilityRing`-functions.
/// 
#[stability::unstable(feature = "enable")]
pub trait PreparedDivisibilityRingStore: RingStore
    where Self::Type: PreparedDivisibilityRing
{
    delegate!{ PreparedDivisibilityRing, fn prepare_divisor(&self, x: &El<Self>) -> <Self::Type as PreparedDivisibilityRing>::PreparedDivisor }
    delegate!{ PreparedDivisibilityRing, fn checked_left_div_prepared(&self, lhs: &El<Self>, rhs: &<Self::Type as PreparedDivisibilityRing>::PreparedDivisor) -> Option<El<Self>> }
    delegate!{ PreparedDivisibilityRing, fn is_unit_prepared(&self, x: &<Self::Type as PreparedDivisibilityRing>::PreparedDivisor) -> bool }

    fn checked_div_prepared(&self, lhs: &El<Self>, rhs: &<Self::Type as PreparedDivisibilityRing>::PreparedDivisor) -> Option<El<Self>> {
        assert!(self.is_commutative());
        self.checked_left_div_prepared(lhs, rhs)
    }

    fn invert_prepared(&self, el: &<Self::Type as PreparedDivisibilityRing>::PreparedDivisor) -> Option<El<Self>> {
        self.checked_div_prepared(&self.one(), el)
    }
}

impl<R> PreparedDivisibilityRingStore for R
    where R: RingStore, R::Type: PreparedDivisibilityRing
{}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::ring::El;
    use super::*;

    pub fn test_divisibility_axioms<R: DivisibilityRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: DivisibilityRing
    {
        let elements = edge_case_elements.collect::<Vec<_>>();

        for a in &elements {
            for b in &elements {
                let ab = ring.mul(ring.clone_el(a), ring.clone_el(b));
                let c = ring.checked_left_div(&ab, &a);
                assert!(c.is_some(), "Divisibility existence failed: there should exist b = {} such that {} = b * {}, but none was found", ring.format(b), ring.format(&ab), ring.format(&a));
                assert!(ring.eq_el(&ab, &ring.mul_ref_snd(ring.clone_el(a), c.as_ref().unwrap())), "Division failed: {} * {} != {} but {} = checked_div({}, {})", ring.format(a), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(&a));

                if !ring.is_unit(a) {
                    let ab_plus_one = ring.add(ring.clone_el(&ab), ring.one());
                    let c = ring.checked_left_div(&ab_plus_one, &a);
                    assert!(c.is_none(), "Unit check failed: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_plus_one), ring.format(a), ring.format(c.as_ref().unwrap()));

                    let ab_minus_one = ring.sub(ring.clone_el(&ab), ring.one());
                    let c = ring.checked_left_div(&ab_minus_one, &a);
                    assert!(c.is_none(), "Unit check failed: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_minus_one), ring.format(a), ring.format(c.as_ref().unwrap()));
                } else {
                    let inv = ring.checked_left_div(&ring.one(), a);
                    assert!(inv.is_some(), "Unit check failed: is_unit({}) is true but checked_div({}, {}) is None", ring.format(a), ring.format(&ring.one()), ring.format(&a));
                    let prod = ring.mul_ref(a, inv.as_ref().unwrap());
                    assert!(ring.eq_el(&ring.one(), &prod), "Division failed: {} != {} * {} but checked_div({}, {}) = {}", ring.format(&ring.one()), ring.format(a), ring.format(inv.as_ref().unwrap()), ring.format(&ring.one()), ring.format(a), ring.format(c.as_ref().unwrap()));
                }
            }
        }
    }
}