use crate::ring::*;
use crate::divisibility::*;

///
/// Trait for rings that are principal ideal rings, i.e. every ideal is generated
/// by a single element. 
/// 
pub trait PrincipalIdealRing: DivisibilityRing {

    ///
    /// Computes a Bezout identity.
    /// 
    /// More concretely, this returns (s, t, g) such that g is a generator 
    /// of the ideal `(lhs, rhs)` and `g = s * lhs + t * rhs`.
    /// 
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element);
}

///
/// [`RingStore`] for [`PrincipalIdealRing`]s
/// 
pub trait PrincipalIdealRingStore: RingStore
    where Self::Type: PrincipalIdealRing
{
    delegate!{ fn ideal_gen(&self, lhs: &El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>, El<Self>) }
}

impl<R> PrincipalIdealRingStore for R
    where R: RingStore,
        R::Type: PrincipalIdealRing
{}

///
/// Trait for rings that support euclidean division.
/// 
/// In other words, there is a degree function d(.) 
/// returning nonnegative integers such that for every `x, y` 
/// with `y != 0` there are `q, r` with `x = qy + r` and 
/// `d(r) < d(y)`. Note that `q, r` do not have to be unique, 
/// and implementations are free to use any choice.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::pid::*;
/// # use feanor_math::primitive_int::*;
/// let ring = StaticRing::<i64>::RING;
/// let (q, r) = ring.euclidean_div_rem(14, &6);
/// assert_el_eq!(&ring, &14, &ring.add(ring.mul(q, 6), r));
/// assert!(ring.euclidean_deg(&r) < ring.euclidean_deg(&6));
/// ```
/// 
pub trait EuclideanRing: PrincipalIdealRing {

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize>;

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).0
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).1
    }
}

///
/// [`RingStore`] for [`EuclideanRing`]s
/// 
pub trait EuclideanRingStore: RingStore + DivisibilityRingStore
    where Self::Type: EuclideanRing
{
    delegate!{ fn euclidean_div_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }
    delegate!{ fn euclidean_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn euclidean_deg(&self, val: &El<Self>) -> Option<usize> }
}

impl<R> EuclideanRingStore for R
    where R: RingStore, R::Type: EuclideanRing
{}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use super::*;
    use crate::ring::El;

    pub fn test_euclidean_ring_axioms<R: EuclideanRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) 
        where R::Type: EuclideanRing
    {
        assert!(ring.is_commutative());
        assert!(ring.is_noetherian());
        let elements = edge_case_elements.collect::<Vec<_>>();
        for a in &elements {
            for b in &elements {
                if ring.is_zero(b) {
                    continue;
                }
                let (q, r) = ring.euclidean_div_rem(ring.clone_el(a), b);
                assert!(ring.euclidean_deg(b).is_none() || ring.euclidean_deg(&r).unwrap_or(usize::MAX) < ring.euclidean_deg(b).unwrap());
                assert_el_eq!(&ring, a, &ring.add(ring.mul(q, ring.clone_el(b)), r));
            }
        }
    }

    pub fn test_principal_ideal_ring_axioms<R: PrincipalIdealRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: PrincipalIdealRing
    {
        assert!(ring.is_commutative());
        assert!(ring.is_noetherian());
        let elements = edge_case_elements.collect::<Vec<_>>();
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    let g1 = ring.mul_ref(a, b);
                    let g2 = ring.mul_ref(a, c);
                    let (s, t, g) = ring.ideal_gen(&g1, &g2);
                    assert!(ring.checked_div(&g, a).is_some(), "Wrong ideal generator: ({}) contains the ideal I = ({}, {}), but ideal_gen() found a generator I = ({}) that does not satisfy {} | {}", ring.format(a), ring.format(&g1), ring.format(&g2), ring.format(&g), ring.format(a), ring.format(&g));
                    assert_el_eq!(&ring, &g, &ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
                }
            }
        }
        for a in &elements {
            for b in &elements {
                let g1 = ring.mul_ref(a, b);
                let g2 = ring.mul_ref_fst(a, ring.add_ref_fst(b, ring.one()));
                let (s, t, g) = ring.ideal_gen(&g1, &g2);
                assert!(ring.checked_div(&g, a).is_some() && ring.checked_div(a, &g).is_some(), "Expected ideals ({}) and I = ({}, {}) to be equal, but ideal_gen() returned generator {} of I", ring.format(a), ring.format(&g1), ring.format(&g2), ring.format(&g));
                assert_el_eq!(&ring, &g, &ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
            }
        }
    }
}