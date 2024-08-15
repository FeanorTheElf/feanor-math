use crate::ring::*;
use crate::divisibility::*;

///
/// Trait for rings that are principal ideal rings, i.e. every ideal is generated
/// by a single element. 
/// 
pub trait PrincipalIdealRing: DivisibilityRing {

    // TODO: when doing the next breaking change, add a function for the annihilator generator of an element / smallest division result

    ///
    /// Computes a Bezout identity for the generator `g` of the ideal `(lhs, rhs)`
    /// as `g = s * lhs + t * rhs`.
    /// 
    /// More concretely, this returns (s, t, g) such that g is a generator 
    /// of the ideal `(lhs, rhs)` and `g = s * lhs + t * rhs`. This `g` is also known
    /// as the greatest common divisor of `lhs` and `rhs`, since `g | lhs, rhs` and
    /// for every `g'` with this property, have `g' | g`. Note that this `g` is only
    /// unique up to multiplication by units.
    /// 
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element);

    ///
    /// Cancels out the gcd of lhs and rhs.
    /// 
    /// More concretely, computes `(a, b, d)` such that `a * d = lhs`, `b * d = rhs` and `a, b` are coprime.
    /// 
    /// # Warning
    /// 
    /// For backwards compatibility, a default implementation that returns `(lhs.checked_div(gcd(lhs, rhs)), rhs.checked_div(gcd(lhs, rhs)), gcd(lhs, rhs))`
    /// is provided. This default implementation **is not guaranteed to always work correctly**. In particular, by the contract of 
    /// [`DivisibilityRing::checked_left_div()`], the returned "quotient" does not have to be minimal w.r.t. divisibility, so it is perfectly
    /// fine to have `6.checked_div(6) = 3 mod 12`, hence the naive implementation would map `(6, 6) -> (3, 3, 6)` with `3, 3` not coprime.
    /// The implementation checks this and panics in such cases, so when the function returns successfully, the result is correct.
    /// 
    /// Note that it will always be correct for [`Domain`]s and [`crate::local::PrincipalLocalRing`]s. Also, natural implementations of
    /// [`DivisibilityRing::checked_left_div()`] often make this work out. Nevertheless, if your ring is neither integral nor local, consider
    /// providing a correct implementation.
    /// 
    /// Also, the default implementation will be removed at the next breaking update of feanor-math.
    /// 
    fn cancel_common_factors(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let gcd = self.ideal_gen(lhs, rhs);
        let fst = self.checked_left_div(lhs, &gcd).unwrap();
        let snd = self.checked_left_div(rhs, &gcd).unwrap();
        assert!(self.is_unit(&self.ideal_gen(&fst, &snd)), "Default implementation for cancel_common_factors() failed for {} and {}; See [`PrincipalIdealRing::cancel_common_factors()`] for an explanation of this error", RingRef::new(self).format(lhs), RingRef::new(self).format(rhs));
        return (fst, snd, gcd);
    }

    ///
    /// Creates a matrix `A` of unit determinant such that `A * (a, b)^T = (d, 0)`.
    /// Returns `(A, d)`.
    /// 
    #[stability::unstable(feature = "enable")]
    fn create_left_elimination_matrix(&self, a: &Self::Element, b: &Self::Element) -> ([Self::Element; 4], Self::Element) {
        let (new_a, new_b, gcd) = self.cancel_common_factors(a, b);
        let (s, t, gcd_new) = self.extended_ideal_gen(&new_a, &new_b);
        debug_assert!(self.is_unit(&gcd_new));
        
        let subtract_factor = self.checked_left_div(&self.sub(self.mul_ref(b, &new_a), self.mul_ref(a, &new_b)), &gcd).unwrap();
        // this has unit determinant and will map `(a, b)` to `(d, b * new_a - a * new_b)`; after a subtraction step, we are done
        let mut result = [s, t, self.negate(new_b), new_a];
    
        let sub1 = self.mul_ref(&result[0], &subtract_factor);
        self.sub_assign(&mut result[2], sub1);
        let sub2 = self.mul_ref_fst(&result[1], subtract_factor);
        self.sub_assign(&mut result[3], sub2);
        debug_assert!(self.is_unit(&self.sub(self.mul_ref(&result[0], &result[3]), self.mul_ref(&result[1], &result[2]))));
        return (result, gcd);
    }
    

    ///
    /// Computes a generator `g` of the ideal `(lhs, rhs) = (g)`, also known as greatest
    /// common divisor.
    /// 
    /// If you require also a Bezout identiy, i.e. `g = s * lhs + t * rhs`, consider
    /// using [`PrincipalIdealRing::extended_ideal_gen()`].
    /// 
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.extended_ideal_gen(lhs, rhs).2
    }

    ///
    /// Computes a generator of the ideal `(lhs) ∩ (rhs)`, also known as least common
    /// multiple.
    /// 
    /// In other words, computes a ring element `g` such that `lhs, rhs | g` and for every
    /// `g'` with this property, have `g | g'`. Note that such an `g` is only unique up to
    /// multiplication by units.
    /// 
    fn lcm(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.checked_left_div(&self.mul_ref(lhs, rhs), &self.ideal_gen(lhs, rhs)).unwrap()
    }
}

///
/// Trait for [`RingStore`]s that store [`PrincipalIdealRing`]s. Mainly used
/// to provide a convenient interface to the `PrincipalIdealRing`-functions.
/// 
pub trait PrincipalIdealRingStore: RingStore
    where Self::Type: PrincipalIdealRing
{
    delegate!{ PrincipalIdealRing, fn extended_ideal_gen(&self, lhs: &El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>, El<Self>) }
    delegate!{ PrincipalIdealRing, fn ideal_gen(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ PrincipalIdealRing, fn lcm(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }

    ///
    /// Alias for [`PrincipalIdealRingStore::ideal_gen()`].
    /// 
    fn gcd(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> {
        self.ideal_gen(lhs, rhs)
    }
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
/// assert_el_eq!(ring, 14, ring.add(ring.mul(q, 6), r));
/// assert!(ring.euclidean_deg(&r) < ring.euclidean_deg(&6));
/// ```
/// 
pub trait EuclideanRing: PrincipalIdealRing {

    ///
    /// Computes euclidean division with remainder.
    /// 
    /// In general, the euclidean division of `lhs` by `rhs` is a tuple `(q, r)` such that
    /// `lhs = q * rhs + r`, and `r` is "smaller" than "rhs". The notion of smallness is
    /// given by the smallness of the euclidean degree function [`EuclideanRing::euclidean_deg()`].
    /// 
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);

    ///
    /// Defines how "small" an element is. For details, see [`EuclideanRing`].
    /// 
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize>;

    ///
    /// Computes euclidean division without remainder.
    /// 
    /// In general, the euclidean division of `lhs` by `rhs` is a tuple `(q, r)` such that
    /// `lhs = q * rhs + r`, and `r` is "smaller" than "rhs". The notion of smallness is
    /// given by the smallness of the euclidean degree function [`EuclideanRing::euclidean_deg()`].
    /// 
    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.euclidean_div_rem(lhs, rhs).0
    }

    ///
    /// Computes only the remainder of euclidean division.
    /// 
    /// In general, the euclidean division of `lhs` by `rhs` is a tuple `(q, r)` such that
    /// `lhs = q * rhs + r`, and `r` is "smaller" than "rhs". The notion of smallness is
    /// given by the smallness of the euclidean degree function [`EuclideanRing::euclidean_deg()`].
    /// 
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
    delegate!{ EuclideanRing, fn euclidean_div_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }
    delegate!{ EuclideanRing, fn euclidean_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ EuclideanRing, fn euclidean_rem(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ EuclideanRing, fn euclidean_deg(&self, val: &El<Self>) -> Option<usize> }
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
                assert_el_eq!(ring, a, ring.add(ring.mul(q, ring.clone_el(b)), r));
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
                    let (s, t, g) = ring.extended_ideal_gen(&g1, &g2);
                    assert!(ring.checked_div(&g, a).is_some(), "Wrong ideal generator: ({}) contains the ideal I = ({}, {}), but extended_ideal_gen() found a generator I = ({}) that does not satisfy {} | {}", ring.format(a), ring.format(&g1), ring.format(&g2), ring.format(&g), ring.format(a), ring.format(&g));
                    assert_el_eq!(ring, g, ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
                }
            }
        }
        for a in &elements {
            for b in &elements {
                let g1 = ring.mul_ref(a, b);
                let g2 = ring.mul_ref_fst(a, ring.add_ref_fst(b, ring.one()));
                let (s, t, g) = ring.extended_ideal_gen(&g1, &g2);
                assert!(ring.checked_div(&g, a).is_some() && ring.checked_div(a, &g).is_some(), "Expected ideals ({}) and I = ({}, {}) to be equal, but extended_ideal_gen() returned generator {} of I", ring.format(a), ring.format(&g1), ring.format(&g2), ring.format(&g));
                assert_el_eq!(ring, g, ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
            }
        }
    }
}