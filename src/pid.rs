use crate::computation::ComputationController;
use crate::ring::*;
use crate::divisibility::*;

///
/// Trait for rings that are principal ideal rings, i.e. every ideal is generated
/// by a single element. 
/// 
/// A principal ideal ring currently must be commutative, since otherwise we would
/// have to distinguish left-, right-and two-sided ideals.
/// 
pub trait PrincipalIdealRing: DivisibilityRing {

    ///
    /// Similar to [`DivisibilityRing::checked_left_div()`] this computes a "quotient" `q`
    /// of `lhs` and `rhs`, if it exists. However, we impose the additional constraint
    /// that this quotient be minimal, i.e. there is no `q'` with `q' | q` properly and
    /// `q' * rhs = lhs`.
    /// 
    /// In domains, this is always satisfied, i.e. this function behaves exactly like
    /// [`DivisibilityRing::checked_left_div()`]. However, if there are zero-divisors, weird
    /// things can happen. For example in `Z/6Z`, `checked_div(4, 4)` may return `4`, however
    /// this is not minimal since `1 | 4` and `1 * 4 = 4`.
    /// 
    /// In particular, this function can be used to compute a generator of the annihilator ideal
    /// of an element `x`, by using it as `checked_div_min(0, x)`.
    /// 
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element>;

    ///
    /// Returns the (w.r.t. divisibility) smallest element `x` such that `x * val = 0`.
    /// 
    /// If the ring is a domain, this returns `0` for all ring elements except zero (for
    /// which it returns any unit).
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::pid::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let Z6 = Zn64B::new(6);
    /// assert_el_eq!(Z6, Z6.int_hom().map(3), Z6.annihilator(&Z6.int_hom().map(2)));
    /// ```
    /// 
    fn annihilator(&self, val: &Self::Element) -> Self::Element {
        self.checked_div_min(&self.zero(), val).unwrap()
    }

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
    /// Creates a matrix `A` of unit determinant such that `A * (a, b)^T = (d, 0)`.
    /// Returns `(A, d)`.
    /// 
    #[stability::unstable(feature = "enable")]
    fn create_elimination_matrix(&self, a: &Self::Element, b: &Self::Element) -> ([Self::Element; 4], Self::Element) {
        assert!(self.is_commutative());
        let old_gcd = self.ideal_gen(a, b);
        let new_a = self.checked_div_min(a, &old_gcd).unwrap();
        let new_b = self.checked_div_min(b, &old_gcd).unwrap();
        let (s, t, gcd_new) = self.extended_ideal_gen(&new_a, &new_b);
        debug_assert!(self.is_unit(&gcd_new));
        
        let subtract_factor = self.checked_left_div(&self.sub(self.mul_ref(b, &new_a), self.mul_ref(a, &new_b)), &gcd_new).unwrap();
        // this has unit determinant and will map `(a, b)` to `(d, b * new_a - a * new_b)`; after a subtraction step, we are done
        let mut result = [s, t, self.negate(new_b), new_a];
    
        let sub1 = self.mul_ref(&result[0], &subtract_factor);
        self.sub_assign(&mut result[2], sub1);
        let sub2 = self.mul_ref_fst(&result[1], subtract_factor);
        self.sub_assign(&mut result[3], sub2);
        debug_assert!(self.is_unit(&self.sub(self.mul_ref(&result[0], &result[3]), self.mul_ref(&result[1], &result[2]))));
        return (result, gcd_new);
    }
    

    ///
    /// Computes a generator `g` of the ideal `(lhs, rhs) = (g)`, also known as greatest
    /// common divisor of `lhs` and `rhs`.
    /// 
    /// If you require also a Bezout identiy, i.e. `g = s * lhs + t * rhs`, consider
    /// using [`PrincipalIdealRing::extended_ideal_gen()`].
    /// 
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.extended_ideal_gen(lhs, rhs).2
    }

    ///
    /// Returns a generator `g` of the ideal `(lhs) n (rhs) = (g)`, also known as least
    /// common multiple of `lhs` and `rhs`.
    /// 
    fn ideal_intersect(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element
        where Self: Domain
    {
        self.checked_div(&self.mul_ref(lhs, rhs), &self.ideal_gen(lhs, rhs)).unwrap()
    }

    ///
    /// As [`PrincipalIdealRing::ideal_gen()`], this computes a generator of the ideal `(lhs, rhs)`.
    /// However, it additionally accepts a [`ComputationController`] to customize the performed
    /// computation.
    /// 
    fn ideal_gen_with_controller<Controller>(&self, lhs: &Self::Element, rhs: &Self::Element, _: Controller) -> Self::Element
        where Controller: ComputationController
    {
        self.ideal_gen(lhs, rhs)
    }

    ///
    /// Returns the ring element `x` that is `= a mod p` and `= b mod q`, assuming
    /// that `p` and `q` are coprime.
    /// 
    /// Panics if `p, q` are not coprime.
    /// 
    fn inv_crt(&self, [a, b]: [&Self::Element; 2], [p, q]: [&Self::Element; 2]) -> Self::Element {
        let (s, t, d) = self.extended_ideal_gen(p, q);
        assert!(self.is_unit(&d));
        self.checked_div(
            &self.fma(
                b, 
                &self.mul_ref_fst(p, s),
                self.mul_ref_fst(a, self.mul_ref_fst(q, t))
            ),
            &d
        ).unwrap()
    }
}

///
/// Trait for [`RingStore`]s that store [`PrincipalIdealRing`]s. Mainly used
/// to provide a convenient interface to the `PrincipalIdealRing`-functions.
/// 
pub trait PrincipalIdealRingStore: RingStore
    where Self::Type: PrincipalIdealRing
{
    delegate!{ PrincipalIdealRing, fn checked_div_min(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
    delegate!{ PrincipalIdealRing, fn extended_ideal_gen(&self, lhs: &El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>, El<Self>) }
    delegate!{ PrincipalIdealRing, fn ideal_gen(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ PrincipalIdealRing, fn inv_crt(&self, congruence: [&El<Self>; 2], modulus: [&El<Self>; 2]) -> El<Self> }
    delegate!{ PrincipalIdealRing, fn annihilator(&self, val: &El<Self>) -> El<Self> }

    ///
    /// See [`PrincipalIdealRing::ideal_intersect()`].
    /// 
    fn ideal_intersect(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self>
        where Self::Type: Domain
    {
        self.get_ring().ideal_intersect(lhs, rhs)
    }

    ///
    /// Alias for [`PrincipalIdealRingStore::ideal_gen()`].
    /// 
    fn gcd(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> {
        self.ideal_gen(lhs, rhs)
    }

    ///
    /// Alias for [`PrincipalIdealRingStore::ideal_intersect()`].
    /// 
    fn lcm(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self>
        where Self::Type: Domain
    {
        self.ideal_intersect(lhs, rhs)
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
/// ```rust
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

#[allow(missing_docs)]
#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use super::*;
    use crate::{algorithms::int_factor::factor, homomorphism::Homomorphism, integer::{int_cast, BigIntRing, IntegerRingStore}, ordered::OrderedRingStore, primitive_int::StaticRing, ring::El};

    pub fn test_euclidean_ring_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) 
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

    pub fn test_principal_ideal_ring_axioms<R: RingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: PrincipalIdealRing
    {
        assert!(ring.is_commutative());
        assert!(ring.is_noetherian());
        let elements = edge_case_elements.collect::<Vec<_>>();

        let expected_unit = ring.checked_div_min(&ring.zero(), &ring.zero()).unwrap();
        assert!(ring.is_unit(&expected_unit), "checked_div_min() returned a non-minimal quotient {} * {} = {}", ring.formatted_el(&expected_unit), ring.formatted_el(&ring.zero()), ring.formatted_el(&ring.zero()));
        let expected_zero = ring.checked_div_min(&ring.zero(), &ring.one()).unwrap();
        assert!(ring.is_zero(&expected_zero), "checked_div_min() returned a wrong quotient {} * {} = {}", ring.formatted_el(&expected_zero), ring.formatted_el(&ring.one()), ring.formatted_el(&ring.zero()));

        for a in &elements {
            let expected_unit = ring.checked_div_min(a, a).unwrap();
            assert!(ring.is_unit(&expected_unit), "checked_div_min() returned a non-minimal quotient {} * {} = {}", ring.formatted_el(&expected_unit), ring.formatted_el(a), ring.formatted_el(a));
        }

        for a in &elements {
            for b in &elements {
                for c in &elements {
                    let g1 = ring.mul_ref(a, b);
                    let g2 = ring.mul_ref(a, c);
                    let (s, t, g) = ring.extended_ideal_gen(&g1, &g2);
                    assert!(ring.checked_div(&g, a).is_some(), "Wrong ideal generator: ({}) contains the ideal I = ({}, {}), but extended_ideal_gen() found a generator I = ({}) that does not satisfy {} | {}", ring.formatted_el(a), ring.formatted_el(&g1), ring.formatted_el(&g2), ring.formatted_el(&g), ring.formatted_el(a), ring.formatted_el(&g));
                    assert_el_eq!(ring, g, ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
                }
            }
        }
        for a in &elements {
            for b in &elements {
                let g1 = ring.mul_ref(a, b);
                let g2 = ring.mul_ref_fst(a, ring.add_ref_fst(b, ring.one()));
                let (s, t, g) = ring.extended_ideal_gen(&g1, &g2);
                assert!(ring.checked_div(&g, a).is_some() && ring.checked_div(a, &g).is_some(), "Expected ideals ({}) and I = ({}, {}) to be equal, but extended_ideal_gen() returned generator {} of I", ring.formatted_el(a), ring.formatted_el(&g1), ring.formatted_el(&g2), ring.formatted_el(&g));
                assert_el_eq!(ring, g, ring.add(ring.mul_ref(&s, &g1), ring.mul_ref(&t, &g2)));
            }
        }

        let ZZbig = BigIntRing::RING;
        let char = ring.characteristic(ZZbig).unwrap();
        if !ZZbig.is_zero(&char) && !ZZbig.is_one(&char) && ZZbig.is_leq(&char, &ZZbig.power_of_two(30)) {
            let p = factor(ZZbig, ZZbig.clone_el(&char)).into_iter().next().unwrap().0;
            let expected = ring.int_hom().map(int_cast(ZZbig.checked_div(&char, &p).unwrap(), StaticRing::<i32>::RING, ZZbig));
            let ann_p = ring.annihilator(&ring.int_hom().map(int_cast(p, StaticRing::<i32>::RING, ZZbig)));
            assert!(ring.checked_div(&ann_p, &expected).is_some());
            assert!(ring.checked_div(&expected, &ann_p).is_some());
        }
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_inv_crt() {
    let ring = StaticRing::<i64>::RING;
    assert_eq!(4, (ring.inv_crt([&4, &3], [&7, &9]) % 7 + 7) % 7);
    assert_eq!(3, (ring.inv_crt([&4, &3], [&7, &9]) % 9 + 9) % 9);
    assert_eq!(2, (ring.inv_crt([&-2, &3], [&4, &5]) % 4 + 4) % 4);
    assert_eq!(3, (ring.inv_crt([&-2, &3], [&4, &5]) % 5 + 5) % 5);
    assert_eq!(1, (ring.inv_crt([&-3, &10], [&4, &25]) % 4 + 4) % 4);
    assert_eq!(10, (ring.inv_crt([&-3, &10], [&4, &25]) % 25 + 25) % 25);
}