use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::matmul::StrassenHint;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::field::Field;
use crate::homomorphism::CanHomFrom;
use crate::integer::{int_cast, IntegerRing, IntegerRingStore};
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::algorithms;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::ring::*;

///
/// An implementation of the rational number `Q`, based on representing them
/// as a tuple `(numerator, denominator)`.
/// 
/// Be careful when instantiating it with finite-precision integers, like `StaticRing<i64>`,
/// since by nature of the rational numbers, both numerator and denominator can increase
/// dramatically, even when the numbers itself are of moderate size.
/// 
#[derive(Debug, Clone, Copy)]
pub struct RationalFieldBase<I: IntegerRingStore>
    where I::Type: IntegerRing
{
    integers: I
}

///
/// [`RingStore`] corresponding to [`RationalFieldBase`]
/// 
pub type RationalField<I> = RingValue<RationalFieldBase<I>>;

impl<I> PartialEq for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.integers.get_ring() == other.integers.get_ring()
    }
}

impl<I> RationalField<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    pub const fn new(integers: I) -> Self {
        RingValue::from(RationalFieldBase { integers })
    }
}

impl<I> RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn reduce(&self, value: (&mut El<I>, &mut El<I>)) {
        // take the denominator first, as in this case gcd will have the same sign, and the final denominator will be positive
        let gcd = algorithms::eea::signed_gcd(self.integers.clone_el(&*value.1), self.integers.clone_el(&*value.0), &self.integers);
        *value.0 = self.integers.checked_div(&*value.0, &gcd).unwrap();
        *value.1 = self.integers.checked_div(&*value.1, &gcd).unwrap();
    }

    fn mul_assign_raw(&self, lhs: &mut <Self as RingBase>::Element, rhs: (&El<I>, &El<I>)) {
        self.integers.mul_assign_ref(&mut lhs.0, rhs.0);
        self.integers.mul_assign_ref(&mut lhs.1, rhs.1);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }
}

impl<I> RingBase for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    type Element = (El<I>, El<I>);

    fn add_assign(&self, lhs: &mut Self::Element, mut rhs: Self::Element) {
        self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
        self.integers.mul_assign_ref(&mut rhs.0, &lhs.1);
        self.integers.mul_assign(&mut lhs.1, rhs.1);
        self.integers.add_assign(&mut lhs.0, rhs.0);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        (self.integers.clone_el(&val.0), self.integers.clone_el(&val.1))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.integers.mul_assign_ref(&mut lhs.0, &rhs.1);
        self.integers.add_assign(&mut lhs.0, self.integers.mul_ref(&lhs.1, &rhs.0));
        self.integers.mul_assign_ref(&mut lhs.1, &rhs.1);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.mul_assign_raw(lhs, (&rhs.0, &rhs.1))
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.integers.mul_assign(&mut lhs.0, rhs.0);
        self.integers.mul_assign(&mut lhs.1, rhs.1);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.integers.negate_inplace(&mut lhs.0);
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.integers.eq_el(&self.integers.mul_ref(&lhs.0, &rhs.1), &self.integers.mul_ref(&lhs.1, &rhs.0))
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.integers.is_zero(&value.0)
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        self.integers.eq_el(&value.0, &value.1)
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        self.integers.eq_el(&value.0, &self.integers.negate(self.integers.clone_el(&value.1)))
    }

    fn is_approximate(&self) -> bool {
        false
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn characteristic<J: IntegerRingStore>(&self, ZZ: &J) -> Option<El<J>>
        where J::Type: IntegerRing
    {
        Some(ZZ.zero())
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        if self.base_ring().is_one(&value.1) {
            write!(out, "{}", self.integers.format(&value.0))
        } else {
            write!(out, "{}/{}", self.integers.format(&value.0), self.integers.format(&value.1))
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        (self.integers.get_ring().from_int(value), self.integers.one())
    }
}

impl<I: IntegerRingStore> StrassenHint for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    default fn strassen_threshold(&self) -> usize {
        usize::MAX
    }
}

impl<I: IntegerRingStore> KaratsubaHint for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    default fn karatsuba_threshold(&self) -> usize {
        usize::MAX
    }
}

impl<I> RingExtension for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    type BaseRing = I;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.integers
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        (x, self.integers.one())
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.integers.mul_assign_ref(&mut lhs.0, rhs);
        self.reduce((&mut lhs.0, &mut lhs.1));
    }
}

impl<I, J> CanHomFrom<RationalFieldBase<J>> for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        J: IntegerRingStore,
        J::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &RationalFieldBase<J>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &RationalFieldBase<J>, el: <RationalFieldBase<J> as RingBase>::Element, (): &Self::Homomorphism) -> Self::Element {
        (int_cast(el.0, self.base_ring(), from.base_ring()), int_cast(el.1, self.base_ring(), from.base_ring()))
    }
}

impl<I, J> CanHomFrom<J> for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        J: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &J) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &J, el: <J as RingBase>::Element, (): &Self::Homomorphism) -> Self::Element {
        (int_cast(el, self.base_ring(), &RingRef::new(from)), self.integers.one())
    }
}

impl<I> DivisibilityRing for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            Some(self.zero())
        } else if self.is_zero(rhs) {
            None
        } else {
            let mut result = self.clone_el(lhs);
            self.mul_assign_raw(&mut result, (&rhs.1, &rhs.0));
            Some(result)
        }
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        !self.is_zero(x)
    }
}

impl<I> PrincipalIdealRing for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return (self.zero(), self.zero(), self.zero());
        } else if self.is_zero(lhs) {
            return (self.zero(), self.one(), self.clone_el(rhs));
        } else {
            return (self.one(), self.zero(), self.clone_el(lhs));
        }
    }
}

impl<I> EuclideanRing for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) { Some(0) } else { Some(1) }
    }

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }
}

impl<I> Domain for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{}

impl<I> Field for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{}

impl<I> OrderedRing for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        assert!(self.integers.is_pos(&lhs.1) && self.integers.is_pos(&rhs.1));
        self.integers.cmp(&self.integers.mul_ref(&lhs.0, &rhs.1), &self.integers.mul_ref(&rhs.0, &lhs.1))
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::homomorphism::Homomorphism;

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = El<RationalField<StaticRing<i64>>>> {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    let incl = ring.into_int_hom();
    (-6..8).flat_map(move |x| (-2..5).filter(|y| *y != 0).map(move |y| ring.checked_div(&incl.map(x), &incl.map(y)).unwrap()))
}

#[test]
fn test_ring_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);

    let half = ring.checked_div(&ring.int_hom().map(1), &ring.int_hom().map(2)).unwrap();
    assert!(!ring.is_one(&half));
    assert!(!ring.is_zero(&half));
    assert_el_eq!(&ring, &ring.one(), &ring.add_ref(&half, &half));
    crate::ring::generic_tests::test_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_divisibility_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::divisibility::generic_tests::test_divisibility_axioms(ring, edge_case_elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::pid::generic_tests::test_euclidean_ring_axioms(ring, edge_case_elements());
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(ring, edge_case_elements());
}

#[test]
fn test_int_hom_axioms() {
    let ring = RationalField::new(StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hom_axioms(&StaticRing::<i64>::RING, ring, -16..15);
}