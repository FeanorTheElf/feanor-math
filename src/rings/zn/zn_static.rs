use crate::algorithms::eea::*;
use crate::euclidean::EuclideanRing;
use crate::field::Field;
use crate::{divisibility::*, Exists, Expr};
use crate::primitive_int::{StaticRing, StaticRingBase};
use crate::ring::*;
use crate::rings::zn::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZnBase<const N: u64, const IS_FIELD: bool>;

pub const fn is_prime(n: u64) -> bool {
    assert!(n >= 2);
    let mut d = 2;
    while d < n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    return true;
}

impl<const N: u64, const IS_FIELD: bool> RingBase for ZnBase<N, IS_FIELD> 
    where Expr<{N as i64 as usize}>: Exists
{
    type Element = u64;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs += rhs;
        if *lhs >= N {
            *lhs -= N;
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        if *lhs != 0 {
            *lhs = N - *lhs;
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = ((*lhs as u128 * rhs as u128) % (N as u128)) as u64
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i64>::RING, value as i64)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        *lhs == *rhs
    }
    
    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", *value)
    }
}

impl<const N: u64, const IS_FIELD: bool> CanonicalHom<StaticRingBase<i64>> for ZnBase<N, IS_FIELD>
    where Expr<{N as i64 as usize}>: Exists
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &StaticRingBase<i64>) -> Option<()> { Some(()) }

    fn map_in(&self, _: &StaticRingBase<i64>, el: i64, _: &()) -> Self::Element {
        let result = ((el % (N as i64)) + (N as i64)) as u64;
        if result >= N {
            result - N
        } else {
            result
        }
    }
}


impl<const N: u64, const IS_FIELD: bool> CanonicalHom<ZnBase<N, IS_FIELD>> for ZnBase<N, IS_FIELD>
    where Expr<{N as i64 as usize}>: Exists
{
    type Homomorphism = ();
    fn has_canonical_hom(&self, _: &Self) -> Option<()> { Some(()) }
    fn map_in(&self, _: &Self, el: Self::Element, _: &()) -> Self::Element { el }
}

impl<const N: u64, const IS_FIELD: bool> CanonicalIso<ZnBase<N, IS_FIELD>> for ZnBase<N, IS_FIELD>
    where Expr<{N as i64 as usize}>: Exists
{
    type Isomorphism = ();
    fn has_canonical_iso(&self, _: &Self) -> Option<()> { Some(()) }
    fn map_out(&self, _: &Self, el: Self::Element, _: &()) -> Self::Element { el }
}

impl<const N: u64, const IS_FIELD: bool> DivisibilityRing for ZnBase<N, IS_FIELD>
    where Expr<{N as i64 as usize}>: Exists
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let (s, _, d) = signed_eea(*rhs as i64, N as i64, StaticRing::<i64>::RING);
        let mut rhs_inv = ((s % (N as i64)) + (N as i64)) as u64;
        if rhs_inv >= N {
            rhs_inv -= N;
        }
        if d == 1 {
            Some(self.mul(*lhs, rhs_inv))
        } else {
            None
        }
    }
}

impl<const N: u64> EuclideanRing for ZnBase<N, true>
    where Expr<{N as i64 as usize}>: Exists
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }
}
pub struct ZnBaseElementsIter<const N: u64> {
    current: u64
}

impl<const N: u64> Iterator for ZnBaseElementsIter<N> {

    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < N {
            self.current += 1;
            return Some(self.current - 1);
        } else {
            return None;
        }
    }
}

impl<const N: u64, const IS_FIELD: bool> FiniteRing for ZnBase<N, IS_FIELD> 
    where Expr<{N as i64 as usize}>: Exists
{
    type ElementsIter<'a> = ZnBaseElementsIter<N>;

    fn elements<'a>(&'a self) -> ZnBaseElementsIter<N> {
        ZnBaseElementsIter { current: 0 }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        generic_impls::generic_random_element(self, rng)
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> El<I>
        where I::Type: IntegerRing
    {
        int_cast(*self.modulus(), ZZ, self.integer_ring())
    }
}

impl<const N: u64, const IS_FIELD: bool> ZnRing for ZnBase<N, IS_FIELD> 
    where Expr<{N as i64 as usize}>: Exists
{
    type IntegerRingBase = StaticRingBase<i64>;
    type Integers = RingValue<StaticRingBase<i64>>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        el as i64
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &(N as i64)
    }
}

impl<const N: u64> Field for ZnBase<N, true>
    where Expr<{N as i64 as usize}>: Exists
{}

impl<const N: u64, const IS_FIELD: bool> RingValue<ZnBase<N, IS_FIELD>>
    where Expr<{N as i64 as usize}>: Exists
{
    pub const RING: Self = Self::from(ZnBase);
}

pub type Zn<const N: u64> = RingValue<ZnBase<N, {is_prime(N)}>>;

#[test]
fn test_is_prime() {
    assert_eq!(true, is_prime(17));
    assert_eq!(false, is_prime(49));
}

pub const Z17: Zn<17> = Zn::<17>::RING;

#[test]
fn test_zn_el_add() {
    let a = Z17.from_int(6);
    let b = Z17.from_int(12);
    assert_eq!(Z17.from_int(1), Z17.add(a, b));
}

#[test]
fn test_zn_el_sub() {
    let a = Z17.from_int(6);
    let b = Z17.from_int(12);
    assert_eq!(Z17.from_int(11), Z17.sub(a, b));
}

#[test]
fn test_zn_el_mul() {
    let a = Z17.from_int(6);
    let b = Z17.from_int(12);
    assert_eq!(Z17.from_int(4), Z17.mul(a, b));
}

#[test]
fn test_zn_el_div() {
    let a = Z17.from_int(6);
    let b = Z17.from_int(12);
    assert_eq!(Z17.from_int(9), Z17.checked_div(&a, &b).unwrap());
}

#[test]
fn fn_test_div_impossible() {
    let _a = Zn::<22>::RING.from_int(4);
    // the following line should give a compiler error
    // Zn::<22>::RING.div(_a, _a);
}

#[test]
fn test_zn_ring_axioms_znbase() {
    generic_test_zn_ring_axioms(Zn::<17>::RING);
    generic_test_zn_ring_axioms(Zn::<63>::RING);
}
