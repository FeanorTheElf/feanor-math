use crate::algorithms::eea::*;
use crate::local::PrincipalLocalRing;
use crate::pid::{EuclideanRing, PrincipalIdealRing, PrincipalIdealRingStore};
use crate::field::Field;
use crate::divisibility::*;
use crate::primitive_int::{StaticRing, StaticRingBase};
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::zn::*;

///
/// Ring that implements arithmetic in `Z/nZ` for a small `n` known
/// at compile time.
/// 
#[stability::unstable(feature = "enable")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZnBase<const N: u64, const IS_FIELD: bool>;

#[stability::unstable(feature = "enable")]
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

impl<const N: u64, const IS_FIELD: bool> ZnBase<N, IS_FIELD> {
    
    #[stability::unstable(feature = "enable")]
    pub const fn new() -> Self {
        assert!(!IS_FIELD || is_prime(N));
        ZnBase
    }
}

impl<const N: u64, const IS_FIELD: bool> RingBase for ZnBase<N, IS_FIELD> {
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
    
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.size(ZZ)
    }
}

impl<const N: u64, const IS_FIELD: bool> CanHomFrom<StaticRingBase<i64>> for ZnBase<N, IS_FIELD> {
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

impl<const N: u64, const IS_FIELD: bool> CanHomFrom<ZnBase<N, IS_FIELD>> for ZnBase<N, IS_FIELD> {
    type Homomorphism = ();
    fn has_canonical_hom(&self, _: &Self) -> Option<()> { Some(()) }
    fn map_in(&self, _: &Self, el: Self::Element, _: &()) -> Self::Element { el }
}

impl<const N: u64, const IS_FIELD: bool> CanIsoFromTo<ZnBase<N, IS_FIELD>> for ZnBase<N, IS_FIELD> {
    type Isomorphism = ();
    fn has_canonical_iso(&self, _: &Self) -> Option<()> { Some(()) }
    fn map_out(&self, _: &Self, el: Self::Element, _: &()) -> Self::Element { el }
}

impl<const N: u64, const IS_FIELD: bool> DivisibilityRing for ZnBase<N, IS_FIELD> {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let (s, _, d) = signed_eea(*rhs as i64, N as i64, StaticRing::<i64>::RING);
        let mut rhs_inv = ((s % (N as i64)) + (N as i64)) as u64;
        if rhs_inv >= N {
            rhs_inv -= N;
        }
        if *lhs % d as u64 == 0 {
            Some(self.mul(*lhs / d as u64, rhs_inv))
        } else {
            None
        }
    }
}

impl<const N: u64, const IS_FIELD: bool> PrincipalIdealRing for ZnBase<N, IS_FIELD> {
    
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = StaticRing::<i64>::RING.extended_ideal_gen(&(*lhs as i64), &(*rhs as i64));
        let quo = RingRef::new(self).into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        (quo.map(s), quo.map(t), quo.map(d))
    }
}

impl<const N: u64> EuclideanRing for ZnBase<N, true> {

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

#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
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

impl<const N: u64, const IS_FIELD: bool> HashableElRing for ZnBase<N, IS_FIELD> {
    
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        h.write_u64(*el);
    }
}

impl<const N: u64, const IS_FIELD: bool> FiniteRing for ZnBase<N, IS_FIELD> {
    type ElementsIter<'a> = ZnBaseElementsIter<N>;

    fn elements<'a>(&'a self) -> ZnBaseElementsIter<N> {
        ZnBaseElementsIter { current: 0 }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        generic_impls::random_element(self, rng)
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        if ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(self.modulus()) < ZZ.get_ring().representable_bits() {
            Some(int_cast(*self.modulus(), ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl<const N: u64, const IS_FIELD: bool> ZnRing for ZnBase<N, IS_FIELD> {
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

    fn is_field(&self) -> bool {
        is_prime(N)
    }

    fn from_int_promise_reduced(&self, x: El<Self::Integers>) -> Self::Element {
        debug_assert!(x >= 0);
        debug_assert!((x as u64) < N);
        x as u64
    }
}

impl<const N: u64> Domain for ZnBase<N, true> {}

impl<const N: u64> Field for ZnBase<N, true> {}

impl<const N: u64> PrincipalLocalRing for ZnBase<N, true> {
    
    fn max_ideal_gen(&self) ->  &Self::Element {
        &0
    }
}

impl<const N: u64, const IS_FIELD: bool> RingValue<ZnBase<N, IS_FIELD>> {

    #[stability::unstable(feature = "enable")]
    pub const RING: Self = Self::from(ZnBase::new());
}

///
/// Ring that implements arithmetic in `Z/nZ` for a small `n` known
/// at compile time. For details, see [`ZnBase`].
/// 
#[stability::unstable(feature = "enable")]
pub type Zn<const N: u64> = RingValue<ZnBase<N, false>>;

///
/// Ring that implements arithmetic in `Z/nZ` for a small `n` known
/// at compile time. For details, see [`ZnBase`].
/// 
#[stability::unstable(feature = "enable")]
pub type Fp<const P: u64> = RingValue<ZnBase<P, true>>;

#[test]
fn test_is_prime() {
    assert_eq!(true, is_prime(17));
    assert_eq!(false, is_prime(49));
}

#[stability::unstable(feature = "enable")]
pub const F17: Fp<17> = Fp::<17>::RING;

#[test]
fn test_finite_field_axioms() {
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&F17);
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::<128>::RING);
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Fp::<257>::RING);
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::<256>::RING);
}

#[test]
fn test_zn_el_add() {
    let a = F17.int_hom().map(6);
    let b = F17.int_hom().map(12);
    assert_eq!(F17.int_hom().map(1), F17.add(a, b));
}

#[test]
fn test_zn_el_sub() {
    let a = F17.int_hom().map(6);
    let b = F17.int_hom().map(12);
    assert_eq!(F17.int_hom().map(11), F17.sub(a, b));
}

#[test]
fn test_zn_el_mul() {
    let a = F17.int_hom().map(6);
    let b = F17.int_hom().map(12);
    assert_eq!(F17.int_hom().map(4), F17.mul(a, b));
}

#[test]
fn test_zn_el_div() {
    let a = F17.int_hom().map(6);
    let b = F17.int_hom().map(12);
    assert_eq!(F17.int_hom().map(9), F17.checked_div(&a, &b).unwrap());
}

#[test]
fn fn_test_div_impossible() {
    let _a = Zn::<22>::RING.int_hom().map(4);
    // the following line should give a compiler error
    // Zn::<22>::RING.div(_a, _a);
}

#[test]
fn test_zn_ring_axioms_znbase() {
    super::generic_tests::test_zn_axioms(Zn::<17>::RING);
    super::generic_tests::test_zn_axioms(Zn::<63>::RING);
}

#[test]
fn test_divisibility_axioms() {
    crate::divisibility::generic_tests::test_divisibility_axioms(Zn::<17>::RING, Zn::<17>::RING.elements());
    crate::divisibility::generic_tests::test_divisibility_axioms(Zn::<9>::RING, Zn::<9>::RING.elements());
    crate::divisibility::generic_tests::test_divisibility_axioms(Zn::<12>::RING, Zn::<12>::RING.elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let R = Zn::<17>::RING;
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
    let R = Zn::<63>::RING;
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
}