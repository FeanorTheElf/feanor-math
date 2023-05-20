use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::rings::zn::*;
use crate::primitive_int::*;

use super::zn_barett;

///
/// Represents the ring `Z/nZ`.
/// A special implementation of non-standard Barett reduction
/// that uses 128-bit integer but provides moduli up to 42 bit.
/// 
/// Any modular reductions are performed lazily.
/// 
#[derive(Clone, Copy)]
pub struct ZnBase {
    // must be 128 bit to deal with very small moduli
    inv_modulus: u128,
    modulus: u64,
    repr_bound: u64,
    modulus_i128: i128
}

pub type Zn = RingValue<ZnBase>;

///
/// The number of bits to which we approximate the quotient `1 / modulus`.
/// In particular, we find `floor(2^b / modulus)` and then approximate
/// `x / modulus` by `(floor(2^b / modulus) * x) / 2^b`.
/// 
const BITSHIFT: u32 = 84;
const MAX_MODULUS_BITS: u32 = BITSHIFT / 2;

#[derive(Copy, Clone, Debug)]
pub struct ZnEl(u64);

impl Zn {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus <= (1 << MAX_MODULUS_BITS));
        let inv_modulus = (1 << BITSHIFT) / modulus as u128;
        let repr_bound = 1 << (inv_modulus.leading_zeros() / 2);
        assert!(2 * modulus < repr_bound);
        return ZnBase {
            modulus: modulus,
            modulus_i128: modulus as i128,
            inv_modulus: inv_modulus,
            repr_bound: repr_bound
        }
    }

    fn potential_reduce(&self, val: &mut u64) {
        if std::intrinsics::unlikely(*val >= self.repr_bound) {
            *val = self.bounded_reduce(*val as u128);
        }
    }

    ///
    /// If input is smaller than `1 << BITSHIFT`, the output is smaller
    /// than `2 * self.modulus` and congruent to the input.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        assert!(value < (1 << BITSHIFT));
        let result = (value - ((value * self.inv_modulus) >> BITSHIFT) * self.modulus as u128) as u64;
        debug_assert!(result < 2 * self.modulus);
        return result;
    }

    fn complete_reduce(&self, value: u128) -> u64 {
        let mut result = self.bounded_reduce(value);
        if result >= self.modulus {
            result -= self.modulus;
        }
        debug_assert!(result < self.modulus);
        return result;
    }

    fn non_lazy(&self) -> zn_barett::Zn<StaticRing<i128>> {
        zn_barett::Zn::new(StaticRing::<i128>::RING, self.modulus as i128)
    }
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        *lhs += rhs;
        self.potential_reduce(lhs);
    }
    
    fn negate_inplace(&self, ZnEl(lhs): &mut Self::Element) {
        *lhs = 2 * self.modulus - self.bounded_reduce(*lhs as u128);
    }

    fn mul_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        *lhs = self.bounded_reduce(*lhs as u128 * rhs as u128);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        if value < 0 {
            return self.negate(ZnEl(self.bounded_reduce(-value as u128)));
        } else {
            return ZnEl(self.bounded_reduce(value as u128));
        }
    }

    fn eq_el(&self, ZnEl(lhs): &Self::Element, ZnEl(rhs): &Self::Element) -> bool {
        if *lhs >= *rhs {
            self.is_zero(&ZnEl(*lhs - *rhs))
        } else {
            self.is_zero(&ZnEl(*rhs - *lhs))
        }
    }

    fn is_one(&self, ZnEl(value): &Self::Element) -> bool {
        *value != 0 && self.is_zero(&ZnEl(*value - 1))
    }

    fn is_zero(&self, ZnEl(value): &Self::Element) -> bool {
        self.complete_reduce(*value as u128) == 0
    }
    
    fn is_neg_one(&self, ZnEl(value): &Self::Element) -> bool {
        self.is_zero(&ZnEl(*value + 1))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn dbg<'a>(&self, ZnEl(value): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", self.complete_reduce(*value as u128))
    }
}

impl CanonicalHom<ZnBase> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _: &ZnBase, el: <ZnBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<ZnBase> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, _: &ZnBase, el: Self::Element, _: &Self::Isomorphism) -> <ZnBase as RingBase>::Element {
        el
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalHom<zn_barett::ZnBase<I>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        if self.modulus as i128 == *from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_barett::ZnBase<I>, el: <zn_barett::ZnBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        ZnEl(from.smallest_positive_lift(el) as u64)
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalIso<zn_barett::ZnBase<I>> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        if self.modulus as i128 == *from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_barett::ZnBase<I>, el: <Self as RingBase>::Element, _: &Self::Homomorphism) -> <zn_barett::ZnBase<I> as RingBase>::Element {
        from.project_gen(el.0 as i64, &StaticRing::<i64>::RING)
    }
}

impl<I: IntegerRing + CanonicalIso<StaticRingBase<i128>>> CanonicalHom<I> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &I) -> Option<Self::Homomorphism> {
       Some(())
    }

    fn map_in(&self, from: &I, el: I::Element, _: &Self::Homomorphism) -> Self::Element {
        let (neg, n) = if from.is_neg(&el) {
            (true, from.negate(el))
        } else {
            (false, el)
        };
        let ZZ128 = StaticRing::<i128>::RING.get_ring();
        let as_u128 = |x: I::Element| from.map_out(ZZ128, x, &from.has_canonical_iso(ZZ128).unwrap()) as u128;
        let from_u128 = |x: u128| from.map_in(ZZ128, x as i128, &from.has_canonical_hom(ZZ128).unwrap());
        let reduced = if from.is_lt(&n, &from.power_of_two(BITSHIFT as usize)) {
            self.bounded_reduce(as_u128(n))
        } else {
            as_u128(from.euclidean_rem(n, &from_u128(self.modulus as u128))) as u64
        };
        if neg {
            self.negate(ZnEl(reduced))
        } else {
            ZnEl(reduced)
        }
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ring = self.non_lazy();
        Some(RingRef::new(self).coerce(&ring, ring.checked_div(&RingRef::new(self).cast(&ring, *lhs), &RingRef::new(self).cast(&ring, *rhs))?))
    }
}

pub struct ZnLazyBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u64
}

impl<'a> Iterator for ZnLazyBaseElementsIter<'a> {

    type Item = ZnEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus {
            let result = self.current;
            self.current += 1;
            return Some(ZnEl(result));
        } else {
            return None;
        }
    }
}

impl ZnRing for ZnBase {

    type IntegerRingBase = StaticRingBase<i128>;
    type Integers = StaticRing<i128>;
    type ElementsIter<'a> = ZnLazyBaseElementsIter<'a>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i128>::RING
    }

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnLazyBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.complete_reduce(el.0 as u128) as i128
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus_i128
    }
}

#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 10] = [0, 1, 3, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms() {
    {
        let ring = Zn::new(63);
        generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)));
    }
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    {
        let from = zn_barett::Zn::new(StaticRing::<i128>::RING, 7 * 11);
        let to = Zn::new(7 * 11);
        generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
        generic_test_canonical_iso_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    }
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    {
        let from = StaticRing::<i128>::RING;
        let to = Zn::new(7 * 11);
        generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    }
}

#[test]
fn test_zn_ring_axioms() {
    {
        generic_test_zn_ring_axioms(Zn::new(17));
        generic_test_zn_ring_axioms(Zn::new(63));
    }
}

#[test]
fn test_divisibility_axioms() {
    {
        let R = Zn::new(17);
        generic_test_divisibility_axioms(&R, R.elements());
    }
}

#[test]
fn test_zn_map_in_large_int() {
    {
        let R = Zn::new(17);
        generic_test_map_in_large_int(R);
    }
}