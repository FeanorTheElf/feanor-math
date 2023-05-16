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
/// # Attention
/// 
/// Reductions are only performed during multiplications.
/// In particular, this means that performing more than
/// `(1 << BITSHIFT) / modulus` non-reducing operations (i.e.
/// additions, subtractions, negations, ...) will result
/// in an integer overflow.
///
#[derive(Clone, Copy)]
pub struct ZnLazyBase {
    // must be 128 bit to deal with very small moduli
    inv_modulus: u128,
    modulus: u64,
    modulus_i128: i128
}

pub type ZnLazy = RingValue<ZnLazyBase>;

// this is the value up to which we can choose the modulus
const MAX_MODULUS_LOG2: u32 = 36;
// this is the value up to which representatives may grow during operations
const MAX_SIZE_LOG2: u32 = 42;
const BITSHIFT: u32 = 2 * MAX_SIZE_LOG2;

#[derive(Copy, Clone)]
pub struct ZnLazyEl(u64);

impl ZnLazy {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnLazyBase::new(modulus))
    }
}

impl ZnLazyBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus < (1 << MAX_MODULUS_LOG2));
        return ZnLazyBase {
            modulus: modulus,
            modulus_i128: modulus as i128,
            inv_modulus: (1 << BITSHIFT) / modulus as u128
        }
    }

    ///
    /// If input is smaller than `1 << BITSHIFT`, the output is smaller
    /// than `2 * self.modulus` and congruent to the input.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        assert!(value <= (1 << BITSHIFT));
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

impl RingBase for ZnLazyBase {

    type Element = ZnLazyEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnLazyEl(lhs): &mut Self::Element, ZnLazyEl(rhs): Self::Element) {
        *lhs += rhs;
    }
    
    fn negate_inplace(&self, ZnLazyEl(lhs): &mut Self::Element) {
        *lhs = 2 * self.modulus - self.bounded_reduce(*lhs as u128);
    }

    fn mul_assign(&self, ZnLazyEl(lhs): &mut Self::Element, ZnLazyEl(rhs): Self::Element) {
        *lhs = self.bounded_reduce(*lhs as u128 * rhs as u128)
    }

    fn from_int(&self, value: i32) -> Self::Element {
        if value < 0 {
            return self.negate(ZnLazyEl(self.bounded_reduce(-value as u128)));
        } else {
            return ZnLazyEl(self.bounded_reduce(value as u128));
        }
    }

    fn eq_el(&self, ZnLazyEl(lhs): &Self::Element, ZnLazyEl(rhs): &Self::Element) -> bool {
        if *lhs >= *rhs {
            self.is_zero(&ZnLazyEl(*lhs - *rhs))
        } else {
            self.is_zero(&ZnLazyEl(*rhs - *lhs))
        }
    }

    fn is_one(&self, ZnLazyEl(value): &Self::Element) -> bool {
        *value != 0 && self.is_zero(&ZnLazyEl(*value - 1))
    }

    fn is_zero(&self, ZnLazyEl(value): &Self::Element) -> bool {
        self.complete_reduce(*value as u128) == 0
    }
    
    fn is_neg_one(&self, ZnLazyEl(value): &Self::Element) -> bool {
        self.is_zero(&ZnLazyEl(*value + 1))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn dbg<'a>(&self, ZnLazyEl(value): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", self.complete_reduce(*value as u128))
    }
}

impl CanonicalHom<ZnLazyBase> for ZnLazyBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnLazyBase) -> Option<Self::Homomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _: &ZnLazyBase, el: <ZnLazyBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<ZnLazyBase> for ZnLazyBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnLazyBase) -> Option<Self::Homomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, _: &ZnLazyBase, el: Self::Element, _: &Self::Isomorphism) -> <ZnLazyBase as RingBase>::Element {
        el
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalHom<zn_barett::ZnBase<I>> for ZnLazyBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        if self.modulus as i128 == *from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_barett::ZnBase<I>, el: <zn_barett::ZnBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        ZnLazyEl(from.smallest_positive_lift(el) as u64)
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalIso<zn_barett::ZnBase<I>> for ZnLazyBase {

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

impl CanonicalHom<StaticRingBase<i128>> for ZnLazyBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &StaticRingBase<i128>) -> Option<Self::Homomorphism> {
       Some(())
    }

    fn map_in(&self, _: &StaticRingBase<i128>, el: i128, _: &Self::Homomorphism) -> Self::Element {
        let (neg, n) = if el < 0 {
            (true, -el as u128)
        } else {
            (false, el as u128)
        };
        let reduced = if n < (1 << BITSHIFT) {
            self.bounded_reduce(n)
        } else {
            (n % self.modulus as u128) as u64
        };
        if neg {
            self.negate(ZnLazyEl(reduced))
        } else {
            ZnLazyEl(reduced)
        }
    }
}

impl DivisibilityRing for ZnLazyBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ring = self.non_lazy();
        Some(RingRef::new(self).coerce(&ring, ring.checked_div(&RingRef::new(self).cast(&ring, *lhs), &RingRef::new(self).cast(&ring, *rhs))?))
    }
}

pub struct ZnLazyBaseElementsIter<'a> {
    ring: &'a ZnLazyBase,
    current: u64
}

impl<'a> Iterator for ZnLazyBaseElementsIter<'a> {

    type Item = ZnLazyEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus {
            let result = self.current;
            self.current += 1;
            return Some(ZnLazyEl(result));
        } else {
            return None;
        }
    }
}

impl ZnRing for ZnLazyBase {

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

#[derive(Clone, Copy)]
pub struct ZnBase {
    base: ZnLazyBase
}

#[derive(Clone, Copy)]
pub struct ZnEl(ZnLazyEl);

pub type Zn = RingValue<ZnBase>;

impl Zn {
    
    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

impl ZnBase {
    
    pub fn new(modulus: u64) -> Self {
        ZnBase { base: ZnLazyBase::new(modulus) }
    }

    fn reduce_once(&self, ZnLazyEl(val): &mut ZnLazyEl) {
        if *val >= self.base.modulus {
            *val -= self.base.modulus;
        }
    }

    fn assert_valid(&self, ZnLazyEl(val): &ZnLazyEl) {
        debug_assert!(*val <= self.base.modulus);
    }
}

impl RingBase for ZnBase {
    
    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        self.assert_valid(lhs);
        self.assert_valid(&rhs);
        lhs.0 += rhs.0;
        self.reduce_once(lhs);
        self.assert_valid(lhs);
    }
    
    fn negate_inplace(&self, ZnEl(lhs): &mut Self::Element) {
        self.assert_valid(lhs);
        lhs.0 = self.base.modulus - lhs.0;
        self.assert_valid(lhs);
    }

    fn mul_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        self.assert_valid(lhs);
        self.assert_valid(&rhs);
        self.base.mul_assign(lhs, rhs);
        self.reduce_once(lhs);
        self.assert_valid(lhs);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        let mut result = self.base.from_int(value);
        self.reduce_once(&mut result);
        self.assert_valid(&result);
        return ZnEl(result);
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.is_zero(&self.sub_ref(lhs, rhs))
    }

    fn is_zero(&self, ZnEl(val): &Self::Element) -> bool {
        self.assert_valid(val);
        val.0 == 0 || val.0 == self.base.modulus
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, ZnEl(val): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.base.dbg(val, out)
    }

}

impl CanonicalHom<ZnBase> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(&from.base)
    }

    fn map_in(&self, from: &ZnBase, ZnEl(el): <ZnBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(self.base.map_in(&from.base, el, hom))
    }
}

impl CanonicalIso<ZnBase> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        self.base.has_canonical_iso(&from.base)
    }

    fn map_out(&self, from: &ZnBase, ZnEl(el): Self::Element, iso: &Self::Isomorphism) -> <ZnBase as RingBase>::Element {
        ZnEl(self.base.map_out(&from.base, el, iso))
    }
}

impl CanonicalHom<ZnLazyBase> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnLazyBase) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, _: &ZnLazyBase, el: <ZnLazyBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        let mut result = ZnLazyEl(self.base.complete_reduce(el.0 as u128));
        self.reduce_once(&mut result);
        return ZnEl(result);
    }
}

impl CanonicalIso<ZnLazyBase> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnLazyBase) -> Option<Self::Homomorphism> {
        self.base.has_canonical_iso(from)
    }

    fn map_out(&self, _: &ZnLazyBase, ZnEl(el): Self::Element, _: &Self::Isomorphism) -> <ZnLazyBase as RingBase>::Element {
        el
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalHom<zn_barett::ZnBase<I>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &zn_barett::ZnBase<I>, el: <zn_barett::ZnBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(self.base.map_in(from, el, hom))
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalIso<zn_barett::ZnBase<I>> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_iso(from)
    }

    fn map_out(&self, from: &zn_barett::ZnBase<I>, ZnEl(el): <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_barett::ZnBase<I> as RingBase>::Element {
        self.base.map_out(from, el, iso)
    }
}

impl CanonicalHom<StaticRingBase<i128>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &StaticRingBase<i128>) -> Option<Self::Homomorphism> {
       self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &StaticRingBase<i128>, el: i128, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = self.base.map_in(from, el, hom);
        self.reduce_once(&mut result);
        self.assert_valid(&result);
        return ZnEl(result);
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, ZnEl(lhs): &Self::Element, ZnEl(rhs): &Self::Element) -> Option<Self::Element> {
        let mut result = self.base.checked_left_div(lhs, rhs)?;
        self.reduce_once(&mut result);
        self.assert_valid(&result);
        return Some(ZnEl(result));
    }
}


pub struct ZnBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

    type Item = ZnEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.base.modulus {
            let result = self.current;
            self.current += 1;
            return Some(ZnEl(ZnLazyEl(result)));
        } else {
            return None;
        }
    }
}

impl ZnRing for ZnBase {
    
    type IntegerRingBase = StaticRingBase<i128>;
    type Integers = StaticRing<i128>;
    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn integer_ring(&self) -> &Self::Integers {
        self.base.integer_ring()
    }

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn smallest_positive_lift(&self, ZnEl(el): Self::Element) -> El<Self::Integers> {
        self.assert_valid(&el);
        if el.0 == self.base.modulus {
            0
        } else {
            el.0 as i128
        }
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.base.modulus()
    }
}

#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 10] = [0, 1, 3, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms() {
    {
        let ring = ZnLazy::new(63);
        generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)));
    }
    {
        let ring = Zn::new(63);
        generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)));
    }
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    {
        let from = zn_barett::Zn::new(StaticRing::<i128>::RING, 7 * 11);
        let to = ZnLazy::new(7 * 11);
        generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
        generic_test_canonical_iso_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    }
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
        let to = ZnLazy::new(7 * 11);
        generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    }
    {
        let from = StaticRing::<i128>::RING;
        let to = Zn::new(7 * 11);
        generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    }
}

#[test]
fn test_zn_ring_axioms() {
    {
        generic_test_zn_ring_axioms(ZnLazy::new(17));
        generic_test_zn_ring_axioms(ZnLazy::new(63));
    }
    {
        generic_test_zn_ring_axioms(Zn::new(17));
        generic_test_zn_ring_axioms(Zn::new(63));
    }
}

#[test]
fn test_divisibility_axioms() {
    {
        let R = ZnLazy::new(17);
        generic_test_divisibility_axioms(&R, R.elements());
    }
    {
        let R = Zn::new(17);
        generic_test_divisibility_axioms(&R, R.elements());
    }
}