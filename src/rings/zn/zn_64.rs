use crate::algorithms::fft::cooley_tuckey::*;
use crate::divisibility::DivisibilityRingStore;
use crate::euclidean::EuclideanRingStore;
use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::rings::zn::*;
use crate::primitive_int::*;
use crate::rings::rust_bigint::RustBigintRingBase;

use super::zn_barett;

pub fn usigned_as_signed_ref<'a>(x: &'a u64) -> &'a i64 {
    assert!(*x <= i64::MAX as u64);
    assert!(std::mem::align_of::<i64>() <= std::mem::align_of::<u64>());
    unsafe { std::mem::transmute(x) }
}

#[allow(non_upper_case_globals)]
const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

fn split(x: u128) -> (u64, u64) {
    ((x & ((1 << 64) - 1)) as u64, (x >> 64) as u64)
}

///
/// Represents the ring `Z/nZ`.
/// A special implementation of non-standard Barett reduction
/// is used that allows moduli up to 63 bits.
/// 
/// # Implementation
/// 
/// For standard Barett reduction, we would compute for a multiplication
/// the expression `floor(a * b * floor(2^k / n) / 2^k)`. In our case, we
/// set `k = 64`. 
/// 
/// Note that `a * b` fits into `u128`, hence we can write it
/// as `a * b = c0 + c1 * 2^64`. Similarly, we write `floor(2^64 / n) = k0 + k1 * 2^64`.
/// Instead of the complete product `(c0 + c1 * 2^64)(k0 + k1 * 2^64)` we now only
/// compute `c0 * k1 * 2^64 + c1 * k0 * 2^64 + c1 * k1 * 2^128`, i.e. ignore
/// the lowest order term. This saves one multiplication, and does not increase
/// the error a lot.
/// 
#[derive(Clone, Copy, PartialEq)]
pub struct ZnBase {
    inv_modulus: (u64, u64),
    modulus: u64,
    /// Representatives of elements may grow up to (including) this bound
    repr_bound: u64
}

pub type Zn = RingValue<ZnBase>;

#[derive(Copy, Clone, Debug)]
pub struct ZnEl(u64);

impl Zn {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus > 1);
        // required to make the product `a * b * floor(2^128 / n)` fit into 256 bits
        // if `a, b <= 2 * n`. Also, we would like to compute `a + b` as `potential_reduce(a + b)`.
        assert!(modulus < (1 << 62));
        let inv_modulus = if modulus == 2 {
            1 << 127
        } else {
            ZZbig.cast(
                &StaticRing::<i128>::RING,
                ZZbig.euclidean_div(ZZbig.power_of_two(128), &ZZbig.coerce(&ZZ, modulus as i64))
            ) as u128
        };
        let repr_bound = 2 * modulus;
        let inv_modulus_lower_bits = inv_modulus & ((1 << 64) - 1);
        // the error comes from two points:
        //  - approximating `2^128 / n` via `floor(2^128 / n)`
        //  - ignoring the lowest bit term `c0 * k0` in `(c0 + c1 2^64)(k0 + k1 2^64)`
        //    where `c0 + c1^64 = a * b` and `k0 + k1 2^64 = floor(2^128 / n)`
        let error = ZZbig.add(
            ZZbig.pow(ZZbig.coerce(&StaticRing::<i128>::RING, repr_bound as i128), 2),
            ZZbig.mul(ZZbig.coerce(&StaticRing::<i128>::RING, inv_modulus_lower_bits as i128), ZZbig.power_of_two(64))
        );
        // if the error is less than one, our reduction reduces into `[0, 2n)`
        assert!(ZZbig.is_lt(&error, &ZZbig.power_of_two(128)));
        return ZnBase {
            modulus: modulus,
            inv_modulus: split(inv_modulus),
            repr_bound: repr_bound
        }
    }

    fn bounded_reduce(&self, value: u128) -> u64 {
        debug_assert!(value <= self.repr_bound as u128 * self.repr_bound as u128);
        let (c0, c1) = split(value);
        let (k0, k1) = self.inv_modulus;
        // note that `c1 * k1` cannot overflow, otherwise the whole product
        // `a * b * floor(2^128 / n)` would be `>= 2^192` which cannot be as `n < 2^62`.
        let numerator = c0 as u128 * k1 as u128 + c1 as u128 * k0 as u128 + (c1 * k1) as u128 * (1 << 64);
        let quotient = (numerator >> 64) as u64;
        let result = (value - quotient as u128 * self.modulus as u128) as u64;
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
    
    fn potential_reduce(&self, val: &mut u64) {
        if std::intrinsics::unlikely(*val >= self.repr_bound) {
            *val -= self.repr_bound;
        }
    }
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        debug_assert!(*lhs <= self.repr_bound);
        debug_assert!(rhs <= self.repr_bound);
        *lhs += rhs;
        self.potential_reduce(lhs);
        debug_assert!(*lhs <= self.repr_bound);
    }
    
    fn negate_inplace(&self, ZnEl(lhs): &mut Self::Element) {
        debug_assert!(*lhs <= self.repr_bound);
        *lhs = self.repr_bound - *lhs;
        debug_assert!(*lhs <= self.repr_bound);
    }

    fn mul_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        debug_assert!(*lhs <= self.repr_bound);
        debug_assert!(rhs <= self.repr_bound);
        *lhs = self.bounded_reduce(*lhs as u128 * rhs as u128);
        debug_assert!(*lhs <= self.repr_bound);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
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
        debug_assert!(*value <= self.repr_bound);
        self.complete_reduce(*value as u128) == 0
    }
    
    fn is_neg_one(&self, ZnEl(value): &Self::Element) -> bool {
        debug_assert!(*value <= self.repr_bound);
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

impl<I: IntegerRingStore> CanonicalHom<zn_barett::ZnBase<I>> for ZnBase
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i64>>
{
    type Homomorphism = <I::Type as CanonicalIso<StaticRingBase<i64>>>::Isomorphism;

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        if self.modulus as i128 == from.integer_ring().cast(&ZZ, from.integer_ring().clone_el(from.modulus())) as i128 {
            from.integer_ring().get_ring().has_canonical_iso(ZZ.get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_barett::ZnBase<I>, el: <zn_barett::ZnBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(from.integer_ring().get_ring().map_out(ZZ.get_ring(), from.smallest_positive_lift(el), hom) as u64)
    }
}

impl<I: IntegerRingStore> CanonicalIso<zn_barett::ZnBase<I>> for ZnBase
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i64>>
{
    type Isomorphism = <zn_barett::ZnBase<I> as CanonicalHom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Isomorphism> {
        if self.modulus as i128 == from.integer_ring().cast(&ZZ, from.integer_ring().clone_el(from.modulus())) as i128 {
            from.has_canonical_hom(self.integer_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_barett::ZnBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_barett::ZnBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0 as i64, iso)
    }
}

trait GenericMapInFromInt: IntegerRing + CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i64>> {}

impl GenericMapInFromInt for StaticRingBase<i8> {}
impl GenericMapInFromInt for StaticRingBase<i16> {}
impl GenericMapInFromInt for StaticRingBase<i32> {}
impl GenericMapInFromInt for StaticRingBase<i64> {}
impl GenericMapInFromInt for StaticRingBase<i128> {}
impl GenericMapInFromInt for RustBigintRingBase {}

#[cfg(feature = "mpir")]
impl GenericMapInFromInt for crate::rings::mpir::MPZBase {}

impl<I: ?Sized + GenericMapInFromInt> CanonicalHom<I> for ZnBase {

    type Homomorphism = generic_impls::GenericIntegerToZnHom<I, StaticRingBase<i128>, Self>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        generic_impls::generic_has_canonical_hom_from_int(from, self, StaticRing::<i128>::RING.get_ring(), Some(&(self.repr_bound as i128 * self.repr_bound as i128)))
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        generic_impls::generic_map_in_from_int(from, self, StaticRing::<i128>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as u64) < self.modulus);
            ZnEl(n as u64)
        }, |n| {
            debug_assert!(n <= (self.repr_bound as i128 * self.repr_bound as i128));
            ZnEl(self.bounded_reduce(n as u128))
        })
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ring = zn_barett::Zn::new(ZZbig, ZZbig.coerce(&ZZ, self.modulus as i64));
        Some(RingRef::new(self).coerce(&ring, ring.checked_div(&RingRef::new(self).cast(&ring, *lhs), &RingRef::new(self).cast(&ring, *rhs))?))
    }
}

pub struct ZnBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

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

impl FiniteRing for ZnBase {

    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        generic_impls::generic_random_element(self, rng)
    }

    fn size<I: IntegerRingStore>(&self, int_ring: &I) -> El<I>
        where I::Type: IntegerRing
    {
        int_cast(*self.modulus(), int_ring, self.integer_ring())
    }
}

impl ZnRing for ZnBase {

    type IntegerRingBase = StaticRingBase<i64>;
    type Integers = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.complete_reduce(el.0 as u128) as i64
    }

    fn modulus(&self) -> &El<Self::Integers> {
        usigned_as_signed_ref(&self.modulus)
    }
}

impl HashableElRing for ZnBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.integer_ring().hash(&self.smallest_positive_lift(*el), h)
    }
}

impl CooleyTuckeyButterfly<ZnBase> for ZnBase {

    #[inline(always)]
    fn butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, _: &ZnBase, _: &<Self as CanonicalHom<ZnBase>>::Homomorphism, values: &mut V, twiddle: &<ZnBase as RingBase>::Element, i1: usize, i2: usize) {
        let a = *values.at(i1);
        let mut b = *values.at(i2);
        self.mul_assign_ref(&mut b, twiddle);

        // this is implied by `bounded_reduce`, check anyway
        debug_assert!(b.0 < self.modulus * 2);
        debug_assert!(self.repr_bound >= self.modulus * 2);

        *values.at_mut(i1) = self.add(a, b);
        *values.at_mut(i2) = self.add(a, ZnEl(2 * self.modulus - b.0));
    }

    #[inline(always)]
    fn inv_butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, _: &ZnBase, _: &<Self as CanonicalHom<ZnBase>>::Homomorphism, values: &mut V, twiddle: &<ZnBase as RingBase>::Element, i1: usize, i2: usize) {
        let a = *values.at(i1);
        let b = *values.at(i2);

        let b_reduced = ZnEl(self.bounded_reduce(b.0 as u128));

        *values.at_mut(i1) = self.add(a, b_reduced);
        *values.at_mut(i2) = self.add(a, ZnEl(2 * self.modulus - b_reduced.0));
        self.mul_assign_ref(values.at_mut(i2), twiddle);
    }
}

#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_ring_axioms() {
    let ring = Zn::new(2);
    generic_test_ring_axioms(&ring, ring.elements());

    let ring = Zn::new(63);
    generic_test_ring_axioms(&ring, ring.elements());

    let ring = Zn::new((1 << 41) - 1);
    generic_test_ring_axioms(&ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1].iter().cloned().map(|x| ring.from_int(x)));

    let ring = Zn::new((1 << 60) - 1);
    generic_test_ring_axioms(&ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1, (1 << 30), (1 << 30) + 1].iter().cloned().map(|x| ring.from_int(x)));
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    let from = zn_barett::Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::new(7 * 11);
    generic_test_canonical_hom_axioms(&from, &to, from.elements());
    generic_test_canonical_iso_axioms(&from, &to, from.elements());
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i128>::RING;
    let to = Zn::new(7 * 11);
    generic_test_canonical_hom_axioms(&from, to, 0..(7 * 11));
}

#[test]
fn test_zn_ring_axioms() {
    generic_test_zn_ring_axioms(Zn::new(17));
    generic_test_zn_ring_axioms(Zn::new(63));
}

#[test]
fn test_divisibility_axioms() {
    let R = Zn::new(17);
    generic_test_divisibility_axioms(&R, R.elements());
}

#[test]
fn test_zn_map_in_large_int() {
    let R = Zn::new(17);
    generic_test_map_in_large_int(R);

    let R = Zn::new(3);
    assert_el_eq!(&R, &R.from_int(0), &R.coerce(&ZZbig, ZZbig.sub(ZZbig.power_of_two(84), ZZbig.one())));
}

#[test]
fn test_zn_map_in_small_int() {
    let R = Zn::new((1 << 41) - 1);
    let hom = generic_impls::generic_has_canonical_hom_from_int(StaticRing::<i8>::RING.get_ring(), R.get_ring(), StaticRing::<i128>::RING.get_ring(), Some(&(*R.modulus() as i128 * *R.modulus() as i128))).unwrap();
    assert_el_eq!(&R, &R.from_int(1), &generic_impls::generic_map_in_from_int(
        StaticRing::<i8>::RING.get_ring(), 
        R.get_ring(), 
        StaticRing::<i128>::RING.get_ring(), 
        1, 
        &hom, 
        |n| ZnEl(n as u64), 
        |n| ZnEl(R.get_ring().bounded_reduce(n as u128))
    ));
}

#[test]
fn test_from_int() {
    let R = Zn::new(2);
    assert_el_eq!(&R, &R.from_int(1), &R.from_int(i32::MAX));

    let R = Zn::new((1 << 41) - 1);
    assert_el_eq!(&R, &R.pow(R.from_int(2), 30), &R.from_int(1 << 30));
}

#[test]
fn test_cooley_tuckey_butterfly() {
    let ring = Zn::new(2);
    generic_test_cooley_tuckey_butterfly(ring, ring, ring.elements(), &ring.one());

    let ring = Zn::new(97);
    generic_test_cooley_tuckey_butterfly(ring, ring, ring.elements(), &ring.from_int(3));

    let ring = Zn::new((1 << 41) - 1);
    generic_test_cooley_tuckey_butterfly(ring, ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1].iter().cloned().map(|x| ring.from_int(x)), &ring.from_int(3));
    
    let ring = Zn::new((1 << 60) - 1);
    generic_test_cooley_tuckey_butterfly(ring, ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1, (1 << 30), (1 << 30) + 1].iter().cloned().map(|x| ring.from_int(x)), &ring.from_int(19));
}