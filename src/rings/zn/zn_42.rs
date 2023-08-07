use crate::delegate::DelegateRing;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::rings::zn::*;
use crate::primitive_int::*;

use super::zn_barett;

fn usigned_as_signed_ref<'a>(x: &'a u64) -> &'a i64 {
    assert!(*x <= i64::MAX as u64);
    assert!(std::mem::align_of::<i64>() <= std::mem::align_of::<u64>());
    unsafe { std::mem::transmute(x) }
}

///
/// Represents the ring `Z/nZ`.
/// A special implementation of non-standard Barett reduction
/// that uses 128-bit integer but provides moduli up to 41 bits.
/// 
/// Any modular reductions are performed lazily on-demand.
/// 
/// # Example
/// 
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// let zn = Zn::new(7);
/// assert_el_eq!(&zn, &zn.one(), &zn.mul(zn.from_int(3), zn.from_int(5)));
/// ```
/// For moduli larger than 42 bit, this will panic
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// let zn = Zn::new((1 << 42) + 1);
/// ```
/// 
#[derive(Clone, Copy, PartialEq)]
pub struct ZnBase {
    /// must be 128 bit to deal with very small moduli
    inv_modulus: u128,
    modulus: u64,
    /// Representatives of elements may grow up to this bound
    repr_bound: u64
}

pub type Zn = RingValue<ZnBase>;

///
/// The number of bits to which we approximate the quotient `1 / modulus`.
/// In particular, we find `floor(2^b / modulus)` and then approximate
/// `x / modulus` by `(floor(2^b / modulus) * x) / 2^b`.
/// 
const BITSHIFT: u32 = 84;
///
/// Subtract one bit, as we need this to efficiently implement negate - 
/// see also constructor assertion `2 * modulus < repr_bound`.
/// 
pub const MAX_MODULUS_BITS: u32 = (BITSHIFT / 2) - 1;

#[derive(Copy, Clone, Debug)]
pub struct ZnEl(u64);

impl Zn {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus < (1 << MAX_MODULUS_BITS));
        let inv_modulus = (1 << BITSHIFT) / modulus as u128;
        let repr_bound = 1 << (inv_modulus.leading_zeros() / 2);
        // necessary for from_int to work
        assert!(repr_bound >= (1 << 16));
        // necessary for negate to work
        assert!(2 * modulus < repr_bound);
        return ZnBase {
            modulus: modulus,
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
    /// Note that we also need `input < repr_bound * repr_bound`, otherwise
    /// the 128-bit multiplication will overflow.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        assert!(value < (1 << BITSHIFT));
        let quotient = ((value * self.inv_modulus) >> BITSHIFT) as u64;
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
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        debug_assert!(*lhs < self.repr_bound);
        debug_assert!(rhs < self.repr_bound);
        *lhs += rhs;
        self.potential_reduce(lhs);
    }
    
    fn negate_inplace(&self, ZnEl(lhs): &mut Self::Element) {
        debug_assert!(*lhs < self.repr_bound);
        // we check this in the constructor also during release
        debug_assert!(2 * self.modulus < self.repr_bound);
        *lhs = 2 * self.modulus - self.bounded_reduce(*lhs as u128);
    }

    fn mul_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        debug_assert!(*lhs < self.repr_bound);
        debug_assert!(rhs < self.repr_bound);
        *lhs = self.bounded_reduce(*lhs as u128 * rhs as u128);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        // we check this in the constructor also during release
        debug_assert!(self.repr_bound >= (1 << 16));
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

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
            where R::Type: IntegerRing,
                Self: SelfIso
    {
        let fastmul_ring = ZnFastmul::from(ZnFastmulBase::new(*self));
        algorithms::sqr_mul::generic_pow(RingRef::new(self).cast(&fastmul_ring, x), power, &fastmul_ring, &RingRef::new(self), &integers)
    }

    fn sum<I>(&self, mut els: I) -> Self::Element 
        where I: Iterator<Item = Self::Element>
    {
        let mut result = self.zero();
        while let Some(ZnEl(start)) = els.next() {
            let mut current = start as u128;
            for ZnEl(c) in els.by_ref().take(self.repr_bound as usize - 1) {
                current += c as u128;
            }
            self.add_assign(&mut result, ZnEl(self.bounded_reduce(current)));
        }
        return result;
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

    type Isomorphism = <zn_barett::ZnBase<I> as CanonicalHom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Isomorphism> {
        if self.modulus as i128 == *from.modulus() {
            from.has_canonical_hom(self.integer_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_barett::ZnBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_barett::ZnBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0 as i64, iso)
    }
}

impl<I: ?Sized + IntegerRing> CanonicalHom<I> for ZnBase 
    where I: CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i64>>
{
    type Homomorphism = generic_impls::GenericIntegerToZnHom<I, StaticRingBase<i128>, Self>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        generic_impls::generic_has_canonical_hom_from_int(from, self, StaticRing::<i128>::RING.get_ring(), Some(&(self.repr_bound as i128 * self.repr_bound as i128)))
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        generic_impls::generic_map_in_from_int(from, self, StaticRing::<i128>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as u64) < self.modulus);
            ZnEl(n as u64)
        }, |n| {
            debug_assert!(n < (self.repr_bound as i128 * self.repr_bound as i128));
            ZnEl(self.bounded_reduce(n as u128))
        })
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ring = zn_barett::Zn::new(StaticRing::<i128>::RING, self.modulus as i128);
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

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> El<I>
        where I::Type: IntegerRing
    {
        int_cast(*self.modulus(), ZZ, self.integer_ring())
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

///
/// Wraps [`ZnBase`] to represent an instance of the ring `Z/nZ`.
/// As opposed to [`ZnBase`], elements are stored with additional information
/// to speed up multiplication `ZnBase x ZnFastmulBase -> ZnBase`, by
/// using [`CanonicalHom::mul_assign_map_in()`].
/// Note that normal arithmetic in this ring is slower than [`ZnBase`].
/// 
/// # Example
/// The following use of the FFT is usually faster than the standard use, as
/// the FFT requires a high amount of multiplications with the internally stored
/// roots of unity.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// # use feanor_math::mempool::*;
/// let ring = Zn::new(1073872897);
/// let fastmul_ring = ZnFastmul::new(ring);
/// // The values stored by the FFT table are elements of `ZnFastmulBase`
/// let fft = FFTTableCooleyTuckey::for_zn(&fastmul_ring, 15).unwrap();
/// // Note that data uses `ZnBase`
/// let mut data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
/// fft.unordered_fft(&mut data[..], &ring, &AllocatingMemoryProvider);
/// ```
/// 
#[derive(PartialEq)]
pub struct ZnFastmulBase {
    base: ZnBase
}

pub type ZnFastmul = RingValue<ZnFastmulBase>;

impl ZnFastmul {

    pub fn new(base: Zn) -> Self {
        RingValue::from(ZnFastmulBase::new(*base.get_ring()))
    }
}

impl ZnFastmulBase {

    pub fn new(base: ZnBase) -> Self {
        ZnFastmulBase { base }
    }
}

pub struct ZnFastmulEl(ZnEl, u128);

impl DelegateRing for ZnFastmulBase {

    type Base = ZnBase;
    type Element = ZnFastmulEl;

    fn get_delegate(&self) -> &Self::Base {
        &self.base
    }
    
    fn delegate(&self, ZnFastmulEl(el, _): Self::Element) -> <Self::Base as RingBase>::Element {
        el
    }

    fn delegate_mut<'a>(&self, ZnFastmulEl(el, _): &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        el
    }

    fn delegate_ref<'a>(&self, ZnFastmulEl(el, _): &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        el
    }

    fn postprocess_delegate_mut(&self, ZnFastmulEl(el, additional): &mut Self::Element) {
        *additional = el.0 as u128 * self.base.inv_modulus;
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        let mut result = ZnFastmulEl(el, 0);
        self.postprocess_delegate_mut(&mut result);
        return result;
    }
}


impl CanonicalHom<ZnFastmulBase> for ZnFastmulBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnFastmulBase) -> Option<Self::Homomorphism> {
        if self.base.modulus == from.base.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _: &ZnFastmulBase, el: Self::Element, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<ZnFastmulBase> for ZnFastmulBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnFastmulBase) -> Option<Self::Isomorphism> {
        if self.base.modulus == from.base.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, _: &ZnFastmulBase, el: Self::Element, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalHom<ZnFastmulBase> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnFastmulBase) -> Option<Self::Homomorphism> {
        if self.modulus == from.base.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &ZnFastmulBase, el: <ZnFastmulBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        from.delegate(el)
    }

    fn mul_assign_map_in_ref(&self, _: &ZnFastmulBase, ZnEl(lhs): &mut Self::Element, ZnFastmulEl(ZnEl(rhs), rhs_inv_mod): &<ZnFastmulBase as RingBase>::Element, _: &Self::Homomorphism) {
        let quotient = ((*lhs as u128 * *rhs_inv_mod) >> BITSHIFT) as u64;
        let result = (*lhs as u128 * *rhs as u128 - quotient as u128 * self.modulus as u128) as u64;
        *lhs = result;
        debug_assert!(*lhs < 2 * self.modulus);
    }

    fn mul_assign_map_in(&self, from: &ZnFastmulBase, lhs: &mut Self::Element, rhs: <ZnFastmulBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_map_in_ref(from, lhs, &rhs, hom);
    }
}

impl CanonicalHom<StaticRingBase<i64>> for ZnFastmulBase {

    type Homomorphism = <ZnBase as CanonicalHom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &StaticRingBase<i64>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &StaticRingBase<i64>, el: i64, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.map_in(from, el, hom))
    }
}

impl CanonicalIso<ZnFastmulBase> for ZnBase {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnFastmulBase) -> Option<Self::Isomorphism> {
        if self.modulus == from.base.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, _: &ZnFastmulBase, el: Self::Element, _: &Self::Isomorphism) -> <ZnFastmulBase as RingBase>::Element {
        ZnFastmulEl(el, el.0 as u128 * self.inv_modulus)
    }
}

impl<R: ZnRingStore<Type = ZnBase>> CanonicalHom<ZnBase> for AsFieldBase<R> {
    
    type Homomorphism = <ZnBase as CanonicalHom<ZnBase>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        <ZnBase as CanonicalHom<ZnBase>>::has_canonical_hom(self.base_ring().get_ring(), from)
    }

    fn map_in(&self, from: &ZnBase, el: <ZnBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from(<ZnBase as CanonicalHom<ZnBase>>::map_in(self.base_ring().get_ring(), from, el, hom))
    }
}

impl<R: ZnRingStore<Type = ZnBase>> CanonicalIso<ZnBase> for AsFieldBase<R> {

    type Isomorphism = <ZnBase as CanonicalIso<ZnBase>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ZnBase) -> Option<Self::Isomorphism> {
        <ZnBase as CanonicalIso<ZnBase>>::has_canonical_iso(self.base_ring().get_ring(), from)
    }

    fn map_out(&self, from: &ZnBase, el: <AsFieldBase<R> as RingBase>::Element, iso: &Self::Isomorphism) -> <ZnBase as RingBase>::Element {
        <ZnBase as CanonicalIso<ZnBase>>::map_out(self.base_ring().get_ring(), from, self.unwrap_element(el), iso)
    }
}

impl<R: ZnRingStore<Type = ZnBase>> CanonicalHom<AsFieldBase<R>> for ZnBase {
    
    type Homomorphism = <ZnBase as CanonicalHom<ZnBase>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFieldBase<R>) -> Option<Self::Homomorphism> {
        self.has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &AsFieldBase<R>, el: <AsFieldBase<R> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(from.base_ring().get_ring(), from.unwrap_element(el), hom)
    }
}

impl<R: ZnRingStore<Type = ZnBase>> CanonicalIso<AsFieldBase<R>> for ZnBase {

    type Isomorphism = <ZnBase as CanonicalIso<ZnBase>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<R>) -> Option<Self::Isomorphism> {
        self.has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &AsFieldBase<R>, el: <ZnBase as RingBase>::Element, iso: &Self::Isomorphism) -> <AsFieldBase<R> as RingBase>::Element {
        from.from(self.map_out(from.base_ring().get_ring(), el, iso))
    }
}

#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 10] = [0, 1, 3, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms() {
    let ring = Zn::new(63);
    generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)));

    let ring = Zn::new((1 << 41) - 1);
    generic_test_ring_axioms(&ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1].iter().cloned().map(|x| ring.from_int(x)));
}

#[test]
fn test_sum() {
    let ring = Zn::new(17);
    assert_el_eq!(&ring, &ring.from_int(10001 * 5000), &ring.sum((0..=10000).map(|x| ring.from_int(x))));

    let ring = Zn::new((1 << 41) - 1);
    assert_el_eq!(&ring, &ring.from_int(10001 * 5000), &ring.sum((0..=10000).map(|x| ring.from_int(x))));
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    let from = zn_barett::Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::new(7 * 11);
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    generic_test_canonical_iso_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i128>::RING;
    let to = Zn::new(7 * 11);
    generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
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

    let ZZbig = DefaultBigIntRing::RING;
    let R = Zn::new(3);
    assert_el_eq!(&R, &R.from_int(0), &R.coerce(&ZZbig, ZZbig.sub(ZZbig.power_of_two(84), ZZbig.one())));
}

#[test]
fn test_zn_map_in_small_int() {
    let R = Zn::new((1 << 41) - 1);
    assert_el_eq!(&R, &R.from_int(1), &R.coerce(&StaticRing::<i8>::RING, 1));
}

#[test]
fn test_from_int() {
    let R = Zn::new(2);
    assert_el_eq!(&R, &R.from_int(1), &R.from_int(i32::MAX));

    let R = Zn::new((1 << 41) - 1);
    assert_el_eq!(&R, &R.pow(R.from_int(2), 30), &R.from_int(1 << 30));
}

#[test]
fn test_canonical_iso_axioms_as_field() {
    let R = Zn::new(17);
    let R2 = R.clone().as_field().ok().unwrap();
    generic_test_canonical_hom_axioms(&R, &R2, R.elements());
    generic_test_canonical_iso_axioms(&R, &R2, R.elements());
    generic_test_canonical_hom_axioms(&R2, &R, R2.elements());
    generic_test_canonical_iso_axioms(&R2, &R, R2.elements());
}