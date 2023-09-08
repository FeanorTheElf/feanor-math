use crate::algorithms::fft::cooley_tuckey::*;
use crate::delegate::DelegateRing;
use crate::divisibility::DivisibilityRingStore;
use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::rings::zn::*;
use crate::primitive_int::*;
use crate::rings::rust_bigint::RustBigintRingBase;

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
/// The basic idea underlying this implementation is the fact
/// that for Barett reduction, we have to multiply three numbers
/// of roughly equal size. If the result should fit into `u128`, each
/// number can be at most 42 bits. Hence, we restrict moduli to at
/// most 42 bits, but for efficiency reasons, we want to allow
/// representatives to grow up to `2 * n`, hence only 41 bits are left.
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
/// For moduli larger than 41 bit, this will panic
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// let zn = Zn::new((1 << 41) + 1);
/// ```
/// 
#[derive(Clone, Copy, PartialEq)]
pub struct ZnBase {
    /// must be 128 bit to deal with very small moduli
    inv_modulus: u128,
    modulus: u64,
    /// Representatives of elements may grow up to (including) this bound
    repr_bound: u64
}

///
/// A heavily optimized implementation of `Z/nZ` for `n` that have at most
/// 41 bits. For details, see [`ZnBase`].
/// 
pub type Zn = RingValue<ZnBase>;

///
/// The number of bits to which we approximate the quotient `1 / modulus`.
/// In particular, we find `floor(2^b / modulus)` and then approximate
/// `x / modulus` by `(floor(2^b / modulus) * x) / 2^b`.
/// 
const BITSHIFT: u32 = 84;
///
/// Subtract one bit, as we need this to efficiently implement negate - 
/// see also constructor assertion `2 * modulus <= repr_bound`.
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
        assert!(modulus > 1);
        let inv_modulus = (1 << BITSHIFT) / modulus as u128;
        let mut repr_bound = 1 << (inv_modulus.leading_zeros() / 2);
        repr_bound -= repr_bound % modulus;

        // necessary for bounded_reduce output to be valid
        assert!(repr_bound >= 2 * modulus);
        // necessary for bounded_reduce to work
        assert!((repr_bound as u128 * repr_bound as u128) < (1 << BITSHIFT));
        // necessary for from_int to work
        assert!(repr_bound >= (1 << 16));
        // necessary for negate to work
        assert!(repr_bound % modulus == 0);
        return ZnBase {
            modulus: modulus,
            inv_modulus: inv_modulus,
            repr_bound: repr_bound
        }
    }

    fn potential_reduce(&self, val: &mut u64) {
        if std::intrinsics::unlikely(*val > self.repr_bound) {
            *val = self.bounded_reduce(*val as u128);
        }
    }

    ///
    /// If input is `< 1 << BITSHIFT` (and `<= repr_bound * repr_bound`), 
    /// then output is smaller than `2 * self.modulus` and congruent to the input.
    /// 
    /// Note that we also need `input < repr_bound * repr_bound`, otherwise
    /// the 128-bit multiplication will overflow.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        debug_assert!((self.repr_bound as u128 * self.repr_bound as u128) < (1 << BITSHIFT));
        debug_assert!(value <= self.repr_bound as u128 * self.repr_bound as u128);
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
        debug_assert!(result.0 <= self.repr_bound);
        return result;
    }
}

impl_eq_based_self_iso!{ ZnBase }

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

trait GenericMapInFromInt: IntegerRing + CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i64>> {}

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

impl CanonicalHom<StaticRingBase<i16>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &StaticRingBase<i16>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _from: &StaticRingBase<i16>, el: i16, _: &()) -> Self::Element {
        // we check this in the constructor also during release
        debug_assert!(self.repr_bound >= (1 << 16));

        if el < 0 {
            self.negate(ZnEl(-(el as i32) as u64))
        } else {
            ZnEl(el as u64)
        }
    }
}

impl CanonicalHom<StaticRingBase<i8>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _from: &StaticRingBase<i8>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _from: &StaticRingBase<i8>, el: i8, _: &()) -> Self::Element {
        // we check this in the constructor also during release
        debug_assert!(self.repr_bound >= (1 << 16));

        if el < 0 {
            self.negate(ZnEl(-(el as i16) as u64))
        } else {
            ZnEl(el as u64)
        }
    }
}

pub struct TrivialMapInFromI32(bool);

impl CanonicalHom<StaticRingBase<i32>> for ZnBase {

    type Homomorphism = TrivialMapInFromI32;

    fn has_canonical_hom(&self, _from: &StaticRingBase<i32>) -> Option<Self::Homomorphism> {
        if self.repr_bound > i32::MAX as u64 {
            Some(TrivialMapInFromI32(true))
        } else {
            Some(TrivialMapInFromI32(false))
        }
    }

    fn map_in(&self, _from: &StaticRingBase<i32>, el: i32, TrivialMapInFromI32(supports_trivial_map_in): &TrivialMapInFromI32) -> Self::Element {
        // we check this in the constructor also during release
        debug_assert!(self.repr_bound >= (1 << 16));

        if std::intrinsics::likely(*supports_trivial_map_in) {
            if el < 0 {
                self.negate(ZnEl(-(el as i64) as u64))
            } else {
                ZnEl(el as u64)
            }
        } else {
            if el < 0 {
                self.negate(ZnEl(self.bounded_reduce(-(el as i64) as u128)))
            } else {
                ZnEl(self.bounded_reduce(el as u128))
            }
        }
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

///
/// Wraps [`ZnBase`] to represent an instance of the ring `Z/nZ`.
/// As opposed to [`ZnBase`], elements are stored with additional information
/// to speed up multiplication `ZnBase x ZnFastmulBase -> ZnBase`, by
/// using [`CanonicalHom::mul_assign_map_in()`].
/// Note that normal arithmetic in this ring is much slower than [`ZnBase`].
/// 
/// # Example
/// The following use of the FFT is usually faster than the standard use, as
/// the FFT requires a high amount of multiplications with the internally stored
/// roots of unity.
/// ```
/// # #![feature(const_type_name)]
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// # use feanor_math::mempool::*;
/// # use feanor_math::default_memory_provider;
/// let ring = Zn::new(1073872897);
/// let fastmul_ring = ZnFastmul::new(ring);
/// // The values stored by the FFT table are elements of `ZnFastmulBase`
/// let fft = FFTTableCooleyTuckey::for_zn(&fastmul_ring, 15).unwrap();
/// // Note that data uses `ZnBase`
/// let mut data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
/// fft.unordered_fft(&mut data[..], &ring, &default_memory_provider!());
/// ```
/// 
#[derive(PartialEq, Clone, Copy)]
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

///
/// An element of [`ZnFastmulBase`].
/// 
pub struct ZnFastmulEl {
    // representatives are always reduced, except temporarily when using `delegate_mut()`
    base: ZnEl,
    // the value `floor((x * 2^42) / p)`
    x_shift_over_p: u64
}

impl DelegateRing for ZnFastmulBase {

    type Base = ZnBase;
    type Element = ZnFastmulEl;

    fn get_delegate(&self) -> &Self::Base {
        &self.base
    }
    
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.base
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.base
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.base
    }

    fn postprocess_delegate_mut(&self, el: &mut Self::Element) {
        el.base.0 = el.base.0 % self.base.modulus;
        el.x_shift_over_p = (el.base.0 as u128 * (1u128 << (BITSHIFT / 2)) as u128 / self.base.modulus as u128) as u64;
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        let mut result = ZnFastmulEl {
            base: el,
            x_shift_over_p: 0
        };
        self.postprocess_delegate_mut(&mut result);
        return result;
    }
}

impl_eq_based_self_iso!{ ZnFastmulBase }

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

    fn mul_assign_map_in_ref(&self, _: &ZnFastmulBase, ZnEl(lhs): &mut Self::Element, twiddle: &<ZnFastmulBase as RingBase>::Element, _: &Self::Homomorphism) {
        debug_assert!(*lhs <= self.repr_bound);
        debug_assert!(twiddle.base.0 < self.modulus);
        // the upper parts of product will cancel out, so only compute the lower parts
        let product = (*lhs).wrapping_mul(twiddle.base.0);
        // the quotient fits into u64 as `*lhs <= self.repr_bound` has at most `BITSHIFT / 2` bits
        let quotient = ((*lhs as u128 * twiddle.x_shift_over_p as u128) >> (BITSHIFT / 2)) as u64;
        *lhs = product.wrapping_sub(quotient.wrapping_mul(self.modulus)) as u64;
        debug_assert!(*lhs < 2 * self.modulus);
    }

    fn mul_assign_map_in(&self, from: &ZnFastmulBase, lhs: &mut Self::Element, rhs: <ZnFastmulBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_map_in_ref(from, lhs, &rhs, hom);
    }
}

impl CooleyTuckeyButterfly<ZnFastmulBase> for ZnBase {

    #[inline(always)]
    fn butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, from: &ZnFastmulBase, hom: &<Self as CanonicalHom<ZnBase>>::Homomorphism, values: &mut V, twiddle: &<ZnFastmulBase as RingBase>::Element, i1: usize, i2: usize) {
        let a = *values.at(i1);
        let mut b = *values.at(i2);
        self.mul_assign_map_in_ref(from, &mut b, twiddle, hom);

        // this is implied by `bounded_reduce`, check anyway
        debug_assert!(b.0 < self.modulus * 2);
        debug_assert!(self.repr_bound >= self.modulus * 2);

        *values.at_mut(i1) = self.add(a, b);
        *values.at_mut(i2) = self.add(a, ZnEl(2 * self.modulus - b.0));
    }

    #[inline(always)]
    fn inv_butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, from: &ZnFastmulBase, hom: &<Self as CanonicalHom<ZnBase>>::Homomorphism, values: &mut V, twiddle: &<ZnFastmulBase as RingBase>::Element, i1: usize, i2: usize) {
        let a = *values.at(i1);
        let b = *values.at(i2);

        let b_reduced = ZnEl(self.bounded_reduce(b.0 as u128));

        *values.at_mut(i1) = self.add(a, b_reduced);
        *values.at_mut(i2) = self.add(a, ZnEl(2 * self.modulus - b_reduced.0));
        self.mul_assign_map_in_ref(from, values.at_mut(i2), twiddle, hom);
    }
}

impl<I: ?Sized + IntegerRing> CanonicalHom<I> for ZnFastmulBase 
    where ZnBase: CanonicalHom<I>
{

    type Homomorphism = <ZnBase as CanonicalHom<I>>::Homomorphism;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
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

    fn map_out(&self, from: &ZnFastmulBase, el: Self::Element, _: &Self::Isomorphism) -> <ZnFastmulBase as RingBase>::Element {
        let mut result = ZnFastmulEl {
            base: el,
            x_shift_over_p: 0
        };
        from.postprocess_delegate_mut(&mut result);
        return result;
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
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_ring_axioms() {
    let ring = Zn::new(2);
    generic_test_ring_axioms(&ring, ring.elements());

    let ring = Zn::new(63);
    generic_test_ring_axioms(&ring, ring.elements());

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

    let ZZbig = RustBigintRing::RING;
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
fn test_canonical_iso_axioms_as_field() {
    let R = Zn::new(17);
    let R2 = R.clone().as_field().ok().unwrap();
    generic_test_canonical_hom_axioms(&R, &R2, R.elements());
    generic_test_canonical_iso_axioms(&R, &R2, R.elements());
    generic_test_canonical_hom_axioms(&R2, &R, R2.elements());
    generic_test_canonical_iso_axioms(&R2, &R, R2.elements());
}

#[test]
fn test_cooley_tuckey_butterfly() {
    let ring = Zn::new(2);
    generic_test_cooley_tuckey_butterfly(ring, ring, ring.elements(), &ring.one());

    let ring = Zn::new(97);
    generic_test_cooley_tuckey_butterfly(ring, ring, ring.elements(), &ring.from_int(3));

    let ring = Zn::new((1 << 41) - 1);
    generic_test_cooley_tuckey_butterfly(ring, ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1].iter().cloned().map(|x| ring.from_int(x)), &ring.from_int(3));
}

#[test]
fn test_cooley_tuckey_butterfly_fastmul() {
    let ring = Zn::new(2);
    let fastmul_ring = ZnFastmul::new(ring);
    generic_test_cooley_tuckey_butterfly(ring, fastmul_ring, ring.elements(), &fastmul_ring.one());

    let ring = Zn::new(97);
    let fastmul_ring = ZnFastmul::new(ring);
    generic_test_cooley_tuckey_butterfly(ring, fastmul_ring, ring.elements(), &fastmul_ring.from_int(3));

    let ring = Zn::new((1 << 41) - 1);
    let fastmul_ring = ZnFastmul::new(ring);
    generic_test_cooley_tuckey_butterfly(ring, fastmul_ring, [0, 1, 2, 3, 4, (1 << 20), (1 << 20) + 1, (1 << 21), (1 << 21) + 1].iter().cloned().map(|x| ring.from_int(x)), &fastmul_ring.from_int(3));
}