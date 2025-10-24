use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyButterfly;
use crate::algorithms::fft::radix3::CooleyTukeyRadix3Butterfly;
use crate::reduce_lift::lift_poly_eval::InterpolationBaseRing;
use crate::delegate::DelegateRing;
use crate::delegate::DelegateRingImplFiniteRing;
use crate::divisibility::*;
use crate::{impl_eq_based_self_iso, impl_localpir_wrap_unwrap_homs, impl_localpir_wrap_unwrap_isos, impl_field_wrap_unwrap_homs, impl_field_wrap_unwrap_isos};
use crate::ordered::OrderedRingStore;
use crate::primitive_int::*;
use crate::integer::*;
use crate::pid::*;
use crate::rings::extension::FreeAlgebraStore;
use crate::ring::*;
use crate::rings::extension::galois_field::*;
use crate::seq::*;
use crate::serialization::*;
use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::matmul::ComputeInnerProduct;
use crate::algorithms::matmul::StrassenHint;
use crate::specialization::*;
use crate::homomorphism::*;

use std::marker::PhantomData;
use std::fmt::Debug;

use feanor_serde::newtype_struct::*;
use serde::de::{DeserializeSeed, Error};
use serde::{Deserialize, Deserializer, Serialize, Serializer}; 

use super::*;
use super::zn_big;

fn high(x: u128) -> u64 {
    (x >> 64) as u64
}

fn low(x: u128) -> u64 {
    (x & ((1 << 64) - 1)) as u64
}

fn mulhi(lhs: u64, rhs: u64) -> u64 {
    high(lhs as u128 * rhs as u128)
}

fn mullo(lhs: u64, rhs: u64) -> u64 {
    lhs.wrapping_mul(rhs)
}

///
/// Assuming that `x <= 2 * bound`, computes `x' <= bound` that is congruent
/// to `x` modulo `bound`
/// 
fn reduce_to_half(x: u64, bound: u64) -> u64 {
    let (result, overflow) = x.overflowing_sub(bound);
    if overflow {
        x
    } else {
        result
    }
}

///
/// Represents the ring `Z/nZ` for `n` somewhat below 64 bits in size.
/// 
/// The `64B` in the name stands for "64-bit Barett", which is the kind
/// of reduction internally used.
/// 
/// More concretely, the currently maximal supported modulus is `floor(2^62 / 9)`.
/// Note that the exact value might change in the future.
/// 
/// # Examples
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// let zn = Zn64B::new(7);
/// assert_el_eq!(zn, zn.one(), zn.mul(zn.int_hom().map(3), zn.int_hom().map(5)));
/// ```
/// Too large moduli will give an error.
/// ```should_panic
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// Zn64B::new((1 << 62) / 9 + 1);
/// ```
/// 
#[derive(Clone, Copy)]
pub struct Zn64BBase {
    modulus: i64,
    modulus_half: i64,
    modulus_times_three: u64,
    modulus_times_six: u64,
    inv_modulus: u128
}

const fn two_pow_128_div(x: u64) -> u128 {
    let q_high = (1u128 << 64) / (x as u128);
    let r_high = (1u128 << 64) % (x as u128);
    let q_low = (r_high << 64) / (x as u128);
    let result = (q_high << 64) | q_low;
    debug_assert!(result.wrapping_mul(x as u128) == 0 || result.wrapping_mul(x as u128) >= 0u128.wrapping_sub(x as u128));
    return result;
}

impl Zn64BBase {

    pub const fn new(modulus: u64) -> Self {
        assert!(modulus > 1);
        // this should imply the statement we need later
        assert!(modulus <= ((1 << 62) / 9));
        // we want representatives to grow up to 6 * modulus
        assert!(modulus as u128 * 6 <= u64::MAX as u128);
        let modulus_i64: i64 = modulus as i64;
        let inv_modulus = two_pow_128_div(modulus);
        // we need the product `inv_modulus * (6 * modulus)^2` to fit into 192 bit, should be implied by `modulus < ((1 << 62) / 9)`
        // debug_assert!(ZZbig.is_lt(&ZZbig.mul_ref_fst(&int_cast(inv_modulus as i128, ZZbig, StaticRing::<i128>::RING), ZZbig.pow(ZZbig.int_hom().mul_map(ZZbig.coerce(&ZZ, modulus_i64), 6), 2)), &ZZbig.power_of_two(192)));
        Self {
            modulus: modulus_i64,
            inv_modulus: inv_modulus,
            modulus_half: (modulus_i64 - 1) / 2 + 1,
            modulus_times_three: modulus * 3,
            modulus_times_six: modulus * 6
        }
    }

    fn modulus_u64(&self) -> u64 {
        self.modulus as u64
    }

    ///
    /// Positive integers bounded by `self.repr_bound()` (inclusive) are considered
    /// valid representatives of ring elements.
    /// 
    fn repr_bound(&self) -> u64 {
        self.modulus_times_six
    }

    ///
    /// Reduces from `[0, self.repr_bound() * self.repr_bound()]` to `[0, 3 * self.modulus()[`.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        debug_assert!(value <= self.repr_bound() as u128 * self.repr_bound() as u128);

        let (in_low, in_high) = (low(value), high(value));
        let (invmod_low, invmod_high) = (low(self.inv_modulus), high(self.inv_modulus));
        // we ignore the lowest part of the sum, causing an error of at most 1;
        // we also assume that `repr_bound * repr_bound * inv_modulus` fits into 192 bit
        let approx_quotient = mulhi(in_low, invmod_high) + mulhi(in_high, invmod_low) + mullo(in_high, invmod_high);
        let result = low(value).wrapping_sub(mullo(approx_quotient, self.modulus_u64()));

        debug_assert!(result < self.modulus_times_three);
        debug_assert!((value - result as u128) % (self.modulus_u64() as u128) == 0);
        return result;
    }

    ///
    /// Reduces from `[0, FACTOR * self.repr_bound() * self.repr_bound()]` to `[0, 3 * self.modulus()]`.
    /// 
    /// As opposed to the faster [`ZnBase::bounded_reduce()`], this should work for all inputs in `u128`
    /// (assuming configured with the right value of `FACTOR`). Currently only used by the specialization of 
    /// [`ComputeInnerProduct::inner_product()`].
    /// 
    #[inline(never)]
    fn bounded_reduce_larger<const FACTOR: usize>(&self, value: u128) -> u64 {
        assert!(FACTOR == 32);
        debug_assert!(value <= FACTOR as u128 * self.repr_bound() as u128 * self.repr_bound() as u128);

        let (in_low, in_high) = (low(value), high(value));
        let invmod_high = high(self.inv_modulus);
        // `approx_quotient` can be just slightly larger than 64 bits, since we optimized for `bounded_reduce()`
        let approx_quotient = in_high as u128 * invmod_high as u128 + mulhi(in_low, invmod_high) as u128;

        return self.bounded_reduce(value - (approx_quotient * self.modulus as u128));
    }

    ///
    /// Creates a `ZnEl` from an integer that is already sufficiently reduced,
    /// i.e. within `[0, self.repr_bound()]`.
    /// 
    /// The reducedness of the input is only checked when debug assertions are
    /// enabled.
    /// 
    fn from_u64_promise_reduced(&self, value: u64) -> Zn64BEl {
        debug_assert!(value <= self.repr_bound());
        Zn64BEl(value)
    }

    ///
    /// Reduces from `[0, self.repr_bound()]` to `[0, self.modulus()[`
    /// 
    fn complete_reduce(&self, mut value: u64) -> u64 {
        debug_assert!(value <= self.repr_bound());
        if value >= 4 * self.modulus_u64() {
            value -= 4 * self.modulus_u64();
        }
        if value >= 2 * self.modulus_u64() {
            value -= 2 * self.modulus_u64();
        }
        if value >= self.modulus_u64() {
            value -= self.modulus_u64();
        }
        debug_assert!(value < self.modulus_u64());
        return value;
    }

    #[stability::unstable(feature = "enable")]
    pub fn from_primitive_int<T: PrimitiveInt>(&self, el: T) -> Zn64BEl {
        let is_neg = StaticRing::<T>::RING.is_neg(&el);
        let el_abs = <T as Into<i128>>::into(el).unsigned_abs();
        if el_abs <= self.modulus_u64() as u128 {
            if is_neg {
                self.negate(self.from_u64_promise_reduced(el_abs as u64))
            } else {
                self.from_u64_promise_reduced(el_abs as u64)
            }
        } else if el_abs <= self.repr_bound() as u128 {
            if is_neg {
                self.negate(self.from_u64_promise_reduced(self.bounded_reduce(el_abs)))
            } else {
                self.from_u64_promise_reduced(self.bounded_reduce(el_abs))
            }
        } else {
            let el_i128 = <T as Into<i128>>::into(el);
            if is_neg {
                self.from_u64_promise_reduced(((el_i128 % self.modulus as i128) as i64 + self.modulus) as u64)
            } else {
                self.from_u64_promise_reduced((el_i128 % self.modulus as i128) as u64)
            }
        }
    }
}

/// 
/// Represents the ring `Z/nZ`, using a variant of Barett reduction
/// for moduli somewhat smaller than 64 bits. For details, see [`ZnBase`].
/// 
pub type Zn64B = RingValue<Zn64BBase>;

impl Zn64B {

    pub const fn new(modulus: u64) -> Self {
        RingValue::from(Zn64BBase::new(modulus))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Zn64BEl(/* a representative within `[0, ring.repr_bound()]` */ u64);

impl PartialEq for Zn64BBase {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl Debug for Zn64BBase {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z/{}Z", self.modulus)
    }
}

impl RingBase for Zn64BBase {

    type Element = Zn64BEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 += rhs.0;
        lhs.0 = reduce_to_half(lhs.0, self.repr_bound());
        debug_assert!(lhs.0 <= self.repr_bound());
    }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        let (result, overflow) = lhs.0.overflowing_sub(rhs.0);
        if overflow {
            lhs.0 = result.wrapping_add(self.repr_bound());
        } else {
            lhs.0 = result;
        }
        debug_assert!(lhs.0 <= self.repr_bound());
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        lhs.0 = self.repr_bound() - lhs.0;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.bounded_reduce(lhs.0 as u128 * rhs.0 as u128);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from_primitive_int(value)
    }

    fn zero(&self) -> Self::Element {
        Zn64BEl(0)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        if lhs.0 > rhs.0 {
            self.is_zero(&self.from_u64_promise_reduced(lhs.0 - rhs.0))
        } else {
            self.is_zero(&self.from_u64_promise_reduced(rhs.0 - lhs.0))
        }
    }

    fn is_zero(&self, val: &Self::Element) -> bool {
        debug_assert!(val.0 <= self.repr_bound());
        self.complete_reduce(val.0) == 0
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _: EnvBindingStrength) -> std::fmt::Result {
        write!(out, "{}", self.complete_reduce(value.0))
    }
    
    fn characteristic<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.size(other_ZZ)
    }
    
    fn sum<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        let mut result = self.zero();
        let mut els_it = els.into_iter();
        while let Some(Zn64BEl(start)) = els_it.next() {
            let mut current = start as u128;
            for Zn64BEl(c) in els_it.by_ref().take(self.repr_bound() as usize / 2 - 1) {
                current += c as u128;
            }
            self.add_assign(&mut result, self.from_u64_promise_reduced(self.bounded_reduce(current)));
        }
        debug_assert!(result.0 <= self.repr_bound());
        return result;
    }

    fn pow_gen<R: RingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        assert!(!integers.is_neg(power));
        algorithms::sqr_mul::generic_abs_square_and_multiply(
            x, 
            power, 
            &integers,
            |mut a| {
                self.square(&mut a);
                a
            },
            |a, b| self.mul_ref_fst(a, b),
            self.one()
        )
    }

    fn is_approximate(&self) -> bool { false }
}

impl SerializableElementRing for Zn64BBase {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        <i64 as Deserialize>::deserialize(deserializer)
            .and_then(|x| if x < 0 || x >= *self.modulus() { Err(Error::custom("ring element value out of bounds for ring Z/nZ")) } else { Ok(x) })
            .map(|x| self.from_int_promise_reduced(x))
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        <i64 as Serialize>::serialize(&self.smallest_positive_lift(*el), serializer)
    }
}

impl FromModulusCreateableZnRing for Zn64BBase {

    fn from_modulus<F, E>(create_modulus: F) -> Result<Self, E>
        where F: FnOnce(&Self::IntegerRingBase) -> Result<El<Self::IntegerRing>, E>
    {
        create_modulus(StaticRing::<i64>::RING.get_ring()).map(|n| Self::new(n as u64))
    }
}

impl InterpolationBaseRing for AsFieldBase<Zn64B> {

    type ExtendedRingBase<'a> = GaloisFieldBaseOver<RingRef<'a, Self>>
        where Self: 'a;

    type ExtendedRing<'a> = GaloisFieldOver<RingRef<'a, Self>>
        where Self: 'a;

    fn in_base<'a, S>(&self, ext_ring: S, el: El<S>) -> Option<Self::Element>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        let wrt_basis = ext_ring.wrt_canonical_basis(&el);
        if wrt_basis.iter().skip(1).all(|x| self.is_zero(&x)) {
            return Some(wrt_basis.at(0));
        } else {
            return None;
        }
    }

    fn in_extension<'a, S>(&self, ext_ring: S, el: Self::Element) -> El<S>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        ext_ring.inclusion().map(el)
    }

    fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>) {
        let ring = super::generic_impls::interpolation_ring(RingRef::new(self), count);
        let points = ring.elements().take(count).collect();
        return (ring, points);
    }
}

impl ComputeInnerProduct for Zn64BBase {

    fn inner_product<I: IntoIterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {

        debug_assert!(u128::MAX / (self.repr_bound() as u128 * self.repr_bound() as u128) >= 36);
        const REDUCE_AFTER_STEPS: usize = 32;
        let mut array_chunks = els.into_iter().array_chunks::<REDUCE_AFTER_STEPS>();
        let mut result = self.zero();
        while let Some(chunk) = array_chunks.next() {
            let mut sum: u128 = 0;
            for (l, r) in chunk {
                debug_assert!(l.0 <= self.repr_bound());
                debug_assert!(r.0 <= self.repr_bound());
                sum += l.0 as u128 * r.0 as u128;
            }
            self.add_assign(&mut result, Zn64BEl(self.bounded_reduce_larger::<REDUCE_AFTER_STEPS>(sum)));
        }
        let mut sum: u128 = 0;
        for (l, r) in array_chunks.into_remainder().unwrap() {
            debug_assert!(l.0 <= self.repr_bound());
            debug_assert!(r.0 <= self.repr_bound());
            sum += l.0 as u128 * r.0 as u128;
        }
        self.add_assign(&mut result, Zn64BEl(self.bounded_reduce_larger::<REDUCE_AFTER_STEPS>(sum)));
        return result;
    }

    fn inner_product_ref_fst<'a, I: IntoIterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product(els.into_iter().map(|(l, r)| (self.clone_el(l), r)))
    }

    fn inner_product_ref<'a, I: IntoIterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product_ref_fst(els.into_iter().map(|(l, r)| (l, self.clone_el(r))))
    }
}

impl_eq_based_self_iso!{ Zn64BBase }

impl<I: RingStore> CanHomFrom<zn_big::ZnGBBase<I>> for Zn64BBase
    where I::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Homomorphism> {
        if from.integer_ring().get_ring().representable_bits().is_none() || from.integer_ring().get_ring().representable_bits().unwrap() >= self.integer_ring().abs_log2_ceil(self.modulus()).unwrap() {
            if from.integer_ring().eq_el(from.modulus(), &int_cast(*self.modulus(), from.integer_ring(), self.integer_ring())) {
                Some(())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_big::ZnGBBase<I>, el: <zn_big::ZnGBBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        self.from_u64_promise_reduced(int_cast(from.smallest_positive_lift(el), self.integer_ring(), from.integer_ring()) as u64)
    }
}

impl<I: RingStore> CanIsoFromTo<zn_big::ZnGBBase<I>> for Zn64BBase
    where I::Type: IntegerRing
{
    type Isomorphism = <zn_big::ZnGBBase<I> as CanHomFrom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Isomorphism> {
        if from.integer_ring().get_ring().representable_bits().is_none() || from.integer_ring().get_ring().representable_bits().unwrap() >= self.integer_ring().abs_log2_ceil(self.modulus()).unwrap() {
            if from.integer_ring().eq_el(from.modulus(), &int_cast(*self.modulus(), from.integer_ring(), self.integer_ring())) {
                from.has_canonical_hom(self.integer_ring().get_ring())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_big::ZnGBBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_big::ZnGBBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0.try_into().unwrap(), iso)
    }
}

impl CanHomFrom<zn_64m::Zn64MBase> for Zn64BBase {
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_64m::Zn64MBase) -> Option<Self::Homomorphism> {
        if self.modulus() == from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_64m::Zn64MBase, el: <zn_64m::Zn64MBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        self.from_int_promise_reduced(from.smallest_positive_lift(el))
    }
}

impl CanIsoFromTo<zn_64m::Zn64MBase> for Zn64BBase {
    type Isomorphism = <zn_64m::Zn64MBase as CanHomFrom<Self>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_64m::Zn64MBase) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &zn_64m::Zn64MBase, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_64m::Zn64MBase as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

///
/// Data associated to an element of [`ZnBase`] that allows for faster division. 
/// For details, see [`DivisibilityRing::prepare_divisor()`].
/// 
#[derive(Copy, Clone, Debug)]
pub struct ZnPreparedDivisorData {
    unit_part: El<Zn64B>,
    is_unit: bool,
    smallest_positive_zero_divisor_part: PreparedDivisor<StaticRingBase<i64>>
}

impl DivisibilityRing for Zn64BBase {

    type PreparedDivisorData = ZnPreparedDivisorData;

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs)
    }

    fn prepare_divisor(&self, x: &Self::Element) -> Self::PreparedDivisorData {
        let (s, _t, d) = self.integer_ring().extended_ideal_gen(&self.smallest_positive_lift(*x), self.modulus());
        debug_assert!(d > 0);
        debug_assert!(d <= *self.modulus());
        return ZnPreparedDivisorData {
            is_unit: d == 1,
            unit_part: if s < 0 { self.negate(self.from_u64_promise_reduced(-s as u64)) } else { self.from_u64_promise_reduced(s as u64) },
            smallest_positive_zero_divisor_part: PreparedDivisor::new(StaticRing::<i64>::RING.get_ring(), d)
        };
    }

    fn checked_left_div_prepared(&self, lhs: &Self::Element, _rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> Option<Self::Element> {
        if rhs_prep.is_unit {
            Some(self.mul_ref(lhs, &rhs_prep.unit_part))
        } else {
            StaticRing::<i64>::RING.get_ring().checked_left_div_prepared(&self.smallest_positive_lift(*lhs), &rhs_prep.smallest_positive_zero_divisor_part.element, &rhs_prep.smallest_positive_zero_divisor_part.data)
                .map(|x| self.mul(self.from_u64_promise_reduced(x as u64), rhs_prep.unit_part))
        }
    }
}

impl<I: ?Sized + IntegerRing> CanHomFrom<I> for Zn64BBase {

    type Homomorphism = super::generic_impls::BigIntToZnHom<I, StaticRingBase<i128>, Self>;

    default fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_bigint(from, self, StaticRing::<i128>::RING.get_ring(), Some(&(self.repr_bound() as i128 * self.repr_bound() as i128)))
    }

    default fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in_from_bigint(from, self, StaticRing::<i128>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as u64) < self.modulus_u64());
            self.from_u64_promise_reduced(n as u64)
        }, |n| {
            debug_assert!(n <= (self.repr_bound() as i128 * self.repr_bound() as i128));
            self.from_u64_promise_reduced(self.bounded_reduce(n as u128))
        })
    }
}

impl Serialize for Zn64BBase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("Zn", *self.modulus()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Zn64BBase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("Zn", PhantomData::<i64>).deserialize(deserializer).map(|n| Zn64BBase::new(n as u64))
    }
}

macro_rules! impl_static_int_to_zn {
    ($($int:ident),*) => {
        $(
            impl CanHomFrom<StaticRingBase<$int>> for Zn64BBase {
                fn map_in(&self, _from: &StaticRingBase<$int>, el: $int, _hom: &Self::Homomorphism) -> Self::Element {
                    self.from_primitive_int(el)
                }
            }
        )*
    };
}

impl_static_int_to_zn!{ i8, i16, i32, i64, i128 }

#[derive(Clone, Copy)]
pub struct ZnBaseElementsIter<'a> {
    ring: &'a Zn64BBase,
    current: u64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

    type Item = Zn64BEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus_u64() {
            let result = self.current;
            self.current += 1;
            return Some(self.ring.from_u64_promise_reduced(result));
        } else {
            return None;
        }
    }
}

impl FiniteRingSpecializable for Zn64BBase {
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.execute()
    }
}

impl FiniteRing for Zn64BBase {

    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        super::generic_impls::random_element(self, rng)
    }

    fn size<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        if other_ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(&(self.modulus() + 1)) <= other_ZZ.get_ring().representable_bits() {
            Some(int_cast(*self.modulus(), other_ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl PrincipalIdealRing for Zn64BBase {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_div_min(RingRef::new(self), lhs, rhs)
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = StaticRing::<i64>::RING.extended_ideal_gen(&lhs.0.try_into().unwrap(), &rhs.0.try_into().unwrap());
        let quo = RingRef::new(self).into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        (quo.map(s), quo.map(t), quo.map(d))
    }
}

impl StrassenHint for Zn64BBase {
    default fn strassen_threshold(&self) -> usize {
        6
    }
}

impl KaratsubaHint for Zn64BBase {
    default fn karatsuba_threshold(&self) -> usize {
        6
    }
}

impl ZnRing for Zn64BBase {

    type IntegerRingBase = StaticRingBase<i64>;
    type IntegerRing = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::IntegerRing {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.complete_reduce(el.0) as i64
    }

    fn smallest_lift(&self, Zn64BEl(mut value_u64): Self::Element) -> El<Self::IntegerRing> {
        debug_assert!(value_u64 <= self.repr_bound());
        // value is in [0, 6 * self.modulus]
        if value_u64 >= 3 * self.modulus_u64() {
            value_u64 -= 3 * self.modulus_u64();
        }
        // value is in [0, 3 * self.modulus]
        let mut value_i64 = value_u64 as i64;
        if value_i64 >= self.modulus + self.modulus_half {
            value_i64 -= 2 * self.modulus;
        }
        // value is in ]-self.modulus_half, self.modulus + self.modulus_half[ if modulus is odd
        // value is in [-self.modulus_half, self.modulus + self.modulus_half[ if modulus is even
        if value_i64 >= self.modulus_half {
            value_i64 -= self.modulus;
        }
        // value is in ]-self.modulus_half, self.modulus_half[ if modulus is odd
        // value is in [-self.modulus_half, self.modulus_half[ if modulus is even
        debug_assert!(value_i64 < self.modulus_half);
        debug_assert!(self.modulus() % 2 == 0 || value_i64 > -self.modulus_half);
        debug_assert!(self.modulus() % 2 == 1 || value_i64 >= -self.modulus_half);
        return value_i64;
    }

    fn modulus(&self) -> &El<Self::IntegerRing> {
        &self.modulus
    }

    fn any_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        el.0 as i64
    }

    ///
    /// If the given integer is within `{ 0, ..., 6 * n }`, returns the corresponding
    /// element in `Z/nZ`. Any other input is considered a logic error.
    /// 
    /// This function follows [`ZnRing::from_int_promise_reduced()`], but is guaranteed
    /// to work on elements `{ 0, ..., 6 * n }` instead of only `{ 0, ..., n - 1 }`.
    /// 
    /// # Examples
    /// ```rust
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let ring = Zn64B::new(7);
    /// assert_el_eq!(ring, ring.zero(), ring.get_ring().from_int_promise_reduced(42));
    /// ```
    /// Larger values lead to a panic in debug mode, and to a logic error in release mode.
    /// ```should_panic
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let ring = Zn64B::new(7);
    /// ring.get_ring().from_int_promise_reduced(43);
    /// ```
    /// 
    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        debug_assert!(self.repr_bound() == 6 * self.modulus_u64());
        debug_assert!(x >= 0);
        debug_assert!(x as u64 <= self.repr_bound());
        self.from_u64_promise_reduced(x as u64)
    }
}

impl HashableElRing for Zn64BBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.integer_ring().hash(&self.smallest_positive_lift(*el), h)
    }
}

///
/// Wraps [`ZnBase`] to represent an instance of the ring `Z/nZ`.
/// As opposed to [`ZnBase`], elements are stored with additional information
/// to speed up multiplication `ZnBase x ZnFastmulBase -> ZnBase`, by
/// using [`CanHomFrom::mul_assign_map_in()`].
/// Note that normal arithmetic in this ring is much slower than [`ZnBase`].
/// 
/// # Example
/// The following use of the FFT is usually faster than the standard use, as
/// the FFT requires a high amount of multiplications with the internally stored
/// roots of unity.
/// ```rust
/// # #![feature(const_type_name)]
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// let ring = Zn64B::new(1073872897);
/// let fastmul_ring = ZnFastmul::new(ring).unwrap();
/// // The values stored by the FFT table are elements of `ZnFastmulBase`
/// let fft = CooleyTuckeyFFT::for_zn_with_hom(ring.can_hom(&fastmul_ring).unwrap(), 15).unwrap();
/// // Note that data uses `ZnBase`
/// let mut data = (0..(1 << 15)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
/// fft.unordered_fft(&mut data[..], &ring);
/// ```
/// 
#[derive(Clone, Copy, Debug)]
pub struct ZnFastmulBase {
    base: RingValue<Zn64BBase>
}

///
/// An implementation of `Z/nZ` for `n` that is optimized to provide fast multiplication with elements
/// of [`Zn`]. For details, see [`ZnFastmulBase`].
/// 
pub type ZnFastmul = RingValue<ZnFastmulBase>;

impl ZnFastmul {

    pub fn new(base: Zn64B) -> Option<Self> {
        Some(RingValue::from(ZnFastmulBase { base }))
    }
}

impl PartialEq for ZnFastmulBase {
    
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl PrincipalIdealRing for ZnFastmulBase {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let result = self.get_delegate().checked_div_min(self.delegate_ref(lhs), self.delegate_ref(rhs));
        result.map(|x| self.rev_delegate(x))
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self.get_delegate().extended_ideal_gen(self.delegate_ref(lhs), self.delegate_ref(rhs));
        (self.rev_delegate(s), self.rev_delegate(t), self.rev_delegate(d))
    }
}

impl_eq_based_self_iso!{ ZnFastmulBase }

#[derive(Clone, Copy, Debug)]
pub struct ZnFastmulEl {
    el: Zn64BEl,
    value_invmod_shifted: u64
}

impl DelegateRing for ZnFastmulBase {

    type Base = Zn64BBase;
    type Element = ZnFastmulEl;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.el
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.el
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.el
    }

    fn postprocess_delegate_mut(&self, el: &mut Self::Element) {
        assert!(el.el.0 <= self.base.get_ring().repr_bound());
        el.el.0 = self.base.get_ring().complete_reduce(el.el.0);
        let value = el.el.0;
        el.value_invmod_shifted = (((value as u128) << 64) / self.base.get_ring().modulus_u64() as u128) as u64;
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        let mut result = ZnFastmulEl {
            el: el,
            value_invmod_shifted: 0
        };
        self.postprocess_delegate_mut(&mut result);
        return result;
    }
}

impl DelegateRingImplFiniteRing for ZnFastmulBase {}

impl CanHomFrom<Zn64BBase> for ZnFastmulBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &Zn64BBase) -> Option<Self::Homomorphism> {
        if from == self.base.get_ring() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _from: &Zn64BBase, el: <Zn64BBase as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(el)
    }
}

impl CanHomFrom<ZnFastmulBase> for Zn64BBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnFastmulBase) -> Option<Self::Homomorphism> {
        if self == from.base.get_ring() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _from: &ZnFastmulBase, el: <ZnFastmulBase as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        el.el
    }

    fn mul_assign_map_in(&self, _from: &ZnFastmulBase, lhs: &mut Self::Element, rhs: <ZnFastmulBase as RingBase>::Element, _hom: &Self::Homomorphism) {
        debug_assert!(lhs.0 <= self.repr_bound());
        let lhs_original = lhs.0;
        let product = mullo(lhs.0, rhs.el.0);
        let approx_quotient = mulhi(lhs.0, rhs.value_invmod_shifted);
        lhs.0 = product.wrapping_sub(mullo(approx_quotient, self.modulus_u64()));
        debug_assert!(lhs.0 < self.modulus_times_three);
        debug_assert!((lhs_original as u128 * rhs.el.0 as u128 - lhs.0 as u128) % (self.modulus_u64() as u128) == 0);
    }

    fn mul_assign_map_in_ref(&self, from: &ZnFastmulBase, lhs: &mut Self::Element, rhs: &<ZnFastmulBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_map_in(from, lhs, *rhs, hom);
    }
}

impl CanIsoFromTo<ZnFastmulBase> for Zn64BBase {

    type Isomorphism = <ZnFastmulBase as CanHomFrom<Self>>::Homomorphism;

    fn has_canonical_iso(&self, from: &ZnFastmulBase) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &ZnFastmulBase, el: Self::Element, iso: &Self::Isomorphism) -> <ZnFastmulBase as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

impl CooleyTuckeyButterfly<ZnFastmulBase> for Zn64BBase {

    #[inline(always)]
    fn butterfly_new<H: Homomorphism<ZnFastmulBase, Self>>(hom: H, a: &mut Zn64BEl, b: &mut Zn64BEl, twiddle: &<ZnFastmulBase as RingBase>::Element) {
        let self_ = hom.codomain().get_ring();
        
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        hom.mul_assign_ref_map(b, twiddle);

        (a.0, b.0) = (a.0 + b.0, a.0 + self_.modulus_times_three - b.0);
    }

    #[inline(always)]
    fn inv_butterfly_new<H: Homomorphism<ZnFastmulBase, Self>>(hom: H, a: &mut Zn64BEl, b: &mut Zn64BEl, twiddle: &<ZnFastmulBase as RingBase>::Element) {
        let self_ = hom.codomain().get_ring();

        // we assume that a, b are already at most `self.modulus_times_three`

        (a.0, b.0) = (a.0 + b.0, a.0 + self_.modulus_times_three - b.0);
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        hom.mul_assign_ref_map(b, twiddle);
    }

    #[inline(always)]
    fn prepare_for_inv_fft(&self, value: &mut Self::Element) {
        if value.0 >= self.modulus_times_three {
            value.0 -= self.modulus_times_three;
        }
    }
}

impl CooleyTukeyRadix3Butterfly<ZnFastmulBase> for Zn64BBase {
    
    fn butterfly<H: Homomorphism<ZnFastmulBase, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element, 
        c: &mut Self::Element, 
        z: &ZnFastmulEl,
        t: &ZnFastmulEl,
        t_sqr_z_sqr: &ZnFastmulEl
    ) {
        let self_ = hom.codomain().get_ring();

        hom.mul_assign_ref_map(b, t);
        hom.mul_assign_ref_map(c, t_sqr_z_sqr);
        let b_ = hom.mul_ref_map(b, z);
        let c_ = hom.mul_ref_map(c, z);

        let mut s1 = b.0 + c_.0;
        let mut s2 = b_.0 + c.0;
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        s1 = reduce_to_half(s1, self_.modulus_times_three);
        s2 = reduce_to_half(s2, self_.modulus_times_three);

        let mut s3 = s1 + s2;
        s3 = reduce_to_half(s3, self_.modulus_times_three);

        b.0 = a.0 + s2;
        c.0 = a.0 + self_.modulus_times_three - s3;
        a.0 = a.0 + s1;
    }
    
    fn inv_butterfly<H: Homomorphism<ZnFastmulBase, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element,
        c: &mut Self::Element,
        z: &ZnFastmulEl,
        t: &ZnFastmulEl,
        t_sqr: &ZnFastmulEl
    ) {
        let self_ = hom.codomain().get_ring();

        // we assume that a, b, c are already at most `self.modulus_times_three`

        let b_ = hom.mul_ref_map(b, z);
        let mut s1 = b.0 + c.0;
        s1 = reduce_to_half(s1, self_.modulus_times_three);

        let s2 = b_.0 + c.0;
        let s2_ = hom.mul_ref_snd_map(self_.from_u64_promise_reduced(s2), z);

        let mut s3 = s1 + s2_.0;
        s3 = reduce_to_half(s3, self_.modulus_times_three);

        b.0 = a.0 + s2_.0;
        c.0 = a.0 + self_.modulus_times_three - s3;
        a.0 = a.0 + s1;

        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        hom.mul_assign_ref_map(b, t);
        hom.mul_assign_ref_map(c, t_sqr);
    }

    #[inline(always)]
    fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    #[inline(always)]
    fn prepare_for_inv_fft(&self, value: &mut Self::Element) {
        if value.0 >= self.modulus_times_three {
            value.0 -= self.modulus_times_three
        }
    }
}

impl CooleyTuckeyButterfly<Zn64BBase> for Zn64BBase {

    #[inline(always)]
    fn butterfly_new<H: Homomorphism<Self, Self>>(hom: H, a: &mut Zn64BEl, b: &mut Zn64BEl, twiddle: &Zn64BEl) {
        let self_ = hom.codomain().get_ring();
        
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        hom.mul_assign_ref_map(b, twiddle);

        (a.0, b.0) = (a.0 + b.0, a.0 + self_.modulus_times_three - b.0);
    }

    #[inline(always)]
    fn inv_butterfly_new<H: Homomorphism<Self, Self>>(hom: H, a: &mut Zn64BEl, b: &mut Zn64BEl, twiddle: &Zn64BEl) {
        let self_ = hom.codomain().get_ring();

        (a.0, b.0) = (a.0 + b.0, a.0 + self_.modulus_times_three - b.0);
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        hom.mul_assign_ref_map(b, twiddle);
    }
    
    #[inline(always)]
    fn prepare_for_inv_fft(&self, value: &mut Self::Element) {
        if value.0 >= self.modulus_times_three {
            value.0 -= self.modulus_times_three;
        }
    }
}

impl CooleyTukeyRadix3Butterfly<Zn64BBase> for Zn64BBase {
    
    fn butterfly<H: Homomorphism<Zn64BBase, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element, 
        c: &mut Self::Element, 
        z: &Zn64BEl,
        t: &Zn64BEl,
        t_sqr_z_sqr: &Zn64BEl
    ) {
        let self_ = hom.codomain().get_ring();

        hom.mul_assign_ref_map(b, t);
        hom.mul_assign_ref_map(c, t_sqr_z_sqr);
        let b_ = hom.mul_ref_map(b, z);
        let c_ = hom.mul_ref_map(c, z);

        let mut s1 = b.0 + c_.0;
        let mut s2 = b_.0 + c.0;
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);
        s1 = reduce_to_half(s1, self_.modulus_times_three);
        s2 = reduce_to_half(s2, self_.modulus_times_three);
        let mut s3 = s1 + s2;
        s3 = reduce_to_half(s3, self_.modulus_times_three);

        b.0 = a.0 + s2;
        c.0 = a.0 + self_.modulus_times_three - s3;
        a.0 = a.0 + s1;
    }
    
    fn inv_butterfly<H: Homomorphism<Zn64BBase, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element,
        c: &mut Self::Element,
        z: &Zn64BEl,
        t: &Zn64BEl,
        t_sqr: &Zn64BEl
    ) {
        let self_ = hom.codomain().get_ring();

        // we assume that a, b, c are already at most `self.modulus_times_three`

        let b_ = hom.mul_ref_map(b, z);
        let mut s1 = b.0 + c.0;
        s1 = reduce_to_half(s1, self_.modulus_times_three);

        let s2 = b_.0 + c.0;
        let s2_ = hom.mul_ref_snd_map(self_.from_u64_promise_reduced(s2), z);

        let mut s3 = s1 + s2_.0;
        s3 = reduce_to_half(s3, self_.modulus_times_three);

        b.0 = a.0 + s2_.0;
        c.0 = a.0 + self_.modulus_times_three - s3;
        a.0 = a.0 + s1;
        a.0 = reduce_to_half(a.0, self_.modulus_times_three);

        hom.mul_assign_ref_map(b, t);
        hom.mul_assign_ref_map(c, t_sqr);
    }

    #[inline(always)]
    fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    #[inline(always)]
    fn prepare_for_inv_fft(&self, value: &mut Self::Element) {
        if value.0 >= self.modulus_times_three {
            value.0 -= self.modulus_times_three
        }
    }
}

impl<I: ?Sized + IntegerRing> CanHomFrom<I> for ZnFastmulBase 
    where Zn64BBase: CanHomFrom<I>
{
    type Homomorphism = <Zn64BBase as CanHomFrom<I>>::Homomorphism;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.get_ring().map_in(from, el, hom))
    }
}

impl_field_wrap_unwrap_homs!{ Zn64BBase, Zn64BBase }
impl_field_wrap_unwrap_isos!{ Zn64BBase, Zn64BBase }
impl_localpir_wrap_unwrap_homs!{ Zn64BBase, Zn64BBase }
impl_localpir_wrap_unwrap_isos!{ Zn64BBase, Zn64BBase }

impl<I> CanHomFrom<zn_big::ZnGBBase<I>> for AsFieldBase<Zn64B>
    where I: RingStore,
        I::Type: IntegerRing
{
    type Homomorphism = <Zn64BBase as CanHomFrom<zn_big::ZnGBBase<I>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from)
    }

    fn map_in(&self, from: &zn_big::ZnGBBase<I>, el: <zn_big::ZnGBBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.get_delegate().map_in(from, el, hom))
    }
}

impl<I> CanIsoFromTo<zn_big::ZnGBBase<I>> for AsFieldBase<Zn64B>
    where I: RingStore,
        I::Type: IntegerRing
{
    type Isomorphism = <Zn64BBase as CanIsoFromTo<zn_big::ZnGBBase<I>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Isomorphism> {
        self.get_delegate().has_canonical_iso(from)
    }

    fn map_out(&self, from: &zn_big::ZnGBBase<I>, el: Self::Element, hom: &Self::Isomorphism) -> <zn_big::ZnGBBase<I> as RingBase>::Element {
        self.get_delegate().map_out(from, self.delegate(el), hom)
    }
}

#[cfg(test)]
use test::Bencher;

#[cfg(test)]
fn elements<'a>(ring: &'a Zn64B) -> impl 'a + Iterator<Item = El<Zn64B>> {
    (0..63).map(|i| ring.coerce(&StaticRing::<i64>::RING, 1 << i))
}

#[cfg(test)]
const LARGE_MODULI: [u64; 6] = [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1];

#[test]
fn test_complete_reduce() {
    let ring = Zn64B::new(32);
    assert_eq!(31, ring.get_ring().complete_reduce(4 * 32 - 1));
}

#[test]
fn test_sum() {
    for n in LARGE_MODULI {
        let Zn = Zn64B::new(n);
        assert_el_eq!(Zn, Zn.int_hom().map(10001 * 5000), Zn.sum((0..=10000).map(|x| Zn.int_hom().map(x))));
    }
}

#[test]
fn test_ring_axioms() {
    for n in 2..=17 {
        let ring = Zn64B::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, (0..=ring.get_ring().repr_bound()).map(|n| Zn64BEl(n)));
    }
    for n in LARGE_MODULI {
        let ring = Zn64B::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, elements(&ring));
    }
}

#[test]
fn test_hash_axioms() {
    for n in 2..=17 {
        let ring = Zn64B::new(n);
        crate::ring::generic_tests::test_hash_axioms(&ring, (0..=ring.get_ring().repr_bound()).map(|n| Zn64BEl(n)));
    }
    for n in LARGE_MODULI {
        let ring = Zn64B::new(n);
        crate::ring::generic_tests::test_hash_axioms(&ring, elements(&ring));
    }
}

#[test]
fn test_divisibility_axioms() {
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, Zn.elements());
    }
    for n in LARGE_MODULI {
        let Zn = Zn64B::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, elements(&Zn));
    }
}

#[test]
fn test_zn_axioms() {
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        super::generic_tests::test_zn_axioms(&Zn);
    }
}

#[test]
fn test_principal_ideal_ring_axioms() {
    for n in 2..=17 {
        let R = Zn64B::new(n);
        crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
    }
    let R = Zn64B::new(63);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
}

#[test]
fn test_hom_from_fastmul() {
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        let Zn_fastmul = ZnFastmul::new(Zn).unwrap();
        crate::ring::generic_tests::test_hom_axioms(Zn_fastmul, Zn, Zn.elements().map(|x| Zn_fastmul.coerce(&Zn, x)));
    }
    for n in [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1] {
        let Zn = Zn64B::new(n);
        let Zn_fastmul = ZnFastmul::new(Zn).unwrap();
        crate::ring::generic_tests::test_hom_axioms(Zn_fastmul, Zn, elements(&Zn).map(|x| Zn_fastmul.coerce(&Zn, x)));
    }
}

#[test]
fn test_finite_ring_axioms() {
    for n in 2..=17 {
        crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn64B::new(n));
    }
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn64B::new(128));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn64B::new(1 << 32));
}

#[test]
fn test_from_int_hom() {
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i8>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i16>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i32>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, Zn, -8..8);
    }
    let Zn = Zn64B::new(5);
    assert_el_eq!(Zn, Zn.int_hom().map(3), Zn.can_hom(&StaticRing::<i64>::RING).unwrap().map(-1596802));
}

#[test]
fn test_bounded_reduce_large() {
    const FACTOR: usize = 32;
    let n_max = (1 << 62) / 9;
    for n in (n_max - 10)..=n_max {
        let Zn = Zn64B::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128 * FACTOR as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(Zn64BEl(Zn.get_ring().bounded_reduce_larger::<FACTOR>(k))));
        }
    }
}

#[test]
fn test_smallest_lift() {
    for n in 2..=17 {
        let ring = Zn64B::new(n);
        for k in 0..=ring.get_ring().repr_bound() {
            let expected = if (k % n) <= n / 2 { (k % n) as i64 } else { (k % n) as i64 - n as i64 };
            if n % 2 == 0 && (k % n) == n / 2 {
                assert!(ring.smallest_lift(Zn64BEl(k)) == n as i64 / 2 || ring.smallest_lift(Zn64BEl(k)) == -(n as i64) / 2)
            } else {
                assert_eq!(expected, ring.smallest_lift(Zn64BEl(k)));
            }
        }
    }
}

#[test]
fn test_smallest_positive_lift() {
    for n in 2..=17 {
        let ring = Zn64B::new(n);
        for k in 0..=ring.get_ring().repr_bound() {
            let expected = (k % n) as i64;
            assert_eq!(expected, ring.smallest_positive_lift(Zn64BEl(k)));
        }
    }
}

#[test]
fn test_bounded_reduce_small() {
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(Zn64BEl(Zn.get_ring().bounded_reduce(k))));
        }
    }
}

#[test]
fn test_bounded_reduce_large_small() {
    const FACTOR: usize = 32;
    for n in 2..=17 {
        let Zn = Zn64B::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128 * FACTOR as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(Zn64BEl(Zn.get_ring().bounded_reduce_larger::<FACTOR>(k))));
        }
    }
}

#[test]
fn test_bounded_reduce() {
    let n_max = (1 << 62) / 9;
    for n in (n_max - 10)..=n_max {
        let Zn = Zn64B::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(Zn64BEl(Zn.get_ring().bounded_reduce(k))));
        }
    }
}

#[bench]
fn bench_hom_from_i64_large_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn64B::new(36028797018963971 /* = 2^55 + 3 */);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(Zn, Zn.int_hom().map(-1300), Zn.sum((0..100).flat_map(|_| (0..=56).map(|k| 1 << k)).map(|x| hom.map(x))))
    });
}

#[bench]
fn bench_hom_from_i64_small_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn64B::new(17);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(Zn, Zn.int_hom().map(2850 * 5699), Zn.sum((0..5700).map(|x| hom.map(x))))
    });
}

#[bench]
fn bench_reduction_map_use_case(bencher: &mut Bencher) {
    // this benchmark is inspired by the use in https://eprint.iacr.org/2023/1510.pdf
    let p = 17;
    let Zp2 = Zn64B::new(p * p);
    let Zp = Zn64B::new(p);
    let Zp2_mod_p = ZnReductionMap::new(&Zp2, &Zp).unwrap();
    let Zp2_p = PreparedDivisor::new(Zp2.get_ring(), Zp2.int_hom().map(p as i32));

    let split_quo_rem = |x: El<Zn64B>| {
        let rem = Zp2_mod_p.map_ref(&x);
        let Zp2_rem = Zp2_mod_p.smallest_lift(rem);
        let quo = Zp2_p.checked_left_div_by(&Zp2.sub(x, Zp2_rem), Zp2.get_ring()).unwrap();
        (rem, Zp2_mod_p.map(quo))
    };

    bencher.iter(|| {
        for x in Zp2.elements() {
            for y in Zp2.elements() {
                let (x_low, x_high) = split_quo_rem(x);
                let (y_low, y_high) = split_quo_rem(y);
                assert_el_eq!(Zp2, Zp2.mul(x, y), &Zp2.add(Zp2.mul(Zp2_mod_p.smallest_lift(x_low), Zp2_mod_p.smallest_lift(y_low)), Zp2_mod_p.mul_quotient_fraction(Zp.add(Zp.mul(x_low, y_high), Zp.mul(x_high, y_low)))));
            }
        }
    });
}

#[bench]
fn bench_inner_product(bencher: &mut Bencher) {
    let Fp = Zn64B::new(65537);
    let len = 1 << 12;
    let lhs = (0..len).map(|i| Fp.int_hom().map(i)).collect::<Vec<_>>();
    let rhs = (0..len).map(|i| Fp.int_hom().map(i)).collect::<Vec<_>>();
    let expected = (0..len).map(|i| Fp.int_hom().map(i * i)).fold(Fp.zero(), |l, r| Fp.add(l, r));

    bencher.iter(|| {
        let actual = <_ as ComputeInnerProduct>::inner_product_ref(Fp.get_ring(), lhs.iter().zip(rhs.iter()).map(|x| std::hint::black_box(x)));
        assert_el_eq!(Fp, expected, actual);
    })
}

#[test]
fn test_serialize() {
    let ring = Zn64B::new(128);
    crate::serialization::generic_tests::test_serialization(ring, ring.elements())
}