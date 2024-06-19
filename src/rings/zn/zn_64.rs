use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyButterfly;
use crate::delegate::DelegateRing;
use crate::divisibility::*;
use crate::impl_eq_based_self_iso;
use crate::impl_wrap_unwrap_homs;
use crate::impl_wrap_unwrap_isos;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use algorithms::convolution::KaratsubaHint;
use algorithms::matmul::ComputeInnerProduct;
use algorithms::matmul::StrassenHint;
use crate::homomorphism::*;
use crate::rings::rust_bigint::*;

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
/// Represents the ring `Z/nZ`.
/// A variant of Barett reduction is used to perform fast modular
/// arithmetic for `n` slightly smaller than 64 bit.
/// 
/// More concretely, the currently maximal supported modulus is `floor(2^62 / 9)`.
/// Note that the exact value might change in the future.
/// 
/// Standard arithmetic in this ring is about the same as in [`super::zn_42::ZnBase`],
/// which supports moduli up to 41 bits. However, this ring is perfectly suited for the
/// number theoretic transform together with [`ZnFastmulBase`], where it is possibly even
/// slightly faster than the 42-bit ring.
/// 
/// # Examples
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// let zn = Zn::new(7);
/// assert_el_eq!(&zn, &zn.one(), &zn.mul(zn.int_hom().map(3), zn.int_hom().map(5)));
/// ```
/// Too large moduli will give an error.
/// ```should_panic
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// Zn::new((1 << 62) / 9 + 1);
/// ```
/// 
#[derive(Clone, Copy)]
pub struct ZnBase {
    modulus: i64,
    modulus_half: i64,
    modulus_times_three: u64,
    inv_modulus: u128
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
#[allow(non_upper_case_globals)]
const ZZbig: BigIntRing = BigIntRing::RING;

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus > 1);
        // this should imply the statement we need later
        assert!(modulus <= ((1 << 62) / 9));
        // we want representatives to grow up to 6 * modulus
        assert!(modulus as u128 * 6 <= u64::MAX as u128);
        let inv_modulus = ZZbig.euclidean_div(ZZbig.power_of_two(128), &ZZbig.coerce(&ZZ, modulus as i64));
        // we need the product `inv_modulus * (6 * modulus)^2` to fit into 192 bit, should be implied by `modulus < ((1 << 62) / 9)`
        debug_assert!(ZZbig.is_lt(&ZZbig.mul_ref_fst(&inv_modulus, ZZbig.pow(ZZbig.int_hom().mul_map(ZZbig.coerce(&ZZ, modulus as i64), 6), 2)), &ZZbig.power_of_two(192)));
        let inv_modulus = if ZZbig.eq_el(&inv_modulus, &ZZbig.power_of_two(127)) {
            1u128 << 127
        } else {
            int_cast(inv_modulus, &StaticRing::<i128>::RING, &ZZbig) as u128
        };
        Self {
            modulus: modulus as i64,
            inv_modulus: inv_modulus,
            modulus_half: (modulus as i64 - 1) / 2 + 1,
            modulus_times_three: modulus * 3
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
        self.modulus_times_three * 2
    }

    ///
    /// If input is bounded by `self.repr_bound() * self.repr_bound()`, then the output
    /// is `< 3 * modulus` and congruent to the input.
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
    /// If input is bounded by `FACTOR * self.repr_bound() * self.repr_bound()`, then the 
    /// output that is `< 3 * modulus` and congruent to the input.
    /// 
    /// As opposed to the faster [`ZnBase::bounded_reduce()`], this should work for all
    /// inputs in `u128`. Currently only used by the specialization of
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

    fn potential_reduce(&self, mut value: u64) -> u64 {
        if value >= self.repr_bound() {
            value -= self.repr_bound();
        }
        if value >= self.modulus_times_three {
            value -= self.modulus_times_three;
        }
        return value;
    }

    fn from_u64_promise_reduced(&self, value: u64) -> ZnEl {
        debug_assert!(value <= self.repr_bound());
        ZnEl(value)
    }

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
}

/// 
/// Represents the ring `Z/nZ`, using a variant of Barett reduction
/// for moduli somewhat smaller than 64 bits. For details, see [`ZnBase`].
/// 
pub type Zn = RingValue<ZnBase>;

impl Zn {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

#[derive(Clone, Copy)]
pub struct ZnEl(u64);

impl PartialEq for ZnBase {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.potential_reduce(lhs.0 + rhs.0);
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
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
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

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", self.complete_reduce(value.0))
    }
    
    fn characteristic<I: IntegerRingStore>(&self, other_ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.size(other_ZZ)
    }
    
    fn sum<I>(&self, mut els: I) -> Self::Element 
        where I: Iterator<Item = Self::Element>
    {
        let mut result = self.zero();
        while let Some(ZnEl(start)) = els.next() {
            let mut current = start as u128;
            for ZnEl(c) in els.by_ref().take(self.repr_bound() as usize / 2 - 1) {
                current += c as u128;
            }
            self.add_assign(&mut result, self.from_u64_promise_reduced(self.bounded_reduce(current)));
        }
        debug_assert!(result.0 <= self.repr_bound());
        return result;
    }

}

impl ComputeInnerProduct for ZnBase {

    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, mut els: I) -> Self::Element
    {
        fn body<I: Iterator<Item = (ZnEl, ZnEl)>>(ring: &ZnBase, els: &mut I) -> Option<ZnEl> {
            debug_assert!(u128::MAX / (ring.repr_bound() as u128 * ring.repr_bound() as u128) >= 36);
            const REDUCE_AFTER_STEPS: usize = 32;
            
            let mut current = 0;
            for i in 0..REDUCE_AFTER_STEPS {
                let next_pair = els.next();
                if let Some((l, r)) = next_pair {
                    debug_assert!(l.0 <= ring.repr_bound());
                    debug_assert!(r.0 <= ring.repr_bound());
                    current += l.0 as u128 * r.0 as u128;
                    debug_assert!(current <= (i + 1) as u128 * ring.repr_bound() as u128 * ring.repr_bound() as u128);
                } else if i == 0 {
                    return None;
                } else {
                    break;
                }
            }
            return Some(ZnEl(ring.bounded_reduce_larger::<REDUCE_AFTER_STEPS>(current)));
        }

        self.sum((0..).map(|_| body(self, &mut els)).take_while(|x| x.is_some()).map(|x| x.unwrap()))
    }
}

impl_eq_based_self_iso!{ ZnBase }

impl<I: IntegerRingStore> CanHomFrom<zn_big::ZnBase<I>> for ZnBase
    where I::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_big::ZnBase<I>) -> Option<Self::Homomorphism> {
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

    fn map_in(&self, from: &zn_big::ZnBase<I>, el: <zn_big::ZnBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        self.from_u64_promise_reduced(int_cast(from.smallest_positive_lift(el), self.integer_ring(), from.integer_ring()) as u64)
    }
}

impl<I: IntegerRingStore> CanIsoFromTo<zn_big::ZnBase<I>> for ZnBase
    where I::Type: IntegerRing
{
    type Isomorphism = <zn_big::ZnBase<I> as CanHomFrom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_big::ZnBase<I>) -> Option<Self::Isomorphism> {
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

    fn map_out(&self, from: &zn_big::ZnBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_big::ZnBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0 as i64, iso)
    }
}

///
/// An element of [`ZnBase`] together with extra information that allows for
/// faster division if this element is the divisor. See also [`PreparedDivisibilityRing`].
/// 
#[derive(Copy, Clone)]
pub struct ZnPreparedDivisor {
    unit_part: El<Zn>,
    is_unit: bool,
    smallest_positive_zero_divisor_part: <StaticRingBase<i64> as PreparedDivisibilityRing>::PreparedDivisor
}

impl PreparedDivisibilityRing for ZnBase {

    type PreparedDivisor = ZnPreparedDivisor;

    fn prepare_divisor(&self, x: &Self::Element) -> Self::PreparedDivisor {
        let (s, _t, d) = algorithms::eea::signed_eea(self.smallest_positive_lift(*x), *self.modulus(), self.integer_ring());
        debug_assert!(d > 0);
        debug_assert!(d <= *self.modulus());
        return ZnPreparedDivisor {
            is_unit: d == 1,
            unit_part: if s < 0 { self.negate(self.from_u64_promise_reduced(-s as u64)) } else { self.from_u64_promise_reduced(s as u64) },
            smallest_positive_zero_divisor_part: StaticRing::<i64>::RING.prepare_divisor(&d)
        }
    }

    fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &Self::PreparedDivisor) -> Option<Self::Element> {
        if rhs.is_unit {
            Some(self.mul_ref(lhs, &rhs.unit_part))
        } else {
            StaticRing::<i64>::RING.checked_div_prepared(&self.smallest_positive_lift(*lhs), &rhs.smallest_positive_zero_divisor_part)
                .map(|x| self.mul(self.from_u64_promise_reduced(x as u64), rhs.unit_part))
        }
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div_prepared(lhs, &self.prepare_divisor(rhs))
    }
}
trait ImplGenericIntHomomorphismMarker: IntegerRing + CanIsoFromTo<StaticRingBase<i128>> + CanIsoFromTo<StaticRingBase<i64>> {}

impl ImplGenericIntHomomorphismMarker for RustBigintRingBase {}

#[cfg(feature = "mpir")]
impl ImplGenericIntHomomorphismMarker for crate::rings::mpir::MPZBase {}

impl<I: ?Sized + ImplGenericIntHomomorphismMarker> CanHomFrom<I> for ZnBase {

    type Homomorphism = super::generic_impls::BigIntToZnHom<I, StaticRingBase<i128>, Self>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_bigint(from, self, StaticRing::<i128>::RING.get_ring(), Some(&(self.repr_bound() as i128 * self.repr_bound() as i128)))
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in_from_bigint(from, self, StaticRing::<i128>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as u64) < self.modulus_u64());
            self.from_u64_promise_reduced(n as u64)
        }, |n| {
            debug_assert!(n <= (self.repr_bound() as i128 * self.repr_bound() as i128));
            self.from_u64_promise_reduced(self.bounded_reduce(n as u128))
        })
    }
}

macro_rules! impl_static_int_to_zn {
    ($($int:ident),*) => {
        $(
            impl CanHomFrom<StaticRingBase<$int>> for ZnBase {
            
                type Homomorphism = (/* bound for direct reduction */ $int, /* bound for bounded reduction */ $int);
            
                fn has_canonical_hom(&self, _from: &StaticRingBase<$int>) -> Option<Self::Homomorphism> {
                    let bounded_reduce_bound = self.repr_bound() as i128 * self.repr_bound() as i128;
                    Some((self.repr_bound() as $int, if bounded_reduce_bound > $int::MAX as i128 { $int::MAX } else { bounded_reduce_bound as $int }))
                }
            
                fn map_in(&self, _from: &StaticRingBase<$int>, el: $int, hom: &($int, $int)) -> Self::Element {
                    if el.abs() <= hom.0 {
                        if el < 0 {
                            self.negate(self.from_u64_promise_reduced(el.unsigned_abs() as u64))
                        } else {
                            self.from_u64_promise_reduced(el as u64)
                        }
                    } else if el.abs() <= hom.1 {
                        if el < 0 {
                            self.negate(self.from_u64_promise_reduced(self.bounded_reduce(el.unsigned_abs() as u128)))
                        } else {
                            self.from_u64_promise_reduced(self.bounded_reduce(el as u128))
                        }
                    } else {
                        if el < 0 {
                            self.from_u64_promise_reduced(((el as i128 % self.modulus as i128) as i64 + self.modulus) as u64)
                        } else {
                            self.from_u64_promise_reduced((el as i128 % self.modulus as i128) as u64)
                        }
                    }
                }
            }
        )*
    };
}

impl_static_int_to_zn!{ i8, i16, i32, i64, i128 }

#[derive(Clone, Copy)]
pub struct ZnBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

    type Item = ZnEl;

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

impl FiniteRing for ZnBase {

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

    fn size<I: IntegerRingStore>(&self, other_ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        if other_ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(&(self.modulus() + 1)) <= other_ZZ.get_ring().representable_bits() {
            Some(int_cast(*self.modulus(), other_ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl PrincipalIdealRing for ZnBase {

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = StaticRing::<i64>::RING.extended_ideal_gen(&(lhs.0 as i64), &(rhs.0 as i64));
        let quo = RingRef::new(self).into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        (quo.map(s), quo.map(t), quo.map(d))
    }
}

impl StrassenHint for ZnBase {
    default fn strassen_threshold(&self) -> usize {
        6
    }
}

impl KaratsubaHint for ZnBase {
    default fn karatsuba_threshold(&self) -> usize {
        6
    }
}

impl ZnRing for ZnBase {

    type IntegerRingBase = StaticRingBase<i64>;
    type Integers = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.complete_reduce(el.0) as i64
    }

    fn smallest_lift(&self, ZnEl(mut value_u64): Self::Element) -> El<Self::Integers> {
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

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus
    }

    fn any_lift(&self, el: Self::Element) -> El<Self::Integers> {
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
    /// ```
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let ring = Zn::new(7);
    /// assert_el_eq!(&ring, &ring.zero(), &ring.get_ring().from_int_promise_reduced(42));
    /// ```
    /// Larger values lead to a panic in debug mode, and to a logic error in release mode.
    /// ```should_panic
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let ring = Zn::new(7);
    /// ring.get_ring().from_int_promise_reduced(43);
    /// ```
    /// 
    fn from_int_promise_reduced(&self, x: El<Self::Integers>) -> Self::Element {
        debug_assert!(self.repr_bound() == 6 * self.modulus_u64());
        debug_assert!(x >= 0);
        debug_assert!(x as u64 <= self.repr_bound());
        self.from_u64_promise_reduced(x as u64)
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
/// using [`CanHomFrom::mul_assign_map_in()`].
/// Note that normal arithmetic in this ring is much slower than [`ZnBase`].
/// 
/// # Example
/// The following use of the FFT is usually faster than the standard use, as
/// the FFT requires a high amount of multiplications with the internally stored
/// roots of unity.
/// ```
/// # #![feature(const_type_name)]
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// let ring = Zn::new(1073872897);
/// let fastmul_ring = ZnFastmul::new(ring);
/// // The values stored by the FFT table are elements of `ZnFastmulBase`
/// let fft = CooleyTuckeyFFT::for_zn_with_hom(ring.can_hom(&fastmul_ring).unwrap(), 15).unwrap();
/// // Note that data uses `ZnBase`
/// let mut data = (0..(1 << 15)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
/// fft.unordered_fft(&mut data[..], ring.get_ring());
/// ```
/// 
#[derive(Clone, Copy)]
pub struct ZnFastmulBase {
    base: RingValue<ZnBase>
}

///
/// An implementation of `Z/nZ` for `n` that have at most 41 bits that is optimized for multiplication with elements
/// of [`Zn`]. For details, see [`ZnFastmulBase`].
/// 
pub type ZnFastmul = RingValue<ZnFastmulBase>;

impl ZnFastmul {

    pub fn new(base: Zn) -> Self {
        RingValue::from(ZnFastmulBase { base })
    }
}

impl PartialEq for ZnFastmulBase {
    
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl PrincipalIdealRing for ZnFastmulBase {

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self.get_delegate().extended_ideal_gen(self.delegate_ref(lhs), self.delegate_ref(rhs));
        (self.rev_delegate(s), self.rev_delegate(t), self.rev_delegate(d))
    }
}

impl_eq_based_self_iso!{ ZnFastmulBase }

#[derive(Clone, Copy)]
pub struct ZnFastmulEl {
    el: ZnEl,
    value_invmod_shifted: u128
}

impl DelegateRing for ZnFastmulBase {

    type Base = ZnBase;
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
        let value = el.el.0;
        el.value_invmod_shifted = ((value as u128) << 64) / self.base.get_ring().modulus_u64() as u128;
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

impl CanHomFrom<ZnBase> for ZnFastmulBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnBase) -> Option<Self::Homomorphism> {
        if from == self.base.get_ring() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _from: &ZnBase, el: <ZnBase as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(el)
    }
}

impl CanHomFrom<ZnFastmulBase> for ZnBase {

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
        let approx_quotient = mullo(lhs.0, high(rhs.value_invmod_shifted)).wrapping_add(mulhi(lhs.0, low(rhs.value_invmod_shifted)));
        lhs.0 = product.wrapping_sub(mullo(approx_quotient, self.modulus_u64()));
        debug_assert!(lhs.0 < self.modulus_times_three);
        debug_assert!((lhs_original as u128 * rhs.el.0 as u128 - lhs.0 as u128) % (self.modulus_u64() as u128) == 0);
    }

    fn mul_assign_map_in_ref(&self, from: &ZnFastmulBase, lhs: &mut Self::Element, rhs: &<ZnFastmulBase as RingBase>::Element, hom: &Self::Homomorphism) {
        self.mul_assign_map_in(from, lhs, *rhs, hom);
    }
}

impl CanIsoFromTo<ZnFastmulBase> for ZnBase {

    type Isomorphism = <ZnFastmulBase as CanHomFrom<Self>>::Homomorphism;

    fn has_canonical_iso(&self, from: &ZnFastmulBase) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &ZnFastmulBase, el: Self::Element, iso: &Self::Isomorphism) -> <ZnFastmulBase as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

impl CooleyTuckeyButterfly<ZnFastmulBase> for ZnBase {

    #[inline(always)]
    fn butterfly<V: crate::seq::VectorViewMut<Self::Element>, H: Homomorphism<ZnFastmulBase, Self>>(&self, hom: &H, values: &mut V, twiddle: &<ZnFastmulBase as RingBase>::Element, i1: usize, i2: usize) {
        let mut a = *values.at(i1);
        if a.0 >= self.modulus_times_three {
            a.0 -= self.modulus_times_three;
        }
        let mut b = *values.at(i2);
        hom.mul_assign_map_ref(&mut b, twiddle);

        // this is implied by `bounded_reduce`, check anyway
        debug_assert!(a.0 <= self.modulus_times_three);
        debug_assert!(b.0 < self.modulus_times_three);
        debug_assert!(self.repr_bound() >= self.modulus_u64() * 6);

        *values.at_mut(i1) = self.from_u64_promise_reduced(a.0 + b.0);
        *values.at_mut(i2) = self.from_u64_promise_reduced(a.0 + self.modulus_times_three - b.0);
    }

    fn inv_butterfly<V: crate::seq::VectorViewMut<Self::Element>, H: Homomorphism<ZnFastmulBase, Self>>(&self, hom: &H, values: &mut V, twiddle: &<ZnFastmulBase as RingBase>::Element, i1: usize, i2: usize) {
        let a = *values.at(i1);
        let b = *values.at(i2);

        *values.at_mut(i1) = self.add(a, b);
        // this works, as mul_assign_map_in_ref() works with values up to 6 * self.modulus
        *values.at_mut(i2) = self.from_u64_promise_reduced(a.0 + self.modulus_times_three - b.0);
        hom.mul_assign_map_ref(values.at_mut(i2), twiddle);
    }
}

impl<I: ?Sized + IntegerRing> CanHomFrom<I> for ZnFastmulBase 
    where ZnBase: CanHomFrom<I>
{
    type Homomorphism = <ZnBase as CanHomFrom<I>>::Homomorphism;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.get_ring().map_in(from, el, hom))
    }
}

impl_wrap_unwrap_homs!{ ZnBase, ZnBase }
impl_wrap_unwrap_isos!{ ZnBase, ZnBase }

#[cfg(test)]
use test::Bencher;

#[cfg(test)]
fn elements<'a>(ring: &'a Zn) -> impl 'a + Iterator<Item = El<Zn>> {
    (0..63).map(|i| ring.coerce(&ZZ, 1 << i))
}

#[cfg(test)]
const LARGE_MODULI: [u64; 6] = [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1];

#[test]
fn test_complete_reduce() {
    let ring = Zn::new(32);
    assert_eq!(31, ring.get_ring().complete_reduce(4 * 32 - 1));
}

#[test]
fn test_sum() {
    for n in LARGE_MODULI {
        let Zn = Zn::new(n);
        assert_el_eq!(&Zn, &Zn.int_hom().map(10001 * 5000), &Zn.sum((0..=10000).map(|x| Zn.int_hom().map(x))));
    }
}

#[test]
fn test_ring_axioms() {
    for n in 2..=17 {
        let ring = Zn::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, (0..=ring.get_ring().repr_bound()).map(|n| ZnEl(n)));
    }
    for n in LARGE_MODULI {
        let ring = Zn::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, elements(&ring));
    }
}

#[test]
fn test_divisibility_axioms() {
    for n in 2..=17 {
        let Zn = Zn::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, Zn.elements());
    }
    for n in LARGE_MODULI {
        let Zn = Zn::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, elements(&Zn));
    }
}

#[test]
fn test_zn_axioms() {
    for n in 2..=17 {
        let Zn = Zn::new(n);
        super::generic_tests::test_zn_axioms(&Zn);
    }
}

#[test]
fn test_principal_ideal_ring_axioms() {
    for n in 2..=17 {
        let R = Zn::new(n);
        crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
    }
    let R = Zn::new(63);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
}

#[test]
fn test_hom_from_fastmul() {
    for n in 2..=17 {
        let Zn = Zn::new(n);
        let Zn_fastmul = ZnFastmul::new(Zn);
        crate::ring::generic_tests::test_hom_axioms(Zn_fastmul, Zn, Zn.elements().map(|x| Zn_fastmul.coerce(&Zn, x)));
    }
    for n in [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1] {
        let Zn = Zn::new(n);
        let Zn_fastmul = ZnFastmul::new(Zn);
        crate::ring::generic_tests::test_hom_axioms(Zn_fastmul, Zn, elements(&Zn).map(|x| Zn_fastmul.coerce(&Zn, x)));
    }
}

#[test]
fn test_finite_ring_axioms() {
    for n in 2..=17 {
        crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(n));
    }
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(128));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(1 << 32));
}

#[test]
fn test_from_int_hom() {
    for n in 2..=17 {
        let Zn = Zn::new(n);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i8>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i16>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i32>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, Zn, -8..8);
    }
    let Zn = Zn::new(5);
    assert_el_eq!(&Zn, &Zn.int_hom().map(3), &Zn.can_hom(&StaticRing::<i64>::RING).unwrap().map(-1596802));
}

#[test]
fn test_bounded_reduce_large() {
    const FACTOR: usize = 32;
    let n_max = (1 << 62) / 9;
    for n in (n_max - 10)..=n_max {
        let Zn = Zn::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128 * FACTOR as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(ZnEl(Zn.get_ring().bounded_reduce_larger::<FACTOR>(k))));
        }
    }
}

#[test]
fn test_smallest_lift() {
    for n in 2..=17 {
        let ring = Zn::new(n);
        for k in 0..=ring.get_ring().repr_bound() {
            let expected = if (k % n) <= n / 2 { (k % n) as i64 } else { (k % n) as i64 - n as i64 };
            if n % 2 == 0 && (k % n) == n / 2 {
                assert!(ring.smallest_lift(ZnEl(k)) == n as i64 / 2 || ring.smallest_lift(ZnEl(k)) == -(n as i64) / 2)
            } else {
                assert_eq!(expected, ring.smallest_lift(ZnEl(k)));
            }
        }
    }
}

#[test]
fn test_smallest_positive_lift() {
    for n in 2..=17 {
        let ring = Zn::new(n);
        for k in 0..=ring.get_ring().repr_bound() {
            let expected = (k % n) as i64;
            assert_eq!(expected, ring.smallest_positive_lift(ZnEl(k)));
        }
    }
}

#[test]
fn test_bounded_reduce_small() {
    for n in 2..=17 {
        let Zn = Zn::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(ZnEl(Zn.get_ring().bounded_reduce(k))));
        }
    }
}

#[test]
fn test_bounded_reduce_large_small() {
    const FACTOR: usize = 32;
    for n in 2..=17 {
        let Zn = Zn::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128 * FACTOR as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(ZnEl(Zn.get_ring().bounded_reduce_larger::<FACTOR>(k))));
        }
    }
}

#[test]
fn test_bounded_reduce() {
    let n_max = (1 << 62) / 9;
    for n in (n_max - 10)..=n_max {
        let Zn = Zn::new(n);
        let val_max = Zn.get_ring().repr_bound() as u128 * Zn.get_ring().repr_bound() as u128;
        for k in (val_max - 100)..=val_max {
            assert_eq!((k % (n as u128)) as i64, Zn.smallest_positive_lift(ZnEl(Zn.get_ring().bounded_reduce(k))));
        }
    }
}

#[bench]
fn bench_hom_from_i64_large_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn::new(36028797018963971 /* = 2^55 + 3 */);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(&Zn, &Zn.int_hom().map(-1300), &Zn.sum((0..100).flat_map(|_| (0..=56).map(|k| 1 << k)).map(|x| hom.map(x))))
    });
}

#[bench]
fn bench_hom_from_i64_small_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn::new(17);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(&Zn, &Zn.int_hom().map(2850 * 5699), &Zn.sum((0..5700).map(|x| hom.map(x))))
    });
}

#[bench]
fn bench_reduction_map_use_case(bencher: &mut Bencher) {
    // this benchmark is inspired by the use in https://eprint.iacr.org/2023/1510.pdf
    let p = 17;
    let Zp2 = Zn::new(p * p);
    let Zp = Zn::new(p);
    let Zp2_mod_p = ReductionMap::new(&Zp2, &Zp).unwrap();
    let Zp2_p = Zp2.prepare_divisor(&Zp2.int_hom().map(p as i32));

    let split_quo_rem = |x: El<Zn>| {
        let rem = Zp2_mod_p.map_ref(&x);
        let Zp2_rem = Zp2_mod_p.smallest_lift(rem);
        let quo = Zp2.checked_div_prepared(&Zp2.sub(x, Zp2_rem), &Zp2_p).unwrap();
        (rem, Zp2_mod_p.map(quo))
    };

    bencher.iter(|| {
        for x in Zp2.elements() {
            for y in Zp2.elements() {
                let (x_low, x_high) = split_quo_rem(x);
                let (y_low, y_high) = split_quo_rem(y);
                assert_el_eq!(&Zp2, &Zp2.mul(x, y), &Zp2.add(Zp2.mul(Zp2_mod_p.smallest_lift(x_low), Zp2_mod_p.smallest_lift(y_low)), Zp2_mod_p.mul_quotient_fraction(Zp.add(Zp.mul(x_low, y_high), Zp.mul(x_high, y_low)))));
            }
        }
    });
}

#[bench]
fn bench_inner_product(bencher: &mut Bencher) {
    let Fp = Zn::new(65537);
    let len = 1 << 12;
    let lhs = (0..len).map(|i| Fp.int_hom().map(i)).collect::<Vec<_>>();
    let rhs = (0..len).map(|i| Fp.int_hom().map(i)).collect::<Vec<_>>();
    let expected = (0..len).map(|i| Fp.int_hom().map(i * i)).fold(Fp.zero(), |l, r| Fp.add(l, r));

    bencher.iter(|| {
        let actual = <_ as ComputeInnerProduct>::inner_product_ref(Fp.get_ring(), lhs.iter().zip(rhs.iter()).map(|x| std::hint::black_box(x)));
        assert_el_eq!(&Fp, &expected, &actual);
    })
}