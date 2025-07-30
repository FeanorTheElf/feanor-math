use libc;
use serde::de::DeserializeSeed;
use serde::ser::SerializeTuple;
use serde::{Deserializer, Serializer, Serialize, Deserialize};

use crate::{impl_interpolation_base_ring_char_zero, impl_poly_gcd_locally_for_ZZ, impl_eval_poly_locally_for_ZZ};
use crate::algorithms;
use crate::pid::*;
use crate::ordered::*;
use crate::primitive_int::*;
use crate::divisibility::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::specialization::*;
use crate::rings::rust_bigint::*;
use crate::algorithms::bigint::deserialize_bigint_from_bytes;
use crate::serialization::*;

mod mpir_bindings;

///
/// An `mpir` integer.
/// 
pub struct MPZEl {
    integer: mpir_bindings::__mpz_struct
}

impl MPZEl {

    fn new() -> Self {
        unsafe {
            let mut result = mpir_bindings::UNINIT_MPZ;
            mpir_bindings::__gmpz_init(&mut result as mpir_bindings::mpz_ptr);
            return MPZEl { integer: result };
        }
    }

    pub fn assign(&mut self, rhs: &MPZEl) {
        unsafe {
            mpir_bindings::__gmpz_set(&mut self.integer as mpir_bindings::mpz_ptr, &rhs.integer as mpir_bindings::mpz_srcptr);
        }
    }
}

impl Clone for MPZEl {
    fn clone(&self) -> Self {
        MPZ::RING.clone_el(self)
    }
}

///
/// Except for random number generation (which we do in Rust), GMP/MPIR
/// is thread-safe (Section 3.7 of the manual)
/// 
unsafe impl Send for MPZEl {}
///
/// Except for random number generation (which we do in Rust), GMP/MPIR
/// is thread-safe (Section 3.7 of the manual)
/// 
unsafe impl Sync for MPZEl {}

impl Drop for MPZEl {

    fn drop(&mut self) {
        unsafe {
            mpir_bindings::__gmpz_clear(&mut self.integer as mpir_bindings::mpz_ptr)
        }
    }
}

impl std::fmt::Debug for MPZEl {

    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        MPZ::RING.get_ring().dbg(self, f)
    }
}

///
/// Arbitrary-precision integer ring, implemented by binding to the well-known
/// and heavily optimized library mpir (fork of gmp).
/// 
/// TODO: Remove `Copy` at next breaking release.
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MPZBase;

///
/// [`RingStore`] corresponding to [`MPZBase`].
/// 
pub type MPZ = RingValue<MPZBase>;

impl MPZ {
    
    pub const RING: MPZ = RingValue::from(MPZBase);
}

impl MPZBase {

    ///
    /// Interprets the bytes in `input` as little endian bytes and returns the
    /// nonnegative integer represented by them. 
    /// 
    pub fn from_le_bytes(&self, dst: &mut MPZEl, input: &[u8]) {
        unsafe {
            assert_eq!(std::mem::size_of::<mpir_bindings::mpir_ui>(), std::mem::size_of::<u64>());
            if input.len() == 0 {
                mpir_bindings::__gmpz_set_ui(&mut dst.integer as mpir_bindings::mpz_ptr, 0);
                return;
            }
            assert!(input.len() > 0);
            mpir_bindings::__gmpz_import(
                &mut dst.integer as mpir_bindings::mpz_ptr, 
                input.len(), 
                -1i32, // least significant digit first
                1 as libc::size_t,
                -1, // little endian 
                0, 
                input.as_ptr() as *const libc::c_void
            );
        }
    }

    ///
    /// Interprets the `u64` as digits w.r.t. base `2^64` (with least-significant digit first)
    /// and returns the nonnegative integer represented by them. 
    /// 
    pub fn from_base_u64_repr(&self, dst: &mut MPZEl, input: &[u64]) {
        unsafe {
            assert_eq!(std::mem::size_of::<mpir_bindings::mpir_ui>(), std::mem::size_of::<u64>());
            if input.len() == 0 {
                mpir_bindings::__gmpz_set_ui(&mut dst.integer as mpir_bindings::mpz_ptr, 0);
                return;
            }
            assert!(input.len() > 0);
            mpir_bindings::__gmpz_import(
                &mut dst.integer as mpir_bindings::mpz_ptr, 
                input.len(), 
                -1i32, // least significant digit first
                (u64::BITS / 8) as libc::size_t,
                0, // native endianess 
                0, 
                input.as_ptr() as *const libc::c_void
            );
        }
    }

    pub fn abs_base_u64_repr_len(&self, src: &MPZEl) -> usize {
        self.abs_highest_set_bit(src).map(|n| (n / u64::BITS as usize) + 1).unwrap_or(0)
    }

    ///
    /// Inverse to [`MPZBase::from_base_u64_repr()`].
    /// 
    pub fn abs_base_u64_repr(&self, src: &MPZEl, out: &mut [u64]) {
        unsafe {
            if self.abs_base_u64_repr_len(src) > 0 {
                assert!(out.len() >= self.abs_base_u64_repr_len(src));
                let mut size = 0;
        
                let result_ptr = mpir_bindings::__gmpz_export(
                    out.as_mut_ptr() as *mut libc::c_void,
                    &mut size,
                    -1, // least significant digit first
                    (u64::BITS / 8) as libc::size_t,
                    0, // native endianess 
                    0,
                    &src.integer as mpir_bindings::mpz_srcptr
                );
                assert_eq!(result_ptr as *const libc::c_void, out.as_ptr() as *const libc::c_void);
                for i in size..out.len() {
                    out[i] = 0;
                }
            } else {
                for i in 0..out.len() {
                    out[i] = 0;
                }
            }
        }
    }

    pub fn to_le_bytes_len(&self, src: &MPZEl) -> usize {
        self.abs_highest_set_bit(src).map(|n| (n / u8::BITS as usize) + 1).unwrap_or(0)
    }

    ///
    /// Inverse to [`MPZBase::from_le_bytes()`].
    /// 
    pub fn to_le_bytes(&self, src: &MPZEl, out: &mut [u8]) {
        unsafe {
            if self.to_le_bytes_len(src) > 0 {
                assert!(out.len() >= self.to_le_bytes_len(src));
                let mut size = 0;
        
                let result_ptr = mpir_bindings::__gmpz_export(
                    out.as_mut_ptr() as *mut libc::c_void,
                    &mut size,
                    -1, // least significant digit first
                    1 as libc::size_t,
                    -1, // little endian
                    0,
                    &src.integer as mpir_bindings::mpz_srcptr
                );
                assert_eq!(result_ptr as *const libc::c_void, out.as_ptr() as *const libc::c_void);
                for i in size..out.len() {
                    out[i] = 0;
                }
            } else {
                for i in 0..out.len() {
                    out[i] = 0;
                }
            }
        }
    }
}

impl RingBase for MPZBase {
    
    type Element = MPZEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new();
            mpir_bindings::__gmpz_set(&mut result.integer as mpir_bindings::mpz_ptr, &val.integer as mpir_bindings::mpz_srcptr);
            return result;
        }
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new();
            mpir_bindings::__gmpz_mul(&mut result.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr);
            return result;
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        unsafe {
            mpir_bindings::__gmpz_neg(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr)
        }
    }

    fn zero(&self) -> Self::Element {
        MPZEl::new()
    }

    fn one(&self) -> Self::Element {
        self.from_int(1)
    }

    fn sub_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        unsafe {
            mpir_bindings::__gmpz_sub(&mut rhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr);
            return rhs;
        }
    }

    fn sub_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        unsafe {
            mpir_bindings::__gmpz_sub(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr);
            return lhs;
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { 
        self.add_assign_ref(lhs, &rhs);
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { 
        unsafe {
            mpir_bindings::__gmpz_add(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs)
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { 
        unsafe {
            mpir_bindings::__gmpz_mul(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr);
        }
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        unsafe {
            mpir_bindings::__gmpz_mul_ui(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr, rhs.abs() as u64);
            if rhs < 0 {
                mpir_bindings::__gmpz_neg(&mut lhs.integer as mpir_bindings::mpz_ptr, &lhs.integer as mpir_bindings::mpz_srcptr)
            }
        }
    }

    fn sub(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.sub_ref_snd(lhs, &rhs)
    }

    fn from_int(&self, value: i32) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new();
            mpir_bindings::__gmpz_set_ui(&mut result.integer as mpir_bindings::mpz_ptr, value.abs() as u64);
            if value < 0 {
                return self.negate(result);
            } else {
                return result;
            }
        }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        unsafe {
            mpir_bindings::__gmpz_cmp(&lhs.integer as mpir_bindings::mpz_srcptr, &rhs.integer as mpir_bindings::mpz_srcptr) == 0
        }
    }

    fn is_zero(&self, val: &Self::Element) -> bool { 
        unsafe {
            mpir_bindings::__gmpz_cmp_si(&val.integer as mpir_bindings::mpz_srcptr, 0) == 0
        }
    }
    
    fn is_one(&self, val: &Self::Element) -> bool { 
        unsafe {
            mpir_bindings::__gmpz_cmp_si(&val.integer as mpir_bindings::mpz_srcptr, 1) == 0
        }
    }

    fn is_neg_one(&self, val: &Self::Element) -> bool {
        unsafe {
            mpir_bindings::__gmpz_cmp_si(&val.integer as mpir_bindings::mpz_srcptr, -1) == 0
        }
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn is_approximate(&self) -> bool {
        false
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        RustBigintRing::RING.get_ring().dbg(&self.map_out(RustBigintRing::RING.get_ring(), self.clone_el(value), &()), out)
    }
    
    fn characteristic<I: IntegerRingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(other_ZZ.zero())
    }
}

impl OrderedRing for MPZBase {

    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        unsafe {
            let res = mpir_bindings::__gmpz_cmp(
                &lhs.integer as mpir_bindings::mpz_srcptr,
                &rhs.integer as mpir_bindings::mpz_srcptr
            );
            if res < 0 {
                return std::cmp::Ordering::Less;
            } else if res > 0 {
                return std::cmp::Ordering::Greater;
            } else {
                return std::cmp::Ordering::Equal;
            }
        }
    }

    fn is_neg(&self, value: &Self::Element) -> bool {
        unsafe {
            mpir_bindings::mpz_is_neg(&value.integer as mpir_bindings::mpz_srcptr)
        }
    }
}

impl Serialize for MPZBase {

    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        serializer.serialize_unit_struct("IntegerRing(MPZ)")
    }
}

impl<'de> Deserialize<'de> for MPZBase {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedUnitStruct::new("IntegerRing(MPZ)", MPZ::RING.into()).deserialize(deserializer)
    }
}

impl SerializableElementRing for MPZBase {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        if deserializer.is_human_readable() {
            return DeserializeWithRing::new(RustBigintRing::RING).deserialize(deserializer).map(|n| int_cast(n, RingRef::new(self), RustBigintRing::RING));
        } else {
            let (negative, mut result) = deserialize_bigint_from_bytes(deserializer, |data| {
                let mut result = self.zero();
                self.from_le_bytes(&mut result, data);
                return result;
            })?;
            if negative {
                self.negate_inplace(&mut result);
            }
            return Ok(result);
        }
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        if serializer.is_human_readable() {
            SerializableNewtype::new("BigInt", format!("{}", RingRef::new(self).format(el)).as_str()).serialize(serializer)
        } else {
            let len = self.to_le_bytes_len(el);
            let mut data = Vec::with_capacity(len);
            data.resize(len, 0u8);
            self.to_le_bytes(el, &mut data);
            let mut seq = serializer.serialize_tuple(2)?;
            seq.serialize_element(&self.is_neg(el))?;
            seq.serialize_element(serde_bytes::Bytes::new(&data[..]))?;
            return seq.end();
        }
    }
}

impl DivisibilityRing for MPZBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(rhs) {
            if self.is_zero(lhs) {
                return Some(self.zero());
            } else {
                return None;
            }
        }
        let (quo, rem) = self.euclidean_div_rem(self.clone_el(lhs), rhs);
        if self.is_zero(&rem) {
            return Some(quo);
        } else {
            return None;
        }
    }

    fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        Some(elements.fold(self.zero(), |a, b| self.ideal_gen(&a, b)))
    }
}

impl PrincipalIdealRing for MPZBase {
    
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        algorithms::eea::eea(self.clone_el(lhs), self.clone_el(rhs), MPZ::RING)
    }

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }
}

impl EuclideanRing for MPZBase {
    
    fn euclidean_div_rem(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        unsafe {
            assert!(!self.is_zero(rhs));
            let mut quo = MPZEl::new();
            mpir_bindings::__gmpz_tdiv_qr(
                &mut quo.integer as mpir_bindings::mpz_ptr, 
                &mut lhs.integer as mpir_bindings::mpz_ptr, 
                &lhs.integer as mpir_bindings::mpz_srcptr, 
                &rhs.integer as mpir_bindings::mpz_srcptr
            );
            return (quo, lhs);
        }
    }

    fn euclidean_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element { 
        unsafe {
            assert!(!self.is_zero(rhs));
            let mut rem = MPZEl::new();
            mpir_bindings::__gmpz_tdiv_r(
                &mut rem.integer as mpir_bindings::mpz_ptr, 
                &lhs.integer as mpir_bindings::mpz_srcptr, 
                &rhs.integer as mpir_bindings::mpz_srcptr
            );
            return rem;
        }
    }

    fn euclidean_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        unsafe {
            assert!(!self.is_zero(rhs));
            let mut rem = MPZEl::new();
            mpir_bindings::__gmpz_tdiv_q(
                &mut rem.integer as mpir_bindings::mpz_ptr, 
                &lhs.integer as mpir_bindings::mpz_srcptr, 
                &rhs.integer as mpir_bindings::mpz_srcptr
            );
            return rem;
        }
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.abs_highest_set_bit(val).unwrap_or(0) < usize::BITS as usize {
            unsafe {
                mpir_bindings::__gmpz_get_ui(&val.integer as mpir_bindings::mpz_srcptr).try_into().ok()
            }
        } else {
            None
        }
    }
}

impl Default for MPZBase {
    
    fn default() -> Self {
        MPZ::RING.into()
    }
}

impl Domain for MPZBase {}

impl IntegerRing for MPZBase {
    
    fn to_float_approx(&self, el: &Self::Element) -> f64 {
        unsafe {
            mpir_bindings::__gmpz_get_d(&el.integer as mpir_bindings::mpz_srcptr)
        }
    }

    fn from_float_approx(&self, el: f64) -> Option<Self::Element> {
        unsafe {
            let mut result = MPZEl::new();
            mpir_bindings::__gmpz_set_d(&mut result.integer as mpir_bindings::mpz_ptr, el);
            return Some(result);
        }
    }

    fn mul_pow_2(&self, el: &mut Self::Element, power: usize) {
        unsafe {
            mpir_bindings::__gmpz_mul_2exp(&mut el.integer as mpir_bindings::mpz_ptr, &el.integer as mpir_bindings::mpz_srcptr, power as u64)
        }
    }

    fn euclidean_div_pow_2(&self, el: &mut Self::Element, power: usize) {
        unsafe {
            mpir_bindings::__gmpz_tdiv_q_2exp(&mut el.integer as mpir_bindings::mpz_ptr, &el.integer as mpir_bindings::mpz_srcptr, power as u64)
        }
    }

    fn abs_is_bit_set(&self, el: &Self::Element, bit: usize) -> bool {
        unsafe {
            if mpir_bindings::mpz_is_neg(&el.integer as mpir_bindings::mpz_srcptr) {
                let value = mpir_bindings::__gmpz_tstbit(&el.integer as mpir_bindings::mpz_srcptr, bit as u64) == 1;
                let least_significant_zero = mpir_bindings::__gmpz_scan1(&el.integer as mpir_bindings::mpz_srcptr, 0);
                if bit <= least_significant_zero as usize {
                    value
                } else {
                    !value
                }
            } else {
                mpir_bindings::__gmpz_tstbit(&el.integer as mpir_bindings::mpz_srcptr, bit as u64) == 1
            }
        }
    }
    
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        unsafe {
            if self.is_zero(value) {
                return None;
            }
            Some(mpir_bindings::__gmpz_sizeinbase(&value.integer as mpir_bindings::mpz_srcptr, 2) - 1)
        }
    }

    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        unsafe {
            let result = mpir_bindings::__gmpz_scan1(&value.integer as mpir_bindings::mpz_srcptr, 0);
            if result == mpir_bindings::mpir_ui::MAX {
                return None;
            } else {
                return Some(result as usize);
            }
        }
    }

    fn rounded_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut rhs_half = self.abs(self.clone_el(rhs));
        self.euclidean_div_pow_2(&mut rhs_half, 1);
        if self.is_neg(&lhs) {
            return self.euclidean_div(self.sub(lhs, rhs_half), rhs);
        } else {
            return self.euclidean_div(self.add(lhs, rhs_half), rhs);
        }
    }

    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, rng: G) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new();
            let len = (log2_bound_exclusive - 1) / u64::BITS as usize + 1;
            let mut data = Vec::new();
            data.resize_with(len, rng);

            mpir_bindings::__gmpz_import(
                &mut result.integer as mpir_bindings::mpz_ptr, 
                data.len(), 
                -1i32,
                (u64::BITS / 8) as libc::size_t,
                0, 
                0, 
                (data.as_ptr() as *const mpir_bindings::mpir_ui) as *const libc::c_void
            );

            self.euclidean_div_pow_2(&mut result, len * u64::BITS as usize - log2_bound_exclusive);
            return result;
        }
    }

    fn representable_bits(&self) -> Option<usize> {
        None
    }
}

impl_interpolation_base_ring_char_zero!{ InterpolationBaseRing for MPZBase }

impl_poly_gcd_locally_for_ZZ!{ IntegerPolyGCDRing for MPZBase }

impl_eval_poly_locally_for_ZZ!{ EvalPolyLocallyRing for MPZBase }

impl FiniteRingSpecializable for MPZBase {
    
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl HashableElRing for MPZBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        unsafe {
            <_ as std::hash::Hash>::hash(&(self.is_neg(&el), self.abs_highest_set_bit(&el), mpir_bindings::__gmpz_get_ui(&el.integer as mpir_bindings::mpz_srcptr)), h)
        }
    }
}

impl IntCast<RustBigintRingBase> for MPZBase {

    fn cast(&self, _: &RustBigintRingBase, el: RustBigint) -> Self::Element {
        let mut result = MPZEl::new();
        self.from_base_u64_repr(&mut result, &RustBigintRing::RING.get_ring().abs_base_u64_repr(&el).collect::<Vec<_>>()[..]);
        if RustBigintRing::RING.is_neg(&el) {
            self.negate_inplace(&mut result);
        }
        return result;
    }
}

impl IntCast<MPZBase> for RustBigintRingBase {

    fn cast(&self, from: &MPZBase, el: MPZEl) -> RustBigint {
        let mut result = (0..from.abs_base_u64_repr_len(&el)).map(|_| 0u64).collect::<Vec<_>>();
        from.abs_base_u64_repr(&el, &mut result[..]);
        let result = RustBigintRing::RING.get_ring().from_base_u64_repr(result.into_iter());
        if from.is_neg(&el) {
            return RustBigintRing::RING.negate(result);
        } else {
            return result;
        }
    }
}

impl IntCast<MPZBase> for MPZBase {

    fn cast(&self, _from: &MPZBase, el: MPZEl) -> MPZEl {
        el
    }
}

impl IntCast<StaticRingBase<i64>> for MPZBase {

    fn cast(&self, _: &StaticRingBase<i64>, el: i64) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new(); 
            mpir_bindings::__gmpz_set_si(&mut result.integer as *mut _, el);
            return result;
        }
    }
}

impl IntCast<MPZBase> for StaticRingBase<i64> {

    fn cast(&self, from: &MPZBase, el: MPZEl) -> i64 {
        assert!(from.abs_highest_set_bit(&el).unwrap_or(0) < u64::BITS as usize);
        let negative = from.is_neg(&el);
        unsafe {
            let result = mpir_bindings::__gmpz_get_ui(&el.integer as mpir_bindings::mpz_srcptr);
            if !negative {
                result.try_into().unwrap()
            } else if result == i64::MAX as u64 + 1 {
                i64::MIN
            } else {
                -i64::try_from(result).unwrap()
            }
        }
    }
}

impl IntCast<StaticRingBase<i128>> for MPZBase {

    fn cast(&self, _: &StaticRingBase<i128>, el: i128) -> MPZEl {
        let el_abs = el.unsigned_abs();
        let data = [(el_abs & u64::MAX as u128) as u64, (el_abs >> u64::BITS) as u64];
        let mut result = MPZEl::new();
        self.from_base_u64_repr(&mut result, &data[..]);
        if el < 0 {
            self.negate_inplace(&mut result);
        }
        return result;
    }
}

impl IntCast<MPZBase> for StaticRingBase<i128> {

    fn cast(&self, from: &MPZBase, el: MPZEl) -> i128 {
        assert!(from.abs_base_u64_repr_len(&el) <= 2);
        let mut data = [0u64; 2];
        from.abs_base_u64_repr(&el, &mut data);
        let result = data[0] as u128 + ((data[1] as u128) << u64::BITS);
        if from.is_neg(&el) {
            if result == i128::MIN.unsigned_abs() {
                return i128::MIN;
            } else {
                return -i128::try_from(result).unwrap();
            }
        } else {
            return result.try_into().unwrap();
        }
    }
}

macro_rules! specialize_int_cast {
    ($($from:ty),*) => {
        $(
            impl IntCast<StaticRingBase<$from>> for MPZBase {

                fn cast(&self, _: &StaticRingBase<$from>, el: $from) -> MPZEl {
                    int_cast(el.try_into().unwrap(), &RingRef::new(self), StaticRing::<i64>::RING)
                }
            }

            impl IntCast<MPZBase> for StaticRingBase<$from> {

                fn cast(&self, _: &MPZBase, el: MPZEl) -> $from {
                    int_cast(el, &StaticRing::<i64>::RING, &MPZ::RING) as $from
                }
            }
        )*
    }
}

specialize_int_cast!{ i8, i16, i32 }

#[cfg(test)]
use crate::pid::EuclideanRingStore;
use crate::serialization::DeserializeWithRing;
use crate::serialization::SerializableElementRing;

#[cfg(test)]
fn edge_case_elements_bigint() -> impl Iterator<Item = RustBigint> {
    [
        RustBigintRing::RING.int_hom().map(0),
        RustBigintRing::RING.int_hom().map(1),
        RustBigintRing::RING.int_hom().map(-1),
        RustBigintRing::RING.int_hom().map(7),
        RustBigintRing::RING.int_hom().map(-7),
        RustBigintRing::RING.int_hom().map(i32::MAX),
        RustBigintRing::RING.int_hom().map(i32::MIN),
        RustBigintRing::RING.power_of_two(64),
        RustBigintRing::RING.negate(RustBigintRing::RING.power_of_two(64)),
        RustBigintRing::RING.sub(RustBigintRing::RING.power_of_two(64), RustBigintRing::RING.one()),
        RustBigintRing::RING.power_of_two(128),
        RustBigintRing::RING.negate(RustBigintRing::RING.power_of_two(128)),
        RustBigintRing::RING.sub(RustBigintRing::RING.power_of_two(128), RustBigintRing::RING.one()),
        RustBigintRing::RING.power_of_two(192),
        RustBigintRing::RING.negate(RustBigintRing::RING.power_of_two(192)),
        RustBigintRing::RING.sub(RustBigintRing::RING.power_of_two(192), RustBigintRing::RING.one())
    ].into_iter()
}

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = MPZEl> {
    edge_case_elements_bigint().map(|n| MPZ::RING.coerce(&RustBigintRing::RING, n))
}

#[test]
fn test_negate_inplace() {
    let mut a = MPZ::RING.power_of_two(64);
    MPZ::RING.negate_inplace(&mut a);
    assert!(MPZ::RING.is_neg(&a));

    for a in edge_case_elements() {
        let mut b = MPZ::RING.clone_el(&a);
        MPZ::RING.negate_inplace(&mut b);
        assert!(MPZ::RING.is_zero(&a) || (MPZ::RING.is_neg(&a) ^ MPZ::RING.is_neg(&b)));
    }
}

#[test]
fn test_ring_axioms() {
    crate::ring::generic_tests::test_ring_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_hash_axioms() {
    crate::ring::generic_tests::test_hash_axioms(MPZ::RING, edge_case_elements());
}

#[test]
fn test_divisibility_ring_axioms() {
    crate::divisibility::generic_tests::test_divisibility_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_euclidean_ring_axioms() {
    crate::pid::generic_tests::test_euclidean_ring_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_integer_ring_axioms() {
    crate::integer::generic_tests::test_integer_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_canonical_iso_axioms_i32() {
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i32>::RING, MPZ::RING, [0, -1, 1, i16::MAX as i32, i16::MIN as i32].into_iter());
    crate::ring::generic_tests::test_iso_axioms(StaticRing::<i32>::RING, MPZ::RING, [0, -1, 1, i16::MIN as i32, i16::MAX as i32].into_iter());
}

#[test]
fn test_canonical_iso_axioms_i64() {
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, MPZ::RING, [0, -1, 1, i32::MAX as i64, i32::MIN as i64].into_iter());
    crate::ring::generic_tests::test_iso_axioms(StaticRing::<i64>::RING, MPZ::RING, [0, -1, 1, i32::MIN as i64, i32::MAX as i64].into_iter());

    assert_el_eq!(MPZ::RING, MPZ::RING.sub(MPZ::RING.power_of_two(63), MPZ::RING.one()), &MPZ::RING.coerce(&StaticRing::<i64>::RING, i64::MAX));
    assert_el_eq!(MPZ::RING, MPZ::RING.negate(MPZ::RING.power_of_two(63)), MPZ::RING.coerce(&StaticRing::<i64>::RING, i64::MIN));
    assert_eq!(i64::MAX, int_cast(MPZ::RING.sub(MPZ::RING.power_of_two(63), MPZ::RING.one()), &StaticRing::<i64>::RING, MPZ::RING));
    assert_eq!(i64::MIN, int_cast(MPZ::RING.negate(MPZ::RING.power_of_two(63)), &StaticRing::<i64>::RING, MPZ::RING));
}

#[test]
fn test_canonical_iso_axioms_i128() {
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, MPZ::RING, [0, -1, 1, i64::MAX as i128, i64::MIN as i128].into_iter());
    crate::ring::generic_tests::test_iso_axioms(StaticRing::<i128>::RING, MPZ::RING, [0, -1, 1, i64::MIN as i128, i64::MAX as i128].into_iter());

    assert_el_eq!(MPZ::RING, MPZ::RING.sub(MPZ::RING.power_of_two(127), MPZ::RING.one()), &MPZ::RING.coerce(&StaticRing::<i128>::RING, i128::MAX));
    assert_el_eq!(MPZ::RING, MPZ::RING.negate(MPZ::RING.power_of_two(127)), MPZ::RING.coerce(&StaticRing::<i128>::RING, i128::MIN));
    assert_eq!(i128::MAX, int_cast(MPZ::RING.sub(MPZ::RING.power_of_two(127), MPZ::RING.one()), &StaticRing::<i128>::RING, MPZ::RING));
    assert_eq!(i128::MIN, int_cast(MPZ::RING.negate(MPZ::RING.power_of_two(127)), &StaticRing::<i128>::RING, MPZ::RING));
}

#[test]
fn test_canonical_iso_axioms_bigint() {
    crate::ring::generic_tests::test_hom_axioms(RustBigintRing::RING, MPZ::RING, edge_case_elements_bigint());
    crate::ring::generic_tests::test_iso_axioms(RustBigintRing::RING, MPZ::RING, edge_case_elements_bigint());
}

#[test]
fn test_abs_is_bit_set() {
    let a = MPZ::RING.int_hom().map(1 << 15);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 15));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 16));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 14));

    let a = MPZ::RING.int_hom().map(-7);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 0));
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 1));
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 2));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 3));

    let a = MPZ::RING.int_hom().map(-1 << 15);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 15));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 16));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 14));
}

#[test]
fn test_highest_set_bit() {
    assert_eq!(None, MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(0)));
    assert_eq!(Some(0), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(1)));
    assert_eq!(Some(0), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(-1)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(2)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(-2)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(3)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(-3)));
    assert_eq!(Some(2), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(4)));
    assert_eq!(Some(2), MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(-4)));
}

#[test]
fn test_lowest_set_bit() {
    assert_eq!(None, MPZ::RING.abs_highest_set_bit(&MPZ::RING.int_hom().map(0)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(1)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(-1)));
    assert_eq!(Some(1), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(2)));
    assert_eq!(Some(1), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(-2)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(3)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(-3)));
    assert_eq!(Some(2), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(4)));
    assert_eq!(Some(2), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.int_hom().map(-4)));
}

#[test]
fn from_to_float_approx() {
    let x: f64 = 83465209236517892563478156042389675783219532497861237985328563.;
    let y = MPZ::RING.to_float_approx(&MPZ::RING.from_float_approx(x).unwrap());
    assert!(x * 0.99 < y);
    assert!(y < x * 1.01);
}

#[bench]
fn bench_mul_300_bits(bencher: &mut test::Bencher) {
    let x = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap());
    let y = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap());
    let z = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("116588006478839442056346504147013274749794691549803163727888681858469844569693215953808606899770104590589390919543097259495176008551856143726436", 10).unwrap());
    bencher.iter(|| {
        let p = MPZ::RING.mul_ref(&x, &y);
        assert_el_eq!(MPZ::RING, z, p);
    })
}

#[bench]
fn bench_div_300_bits(bencher: &mut test::Bencher) {
    let x = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap());
    let y = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap());
    let z = MPZ::RING.coerce(&RustBigintRing::RING, RustBigintRing::RING.get_ring().parse("116588006478839442056346504147013274749794691549803163727888681858469844569693215953808606899770104590589390919543097259495176008551856143726436", 10).unwrap());
    bencher.iter(|| {
        let q = MPZ::RING.euclidean_div(MPZ::RING.clone_el(&z), &y);
        assert_el_eq!(MPZ::RING, x, q);
    })
}

#[test]
fn test_serialization() {
    crate::serialization::generic_tests::test_serialization(MPZ::RING, edge_case_elements());
}