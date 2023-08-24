use libc;

use crate::divisibility::DivisibilityRing;
use crate::euclidean::EuclideanRing;
use crate::ordered::OrderedRing;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::integer::*;
use super::bigint::*;

mod mpir_bindings;

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

impl Drop for MPZEl {

    fn drop(&mut self) {
        unsafe {
            mpir_bindings::__gmpz_clear(&mut self.integer as mpir_bindings::mpz_ptr)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MPZ;

impl MPZ {

    pub const RING: RingValue<MPZ> = RingValue::from(MPZ);

    pub fn from_base_u64_repr<I: IntoIterator<Item = u64>>(&self, dst: &mut MPZEl, input: I) {
        unsafe {
            assert_eq!(std::mem::size_of::<mpir_bindings::mpir_ui>(), std::mem::size_of::<u64>());
            let data: Vec<mpir_bindings::mpir_ui> = input.into_iter().collect();
            if data.len() == 0 {
                mpir_bindings::__gmpz_set_ui(&mut dst.integer as mpir_bindings::mpz_ptr, 0);
                return;
            }
            assert!(data.len() > 0);
            mpir_bindings::__gmpz_import(
                &mut dst.integer as mpir_bindings::mpz_ptr, 
                data.len(), 
                -1i32,
                (u64::BITS / 8) as libc::size_t,
                0, 
                0, 
                (data.as_ptr() as *const mpir_bindings::mpir_ui) as *const libc::c_void
            );
        }
    }

    pub fn abs_base_u64_repr(&self, src: &MPZEl) -> impl 'static + Iterator<Item = u64> {
        unsafe {
            let size = self.abs_highest_set_bit(src).map(|n| n + 1).unwrap_or(0);
            let mut data = Vec::new();

            if size > 0 {
                data.resize(size, 0u64);
                let mut size = 0;
        
                mpir_bindings::__gmpz_export(
                    (data.as_mut_ptr() as *mut mpir_bindings::mpir_ui) as *mut libc::c_void,
                    &mut size,
                    -1,
                    (u64::BITS / 8) as libc::size_t,
                    0,
                    0,
                    &src.integer as mpir_bindings::mpz_srcptr
                );
                assert_eq!(size, size);
            }
            return data.into_iter();
        }
    }
}

impl RingBase for MPZ {
    
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

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        DefaultBigIntRing::RING.get_ring().dbg(&self.map_out(DefaultBigIntRing::RING.get_ring(), self.clone_el(value), &()), out)
    }
}

impl OrderedRing for MPZ {

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

impl DivisibilityRing for MPZ {

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
}

impl EuclideanRing for MPZ {
    
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

impl IntegerRing for MPZ {
    
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

            self.euclidean_div_pow_2(&mut result, len * 8 - log2_bound_exclusive);
            return result;
        }
    }
}

impl HashableElRing for MPZ {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        unsafe {
            <_ as std::hash::Hash>::hash(&(self.is_neg(&el), self.abs_highest_set_bit(&el), mpir_bindings::__gmpz_get_ui(&el.integer as mpir_bindings::mpz_srcptr)), h)
        }
    }
}

impl CanonicalHom<MPZ> for MPZ {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &MPZ) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _: &MPZ, el: MPZEl, _: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<MPZ> for MPZ {

    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &MPZ) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, _: &MPZ, el: Self::Element, _: &Self::Isomorphism) -> <MPZ as RingBase>::Element {
        el
    }
}

impl CanonicalHom<DefaultBigIntRing> for MPZ {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &DefaultBigIntRing) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in_ref(&self, _: &DefaultBigIntRing, el: &DefaultBigIntRingEl, _: &Self::Homomorphism) -> Self::Element {
        let mut result = MPZEl::new();
        self.from_base_u64_repr(&mut result, DefaultBigIntRing::RING.get_ring().abs_base_u64_repr(&el));
        if DefaultBigIntRing::RING.is_neg(el) {
            self.negate_inplace(&mut result);
        }
        return result;
    }

    fn map_in(&self, from: &DefaultBigIntRing, el: <DefaultBigIntRing as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl CanonicalIso<DefaultBigIntRing> for MPZ {

    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &DefaultBigIntRing) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, _: &DefaultBigIntRing, el: Self::Element, _: &Self::Isomorphism) -> DefaultBigIntRingEl {
        let result = DefaultBigIntRing::RING.get_ring().from_base_u64_repr(self.abs_base_u64_repr(&el));
        if self.is_neg(&el) {
            return DefaultBigIntRing::RING.negate(result);
        } else {
            return result;
        }
    }
}

impl CanonicalHom<StaticRingBase<i32>> for MPZ {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &StaticRingBase<i32>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _: &StaticRingBase<i32>, el: i32, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(StaticRing::<i64>::RING.get_ring(), el.into(), hom)
    }
}

impl CanonicalIso<StaticRingBase<i32>> for MPZ {

    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &StaticRingBase<i32>) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, _: &StaticRingBase<i32>, el: Self::Element, iso: &Self::Isomorphism) -> <StaticRingBase<i32> as RingBase>::Element {
        self.map_out(StaticRing::<i64>::RING.get_ring(), el, iso).try_into().unwrap()
    }
}

impl CanonicalHom<StaticRingBase<i64>> for MPZ {

    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &StaticRingBase<i64>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, _: &StaticRingBase<i64>, el: i64, _: &Self::Homomorphism) -> Self::Element {
        unsafe {
            let mut result = MPZEl::new(); 
            mpir_bindings::__gmpz_set_si(&mut result.integer as *mut _, el);
            return result;
        }
    }
}

impl CanonicalIso<StaticRingBase<i64>> for MPZ {

    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &StaticRingBase<i64>) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, _: &StaticRingBase<i64>, el: Self::Element, _: &Self::Isomorphism) -> <StaticRingBase<i64> as RingBase>::Element {
        assert!(self.abs_highest_set_bit(&el).unwrap_or(0) < u64::BITS as usize);
        let negative = self.is_neg(&el);
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

#[cfg(test)]
use crate::euclidean::EuclideanRingStore;
#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;
#[cfg(test)]
use crate::euclidean::generic_test_euclidean_axioms;

#[cfg(test)]
fn edge_case_elements_bigint() -> impl Iterator<Item = DefaultBigIntRingEl> {
    [
        DefaultBigIntRing::RING.from_int(0),
        DefaultBigIntRing::RING.from_int(1),
        DefaultBigIntRing::RING.from_int(-1),
        DefaultBigIntRing::RING.from_int(7),
        DefaultBigIntRing::RING.from_int(-7),
        DefaultBigIntRing::RING.from_int(i32::MAX),
        DefaultBigIntRing::RING.from_int(i32::MIN),
        DefaultBigIntRing::RING.power_of_two(64),
        DefaultBigIntRing::RING.negate(DefaultBigIntRing::RING.power_of_two(64)),
        DefaultBigIntRing::RING.sub(DefaultBigIntRing::RING.power_of_two(64), DefaultBigIntRing::RING.one()),
        DefaultBigIntRing::RING.power_of_two(128),
        DefaultBigIntRing::RING.negate(DefaultBigIntRing::RING.power_of_two(128)),
        DefaultBigIntRing::RING.sub(DefaultBigIntRing::RING.power_of_two(128), DefaultBigIntRing::RING.one()),
        DefaultBigIntRing::RING.power_of_two(192),
        DefaultBigIntRing::RING.negate(DefaultBigIntRing::RING.power_of_two(192)),
        DefaultBigIntRing::RING.sub(DefaultBigIntRing::RING.power_of_two(192), DefaultBigIntRing::RING.one())
    ].into_iter()
}

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = MPZEl> {
    edge_case_elements_bigint().map(|n| MPZ::RING.coerce(&DefaultBigIntRing::RING, n))
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
    generic_test_ring_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_divisibility_ring_axioms() {
    generic_test_divisibility_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_euclidean_ring_axioms() {
    generic_test_euclidean_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_integer_ring_axioms() {
    generic_test_integer_axioms(MPZ::RING, edge_case_elements())
}

#[test]
fn test_canonical_iso_axioms_i32() {
    generic_test_canonical_hom_axioms(StaticRing::<i32>::RING, MPZ::RING, [0, -1, 1, i16::MAX as i32, i16::MIN as i32].into_iter());
    generic_test_canonical_iso_axioms(StaticRing::<i32>::RING, MPZ::RING, [0, -1, 1, i16::MIN as i32, i16::MAX as i32].into_iter());
}

#[test]
fn test_canonical_iso_axioms_i64() {
    generic_test_canonical_hom_axioms(StaticRing::<i64>::RING, MPZ::RING, [0, -1, 1, i32::MAX as i64, i32::MIN as i64].into_iter());
    generic_test_canonical_iso_axioms(StaticRing::<i64>::RING, MPZ::RING, [0, -1, 1, i32::MIN as i64, i32::MAX as i64].into_iter());
}

#[test]
fn test_canonical_iso_axioms_bigint() {
    generic_test_canonical_hom_axioms(DefaultBigIntRing::RING, MPZ::RING, edge_case_elements_bigint());
    generic_test_canonical_iso_axioms(DefaultBigIntRing::RING, MPZ::RING, edge_case_elements_bigint());
}

#[test]
fn test_abs_is_bit_set() {
    let a = MPZ::RING.from_int(1 << 15);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 15));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 16));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 14));

    let a = MPZ::RING.from_int(-7);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 0));
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 1));
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 2));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 3));

    let a = MPZ::RING.from_int(-1 << 15);
    assert_eq!(true, MPZ::RING.abs_is_bit_set(&a, 15));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 16));
    assert_eq!(false, MPZ::RING.abs_is_bit_set(&a, 14));
}

#[test]
fn test_highest_set_bit() {
    assert_eq!(None, MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(0)));
    assert_eq!(Some(0), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(1)));
    assert_eq!(Some(0), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(-1)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(2)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(-2)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(3)));
    assert_eq!(Some(1), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(-3)));
    assert_eq!(Some(2), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(4)));
    assert_eq!(Some(2), MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(-4)));
}

#[test]
fn test_lowest_set_bit() {
    assert_eq!(None, MPZ::RING.abs_highest_set_bit(&MPZ::RING.from_int(0)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(1)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(-1)));
    assert_eq!(Some(1), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(2)));
    assert_eq!(Some(1), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(-2)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(3)));
    assert_eq!(Some(0), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(-3)));
    assert_eq!(Some(2), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(4)));
    assert_eq!(Some(2), MPZ::RING.abs_lowest_set_bit(&MPZ::RING.from_int(-4)));
}

#[bench]
fn bench_mul(bencher: &mut test::Bencher) {
    let x = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap());
    let y = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap());
    let z = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("116588006478839442056346504147013274749794691549803163727888681858469844569693215953808606899770104590589390919543097259495176008551856143726436", 10).unwrap());
    bencher.iter(|| {
        let p = MPZ::RING.mul_ref(&x, &y);
        assert_el_eq!(&MPZ::RING, &z, &p);
    })
}

#[bench]
fn bench_div(bencher: &mut test::Bencher) {
    let x = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap());
    let y = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap());
    let z = MPZ::RING.coerce(&DefaultBigIntRing::RING, DefaultBigIntRing::RING.get_ring().parse("116588006478839442056346504147013274749794691549803163727888681858469844569693215953808606899770104590589390919543097259495176008551856143726436", 10).unwrap());
    bencher.iter(|| {
        let q = MPZ::RING.euclidean_div(MPZ::RING.clone_el(&z), &y);
        assert_el_eq!(&MPZ::RING, &x, &q);
    })
}
