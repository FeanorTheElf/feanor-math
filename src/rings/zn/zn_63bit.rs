use crate::euclidean::EuclideanRingStore;
use crate::integer::IntegerRingStore;
use crate::primitive_int::*;
use crate::ring::*;
use crate::rings::bigint::DefaultBigIntRing;

///
/// Represents the ring `Z/nZ`.
/// A special implementation of non-standard Barett reduction
/// that uses 128-bit integer but provides moduli up to 64 bit.
///
#[derive(Clone, Copy)]
pub struct ZnBase {
    inv_modulus: u128,
    modulus: u64
}

fn mul_128_128_hw128(a: u128, b: u128) -> u128 {
    let (a_high, a_low) = (a >> u64::BITS, a & ((1 << u64::BITS) - 1));
    let (b_high, b_low) = (b >> u64::BITS, b & ((1 << u64::BITS) - 1));
    return a_high * b_high + ((a_high * b_low) >> u64::BITS) + ((a_low * b_high) >> u64::BITS);
}

#[derive(Copy, Clone)]
pub struct ZnEl(u64);

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus > i32::MAX as u64, "if the modulus is <= i32::MAX, use `zn_barett::Zn::<StaticRing<i128>>` instead");
        // to allow addition in u64
        assert!(modulus <= u64::MAX / 2);
        let big_ZZ = DefaultBigIntRing::RING;
        return ZnBase {
            modulus: modulus,
            inv_modulus: big_ZZ.cast(
                &StaticRing::<i128>::RING,
                big_ZZ.euclidean_div(
                    big_ZZ.power_of_two(u128::BITS as usize), 
                    &big_ZZ.coerce(&StaticRing::<i64>::RING, modulus as i64)
                )
            ) as u128
        }
    }
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        *lhs += rhs;
        if *lhs > self.modulus {
            *lhs -= self.modulus;
        }
    }
    
    fn negate_inplace(&self, ZnEl(lhs): &mut Self::Element) {
        if *lhs != 0 {
            *lhs = self.modulus - *lhs;
        }
    }

    fn mul_assign(&self, ZnEl(lhs): &mut Self::Element, ZnEl(rhs): Self::Element) {
        let product = *lhs as u128 * rhs as u128;

        // perform Barett reduction
        let quotient = mul_128_128_hw128(product, self.inv_modulus);
        *lhs = (product - quotient * self.modulus as u128) as u64;
        if *lhs > self.modulus {
            *lhs -= self.modulus;
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        debug_assert!(self.modulus > i32::MAX as u64);
        if value < 0 {
            return ZnEl(self.modulus - (-value as u64));
        } else {
            return ZnEl(value as u64);
        }
    }

    fn eq(&self, ZnEl(lhs): &Self::Element, ZnEl(rhs): &Self::Element) -> bool {
        *lhs == *rhs
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn dbg<'a>(&self, ZnEl(value): &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", *value)
    }

}
