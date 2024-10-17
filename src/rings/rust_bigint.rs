use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer}; 

use crate::algorithms::bigint::highest_set_block;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::Domain;
use crate::{impl_interpolation_base_ring_char_zero, pid::*};
use crate::integer::*;
use crate::ordered::*;
use crate::primitive_int::*;
use crate::ring::*;
use crate::algorithms;
use crate::serialization::SerializableElementRing;
use std::alloc::Allocator;
use std::alloc::Global;
use std::cmp::Ordering::*;

#[derive(Clone, Debug)]
pub struct RustBigint<A: Allocator = Global>(bool, Vec<u64, A>);

///
/// Arbitrary-precision integer implementation.
/// 
/// This is a not-too-well optimized implementation, written in pure Rust.
/// If you need very high performance, consider using [`crate::rings::mpir::MPZ`]
/// (requires an installation of mpir and activating the feature "mpir").
/// 
#[derive(Copy, Clone)]
pub struct RustBigintRingBase<A: Allocator + Clone = Global> {
    allocator: A
}

///
/// [`RingStore`] corresponding to [`RustBigintRingBase`].
/// 
pub type RustBigintRing<A = Global> = RingValue<RustBigintRingBase<A>>;

impl<A: Allocator + Clone> RustBigintRing<A> {
    
    #[stability::unstable(feature = "enable")]
    pub fn new_with(allocator: A) -> RustBigintRing<A> {
        Self::from(RustBigintRingBase { allocator })
    }
}

impl RustBigintRing {
    
    ///
    /// Default instance of [`RustBigintRing`], the ring of arbitrary-precision integers.
    /// 
    pub const RING: RustBigintRing = RingValue::from(RustBigintRingBase { allocator: Global });
}

impl<A: Allocator + Clone + Default> Default for RustBigintRingBase<A> {

    fn default() -> Self {
        RustBigintRingBase { allocator: A::default() }
    }
}

impl<A: Allocator + Clone> RustBigintRingBase<A> {

    pub fn map_i128(&self, val: &RustBigint<A>) -> Option<i128> {
        match algorithms::bigint::highest_set_block(&val.1) {
            None => Some(0),
            Some(0) if val.0 => Some(-(val.1[0] as i128)),
            Some(0) if !val.0 => Some(val.1[0] as i128),
            Some(1) if val.0 => {
                let value = val.1[0] as u128 + ((val.1[1] as u128) << u64::BITS);
                if value == 1 << (u128::BITS - 1) {
                    Some(i128::MIN)
                } else {
                    i128::try_from(value).ok().map(|x| -x)
                }
            },
            Some(1) if !val.0 => i128::try_from(val.1[0] as u128 + ((val.1[1] as u128) << u64::BITS)).ok(),
            Some(_) => None
        }
    }

    pub fn parse(&self, string: &str, base: u32) -> Result<RustBigint<A>, ()> {
        let (negative, rest) = if string.chars().next() == Some('-') {
            (true, string.split_at(1).1)
        } else if string.chars().next() == Some('+') {
            (false, string.split_at(1).1)
        } else {
            (false, string)
        };
        Ok(RustBigint(negative, algorithms::bigint::from_str_radix(rest, base, self.zero().1)?))
    }

    pub fn abs_base_u64_repr<'a>(&self, el: &'a RustBigint) -> impl 'a + Iterator<Item = u64> {
        el.1.iter().copied()
    }

    pub fn from_base_u64_repr<I>(&self, data: I) -> RustBigint
        where I: Iterator<Item = u64>
    {
        RustBigint(false, data.collect())
    }
}

impl<A: Allocator + Clone> PartialEq for RustBigintRingBase<A> {

    fn eq(&self, _other: &Self) -> bool {
        // it is perfectly valid to swap elements between two different `RustBigintRing`s,
        // even if they have different allocators. Every element keeps track of their allocator 
        // themselves
        true
    }
}

impl<A: Allocator + Clone> RingBase for RustBigintRingBase<A> {
    
    type Element = RustBigint<A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        // allocate it with our allocator
        let mut result_data = Vec::with_capacity_in(val.1.len(), self.allocator.clone());
        result_data.extend(val.1.iter().copied());
        RustBigint(val.0, result_data)
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        match (lhs, rhs) {
            (RustBigint(false, lhs_val), RustBigint(false, rhs_val)) |
            (RustBigint(true, lhs_val), RustBigint(true, rhs_val)) => {
                algorithms::bigint::bigint_add(lhs_val, rhs_val, 0);
            },
            (RustBigint(lhs_sgn, lhs_val), RustBigint(_, rhs_val)) => {
                match algorithms::bigint::bigint_cmp(lhs_val, rhs_val) {
                    Less => {
                        algorithms::bigint::bigint_sub_self(lhs_val, rhs_val);
                        *lhs_sgn = !*lhs_sgn;
                    },
                    Equal => {
                        lhs_val.clear();
                    },
                    Greater => {
                        algorithms::bigint::bigint_sub(lhs_val, rhs_val, 0);
                    }
                }
            }
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { 
        self.negate_inplace(lhs);
        self.add_assign_ref(lhs, rhs);
        self.negate_inplace(lhs);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        lhs.0 = !lhs.0;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let result = algorithms::bigint::bigint_mul(&lhs.1, &rhs.1, Vec::new_in(self.allocator.clone()));
        *lhs = RustBigint(lhs.0 ^ rhs.0, result);
    }

    fn zero(&self) -> Self::Element {
        RustBigint(false, Vec::new_in(self.allocator.clone()))
    }

    fn from_int(&self, value: i32) -> Self::Element {
        let mut data = Vec::with_capacity_in(1, self.allocator.clone());
        data.push((value as i64).abs() as u64);
        RustBigint(value < 0, data)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.0 == rhs.0 {
            algorithms::bigint::bigint_cmp(&lhs.1, &rhs.1) == Equal
        } else {
            self.is_zero(lhs) && self.is_zero(rhs)
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        algorithms::bigint::highest_set_block(&value.1).is_none()
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.0 == false && algorithms::bigint::highest_set_block(&value.1) == Some(0) && value.1[0] == 1
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.0 == true && algorithms::bigint::highest_set_block(&value.1) == Some(0) && value.1[0] == 1
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        ///
        /// 10 to this power fits still in a u64
        /// 
        const BIG_POWER_TEN_ZEROS: usize = 19;
        const BIG_POWER_TEN: u64 = 10u64.pow(BIG_POWER_TEN_ZEROS as u32);

        if value.0 {
            write!(out, "-")?;
        }
        let mut copy = value.clone();
        let mut remainders: Vec<u64> = Vec::with_capacity(
            (algorithms::bigint::highest_set_block(&value.1).unwrap_or(0) + 1) * u64::BITS as usize / 3
        );
        while !self.is_zero(&copy) {
            let rem = algorithms::bigint::bigint_div_small(&mut copy.1, BIG_POWER_TEN);
            remainders.push(rem);
        }
        remainders.reverse();
        let mut it = remainders.into_iter();
        if let Some(fst) = it.next() {
            write!(out, "{}", fst)?;
            for rem in it {
                write!(out, "{:0>width$}", rem, width = BIG_POWER_TEN_ZEROS)?;
            }
        } else {
            write!(out, "0")?;
        }
        return Ok(());
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(other_ZZ.zero())
    }
    
    fn is_approximate(&self) -> bool { false }
}

impl<A1: Allocator + Clone, A2: Allocator + Clone> IntCast<RustBigintRingBase<A2>> for RustBigintRingBase<A1> {
    
    fn cast(&self, _: &RustBigintRingBase<A2>, value: RustBigint<A2>) -> Self::Element {
        // allocate it with our allocator
        let mut result_data = Vec::with_capacity_in(value.1.len(), self.allocator.clone());
        result_data.extend(value.1.iter().copied());
        RustBigint(value.0, result_data)
    }
}

macro_rules! specialize_int_cast {
    ($($from:ty),*) => {
        $(
            impl<A: Allocator + Clone> IntCast<StaticRingBase<$from>> for RustBigintRingBase<A> {

                fn cast(&self, _: &StaticRingBase<$from>, value: $from) -> RustBigint<A> {
                    let negative = value < 0;
                    let value = <_ as Into<i128>>::into(value).checked_abs().map(|x| x as u128).unwrap_or(1 << (u128::BITS - 1));
                    let mut result = Vec::with_capacity_in(2, self.allocator.clone());
                    result.extend([(value & ((1 << u64::BITS) - 1)) as u64, (value >> u64::BITS) as u64].into_iter());
                    RustBigint(negative, result)
                }
            }

            impl<A: Allocator + Clone> IntCast<RustBigintRingBase<A>> for StaticRingBase<$from> {

                fn cast(&self, from: &RustBigintRingBase<A>, value: RustBigint<A>) -> $from {
                    <$from>::try_from(from.map_i128(&value).unwrap()).ok().unwrap()
                }
            }
        )*
    };
}

specialize_int_cast!{ i8, i16, i32, i64, i128 }

impl<A: Allocator + Clone> Domain for RustBigintRingBase<A> {}

impl<A: Allocator + Clone> OrderedRing for RustBigintRingBase<A> {

    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        match (lhs.0, rhs.0) {
            (true, true) => algorithms::bigint::bigint_cmp(&rhs.1, &lhs.1),
            (false, false) => algorithms::bigint::bigint_cmp(&lhs.1, &rhs.1),
            (_, _) if self.is_zero(lhs) && self.is_zero(rhs) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater
        }
    }

    fn abs_cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> std::cmp::Ordering {
        algorithms::bigint::bigint_cmp(&rhs.1, &lhs.1)
    }
}

impl<A: Allocator + Clone> DivisibilityRing for RustBigintRingBase<A> {
    
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(rhs) && self.is_zero(lhs) {
            return Some(self.zero());
        } else if self.is_zero(rhs) {
            return None;
        }
        let (quo, rem) = self.euclidean_div_rem(lhs.clone(), rhs);
        if self.is_zero(&rem) {
            Some(quo)
        } else {
            None
        }
    }
}

impl<A: Allocator + Clone> PrincipalIdealRing for RustBigintRingBase<A> {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }
    
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        algorithms::eea::eea(self.clone_el(lhs), self.clone_el(rhs), RingRef::new(self))
    }
}

impl<A: Allocator + Clone> EuclideanRing for RustBigintRingBase<A> {

    fn euclidean_div_rem(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        let mut quo = RustBigint(false, algorithms::bigint::bigint_div(&mut lhs.1, &rhs.1, self.zero().1));
        if rhs.0 ^ lhs.0 {// if result of division is zero, `.is_neg(&lhs)` does not work as expected
            self.negate_inplace(&mut quo);
        }
        return (quo, lhs);
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        self.map_i128(val).and_then(|x| x.checked_abs()).and_then(|x| usize::try_from(x).ok())
    }
}

impl<A: Allocator + Clone> HashableElRing for RustBigintRingBase<A> {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        let block = algorithms::bigint::highest_set_block(&el.1);
        if let Some(b) = block {
            for i in 0..=b {
                h.write_u64(el.1[i])
            }
        }
    }
}

impl<A: Allocator + Clone> SerializableElementRing for RustBigintRingBase<A> {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        if deserializer.is_human_readable() {
            let string = <String as Deserialize>::deserialize(deserializer)?;
            return self.parse(string.as_str(), 10).map_err(|()| de::Error::custom(format!("cannot parse \"{}\" as number", string)));
        } else {
            let (negative, data) = <(bool, &serde_bytes::Bytes) as Deserialize>::deserialize(deserializer)?;
            let mut result_data = Vec::with_capacity_in(data.len() / size_of::<u64>(), self.allocator.clone());
            for digit in data.array_chunks() {
                result_data.push(u64::from_le_bytes(*digit));
            }
            return Ok(RustBigint(negative, result_data));
        }
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        if serializer.is_human_readable() {
            <String as Serialize>::serialize(&format!("{}", RingRef::new(self).format(el)), serializer)
        } else {
            let len = highest_set_block(&el.1).map(|n| n + 1).unwrap_or(0);
            let mut data = Vec::with_capacity_in(len * size_of::<u64>(), &self.allocator);
            for digit in &el.1 {
                data.extend(digit.to_le_bytes().into_iter());
            }
            <(bool, &serde_bytes::Bytes) as Serialize>::serialize(&(el.0, serde_bytes::Bytes::new(&data)), serializer)
        }
    }
}

impl_interpolation_base_ring_char_zero!{ <{A}> InterpolationBaseRing for RustBigintRingBase<A> where A: Allocator + Clone }

impl<A: Allocator + Clone> IntegerRing for RustBigintRingBase<A> {

    fn to_float_approx(&self, value: &Self::Element) -> f64 {
        let sign = if value.0 { -1. } else { 1. };
        match algorithms::bigint::highest_set_block(&value.1) {
            None => 0.,
            Some(0) => value.1[0] as f64 * sign,
            Some(d) => (value.1[d] as f64 * 2f64.powi(d as i32 * u64::BITS as i32) + value.1[d - 1] as f64 * 2f64.powi((d - 1) as i32 * u64::BITS as i32)) * sign
        }
    }

    fn from_float_approx(&self, mut value: f64) -> Option<Self::Element> {
        if value.round() == 0. {
            return Some(self.zero());
        }
        let sign = value < 0.;
        value = value.abs();
        let scale = value.log2().ceil() as i32;
        let significant_digits = std::cmp::min(scale, u64::BITS as i32);
        let most_significant_bits = (value / 2f64.powi(scale - significant_digits)) as u64;
        let mut result = self.one();
        result.1[0] = most_significant_bits;
        result.0 = sign;
        self.mul_pow_2(&mut result, (scale - significant_digits) as usize);
        return Some(result);
    }

    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        if self.is_zero(value) {
            return None;
        }
        for i in 0..value.1.len() {
            if value.1[i] != 0 {
                return Some(i * u64::BITS as usize + value.1[i].trailing_zeros() as usize)
            }
        }
        unreachable!()
    }

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool {
        if i / u64::BITS as usize >= value.1.len() {
            false
        } else {
            (value.1[i / u64::BITS as usize] >> (i % u64::BITS as usize)) & 1 == 1
        }
    }

    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize> {
        let block = algorithms::bigint::highest_set_block(&value.1)?;
        Some(block * u64::BITS as usize + u64::BITS as usize - value.1[block].leading_zeros() as usize - 1)
    }

    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize) {
        algorithms::bigint::bigint_rshift(&mut value.1, power);
    }

    fn mul_pow_2(&self, value: &mut Self::Element, power: usize) {
        algorithms::bigint::bigint_lshift(&mut value.1, power)
    }

    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, mut rng: G) -> Self::Element {
        let blocks = log2_bound_exclusive / u64::BITS as usize;
        let in_block = log2_bound_exclusive % u64::BITS as usize;
        let mut result = Vec::with_capacity_in(blocks + if in_block > 0 { 1 } else { 0 }, self.allocator.clone());
        if in_block == 0 {
            result.extend((0..blocks).map(|_| rng()));
        } else {
            let last = rng() & 1u64.overflowing_shl(in_block as u32).0.overflowing_sub(1).0;
            result.extend((0..blocks).map(|_| rng()).chain(std::iter::once(last)));
        }
        return RustBigint(false, result);
    }
    
    fn representable_bits(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
use crate::homomorphism::*;

#[cfg(test)]
const ZZ: RustBigintRing = RustBigintRing::RING;

#[test]
fn test_print_power_2() {
    let x = RustBigint(false, vec![0, 0, 1]);
    assert_eq!("340282366920938463463374607431768211456", format!("{}", RustBigintRing::RING.format(&x)));
}

#[test]
fn test_from() {
    assert!(ZZ.eq_el(&RustBigint(false, vec![]), &ZZ.int_hom().map(0)));
    assert!(ZZ.eq_el(&RustBigint(false, vec![2138479]), &ZZ.int_hom().map(2138479)));
    assert!(ZZ.eq_el(&RustBigint(true, vec![2138479]), &ZZ.int_hom().map(-2138479)));
    // assert!(ZZ.eq(&RustBigint(false, vec![0x38691a350bf12fca, 0x1]), &ZZ.from_z_gen(0x138691a350bf12fca, &i128::RING)));
}

#[test]
fn test_to_i128() {
    let iso = ZZ.can_iso(&StaticRing::<i128>::RING).unwrap();
    assert_eq!(0, iso.map(RustBigint(false, vec![])));
    assert_eq!(2138479, iso.map(RustBigint(false, vec![2138479])));
    assert_eq!(-2138479, iso.map(RustBigint(true, vec![2138479])));
    assert_eq!(0x138691a350bf12fca, iso.map(RustBigint(false, vec![0x38691a350bf12fca, 0x1])));
    assert_eq!(i128::MAX, iso.map(RustBigint(false, vec![(i128::MAX & ((1 << 64) - 1)) as u64, (i128::MAX >> 64) as u64])));
    assert_eq!(i128::MIN + 1, iso.map(RustBigint(true, vec![(i128::MAX & ((1 << 64) - 1)) as u64, (i128::MAX >> 64) as u64])));
    assert_eq!(i64::MAX as i128 + 1, iso.map(RustBigint(false, vec![i64::MAX as u64 + 1])));
    assert_eq!(u64::MAX as i128, iso.map(RustBigint(false, vec![u64::MAX])));
}

#[test]
fn test_sub_assign() {
    let mut x = RustBigintRing::RING.get_ring().parse("4294836225", 10).unwrap();
    let y = RustBigintRing::RING.get_ring().parse("4294967297", 10).unwrap();
    let z = RustBigintRing::RING.get_ring().parse("-131072", 10).unwrap();
    x = ZZ.sub_ref_fst(&x, y);
    assert!(ZZ.eq_el(&z, &x));
}

#[test]
fn test_assumptions_integer_division() {
    assert_eq!(-1, -3 / 2);
    assert_eq!(-1, 3 / -2);
    assert_eq!(1, -3 / -2);
    assert_eq!(1, 3 / 2);

    assert_eq!(-1, -3 % 2);
    assert_eq!(1, 3 % -2);
    assert_eq!(-1, -3 % -2);
    assert_eq!(1, 3 % 2);
}

#[cfg(test)]
fn edge_case_elements() -> impl Iterator<Item = RustBigint> {
    
    const NUMBERS: [&'static str; 10] = [
        "5444517870735015415413993718908291383295", // power of two - 1
        "5444517870735015415413993718908291383296", // power of two
        "-5444517870735015415413993718908291383295",
        "-5444517870735015415413993718908291383296",
        "3489", // the rest is random
        "891023591340178345678931246518793456983745682137459364598623489512389745698237456890239238476873429872346579",
        "172365798123602365091834765607185713205612370956192783561461248973265193754762751378496572896497125361819754136",
        "0",
        "-231780567812394562346324763251741827457123654871236548715623487612384752328164",
        "+1278367182354612381234568509783420989356938472561078564732895634928563482349872698723465"
    ];

    NUMBERS.iter().cloned().map(|s| RustBigintRing::RING.get_ring().parse(s, 10)).map(Result::unwrap)
}

#[test]
fn test_bigint_ring_axioms() {
    crate::ring::generic_tests::test_ring_axioms(ZZ, edge_case_elements())
}

#[test]
fn test_hash_axioms() {
    crate::ring::generic_tests::test_hash_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_divisibility_ring_axioms() {
    crate::divisibility::generic_tests::test_divisibility_axioms(ZZ, edge_case_elements())
}

#[test]
fn test_bigint_euclidean_ring_axioms() {
    crate::pid::generic_tests::test_euclidean_ring_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_principal_ideal_ring_axioms() {
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_integer_ring_axioms() {
    crate::integer::generic_tests::test_integer_axioms(ZZ, edge_case_elements())
}

#[test]
fn from_to_float_approx() {
    let x: f64 = 83465209236517892563478156042389675783219532497861237985328563.;
    let y = ZZ.to_float_approx(&ZZ.from_float_approx(x).unwrap());
    assert!(x * 0.99 < y);
    assert!(y < x * 1.01);
}

#[bench]
fn bench_div_300_bits(bencher: &mut test::Bencher) {
    let x = RustBigintRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap();
    let y = RustBigintRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap();
    let z = RustBigintRing::RING.get_ring().parse("48682207850683149082203680872586784064678018", 10).unwrap();
    bencher.iter(|| {
        let q = ZZ.euclidean_div(x.clone(), &y);
        assert!(ZZ.eq_el(&z, &q));
    })
}

#[bench]
fn bench_mul_300_bits(bencher: &mut test::Bencher) {
    let x = RustBigintRing::RING.get_ring().parse("2382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap();
    let y = RustBigintRing::RING.get_ring().parse("48937502893645789234569182735646324895723409587234", 10).unwrap();
    let z = RustBigintRing::RING.get_ring().parse("116588006478839442056346504147013274749794691549803163727888681858469844569693215953808606899770104590589390919543097259495176008551856143726436", 10).unwrap();
    bencher.iter(|| {
        let p = ZZ.mul_ref(&x, &y);
        assert!(ZZ.eq_el(&z, &p));
    })
}

#[test]
fn test_is_zero() {
    let zero = ZZ.zero();
    let mut nonzero = ZZ.one();
    ZZ.mul_pow_2(&mut nonzero, 83124);
    assert!(ZZ.is_zero(&zero));
    assert!(ZZ.is_zero(&ZZ.negate(zero)));
    assert!(!ZZ.is_zero(&nonzero));
    assert!(!ZZ.is_zero(&ZZ.negate(nonzero)));
}

#[test]
fn test_cmp() {
    assert_eq!(true, ZZ.is_lt(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(2)));
    assert_eq!(true, ZZ.is_lt(&ZZ.int_hom().map(1), &ZZ.int_hom().map(2)));
    assert_eq!(false, ZZ.is_lt(&ZZ.int_hom().map(2), &ZZ.int_hom().map(2)));
    assert_eq!(false, ZZ.is_lt(&ZZ.int_hom().map(3), &ZZ.int_hom().map(2)));
    assert_eq!(true, ZZ.is_gt(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(-2)));
}

#[test]
fn test_get_uniformly_random() {
    crate::integer::generic_tests::test_integer_get_uniformly_random(ZZ);

    let ring = ZZ;
    let bound = RustBigintRing::RING.get_ring().parse("11000000000000000", 16).unwrap();
    let block_bound = RustBigintRing::RING.get_ring().parse("10000000000000000", 16).unwrap();
    let mut rng = oorandom::Rand64::new(0);
    let elements: Vec<_> = (0..1000).map(|_| ring.get_uniformly_random(&bound, || rng.rand_u64())).collect();
    assert!(elements.iter().any(|x| ring.is_lt(x, &block_bound)));
    assert!(elements.iter().any(|x| ring.is_gt(x, &block_bound)));
    assert!(elements.iter().all(|x| ring.is_lt(x, &bound)));
}

#[test]
fn test_canonical_iso_static_int() {
    // for the hom test, we have to be able to multiply elements in `StaticRing::<i128>::RING`, so we cannot test `i128::MAX` or `i128::MIN`
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, ZZ, [0, 1, -1, -100, 100, i64::MAX as i128, i64::MIN as i128].iter().copied());
    crate::ring::generic_tests::test_iso_axioms(StaticRing::<i128>::RING, ZZ, [0, 1, -1, -100, 100, i64::MAX as i128, i64::MIN as i128, i128::MAX, i128::MIN].iter().copied());
}

#[test]
fn test_serialize() {
    crate::serialization::generic_tests::test_serialization(ZZ, edge_case_elements())
}