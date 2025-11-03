use serde::de::{self, DeserializeSeed, Visitor};
use serde::Deserializer;
use tracing::instrument;

use crate::seq::*;

use core::fmt;
use std::cmp::{Ordering, max};
use std::alloc::Allocator;

type BlockInt = u64;
type DoubleBlockInt = u128;
const BLOCK_BITS: u32 = u64::BITS;

fn expand<A: Allocator>(x: &mut Vec<BlockInt, A>, len: usize) {
	if len > x.len() {
		x.resize(len, 0)
	}
}

#[cfg(test)]
fn truncate_zeros(mut x: Vec<BlockInt>) -> Vec<BlockInt> {
	x.truncate(x.len() - x.iter().rev().take_while(|a| **a == 0).count());
	return x;
}

#[stability::unstable(feature = "enable")]
#[inline(always)]
pub fn effective_length(x: &[BlockInt]) -> usize {
	for i in (0..x.len()).rev() {
		if x[i] != 0 {
			return i + 1;
		}
	}
	return 0;
}

#[inline(always)]
fn core_fma_small_ref_fst<A: Allocator>(lhs: &[BlockInt], factor: BlockInt, summand: &mut Vec<u64, A>, shift: u32) {
	let block_offset = (shift / BLOCK_BITS) as usize;
	let remaining_shift = shift % BLOCK_BITS;

	let mut implementation = || {
		// a little trick: using `wrapping_sub` will result in an out-of-bounds index if
		// signed arithmetic would result in a negative index
		assert!(lhs.len() as u128 + block_offset as u128 + 1 < usize::MAX as u128);
		let lhs_shifted_block = |i: usize| (*lhs.get(i.wrapping_sub(block_offset)).unwrap_or(&0) >> remaining_shift) | lhs.get((i + 1).wrapping_sub(block_offset)).unwrap_or(&0).checked_shl(BLOCK_BITS - remaining_shift).unwrap_or(0);

		let up_to = max(effective_length(lhs.as_ref()) + block_offset + 1, effective_length(summand));
		summand.truncate(up_to + 1);
		expand(summand, up_to + 1);
		let mut buffer: u64 = 0;
		for i in block_offset..up_to {
			let prod = lhs_shifted_block(i) as u128 * factor as u128 + buffer as u128 + summand[i] as u128;
			summand[i] = (prod & ((1u128 << BLOCK_BITS) - 1)) as u64;
			buffer = (prod >> BLOCK_BITS) as u64;
		}
		summand[up_to] = buffer;
		let result_len = effective_length(&summand);
		debug_assert!(summand.len() <= result_len + 2);
		summand.truncate(result_len);
	};

	// allow the compiler to produce specialized code for the case remaining_shift == 0
	if factor == 0 {
		// do nothing
	} else if remaining_shift == 0 {
		implementation();
	} else {
		implementation();
	}
}

#[inline(always)]
pub fn core_fma_small_ref_snd<A: Allocator>(lhs: &mut Vec<u64, A>, factor: BlockInt, summand: &[u64], shift: usize) {
	unimplemented!()
}

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_add_small<A: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: BlockInt) {
	if lhs.len() > 0 {
		let (sum, mut buffer) = lhs[0].overflowing_add(rhs);
		*lhs.at_mut(0) = sum;
		let mut i = 1;
		while buffer {
			expand(lhs, i + 1);
			let (sum, overflow) = lhs[i].overflowing_add(1);
			buffer = overflow;
			*lhs.at_mut(i) = sum;
			i += 1;
		}
	} else {
		expand(lhs, 1);
		*lhs.at_mut(0) = rhs;
	}
}
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_cmp(lhs: &[BlockInt], rhs: &[BlockInt]) -> Ordering {
	let llen = effective_length(lhs);
	let rlen = effective_length(rhs);
	match llen.cmp(&rlen) {
		Ordering::Less => Ordering::Less,
		Ordering::Greater => Ordering::Greater,
		Ordering::Equal => {
			for i in (0..llen).rev() {
				match lhs[i].cmp(&rhs[i]) {
					Ordering::Less => return Ordering::Less,
					Ordering::Greater => return Ordering::Greater,
					_ => {}
				}
			}
			return Ordering::Equal;
		}
	}
}

#[stability::unstable(feature = "enable")]
pub fn bigint_cmp_small(lhs: &[BlockInt], rhs: DoubleBlockInt) -> Ordering {
	bigint_cmp(lhs, &[(rhs & ((1 << BLOCK_BITS) - 1)) as BlockInt, (rhs >> BLOCK_BITS) as BlockInt])
}


#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_add<A: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: &[BlockInt], shift: u32) {
	core_fma_small_ref_fst(rhs, 1, lhs, shift);
}

///
/// Calculate lhs -= rhs * 2**shift
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_sub(lhs: &mut [BlockInt], rhs: &[BlockInt], shift: u32) -> Result<(), ()> {
	unimplemented!()
}

///
/// Calculate lhs = rhs - lhs
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_sub_self<A: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: &[BlockInt]) -> Result<(), ()> {
	unimplemented!()
}

///
/// Computes `summand := lhs * factor * 2**shift + summand`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_fma_small_ref_fst<A: Allocator>(lhs: &[BlockInt], factor: BlockInt, summand: &mut Vec<u64, A>, shift: u32) {
	core_fma_small_ref_fst(lhs, factor, summand, shift);
}

///
/// Computes `lhs := lhs * factor * 2**shift + summand`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_fma_small_ref_snd<A: Allocator>(lhs: &mut Vec<u64, A>, factor: BlockInt, summand: &[u64], shift: usize) {
	core_fma_small_ref_snd(lhs, factor, summand, shift);
}

///
/// Computes `lhs := lhs << power`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_lshift<A: Allocator>(lhs: &mut Vec<BlockInt, A>, power: usize) {
	core_fma_small_ref_snd(lhs, 1, &[], power);
}

///
/// Computes `lhs := lhs >> power`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_rshift(lhs: &mut [BlockInt], power: usize) {
	unimplemented!()
}

///
/// Computes `summand + lhs * rhs`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_fma<A: Allocator>(lhs: &[BlockInt], rhs: &[BlockInt], mut summand: Vec<BlockInt, A>) -> Vec<BlockInt, A> {
	summand.resize(max(effective_length(lhs) + effective_length(rhs), effective_length(summand.as_ref())), 0);
	for i in 0..effective_length(rhs) {
		core_fma_small_ref_fst(lhs, rhs[i], &mut summand, BLOCK_BITS * i as u32);
	}
	let result_len = effective_length(&summand);
	debug_assert!(summand.len() <= result_len + 2);
	summand.truncate(result_len);
	return summand;
}

///
/// Same as division_step, but for self_high == rhs_high == d
/// 
#[instrument(skip_all, level = "trace")]
fn division_step_last<A: Allocator>(lhs: &mut [BlockInt], rhs: &[BlockInt], d: usize, tmp: &mut Vec<BlockInt, A>) -> u64 {
	assert!(lhs[d] != 0);
	assert!(rhs[d] != 0);

	let self_high_blocks: u128 = ((lhs[d] as u128) << BLOCK_BITS) | (lhs[d - 1] as u128);
	let rhs_high_blocks: u128 = ((rhs[d] as u128) << BLOCK_BITS) | (rhs[d - 1] as u128);

	if rhs_high_blocks == u128::MAX {
		if bigint_cmp(lhs, rhs) != Ordering::Less {
			let sub_result = bigint_sub(lhs, rhs, 0);
			debug_assert!(sub_result.is_ok());
			return 1;
		} else {
			return 0;
		}
	} else {
		let mut quotient = (self_high_blocks / (rhs_high_blocks + 1)) as u64;
		tmp.clear();
		core_fma_small_ref_fst(rhs, quotient, tmp, 0);
		let sub_result = bigint_sub(lhs, tmp.as_ref(), 0);
		debug_assert!(sub_result.is_ok());

		if bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Less {
			let sub_result = bigint_sub(lhs, rhs.as_ref(), 0);
			debug_assert!(sub_result.is_ok());
			quotient += 1;
		}
		if bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Less {
			let sub_result = bigint_sub(lhs, rhs.as_ref(), 0);
			debug_assert!(sub_result.is_ok());
			quotient += 1;
		}
		
		debug_assert!(bigint_cmp(lhs.as_ref(), rhs) == Ordering::Less);
		return quotient;
	}
}

///
/// Finds some integer d such that subtracting d * rhs from self clears the top
/// block of self. self will be assigned the value after the subtraction and d
/// will be returned as d = (u * 2**block_bits + l) * 2**(k * block_bits) 
/// where the return value is (u, l, k)
/// 
/// Complexity O(log(n))
/// 
#[instrument(skip_all, level = "trace")]
fn division_step<A: Allocator>(
	lhs: &mut [BlockInt], 
	rhs: &[BlockInt], 
	lhs_high: usize, 
	rhs_high: usize, 
	tmp: &mut Vec<BlockInt, A>
) -> (u64, u64, usize) {
	assert!(lhs_high > rhs_high);
	assert!(lhs[lhs_high] != 0);
	assert!(rhs[rhs_high] != 0);

	// the basic idea is as follows:
	// we know that for a and b, have a - a//(b+1) * b <= b + a/(b+1)
	// Hence, we perform two steps:
	//  - by choosing a and b as the top two blocks of lhs resp. lhs, achieve that a - a//(b+1) * b <= b + 2^k
	//    (where k = BLOCK_BITS); hence, lhs - a//(b+1) * rhs <= rhs + 2^k, and so possibly subtracting rhs
	//    achieves new_lhs <= rhs
	//  - by choosing a as the top two blocks and b as only the top block of lhs resp. lhs (now b < 2^k), achieve
	//    that lhs - a//(b+1) * rhs < 2^k + 2^k = 2 * 2^k, and so after possibly subtracting rhs we find
	//    that the top block of lhs is cleared

	let mut result_upper = 0;
	let mut result_lower = 0;

	// first step
	{
		let lhs_high_blocks = ((lhs[lhs_high] as DoubleBlockInt) << BLOCK_BITS) | (lhs[lhs_high - 1] as DoubleBlockInt);
		let rhs_high_blocks = ((rhs[rhs_high] as DoubleBlockInt) << BLOCK_BITS) | (rhs[rhs_high - 1] as DoubleBlockInt);

		if rhs_high_blocks != DoubleBlockInt::MAX && lhs_high_blocks >= (rhs_high_blocks + 1) {
			let mut quotient = (lhs_high_blocks / (rhs_high_blocks + 1)) as u64;
			debug_assert!(quotient != 0);
			tmp.clear();
			core_fma_small_ref_fst(rhs.as_ref(), quotient, tmp, 0);
			let sub_result = bigint_sub(lhs, tmp.as_ref(), (lhs_high - rhs_high).try_into().unwrap());
			debug_assert!(sub_result.is_ok());

			let lhs_high_blocks = ((lhs[lhs_high] as DoubleBlockInt) << BLOCK_BITS) | (lhs[lhs_high - 1] as DoubleBlockInt);

			if lhs_high_blocks > rhs_high_blocks {
				let sub_result = bigint_sub(lhs, rhs.as_ref(), (lhs_high - rhs_high).try_into().unwrap());
				debug_assert!(sub_result.is_ok());
				quotient += 1;
			}
			result_upper = quotient;
		}

		// this is what we wanted to achieve in the first step
		let lhs_high_blocks = ((lhs[lhs_high] as DoubleBlockInt) << BLOCK_BITS) | (lhs[lhs_high - 1] as DoubleBlockInt);
		debug_assert!(lhs_high_blocks <= rhs_high_blocks);
	}

	// second step
	{
		let lhs_high_blocks = ((lhs[lhs_high] as DoubleBlockInt) << BLOCK_BITS) | (lhs[lhs_high - 1] as DoubleBlockInt);
		let rhs_high_block = rhs[rhs_high] as DoubleBlockInt;

		if lhs[lhs_high] != 0 {
			let mut quotient = (lhs_high_blocks / (rhs_high_block + 1)) as BlockInt;
			tmp.clear();
			core_fma_small_ref_fst(rhs.as_ref(), quotient, tmp, 0);
			let sub_result = bigint_sub(lhs, tmp.as_ref(), (lhs_high - rhs_high - 1).try_into().unwrap());
			debug_assert!(sub_result.is_ok());

			if lhs[lhs_high] != 0 {
				let sub_result = bigint_sub(lhs, rhs.as_ref(), (lhs_high - rhs_high - 1).try_into().unwrap());
				debug_assert!(sub_result.is_ok());
				quotient += 1;
			}
			result_lower = quotient;
		}

		debug_assert!(lhs[lhs_high] == 0);
	}
	return (result_upper, result_lower, lhs_high - rhs_high - 1);
}

///
/// Calculates abs(self) = abs(self) % abs(rhs) and returns the quotient
/// of the division abs(self) / abs(rhs).
/// 
/// Complexity O(log(n)^2)
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_div<A: Allocator, A2: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: &[BlockInt], mut out: Vec<BlockInt, A>, scratch_alloc: A2) -> Vec<BlockInt, A> {
	assert!(effective_length(rhs) > 0);

	out.clear();
	if effective_length(rhs) == 1 {
		let rem = bigint_div_small(lhs, rhs[0]);
		let lhs_len = effective_length(lhs);
		for i in 0..lhs_len {
			out.push(lhs[i]);
			lhs[i] = 0;
		}
		expand(lhs, 1);
		lhs[0] = rem;
		lhs.truncate(effective_length(lhs));
		return out;
	} else {
		let mut lhs_len = effective_length(lhs);
		let rhs_len = effective_length(rhs);
		let mut tmp = Vec::new_in(scratch_alloc);
		expand(&mut out, (lhs_len + 1).saturating_sub(rhs_len));
		while lhs_len > 0 && lhs_len > rhs_len {
			if lhs[lhs_len - 1] != 0 {
				let (quo_upper, quo_lower, quo_power) = division_step(lhs, rhs.as_ref(), lhs_len - 1, rhs_len - 1, &mut tmp);
				out[quo_power] = quo_lower;
				bigint_add(&mut out, &[quo_upper][..], BLOCK_BITS * (quo_power as u32 + 1));
				debug_assert!(lhs[lhs_len - 1] == 0);
			}
			lhs_len -= 1;
		}
		debug_assert_eq!(lhs_len, rhs_len);
		let quo = if lhs[lhs_len - 1] != 0 {
			division_step_last(lhs, rhs, lhs_len - 1, &mut tmp)
		} else {
			0
		};
		bigint_add(&mut out, &[quo], 0);
		lhs.truncate(effective_length(lhs));
		return out;
	}
}

///
/// Calculates self /= divisor and returns the remainder of the division.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn bigint_div_small(lhs: &mut [BlockInt], rhs: BlockInt) -> BlockInt {
	assert!(rhs != 0);
	let lhs_len = effective_length(lhs);
	if lhs_len == 0 {
		return 0;
	} else {
		let (quo, rem) = (lhs[lhs_len - 1] / rhs, lhs[lhs_len - 1] % rhs);
		let mut buffer = rem as DoubleBlockInt;
		lhs[lhs_len - 1] = quo;
		for i in (0..(lhs_len - 1)).rev() {
			buffer = (buffer << BLOCK_BITS) | (lhs[i] as DoubleBlockInt);
			let (quo, rem) = (buffer / rhs as DoubleBlockInt, buffer % rhs as DoubleBlockInt);
			debug_assert!(quo <= BlockInt::MAX as DoubleBlockInt);
			lhs[i] = quo as BlockInt;
			buffer = rem;
		}
		return buffer as BlockInt;
	}
}

///
/// Deserializes a 2-element tuple, consisting of a sign bit and a list of bytes
/// in little endian order to represent a number.
/// The list of bytes is converted into a `T` by the given closure, and the resulting
/// tuple is returned.
/// 
/// The main difference between using this function and `<(bool, &serde_bytes::Bytes)>::deserialize`
/// is that this function can accept a byte array with a shorter lifetime.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn deserialize_bigint_from_bytes<'de, D, F, T>(deserializer: D, from_bytes: F) -> Result<(bool, T), D::Error>
	where D: Deserializer<'de>,
		F: FnOnce(&[u8]) -> T
{
	struct ResultVisitor<F, T>
		where F: FnOnce(&[u8]) -> T
	{
		from_bytes: F
	}
	impl<'de, F, T> Visitor<'de> for ResultVisitor<F, T>
		where F: FnOnce(&[u8]) -> T
	{
		type Value = (bool, T);

		fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
			write!(f, "a sign bit as `bool` and a list of bytes in little endian order")
		}

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where A: serde::de::SeqAccess<'de>
        {
			let is_negative = seq.next_element()?;
			if is_negative.is_none() {
				return Err(de::Error::invalid_length(0, &"expected a sign bit as `bool`" as &'static dyn de::Expected));
			}
			let is_negative: bool = is_negative.unwrap();

			struct BytesVisitor<F, T>
				where F: FnOnce(&[u8]) -> T
			{
				from_bytes: F
			}
			impl<'de, F, T> Visitor<'de> for BytesVisitor<F, T>
				where F: FnOnce(&[u8]) -> T
			{
				type Value = T;

				fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
					write!(f, "a list of bytes in little endian order")
				}

				fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
					where E: de::Error,
				{
					Ok((self.from_bytes)(v))
				}
			}
			struct DeserializeBytes<F, T>
				where F: FnOnce(&[u8]) -> T
			{
				from_bytes: F
			}
			impl<'de, F, T> DeserializeSeed<'de> for DeserializeBytes<F, T>
				where F: FnOnce(&[u8]) -> T
			{
				type Value = T;

				fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
					where D: Deserializer<'de>
				{
					deserializer.deserialize_bytes(BytesVisitor { from_bytes: self.from_bytes })
				}
			}

			let data = seq.next_element_seed(DeserializeBytes { from_bytes: self.from_bytes })?;
			if data.is_none() {
				return Err(de::Error::invalid_length(1, &"expected a representation of the number as list of little endian bytes" as &'static dyn de::Expected));
			}
			let data: T = data.unwrap();

            return Ok((is_negative, data));
        }
	}
	return deserializer.deserialize_tuple(2, ResultVisitor { from_bytes: from_bytes });
}

#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[cfg(test)]
fn parse(s: &str, base: u32) -> Vec<BlockInt> {
    use crate::rings::rust_bigint::RustBigintRing;
	use crate::integer::*;
	use crate::ring::*;

	truncate_zeros(RustBigintRing::RING.get_ring().abs_base_u64_repr(&RustBigintRing::RING.parse(s, base).unwrap()).collect::<Vec<_>>())
}

#[test]
fn test_sub() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("923645871236598172365987287530543", 10);
    let y = parse("58430657823473456743684735863478", 10);
    let z = parse("865215213413124715622302551667065", 10);
    bigint_sub(&mut x, &y, 0).unwrap();
    assert_eq!(truncate_zeros(z), truncate_zeros(x));

    let x = parse("4294836225", 10);
    let mut y = parse("4294967297", 10);
    let z = parse("131072", 10);
    bigint_sub(&mut y, &x, 0).unwrap();
    assert_eq!(truncate_zeros(y), truncate_zeros(z));
}

#[test]
fn test_sub_with_carry() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("1000000000000000000", 16);
    let y = parse("FFFFFFFFFFFFFFFF00", 16);
    bigint_sub(&mut x, &y, 0).unwrap();
    assert_eq!(vec![256], truncate_zeros(x));
}

#[test]
fn test_add() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("923645871236598172365987287530543", 10);
    let y = parse("58430657823473456743684735863478", 10);
    let z = parse("982076529060071629109672023394021", 10);
    bigint_add(&mut x, &y, 0);
    assert_eq!(truncate_zeros(z), truncate_zeros(x));
}

#[test]
fn test_add_with_carry() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("1BC00000000000000BC", 16);
    let y =  parse("FFFFFFFFFFFFFFFF0000000000000000BC", 16);
    let z = parse("10000000000000000BC0000000000000178", 16);
    bigint_add(&mut x, &y, 0);
    assert_eq!(truncate_zeros(z), truncate_zeros(x));
}

#[test]
fn test_mul() {
    LogAlgorithmSubscriber::init_test();
    let x = parse("57873674586797895671345345", 10);
    let y = parse("21308561789045691782534873921650342768903561413264128756389247568729346542359871235465", 10);
    let z = parse("1233204770891906354921751949503652431220138020953161094405729272872607166072371117664593787957056214903826660425", 10);
    assert_eq!(truncate_zeros(z), truncate_zeros(bigint_fma(&x, &y, Vec::new())));
}

#[test]
fn test_fma() {
    LogAlgorithmSubscriber::init_test();
    let x = parse("543929578293075482904560982347609823468792", 10);
    let y = parse("598147578092315980234089723484389243859743", 10);
    let a = parse("98734435342", 10);
    let z = parse("325350159908777866468983871740437853305599423427707736569559476320508903561720075798", 10);
    assert_eq!(truncate_zeros(z), truncate_zeros(bigint_fma(&x, &y, a)));
}

#[test]
fn test_div_no_remainder() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("578435387FF0582367863200000000000000000000", 16);
    let y = parse("200000000000000000000", 16);
    let z = parse("2BC21A9C3FF82C11B3C319", 16);
    let quotient = bigint_div(&mut x, &y, Vec::new(), Global);
    assert_eq!(Vec::<BlockInt>::new(), truncate_zeros(x));
    assert_eq!(truncate_zeros(z), truncate_zeros(quotient));
}

#[test]
fn test_div_with_remainder() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("578435387FF0582367863200000000007651437856", 16);
    let y = parse("200000000000000000000", 16);
    let z = parse("2BC21A9C3FF82C11B3C319", 16);
    let r = parse("7651437856", 16);
    let quotient = bigint_div(&mut x, &y, Vec::new(), Global);
    assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(z), truncate_zeros(quotient));
}

#[test]
fn test_div_big() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("581239456149785691238569872349872348569871269871234657986123987237865847935698734296434575367565723846982523852347", 10);
    let y = parse("903852718907268716125180964783634518356783568793426834569872365791233387356325", 10);
    let q = parse("643068769934649368349591185247155725", 10);
    let r = parse("265234469040774335115597728873888165088018116561138613092906563355599185141722", 10);
    let actual = bigint_div(&mut x, &y, Vec::new(), Global);
    assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(q), truncate_zeros(actual));

	let mut x = vec![0, 0, 0, 0, 1];
	let y = parse("170141183460469231731687303715884105727", 10);
	let q = parse("680564733841876926926749214863536422916", 10);
	let r = vec![4];
	let actual = bigint_div(&mut x, &y, Vec::new(), Global);
	assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(q), truncate_zeros(actual));
}

#[test]
fn test_div_last_block_overflow() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("3227812347608635737069898965003764842912132241036529391038324195675809527521051493287056691600172289294878964965934366720", 10);
    let y = parse("302231454903657293676544", 10);
    let q = parse("10679935179604550411975108530847760573013522611783263849735208039111098628903202750114810434682880", 10);
    let quotient = bigint_div(&mut x, &y, Vec::new(), Global);
    assert_eq!(truncate_zeros(q), truncate_zeros(quotient));
    assert_eq!(Vec::<BlockInt>::new(), truncate_zeros(x));
}

#[test]
fn test_div_small() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("891023591340178345678931246518793456983745682137459364598623489512389745698237456890239238476873429872346579", 10);
    let q = parse("255380794307875708133829534685810678413226048190730686328066348384175908769916152734376393945793473738133", 10);
    _ = bigint_div_small(&mut x, 3489);
    assert_eq!(truncate_zeros(q), truncate_zeros(x));
}

#[test]
fn test_bigint_rshift() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("9843a756781b34567f81394", 16);
    let z = parse("9843a756781b34567", 16);
	bigint_rshift(&mut x, 24);
    assert_eq!(truncate_zeros(x), truncate_zeros(z));

    let mut x = parse("9843a756781b34567f81394", 16);
	bigint_rshift(&mut x, 1000);
    assert_eq!(truncate_zeros(x), Vec::<u64>::new());
}

#[test]
fn test_bigint_lshift() {
    LogAlgorithmSubscriber::init_test();
    let mut x = parse("2", 10);
	bigint_lshift(&mut x, 0);
    assert_eq!(parse("2", 10), truncate_zeros(x));

    let mut x = parse("4829192", 10);
	bigint_lshift(&mut x, 3);
    assert_eq!(parse("38633536", 10), truncate_zeros(x));

    let mut x = parse("4829192", 10);
	bigint_lshift(&mut x, 64);
    assert_eq!(parse("89082868906805576987574272", 10), truncate_zeros(x));
}

#[test]
fn test_core_fma_small() {
    LogAlgorithmSubscriber::init_test();
	let x = parse("10000000000000000000000000000000", 10);
	let mut z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_fst(&x, 42, &mut z, 0);
	assert_eq!(parse("520000000000000000000000000000000", 10), truncate_zeros(z));

	let mut x = parse("10000000000000000000000000000000", 10);
	let z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_snd(&mut x, 42, &z, 0);
	assert_eq!(parse("520000000000000000000000000000000", 10), truncate_zeros(x));

	let x = parse("10000000000000000000000000000000", 10);
	let mut z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_fst(&x, 42, &mut z, 1);
	assert_eq!(parse("940000000000000000000000000000000", 10), truncate_zeros(z));

	let mut x = parse("10000000000000000000000000000000", 10);
	let z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_snd(&mut x, 42, &z, 1);
	assert_eq!(parse("940000000000000000000000000000000", 10), truncate_zeros(x));
	
	let x = parse("10000000000000000000000000000000", 10);
	let mut z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_fst(&x, 42, &mut z, 63);
	assert_eq!(parse("92233720368547758180000000000000000000000000000000", 10), truncate_zeros(z));

	let mut x = parse("10000000000000000000000000000000", 10);
	let z = parse("100000000000000000000000000000000", 10);
	core_fma_small_ref_snd(&mut x, 42, &z, 63);
	assert_eq!(parse("92233720368547758180000000000000000000000000000000", 10), truncate_zeros(x));
}
