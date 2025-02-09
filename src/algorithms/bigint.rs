use serde::de::{self, DeserializeSeed, Visitor};
use serde::Deserializer;

use crate::seq::*;

use core::fmt;
use std::cmp::{Ordering, min, max};
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

fn assign<A: Allocator>(x: &mut Vec<BlockInt, A>, rhs: &[BlockInt]) {
    x.clear();
    x.extend((0..rhs.len()).map(|i| rhs[i]))
}

#[stability::unstable(feature = "enable")]
pub fn bigint_add<A: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: &[BlockInt], block_offset: usize) {
	let prev_len = lhs.len();
	let mut buffer: bool = false;
	let mut i = 0;
	if let Some(rhs_d) = highest_set_block(rhs) {
		while i <= rhs_d || buffer {
			let rhs_val = *rhs.get(i).unwrap_or(&0);
			let j = i + block_offset;
			expand(lhs, j + 1);
			let (sum, overflow) = lhs[j].overflowing_add(rhs_val);
			if buffer {
				let (carry_sum, carry_overflow) = sum.overflowing_add(1);
				*lhs.at_mut(j) = carry_sum;
				buffer = overflow || carry_overflow;
			} else {
				*lhs.at_mut(j) = sum;
				buffer = overflow;
			}
			i += 1;
		}
	}
	let new_highest_set_block = highest_set_block(&lhs);
	debug_assert!(new_highest_set_block.is_none() || max(prev_len, new_highest_set_block.unwrap() + 1) == lhs.len());
}

#[stability::unstable(feature = "enable")]
pub fn highest_set_block(x: &[BlockInt]) -> Option<usize> {
	for i in (0..x.len()).rev() {
		if x[i] != 0 {
			return Some(i);
		}
	}
	return None;
}

#[stability::unstable(feature = "enable")]
pub fn bigint_cmp(lhs: &[BlockInt], rhs: &[BlockInt]) -> Ordering {
	match (highest_set_block(lhs.as_ref()), highest_set_block(rhs.as_ref())) {
		(None, None) => return Ordering::Equal,
		(Some(_), None) => return Ordering::Greater,
		(None, Some(_)) => return Ordering::Less,
		(Some(x), Some(y)) => match x.cmp(&y) {
			Ordering::Less => return Ordering::Less,
			Ordering::Greater => return Ordering::Greater,
			Ordering::Equal => {
				for i in (0..=x).rev() {
					match lhs[i].cmp(&rhs[i]) {
						Ordering::Less => return Ordering::Less,
						Ordering::Greater => return Ordering::Greater,
						_ => {}
					}
				}
				return Ordering::Equal;
			}
		}
	};
}

#[stability::unstable(feature = "enable")]
pub fn bigint_cmp_small(lhs: &[BlockInt], rhs: DoubleBlockInt) -> Ordering {
	match highest_set_block(lhs.as_ref()) {
	   None => 0.cmp(&rhs),
	   Some(0) => (lhs[0] as DoubleBlockInt).cmp(&rhs),
	   Some(1) => (((lhs[1] as DoubleBlockInt) << BLOCK_BITS) | (lhs[0] as DoubleBlockInt)).cmp(&rhs),
	   Some(_) => Ordering::Greater,
	}
}

///
/// Calculate lhs -= rhs * (1 << BLOCK_BITS)^block_offset
/// 
/// This will panic if the subtraction would result in a negative number
/// 
#[stability::unstable(feature = "enable")]
pub fn bigint_sub(lhs: &mut [BlockInt], rhs: &[BlockInt], block_offset: usize) {
	assert!(bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Less);

	if let Some(rhs_high) = highest_set_block(rhs.as_ref()) {
		let mut buffer: bool = false;
		let mut i = 0;
		while i <= rhs_high || buffer {
			let rhs_val = *rhs.get(i).unwrap_or(&0);
			let j = i + block_offset;
			debug_assert!(j < lhs.len());
			let (difference, overflow) = lhs[j].overflowing_sub(rhs_val);
			if buffer {
				let (carry_difference, carry_overflow) = difference.overflowing_sub(1);
				lhs[j] = carry_difference;
				buffer = overflow || carry_overflow;
			} else {
				lhs[j] = difference;
				buffer = overflow;
			}
			i += 1;
		}
	}
}

///
/// Calculate lhs = rhs - lhs
/// 
/// This will panic or give a wrong result if the subtraction would result in a negative number
/// 
#[stability::unstable(feature = "enable")]
pub fn bigint_sub_self<A: Allocator>(lhs: &mut Vec<BlockInt, A>, rhs: &[BlockInt]) {
	debug_assert!(bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Greater);

	let rhs_high = highest_set_block(rhs.as_ref()).expect("rhs must be larger than lhs");
	expand(lhs, rhs_high + 1);
	let mut buffer: bool = false;
	let mut i = 0;
	while i <= rhs_high {
		let (difference, overflow) = rhs[i].overflowing_sub(lhs[i]);
		if buffer {
			let (carry_difference, carry_overflow) = difference.overflowing_sub(1);
			lhs[i] = carry_difference;
			buffer = overflow || carry_overflow;
		} else {
			lhs[i] = difference;
			buffer = overflow;
		}
		i += 1;
	}
	assert!(!buffer);
}

#[stability::unstable(feature = "enable")]
pub fn bigint_lshift<A: Allocator>(lhs: &mut Vec<BlockInt, A>, power: usize) {
	if let Some(high) = highest_set_block(&lhs) {
		let mut buffer: BlockInt = 0;
		let mut i = 0;
		let in_block = (power % BlockInt::BITS as usize) as u32;
		if in_block != 0 {
			while i <= high || buffer != 0 {
				expand(lhs, i + 1);
				let tmp = lhs[i].overflowing_shr(BlockInt::BITS - in_block).0;
				lhs[i] = (lhs[i] << in_block) | buffer;
				buffer = tmp;
				i += 1;
			}
		}
		lhs.reverse();
		lhs.extend((0..(power / BlockInt::BITS as usize)).map(|_| 0));
		lhs.reverse();
	}
}

#[stability::unstable(feature = "enable")]
pub fn bigint_rshift(lhs: &mut [BlockInt], power: usize) {
	if let Some(high) = highest_set_block(lhs) {
		let mut buffer: BlockInt = 0;
		let in_block = (power % BlockInt::BITS as usize) as u32;
		let mut i = high as isize;
		if in_block != 0 {
			while i >= 0 {
				let tmp = lhs[i as usize].overflowing_shl(BlockInt::BITS - in_block).0;
				lhs[i as usize] = (lhs[i as usize] >> in_block) | buffer;
				buffer = tmp;
				i -= 1;
			}
		}
		let blocks = power / BlockInt::BITS as usize;
		if blocks != 0 {
			for i in 0..min(blocks, lhs.len()) {
				lhs[i] = 0;
			}
			for i in blocks..=high {
				lhs[i - blocks] = lhs[i];
				lhs[i] = 0;
			}
		}
	}
}

#[stability::unstable(feature = "enable")]
pub fn bigint_mul<A: Allocator>(lhs: &[BlockInt], rhs: &[BlockInt], mut out: Vec<BlockInt, A>) -> Vec<BlockInt, A> {
	out.clear();
	out.resize(
		highest_set_block(lhs.as_ref()).unwrap_or(0) + 
		highest_set_block(rhs.as_ref()).unwrap_or(0) + 2, 
		0
	);
	if let Some(d) = highest_set_block(rhs.as_ref()) {
		let mut val = Vec::new();
		for i in 0..=d {
			assign(&mut val, lhs.as_ref());
			bigint_mul_small(&mut val, rhs[i]);
			bigint_add(&mut out, val.as_ref(), i);
		}
	}
	debug_assert!(highest_set_block(&out).is_none() || highest_set_block(&out).unwrap() as isize >= out.len() as isize - 2);
	return out;
}

///
/// Complexity O(log(n))
/// 
#[stability::unstable(feature = "enable")]
pub fn bigint_mul_small<A: Allocator>(lhs: &mut Vec<BlockInt, A>, factor: BlockInt) {
	if let Some(d) = highest_set_block(lhs.as_ref()) {
		let mut buffer: u64 = 0;
		for i in 0..=d {
			let prod = lhs[i] as u128 * factor as u128 + buffer as u128;
			*lhs.at_mut(i) = (prod & ((1u128 << BLOCK_BITS) - 1)) as u64;
			buffer = (prod >> BLOCK_BITS) as u64;
		}
		expand(lhs, d + 2);
		*lhs.at_mut(d + 1) = buffer;
	}
}

#[stability::unstable(feature = "enable")]
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

///
/// Same as division_step, but for self_high == rhs_high == d
/// 
fn division_step_last<A: Allocator>(lhs: &mut [BlockInt], rhs: &[BlockInt], d: usize, tmp: &mut Vec<BlockInt, A>) -> u64 {
	assert!(lhs[d] != 0);
	assert!(rhs[d] != 0);

	let self_high_blocks: u128 = ((lhs[d] as u128) << BLOCK_BITS) | (lhs[d - 1] as u128);
	let rhs_high_blocks: u128 = ((rhs[d] as u128) << BLOCK_BITS) | (rhs[d - 1] as u128);

	if rhs_high_blocks == u128::MAX {
		if bigint_cmp(lhs, rhs) != Ordering::Less {
			bigint_sub(lhs, rhs, 0);
			return 1;
		} else {
			return 0;
		}
	} else {
		let mut quotient = (self_high_blocks / (rhs_high_blocks + 1)) as u64;
		assign(tmp, rhs.as_ref());
		bigint_mul_small(tmp, quotient);
		bigint_sub(lhs, tmp.as_ref(), 0);

		if bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Less {
			bigint_sub(lhs, rhs.as_ref(), 0);
			quotient += 1;
		}
		if bigint_cmp(lhs.as_ref(), rhs.as_ref()) != Ordering::Less {
			bigint_sub(lhs, rhs.as_ref(), 0);
			quotient += 1;
		}
		
		debug_assert!(bigint_cmp(lhs.as_ref(), rhs) == Ordering::Less);
		return quotient;
	}
}

///
/// Finds some integer d such that subtracting d * rhs from self clears the top
/// block of self. self will be assigned the value after the subtraction and d
/// will be returned as d = (u * 2 ^ block_bits + l) * 2 ^ (k * block_bits) 
/// where the return value is (u, l, k)
/// 
/// Complexity O(log(n))
/// 
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
	//  - by choosing a as the top two blocks and b as only the top block of lhs resp. lhs (now b < 2^k), achieve that
	//    lhs - a//(b+1) * rhs < 2^k + 2^k = 2 * 2^k, and so after possibly subtracting rhs we find
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
			assign(tmp, rhs.as_ref());
			bigint_mul_small(tmp, quotient);
			bigint_sub(lhs, tmp.as_ref(), lhs_high - rhs_high);

			let lhs_high_blocks = ((lhs[lhs_high] as DoubleBlockInt) << BLOCK_BITS) | (lhs[lhs_high - 1] as DoubleBlockInt);

			if lhs_high_blocks > rhs_high_blocks {
				bigint_sub(lhs, rhs.as_ref(), lhs_high - rhs_high);
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
			assign(tmp, rhs.as_ref());
			bigint_mul_small(tmp, quotient);
			bigint_sub(lhs, tmp.as_ref(), lhs_high - rhs_high - 1);

			if lhs[lhs_high] != 0 {
				bigint_sub(lhs, rhs.as_ref(), lhs_high - rhs_high - 1);
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
/// of the division abs(self) / abs(rhs). The sign bit of self is ignored
/// and left unchanged.
/// 
/// Complexity O(log(n)^2)
/// 
#[stability::unstable(feature = "enable")]
pub fn bigint_div<A: Allocator>(lhs: &mut [BlockInt], rhs: &[BlockInt], mut out: Vec<BlockInt, A>) -> Vec<BlockInt, A> {
	assert!(highest_set_block(rhs.as_ref()).is_some());
	
	out.clear();
	match (highest_set_block(lhs), highest_set_block(rhs)) {
		(_, None) => panic!("division by zero"),
		(None, Some(_)) => return out,
		(Some(d), Some(k)) if d < k => return out,
		(Some(d), Some(k)) if k == 0 => {
			let rem = bigint_div_small(lhs, rhs[0]);
			for i in 0..=d {
				out.push(lhs[i]);
				lhs[i] = 0;
			}
			lhs[0] = rem;
			return out;
		},
		(Some(mut d), Some(k)) => {
			let mut tmp = Vec::new();
			expand(&mut out, d - k + 1);
			while d > k {
				if lhs[d] != 0 {
					let (quo_upper, quo_lower, quo_power) = division_step(lhs, rhs.as_ref(), d, k, &mut tmp);
					*out.at_mut(quo_power) = quo_lower;
					bigint_add(&mut out, &[quo_upper][..], quo_power + 1);
					debug_assert!(lhs[d] == 0);
				}
				d -= 1;
			}
			let quo = if lhs[d] != 0 {
				division_step_last(lhs, rhs, d, &mut tmp)
			} else {
				0
			};
			bigint_add(&mut out, &[quo], 0);
			return out;
		}
	}
}

///
/// Calculates self /= divisor and returns the remainder of the division.
/// 
#[stability::unstable(feature = "enable")]
pub fn bigint_div_small(lhs: &mut [BlockInt], rhs: BlockInt) -> BlockInt {
	assert!(rhs != 0);
	let highest_block_opt = highest_set_block(lhs.as_ref());
	if highest_block_opt == Some(0) {
		let (quo, rem) = (lhs[0] / rhs, lhs[0] % rhs);
		*lhs.at_mut(0) = quo;
		return rem;
	} else if let Some(highest_block) = highest_block_opt {
		let (quo, rem) = (lhs[highest_block] / rhs, lhs[highest_block] % rhs);
		let mut buffer = rem as DoubleBlockInt;
		*lhs.at_mut(highest_block) = quo;
		for i in (0..highest_block).rev() {
			buffer = (buffer << BLOCK_BITS) | (lhs[i] as DoubleBlockInt);
			let (quo, rem) = (buffer / rhs as DoubleBlockInt, buffer % rhs as DoubleBlockInt);
			debug_assert!(quo <= BlockInt::MAX as DoubleBlockInt);
			*lhs.at_mut(i) = quo as BlockInt;
			buffer = rem;
		}
		return buffer as BlockInt;
	} else {
		return 0;
	}
}

#[stability::unstable(feature = "enable")]
pub fn from_radix<A: Allocator, I: Iterator<Item = Result<u64, E>>, E>(data: I, base: u64, mut out: Vec<BlockInt, A>) -> Result<Vec<BlockInt, A>, E> {
	out.clear();
	for value in data {
		let val = value?;
		debug_assert!(val < base);
		bigint_mul_small(&mut out, base);
		bigint_add_small(&mut out, val);
	}
	return Ok(out);
}

#[stability::unstable(feature = "enable")]
pub fn from_str_radix<A: Allocator>(string: &str, base: u32, out: Vec<BlockInt, A>) -> Result<Vec<BlockInt, A>, ()> {
	assert!(base >= 2);
	// we need the -1 in BLOCK_BITS to ensure that base^chunk_size is 
	// really smaller than 2^64
	let chunk_size = ((BLOCK_BITS - 1) as f32 / (base as f32).log2()).floor() as usize;
	let it = <str as AsRef<[u8]>>::as_ref(string)
		.rchunks(chunk_size)
		.rev()
		.map(std::str::from_utf8)
		.map(|chunk| chunk.map_err(|_| ()))
		.map(|chunk| chunk.and_then(|n| 
			u64::from_str_radix(n, base).map_err(|_| ()))
		);
	return from_radix::<A, _, ()>(it, (base as u64).pow(chunk_size as u32), out);
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
fn parse(s: &str) -> Vec<BlockInt> {
	from_str_radix(s, 10, Vec::new()).unwrap()
}

#[test]
fn test_sub() {
    let mut x = parse("923645871236598172365987287530543");
    let y = parse("58430657823473456743684735863478");
    let z = parse("865215213413124715622302551667065");
    bigint_sub(&mut x, &y, 0);
    assert_eq!(truncate_zeros(z), truncate_zeros(x));

    let x = parse("4294836225");
    let mut y = parse("4294967297");
    let z = parse("131072");
    bigint_sub(&mut y, &x, 0);
    assert_eq!(truncate_zeros(y), truncate_zeros(z));
}

#[test]
fn test_sub_with_carry() {
    let mut x = from_str_radix("1000000000000000000", 16, Vec::new()).unwrap();
    let y = from_str_radix("FFFFFFFFFFFFFFFF00", 16, Vec::new()).unwrap();
    bigint_sub(&mut x, &y, 0);
    assert_eq!(vec![256], truncate_zeros(x));
}

#[test]
fn test_add() {
    let mut x = parse("923645871236598172365987287530543");
    let y = parse("58430657823473456743684735863478");
    let z = parse("982076529060071629109672023394021");
    bigint_add(&mut x, &y, 0);
    assert_eq!(truncate_zeros(z), truncate_zeros(x));
}

#[test]
fn test_add_with_carry() {
    let mut x = from_str_radix("1BC00000000000000BC", 16, Vec::new()).unwrap();
    let y =  from_str_radix("FFFFFFFFFFFFFFFF0000000000000000BC", 16, Vec::new()).unwrap();
    let z = from_str_radix("10000000000000000BC0000000000000178", 16, Vec::new()).unwrap();
    bigint_add(&mut x, &y, 0);
    assert_eq!(truncate_zeros(z), truncate_zeros(x));
}

#[test]
fn test_mul() {
    let x = parse("57873674586797895671345345");
    let y = parse("21308561789045691782534873921650342768903561413264128756389247568729346542359871235465");
    let z = parse("1233204770891906354921751949503652431220138020953161094405729272872607166072371117664593787957056214903826660425");
    assert_eq!(truncate_zeros(z), truncate_zeros(bigint_mul(&x, &y, Vec::new())));
}

#[test]
fn test_div_no_remainder() {
    let mut x = from_str_radix("578435387FF0582367863200000000000000000000", 16, Vec::new()).unwrap();
    let y = from_str_radix("200000000000000000000", 16, Vec::new()).unwrap();
    let z = from_str_radix("2BC21A9C3FF82C11B3C319", 16, Vec::new()).unwrap();
    let quotient = bigint_div(&mut x, &y, Vec::new());
    assert_eq!(Vec::<BlockInt>::new(), truncate_zeros(x));
    assert_eq!(truncate_zeros(z), truncate_zeros(quotient));
}

#[test]
fn test_div_with_remainder() {
    let mut x = from_str_radix("578435387FF0582367863200000000007651437856", 16, Vec::new()).unwrap();
    let y = from_str_radix("200000000000000000000", 16, Vec::new()).unwrap();
    let z = from_str_radix("2BC21A9C3FF82C11B3C319", 16, Vec::new()).unwrap();
    let r = from_str_radix("7651437856", 16, Vec::new()).unwrap();
    let quotient = bigint_div(&mut x, &y, Vec::new());
    assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(z), truncate_zeros(quotient));
}

#[test]
fn test_div_big() {
    let mut x = parse("581239456149785691238569872349872348569871269871234657986123987237865847935698734296434575367565723846982523852347");
    let y = parse("903852718907268716125180964783634518356783568793426834569872365791233387356325");
    let q = parse("643068769934649368349591185247155725");
    let r = parse("265234469040774335115597728873888165088018116561138613092906563355599185141722");
    let actual = bigint_div(&mut x, &y, Vec::new());
    assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(q), truncate_zeros(actual));

	let mut x = vec![0, 0, 0, 0, 1];
	let y = parse("170141183460469231731687303715884105727");
	let q = parse("680564733841876926926749214863536422916");
	let r = vec![4];
	let actual = bigint_div(&mut x, &y, Vec::new());
	assert_eq!(truncate_zeros(r), truncate_zeros(x));
    assert_eq!(truncate_zeros(q), truncate_zeros(actual));
}

#[test]
fn test_div_last_block_overflow() {
    let mut x = parse("3227812347608635737069898965003764842912132241036529391038324195675809527521051493287056691600172289294878964965934366720");
    let y = parse("302231454903657293676544");
    let q = parse("10679935179604550411975108530847760573013522611783263849735208039111098628903202750114810434682880");
    let quotient = bigint_div(&mut x, &y, Vec::new());
    assert_eq!(truncate_zeros(q), truncate_zeros(quotient));
    assert_eq!(Vec::<BlockInt>::new(), truncate_zeros(x));
}

#[test]
fn test_div_small() {
    let mut x = parse("891023591340178345678931246518793456983745682137459364598623489512389745698237456890239238476873429872346579");
    let q = parse("255380794307875708133829534685810678413226048190730686328066348384175908769916152734376393945793473738133");
    _ = bigint_div_small(&mut x, 3489);
    assert_eq!(truncate_zeros(q), truncate_zeros(x));
}

#[test]
fn test_bigint_rshift() {
    let mut x = from_str_radix("9843a756781b34567f81394", 16, Vec::new()).unwrap();
    let z = from_str_radix("9843a756781b34567", 16, Vec::new()).unwrap();
	bigint_rshift(&mut x, 24);
    assert_eq!(truncate_zeros(x), truncate_zeros(z));

    let mut x = from_str_radix("9843a756781b34567f81394", 16, Vec::new()).unwrap();
	bigint_rshift(&mut x, 1000);
    assert_eq!(truncate_zeros(x), Vec::<u64>::new());
}

#[test]
fn test_bigint_lshift() {
    let mut x = parse("2");
	bigint_lshift(&mut x, 0);
    assert_eq!(parse("2"), truncate_zeros(x));

    let mut x = parse("4829192");
	bigint_lshift(&mut x, 3);
    assert_eq!(parse("38633536"), truncate_zeros(x));

    let mut x = parse("4829192");
	bigint_lshift(&mut x, 64);
    assert_eq!(parse("89082868906805576987574272"), truncate_zeros(x));
}
