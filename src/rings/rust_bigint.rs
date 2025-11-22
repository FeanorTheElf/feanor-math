use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use serde::de::{self, DeserializeSeed};
use serde::ser::SerializeTuple;
use serde::{Deserializer, Serialize, Deserialize, Serializer};
use tracing::instrument; 

use crate::algorithms::bigint_ops::*;
use crate::algorithms::eea::{eea, gcd, partial_eea_int};
use crate::divisibility::{DivisibilityRing, Domain};
use crate::pid::*;
use crate::{impl_interpolation_base_ring_char_zero, impl_poly_gcd_locally_for_ZZ, impl_eval_poly_locally_for_ZZ};
use crate::integer::*;
use crate::ordered::*;
use crate::primitive_int::*;
use crate::ring::*;
use crate::serialization::*;
use crate::specialization::*;
use crate::homomorphism::*;

use std::alloc::Allocator;
use std::alloc::Global;
use std::cmp::{Ordering, max, min};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{replace, swap};

///
/// An element of the integer ring implementation [`RustBigintRing`].
/// 
/// An object of this struct does represent an arbitrary-precision integer,
/// but it follows the general approach of `feanor-math` to expose its actual
/// behavior through the ring, and not through the elements. For example, to
/// add integers, use
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::rust_bigint::*;
/// const ZZ: RustBigintRing = RustBigintRing::RING;
/// let a: RustBigint = ZZ.add(ZZ.power_of_two(50), ZZ.power_of_two(100));
/// assert_eq!("1267650600228230527396610048000", format!("{}", ZZ.formatted_el(&a)));
/// ```
/// and not
/// ```compile_fail
/// assert_eq!("1267650600228230527396610048000", format!("{}", RustBigint::from(2).pow(100) + RustBigint::from(2).pow(50)));
/// ```
/// 
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
pub struct RustBigintRingBase<A: Allocator + Send + Sync + Clone = Global> {
    allocator: A
}

///
/// [`RingStore`] corresponding to [`RustBigintRingBase`].
/// 
pub type RustBigintRing<A = Global> = RingValue<RustBigintRingBase<A>>;

impl<A: Allocator + Send + Sync + Clone> RustBigintRing<A> {
    
    #[stability::unstable(feature = "enable")]
    pub fn new_with_alloc(allocator: A) -> RustBigintRing<A> {
        Self::from(RustBigintRingBase { allocator })
    }
}

impl RustBigintRing {
    
    ///
    /// Default instance of [`RustBigintRing`], the ring of arbitrary-precision integers.
    /// 
    pub const RING: RustBigintRing = RingValue::from(RustBigintRingBase { allocator: Global });
}

impl<A: Allocator + Send + Sync + Clone + Default> Default for RustBigintRingBase<A> {

    fn default() -> Self {
        RustBigintRingBase { allocator: A::default() }
    }
}

impl<A: Allocator + Send + Sync + Clone> RustBigintRingBase<A> {

    ///
    /// If the given big integer fits into a `i128`, this will be
    /// returned. Otherwise, `None` is returned.
    /// 
    pub fn map_i128(&self, val: &RustBigint<A>) -> Option<i128> {
        match effective_length(&val.1) {
            0 => Some(0),
            1 if val.0 => Some(-(val.1[0] as i128)),
            1 if !val.0 => Some(val.1[0] as i128),
            2 if val.0 => {
                let value = val.1[0] as u128 + ((val.1[1] as u128) << u64::BITS);
                if value == 1 << (u128::BITS - 1) {
                    Some(i128::MIN)
                } else {
                    i128::try_from(value).ok().map(|x| -x)
                }
            },
            2 if !val.0 => i128::try_from(val.1[0] as u128 + ((val.1[1] as u128) << u64::BITS)).ok(),
            _ => None
        }
    }

    ///
    /// Returns an iterator over the digits of the `2^64`-adic digit
    /// representation of the absolute value of the given element.
    /// 
    pub fn abs_base_u64_repr<'a>(&self, el: &'a RustBigint) -> impl 'a + Iterator<Item = u64> {
        el.1.iter().copied()
    }

    ///
    /// Interprets the elements of the iterator as digits in a `2^64`-adic
    /// digit representation, and returns the big integer represented by it.
    /// 
    pub fn from_base_u64_repr<I>(&self, data: I) -> RustBigint<A>
        where I: Iterator<Item = u64>
    {
        let mut result = Vec::with_capacity_in(data.size_hint().0, self.allocator.clone());
        result.extend(data);
        RustBigint(false, result)
    }

    ///
    /// Computes `lhs * rhs * 2**shift + summand`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn fma_i64_with_shift(&self, lhs: &RustBigint<A>, rhs: i64, mut summand: RustBigint<A>, shift: u32) -> RustBigint<A> {
        if (rhs < 0) ^ lhs.0 == summand.0 {
            bigint_fma_small_ref_fst(&lhs.1, rhs.unsigned_abs(), &mut summand.1, shift);
        } else {
            let negated = bigint_fms_small_ref_fst(&lhs.1, rhs.unsigned_abs(), &mut summand.1, shift).is_err();
            summand.0 ^= negated;
        }
        return summand;
    }
}

impl<A: Allocator + Send + Sync + Clone> Debug for RustBigintRingBase<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z")
    }
}

impl<A: Allocator + Send + Sync + Clone> PartialEq for RustBigintRingBase<A> {

    fn eq(&self, _other: &Self) -> bool {
        // it is perfectly valid to swap elements between two different `RustBigintRing`s,
        // even if they have different allocators. Every element keeps track of their allocator 
        // themselves
        true
    }
}

impl<A: Allocator + Send + Sync + Clone> RingBase for RustBigintRingBase<A> {
    
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
                bigint_add(lhs_val, rhs_val, 0);
            },
            (RustBigint(lhs_sgn, lhs_val), RustBigint(_, rhs_val)) => {
                let negated = bigint_sub(lhs_val, &rhs_val, 0).is_err();
                *lhs_sgn ^= negated;
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
        let result = bigint_fma(&lhs.1, &rhs.1, Vec::new_in(self.allocator.clone()));
        *lhs = RustBigint(lhs.0 ^ rhs.0, result);
    }

    fn fma(&self, lhs: &Self::Element, rhs: &Self::Element, summand: Self::Element) -> Self::Element {
        if lhs.0 ^ rhs.0 == summand.0 {
            let result = bigint_fma(&lhs.1, &rhs.1, summand.1);
            RustBigint(summand.0, result)
        } else {
            self.add(summand, self.mul_ref(lhs, rhs))
        }
    }

    fn mul_int(&self, mut lhs: Self::Element, rhs: i32) -> Self::Element {
        lhs.0 ^= rhs < 0;
        core_fma_small_ref_snd(&mut lhs.1, rhs.unsigned_abs().into(), &[], 0);
        return lhs;
    }

    fn zero(&self) -> Self::Element {
        RustBigint(false, Vec::new_in(self.allocator.clone()))
    }

    fn from_int(&self, value: i32) -> Self::Element {
        let mut data = Vec::with_capacity_in(1, self.allocator.clone());
        data.push(value.unsigned_abs() as u64);
        RustBigint(value < 0, data)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.0 == rhs.0 {
            bigint_cmp(&lhs.1, &rhs.1) == Ordering::Equal
        } else {
            self.is_zero(lhs) && self.is_zero(rhs)
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        effective_length(&value.1) == 0
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.0 == false && effective_length(&value.1) == 1 && value.1[0] == 1
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.0 == true && effective_length(&value.1) == 1 && value.1[0] == 1
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    
    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _: EnvBindingStrength) -> std::fmt::Result {
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
            effective_length(&value.1) * u64::BITS as usize / 3
        );
        while !self.is_zero(&copy) {
            let rem = bigint_div_small(&mut copy.1, BIG_POWER_TEN);
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

impl<A1: Allocator + Send + Sync + Clone, A2: Allocator + Send + Sync + Clone> IntCast<RustBigintRingBase<A2>> for RustBigintRingBase<A1> {
    
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
            impl<A: Allocator + Send + Sync + Clone> IntCast<StaticRingBase<$from>> for RustBigintRingBase<A> {

                fn cast(&self, _: &StaticRingBase<$from>, value: $from) -> RustBigint<A> {
                    let negative = value < 0;
                    let value = <_ as Into<i128>>::into(value).checked_abs().map(|x| x as u128).unwrap_or(1 << (u128::BITS - 1));
                    let mut result = Vec::with_capacity_in(2, self.allocator.clone());
                    result.extend([(value & ((1 << u64::BITS) - 1)) as u64, (value >> u64::BITS) as u64].into_iter());
                    RustBigint(negative, result)
                }
            }

            impl<A: Allocator + Send + Sync + Clone> IntCast<RustBigintRingBase<A>> for StaticRingBase<$from> {

                fn cast(&self, from: &RustBigintRingBase<A>, value: RustBigint<A>) -> $from {
                    <$from>::try_from(from.map_i128(&value).expect(concat!("integer does not fit into a ", stringify!($from)))).ok().expect(concat!("integer does not fit into a ", stringify!($from)))
                }
            }
        )*
    };
}

specialize_int_cast!{ i8, i16, i32, i64, i128 }

impl<A: Allocator + Send + Sync + Clone> Domain for RustBigintRingBase<A> {}

impl<A: Allocator + Send + Sync + Clone> OrderedRing for RustBigintRingBase<A> {

    fn cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> Ordering {
        match (lhs.0, rhs.0) {
            (true, true) => bigint_cmp(&rhs.1, &lhs.1),
            (false, false) => bigint_cmp(&lhs.1, &rhs.1),
            (_, _) if self.is_zero(lhs) && self.is_zero(rhs) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater
        }
    }

    fn abs_cmp(&self, lhs: &Self::Element, rhs: &Self::Element) -> Ordering {
        bigint_cmp(&lhs.1, &rhs.1)
    }
}

impl<A: Allocator + Send + Sync + Clone> DivisibilityRing for RustBigintRingBase<A> {
    
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

    fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        Some(elements.fold(self.zero(), |a, b| self.ideal_gen(&a, b)))
    }

    fn prepare_divisor(&self, _: &Self::Element) -> Self::PreparedDivisorData {
        ()
    }
}

impl<A: Allocator + Send + Sync + Clone> PrincipalIdealRing for RustBigintRingBase<A> {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }
    
    #[instrument(skip_all, level = "trace")]
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {

        let from_i64 = RingValue::from_ref(self).can_hom(&StaticRing::<i64>::RING).unwrap();

        let get_abs_head_bits = |x: &RustBigint<A>, shift: usize| {
            let block_idx = shift / 64;
            let remaining_shift = shift % 64;
            let result = if x.1.len() <= block_idx {
                0
            } else if x.1.len() == block_idx + 1 || remaining_shift == 0 {
                x.1[block_idx] >> remaining_shift
            } else {
                (x.1[block_idx] >> remaining_shift) | (x.1[block_idx + 1] << (64 - remaining_shift))
            };
            debug_assert!(result < (1 << 63));
            return result as i64;
        };

        let [mut a, mut b] = [self.clone_el(lhs), self.clone_el(rhs)];
        let [mut sa, mut ta, mut sb, mut tb] = [self.one(), self.zero(), self.zero(), self.one()];

        while let Some(mut i_a) = self.abs_highest_set_bit(&a) && let Some(mut i_b) = self.abs_highest_set_bit(&b) && i_a >= 63 && i_b >= 63 {
            if self.is_neg(&a) {
                self.negate_inplace(&mut a);
                self.negate_inplace(&mut sa);
                self.negate_inplace(&mut ta);
            }
            if self.is_neg(&b) {
                self.negate_inplace(&mut b);
                self.negate_inplace(&mut sb);
                self.negate_inplace(&mut tb);
            }
            if i_a < i_b {
                swap(&mut a, &mut b);
                swap(&mut i_a, &mut i_b);
                swap(&mut sa, &mut sb);
                swap(&mut ta, &mut tb);
            }
            if i_a - i_b >= 16 {
                let a_shift = i_a - 62;
                let b_shift = min(i_b - 31, a_shift);
                let quo = get_abs_head_bits(&a, a_shift) / (get_abs_head_bits(&b, b_shift) + 1);

                a = self.fma_i64_with_shift(&b, -quo, a, (a_shift - b_shift).try_into().unwrap());
                sa = self.fma_i64_with_shift(&sb, -quo, sa, (a_shift - b_shift).try_into().unwrap());
                ta = self.fma_i64_with_shift(&tb, -quo, ta, (a_shift - b_shift).try_into().unwrap());

                debug_assert!(self.eq_el(&a, &self.add(self.mul_ref(&sa, lhs), self.mul_ref(&ta, rhs))));
                debug_assert!(self.eq_el(&b, &self.add(self.mul_ref(&sb, lhs), self.mul_ref(&tb, rhs))));
            } else {
                let shift = max(i_a, i_b) - 62;
                let a_head = get_abs_head_bits(&a, shift);
                debug_assert!(a_head.abs() >= (1 << 47));
                let b_head = get_abs_head_bits(&b, shift);
                debug_assert!(b_head.abs() >= (1 << 47));
                
                // by the bound given by `partial_eea_int()`, we find that `b_new < 2**32`
                // and also `|transform[2]|, |transform[3]| <= a / (a_new - b_new) < 2**64 / 2**31 = 2**33`;
                // thus we decreased the size of at least one new value from >= 2**47 to at most 3 * 2**33
                let (transform, _) = partial_eea_int(StaticRing::<i64>::RING, a_head, b_head, &(1 << 32));

                let apply_transform = |[a, b]: [&mut RustBigint<A>; 2]| {
                    let new_b = from_i64.fma_map(&b, &transform[3], from_i64.mul_ref_map(&a, &transform[2]));
                    *a = from_i64.fma_map(&b, &transform[1], from_i64.mul_map(replace(a, self.zero()), transform[0]));
                    *b = new_b;
                };
                apply_transform([&mut a, &mut b]);
                apply_transform([&mut sa, &mut sb]);
                apply_transform([&mut ta, &mut tb]);

                debug_assert!(self.eq_el(&a, &self.add(self.mul_ref(&sa, lhs), self.mul_ref(&ta, rhs))));
                debug_assert!(self.eq_el(&b, &self.add(self.mul_ref(&sb, lhs), self.mul_ref(&tb, rhs))));
            }
        }
        if self.is_zero(&a) {
            return (sb, tb, b);
        } else if self.is_neg(&a) {
            self.negate_inplace(&mut a);
            self.negate_inplace(&mut sa);
            self.negate_inplace(&mut ta);
        }
        if self.is_zero(&b) {
            return (sa, ta, a);
        } else if self.is_neg(&b) {
            self.negate_inplace(&mut b);
            self.negate_inplace(&mut sb);
            self.negate_inplace(&mut tb);
        }

        let mut i_a = self.abs_highest_set_bit(&a).unwrap();
        let mut i_b = self.abs_highest_set_bit(&b).unwrap();
        if i_a < i_b {
            swap(&mut a, &mut b);
            swap(&mut i_a, &mut i_b);
            swap(&mut sa, &mut sb);
            swap(&mut ta, &mut tb);
        }
        debug_assert!(i_b < 63);
        let b = b.1[0];
        let mut quo = a;
        let a = bigint_div_small(&mut quo.1, b);
        debug_assert!(a <= i64::MAX as u64);
        
        self.negate_inplace(&mut quo);
        sa = self.fma(&quo, &sb, sa);
        ta = self.fma(&quo, &tb, ta);

        let (s_final, t_final, d) = eea(a as i64, b as i64, StaticRing::<i64>::RING);
        return (
            from_i64.fma_map(&sb, &t_final, from_i64.mul_map(sa, s_final)),
            from_i64.fma_map(&tb, &t_final, from_i64.mul_map(ta, s_final)),
            from_i64.map(d)
        );
    }

    #[instrument(skip_all, level = "trace")]
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {

        let from_i64 = RingValue::from_ref(self).can_hom(&StaticRing::<i64>::RING).unwrap();

        let get_abs_head_bits = |x: &RustBigint<A>, shift: usize| {
            let block_idx = shift / 64;
            let remaining_shift = shift % 64;
            let result = if x.1.len() <= block_idx {
                0
            } else if x.1.len() == block_idx + 1 || remaining_shift == 0 {
                x.1[block_idx] >> remaining_shift
            } else {
                (x.1[block_idx] >> remaining_shift) | (x.1[block_idx + 1] << (64 - remaining_shift))
            };
            debug_assert!(result < (1 << 63));
            return result as i64;
        };

        let [mut a, mut b] = [self.clone_el(lhs), self.clone_el(rhs)];

        while let Some(mut i_a) = self.abs_highest_set_bit(&a) && let Some(mut i_b) = self.abs_highest_set_bit(&b) && i_a >= 63 && i_b >= 63 {
            if self.is_neg(&a) {
                self.negate_inplace(&mut a);
            }
            if self.is_neg(&b) {
                self.negate_inplace(&mut b);
            }
            if i_a < i_b {
                swap(&mut a, &mut b);
                swap(&mut i_a, &mut i_b);
            }
            if i_a - i_b >= 16 {
                let a_shift = i_a - 62;
                let b_shift = min(i_b - 31, a_shift);
                let quo = get_abs_head_bits(&a, a_shift) / (get_abs_head_bits(&b, b_shift) + 1);
                a = self.fma_i64_with_shift(&b, -quo, a, (a_shift - b_shift).try_into().unwrap());
            } else {
                let shift = max(i_a, i_b) - 62;
                let a_head = get_abs_head_bits(&a, shift);
                debug_assert!(a_head.abs() >= (1 << 47));
                let b_head = get_abs_head_bits(&b, shift);
                debug_assert!(b_head.abs() >= (1 << 47));
                
                let (transform, _) = partial_eea_int(StaticRing::<i64>::RING, a_head, b_head, &(1 << 32));

                let new_b = from_i64.fma_map(&b, &transform[3], from_i64.mul_ref_map(&a, &transform[2]));
                a = from_i64.fma_map(&b, &transform[1], from_i64.mul_map(a, transform[0]));
                b = new_b;
            }
        }
        if self.is_zero(&a) {
            return b;
        } else if self.is_neg(&a) {
            self.negate_inplace(&mut a);
        }
        if self.is_zero(&b) {
            return a;
        } else if self.is_neg(&b) {
            self.negate_inplace(&mut b);
        }

        let mut i_a = self.abs_highest_set_bit(&a).unwrap();
        let mut i_b = self.abs_highest_set_bit(&b).unwrap();
        if i_a < i_b {
            swap(&mut a, &mut b);
            swap(&mut i_a, &mut i_b);
        }
        debug_assert!(i_b < 63);
        let b = b.1[0];
        let mut quo = a;
        let a = bigint_div_small(&mut quo.1, b);
        debug_assert!(a <= i64::MAX as u64);
        
        self.negate_inplace(&mut quo);
        let d = gcd(a as i64, b as i64, StaticRing::<i64>::RING);
        return from_i64.map(d);
    }
}

impl<A: Allocator + Send + Sync + Clone> EuclideanRing for RustBigintRingBase<A> {

    fn euclidean_div_rem(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        let mut quo = RustBigint(false, bigint_div(&mut lhs.1, &rhs.1, self.zero().1));
        // if result of division is zero, `.is_neg(&lhs)` does not work as expected
        if rhs.0 ^ lhs.0 {
            self.negate_inplace(&mut quo);
        }
        return (quo, lhs);
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        self.map_i128(val).and_then(|x| x.checked_abs()).and_then(|x| usize::try_from(x).ok())
    }
}

impl<A: Allocator + Send + Sync + Clone> HashableElRing for RustBigintRingBase<A> {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        for i in 0..effective_length(&el.1) {
            h.write_u64(el.1[i])
        }
    }
}

impl Serialize for RustBigintRingBase<Global> {

    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("IntegerRing(RustBigInt)", ()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RustBigintRingBase<Global> {

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("IntegerRing(RustBigInt)", PhantomData::<()>).deserialize(deserializer).map(|_| RustBigintRing::RING.into())
    }
}

impl<A: Allocator + Send + Sync + Clone> SerializableElementRing for RustBigintRingBase<A> {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        if deserializer.is_human_readable() {
            // this makes an unnecessary temporary allocation, but then the cost is probably negligible compared
            // to the parsing of a string as a number
            let string = DeserializeSeedNewtypeStruct::new("BigInt", PhantomData::<String>).deserialize(deserializer)?;
            return self.parse(string.as_str(), 10).map_err(|()| de::Error::custom(format!("cannot parse \"{}\" as number", string)));
        } else {
            let (negative, data) = deserialize_bigint_from_bytes(deserializer, |data| {
                let mut result_data = Vec::with_capacity_in(data.len().div_ceil(size_of::<u64>()), self.allocator.clone());
                let (chunks, last) = data.as_chunks();
                for digit in chunks {
                    result_data.push(u64::from_le_bytes(*digit));
                }
                result_data.push(u64::from_le_bytes(std::array::from_fn(|i| if i >= last.len() { 0 } else { last[i] })));
                return result_data;
            })?;
            return Ok(RustBigint(negative, data));
        }
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        if serializer.is_human_readable() {
            SerializableNewtypeStruct::new("BigInt", format!("{}", RingRef::new(self).formatted_el(el)).as_str()).serialize(serializer)
        } else {
            let len = effective_length(&el.1);
            let mut data = Vec::with_capacity_in(len * size_of::<u64>(), &self.allocator);
            for digit in &el.1 {
                data.extend(digit.to_le_bytes().into_iter());
            }
            let mut seq = serializer.serialize_tuple(2)?;
            seq.serialize_element(&self.is_neg(el))?;
            seq.serialize_element(serde_bytes::Bytes::new(&data[..]))?;
            return seq.end();
        }
    }
}

impl_interpolation_base_ring_char_zero!{ <{A}> InterpolationBaseRing for RustBigintRingBase<A> where A: Allocator + Send + Sync + Clone }

impl_poly_gcd_locally_for_ZZ!{ <{A}> IntegerPolyGCDRing for RustBigintRingBase<A> where A: Allocator + Send + Sync + Clone }

impl_eval_poly_locally_for_ZZ!{ <{A}> LiftPolyEvalRing for RustBigintRingBase<A> where A: Allocator + Send + Sync + Clone }

impl<A> FiniteRingSpecializable for RustBigintRingBase<A>
    where A: Allocator + Send + Sync + Clone
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<A: Allocator + Send + Sync + Clone> CanHomFrom<StaticRingBase<i64>> for RustBigintRingBase<A> {

    fn fma_map_in(&self, _: &StaticRingBase<i64>, lhs: &RustBigint<A>, rhs: &<StaticRingBase<i64> as RingBase>::Element, summand: RustBigint<A>, (): &()) -> RustBigint<A> {
        self.fma_i64_with_shift(lhs, *rhs, summand, 0)
    }

    fn mul_assign_map_in(&self, from: &StaticRingBase<i64>, lhs: &mut RustBigint<A>, rhs: <StaticRingBase<i64> as RingBase>::Element, (): &()) {
        self.mul_assign_map_in_ref(from, lhs, &rhs, &());
    }

    fn mul_assign_map_in_ref(&self, _: &StaticRingBase<i64>, lhs: &mut RustBigint<A>, rhs: &<StaticRingBase<i64> as RingBase>::Element, (): &()) {
        core_fma_small_ref_snd(&mut lhs.1, rhs.unsigned_abs() as u64, &[], 0);
        lhs.0 ^= *rhs < 0;
    }
}

impl<A: Allocator + Send + Sync + Clone> IntegerRing for RustBigintRingBase<A> {

    fn to_float_approx(&self, value: &Self::Element) -> f64 {
        let sign = if value.0 { -1. } else { 1. };
        match effective_length(&value.1) {
            0 => 0.,
            1 => value.1[0] as f64 * sign,
            l => (value.1[l - 1] as f64 * 2f64.powi(((l - 1) * u64::BITS as usize).try_into().unwrap()) + value.1[l - 2] as f64 * 2f64.powi(((l - 2) * u64::BITS as usize).try_into().unwrap())) * sign
        }
    }

    fn from_float_approx(&self, mut value: f64) -> Option<Self::Element> {
        if value.round() == 0. {
            return Some(self.zero());
        }
        let sign = value < 0.;
        value = value.abs();
        let scale: i32 = (value.log2().ceil() as i64).try_into().unwrap();
        let significant_digits = std::cmp::min(scale, u64::BITS.try_into().unwrap());
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
        let block = effective_length(&value.1).checked_sub(1)?;
        Some(block * u64::BITS as usize + u64::BITS as usize - value.1[block].leading_zeros() as usize - 1)
    }

    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize) {
        bigint_rshift(&mut value.1, power);
    }

    fn mul_pow_2(&self, value: &mut Self::Element, power: usize) {
        bigint_lshift(&mut value.1, power.try_into().unwrap())
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
use crate::tracing::LogAlgorithmSubscriber;

#[cfg(test)]
const ZZ: RustBigintRing = RustBigintRing::RING;

#[test]
fn test_print_power_2() {
    LogAlgorithmSubscriber::init_test();
    let x = RustBigint(false, vec![0, 0, 1]);
    assert_eq!("340282366920938463463374607431768211456", format!("{}", RustBigintRing::RING.formatted_el(&x)));
}

#[test]
fn test_from() {
    LogAlgorithmSubscriber::init_test();
    assert!(ZZ.eq_el(&RustBigint(false, vec![]), &ZZ.int_hom().map(0)));
    assert!(ZZ.eq_el(&RustBigint(false, vec![2138479]), &ZZ.int_hom().map(2138479)));
    assert!(ZZ.eq_el(&RustBigint(true, vec![2138479]), &ZZ.int_hom().map(-2138479)));
    // assert!(ZZ.eq(&RustBigint(false, vec![0x38691a350bf12fca, 0x1]), &ZZ.from_z_gen(0x138691a350bf12fca, &i128::RING)));
}

#[test]
fn test_to_i128() {
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
    let mut x = RustBigintRing::RING.get_ring().parse("4294836225", 10).unwrap();
    let y = RustBigintRing::RING.get_ring().parse("4294967297", 10).unwrap();
    let z = RustBigintRing::RING.get_ring().parse("-131072", 10).unwrap();
    x = ZZ.sub_ref_fst(&x, y);
    assert!(ZZ.eq_el(&z, &x));
}

#[test]
fn test_assumptions_integer_division() {
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
    crate::ring::generic_tests::test_ring_axioms(ZZ, edge_case_elements())
}

#[test]
fn test_hash_axioms() {
    LogAlgorithmSubscriber::init_test();
    crate::ring::generic_tests::test_hash_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_divisibility_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    crate::divisibility::generic_tests::test_divisibility_axioms(ZZ, edge_case_elements())
}

#[test]
fn test_bigint_euclidean_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    crate::pid::generic_tests::test_euclidean_ring_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_principal_ideal_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(ZZ, edge_case_elements());
}

#[test]
fn test_bigint_integer_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    crate::integer::generic_tests::test_integer_axioms(ZZ, edge_case_elements())
}

#[test]
fn from_to_float_approx() {
    LogAlgorithmSubscriber::init_test();
    let x: f64 = 83465209236517892563478156042389675783219532497861237985328563.;
    let y = ZZ.to_float_approx(&ZZ.from_float_approx(x).unwrap());
    assert!(x * 0.99999 < y);
    assert!(y < x * 1.00001);

    let x = RustBigintRing::RING.get_ring().parse("238238568756187236598172345698172345698713465983465981349196413289715928374691873256349862875423823856875618723659817234569817234569871346598346598134919641328971592837469187325634986287542382385687561872365981723456981723456987134659834659813491964132897159283746918732563498628754", 10).unwrap();
    let x_f64 = RustBigintRing::RING.to_float_approx(&x);
    assert!(x_f64 > 2.38e281);
    assert!(x_f64 < 2.39e281);
    
    let x = RustBigintRing::RING.get_ring().parse("17612270634266603266562983043609488425884596885216274157019263443846280837737053004240660825056953589054705681188523315954751538958870401258595088307206392227789986855433848759623746265294744028223494023320398812305222823465701205669244602427862540158018457529069827142861864092328740970800137803882190383463", 10).unwrap();
    let x_f64 = RustBigintRing::RING.to_float_approx(&x);
    assert!(x_f64 > 1.76e307);
    assert!(x_f64 < 1.77e307);
}

#[bench]
fn bench_div_300_bits(bencher: &mut test::Bencher) {
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
    assert_eq!(true, ZZ.is_lt(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(2)));
    assert_eq!(true, ZZ.is_lt(&ZZ.int_hom().map(1), &ZZ.int_hom().map(2)));
    assert_eq!(false, ZZ.is_lt(&ZZ.int_hom().map(2), &ZZ.int_hom().map(2)));
    assert_eq!(false, ZZ.is_lt(&ZZ.int_hom().map(3), &ZZ.int_hom().map(2)));
    assert_eq!(true, ZZ.is_gt(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(-2)));

    assert_eq!(Ordering::Less, ZZ.abs_cmp(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(2)));
    assert_eq!(Ordering::Less, ZZ.abs_cmp(&ZZ.int_hom().map(1), &ZZ.int_hom().map(2)));
    assert_eq!(Ordering::Equal, ZZ.abs_cmp(&ZZ.int_hom().map(2), &ZZ.int_hom().map(2)));
    assert_eq!(Ordering::Greater, ZZ.abs_cmp(&ZZ.int_hom().map(-3), &ZZ.int_hom().map(2)));
    assert_eq!(Ordering::Greater, ZZ.abs_cmp(&ZZ.int_hom().map(3), &ZZ.int_hom().map(2)));
    assert_eq!(Ordering::Less, ZZ.abs_cmp(&ZZ.int_hom().map(-1), &ZZ.int_hom().map(-2)));
}

#[test]
fn test_get_uniformly_random() {
    LogAlgorithmSubscriber::init_test();
    crate::integer::generic_tests::test_integer_get_uniformly_random(ZZ);

    let ring = ZZ;
    let bound = RustBigintRing::RING.get_ring().parse("11000000000000000", 16).unwrap();
    let block_bound = RustBigintRing::RING.get_ring().parse("10000000000000000", 16).unwrap();
    let mut rng = oorandom::Rand64::new(1);
    let elements: Vec<_> = (0..1000).map(|_| ring.get_uniformly_random(&bound, || rng.rand_u64())).collect();
    assert!(elements.iter().any(|x| ring.is_lt(x, &block_bound)));
    assert!(elements.iter().any(|x| ring.is_gt(x, &block_bound)));
    assert!(elements.iter().all(|x| ring.is_lt(x, &bound)));
}

#[test]
fn test_canonical_iso_static_int() {
    LogAlgorithmSubscriber::init_test();
    // for the hom test, we have to be able to multiply elements in `StaticRing::<i128>::RING`, so we cannot test `i128::MAX` or `i128::MIN`
    crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, ZZ, [0, 1, -1, -100, 100, i64::MAX as i128, i64::MIN as i128].iter().copied());
    crate::ring::generic_tests::test_iso_axioms(StaticRing::<i128>::RING, ZZ, [0, 1, -1, -100, 100, i64::MAX as i128, i64::MIN as i128, i128::MAX, i128::MIN].iter().copied());
}

#[test]
fn test_fma_map_in() {
    LogAlgorithmSubscriber::init_test();
    let from_i64 = ZZ.can_hom(&StaticRing::<i64>::RING).unwrap();
    assert_el_eq!(ZZ, ZZ.get_ring().parse("10000000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &100, ZZ.zero()));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("-10000000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &-100, ZZ.zero()));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("10000000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &100, ZZ.negate(ZZ.zero())));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("-10000000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &-100, ZZ.negate(ZZ.zero())));

    assert_el_eq!(ZZ, ZZ.get_ring().parse("10100000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &100, ZZ.get_ring().parse("100000000000000000000", 10).unwrap()));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("-9900000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &-100, ZZ.get_ring().parse("100000000000000000000", 10).unwrap()));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("9900000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &100, ZZ.get_ring().parse("-100000000000000000000", 10).unwrap()));
    assert_el_eq!(ZZ, ZZ.get_ring().parse("-10100000000000000000000", 10).unwrap(), from_i64.fma_map(&ZZ.get_ring().parse("100000000000000000000", 10).unwrap(), &-100, ZZ.get_ring().parse("-100000000000000000000", 10).unwrap()));
}

#[test]
fn test_serialize() {
    LogAlgorithmSubscriber::init_test();
    crate::serialization::generic_tests::test_serialization(ZZ, edge_case_elements())
}

#[test]
fn test_serialize_postcard() {
    LogAlgorithmSubscriber::init_test();
    let serialized = postcard::to_allocvec(&SerializeWithRing::new(&ZZ.power_of_two(10000), ZZ)).unwrap();
    let result = DeserializeWithRing::new(ZZ).deserialize(
        &mut postcard::Deserializer::from_flavor(postcard::de_flavors::Slice::new(&serialized))
    ).unwrap();

    assert_el_eq!(ZZ, ZZ.power_of_two(10000), result);
}