use crate::algorithms::eea::signed_gcd;
use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::rings::zn::zn_64;
use crate::rings::zn::ZnOperation;
use crate::rings::zn::ZnRing;
use crate::primitive_int::*;
use crate::rings::zn::ZnRingStore;
use crate::rings::zn::choose_zn_impl;
use crate::DEFAULT_PROBABILISTIC_REPETITIONS;

use oorandom;

struct CheckIsFieldMillerRabin {
    probability_param: usize
}

impl ZnOperation for CheckIsFieldMillerRabin {

    type Output<'a> = bool
        where Self: 'a;

    fn call<'a, R>(self, ring: R) -> bool
        where R: 'a + ZnRingStore, R::Type: ZnRing
    {
        is_prime_base(ring, self.probability_param)
    }
}

///
/// Miller-Rabin primality test.
/// 
/// If n is a prime, this returns true.
/// If n is not a prime, this returns false with probability greater or 
/// equal than 1 - 4^(-k).
/// For very small primes, a lookup table may be used.
/// 
/// For details, see [`is_prime_base()`]
/// 
pub fn is_prime<I>(ZZ: I, n: &El<I>, k: usize) -> bool 
    where I: IntegerRingStore + HashableElRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i128>>
{
    assert!(ZZ.is_pos(n));
    if ZZ.is_zero(n) || ZZ.is_one(n) {
        false
    } else {
        let n_copy = ZZ.clone_el(n);
        choose_zn_impl(ZZ, n_copy, CheckIsFieldMillerRabin { probability_param: k })
    }
}

const SMALL_IS_COPRIME_TABLE: [bool; 30] = [
    false,
    true,
    false,
    false,
    false,
    false,
    false,
    true,
    false,
    false,
    false,
    true,
    false,
    true,
    false,
    false,
    false,
    true,
    false,
    true,
    false,
    false,
    false,
    true,
    false,
    false,
    false,
    false,
    false,
    true
];

fn search_prime<I: IntegerRingStore>(ZZ: I, mut n: El<I>, delta: i64) -> Option<El<I>>
    where I::Type: IntegerRing,
        zn_64::ZnBase: CanHomFrom<I::Type>
{
    assert!(ZZ.is_pos(&n));

    let m = SMALL_IS_COPRIME_TABLE.len();
    let Zm = zn_64::Zn::new(m as u64);
    let mut n_mod_m = Zm.coerce(&ZZ, ZZ.clone_el(&n));
    let mut diff_to_n = 0;
    let Zi64_to_Zm = Zm.can_hom::<StaticRing<i64>>(&StaticRing::<i64>::RING).unwrap();
    let Zi64_to_ZZ = ZZ.can_hom(&StaticRing::<i64>::RING).unwrap();
    debug_assert!(ZZ.is_one(&signed_gcd(ZZ.clone_el(&n), Zi64_to_ZZ.map(delta), &ZZ)));
    let Zm_delta = Zi64_to_Zm.map(delta);
    let ZZ_m = Zi64_to_ZZ.map(m as i64);

    // we continue the main loop until we reach `n <= m`; at this point, the main loop is not correct
    // anymore, since being in `Z/mZ \ Z/mZ*` does not imply nonprimality anymore
    let mut remaining_steps = if ZZ.is_leq(&n, &ZZ_m) {
        0
    } else if delta < 0 {
        let max_steps = ZZ.ceil_div(ZZ.sub_ref(&n, &Zi64_to_ZZ.map(m as i64)), &Zi64_to_ZZ.map(-delta));
        if ZZ.is_lt(&max_steps, &Zi64_to_ZZ.map(i64::MAX)) {
            int_cast(max_steps, StaticRing::<i64>::RING, &ZZ)
        } else {
            i64::MAX
        }
    } else {
        i64::MAX
    };

    while remaining_steps != 0 {
        while !SMALL_IS_COPRIME_TABLE[Zm.smallest_positive_lift(n_mod_m) as usize] && remaining_steps != 0 {
            diff_to_n += delta;
            Zm.add_assign(&mut n_mod_m, Zm_delta);
            remaining_steps -= 1;
        }
        ZZ.add_assign(&mut n, Zi64_to_ZZ.map(diff_to_n));
        if remaining_steps == 0 {
            break;
        }
        if is_prime(&ZZ, &n, DEFAULT_PROBABILISTIC_REPETITIONS) {
            return Some(n);
        } else {
            diff_to_n = delta;
            Zm.add_assign(&mut n_mod_m, Zm_delta);
            remaining_steps -= 1;
        }
    }
    let mut n = int_cast(n, StaticRing::<i64>::RING, &ZZ);
    assert!(n <= m as i64);
    while n > 0 {
        if is_prime(&StaticRing::<i64>::RING, &n, DEFAULT_PROBABILISTIC_REPETITIONS) {
            return Some(Zi64_to_ZZ.map(n));
        }
        n += delta;
    }
    return None;
}

///
/// Returns the largest prime smaller than the given integer.
/// 
#[stability::unstable(feature = "enable")]
pub fn prev_prime<I: IntegerRingStore>(ZZ: I, n: El<I>) -> Option<El<I>>
    where I::Type: IntegerRing,
        zn_64::ZnBase: CanHomFrom<I::Type>
{
    assert!(ZZ.is_pos(&n));
    if ZZ.is_one(&n) {
        return None;
    }
    let n_minus_one = ZZ.sub(n, ZZ.one());
    search_prime(ZZ, n_minus_one, -1)
}

///
/// Returns the smallest prime larger than the given integer.
/// 
#[stability::unstable(feature = "enable")]
pub fn next_prime<I: IntegerRingStore>(ZZ: I, n: El<I>) -> El<I>
    where I::Type: IntegerRing,
        zn_64::ZnBase: CanHomFrom<I::Type>
{
    assert!(ZZ.is_pos(&n));
    let n_plus_one = ZZ.add(n, ZZ.one());
    search_prime(ZZ, n_plus_one, 1).unwrap()
}

///
/// Miller-Rabin primality test.
/// 
/// If the characteristic `n` of the given ring is a prime, this returns true.
/// If n is not a prime, this returns false with probability greater or 
/// equal than 1 - 4^(-k).
/// 
/// Complexity O(k log(n)^3)
/// 
/// # Randomness
/// 
/// Note that the randomness used for this function is derived only from
/// the input, hence it will always yield the same output on the same input.
/// Technically, it follows that the probability of a wrong output is greater
/// than 4^(-k) on some outputs (as it is either 0 or 1), but of course
/// this is not helpful. To be completely precise: If the seed of the used
/// PRNG would be random, then the probability of a wrong output is at 
/// most 4^(-k).
/// 
pub fn is_prime_base<R>(Zn: R, k: usize) -> bool 
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let ZZ = Zn.integer_ring();
    let n = Zn.modulus();
    assert!(ZZ.is_pos(n));

    let mut rng = oorandom::Rand64::new(ZZ.default_hash(n) as u128);
    let mut n_minus_one = ZZ.sub_ref_fst(n, ZZ.one());
    let s = ZZ.abs_lowest_set_bit(&n_minus_one).unwrap();
    ZZ.euclidean_div_pow_2(&mut n_minus_one, s as usize);
    let d = n_minus_one;

    for _i in 0..k {
        let a = Zn.random_element(|| rng.rand_u64());
        if Zn.is_zero(&a) {
            continue;
        }
        let mut current = Zn.pow_gen(a, &d, &ZZ);
        let mut miller_rabin_condition = Zn.is_one(&current);
        for _r in 0..s {
            miller_rabin_condition |= Zn.is_neg_one(&current);
            if miller_rabin_condition {
                break;
            }
            Zn.square(&mut current);
        }
        if Zn.is_zero(&current) || !miller_rabin_condition {
            return false;
        }
    }
    return true;
}

#[cfg(test)]
use crate::rings::rust_bigint::RustBigintRing;

#[test]
pub fn test_is_prime() {
    assert!(is_prime(StaticRing::<i128>::RING, &2, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &3, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &5, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &7, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &11, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &22531, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &417581, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &68719476767, 5));

    assert!(!is_prime(StaticRing::<i128>::RING, &1, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &4, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &6, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &8, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &9, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &10, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &22532, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &347584, 5));

    assert!(is_prime(RustBigintRing::RING, &RustBigintRing::RING.get_ring().parse("170141183460469231731687303715884105727", 10).unwrap(), 10));
}

#[test]
fn test_prev_prime() {
    assert_eq!(Some(7), prev_prime(StaticRing::<i64>::RING, 11));
    assert_eq!(Some(7), prev_prime(StaticRing::<i64>::RING, 10));
    assert_eq!(Some(7), prev_prime(StaticRing::<i64>::RING, 9));
    assert_eq!(Some(7), prev_prime(StaticRing::<i64>::RING, 8));
    assert_eq!(Some(5), prev_prime(StaticRing::<i64>::RING, 7));
    assert_eq!(Some(5), prev_prime(StaticRing::<i64>::RING, 6));
    assert_eq!(Some(3), prev_prime(StaticRing::<i64>::RING, 5));
    assert_eq!(Some(3), prev_prime(StaticRing::<i64>::RING, 4));
    assert_eq!(Some(2), prev_prime(StaticRing::<i64>::RING, 3));
    assert_eq!(None, prev_prime(StaticRing::<i64>::RING, 2));
    assert_eq!(None, prev_prime(StaticRing::<i64>::RING, 1));

    let mut last_prime = 11;
    for i in 12..1000 {
        assert_eq!(Some(last_prime), prev_prime(StaticRing::<i64>::RING, i));
        if is_prime(StaticRing::<i64>::RING, &i, DEFAULT_PROBABILISTIC_REPETITIONS) {
            last_prime = i;
        }
    }
}

#[test]
fn test_next_prime() {
    let mut last_prime = 1009;
    for i in (2..1000).rev() {
        assert_eq!(last_prime, next_prime(StaticRing::<i64>::RING, i));
        if is_prime(StaticRing::<i64>::RING, &i, DEFAULT_PROBABILISTIC_REPETITIONS) {
            last_prime = i;
        }
    }
    assert_eq!(2, next_prime(StaticRing::<i64>::RING, 1));
}

#[test]
fn test_search_prime() {
    assert_eq!(None, search_prime(StaticRing::<i64>::RING, 1, -2));
    for (p, n) in [(3, 3), (5, 5), (7, 7), (7, 9), (11, 11)] {
        assert_eq!(Some(p), search_prime(StaticRing::<i64>::RING, n, -2));
    }

    assert_eq!(None, search_prime(StaticRing::<i64>::RING, 1, -3));
    assert_eq!(None, search_prime(StaticRing::<i64>::RING, 4, -3));
    for (p, n) in [(2, 2), (5, 5), (7, 7), (5, 8), (7, 10), (11, 11), (13, 13), (11, 14), (13, 16), (17, 17)] {
        assert_eq!(Some(p), search_prime(StaticRing::<i64>::RING, n, -3));
    }
    assert_eq!(Some(359), search_prime(StaticRing::<i64>::RING, 380, -3));
}