use oorandom;
use tracing::instrument;

use crate::DEFAULT_PROBABILISTIC_REPETITIONS;
use crate::homomorphism::*;
use crate::ring_properties::ordered::OrderedRingStore;
use crate::ring_properties::pid::PrincipalIdealRingStore;
use crate::primitive_int::*;
use crate::prelude::*;
use crate::ring::HashableElRingStore;
use crate::ring_impls::zn::{ZnRingStore, choose_zn_impl, *};

struct CheckIsFieldMillerRabin {
    probability_param: usize,
}

impl ZnOperation for CheckIsFieldMillerRabin {
    type Output<'a>
        = bool
    where
        Self: 'a;

    fn call<'a, R>(self, ring: R) -> bool
    where
        R: 'a + ZnRingStore,
        R::Ring: ZnRing,
    {
        is_prime_base(ring, self.probability_param)
    }
}

/// Miller-Rabin primality test.
///
/// If n is a prime, this returns true.
/// If n is not a prime, this returns false with probability greater or
/// equal than 1 - 4^(-k).
/// For very small primes, a lookup table may be used.
///
/// For details, see [`is_prime_base()`]
pub fn is_prime<I>(ZZ: I, n: &El<I>, k: usize) -> bool
where
    I: IntegerRingStore,
    I::Ring: IntegerRing + CanIsoFromTo<StaticRingBase<i128>>,
{
    assert!(ZZ.is_pos(n));
    if ZZ.is_zero(n) || ZZ.is_one(n) {
        false
    } else {
        let n_copy = n.clone();
        choose_zn_impl(ZZ, n_copy, CheckIsFieldMillerRabin { probability_param: k })
    }
}

const SMALL_IS_COPRIME_TABLE: [bool; 30] = [
    false, true, false, false, false, false, false, true, false, false, false, true, false, true, false, false, false,
    true, false, true, false, false, false, true, false, false, false, false, false, true,
];

#[instrument(skip_all, level = "trace")]
fn search_prime<I: IntegerRingStore>(ZZ: I, mut n: El<I>, delta: i64) -> Option<El<I>>
where
    I::Ring: IntegerRing,
    zn_64b::Zn64BBase: CanHomFrom<I::Ring>,
{
    assert!(!ZZ.is_neg(&n));

    let m = SMALL_IS_COPRIME_TABLE.len();
    let Zm = zn_64b::Zn64B::new(m as u64);
    let mut n_mod_m = Zm.coerce(&ZZ, n.clone());
    let mut diff_to_n = 0;
    let Zi64_to_Zm = Zm.can_hom::<StaticRing<i64>>(&ZZi64).unwrap();
    let Zi64_to_ZZ = ZZ.can_hom(&ZZi64).unwrap();
    debug_assert!(ZZ.is_unit(&ZZ.ideal_gen(&n, &Zi64_to_ZZ.map(delta))));
    let Zm_delta = Zi64_to_Zm.map(delta);
    let ZZ_m = Zi64_to_ZZ.map(m.try_into().unwrap());

    // we continue the main loop until we reach `n <= m`; at this point, the main loop is not correct
    // anymore, since being in `Z/mZ \ Z/mZ*` does not imply nonprimality anymore
    let mut remaining_steps = if ZZ.is_leq(&n, &ZZ_m) {
        0
    } else if delta < 0 {
        let max_steps = ZZ.ceil_div(
            ZZ.sub_ref(&n, &Zi64_to_ZZ.map(m.try_into().unwrap())),
            &Zi64_to_ZZ.map(-delta),
        );
        if ZZ.is_lt(&max_steps, &Zi64_to_ZZ.map(i64::MAX)) {
            int_cast(max_steps, ZZi64, &ZZ)
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
    let mut n = int_cast(n, ZZi64, &ZZ);
    assert!(n <= m.try_into().unwrap());
    while n > 0 {
        if is_prime(&ZZi64, &n, DEFAULT_PROBABILISTIC_REPETITIONS) {
            return Some(Zi64_to_ZZ.map(n));
        }
        n += delta;
    }
    return None;
}

/// Returns the largest prime smaller than the given integer.
#[stability::unstable(feature = "enable")]
pub fn prev_prime<I: IntegerRingStore>(ZZ: I, n: El<I>) -> Option<El<I>>
where
    I::Ring: IntegerRing,
    zn_64b::Zn64BBase: CanHomFrom<I::Ring>,
{
    assert!(!ZZ.is_neg(&n));
    if ZZ.is_zero(&n) || ZZ.is_one(&n) {
        return None;
    }
    let n_minus_one = ZZ.sub(n, ZZ.one());
    search_prime(ZZ, n_minus_one, -1)
}

/// Returns the smallest prime larger than the given integer.
#[stability::unstable(feature = "enable")]
pub fn next_prime<I: IntegerRingStore>(ZZ: I, n: El<I>) -> El<I>
where
    I::Ring: IntegerRing,
    zn_64b::Zn64BBase: CanHomFrom<I::Ring>,
{
    assert!(!ZZ.is_neg(&n));
    let n_plus_one = ZZ.add(n, ZZ.one());
    search_prime(ZZ, n_plus_one, 1).unwrap()
}

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
#[instrument(skip_all, level = "trace")]
pub fn is_prime_base<R>(Zn: R, k: usize) -> bool
where
    R: ZnRingStore,
    R::Ring: ZnRing,
{
    let ZZ = Zn.integer_ring();
    let n = Zn.modulus();
    assert!(ZZ.is_pos(n));
    if ZZ.is_one(n) {
        return false;
    }

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
use crate::ring_impls::rust_bigint::RustBigintRing;

#[test]
fn test_is_prime() {
    feanor_tracing::DelayedLogger::init_test();
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

    assert!(is_prime(
        RustBigintRing::RING,
        &RustBigintRing::RING
            .get_ring()
            .parse("170141183460469231731687303715884105727", 10)
            .unwrap(),
        10
    ));
}

#[test]
fn test_prev_prime() {
    feanor_tracing::DelayedLogger::init_test();
    assert_eq!(Some(7), prev_prime(ZZi64, 11));
    assert_eq!(Some(7), prev_prime(ZZi64, 10));
    assert_eq!(Some(7), prev_prime(ZZi64, 9));
    assert_eq!(Some(7), prev_prime(ZZi64, 8));
    assert_eq!(Some(5), prev_prime(ZZi64, 7));
    assert_eq!(Some(5), prev_prime(ZZi64, 6));
    assert_eq!(Some(3), prev_prime(ZZi64, 5));
    assert_eq!(Some(3), prev_prime(ZZi64, 4));
    assert_eq!(Some(2), prev_prime(ZZi64, 3));
    assert_eq!(None, prev_prime(ZZi64, 2));
    assert_eq!(None, prev_prime(ZZi64, 1));
    assert_eq!(None, prev_prime(ZZi64, 0));

    let mut last_prime = 11;
    for i in 12..1000 {
        assert_eq!(Some(last_prime), prev_prime(ZZi64, i));
        if is_prime(ZZi64, &i, DEFAULT_PROBABILISTIC_REPETITIONS) {
            last_prime = i;
        }
    }
}

#[test]
fn test_next_prime() {
    feanor_tracing::DelayedLogger::init_test();
    let mut last_prime = 1009;
    for i in (2..1000).rev() {
        assert_eq!(last_prime, next_prime(ZZi64, i));
        if is_prime(ZZi64, &i, DEFAULT_PROBABILISTIC_REPETITIONS) {
            last_prime = i;
        }
    }
    assert_eq!(2, next_prime(ZZi64, 1));
    assert_eq!(2, next_prime(ZZi64, 0));
}

#[test]
fn test_search_prime() {
    feanor_tracing::DelayedLogger::init_test();
    assert_eq!(None, search_prime(ZZi64, 1, -2));
    for (p, n) in [(3, 3), (5, 5), (7, 7), (7, 9), (11, 11)] {
        assert_eq!(Some(p), search_prime(ZZi64, n, -2));
    }

    assert_eq!(None, search_prime(ZZi64, 1, -3));
    assert_eq!(None, search_prime(ZZi64, 4, -3));
    for (p, n) in [
        (2, 2),
        (5, 5),
        (7, 7),
        (5, 8),
        (7, 10),
        (11, 11),
        (13, 13),
        (11, 14),
        (13, 16),
        (17, 17),
    ] {
        assert_eq!(Some(p), search_prime(ZZi64, n, -3));
    }
    assert_eq!(Some(359), search_prime(ZZi64, 380, -3));
}
