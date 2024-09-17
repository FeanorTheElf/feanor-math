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

use oorandom;

struct CheckIsFieldMillerRabin {
    probability_param: usize
}

impl ZnOperation<bool> for CheckIsFieldMillerRabin {

    fn call<R: ZnRingStore>(self, ring: R) -> bool
        where R::Type: ZnRing
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

///
/// Returns the largest prime smaller than the given integer.
/// 
#[stability::unstable(feature = "enable")]
pub fn prev_prime<I: IntegerRingStore>(ZZ: I, mut n: El<I>) -> Option<El<I>>
    where I::Type: IntegerRing,
        zn_64::ZnBase: CanHomFrom<I::Type>
{
    assert!(ZZ.is_pos(&n));
    ZZ.sub_assign(&mut n, ZZ.one());
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
    let Z30 = zn_64::Zn::new(30);
    let mut n_mod_30 = Z30.coerce(&ZZ, ZZ.clone_el(&n));
    let mut diff_to_n = 0;
    let ZZ_30 = ZZ.int_hom().map(30);
    while ZZ.is_geq(&n, &ZZ_30) {
        while !SMALL_IS_COPRIME_TABLE[Z30.smallest_positive_lift(n_mod_30) as usize] {
            Z30.sub_assign(&mut n_mod_30, Z30.one());
            diff_to_n += 1;
        }
        ZZ.sub_assign(&mut n, ZZ.int_hom().map(diff_to_n));
        if is_prime(&ZZ, &n, 10) {
            return Some(n);
        } else {
            diff_to_n = 1;
            Z30.sub_assign(&mut n_mod_30, Z30.one());
        }
    }
    const TABLE: [Option<i32>; 30] = [
        None,
        None,
        Some(2),
        Some(3),
        Some(3),
        Some(5),
        Some(5),
        Some(7),
        Some(7),
        Some(7),
        Some(7),
        Some(11),
        Some(11),
        Some(13),
        Some(13),
        Some(13),
        Some(13),
        Some(17),
        Some(17),
        Some(19),
        Some(19),
        Some(19),
        Some(19),
        Some(23),
        Some(23),
        Some(23),
        Some(23),
        Some(23),
        Some(23),
        Some(29),
    ];
    return TABLE[Z30.smallest_positive_lift(n_mod_30) as usize].map(|n| ZZ.int_hom().map(n));
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

    let mut last_prime = 29;
    for i in 30..1000 {
        assert_eq!(Some(last_prime), prev_prime(StaticRing::<i64>::RING, i));
        if is_prime(StaticRing::<i64>::RING, &i, 10) {
            last_prime = i;
        }
    }
}