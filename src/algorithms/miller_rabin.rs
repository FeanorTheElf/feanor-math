use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::integer::*;
use crate::rings::zn::ZnRing;
use crate::primitive_int::*;
use crate::rings::zn::ZnRingStore;
use crate::rings::zn::choose_zn_impl;
use crate::zn_op;

use oorandom;

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
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i128>>
{
    assert!(ZZ.is_pos(n));
    if ZZ.is_zero(n) || ZZ.is_one(n) {
        false
    } else {
        let mut result = false;
        let n_copy = ZZ.clone_el(n);
        choose_zn_impl(ZZ, n_copy, zn_op!{ <{'a}> [args: (&'a mut bool, usize) = (&mut result, k)] |ring, (result, k): (&mut bool, usize)| {
            *result = is_prime_base(ring, k);
        } });
        return result;
    }
}

///
/// Miller-Rabin primality test.
/// 
/// If n is a prime, this returns true.
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
}