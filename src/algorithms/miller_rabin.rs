use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::integer::*;
use crate::algorithms;
use crate::rings::zn::zn_barett::*;

use oorandom;

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
pub fn is_prime<I>(ZZ: I, n: &El<I>, k: usize) -> bool 
    where I: IntegerRingStore + HashableElRingStore
{
    if ZZ.is_leq(n, &ZZ.from_int(2)) {
        return ZZ.eq(n, &ZZ.from_int(2));
    }

    let mut rng = oorandom::Rand64::new(ZZ.default_hash(n) as u128);
    let mut n_minus_one = ZZ.sub_ref_fst(n, ZZ.one());
    let s = ZZ.abs_lowest_set_bit(&n_minus_one).unwrap();
    ZZ.euclidean_div_pow_2(&mut n_minus_one, s as usize);
    let d = n_minus_one;
    let Zn = Zn::new(&ZZ, ZZ.clone(n));

    for _i in 0..k {
        let a = Zn.get_ring().project(ZZ.add(ZZ.get_uniformly_random(n, || rng.rand_u64()), ZZ.one()));
        if Zn.is_zero(&a) {
            continue;
        }
        let mut current = algorithms::sqr_mul::generic_abs_square_and_multiply(a, &d, &ZZ, |a, b| Zn.mul(a, b), |a, b| Zn.mul_ref(a, b), Zn.one());
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
use crate::primitive_int::*;

#[test]
pub fn test_is_prime() {
    assert!(is_prime(StaticRing::<i128>::RING, &2, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &3, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &5, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &7, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &11, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &22531, 5));
    assert!(is_prime(StaticRing::<i128>::RING, &417581, 5));

    assert!(!is_prime(StaticRing::<i128>::RING, &4, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &6, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &8, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &9, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &10, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &22532, 5));
    assert!(!is_prime(StaticRing::<i128>::RING, &347584, 5));
}