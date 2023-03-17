use crate::ordered::OrderedRingWrapper;
use crate::ring::*;
use crate::integer::*;
use crate::algorithms;
use crate::rings::zn::zn_dyn::*;

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
#[allow(non_snake_case)]
pub fn is_prime<I>(ring: I, n: &El<I>, k: usize) -> bool 
    where I: IntegerRingWrapper + HashableElRingWrapper
{
    if ring.is_leq(n, &ring.from_z(2)) {
        return ring.eq(n, &ring.from_z(2));
    }

    let mut rng = oorandom::Rand64::new(ring.default_hash(n) as u128);
    let mut n_minus_one = ring.sub_ref_fst(n, ring.one());
    let s = ring.abs_lowest_set_bit(&n_minus_one).unwrap();
    ring.euclidean_div_pow_2(&mut n_minus_one, s as usize);
    let d = n_minus_one;
    let Zn = Zn::new(&ring, n.clone());

    for _i in 0..k {
        let a = Zn.project(ring.add(ring.get_uniformly_random(n, || rng.rand_u64()), ring.one()));
        let mut current = algorithms::sqr_mul::generic_abs_square_and_multiply(&a, &d, &ring, |a, b| Zn.mul(a, b), |a, b| Zn.mul_ref(a, b), Zn.one());
        let mut miller_rabin_condition = Zn.is_one(&current);
        for _r in 0..s {
            miller_rabin_condition |= Zn.is_neg_one(&current);
            if miller_rabin_condition {
                break;
            }
            current = Zn.mul(current.clone(), current);
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