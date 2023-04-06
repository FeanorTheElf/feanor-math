use crate::divisibility::DivisibilityRingStore;
use crate::euclidean::EuclideanRingStore;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::integer::*;
use crate::algorithms;
use crate::rings::bigint::DefaultBigIntRing;

fn ec_factor_ring() -> RingValue<DefaultBigIntRing> {
    DefaultBigIntRing::RING
}

fn is_prime_power<I: IntegerRingStore>(ZZ: I, n: &El<I>) -> Option<(El<I>, usize)> {
    for n in 1..ZZ.abs_highest_set_bit(n) {
        
    }
    unimplemented!()
}

#[allow(non_snake_case)]
pub fn factor<I: IntegerRingStore>(ZZ: I, n: &El<I>) -> Vec<(El<I>, usize)> {
    let mut result = Vec::new();
    let mut current = n.clone();

    // first we remove small factors
    for p in algorithms::primes::enumerate_primes(StaticRing::<i64>::RING, &1000) {
        let (mut q, mut r) = (current, ZZ.zero());
        let ZZ_p = ZZ.coerce::<StaticRing<i64>>(&StaticRing::<i64>::RING, p);
        let mut count = 0;
        while ZZ.is_zero(&r) {
            count += 1;
            current = q;
            (q, r) = ZZ.euclidean_div_rem(current, &ZZ_p);
        }
        current = ZZ.add(ZZ.mul_ref_snd(q, &ZZ_p), r);
        if count > 1 {
            result.push((ZZ_p, count - 1));
        }
        if ZZ.is_unit(&current) {
            break;
        }
    }

    // then we use EC factor to factor the result
    let mut ungrouped_prime_factors = Vec::new();
    let mut to_factor = Vec::new();
    to_factor.push(current);

    let ec_factor_ZZ = ec_factor_ring();

    while let Some(n) = to_factor.pop() {
        if ZZ.is_unit(&n) {
            continue;
        } if !algorithms::miller_rabin::is_prime(&ZZ, &n, 6) {
            let p = ZZ.coerce::<RingValue<DefaultBigIntRing>>(&ec_factor_ZZ, algorithms::ec_factor::lenstra_ec_factor(&ec_factor_ZZ, &ec_factor_ZZ.coerce::<I>(&ZZ, n.clone())));
            to_factor.push(ZZ.checked_div(&n, &p).unwrap());
            to_factor.push(p);
        } else {
            ungrouped_prime_factors.push(n);
        }
    }

    // and finally group the primes
    if ungrouped_prime_factors.len() == 0 {
        return result;
    }
    ungrouped_prime_factors.sort_by(|a, b| ZZ.cmp(a, b));
    result.push((ungrouped_prime_factors.pop().unwrap(), 1));
    while let Some(p) = ungrouped_prime_factors.pop() {
        if ZZ.eq(&p, &result.last().unwrap().0) {
            result.last_mut().unwrap().1 += 1;
        } else {
            result.push((p, 1));
        }
    }
    return result;
}

#[test]
fn test_factor() {
    assert_eq!(vec![(3, 2), (5, 1), (29, 1)], factor(StaticRing::<i64>::RING, &(3 * 3 * 5 * 29)));
    assert_eq!(vec![(2, 8)], factor(StaticRing::<i64>::RING, &256));
    assert_eq!(vec![(1009, 2)], factor(StaticRing::<i64>::RING, &(1009 * 1009)));
}