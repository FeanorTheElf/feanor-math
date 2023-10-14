use crate::algorithms::ec_factor::lenstra_ec_factor;
use crate::divisibility::DivisibilityRingStore;
use crate::ordered::OrderedRing;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::integer::*;
use crate::algorithms;
use crate::rings::zn::choose_zn_impl;
use crate::generate_zn_function;

fn is_power<I: IntegerRingStore>(ZZ: &I, n: &El<I>) -> Option<(El<I>, usize)>
    where I::Type: IntegerRing
{
    for i in 2..ZZ.abs_log2_ceil(n).unwrap() {
        let root = algorithms::int_bisect::root_floor(ZZ, ZZ.clone_el(n), i);
        if ZZ.eq_el(&ZZ.pow(root, i), n) {
            return Some((algorithms::int_bisect::root_floor(ZZ, ZZ.clone_el(n), i), i));
        }
    }
    return None;
}

pub fn factor<I>(ZZ: &I, mut n: El<I>) -> Vec<(El<I>, usize)> 
    where I: IntegerRingStore + OrderedRingStore, 
        I::Type: IntegerRing + OrderedRing + CanonicalIso<BigIntRingBase> + CanonicalIso<StaticRingBase<i128>>
{
    const SMALL_PRIME_BOUND: i32 = 1000;
    let mut result = Vec::new();

    // first make it nonnegative
    if ZZ.is_neg(&n) {
        result.push((ZZ.neg_one(), 1));
        ZZ.negate_inplace(&mut n);
    }

    // check for special cases
    if ZZ.is_zero(&n) {
        result.push((ZZ.zero(), 1));
        return result;
    }

    // check if we are done
    if ZZ.is_one(&n) {
        return result;
    } else if algorithms::miller_rabin::is_prime(ZZ, &n, 8) {
        result.push((n, 1));
        return result;
    }

    // then we remove small factors
    for p in algorithms::erathostenes::enumerate_primes(StaticRing::<i32>::RING, &SMALL_PRIME_BOUND) {
        let ZZ_p = ZZ.coerce(&StaticRing::<i32>::RING, p);
        let mut count = 0;
        while let Some(quo) = ZZ.checked_div(&n, &ZZ_p) {
            n = quo;
            count += 1;
        }
        if count >= 1 {
            result.push((ZZ_p, count));
        }
    }

    // check again if we are done
    if ZZ.is_one(&n) {
        return result;
    } else if algorithms::miller_rabin::is_prime(ZZ, &n, 8) {
        result.push((n, 1));
        return result;
    }

    // then check for powers, as EC factor fails for those
    if let Some((m, k)) = is_power(&ZZ, &n) {
        let mut power_factors = factor(ZZ, m);
        for (_, multiplicity) in &mut power_factors {
            *multiplicity *= k;
        }
        result.extend(power_factors.into_iter());
        return result;
    }

    // then we use EC factor to factor the result
    let mut m = None;
    choose_zn_impl(
        ZZ, 
        ZZ.clone_el(&n), 
        generate_zn_function!{ 
            <{'a}, {I: IntegerRingStore<Type = J>}, {J: ?Sized + IntegerRing}> 
            [_: &'a mut Option<El<I>> = &mut m, _: &'a I = ZZ] 
            |ring, (result, ZZ): (&mut Option<El<I>>, &I)| {
                *result = Some(int_cast(
                    lenstra_ec_factor::<&R>(&ring), ZZ, ring.integer_ring()
                ));
            }
        }
    );
    let m = m.unwrap();
    let mut factors1 = factor(ZZ, ZZ.checked_div(&n, &m).unwrap());
    let mut factors2 = factor(ZZ, m);

    // finally group the prime factors
    factors1.sort_by(|(a, _), (b, _)| ZZ.cmp(a, b));
    factors2.sort_by(|(a, _), (b, _)| ZZ.cmp(a, b));
    let mut iter1 = factors1.into_iter().peekable();
    let mut iter2 = factors2.into_iter().peekable();
    loop {
        match (iter1.peek(), iter2.peek()) {
            (Some((p1, m1)), Some((p2, m2))) if ZZ.eq_el(p1, p2) => {
                result.push((ZZ.clone_el(p1), m1 + m2));
                iter1.next();
                iter2.next();
            },
            (Some((p1, m1)), Some((p2, _m2))) if ZZ.is_lt(p1, p2) => {
                result.push((ZZ.clone_el(p1), *m1));
                iter1.next();
            },
            (Some((_p1, _m1)), Some((p2, m2))) => {
                result.push((ZZ.clone_el(p2), *m2));
                iter2.next();
            },
            (Some((p1, m1)), None) => {
                result.push((ZZ.clone_el(p1), *m1));
                iter1.next();
            },
            (None, Some((p2, m2))) => {
                result.push((ZZ.clone_el(p2), *m2));
                iter2.next();
            },
            (None, None) => {
                return result;
            }
        }
    }
}

#[test]
fn test_factor() {
    let ZZbig = BigIntRing::RING;
    assert_eq!(vec![(3, 2), (5, 1), (29, 1)], factor(&StaticRing::<i64>::RING, 3 * 3 * 5 * 29));
    assert_eq!(vec![(2, 8)], factor(&StaticRing::<i64>::RING, 256));
    assert_eq!(vec![(1009, 2)], factor(&StaticRing::<i64>::RING, 1009 * 1009));
    assert_eq!(vec![(0, 1)], factor(&StaticRing::<i64>::RING, 0));
    assert_eq!(Vec::<(i64, usize)>::new(), factor(&StaticRing::<i64>::RING, 1));
    assert_eq!(vec![(-1, 1)], factor(&StaticRing::<i64>::RING, -1));
    assert_eq!(vec![(257, 1), (1009, 2)], factor(&StaticRing::<i128>::RING, 257 * 1009 * 1009));

    let expected = vec![(ZZbig.from_int(-1), 1), (ZZbig.from_int(32771), 1), (ZZbig.from_int(65537), 1)];
    let actual = factor(&ZZbig, ZZbig.mul(ZZbig.from_int(-32771), ZZbig.from_int(65537)));
    assert_eq!(expected.len(), actual.len());
    for ((expected_factor, expected_multiplicity), (actual_factor, actual_multiplicity)) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected_multiplicity, actual_multiplicity);
        assert!(ZZbig.eq_el(expected_factor, actual_factor));
    }

    let expected = vec![(ZZbig.from_int(257), 2), (ZZbig.from_int(32771), 1), (ZZbig.from_int(65537), 2)];
    let actual = factor(&ZZbig, ZZbig.prod([ZZbig.from_int(257 * 257), ZZbig.from_int(32771), ZZbig.from_int(65537), ZZbig.from_int(65537)].into_iter()));
    assert_eq!(expected.len(), actual.len());
    for ((expected_factor, expected_multiplicity), (actual_factor, actual_multiplicity)) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected_multiplicity, actual_multiplicity);
        assert!(ZZbig.eq_el(expected_factor, actual_factor));
    }
}