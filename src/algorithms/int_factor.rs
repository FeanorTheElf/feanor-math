use crate::algorithms::ec_factor::lenstra_ec_factor;
use crate::computation::no_error;
use crate::computation::DontObserve;
use crate::divisibility::DivisibilityRingStore;
use crate::ordered::OrderedRing;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::algorithms;
use crate::rings::zn::choose_zn_impl;
use crate::rings::zn::ZnOperation;
use crate::rings::zn::ZnRing;
use crate::rings::zn::ZnRingStore;
use crate::DEFAULT_PROBABILISTIC_REPETITIONS;

struct ECFactorInt<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing 
{
    result_ring: I
}

impl<I> ZnOperation for ECFactorInt<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    type Output<'a> = El<I>
        where Self: 'a;

    fn call<'a, R>(self, ring: R) -> El<I>
        where R: 'a + ZnRingStore, R::Type: ZnRing
    {
        int_cast(lenstra_ec_factor(&ring, DontObserve).unwrap_or_else(no_error), self.result_ring, ring.integer_ring())
    }
}

pub fn is_prime_power<I>(ZZ: I, n: &El<I>) -> Option<(El<I>, usize)>
    where I: IntegerRingStore + Copy,
        I::Type: IntegerRing
{
    if algorithms::miller_rabin::is_prime(ZZ, n, DEFAULT_PROBABILISTIC_REPETITIONS) {
        return Some((ZZ.clone_el(n), 1));
    }
    let (p, e) = is_power(ZZ, n)?;
    if algorithms::miller_rabin::is_prime(ZZ, &p, DEFAULT_PROBABILISTIC_REPETITIONS) {
        return Some((p, e));
    } else {
        return None;
    }
}

fn is_power<I>(ZZ: I, n: &El<I>) -> Option<(El<I>, usize)>
    where I: IntegerRingStore + Copy,
        I::Type: IntegerRing
{
    assert!(!ZZ.is_zero(n));
    for i in (2..=ZZ.abs_log2_ceil(n).unwrap()).rev() {
        let root = algorithms::int_bisect::root_floor(ZZ, ZZ.clone_el(n), i);
        if ZZ.eq_el(&ZZ.pow(root, i), n) {
            return Some((algorithms::int_bisect::root_floor(ZZ, ZZ.clone_el(n), i), i));
        }
    }
    return None;
}

pub fn factor<I>(ZZ: I, mut n: El<I>) -> Vec<(El<I>, usize)> 
    where I: IntegerRingStore + OrderedRingStore + Copy, 
        I::Type: IntegerRing + OrderedRing + CanIsoFromTo<BigIntRingBase> + CanIsoFromTo<StaticRingBase<i128>>
{
    const SMALL_PRIME_BOUND: i32 = 10000;
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
    } else if algorithms::miller_rabin::is_prime(ZZ, &n, DEFAULT_PROBABILISTIC_REPETITIONS) {
        result.push((n, 1));
        return result;
    }

    // then we remove small factors
    for p in algorithms::erathostenes::enumerate_primes(StaticRing::<i32>::RING, &SMALL_PRIME_BOUND) {
        let ZZ_p = ZZ.int_hom().map(p);
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
    } else if algorithms::miller_rabin::is_prime(ZZ, &n, DEFAULT_PROBABILISTIC_REPETITIONS) {
        result.push((n, 1));
        return result;
    }

    // then check for powers, as EC factor fails for those
    if let Some((m, k)) = is_power(ZZ, &n) {
        let mut power_factors = factor(ZZ, m);
        for (_, multiplicity) in &mut power_factors {
            *multiplicity *= k;
        }
        result.extend(power_factors.into_iter());
        return result;
    }

    // then we use EC factor to factor the result
    let m = choose_zn_impl(ZZ, ZZ.clone_el(&n), ECFactorInt { result_ring: ZZ });

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

    let expected = vec![(ZZbig.int_hom().map(-1), 1), (ZZbig.int_hom().map(32771), 1), (ZZbig.int_hom().map(65537), 1)];
    let actual = factor(&ZZbig, ZZbig.mul(ZZbig.int_hom().map(-32771), ZZbig.int_hom().map(65537)));
    assert_eq!(expected.len(), actual.len());
    for ((expected_factor, expected_multiplicity), (actual_factor, actual_multiplicity)) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected_multiplicity, actual_multiplicity);
        assert!(ZZbig.eq_el(expected_factor, actual_factor));
    }

    let expected = vec![(ZZbig.int_hom().map(257), 2), (ZZbig.int_hom().map(32771), 1), (ZZbig.int_hom().map(65537), 2)];
    let actual = factor(&ZZbig, ZZbig.prod([ZZbig.int_hom().map(257 * 257), ZZbig.int_hom().map(32771), ZZbig.int_hom().map(65537), ZZbig.int_hom().map(65537)].into_iter()));
    assert_eq!(expected.len(), actual.len());
    for ((expected_factor, expected_multiplicity), (actual_factor, actual_multiplicity)) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected_multiplicity, actual_multiplicity);
        assert!(ZZbig.eq_el(expected_factor, actual_factor));
    }
}

#[test]
fn test_is_prime_power() {
    assert_eq!(Some((2, 6)), is_prime_power(&StaticRing::<i64>::RING, &64));
}

#[test]
fn test_is_prime_power_large_n() {
    assert_eq!(Some((5, 25)), is_prime_power(&StaticRing::<i64>::RING, &StaticRing::<i64>::RING.pow(5, 25)));
}