use crate::integer::IntegerRingWrapper;
use crate::ring::*;
use crate::algorithms;

use std::hash::Hash;
use std::collections::HashMap;

///
/// Computes the discrete logarithm of value w.r.t base in the monoid given by
/// op and identity. The parameter `base_order` is only required to be a bound
/// on the size of the discrete logarithm, but in many use cases it will be the
/// order of the base in the monoid.
/// 
pub fn baby_giant_step<T, F, I>(value: T, base: &T, integer_ring: I, base_order: &El<I>, op: F, identity: T) -> Option<El<I>> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq, I: IntegerRingWrapper
{
    let n = base_order.root_floor(2) + 1;
    let mut giant_steps = HashMap::new();
    let giant_step = algorithms::sqr_mul::generic_abs_square_and_multiply(
        base, 
        n, 
        &integer_ring, 
        |a, b| op(a, b), 
        |a, b| op(a.clone(), b.clone()), 
        identity.clone()
    );
    let mut current = identity;
    for j in n.ring().zero()..n.clone() {
        giant_steps.insert(current.clone(), j);
        current = op(current, giant_step.clone());
    }
    current = value;
    for i in n.ring().zero()..n.clone() {
        if let Some(j) = giant_steps.get(&current) {
            return Some(j * n - i);
        }
        current = op(current, base.clone());
    }
    return None;
}

///
/// Computes the coefficients of a linear combination of the given value in terms of
/// the base elements in the monoid given by op and identity. The parameter `base_order`
/// is only required to be a bound on the order of those coefficients, but in most use
/// cases, it will be the order of the base elements.
/// 
pub fn baby_giant_step_2d<T, F, I>(value: T, base: (&T, &T), integer_ring: I, base_order: (&El<I>, &El<I>), op: F, identity: T) -> Option<(El<I>, El<I>)> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq, I: IntegerRingWrapper
{
    let n1 = base_order.0.root_floor(2) + 1;
    let n2 = base_order.1.root_floor(2) + 1;
    let mut giant_steps = HashMap::new();
    let giant_step1 = algorithms::sqr_mul::generic_abs_square_and_multiply(base.0, n1.val(), n1.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    let giant_step2 = algorithms::sqr_mul::generic_abs_square_and_multiply(base.1, n2.val(), n2.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    let mut current1 = identity;
    for k in n1.ring().zero()..n1.clone() {
        let mut current2 = current1.clone();
        for l in n2.ring().zero()..n2.clone() {
            giant_steps.insert(current2.clone(), (k.clone(), l));
            current2 = op(current2, giant_step2.clone());
        }
        current1 = op(current1, giant_step1.clone());
    }
    current1 = value;
    for i in n1.ring().zero()..n1.clone() {
        let mut current2 = current1.clone();
        for j in n2.ring().zero()..n2.clone() {
            if let Some((k, l)) = giant_steps.get(&current2) {
                return Some((k * n1 - i, l * n2 - j));
            }
            current2 = op(current2, base.1.clone());
        }
        current1 = op(current1, base.0.clone());
    }
    return None;
}

fn power_p_discrete_log<T, F, I>(value: T, p_e_base: &T, integer_ring: I, p: &El<I>, e: u32, op: F, identity: T) -> Option<El<I>> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq + std::fmt::Debug, I: IntegerRingWrapper
{
    let pow = |x: &T, e: &El<I>| algorithms::sqr_mul::generic_abs_square_and_multiply(x, e.val(), e.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    let p_base = pow(p_e_base, &p.pow(e - 1));
    debug_assert_ne!(p_base, identity);
    debug_assert_eq!(pow(&p_base, p), identity);
    let mut fill_log = p.ring().zero();
    let mut current = value;
    for i in 0..e {
        let log = baby_giant_step(pow(&current, &p.pow(e - i - 1)), &p_base, &p, &op, identity.clone())?;
        let p_i = p.pow(i);
        let fill = (p - log) * p_i;
        current = op(current, pow(p_e_base, &fill));
        fill_log += fill;
    }
    return Some(p.pow(e) - fill_log);
}

///
/// Computes the coefficients of the linear combination of the given value in terms of
/// the given base elements in the monoid given by op and identity. The base elements
/// here must be linearly independent elements of exact order `p^e.0` resp `p^e.1`.
/// 
fn power_p_discrete_log_2d<T, F, I>(value: T, p_e_base: (&T, &T), integer_ring: I, p: &El<I>, e: (u32, u32), op: F, identity: T) -> Option<(El<I>, El<I>)> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq + std::fmt::Debug, I: IntegerRingWrapper
{
    if e.0 > e.1 {
        let (a, b) = power_p_discrete_log_2d(value, (p_e_base.1, p_e_base.0), integer_ring, p, (e.1, e.0), op, identity)?;
        return Some((b, a));
    }
    if e.0 == 0 {
        return Some((p.ring().zero(), power_p_discrete_log(value, p_e_base.1, integer_ring, p, e.1, op, identity)?));
    }
    let pow = |x: &T, e: &El<I>| algorithms::sqr_mul::generic_abs_square_and_multiply(x, e.val(), e.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    let p_base = (pow(p_e_base.0, &p.pow(e.0 - 1)), pow(p_e_base.1, &p.pow(e.1 - 1)));
    debug_assert_ne!(p_base.0, identity);
    debug_assert_ne!(p_base.1, identity);
    debug_assert_eq!(pow(&p_base.0, p), identity);
    debug_assert_eq!(pow(&p_base.1, p), identity);
    let mut fill_log = (p.ring().zero(), p.ring().zero());
    let mut current = value;
    for i in 0..(e.1 - e.0) {
        let log = baby_giant_step(
            pow(&current, &p.pow(e.1 - i - 1)), 
            &p_base.1, 
            integer_ring,
            &p, 
            &op, 
            identity.clone()
        )?;
        let p_i = p.pow(i);
        let fill = (p - log) * p_i;
        current = op(current, pow(p_e_base.1, &fill));
        fill_log.1 += fill;
    }
    for i in 0..e.0 {
        let log = baby_giant_step_2d(
            pow(&current, &p.pow(e.0 - i - 1)), 
            (&p_base.0, &p_base.1), 
            integer_ring,
            (&p, &p), 
            &op, 
            identity.clone()
        )?;
        let p_i = (p.pow(i), p.pow(i + e.1 - e.0));
        let fill = ((p - log.0) * p_i.0, (p - log.1) * p_i.1);
        current = op(current, op(pow(p_e_base.0, &fill.0), pow(p_e_base.1, &fill.1)));
        fill_log.0 += fill.0;
        fill_log.1 += fill.1;
    }
    return Some((p.pow(e.0) - fill_log.0, p.pow(e.1) - fill_log.1));
}

///
/// Computes x such that `x = a mod p` and `x = b mod q`. Requires that p and q are coprime.
/// 
fn crt<I>(a: El<I>, b: El<I>, p: &El<I>, q: &El<I>, integer_ring: I) -> (El<I>, El<I>)
    where I: IntegerRingWrapper
{
    let (s, t, d) = algorithms::eea::signed_eea(p.clone(), q.clone(), &p.ring());
    assert!(d == 1 || d == -1);
    let mut result = a * t * q + b * s * p;

    let n = p * q;
    result = result % &n;
    if result < 0 {
        result += &n;
    }
    return (result, n);
}

///
/// Computes the discrete logarithm of value w.r.t the given base in the monoid given by op and identity.
/// It is required that `order` is the order of the base element and this is finite. If the given value is
/// not contained in the submonoid generated by the base element, then None is returned.
/// 
pub fn discrete_log<T, F, I>(value: T, base: &T, integer_ring: I, order: &El<I>, op: F, identity: T) -> Option<El<I>> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq + std::fmt::Debug, I: IntegerRingWrapper + UfdInfoRing
{
    let pow = |x: &T, e: &RingElWrapper<I>| abs_square_and_multiply(x, e.val(), e.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    debug_assert!(pow(&base, &order) == identity);
    let mut current_log = order.ring().one();
    let mut current_size = order.ring().one();
    for (p, e) in order.clone().factor() {
        let size = p.pow(e as u32);
        let power = order / &size;
        let log = power_p_discrete_log(
            pow(&value, &power), 
            &pow(&base, &power), 
            integer_ring,
            &p,
            e as u32, 
            &op, 
            identity.clone()
        )?;
        (current_log, current_size) = crt(log, current_log, &size, &current_size);
    }
    return Some(current_log);
}

///
/// Computes the coefficients of the linear combination of the given value in terms of
/// the given base elements in the monoid given by op and identity. The base elements
/// here must be linearly independent elements of exact order `order`. With linearly
/// independent here we mean that there is no linear combination
/// ```text
/// 0 = a0 * base.0 + a1 * base.1
/// ```
/// where a, b are less than the respective orders of the base elements. In other words,
/// the monoid we are working in must be the direct sum of the monoids generated by
/// `base.0` resp. `base.1`.
/// 
/// If no such linear combination exists (i.e. the value is not contained in the monoid
/// generated by `base.0` and `base.1`), then None is returned.
/// 
/// # Example
/// 
/// ```
/// # use feanor_la::wrapper::*;
/// # use feanor_la::primitive::*;
/// # use feanor_la::fq::zn_small::*;
/// # use feanor_la::discrete_log::*;
/// let e0 = (ZnEl::<45>::project(1), ZnEl::<162>::project(0));
/// let e1 = (ZnEl::<45>::project(0), ZnEl::<162>::project(1));
/// let zero = (ZnEl::<45>::project(0), ZnEl::<162>::project(0));
/// assert_eq!(
///     Some((i64::WRAPPED_RING.wrap(30), i64::WRAPPED_RING.wrap(28))),
///     discrete_log_2d(
///         (ZnEl::<45>::project(30), ZnEl::<162>::project(28)),
///         (&e0, &e1),
///         (&i64::WRAPPED_RING.wrap(45), &i64::WRAPPED_RING.wrap(162)),
///         |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2),
///         zero
///     )
/// );
/// ```
/// 
/// # Complexity
/// 
/// The algorithm is based on a generic baby-step-giant step method, and thus has complexity
/// O(poly(log(n)) * sqrt(p) + T) where n is the product of the orders of the base elements,
/// p is the largest prime factor of n and T is the time required for factoring n.
/// 
pub fn discrete_log_2d<T, F, I>(value: T, base: (&T, &T), order: (&RingElWrapper<I>, &RingElWrapper<I>), op: F, identity: T) -> Option<(RingElWrapper<I>, RingElWrapper<I>)> 
    where F: Fn(T, T) -> T, T: Clone + Hash + Eq + std::fmt::Debug, I: IntegerRing + UfdInfoRing
{
    let pow = |x: &T, e: &RingElWrapper<I>| abs_square_and_multiply(x, e.val(), e.parent_ring(), |a, b| op(a, b), |a, b| op(a.clone(), b.clone()), identity.clone());
    let mut current_log = (order.0.ring().one(), order.1.ring().one());
    let mut current_size = (order.0.ring().one(), order.1.ring().one());
    let factorizations = (order.0.clone().factor(), order.1.clone().factor());
    let mut all_factors = VecMap::new();
    all_factors.extend(factorizations.0.iter());
    all_factors.extend(factorizations.1.iter());

    for (p, _) in all_factors {
        let e = (*factorizations.0.get(p).unwrap_or(&0) as u32, *factorizations.1.get(p).unwrap_or(&0) as u32);
        let size = (p.pow(e.0), p.pow(e.1));
        let power = (order.0 * order.1) / p.pow(e.0 + e.1);
        let log = power_p_discrete_log_2d(
            pow(&value, &power), 
            (&pow(base.0, &power), &pow(base.1, &power)), 
            &p, 
            e, 
            &op, 
            identity.clone()
        )?;
        (current_log.0, current_size.0) = crt(log.0, current_log.0, &size.0, &current_size.0);
        (current_log.1, current_size.1) = crt(log.1, current_log.1, &size.1, &current_size.1);
    }
    return Some(current_log);
}

#[cfg(test)]
use super::fq::zn_small::*;

#[test]
fn test_baby_giant_step() {
    assert_eq!(
        Some(i64::WRAPPED_RING.wrap(6)), 
        baby_giant_step(6, &1, &i64::WRAPPED_RING.wrap(20), |a, b| a + b, 0)
    );
}

#[test]
fn test_baby_giant_step_2d() {
    assert_eq!(
        Some((i64::WRAPPED_RING.wrap(13), i64::WRAPPED_RING.wrap(6))), 
        baby_giant_step_2d::<_, _, StaticRing<i64>>(
            (13, 12), 
            (&(1, 0), &(0, 2)), 
            (&i64::WRAPPED_RING.wrap(20), &i64::WRAPPED_RING.wrap(20)), 
            |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2), 
            (0,0)
        )
    );
}

#[test]
fn test_power_p_discrete_log() {
    assert_eq!(
        Some(i64::WRAPPED_RING.wrap(6)), 
        power_p_discrete_log(
            ZnEl::<81>::project(6), 
            &ZnEl::project(1),
            &i64::WRAPPED_RING.wrap(3), 
            4, 
            |a, b| a + b, 
            ZnEl::project(0),
        )
    );
}

#[test]
fn test_power_p_discrete_log_2d() {
    assert_eq!(
        Some((i64::WRAPPED_RING.wrap(23), i64::WRAPPED_RING.wrap(5))), 
        power_p_discrete_log_2d(
            (ZnEl::<81>::project(23), ZnEl::<9>::project(5)), 
            (&(ZnEl::project(1), ZnEl::project(0)), &(ZnEl::project(0), ZnEl::project(1))),
            &i64::WRAPPED_RING.wrap(3), 
            (4, 2), 
            |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2), 
            (ZnEl::project(0), ZnEl::project(0)),
        )
    );
}

#[test]
fn test_power_p_discrete_log_2d_non_unit_basis() {
    assert_eq!(
        Some((i64::WRAPPED_RING.wrap(-1 /* 8 would be correct as well */), i64::WRAPPED_RING.wrap(6))), 
        power_p_discrete_log_2d(
            (ZnEl::<495>::project(110), ZnEl::<189>::project(42)), 
            (&(ZnEl::project(385), ZnEl::project(0)), &(ZnEl::project(0), ZnEl::project(7))),
            &i64::WRAPPED_RING.wrap(3), 
            (2, 3), 
            |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2), 
            (ZnEl::project(0), ZnEl::project(0)),
        )
    );
}
#[test]
fn test_discrete_log() {
    assert_eq!(Some(i64::WRAPPED_RING.wrap(78)), 
        discrete_log(
            ZnEl::<132>::project(78), 
            &ZnEl::project(1),
            &i64::WRAPPED_RING.wrap(132), 
            |a, b| a + b, 
            ZnEl::project(0),
        )
    );
}

#[test]
fn test_discrete_log_2d() {
    const N: u64 = 3 * 3 * 5 * 11;
    const M: u64 = 3 * 3 * 3 * 7;
    let e0 = (ZnEl::<N>::project(1), ZnEl::<M>::project(0));
    let e1 = (ZnEl::<N>::project(0), ZnEl::<M>::project(1));
    assert_eq!(
        Some((i64::WRAPPED_RING.wrap(16 * 5), i64::WRAPPED_RING.wrap(3 * 4 * 5))),
        discrete_log_2d(
            (ZnEl::<N>::project(16 * 5), ZnEl::<M>::project(3 * 4 * 5)), 
            (&e0, &e1),
            (&i64::WRAPPED_RING.wrap(N as i64), &i64::WRAPPED_RING.wrap(M as i64)), 
            |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2), 
            (ZnEl::project(0), ZnEl::project(0))
        )
    );
}