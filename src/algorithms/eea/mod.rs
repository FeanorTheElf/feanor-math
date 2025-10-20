use crate::pid::*;
use crate::integer::*;
use crate::ordered::{OrderedRingStore, OrderedRing};
use crate::ring::*;

use std::mem::swap;
use std::cmp::Ordering;

///
/// For `a, b` computes `s, t, d` such that `s*a + t*b == d` is a greatest 
/// common divisor of `a` and `b`. 
/// 
/// The gcd `d` is only unique up to units, and `s, t` are not unique at all.
/// No guarantees are given on which of these solutions is returned. For integers, 
/// see [`signed_eea()`] which gives more guarantees.
/// 
/// Note that this function always uses the euclidean algorithm to compute these values.
/// In most cases, it is instead recommended to use [`PrincipalIdealRing::extended_ideal_gen()`], 
/// which uses a ring-specific algorithm to compute the Bezout identity (which will of
/// course be [`eea()`] in some cases).
/// 
pub fn eea<R>(a: El<R>, b: El<R>, ring: R) -> (El<R>, El<R>, El<R>) 
    where R: RingStore,
        R::Type: EuclideanRing
{
    let (mut a, mut b) = (a, b);

    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());

    while !ring.is_zero(&b) {
        // loop invariants; unfortunately, this evaluation might cause an integer overflow
        // debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &fst), ring.mul_ref(&ta, &snd))));
        // debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &fst), ring.mul_ref(&tb, &snd))));

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quo, &tb));
        sa = ring.sub(sa, ring.mul_ref_snd(quo, &sb));
        a = rem;

        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return (sa, ta, a);
}

#[stability::unstable(feature = "enable")]
pub const fn const_eea(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut a, mut b) = (a, b);

    let (mut sa, mut ta) = (1, 0);
    let (mut sb, mut tb) = (0, 1);

    while b != 0 {
        let (quo, rem) = (a / b, a % b);
        ta -= quo * tb;
        sa -= quo * sb;
        a = rem;

        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return (sa, ta, a);
}

///
/// Computes the gcd `d` of `a` and `b`, together with "half a Bezout identity", i.e.
/// some `s` such that `s * a = d mod b`.
/// 
/// For details, see [`eea()`].
/// 
#[stability::unstable(feature = "enable")]
pub fn half_eea<R>(a: El<R>, b: El<R>, ring: R) -> (El<R>, El<R>) 
    where R: RingStore,
        R::Type: EuclideanRing
{
    let (mut a, mut b) = (a, b);
    let (mut s, mut t) = (ring.one(), ring.zero());

    // invariant: `s * a == a mod b` and `t * a == b mod b`
    while !ring.is_zero(&b) {
        let (q, r) = ring.euclidean_div_rem(a, &b);
        a = r;
        ring.sub_assign(&mut s, ring.mul_ref_snd(q, &t));
        swap(&mut a, &mut b);
        swap(&mut s, &mut t);
    }
    return (s, a);
}

/// 
/// For integers a, b finds the smallest integers s, t so that 
/// `s*a + t*b == gcd(a, b)` is the greatest common divisor of a, b.
/// 
/// Details: s and t are not unique, this function will return 
/// the smallest tuple (s, t) (ordered by the total ordering
/// given by `(s, t) ≤ (u, v) :<=> |s| ≤ |u| and |t| ≤ |v|`). 
/// In the case |a| = |b|, there are two minimal elements, in this case, it is 
/// unspecified whether this function returns (±1, 0, a) or (0, ±1, a). 
/// We define the greatest common divisor gcd(a, b) as the minimal
/// element of the set of integers dividing a and b (ordered by divisibility), 
/// whose sign matches the sign of a.
/// 
/// In particular, have 
/// ```rust
/// # use feanor_math::algorithms::eea::signed_gcd;
/// # use feanor_math::primitive_int::*;
/// assert_eq!(2, signed_gcd(6, 8, &StaticRing::<i64>::RING));
/// assert_eq!(0, signed_gcd(0, 0, &StaticRing::<i64>::RING)); 
/// assert_eq!(5, signed_gcd(0, -5, &StaticRing::<i64>::RING));
/// assert_eq!(-5, signed_gcd(-5, 0, &StaticRing::<i64>::RING)); 
/// assert_eq!(-1, signed_gcd(-1, 1, &StaticRing::<i64>::RING));
/// assert_eq!(1, signed_gcd(1, -1, &StaticRing::<i64>::RING));
/// ```
/// and therefore `signed_eea(6, 8) == (-1, 1, 2)`, 
/// `signed_eea(-6, 8) == (-1, -1, -2)`, 
/// `signed_eea(8, -6) == (1, 1, 2)`, 
/// `signed_eea(0, 0) == (0, 0, 0)`
/// 
pub fn signed_eea<R>(fst: El<R>, snd: El<R>, ring: R) -> (El<R>, El<R>, El<R>)
    where R: RingStore,
        R::Type: EuclideanRing + OrderedRing
{
    if ring.is_zero(&fst) {
        return match ring.cmp(&snd, &ring.zero()) {
            Ordering::Equal => (ring.zero(), ring.zero(), ring.zero()),
            Ordering::Less => (ring.zero(), ring.negate(ring.one()), ring.negate(snd)),
            Ordering::Greater => (ring.zero(), ring.one(), snd)
        };
    }
    let fst_negative = ring.cmp(&fst, &ring.zero());

    let (s, t, d) = eea(fst, snd, &ring);
    
    // the sign is not consistent (potentially toggled each iteration), 
    // so normalize here
    if ring.cmp(&d, &ring.zero()) == fst_negative {
        return (s, t, d);
    } else {
        return (ring.negate(s), ring.negate(t), ring.negate(d));
    }
}

/// 
/// Finds a greatest common divisor of a and b.
///  
/// The gcd of two elements `a, b` in a euclidean ring is the (w.r.t divisibility) greatest
/// element that divides both elements, i.e. the greatest element (w.r.t. divisibility) `g` such 
/// that `g | a, b`.
/// 
/// Note that this function always uses the euclidean algorithm to compute the gcd. In most
/// cases, it is instead recommended to use [`PrincipalIdealRing::ideal_gen()`], which uses
/// a ring-specific algorithm to compute the gcd (which will of course be [`gcd()`] in some cases).
/// 
/// In general, the gcd is only unique up to multiplication by units. For integers, the function
/// [`signed_gcd()`] gives more guarantees.
/// 
#[stability::unstable(feature = "enable")]
pub fn gcd<R>(a: El<R>, b: El<R>, ring: R) -> El<R>
    where R: RingStore,
        R::Type: EuclideanRing
{
    let (mut a, mut b) = (a, b);
    
    // invariant: `gcd(a, b) = gcd(original_a, original_b)`
    while !ring.is_zero(&b) {
        let (_, r) = ring.euclidean_div_rem(a, &b);
        a = b;
        b = r;
    }
    return a;
}

/// 
/// Finds the greatest common divisor of a and b.
/// 
/// The gcd is only unique up to multiplication by units, so in this case up to sign.
/// However, this function guarantees the following behavior w.r.t different signs:
/// 
/// ```text
///   a < 0 => gcd(a, b) < 0
///   a > 0 => gcd(a, b) > 0
///   sign of b is irrelevant
///   gcd(0, 0) = 0
/// ```
/// 
pub fn signed_gcd<R>(a: El<R>, b: El<R>, ring: R) -> El<R>
    where R: RingStore,
        R::Type: EuclideanRing + OrderedRing
{
    let (_, _, d) = signed_eea(a, b, ring);
    return d;
}

///
/// Finds the least common multiple of two elements in an ordered euclidean ring,
/// e.g. of two integers.
/// 
/// The general lcm is only unique up to multiplication by units. For `signed_lcm`,
/// the following behavior is guaranteed:
/// ```text
///   b > 0 => lcm(a, b) >= 0
///   b < 0 => lcm(a, b) <= 0
///   lcm(0, b) = lcm(a, 0) = lcm(0, 0) = 0
/// ```
/// 
pub fn signed_lcm<R>(fst: El<R>, snd: El<R>, ring: R) -> El<R>
    where R: RingStore,
        R::Type: EuclideanRing + OrderedRing
{
    if ring.is_zero(&fst) || ring.is_zero(&snd) {
        ring.zero()
    } else {
        ring.mul(ring.euclidean_div(ring.clone_el(&fst), &signed_gcd(fst, ring.clone_el(&snd), &ring)), snd)
    }
}

///
/// Finds the least common multiple of two elements `a, b` in a euclidean ring, i.e. the smallest
/// (w.r.t. divisibility) element `y` with `a, b | y`.
/// 
/// In general, the lcm is only unique up to multiplication by units. For integers, the function
/// [`signed_lcm()`] gives more guarantees.
/// 
#[stability::unstable(feature = "enable")]
pub fn lcm<R>(fst: El<R>, snd: El<R>, ring: R) -> El<R>
    where R: RingStore,
        R::Type: EuclideanRing
{
    if ring.is_zero(&fst) || ring.is_zero(&snd) {
        ring.zero()
    } else {
        ring.euclidean_div(ring.mul_ref(&fst, &snd), &gcd(fst, snd, &ring))
    }
}

///
/// Computes x such that `x = a mod p` and `x = b mod q`. Requires that p and q are coprime.
/// 
pub fn inv_crt<I>(a: El<I>, b: El<I>, p: &El<I>, q: &El<I>, ZZ: I) -> El<I>
    where I: RingStore, 
        I::Type: IntegerRing
{
    let (s, t, d) = signed_eea(ZZ.clone_el(p), ZZ.clone_el(q), &ZZ);
    assert!(ZZ.is_one(&d) || ZZ.is_neg_one(&d));
    let mut result = ZZ.add(ZZ.prod([a, t, ZZ.clone_el(q)].into_iter()), ZZ.prod([b, s, ZZ.clone_el(p)].into_iter()));

    let n = ZZ.mul_ref(p, q);
    result = ZZ.euclidean_rem(result, &n);
    if ZZ.is_neg(&result) {
        ZZ.add_assign(&mut result, n);
    }
    return result;
}

#[cfg(test)]
use crate::primitive_int::*;

#[test]
fn test_gcd() {
    assert_eq!(3, gcd(15, 6, &StaticRing::<i64>::RING).abs());
    assert_eq!(3, gcd(6, 15, &StaticRing::<i64>::RING).abs());

    assert_eq!(7, gcd(0, 7, &StaticRing::<i64>::RING).abs());
    assert_eq!(7, gcd(7, 0, &StaticRing::<i64>::RING).abs());
    assert_eq!(0, gcd(0, 0, &StaticRing::<i64>::RING).abs());

    assert_eq!(1, gcd(9, 1, &StaticRing::<i64>::RING).abs());
    assert_eq!(1, gcd(1, 9, &StaticRing::<i64>::RING).abs());

    assert_eq!(1, gcd(13, 300, &StaticRing::<i64>::RING).abs());
    assert_eq!(1, gcd(300, 13, &StaticRing::<i64>::RING).abs());

    assert_eq!(3, gcd(-15, 6, &StaticRing::<i64>::RING).abs());
    assert_eq!(3, gcd(6, -15, &StaticRing::<i64>::RING).abs());
    assert_eq!(3, gcd(-6, -15, &StaticRing::<i64>::RING).abs());
}

#[test]
fn test_signed_gcd() {
    assert_eq!(3, signed_gcd(15, 6, &StaticRing::<i64>::RING));
    assert_eq!(3, signed_gcd(6, 15, &StaticRing::<i64>::RING));

    assert_eq!(7, signed_gcd(0, 7, &StaticRing::<i64>::RING));
    assert_eq!(7, signed_gcd(7, 0, &StaticRing::<i64>::RING));
    assert_eq!(0, signed_gcd(0, 0, &StaticRing::<i64>::RING));

    assert_eq!(1, signed_gcd(9, 1, &StaticRing::<i64>::RING));
    assert_eq!(1, signed_gcd(1, 9, &StaticRing::<i64>::RING));

    assert_eq!(1, signed_gcd(13, 300, &StaticRing::<i64>::RING));
    assert_eq!(1, signed_gcd(300, 13, &StaticRing::<i64>::RING));

    assert_eq!(-3, signed_gcd(-15, 6, &StaticRing::<i64>::RING));
    assert_eq!(3, signed_gcd(6, -15, &StaticRing::<i64>::RING));
    assert_eq!(-3, signed_gcd(-6, -15, &StaticRing::<i64>::RING));
}

#[test]
fn test_eea_sign() {
    assert_eq!((2, -1, 1), signed_eea(3, 5, &StaticRing::<i64>::RING));
    assert_eq!((-1, 2, 1), signed_eea(5, 3, &StaticRing::<i64>::RING));
    assert_eq!((2, 1, -1), signed_eea(-3, 5, &StaticRing::<i64>::RING));
    assert_eq!((-1, -2, 1), signed_eea(5, -3, &StaticRing::<i64>::RING));
    assert_eq!((2, 1, 1), signed_eea(3, -5, &StaticRing::<i64>::RING));
    assert_eq!((-1, -2, -1), signed_eea(-5, 3, &StaticRing::<i64>::RING));
    assert_eq!((2, -1, -1), signed_eea(-3, -5, &StaticRing::<i64>::RING));
    assert_eq!((-1, 2, -1), signed_eea(-5, -3, &StaticRing::<i64>::RING));
    assert_eq!((0, 0, 0), signed_eea(0, 0, &StaticRing::<i64>::RING));
    assert_eq!((1, 0, 4), signed_eea(4, 0, &StaticRing::<i64>::RING));
    assert_eq!((0, 1, 4), signed_eea(0, 4, &StaticRing::<i64>::RING));
    assert_eq!((1, 0, -4), signed_eea(-4, 0, &StaticRing::<i64>::RING));
    assert_eq!((0, -1, 4), signed_eea(0, -4, &StaticRing::<i64>::RING));
}

#[test]
fn test_signed_eea() {
    assert_eq!((-1, 1, 2), signed_eea(6, 8, &StaticRing::<i64>::RING));
    assert_eq!((2, -1, 5), signed_eea(15, 25, &StaticRing::<i64>::RING));
    assert_eq!((4, -7, 2), signed_eea(32, 18, &StaticRing::<i64>::RING));
}

#[test]
fn test_signed_lcm() {
    assert_eq!(24, signed_lcm(6, 8, &StaticRing::<i64>::RING));
    assert_eq!(24, signed_lcm(-6, 8, &StaticRing::<i64>::RING));
    assert_eq!(-24, signed_lcm(6, -8, &StaticRing::<i64>::RING));
    assert_eq!(-24, signed_lcm(-6, -8, &StaticRing::<i64>::RING));
    assert_eq!(0, signed_lcm(0, 0, &StaticRing::<i64>::RING));
    assert_eq!(0, signed_lcm(6, 0, &StaticRing::<i64>::RING));
    assert_eq!(0, signed_lcm(0, 8, &StaticRing::<i64>::RING));
}
