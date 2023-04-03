use crate::euclidean::EuclideanRingStore;
use crate::ordered::OrderedRingStore;
use crate::ring::*;

use std::mem::swap;
use std::cmp::Ordering;

///
/// For a, b computes s, t, d such that `s*a + t*b == d` is a greatest 
/// common divisor of a and b. d is only unique up to units, and s, t 
/// are not unique at all. No guarantees are given on which
/// of these solutions is returned. For integers, see signed_eea 
/// which gives more guarantees.
/// 
/// The given ring must be euclidean
/// 
pub fn eea<R>(fst: El<R>, snd: El<R>, ring: R) -> (El<R>, El<R>, El<R>) 
    where R: EuclideanRingStore
{
    let (mut a, mut b) = (fst, snd);
    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());

    // invariant: sa * a + ta * b = fst, sb * a + tb * b = snd
    while !ring.eq(&b, &ring.zero()) {
        let (quot, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quot, &tb));
        sa = ring.sub(sa, ring.mul_ref(&quot, &sb));
        a = rem;
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return (sa, ta, a);
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
/// ```
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
    where R: EuclideanRingStore + OrderedRingStore
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
/// The gcd of two elements in a euclidean ring is the (w.r.t divisibility) greatest
/// element that divides both elements. It is unique up to multiplication with units. 
/// This function makes no guarantees on which of these will be returned.
/// 
/// If this is required, see also signed_gcd that gives precise statement on the
/// sign of the gcd in case of two integers.
/// 
/// The given ring must be euclidean
/// 
pub fn gcd<R>(a: El<R>, b: El<R>, ring: R) -> El<R>
    where R: EuclideanRingStore
{
    let (_, _, d) = eea(a, b, ring);
    return d;
}

/// 
/// Finds the greatest common divisor of a and b.
/// 
/// The gcd is only unique up to multiplication by units, so in this case up to sign.
/// However, this function guarantees the following behavior w.r.t different signs:
/// 
/// ```text
/// a < 0 => gcd(a, b) < 0
/// a > 0 => gcd(a, b) > 0
/// sign of b is irrelevant
/// gcd(0, 0) = 0
/// ```
/// 
pub fn signed_gcd<R>(a: El<R>, b: El<R>, ring: R) -> El<R>
    where R: EuclideanRingStore + OrderedRingStore
{
    let (_, _, d) = signed_eea(a, b, ring);
    return d;
}

pub fn signed_lcm<R>(fst: El<R>, snd: El<R>, ring: R) -> El<R>
    where R: EuclideanRingStore + OrderedRingStore
{
    ring.mul(ring.euclidean_div(fst.clone(), &signed_gcd(fst, snd.clone(), &ring)), snd)
}

pub fn lcm<R>(fst: El<R>, snd: El<R>, ring: R) -> El<R>
    where R: EuclideanRingStore
{
    ring.euclidean_div(ring.mul_ref(&fst, &snd), &gcd(fst, snd, &ring))
}

#[cfg(test)]
use crate::primitive_int::*;

#[test]
fn test_gcd() {
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