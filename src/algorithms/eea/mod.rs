use crate::pid::*;
use crate::ring::*;

use std::mem::swap;

///
/// For `a, b` computes `s, t, d` such that `s*a + t*b == d` is a greatest 
/// common divisor of `a` and `b`. 
/// 
/// In most cases, prefer [`PrincipalIdealRing::extended_ideal_gen()`].
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

///
/// The same as [`eea()`], but defined as const-fn and only for `i128`.
/// 
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
/// In most cases, prefer [`PrincipalIdealRing::extended_ideal_gen()`].
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
/// Finds a greatest common divisor of a and b.
/// 
/// In most cases, prefer [`PrincipalIdealRing::ideal_gen()`].
/// 
/// The gcd of two elements `a, b` in a euclidean ring is the (w.r.t divisibility) greatest
/// element that divides both elements, i.e. the greatest element (w.r.t. divisibility) `g` such 
/// that `g | a, b`.
/// 
/// Note that this function always uses the euclidean algorithm to compute the gcd. In most
/// cases, it is instead recommended to use [`PrincipalIdealRing::ideal_gen()`], which uses
/// a ring-specific algorithm to compute the gcd (which will of course be [`gcd()`] in some cases).
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
