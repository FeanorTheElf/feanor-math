use std::cmp::Ordering;
use std::mem::swap;

use tracing::instrument;

use crate::prelude::*;
use crate::ring_properties::integer::IntegerRing;
use crate::ring_properties::ordered::OrderedRingStore;

/// For `a, b` computes `s, t, d` such that `s*a + t*b == d` is a greatest
/// common divisor of `a` and `b`.
///
/// In most cases, prefer [`PrincipalIdealRing::extended_ideal_gen()`].
///
/// The gcd `d` is only unique up to units, and `s, t` are not unique at all.
/// No guarantees are given on which of these solutions is returned.
///
/// Note that this function always uses the euclidean algorithm to compute these values.
/// In most cases, it is instead recommended to use [`PrincipalIdealRing::extended_ideal_gen()`],
/// which uses a ring-specific algorithm to compute the Bezout identity (which can often be
/// more efficient than [`general_extended_euclid()`]).
#[instrument(skip_all, level = "trace")]
pub fn general_extended_euclid<R>(a: El<R>, b: El<R>, ring: R) -> (El<R>, El<R>, El<R>)
where
    R: RingStore,
    R::Ring: EuclideanRing,
{
    let (mut a, mut b) = (a, b);
    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());

    while !ring.is_zero(&b) {
        // loop invariants; unfortunately, this evaluation might cause an integer overflow
        // debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &fst), ring.mul_ref(&ta, &snd))));
        // debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &fst), ring.mul_ref(&tb, &snd))));

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        let tb_new = ring.sub(ta, ring.mul_ref(&quo, &tb));
        let sb_new = ring.sub(sa, ring.mul_ref_snd(quo, &sb));
        let b_new = rem;

        ta = tb;
        sa = sb;
        a = b;
        tb = tb_new;
        sb = sb_new;
        b = b_new;
    }
    return (sa, ta, a);
}

/// The same as [`eea()`], but defined as const-fn and only for `i128`.
#[stability::unstable(feature = "enable")]
pub const fn const_extended_euclid(a: i128, b: i128) -> (i128, i128, i128) {
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

/// Computes the gcd `d` of `a` and `b`, together with "half a Bezout identity", i.e.
/// some `s` such that `s * a = d mod b`.
///
/// In most cases, prefer [`PrincipalIdealRing::extended_ideal_gen()`].
///
/// For details, see [`eea()`].
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn half_extended_euclid<R>(a: El<R>, b: El<R>, ring: R) -> (El<R>, El<R>)
where
    R: RingStore,
    R::Ring: EuclideanRing,
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
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn general_euclid<R>(a: El<R>, b: El<R>, ring: R) -> El<R>
where
    R: RingStore,
    R::Ring: EuclideanRing,
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

/// Computes `[s, t, s', t']` such that `x := s * a + t * b` and `x' := s' * a + t' * b`
/// are the current pair of values during the Euclidean Algorithm when, for the first time, we
/// find `|x'| <= target_size`.
///
/// In particular, we have `|x| > target_size` and `|x'| <= target_size`, except if `a, b`
/// already both are `<= target_size` in absolute value.
///
/// The size of `s, t, s', t'` are bounded as
/// ```text
///   |s| <= |b| / |x|
///   |t| <= |a| / |x|
///   |s'| <= |b| / (|x| - |x'|)
///   |t'| <= |a| / (|x| - |x'|)
/// ```
/// except if `min(|a|, |b|) <= target_size`, i.e. no step is performed during the
/// Euclidean algorithm. In such cases, the bounds involving the larger one of `|a|`
/// resp. `|b|` still hold, but the other two don't.
///
/// # Proof
///
/// As in the euclidean algorithm, define `s_0 = 1, t_0 = 0, s_1 = 0, t_1 = 1` and
/// `a_0 = a, a_1 = b`. Then define recursively
/// ```text
///   q_i = floor(a_(i - 1) / a_i),
///   a_(i + 1) = a_(i - 1) - q_i a_i,
///   s_(i + 1) = s_(i - 1) - q_i s_i,
///   t_(i + 1) = t_(i - 1) - q_i t_i
/// ```
/// We assume for the proof that `a > b > 0`.
/// Now observe that
///  - the `a_i` are decreasing
///  - the signs of `s_i` and `t_i` are alternating for `i >= 2`
///  - we have `a_i |s_i| <= |s_2| a_2 q_2 / q_i` and `a_i |t_i| <= |t_2| a_2 q_2 / q_i` for `i >=
///    2`; this can be shown by induction, using the first two points
///  - finally, this implies for `i >= 1` that `a_i |s_i| <= a_1 / q_i` and `a_i |t_i| <= a_0 /
///    q_i`, using that `q_2 a_2 <= a_1` and `q_1 a_1 <= a_0`
///  - unfortunately, it does not apply to `i = 0`
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn partial_extended_euclid_int<R>(ring: R, lhs: El<R>, rhs: El<R>, target_size: &El<R>) -> ([El<R>; 4], [El<R>; 2])
where
    R: RingStore + Copy,
    R::Ring: IntegerRing,
{
    assert!(!ring.is_neg(target_size));
    let (mut a, mut b) = (lhs.clone(), rhs.clone());
    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());

    if ring.abs_cmp(&a, &b) == Ordering::Less {
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    if ring.is_zero(&a) || ring.is_zero(&b) {
        return ([sa, ta, sb, tb], [a, b]);
    }

    while ring.abs_cmp(&b, target_size) == Ordering::Greater {
        // these might trigger integer overflow
        // debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &lhs), ring.mul_ref(&ta, &rhs))));
        // debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &lhs), ring.mul_ref(&tb, &rhs))));

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        let tb_new = ring.sub(ta, ring.mul_ref(&quo, &tb));
        let sb_new = ring.sub(sa, ring.mul_ref_snd(quo, &sb));
        let b_new = rem;

        ta = tb;
        sa = sb;
        a = b;
        tb = tb_new;
        sb = sb_new;
        b = b_new;

        // these might trigger integer overflow
        // debug_assert!(ring.abs_cmp(&ring.mul_ref(&sb, &b), &rhs) != Ordering::Greater);
        // debug_assert!(ring.abs_cmp(&ring.mul_ref(&tb, &b), &lhs) != Ordering::Greater);
    }
    return ([sa, ta, sb, tb], [a, b]);
}

#[test]
fn test_gcd() {
    feanor_tracing::DelayedLogger::init_test();
    assert_eq!(3, general_euclid(15, 6, &ZZi64).abs());
    assert_eq!(3, general_euclid(6, 15, &ZZi64).abs());

    assert_eq!(7, general_euclid(0, 7, &ZZi64).abs());
    assert_eq!(7, general_euclid(7, 0, &ZZi64).abs());
    assert_eq!(0, general_euclid(0, 0, &ZZi64).abs());

    assert_eq!(1, general_euclid(9, 1, &ZZi64).abs());
    assert_eq!(1, general_euclid(1, 9, &ZZi64).abs());

    assert_eq!(1, general_euclid(13, 300, &ZZi64).abs());
    assert_eq!(1, general_euclid(300, 13, &ZZi64).abs());

    assert_eq!(3, general_euclid(-15, 6, &ZZi64).abs());
    assert_eq!(3, general_euclid(6, -15, &ZZi64).abs());
    assert_eq!(3, general_euclid(-6, -15, &ZZi64).abs());
}

#[test]
fn test_partial_int() {
    feanor_tracing::DelayedLogger::init_test();
    let test_on_input = |a: i64, b: i64, size: i64| {
        assert!(a != 0);
        assert!(b != 0);
        let ([s, t, s_, t_], [x, x_]) = partial_extended_euclid_int(ZZi64, a, b, &size);
        assert_eq!(x, s * a + t * b);
        assert_eq!(x_, s_ * a + t_ * b);
        if a.abs() <= size && b.abs() <= size {
            assert!(x.abs() <= size);
            assert!(x_.abs() <= size);
        } else if a.abs() <= size || b.abs() <= size {
            assert!(x.abs() > size);
            assert!(x_.abs() <= size);
            if a.abs() >= b.abs() {
                assert!(t.abs() * x.abs() <= a.abs());
                assert!(t_.abs() * (x.abs() - x_.abs()) <= a.abs());
            } else {
                assert!(s.abs() * x.abs() <= b.abs());
                assert!(s_.abs() * (x.abs() - x_.abs()) <= b.abs());
            }
        } else {
            assert!(x.abs() > size);
            assert!(x_.abs() <= size);
            assert!(t.abs() * x.abs() <= a.abs());
            assert!(t_.abs() * (x.abs() - x_.abs()) <= a.abs());
            assert!(s.abs() * x.abs() <= b.abs());
            assert!(s_.abs() * (x.abs() - x_.abs()) <= b.abs());
        }
    };
    for i in 0..9 {
        test_on_input(6, 9, i);
        test_on_input(9, 6, i);
        test_on_input(7, 9, i);
        test_on_input(9, 7, i);
        test_on_input(7, 1, i);
        test_on_input(1, 7, i);
    }
}
