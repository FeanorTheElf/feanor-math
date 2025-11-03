use tracing::instrument;

use crate::integer::IntegerRing;
use crate::pid::*;
use crate::ring::*;
use crate::ordered::OrderedRingStore;
use crate::rings::poly::{PolyRing, PolyRingStore};

use std::cmp::Ordering;
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
#[instrument(skip_all, level = "trace")]
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
#[instrument(skip_all, level = "trace")]
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
#[instrument(skip_all, level = "trace")]
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
/// Computes `[s, t, s', t']` such that `x := s * a + t * b` and `x' := s' * a + t' * b`
/// are the current pair of values during the Euclidean Algorithm when, for the first
/// time, we find `deg(x') <= target_deg`.
/// 
/// In particular, we have `deg(x) > target_deg` and `deg(x') <= target_deg`, except if `a, b`
/// already both have degree `<= target_deg`.
/// 
/// The degrees of `s, t, s', t'` are bounded as
/// ```text
///   deg(s) <= deg(b) - deg(x)
///   deg(t) <= deg(a) - deg(x)
///   deg(s') <= deg(b) - deg(x')
///   deg(t') <= deg(a) - deg(x')
/// ```
/// except if either `a` or `b` already have degree `<= target_deg` (i.e. no steps in
/// the Euclidean algorithm are performed), in which case only the two bounds involving
/// the larger one of `deg(a)` resp. `deg(b)` hold.
/// 
/// The proof of this is similar to the one outlined in [`partial_eea_int()`], but simpler, because
/// the degree-valuation is non-Archimedean.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn partial_eea_poly<P>(ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize) -> ([El<P>; 4], [El<P>; 2])
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing
{
    if ring.is_zero(&lhs) || ring.is_zero(&rhs) {
        return ([ring.one(), ring.zero(), ring.zero(), ring.one()], [lhs, rhs]);
    }
    let (mut a, mut b) = (ring.clone_el(&lhs), ring.clone_el(&rhs));
    let (mut sa, mut ta) = (ring.one(), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.one());
    
    if ring.degree(&a).unwrap() < ring.degree(&b).unwrap() {
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }

    while ring.degree(&b).unwrap_or(0) > target_deg {
        debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &lhs), ring.mul_ref(&ta, &rhs))));
        debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &lhs), ring.mul_ref(&tb, &rhs))));

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
        
        debug_assert!(ring.degree(&sb).unwrap() <= ring.degree(&rhs).unwrap() - ring.degree(&b).unwrap_or(0));
        debug_assert!(ring.degree(&tb).unwrap() <= ring.degree(&lhs).unwrap() - ring.degree(&b).unwrap_or(0));
    }
    return ([sa, ta, sb, tb], [a, b]);
}


///
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
///  - we have `a_i |s_i| <= |s_2| a_2 q_2 / q_i` and `a_i |t_i| <= |t_2| a_2 q_2 / q_i` for `i >= 2`;
///    this can be shown by induction, using the first two points
///  - finally, this implies for `i >= 1` that `a_i |s_i| <= a_1 / q_i` and `a_i |t_i| <= a_0 / q_i`,
///    using that `q_2 a_2 <= a_1` and `q_1 a_1 <= a_0`
///  - unfortunately, it does not apply to `i = 0`
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn partial_eea_int<R>(ring: R, lhs: El<R>, rhs: El<R>, target_size: &El<R>) -> ([El<R>; 4], [El<R>; 2])
    where R: RingStore + Copy,
        R::Type: IntegerRing
{
    assert!(!ring.is_neg(target_size));
    let (mut a, mut b) = (ring.clone_el(&lhs), ring.clone_el(&rhs));
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

#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_64::*;
#[cfg(test)]
use crate::rings::zn::*;

#[test]
fn test_gcd() {
    LogAlgorithmSubscriber::init_test();
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
fn test_partial_eea_poly() {
    LogAlgorithmSubscriber::init_test();
    let field = Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let test_on_input = |a, b, deg| {
        let ([s, t, s_, t_], [x, x_]) = partial_eea_poly(&poly_ring, poly_ring.clone_el(a), poly_ring.clone_el(b), deg);
        assert_el_eq!(&poly_ring, poly_ring.add(poly_ring.mul_ref(&s, a), poly_ring.mul_ref(&t, b)), &x);
        assert_el_eq!(&poly_ring, poly_ring.add(poly_ring.mul_ref(&s_, a), poly_ring.mul_ref(&t_, b)), &x_);
        let a_deg = poly_ring.degree(a).unwrap_or(0);
        let b_deg = poly_ring.degree(b).unwrap_or(0);
        if a_deg <= deg && b_deg <= deg {
            assert!(poly_ring.degree(&x).unwrap_or(0) <= deg);
            assert!(poly_ring.degree(&x_).unwrap_or(0) <= deg);
        } else if a_deg <= deg || b_deg <= deg {
            assert!(poly_ring.degree(&x).unwrap_or(0) > deg);
            assert!(poly_ring.degree(&x_).unwrap_or(0) <= deg);
            if a_deg >= b_deg {
                assert!(poly_ring.degree(&t).unwrap_or(0) <= a_deg - poly_ring.degree(&x).unwrap_or(0));
                assert!(poly_ring.degree(&t_).unwrap_or(0) <= a_deg - poly_ring.degree(&x_).unwrap_or(0));
            } else {
                assert!(poly_ring.degree(&s).unwrap_or(0) <= b_deg - poly_ring.degree(&x).unwrap_or(0));
                assert!(poly_ring.degree(&s_).unwrap_or(0) <= b_deg - poly_ring.degree(&x_).unwrap_or(0));
            }
        } else {
            assert!(poly_ring.degree(&x).unwrap_or(0) > deg);
            assert!(poly_ring.degree(&x_).unwrap_or(0) <= deg);
            assert!(poly_ring.degree(&t).unwrap_or(0) <= a_deg - poly_ring.degree(&x).unwrap_or(0));
            assert!(poly_ring.degree(&t_).unwrap_or(0) <= a_deg - poly_ring.degree(&x_).unwrap_or(0));
            assert!(poly_ring.degree(&s).unwrap_or(0) <= b_deg - poly_ring.degree(&x).unwrap_or(0));
            assert!(poly_ring.degree(&s_).unwrap_or(0) <= b_deg - poly_ring.degree(&x_).unwrap_or(0));
        }
    };

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
        X.pow_ref(10) + X.pow_ref(6) + 1,
    ]);
    for deg in (0..10).rev() {
        test_on_input(&f, &g, deg);
        test_on_input(&g, &f, deg);
    }

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(5) - 1,
        X.pow_ref(10) - 1,
    ]);
    for deg in (0..5).rev() {
        test_on_input(&f, &g, deg);
        test_on_input(&g, &f, deg);
    }
}

#[test]
fn test_partial_int() {
    LogAlgorithmSubscriber::init_test();
    let test_on_input = |a: i64, b: i64, size: i64| {
        assert!(a != 0);
        assert!(b != 0);
        let ([s, t, s_, t_], [x, x_]) = partial_eea_int(StaticRing::<i64>::RING, a, b, &size);
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