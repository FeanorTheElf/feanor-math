use crate::ring::*;
use crate::integer::*;
use crate::ordered::*;

///
/// Finds some integer `left <= n < right` such that `f(n) <= 0` and `f(n + 1) > 0`, given
/// that `f(left) <= 0` and `f(right) > 0`.
/// 
/// If we consider a continuous extension of `f` to the real numbers, this means that
/// the function finds `floor(x)` for some root `x` of `f` between `left` and `right`.
/// 
pub fn bisect_floor<R, F>(ZZ: R, left: El<R>, right: El<R>, mut func: F) -> El<R>
    where R: RingStore, R::Type: IntegerRing, F: FnMut(&El<R>) -> El<R>
{
    assert!(ZZ.is_lt(&left, &right));
    let mut l = left;
    let mut r = right;
    assert!(!ZZ.is_pos(&func(&l)));
    assert!(ZZ.is_pos(&func(&r)));
    loop {
        let mut mid = ZZ.add_ref(&l, &r);
        ZZ.euclidean_div_pow_2(&mut mid, 1);

        if ZZ.eq_el(&mid, &l) || ZZ.eq_el(&mid, &r) {
            return l;
        } else if ZZ.is_pos(&func(&mid)) {
            r = mid;
        } else {
            l = mid;
        }
    }
}

///
/// Given a function `f` with `lim_{x -> -inf} f(x) = -inf` and `lim_{x -> inf} f(x) = inf`,
/// finds some integer `n` such that `f(n) <= 0` and `f(n + 1) > 0`.
/// 
/// Good performance is only guaranteed for increasing functions `f`. In this case, the
/// solution is unique, and the function terminates `O(log|approx - solution|)` steps.
/// 
/// If we consider a continuous extension of `f` to the real numbers, this means that
/// the function finds `floor(x)` for some root `x` of `f`.
/// 
pub fn find_root_floor<R, F>(ZZ: R, approx: El<R>, mut func: F) -> El<R>
    where R: RingStore, R::Type: IntegerRing, F: FnMut(&El<R>) -> El<R>
{
    let mut left = ZZ.clone_el(&approx);
    let mut step = ZZ.one();
    while ZZ.is_pos(&func(&left)) {
        ZZ.sub_assign_ref(&mut left, &step);
        ZZ.mul_pow_2(&mut step, 1);
    }
    step = ZZ.one();
    let mut right = approx;
    while !ZZ.is_pos(&func(&right)) {
        ZZ.add_assign_ref(&mut right, &step);
        ZZ.mul_pow_2(&mut step, 1);
    }
    return bisect_floor(ZZ, left, right, func);
}

///
/// Computes the largest integer `x` such that `x^root <= n`, where `n` is a positive integer.
/// 
/// # Integer overflow
/// 
/// This function is designed to avoid integer overflow if possible without a huge performance
/// benefit. In particular, in some cases the naive way would require evaluating `(x + 1)^root`
/// to ensure this is really larger than `n` - however, this might lead to integer overflow as in
/// ```should_panic
/// // compute the 62nd root of 2^62
/// let result: i64 = 2;
/// assert!(result.pow(62) <= (1 << 62) && (result + 1).pow(62) > (1 << 62));
/// ```
/// These cases can somewhat be avoided by first doing a size estimate.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::int_bisect::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i64>::RING;
/// assert_eq!(2, root_floor(&ZZ, 1 << 62, 62));
/// ```
/// I some edge cases this is not enough, and so integer overflow can occur.
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::int_bisect::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i64>::RING;
/// assert_eq!(26, root_floor(&ZZ, ZZ.pow(5, 26), 26));
/// ```
/// 
pub fn root_floor<R>(ZZ: R, n: El<R>, root: usize) -> El<R>
    where R: RingStore, R::Type: IntegerRing
{
    assert!(root > 0);
    if root == 1 {
        return n;
    }
    if ZZ.is_zero(&n) {
        return ZZ.zero();
    }
    assert!(!ZZ.is_neg(&n));
    let log2_ceil_n = ZZ.abs_log2_ceil(&n).unwrap();

    return find_root_floor(
        &ZZ, 
        ZZ.from_float_approx(ZZ.to_float_approx(&n).powf(1. / root as f64)).unwrap_or(ZZ.zero()), 
        |x| {
            let x_pow_root_half = ZZ.pow(ZZ.clone_el(x), root / 2);
            // we first make a size estimate, mainly to avoid situations (high `root`) where we get avoidable integer overflow (also might improve performance)
            if ZZ.abs_log2_ceil(x).unwrap_or(0) >= 2 && (ZZ.abs_log2_ceil(&x_pow_root_half).unwrap() - 1) * 2 >= log2_ceil_n {
                if ZZ.is_neg(x) {
                    return ZZ.negate(ZZ.clone_el(&n));
                } else {
                    return ZZ.clone_el(&n);
                }
            } else {
                let x_pow_root = if root % 2 == 0 {
                    ZZ.pow(x_pow_root_half, 2)
                } else {
                    ZZ.mul_ref_snd(ZZ.pow(x_pow_root_half, 2), x)
                };
                return ZZ.sub_ref_snd(x_pow_root, &n);
            }
        }
    );
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_bisect_floor() {
    assert_eq!(0, bisect_floor(&StaticRing::<i64>::RING, 0, 10, |x| if *x == 0 { 0 } else { 1 }));
    assert_eq!(9, bisect_floor(&StaticRing::<i64>::RING, 0, 10, |x| if *x == 10 { 1 } else { 0 }));
    assert_eq!(-15, bisect_floor(&StaticRing::<i64>::RING, -20, -10, |x| *x + 15));
}

#[test]
fn test_root_floor() {
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, 16, 2));
    assert_eq!(3, root_floor(&StaticRing::<i64>::RING, 27, 3));
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, 17, 2));
    assert_eq!(3, root_floor(&StaticRing::<i64>::RING, 28, 3));
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, 24, 2));
    assert_eq!(3, root_floor(&StaticRing::<i64>::RING, 63, 3));
    assert_eq!(5, root_floor(&StaticRing::<i64>::RING, StaticRing::<i64>::RING.pow(5, 25), 25));
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, StaticRing::<i64>::RING.pow(5, 25), 26));
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, StaticRing::<i64>::RING.pow(5, 25), 27));
    assert_eq!(4, root_floor(&StaticRing::<i64>::RING, StaticRing::<i64>::RING.pow(5, 25), 28));
}