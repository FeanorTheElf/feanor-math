use std::mem::swap;

use crate::ring::*;
use crate::rings::zn::*;
use crate::pid::*;
use crate::ordered::OrderedRingStore;

///
/// Returns two integers `(a, b)` with `b > 0` such that `a/b = x mod n`.
/// 
/// If there exist |a| b < sqrt(n)/3` with `a/b = x mod n`, these will be
/// returned. Otherwise (i.e. if every `a, b` with `a/b = x mod n` satisfies
/// `|a| b > sqrt(3)/3`), no further guarantees on the result are given, in
/// particular, `(a, b) = (x, 1)` would be a valid result.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::algorithms::rational_reconstruction::*;
/// let ring = Zn::new(100000000);
/// assert_eq!((3, 7), rational_reconstruction(&ring, ring.checked_div(&ring.int_hom().map(3), &ring.int_hom().map(7)).unwrap()));
///  ```
/// 
#[stability::unstable(feature = "enable")]
pub fn rational_reconstruction<R>(Zn: R, x: El<R>) -> (El<<R::Type as ZnRing>::IntegerRing>, El<<R::Type as ZnRing>::IntegerRing>)
    where R: RingStore,
        R::Type: ZnRing
{
    if Zn.is_zero(&x) {
        return (Zn.integer_ring().zero(), Zn.integer_ring().one());
    }

    let ZZ = Zn.integer_ring();
    let mut a = ZZ.clone_el(Zn.modulus());
    let mut s = ZZ.zero();

    let mut b = Zn.smallest_positive_lift(x);
    let mut t = ZZ.one();

    let mut current_max = (ZZ.zero(), ZZ.zero(), ZZ.zero());

    // invariant: `s * x == a mod n` and `t * x == b mod n`
    while !ZZ.is_zero(&b) {
        let (q, r) = ZZ.euclidean_div_rem(ZZ.clone_el(&a), &b);
        if ZZ.is_gt(&q, &current_max.0) {
            debug_assert!(!ZZ.is_zero(&t));
            current_max = (ZZ.clone_el(&q), ZZ.clone_el(&b), ZZ.clone_el(&t));
        }

        a = r;
        ZZ.sub_assign(&mut s, ZZ.mul_ref_snd(q, &t));
        swap(&mut a, &mut b);
        swap(&mut s, &mut t);
    }

    return (current_max.1, current_max.2);
}

#[cfg(test)]
use crate::divisibility::DivisibilityRingStore;
#[cfg(test)]
use crate::homomorphism::Homomorphism;

#[test]
fn test_rational_reconstruction() {
    let n = 2021027;
    let Zn = zn_64::Zn::new(n);
    let ab_bound = 472;
    for a in -ab_bound..=ab_bound {
        for b in 1..-ab_bound {
            if a * b <= ab_bound {
                let x = Zn.checked_div(&Zn.int_hom().map(a), &Zn.int_hom().map(b)).unwrap();
                assert_eq!((a as i64, b as i64), rational_reconstruction(Zn, x));
            }
        }
    }
}