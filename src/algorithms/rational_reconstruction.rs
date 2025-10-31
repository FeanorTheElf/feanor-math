use std::cmp::Ordering;
use std::mem::swap;

use tracing::instrument;

use crate::ring::*;
use crate::rings::zn::*;
use crate::integer::*;
use crate::pid::*;
use crate::ordered::OrderedRingStore;

///
/// Uses an optimized version of the LLL algorithm to compute a reduced
/// basis of the lattice
/// ```text
///   L = { u | u[0] x - u[1] = 0 mod n }
/// ```
/// i.e. `(u, v)` such that `|u|` is minimal among all nonzero vectors of
/// `L`, and `|v|` is minimal among all vectors of `L` that are not linearly
/// dependent with `u`.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn reduce_2d_modular_relation_basis<R>(Zn: R, x: El<R>) -> (
    [El<<R::Type as ZnRing>::IntegerRing>; 2], 
    [El<<R::Type as ZnRing>::IntegerRing>; 2]
)
    where R: RingStore,
        R::Type: ZnRing
{
    let ZZ = Zn.integer_ring();
    if Zn.is_zero(&x) {
        return ([ZZ.one(), ZZ.zero()], [ZZ.zero(), ZZ.clone_el(Zn.modulus())]);
    }

    let mut u = [ZZ.zero(), ZZ.clone_el(Zn.modulus())];
    let mut v = [ZZ.one(), Zn.smallest_positive_lift(x)];

    // at the beginning, `|v[1]| >> |v[0]|` (and similar for `u`), thus we estimate
    // the size reduction coefficient `round(<u, v>/<u, u>)` by `round(v[1]/u[1])`,
    // until `|v[0]| ~ |v[1]|`; note also that always `|v[1]| < |u[1]|`
    while ZZ.abs_cmp(&v[1], &v[0]) == Ordering::Greater {
        let (q, r) = ZZ.euclidean_div_rem(ZZ.clone_el(&v[1]), &u[1]);
        v[1] = r;
        ZZ.sub_assign(&mut v[0], ZZ.mul_ref_fst(&u[0], q));
        swap(&mut u, &mut v);
    }

    // now use real LLL, it is likely that this does not run for many rounds anymore
    loop {
        let norm_u_sqr = ZZ.add(ZZ.pow(ZZ.clone_el(&u[0]), 2), ZZ.pow(ZZ.clone_el(&u[1]), 2));
        let q = ZZ.rounded_div(
            ZZ.add(ZZ.mul_ref(&u[0], &v[0]), ZZ.mul_ref(&u[1], &v[1])),
            &norm_u_sqr
        );
        ZZ.sub_assign(&mut v[0], ZZ.mul_ref(&u[0], &q));
        ZZ.sub_assign(&mut v[1], ZZ.mul_ref(&u[1], &q));
        let norm_v_sqr = ZZ.add(ZZ.pow(ZZ.clone_el(&v[0]), 2), ZZ.pow(ZZ.clone_el(&v[1]), 2));
        if ZZ.is_geq(&norm_v_sqr, &norm_u_sqr) {
            return (u, v);
        } else {
            swap(&mut u, &mut v);
        }
    }
}

///
/// Returns two integers `(a, b)` with `b > 0` such that `a = bx mod n`.
/// 
/// More concretely, this returns the `a, b` with smallest l2-norm `a^2 + b^2`
/// such that `a = bx mod n`. In particular, if there are `a, b` with
/// `|a| <= sqrt(n / 2)` and `b <= sqrt(n / 2)`, then these are the unique
/// solution (up to scalar multiples).
/// 
/// # Example
/// 
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::algorithms::rational_reconstruction::*;
/// let ring = Zn64B::new(100000000);
/// assert_eq!((3, 7), balanced_rational_reconstruction(&ring, ring.checked_div(&ring.int_hom().map(3), &ring.int_hom().map(7)).unwrap()));
/// ```
/// 
/// # Proof
/// 
/// We show that if there are `a, b` with `|a|, |b| <= sqrt(n / 2)`, then these
/// form (up to scalar multiples) the unique shortest vector. If this were not the case,
/// we would have a basis of `L = { u | u[0] x - u[1] = 0 mod n }` of two vectors both
/// of length `< sqrt(n)`. wlog we assume it is size-reduced, however this means that
/// the determinant of the basis is `< sqrt(n) * sqrt(n) = n = det(L)`, a contradiction.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn balanced_rational_reconstruction<R>(Zn: R, x: El<R>) -> (El<<R::Type as ZnRing>::IntegerRing>, El<<R::Type as ZnRing>::IntegerRing>)
    where R: RingStore,
        R::Type: ZnRing
{
    let [b, a] = reduce_2d_modular_relation_basis(&Zn, x).0;
    if Zn.integer_ring().is_neg(&b) {
        return (Zn.integer_ring().negate(a), Zn.integer_ring().negate(b));
    } else {
        return (a, b);
    }
}

#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::divisibility::DivisibilityRingStore;
#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_rational_reconstruction() {
    LogAlgorithmSubscriber::init_test();
    let n = 2021027;
    let Zn = zn_64::Zn64B::new(n as u64);
    let ab_bound = (n as f64 / 2.).sqrt().floor() as i32;
    for a in -ab_bound..ab_bound {
        for b in 1..ab_bound {
            if a * b <= n / 2 && StaticRing::<i32>::RING.ideal_gen(&b, &a) == 1 {
                let x = Zn.checked_div(&Zn.int_hom().map(a), &Zn.int_hom().map(b)).unwrap();
                assert_eq!((a as i64, b as i64), balanced_rational_reconstruction(Zn, x));
            }
        }
    }

    // this example leads to `a, b` that are divisible by `17`.
    let ring = zn_64::Zn64B::new(17 * 17 * 17);
    let hom = ring.can_hom(ring.integer_ring()).unwrap();
    let (a, b) = balanced_rational_reconstruction(&ring, hom.map(4048));
    assert_el_eq!(&ring, &hom.map(a), ring.mul(hom.map(b), hom.map(4048)));
}
