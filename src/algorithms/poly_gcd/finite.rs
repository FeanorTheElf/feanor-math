use std::cmp::max;
use std::mem::swap;

use tracing::instrument;

use crate::algorithms::int_factor::is_prime_power;
use crate::algorithms::matmul::strassen::{dispatch_strassen_impl, strassen_mem_size};
use crate::matrix::*;
use crate::prelude::*;
use crate::ring_impls::poly::*;

/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_power_decomposition_finite_field<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
    <BaseRingStore<P> as RingStore>::Ring: FiniteRing + Field,
{
    assert!(!poly_ring.is_zero(&poly));
    let squarefree_part = poly_squarefree_part_finite_field(poly_ring, poly);
    if poly_ring.degree(&squarefree_part).unwrap() == poly_ring.degree(&poly).unwrap() {
        return vec![(squarefree_part, 1)];
    } else {
        let square_part = poly_ring.checked_div(&poly, &squarefree_part).unwrap();
        let square_part_decomposition = poly_power_decomposition_finite_field(poly_ring, &square_part);
        let mut result = square_part_decomposition;
        let mut degree = 0;
        for (g, k) in &mut result {
            *k += 1;
            degree += poly_ring.degree(g).unwrap() * *k;
        }
        if degree != poly_ring.degree(&poly).unwrap() {
            let remaining_part = poly_ring
                .checked_div(
                    &poly,
                    &poly_ring.prod(result.iter().map(|(g, e)| poly_ring.pow(g.clone(), *e))),
                )
                .unwrap();
            result.insert(0, (poly_ring.normalize(remaining_part).0, 1));
        }
        return result;
    }
}

/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
///
/// The returned polynomial is always monic, and with this restriction, it
/// is unique.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_squarefree_part_finite_field<P>(poly_ring: P, poly: &El<P>) -> El<P>
where
    P: RingStore,
    P::Ring: PolyRing + PrincipalIdealRing,
    <BaseRingStore<P> as RingStore>::Ring: FiniteRing + Field,
{
    assert!(!poly_ring.is_zero(&poly));
    if poly_ring.degree(poly).unwrap() == 0 {
        return poly_ring.one();
    }
    let derivate = derive_poly(&poly_ring, poly);
    if poly_ring.is_zero(&derivate) {
        let q = poly_ring.base_ring().size(&ZZbig).unwrap();
        let (p, e) = is_prime_power(ZZbig, &q).unwrap();
        let p_usize = int_cast(p.clone(), ZZi64, ZZbig) as usize;
        assert!(p_usize > 0);
        let power = ZZbig.pow(p, e - 1);
        let undo_frobenius = |x: &El<BaseRingStore<P>>| poly_ring.base_ring().pow_gen(x.clone(), &power, ZZbig);
        let base_poly = poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| {
            debug_assert!(i % p_usize == 0);
            (undo_frobenius(c), i / p_usize)
        }));
        return poly_squarefree_part_finite_field(poly_ring, &base_poly);
    } else {
        let square_part = poly_ring.ideal_gen(poly, &derivate);
        let result = poly_ring.checked_div(poly, &square_part).unwrap();
        return poly_ring.normalize(result).0;
    }
}

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
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn partial_eea_poly<P>(ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize) -> ([El<P>; 4], [El<P>; 2])
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
{
    if ring.is_zero(&lhs) || ring.is_zero(&rhs) {
        return ([ring.one(), ring.zero(), ring.zero(), ring.one()], [lhs, rhs]);
    }
    let (mut a, mut b) = (lhs.clone(), rhs.clone());
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

const FAST_POLY_EEA_THRESHOLD: usize = 32;

/// Computes a Bezout identity for polynomials, using a fast divide-and-conquer
/// polynomial gcd algorithm. Unless you are implementing
/// [`crate::ring_properties::pid::PrincipalIdealRing`] for a custom type, you should use
/// [`crate::ring_properties::pid::PrincipalIdealRing::extended_ideal_gen()`] to get a Bezout
/// identity instead.
///
/// A Bezout identity is exactly as specified by
/// [`crate::ring_properties::pid::PrincipalIdealRing::extended_ideal_gen()`], i.e. `s, t, d` such
/// that `d` is the gcd of `lhs` and `rhs`, and `d = lhs * s + rhs * t`. Note that this algorithm
/// does not try to avoid coefficient growth, and thus is only fast over finite fields. Furthermore,
/// it will fall back to a slightly less efficient variant of the standard Euclidean algorithm on
/// small inputs.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn fast_poly_eea<P>(poly_ring: P, lhs: El<P>, rhs: El<P>) -> (El<P>, El<P>, El<P>)
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
    <BaseRingStore<P> as RingStore>::Ring: Field,
{
    fn fast_poly_eea_impl<P>(
        poly_ring: P,
        lhs: El<P>,
        rhs: El<P>,
        target_deg: usize,
        memory: &mut [El<P>],
    ) -> ([El<P>; 4], [El<P>; 2])
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        <BaseRingStore<P> as RingStore>::Ring: Field,
    {
        if poly_ring.is_zero(&lhs) || poly_ring.is_zero(&rhs) {
            return (
                [poly_ring.one(), poly_ring.zero(), poly_ring.zero(), poly_ring.one()],
                [lhs, rhs],
            );
        }
        let ldeg = poly_ring.degree(&lhs).unwrap();
        let rdeg = poly_ring.degree(&rhs).unwrap();
        if ldeg < target_deg + FAST_POLY_EEA_THRESHOLD || rdeg < target_deg + FAST_POLY_EEA_THRESHOLD {
            return partial_eea_poly(poly_ring, lhs, rhs, target_deg);
        } else if 2 * ldeg >= 3 * rdeg {
            let (mut q, r) = poly_ring.euclidean_div_rem(lhs, &rhs);
            poly_ring.negate_inplace(&mut q);
            assert!(poly_ring.degree(&r).unwrap_or(0) <= rdeg);
            let (transform, rest) = fast_poly_eea_impl(poly_ring, r, rhs, target_deg, memory);
            let mut transform: (_, _, _, _) = transform.into();
            transform.1 = poly_ring.fma(&q, &transform.0, transform.1);
            transform.3 = poly_ring.fma(&q, &transform.2, transform.3);
            return (transform.into(), rest);
        } else if 2 * rdeg >= 3 * ldeg {
            let (transform, rest) = fast_poly_eea_impl(poly_ring, rhs, lhs, target_deg, memory);
            let transform: (_, _, _, _) = transform.into();
            return ([transform.1, transform.0, transform.3, transform.2], rest);
        }

        let split_deg = max(ldeg, rdeg) / 3;
        assert!(2 * split_deg + 1 < max(ldeg, rdeg));
        let part_target_deg = max(split_deg, target_deg.saturating_sub(split_deg));

        let lhs_upper = poly_ring.from_terms(
            poly_ring
                .terms(&lhs)
                .filter(|(_, i)| *i >= split_deg)
                .map(|(c, i)| (c.clone(), i - split_deg)),
        );
        let mut lhs_lower = lhs;
        poly_ring.truncate_monomials(&mut lhs_lower, split_deg);
        let rhs_upper = poly_ring.from_terms(
            poly_ring
                .terms(&rhs)
                .filter(|(_, i)| *i >= split_deg)
                .map(|(c, i)| (c.clone(), i - split_deg)),
        );
        let mut rhs_lower = rhs;
        poly_ring.truncate_monomials(&mut rhs_lower, split_deg);

        assert!(
            poly_ring.degree(&lhs_upper).unwrap_or(0) + poly_ring.degree(&rhs_upper).unwrap_or(0)
                <= ldeg + rdeg - split_deg
        );
        let (fst_transform, [mut lhs_rest, mut rhs_rest]) =
            fast_poly_eea_impl(poly_ring, lhs_upper, rhs_upper, part_target_deg, memory);

        poly_ring.mul_assign_monomial(&mut lhs_rest, split_deg);
        poly_ring.mul_assign_monomial(&mut rhs_rest, split_deg);

        lhs_rest = poly_ring.fma(&fst_transform[0], &lhs_lower, lhs_rest);
        lhs_rest = poly_ring.fma(&fst_transform[1], &rhs_lower, lhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[2], &lhs_lower, rhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[3], &rhs_lower, rhs_rest);

        assert!(
            poly_ring.degree(&lhs_rest).unwrap_or(0) + poly_ring.degree(&rhs_rest).unwrap_or(0)
                <= ldeg + rdeg - split_deg
        );
        let (snd_transform, rest) = fast_poly_eea_impl(poly_ring, lhs_rest, rhs_rest, target_deg, memory);

        // multiply snd_transform * fst_transform
        let mut result = [poly_ring.zero(), poly_ring.zero(), poly_ring.zero(), poly_ring.zero()];
        dispatch_strassen_impl::<_, _, _, _, false, _, _, _>(
            1,
            0,
            TransposableSubmatrix::from(Submatrix::from_1d(&snd_transform, 2, 2)),
            TransposableSubmatrix::from(Submatrix::from_1d(&fst_transform, 2, 2)),
            TransposableSubmatrixMut::from(SubmatrixMut::from_1d(&mut result, 2, 2)),
            poly_ring,
            memory,
        );

        return (result, rest);
    }

    if poly_ring.is_zero(&lhs) {
        return (poly_ring.zero(), poly_ring.one(), rhs);
    } else if poly_ring.is_zero(&rhs) {
        return (poly_ring.one(), poly_ring.zero(), lhs);
    }
    let ([s1, t1, s2, t2], [a1, a2]) = fast_poly_eea_impl(
        poly_ring,
        lhs,
        rhs,
        0,
        &mut (0..strassen_mem_size(false, 2, 0))
            .map(|_| poly_ring.zero())
            .collect::<Vec<_>>(),
    );
    if poly_ring.is_zero(&a1) {
        return (s2, t2, a2);
    } else if poly_ring.is_zero(&a2) || poly_ring.degree(&a1).unwrap() == 0 {
        return (s1, t1, a1);
    } else {
        assert!(poly_ring.degree(&a2).unwrap() == 0);
        return (s2, t2, a2);
    }
}

#[cfg(test)]
use crate::ring_impls::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::ring_impls::zn::zn_64b::Zn64B;
#[cfg(test)]
use crate::ring_impls::zn::*;

#[test]
fn test_partial_eea_poly() {
    feanor_tracing::DelayedLogger::init_test();
    let field = Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let test_on_input = |a: &El<DensePolyRing<_>>, b: &El<DensePolyRing<_>>, deg| {
        let ([s, t, s_, t_], [x, x_]) = partial_eea_poly(&poly_ring, a.clone(), b.clone(), deg);
        assert_el_eq!(
            &poly_ring,
            poly_ring.add(poly_ring.mul_ref(&s, a), poly_ring.mul_ref(&t, b)),
            &x
        );
        assert_el_eq!(
            &poly_ring,
            poly_ring.add(poly_ring.mul_ref(&s_, a), poly_ring.mul_ref(&t_, b)),
            &x_
        );
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

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
            X.pow_ref(10) + X.pow_ref(6) + 1,
        ]
    });
    for deg in (0..10).rev() {
        test_on_input(&f, &g, deg);
        test_on_input(&g, &f, deg);
    }

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(5) - 1, X.pow_ref(10) - 1]);
    for deg in (0..5).rev() {
        test_on_input(&f, &g, deg);
        test_on_input(&g, &f, deg);
    }
}

#[test]
fn test_fast_poly_eea() {
    feanor_tracing::DelayedLogger::init_test();

    let field = Zn64B::new(2).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(96) + X.pow_ref(55), X.pow_ref(54)]);
    let (s, t, d) = fast_poly_eea(&poly_ring, f.clone(), g.clone());
    assert_el_eq!(
        &poly_ring,
        &d,
        poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g))
    );
    assert_el_eq!(
        &poly_ring,
        poly_ring.pow(poly_ring.indeterminate(), 54),
        poly_ring.normalize(d).0
    );

    let field = zn_64b::Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X.pow_ref(90) - X.pow_ref(70) + 3 * X.pow_ref(20) - 1,
            X.pow_ref(100) + X.pow_ref(60) + 1,
        ]
    });
    let (s, t, d) = fast_poly_eea(&poly_ring, f.clone(), g.clone());
    assert!(poly_ring.is_unit(&d));
    assert_el_eq!(
        &poly_ring,
        &d,
        poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g))
    );

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
            X.pow_ref(100) + X.pow_ref(60) + 1,
        ]
    });
    let (s, t, d) = fast_poly_eea(&poly_ring, f.clone(), g.clone());
    assert!(poly_ring.is_unit(&d));
    assert_el_eq!(
        &poly_ring,
        &d,
        poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g))
    );
}
