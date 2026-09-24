use std::mem::swap;

use tracing::instrument;

use crate::algorithms::euclid::general_extended_euclid;
use crate::algorithms::int_factor::is_prime_power;
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
/// # Invariants
///
/// Setting up euclid's algorithm as `s0 = t1 = 1`, `s1 = t0 = 0`, `a0 = a` and `a1 = b`,
/// and then using the recursion `a(i - 2) = qi * a(i - 1) + ri`, `ai = ri`, `si = s(i - 2) - qi s(i
/// - 1)`, `ti = t(i - 2) - qi t(i - 1)`, we find that (assuming at least a single remainder step
/// was done)
/// ```text
///   deg(s), deg(s') <= deg(b) - deg(x) <= deg(b) - target_deg
///   deg(t), deg(t') <= deg(a) - deg(x) <= deg(a) - target_deg
/// ```
/// If one of `a, b` is zero, the invariants don't apply; in this case, the zero value is returned
/// as `x'` and the other one as `x`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn partial_eea_poly<P>(ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize) -> ([El<P>; 4], [El<P>; 2])
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
{
    if ring.is_zero(&lhs) {
        return ([ring.zero(), ring.one(), ring.one(), ring.zero()], [rhs, lhs]);
    } else if ring.is_zero(&rhs) {
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

        debug_assert!(ring.degree(&sb).unwrap() <= ring.degree(&rhs).unwrap() - ring.degree(&a).unwrap_or(0));
        debug_assert!(ring.degree(&tb).unwrap() <= ring.degree(&lhs).unwrap() - ring.degree(&a).unwrap_or(0));
        debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &lhs), ring.mul_ref(&tb, &rhs))));
    }
    if usize::min(ring.degree(&lhs).unwrap(), ring.degree(&rhs).unwrap()) > target_deg {
        let deg_x = ring.degree(&a).unwrap();
        debug_assert!(deg_x > target_deg);
        debug_assert!(ring.degree(&sa).unwrap_or(0) <= ring.degree(&rhs).unwrap() - deg_x);
        debug_assert!(ring.degree(&ta).unwrap_or(0) <= ring.degree(&lhs).unwrap() - deg_x);
        debug_assert!(ring.degree(&sb).unwrap_or(0) <= ring.degree(&rhs).unwrap() - deg_x);
        debug_assert!(ring.degree(&tb).unwrap_or(0) <= ring.degree(&lhs).unwrap() - deg_x);
    }
    return ([sa, ta, sb, tb], [a, b]);
}

const HALF_GCD_THRESHOLD: usize = 32;

/// The half-gcd algorithm. Computes the same output as [`partial_eea_poly()`] but with
/// sub-quadratic complexity.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn half_gcd<P>(poly_ring: P, lhs: El<P>, rhs: El<P>, n_half: usize) -> ([El<P>; 4], [El<P>; 2])
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
    <BaseRingStore<P> as RingStore>::Ring: Field,
{
    if poly_ring.is_zero(&lhs) {
        return (
            [poly_ring.zero(), poly_ring.one(), poly_ring.one(), poly_ring.zero()],
            [rhs, lhs],
        );
    } else if poly_ring.is_zero(&rhs) {
        return (
            [poly_ring.one(), poly_ring.zero(), poly_ring.zero(), poly_ring.one()],
            [lhs, rhs],
        );
    }
    let ldeg = poly_ring.degree(&lhs).unwrap();
    let rdeg = poly_ring.degree(&rhs).unwrap();
    assert!(ldeg <= 2 * n_half);
    assert!(rdeg <= 2 * n_half);
    let split = |poly, k| {
        let upper = poly_ring.from_terms(
            poly_ring
                .terms(&poly)
                .filter_map(|(c, i)| i.checked_sub(k).map(|i| (c.clone(), i))),
        );
        let mut lower = poly;
        poly_ring.truncate_monomials(&mut lower, k);
        return (upper, lower);
    };

    let (result, [f, g]) = if ldeg < n_half + HALF_GCD_THRESHOLD || rdeg < n_half + HALF_GCD_THRESHOLD {
        partial_eea_poly(poly_ring, lhs, rhs, n_half)
    } else {
        let n_quarter = n_half.div_ceil(2);
        debug_assert!(n_quarter > 0 && n_quarter < n_half);

        let (f_upper, f_lower) = split(lhs, n_half);
        let (g_upper, g_lower) = split(rhs, n_half);
        let ([s1, t1, s2, t2], [mut f, mut g]) = half_gcd(poly_ring, f_upper, g_upper, n_quarter);

        poly_ring.mul_assign_monomial(&mut f, n_half);
        f = poly_ring.fma(&s1, &f_lower, f);
        f = poly_ring.fma(&t1, &g_lower, f);
        poly_ring.mul_assign_monomial(&mut g, n_half);
        g = poly_ring.fma(&s2, &f_lower, g);
        g = poly_ring.fma(&t2, &g_lower, g);
        debug_assert!(poly_ring.degree(&f).unwrap_or(0) > n_half);
        debug_assert!(poly_ring.degree(&g).unwrap_or(0) <= n_half + n_quarter);

        if poly_ring.degree(&g).unwrap_or(0) <= n_half {
            ([s1, t1, s2, t2], [f, g])
        } else {
            let (mut q, r) = poly_ring.euclidean_div_rem(f, &g);
            poly_ring.negate_inplace(&mut q);
            let new_s2 = poly_ring.fma(&q, &s2, s1);
            let new_t2 = poly_ring.fma(&q, &t2, t1);
            let [s1, t1, s2, t2] = [s2, t2, new_s2, new_t2];
            let (f, g) = (g, r);

            if poly_ring.degree(&g).unwrap_or(0) <= n_half {
                ([s1, t1, s2, t2], [f, g])
            } else {
                let k = n_half - n_quarter;
                let (f_upper, f_lower) = split(f, k);
                let (g_upper, g_lower) = split(g, k);

                let ([s1_, t1_, s2_, t2_], [mut f, mut g]) = half_gcd(poly_ring, f_upper, g_upper, n_quarter);
                poly_ring.mul_assign_monomial(&mut f, k);
                f = poly_ring.fma(&s1_, &f_lower, f);
                f = poly_ring.fma(&t1_, &g_lower, f);
                poly_ring.mul_assign_monomial(&mut g, k);
                g = poly_ring.fma(&s2_, &f_lower, g);
                g = poly_ring.fma(&t2_, &g_lower, g);
                let matrix = [
                    poly_ring.fma(&s1_, &s1, poly_ring.mul_ref(&t1_, &s2)),
                    poly_ring.fma(&s1_, &t1, poly_ring.mul_ref_snd(t1_, &t2)),
                    poly_ring.fma(&s2_, &s1, poly_ring.mul_ref_fst(&t2_, s2)),
                    poly_ring.fma(&s2_, &t1, poly_ring.mul(t2_, t2)),
                ];
                (matrix, [f, g])
            }
        }
    };
    debug_assert!(poly_ring.degree(&g).unwrap_or(0) <= n_half);
    if usize::min(ldeg, rdeg) > n_half {
        let deg_f = poly_ring.degree(&f).unwrap();
        debug_assert!(deg_f > n_half);
        debug_assert!(poly_ring.degree(&result[0]).unwrap_or(0) <= rdeg - deg_f);
        debug_assert!(poly_ring.degree(&result[1]).unwrap_or(0) <= ldeg - deg_f);
        debug_assert!(poly_ring.degree(&result[2]).unwrap_or(0) <= rdeg - deg_f);
        debug_assert!(poly_ring.degree(&result[3]).unwrap_or(0) <= ldeg - deg_f);
    }
    return (result, [f, g]);
}

/// Computes a Bezout identity for polynomials, using a fast divide-and-conquer
/// polynomial gcd algorithm. Unless you are implementing [`PrincipalIdealRing`] for a custom type,
/// you should use [`PrincipalIdealRing::extended_ideal_gen()`] to get a Bezout
/// identity instead.
///
/// A Bezout identity is exactly as specified by [`PrincipalIdealRing::extended_ideal_gen()`], i.e.
/// `s, t, d` such that `d` is the gcd of `lhs` and `rhs`, and `d = lhs * s + rhs * t`. Note that
/// this algorithm does not try to avoid coefficient growth, and thus is only fast over finite
/// fields. Furthermore, it will fall back to a slightly less efficient variant of the standard
/// Euclidean algorithm on small inputs.
///
/// [`PrincipalIdealRing`]: crate::ring_properties::pid::PrincipalIdealRing
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn fast_poly_eea<P>(poly_ring: P, lhs: El<P>, rhs: El<P>) -> (El<P>, El<P>, El<P>)
where
    P: RingStore + Copy,
    P::Ring: PolyRing + EuclideanRing,
    <BaseRingStore<P> as RingStore>::Ring: Field,
{
    if poly_ring.is_zero(&lhs) {
        return (poly_ring.zero(), poly_ring.one(), rhs);
    } else if poly_ring.is_zero(&rhs) {
        return (poly_ring.one(), poly_ring.zero(), lhs);
    }
    let (mut f, mut g) = (lhs, rhs);
    let mut matrix = [poly_ring.one(), poly_ring.zero(), poly_ring.zero(), poly_ring.one()];
    while !poly_ring.is_zero(&g)
        && (poly_ring.degree(&f).unwrap_or(0) > HALF_GCD_THRESHOLD
            || poly_ring.degree(&g).unwrap_or(0) > HALF_GCD_THRESHOLD)
    {
        let n_half = (usize::max(poly_ring.degree(&f).unwrap_or(0), poly_ring.degree(&g).unwrap_or(0)) + 1) / 2;
        let ([s1_, t1_, s2_, t2_], [f_, g_]) = half_gcd(poly_ring, f, g, n_half);
        let [s1, t1, s2, t2] = matrix;
        matrix = [
            poly_ring.fma(&s1_, &s1, poly_ring.mul_ref(&t1_, &s2)),
            poly_ring.fma(&s1_, &t1, poly_ring.mul_ref_snd(t1_, &t2)),
            poly_ring.fma(&s2_, &s1, poly_ring.mul_ref_fst(&t2_, s2)),
            poly_ring.fma(&s2_, &t1, poly_ring.mul(t2_, t2)),
        ];
        f = f_;
        g = g_;

        // `half_gcd()` does nothing at all if `deg(g) <= n_half` already holds, which can happen
        // whenever `deg(g) <= deg(f)/2`; hence perform one step of the Euclidean algorithm to
        // ensure that the degrees decrease during every iteration
        if !poly_ring.is_zero(&g) {
            debug_assert!(poly_ring.degree(&f).unwrap() >= poly_ring.degree(&g).unwrap());
            let (mut quo, rem) = poly_ring.euclidean_div_rem(f, &g);
            poly_ring.negate_inplace(&mut quo);
            let [s1, t1, s2, t2] = matrix;
            let new_s2 = poly_ring.fma(&quo, &s2, s1);
            let new_t2 = poly_ring.fma(&quo, &t2, t1);
            matrix = [s2, t2, new_s2, new_t2];
            f = g;
            g = rem;
        }
    }
    let (s_, t_, d) = general_extended_euclid(f, g, poly_ring);
    let [s1, t1, s2, t2] = matrix;
    return (
        poly_ring.fma(&s_, &s1, poly_ring.mul_ref_fst(&t_, s2)),
        poly_ring.fma(&s_, &t1, poly_ring.mul(t_, t2)),
        d,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_impls::poly::dense_poly::DensePolyRing;
    use crate::ring_impls::zn::zn_64b::Zn64B;
    use crate::ring_impls::zn::*;
    use crate::ring_properties::finite::FiniteRingStore;

    #[test]
    fn test_poly_squarefree_part() {
        let F2 = Zn64B::new(2).as_field().unwrap();
        let F2X = DensePolyRing::new(F2, "X");
        let [f] = F2X.with_wrapped_indeterminate(|X| [X.pow_ref(3) + X.pow_ref(2) + 1]);
        assert_el_eq!(&F2X, &f, poly_squarefree_part_finite_field(&F2X, &f));
    }

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

        let [f, g] = poly_ring.with_wrapped_indeterminate(|X| {
            [
                X.pow_ref(976) + X.pow_ref(943) + X.pow_ref(765) + X.pow_ref(602) + 1,
                X.pow_ref(942) + X.pow_ref(764),
            ]
        });
        let (s, t, d) = fast_poly_eea(&poly_ring, f.clone(), g.clone());
        assert_el_eq!(
            &poly_ring,
            &d,
            poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g))
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

    #[test]
    fn test_half_gcd_random() {
        let field = Zn64B::new(65537).as_field().ok().unwrap();
        let poly_ring = DensePolyRing::new(field, "X");
        let mut rng = oorandom::Rand64::new(1);
        let mut random_poly =
            |deg: usize| poly_ring.from_terms((0..=deg).map(|i| (field.random_element(|| rng.rand_u64()), i)));
        for ldeg in [40, 41, 100, 101, 128, 129, 200, 201] {
            for rdeg_ in [3, 39, 40, 41, 70, 99, 100, 101] {
                let rdeg = usize::min(rdeg_, ldeg);
                let f = random_poly(ldeg);
                let g = random_poly(rdeg);
                for n_half in [
                    (usize::max(ldeg, rdeg) + 1) / 2,
                    usize::max(ldeg, rdeg),
                    usize::max(ldeg, rdeg) * 2 / 3,
                ] {
                    // also test with swapped inputs, i.e. `deg(lhs) < deg(rhs)`
                    for (lhs, rhs) in [(&f, &g), (&g, &f)] {
                        let (expected_m, [expected_f, expected_g]) =
                            partial_eea_poly(&poly_ring, lhs.clone(), rhs.clone(), n_half);
                        let (actual_m, [actual_f, actual_g]) = half_gcd(&poly_ring, lhs.clone(), rhs.clone(), n_half);
                        assert_el_eq!(&poly_ring, &expected_f, &actual_f);
                        assert_el_eq!(&poly_ring, &expected_g, &actual_g);
                        for i in 0..4 {
                            assert_el_eq!(&poly_ring, &expected_m[i], &actual_m[i]);
                        }
                        assert_el_eq!(
                            &poly_ring,
                            &actual_f,
                            poly_ring.add(
                                poly_ring.mul_ref(&actual_m[0], lhs),
                                poly_ring.mul_ref(&actual_m[1], rhs)
                            )
                        );
                        assert_el_eq!(
                            &poly_ring,
                            &actual_g,
                            poly_ring.add(
                                poly_ring.mul_ref(&actual_m[2], lhs),
                                poly_ring.mul_ref(&actual_m[3], rhs)
                            )
                        );
                    }
                }
                // and the full gcd, again in both orders
                for (lhs, rhs) in [(&f, &g), (&g, &f)] {
                    let (s, t, d) = fast_poly_eea(&poly_ring, lhs.clone(), rhs.clone());
                    assert_el_eq!(
                        &poly_ring,
                        &d,
                        poly_ring.add(poly_ring.mul_ref(&s, lhs), poly_ring.mul_ref(&t, rhs))
                    );
                    let expected_d = poly_ring.ideal_gen(lhs, rhs);
                    assert_el_eq!(&poly_ring, poly_ring.normalize(expected_d).0, poly_ring.normalize(d).0);
                }
            }
        }
    }
}
