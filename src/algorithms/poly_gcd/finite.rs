use std::mem::swap;
use std::cmp::max;

use tracing::instrument;

use crate::algorithms::matmul::strassen::{dispatch_strassen_impl, strassen_mem_size};
use crate::algorithms::int_factor::is_prime_power;
use crate::matrix::*;
use crate::primitive_int::*;
use crate::ring::*;
use crate::rings::poly::*;
use crate::field::*;
use crate::pid::*;
use crate::divisibility::*;
use crate::integer::*;
use crate::rings::finite::*;

///
/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_power_decomposition_finite_field<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
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
            let remaining_part = poly_ring.checked_div(&poly, &poly_ring.prod(result.iter().map(|(g, e)| poly_ring.pow(poly_ring.clone_el(g), *e)))).unwrap();
            result.insert(0, (poly_ring.normalize(remaining_part), 1));
        }
        return result;
    }
}

///
/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
/// 
/// The returned polynomial is always monic, and with this restriction, it
/// is unique.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_squarefree_part_finite_field<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: RingStore,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(!poly_ring.is_zero(&poly));
    if poly_ring.degree(poly).unwrap() == 0 {
        return poly_ring.one();
    }
    let derivate = derive_poly(&poly_ring, poly);
    if poly_ring.is_zero(&derivate) {
        let q = poly_ring.base_ring().size(&BigIntRing::RING).unwrap();
        let (p, e) = is_prime_power(BigIntRing::RING, &q).unwrap();
        let p_usize = int_cast(BigIntRing::RING.clone_el(&p), StaticRing::<i64>::RING, BigIntRing::RING) as usize;
        assert!(p_usize > 0);
        let power = BigIntRing::RING.pow(p, e - 1);
        let undo_frobenius = |x| poly_ring.base_ring().pow_gen(poly_ring.base_ring().clone_el(x), &power, BigIntRing::RING);
        let base_poly = poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| {
            debug_assert!(i % p_usize == 0);
            (undo_frobenius(c), i / p_usize)
        }));
        return poly_squarefree_part_finite_field(poly_ring, &base_poly);
    } else {
        let square_part = poly_ring.ideal_gen(poly, &derivate);
        let result = poly_ring.checked_div(poly, &square_part).unwrap();
        return poly_ring.normalize(result);
    }
}

const FAST_POLY_EEA_THRESHOLD: usize = 32;

///
/// Computes linearly independent vectors `(s, t)` and `(s', t')` and `a, a'` 
/// such that `a = s * lhs + t * rhs` and `a' = s' * lhs + t' * rhs` are both of
/// degree at most `target_deg`, or alternatively one of them is zero (in which
/// case the degree cannot be reduced further).
/// 
/// The degrees of `s, t, s', t'` are bounded as
/// ```text
///   deg(s) < deg(rhs) - deg(a)
///   deg(t) < deg(lhs) - deg(a)
///   deg(s') < deg(rhs) - deg(a')
///   deg(t') < deg(lhs) - deg(a')
/// ```
///
#[instrument(skip_all, level = "trace")]
fn partial_eea<P>(ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize) -> ([El<P>; 4], [El<P>; 2])
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field
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

    while ring.degree(&a).unwrap() > target_deg && !ring.is_zero(&b) {
        debug_assert!(ring.eq_el(&a, &ring.add(ring.mul_ref(&sa, &lhs), ring.mul_ref(&ta, &rhs))));
        debug_assert!(ring.eq_el(&b, &ring.add(ring.mul_ref(&sb, &lhs), ring.mul_ref(&tb, &rhs))));

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quo, &tb));
        sa = ring.sub(sa, ring.mul_ref_snd(quo, &sb));
        a = rem;

        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
        
        debug_assert_eq!(ring.degree(&sb).unwrap(), ring.degree(&rhs).unwrap() - ring.degree(&a).unwrap());
        debug_assert_eq!(ring.degree(&tb).unwrap(), ring.degree(&lhs).unwrap() - ring.degree(&a).unwrap());
    }
    return ([sa, ta, sb, tb], [a, b]);
}

///
/// Computes a Bezout identity for polynomials, using a fast divide-and-conquer
/// polynomial gcd algorithm. Unless you are implementing [`crate::pid::PrincipalIdealRing`]
/// for a custom type, you should use [`crate::pid::PrincipalIdealRing::extended_ideal_gen()`]
/// to get a Bezout identity instead.
/// 
/// A Bezout identity is exactly as specified by [`crate::pid::PrincipalIdealRing::extended_ideal_gen()`],
/// i.e. `s, t, d` such that `d` is the gcd of `lhs` and `rhs`, and `d = lhs * s + rhs * t`.
/// Note that this algorithm does not try to avoid coefficient growth, and thus is only fast
/// over finite fields. Furthermore, it will fall back to a slightly less efficient variant of
/// the standard Euclidean algorithm on small inputs.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn fast_poly_eea<P>(poly_ring: P, lhs: El<P>, rhs: El<P>) -> (El<P>, El<P>, El<P>)
    where P: RingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    fn fast_poly_eea_impl<P>(poly_ring: P, lhs: El<P>, rhs: El<P>, target_deg: usize, memory: &mut [El<P>]) -> ([El<P>; 4], [El<P>; 2])
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field
    {
        if poly_ring.is_zero(&lhs) || poly_ring.is_zero(&rhs) {
            return ([poly_ring.one(), poly_ring.zero(), poly_ring.zero(), poly_ring.one()], [lhs, rhs]);
        }
        let ldeg = poly_ring.degree(&lhs).unwrap();
        let rdeg = poly_ring.degree(&rhs).unwrap();
        if ldeg < target_deg + FAST_POLY_EEA_THRESHOLD || rdeg < target_deg + FAST_POLY_EEA_THRESHOLD {
            return partial_eea(poly_ring, lhs, rhs, target_deg);
        } else if ldeg >= 2 * rdeg {
            let (mut q, r) = poly_ring.euclidean_div_rem(lhs, &rhs);
            poly_ring.negate_inplace(&mut q);
            let (transform, rest) = fast_poly_eea_impl(poly_ring, r, rhs, target_deg, memory);
            let mut transform: (_, _, _, _) = transform.into();
            transform.1 = poly_ring.fma(&q, &transform.0, transform.1);
            transform.3 = poly_ring.fma(&q, &transform.2, transform.3);
            return (transform.into(), rest);
        } else if rdeg >= 2 * ldeg {
            let (transform, rest) = fast_poly_eea_impl(poly_ring, rhs, lhs, target_deg, memory);
            let transform: (_, _, _, _) = transform.into();
            return ([transform.1, transform.0, transform.3, transform.2], rest);
        }
        let split_deg = max(ldeg, rdeg) / 3;
        assert!(2 * split_deg + 1 < max(ldeg, rdeg));
        let part_target_deg = max(split_deg, target_deg.saturating_sub(split_deg));

        let lhs_upper = poly_ring.from_terms(poly_ring.terms(&lhs).filter(|(_, i)| *i >= split_deg).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_deg)));
        let mut lhs_lower = lhs;
        poly_ring.truncate_monomials(&mut lhs_lower, split_deg);
        let rhs_upper = poly_ring.from_terms(poly_ring.terms(&rhs).filter(|(_, i)| *i >= split_deg).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_deg)));
        let mut rhs_lower = rhs;
        poly_ring.truncate_monomials(&mut rhs_lower, split_deg);

        let (fst_transform, [mut lhs_rest, mut rhs_rest]) = fast_poly_eea_impl(poly_ring, lhs_upper, rhs_upper, part_target_deg, memory);
        poly_ring.mul_assign_monomial(&mut lhs_rest, split_deg);
        poly_ring.mul_assign_monomial(&mut rhs_rest, split_deg);

        lhs_rest = poly_ring.fma(&fst_transform[0], &lhs_lower, lhs_rest);
        lhs_rest = poly_ring.fma(&fst_transform[1], &rhs_lower, lhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[2], &lhs_lower, rhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[3], &rhs_lower, rhs_rest);

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
            memory
        );

        return (result, rest);
    }

    if poly_ring.is_zero(&lhs) {
        return (poly_ring.zero(), poly_ring.one(), rhs);
    } else if poly_ring.is_zero(&rhs) {
        return (poly_ring.one(), poly_ring.zero(), lhs);
    }
    let ([s1, t1, s2, t2], [a1, a2]) = fast_poly_eea_impl(poly_ring, lhs, rhs, 0, &mut (0..strassen_mem_size(false, 2, 0)).map(|_| poly_ring.zero()).collect::<Vec<_>>());
    if poly_ring.is_zero(&a1) {
        return (s2, t2, a2);
    } else {
        assert!(poly_ring.is_zero(&a2));
        return (s1, t1, a1);
    }
}

#[cfg(test)]
use crate::rings::zn::zn_64;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_partial_eea() {
    LogAlgorithmSubscriber::init_test();
    let field = zn_64::Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
        X.pow_ref(10) + X.pow_ref(6) + 1,
    ]);

    for k in (1..10).rev() {
        let ([s1, t1, s2, t2], [a, b]) = partial_eea(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g), k);
        assert_el_eq!(&poly_ring, poly_ring.add(poly_ring.mul_ref(&s1, &f), poly_ring.mul_ref(&t1, &g)), &a);
        assert_el_eq!(&poly_ring, poly_ring.add(poly_ring.mul_ref(&s2, &f), poly_ring.mul_ref(&t2, &g)), &b);
        assert_eq!(k, poly_ring.degree(&a).unwrap());
        assert_eq!(k - 1, poly_ring.degree(&b).unwrap());
        assert!(poly_ring.degree(&s1).is_none() || poly_ring.degree(&s1).unwrap() < 10 - poly_ring.degree(&a).unwrap());
        assert!(poly_ring.degree(&t1).is_none() || poly_ring.degree(&t1).unwrap() < 9 - poly_ring.degree(&a).unwrap());
        assert!(poly_ring.degree(&s2).is_none() || poly_ring.degree(&s2).unwrap() < 10 - poly_ring.degree(&b).unwrap());
        assert!(poly_ring.degree(&t2).is_none() || poly_ring.degree(&t2).unwrap() < 9 - poly_ring.degree(&b).unwrap());
    }
}

#[test]
fn test_fast_poly_eea() {
    LogAlgorithmSubscriber::init_test();
    let field = zn_64::Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(90) - X.pow_ref(70) + 3 * X.pow_ref(20) - 1,
        X.pow_ref(100) + X.pow_ref(60) + 1,
    ]);

    let (s, t, d) = fast_poly_eea(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g));
    assert!(poly_ring.is_unit(&d));
    assert_el_eq!(&poly_ring, &d, poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g)));

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(9) - X.pow_ref(7) + 3 * X.pow_ref(2) - 1,
        X.pow_ref(100) + X.pow_ref(60) + 1,
    ]);

    let (s, t, d) = fast_poly_eea(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g));
    assert!(poly_ring.is_unit(&d));
    assert_el_eq!(&poly_ring, &d, poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g)));
}