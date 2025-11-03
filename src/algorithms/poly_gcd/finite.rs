use std::cmp::max;

use tracing::instrument;

use crate::algorithms::eea::partial_eea_poly;
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

        let lhs_upper = poly_ring.from_terms(poly_ring.terms(&lhs).filter(|(_, i)| *i >= split_deg).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_deg)));
        let mut lhs_lower = lhs;
        poly_ring.truncate_monomials(&mut lhs_lower, split_deg);
        let rhs_upper = poly_ring.from_terms(poly_ring.terms(&rhs).filter(|(_, i)| *i >= split_deg).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_deg)));
        let mut rhs_lower = rhs;
        poly_ring.truncate_monomials(&mut rhs_lower, split_deg);

        assert!(poly_ring.degree(&lhs_upper).unwrap_or(0) + poly_ring.degree(&rhs_upper).unwrap_or(0) <= ldeg + rdeg - split_deg);
        let (fst_transform, [mut lhs_rest, mut rhs_rest]) = fast_poly_eea_impl(poly_ring, lhs_upper, rhs_upper, part_target_deg, memory);

        poly_ring.mul_assign_monomial(&mut lhs_rest, split_deg);
        poly_ring.mul_assign_monomial(&mut rhs_rest, split_deg);

        lhs_rest = poly_ring.fma(&fst_transform[0], &lhs_lower, lhs_rest);
        lhs_rest = poly_ring.fma(&fst_transform[1], &rhs_lower, lhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[2], &lhs_lower, rhs_rest);
        rhs_rest = poly_ring.fma(&fst_transform[3], &rhs_lower, rhs_rest);

        assert!(poly_ring.degree(&lhs_rest).unwrap_or(0) + poly_ring.degree(&rhs_rest).unwrap_or(0) <= ldeg + rdeg - split_deg);
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
    } else if poly_ring.is_zero(&a2) || poly_ring.degree(&a1).unwrap() == 0 {
        return (s1, t1, a1);
    } else {
        assert!(poly_ring.degree(&a2).unwrap() == 0);
        return (s2, t2, a2);
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
fn test_fast_poly_eea() {
    LogAlgorithmSubscriber::init_test();

    let field = zn_64::Zn64B::new(2).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");

    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(96) + X.pow_ref(55),
        X.pow_ref(54),
    ]);
    let (s, t, d) = fast_poly_eea(&poly_ring, poly_ring.clone_el(&f), poly_ring.clone_el(&g));
    assert_el_eq!(&poly_ring, &d, poly_ring.add(poly_ring.mul_ref(&s, &f), poly_ring.mul_ref(&t, &g)));
    assert_el_eq!(&poly_ring, poly_ring.pow(poly_ring.indeterminate(), 54), poly_ring.normalize(d));

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