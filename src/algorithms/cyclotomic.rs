use crate::divisibility::*;
use crate::primitive_int::StaticRing;
use crate::rings::poly::*;
use crate::ring::*;
use crate::algorithms;

///
/// Computes the `n`-th cyclotomic polynomial.
/// 
/// The `n`-cyclotomic polynomial is the unique polynomial with integer
/// coefficients whose roots are the primitive `n`-th roots of unity.
/// In other words, we find
/// ```text
///   cyclotomic_polynomial(n) = prod_(i in (Z/nZ)*) X - exp(2 pi i / n)
/// ```
/// 
/// # Performance
/// 
/// The implementation uses an efficient algorithm based on the factorization of `n`.
/// However, since the degree of the result is `phi(n)` (about the same size as `n`),
/// it is recommended to pass a polynomial ring using a sparse representation, as this
/// will significantly improve performance (both within the algorithm, and the size of
/// the result).
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::sparse_poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::assert_el_eq;
/// 
/// let poly_ring = SparsePolyRing::new(StaticRing::<i64>::RING, "X");
/// let cyclo_poly = feanor_math::algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, 3);
/// assert_el_eq!(poly_ring, poly_ring.from_terms([(1, 0), (1, 1), (1, 2)].into_iter()), &cyclo_poly);
/// ```
/// 
pub fn cyclotomic_polynomial<P>(P: P, n: usize) -> El<P>
    where P: RingStore, 
        P::Type: PolyRing + DivisibilityRing
{
    let mut current = P.sub(P.indeterminate(), P.one());
    let ZZ = StaticRing::<i128>::RING;
    let mut power_of_x = 1;
    for (p, e) in algorithms::int_factor::factor(&ZZ, n as i128) {
        power_of_x *= ZZ.pow(p, e - 1) as usize;
        current = P.checked_div(
            &P.from_terms(P.terms(&current).map(|(c, d)| (P.base_ring().clone_el(c), d * p as usize))), 
            &current, 
        ).unwrap();
    }
    return P.from_terms(P.terms(&current).map(|(c, d)| (P.base_ring().clone_el(c), d * power_of_x)));
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::poly::sparse_poly::SparsePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;

#[test]
fn test_cyclotomic_polynomial() {
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 2)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 3)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 4)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 5)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (6, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 6)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 6), (6, 3), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 18)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([
            (1, 48), (1, 47), (1, 46), (6, 43), (6, 42), (5, 41), (6, 40), (6, 39), (1, 36), (1, 35), (1, 34), (1, 33), (1, 32), (1, 31), (6, 28), (6, 26), (6, 24), 
            (6, 22), (6, 20), (1, 17), (1, 16), (1, 15), (1, 14), (1, 13), (1, 12), (6, 9), (6, 8), (5, 7), (6, 6), (6, 5), (1, 2), (1, 1), (1, 0)
        ].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 105)
    ));
}

#[bench]
fn bench_cyclotomic_polynomial_3965585_dense(bencher: &mut test::Bencher) {
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    bencher.iter(|| {
        _ = std::hint::black_box(cyclotomic_polynomial(&poly_ring, std::hint::black_box(257 * 257 * 65)));
    });
}

#[bench]
fn bench_cyclotomic_polynomial_3965585_sparse(bencher: &mut test::Bencher) {
    let poly_ring = SparsePolyRing::new(Fp::<7>::RING, "X");
    bencher.iter(|| {
        _ = std::hint::black_box(cyclotomic_polynomial(&poly_ring, std::hint::black_box(257 * 257 * 65)));
    });
}