use crate::algorithms;
use crate::divisibility::DivisibilityRingStore;
use crate::pid::{EuclideanRing, EuclideanRingStore, PrincipalIdealRingStore};
use crate::field::{Field, FieldStore};
use crate::integer::*;
use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::finite::{FiniteRing, FiniteRingStore};

use oorandom;

fn pow_mod_f<P, I>(poly_ring: P, g: El<P>, f: &El<P>, pow: &El<I>, ZZ: I) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        I: IntegerRingStore,
        I::Type: IntegerRing
{
    assert!(!ZZ.is_neg(pow));
    return algorithms::sqr_mul::generic_abs_square_and_multiply(
        g, 
        pow, 
        ZZ, 
        |a| poly_ring.euclidean_rem(poly_ring.pow(a, 2), f), 
        |a, b| poly_ring.euclidean_rem(poly_ring.mul_ref_fst(a, b), f),
        poly_ring.one()
    );
}

///
/// Computes the distinct-degree factorization of a polynomial over a finite
/// field. The given polynomial must be square-free.
/// 
/// Concretely, if a univariate polynomial `f(X)` factors uniquely as
/// `f(X) = f1(X) ... fr(X)`, then the `d`-th distinct-degree factor of `f` is
/// `prod_i fi(X)` where `i` runs through all indices with `deg(fi(X)) = d`.
/// This function returns a list whose `d`-th entry is the `d`-th distinct degree
/// factor. Once the list ends, all further `d`-th distinct degree factors should
/// be considered to be `1`.
/// 
/// To get a square-free polynomial, consider using [`super::poly_squarefree_part()`].
/// 
/// # Example
/// 
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::divisibility::*;
/// # use crate::feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::algorithms::poly_factor::cantor_zassenhaus::*;
/// let Fp = Zn::new(3).as_field().ok().unwrap();
/// let FpX = DensePolyRing::new(Fp, "X");
/// // f = (X^3 + 2 X^2 + 1) (X^3 + 2 X + 1) (X^2 + 1)
/// let f = FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (Fp.one(), 3)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 2), (Fp.one(), 3)].into_iter())
/// ].into_iter());
/// let factorization = distinct_degree_factorization(&FpX, f);
/// assert_eq!(4, factorization.len());
/// assert!(FpX.is_unit(&factorization[0]));
/// assert!(FpX.is_unit(&factorization[1]));
/// assert!(!FpX.is_unit(&factorization[2])); // factorization[2] is some scalar multiple of (X^2 + 1)
/// assert!(!FpX.is_unit(&factorization[3])); // factorization[3] is some scalar multiple of (X^3 + 2 X^2 + 1) (X^3 + 2 X + 1)
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn distinct_degree_factorization<P>(poly_ring: P, mut f: El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: FieldStore + FiniteRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    let ZZ = BigIntRing::RING;
    let q = poly_ring.base_ring().size(&ZZ).unwrap();
    debug_assert!(ZZ.eq_el(&algorithms::int_factor::is_prime_power(&ZZ, &q).unwrap().0, &poly_ring.base_ring().characteristic(&ZZ).unwrap()));
    assert!(!poly_ring.is_zero(&f));

    let mut result = Vec::new();
    result.push(poly_ring.one());
    let mut x_power_Q_mod_f = poly_ring.indeterminate();
    while poly_ring.degree(&f) != Some(0) {
        // technically, we could just compute gcd(f, X^(q^i) - X), however q^i might be
        // really large and eea will be very slow. Hence, we do the first modulo operation
        // X^(q^i) mod f using square-and-multiply in the ring F[X]/(f)
        x_power_Q_mod_f = pow_mod_f(&poly_ring, x_power_Q_mod_f, &f, &q, ZZ);
        let fq_defining_poly_mod_f = poly_ring.sub_ref_fst(&x_power_Q_mod_f, poly_ring.indeterminate());
        let deg_i_factor = poly_ring.ideal_gen(&f, &fq_defining_poly_mod_f);
        f = poly_ring.euclidean_div(f, &deg_i_factor);
        result.push(deg_i_factor);
    }
    result[0] = poly_ring.mul_ref(&result[0], &f);
    return result;
}

///
/// Uses the Cantor-Zassenhaus algorithm to find a nontrivial, factor of a polynomial f
/// over a finite field, that is squarefree and consists only of irreducible factors of 
/// degree d.
/// 
/// # Algorithm
/// 
/// The algorithm relies on the fact that for some monic polynomial T over Fq have
/// ```text
/// T^Q - T = T (T^((Q - 1)/2) + 1) (T^((Q - 1)/2) - 1)
/// ```
/// where `Q = q^d`. Furthermore, the three factors are pairwise coprime.
/// Since `X^Q - X` divides `T^Q - T`, and f is squarefree (so divides `X^Q - X`), 
/// we see that `f` also divides `T^Q - T` and so
/// ```text
/// f = gcd(T, f) gcd((T^((Q - 1)/2) + 1, f) gcd(T^((Q - 1)/2) - 1, f)
/// ```
/// The idea is now to choose a random T and check whether `gcd(T^((Q - 1)/2) - 1, f)`
/// gives a nontrivial factor of f. When f has two irreducible factors, with roots a, b
/// in FQ, then this works if exactly one of them maps to zero under the polynomial
/// `T^((Q - 1)/2) - 1`. Now observe that this is the case if and only if `T(a)` resp.
/// `T(b)` is a square in FQ. Now apparently, for a polynomial chosen uniformly at random
/// among all monic polynomials of degree 2d in `Fq[X]`, the values `T(a)` and `T(b)` are close
/// to independent and uniform on FQ, and thus the probability that one is a square and
/// the other is not is approximately 1/2.
/// 
/// ## Why is the degree of T equal to 2d ?
/// 
/// Pick an Fq-vector space basis of FQ and write a, b as dxs matrices A, B over Fq, where the
/// i-th column is the representation of `a^i` resp. `b^i` w.r.t. that basis. Then the
/// evaluation of `T(a)` resp. `T(b)` is the matrix-vector product `w^T A` resp. `w^T B` where
/// w is the coefficient vector of T (a vector over Fq). We want that `w^T A` and `w^T B` are
/// uniform and independent. Hence, we want all `w^T c` to be uniform and independent, where
/// c runs through the columns of A resp. B. There are 2d such columns in total, thus s = 2d
/// will do (note that all columns are different, as `1, a, ..., a^(d - 1)` is a basis of FQ
/// and similarly for b). 
///
#[stability::unstable(feature = "enable")]
pub fn cantor_zassenhaus<P>(poly_ring: P, f: El<P>, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: FiniteRingStore + FieldStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    let ZZ = BigIntRing::RING;
    let q = poly_ring.base_ring().size(&ZZ).unwrap();
    debug_assert!(ZZ.eq_el(&algorithms::int_factor::is_prime_power(&ZZ, &q).unwrap().0, &poly_ring.base_ring().characteristic(&ZZ).unwrap()));
    assert!(ZZ.is_odd(&q));
    assert!(poly_ring.degree(&f).unwrap() % d == 0);
    assert!(poly_ring.degree(&f).unwrap() > d);
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&q) as u128);
    let exp = ZZ.half_exact(ZZ.sub(ZZ.pow(q, d), ZZ.one()));

    loop {
        let T = poly_ring.from_terms(
            (0..(2 * d)).map(|i| (poly_ring.base_ring().random_element(|| rng.rand_u64()), i))
                .chain(Some((poly_ring.base_ring().one(), 2 * d)))
        );
        let G = poly_ring.sub(pow_mod_f(&poly_ring, T, &f, &exp, ZZ), poly_ring.one());
        let g = poly_ring.ideal_gen(&f, &G);
        if !poly_ring.is_unit(&g) && poly_ring.checked_div(&g, &f).is_none() {
            return g;
        }
    }
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;
#[cfg(test)]
use super::normalize_poly;

#[test]
fn test_distinct_degree_factorization() {
    let field = Fp::<2>::RING;
    let ring = DensePolyRing::new(field, "X");
    let a0 = ring.one();
    let a1 = ring.mul(ring.indeterminate(), ring.from_terms([(1, 0), (1, 1)].into_iter()));
    let a2 = ring.from_terms([(1, 0), (1, 1), (1, 2)].into_iter());
    let a3 = ring.mul(ring.from_terms([(1, 0), (1, 1), (1, 3)].into_iter()), ring.from_terms([(1, 0), (1, 2), (1, 3)].into_iter()));
    let a = ring.prod([&a0, &a1, &a2, &a3].into_iter().map(|x| ring.clone_el(x)));
    let expected = vec![a0, a1, a2, a3];
    let distinct_degree_factorization = distinct_degree_factorization(&ring, a);
    assert_eq!(expected.len(), distinct_degree_factorization.len());
    for (mut f, e) in distinct_degree_factorization.into_iter().zip(expected.into_iter()) {
        normalize_poly(&ring, &mut f);
        assert_el_eq!(&ring, &e, &f);
    }
}

#[test]
fn test_cantor_zassenhaus() {
    let ring = DensePolyRing::new(Fp::<7>::RING, "X");
    let f = ring.from_terms([(1, 0), (1, 2)].into_iter());
    let g = ring.from_terms([(3, 0), (1, 1), (1, 2)].into_iter());
    let p = ring.mul_ref(&f, &g);
    let mut factor = cantor_zassenhaus(&ring, p, 2);
    normalize_poly(&ring, &mut factor);
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
}
