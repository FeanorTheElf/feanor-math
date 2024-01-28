use crate::algorithms;
use crate::divisibility::DivisibilityRingStore;
use crate::pid::{EuclideanRingStore, EuclideanRing};
use crate::field::{Field, FieldStore};
use crate::integer::*;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::poly::{PolyRingStore, PolyRing};
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

#[cfg(test)]
fn normalize_poly<P>(poly_ring: P, poly: &mut El<P>)
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    let inv_lc = poly_ring.base_ring().div(&poly_ring.base_ring().one(), poly_ring.lc(poly).unwrap());
    poly_ring.inclusion().mul_assign_map_ref(poly, &inv_lc);
}

fn derive_poly<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing
{
    poly_ring.from_terms(poly_ring.terms(poly)
        .filter(|(_, i)| *i > 0)
        .map(|(c, i)| (poly_ring.base_ring().int_hom().mul_ref_fst_map(c, i as i32), i - 1))
    )
}

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
        let deg_i_factor = algorithms::eea::gcd(poly_ring.clone_el(&f), poly_ring.clone_el(&fq_defining_poly_mod_f), &poly_ring);
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
/// among all monic polynomials of degree 2d in Fq[X], the values T(a) and T(b) are close
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
#[allow(non_snake_case)]
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
        let g = algorithms::eea::gcd(poly_ring.clone_el(&f), G, &poly_ring);
        if !poly_ring.is_unit(&g) && poly_ring.checked_div(&g, &f).is_none() {
            return g;
        }
    }
}

pub fn poly_squarefree_part<P>(poly_ring: P, poly: El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    assert!(!poly_ring.is_zero(&poly));
    let derivate = derive_poly(&poly_ring, &poly);
    if poly_ring.is_zero(&derivate) {
        let p = poly_ring.base_ring().characteristic(&StaticRing::<i64>::RING).unwrap() as usize;
        if poly_ring.terms(&poly).all(|(_, i)| i == 0) {
            return poly;
        } else {
            assert!(p > 0);
        }
        let base_poly = poly_ring.from_terms(poly_ring.terms(&poly).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i / p)));
        return poly_squarefree_part(poly_ring, base_poly);
    } else {
        let square_part = algorithms::eea::gcd(poly_ring.clone_el(&poly), derivate, &poly_ring);
        return poly_ring.checked_div(&poly, &square_part).unwrap();
    }
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;
#[cfg(test)]
use crate::rings::zn::zn_42;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;

#[test]
fn test_poly_squarefree_part() {
    let ring = DensePolyRing::new(Fp::<257>::RING, "X");
    let a = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let b = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let mut squarefree_part = poly_squarefree_part(&ring, a);
    normalize_poly(&ring, &mut squarefree_part);
    assert_el_eq!(&ring, &b, &squarefree_part);
}

#[test]
fn test_poly_squarefree_part_multiplicity_p() {
    let ring = DensePolyRing::new(zn_42::Zn::new(5).as_field().ok().unwrap(), "X");
    let f = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 10)].into_iter());
    let g = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 2)].into_iter());
    let mut actual = poly_squarefree_part(&ring, f);
    normalize_poly(&ring, &mut actual);
    assert_el_eq!(&ring, &g, &actual);
}

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
