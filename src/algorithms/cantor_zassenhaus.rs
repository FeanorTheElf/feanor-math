use crate::algorithms;
use crate::divisibility::DivisibilityRingStore;
use crate::euclidean::{EuclideanRingStore, EuclideanRing};
use crate::field::{Field, FieldStore};
use crate::integer::{IntegerRingStore, IntegerRing, BigIntRing};
use crate::ordered::OrderedRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::poly::{PolyRingStore, PolyRing};
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::rings::finite::FiniteRingStore;

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
    poly_ring.mul_assign_base(poly, &inv_lc);
}

fn derive_poly<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing
{
    poly_ring.from_terms(poly_ring.terms(poly)
        .filter(|(_, i)| *i > 0)
        .map(|(c, i)| (poly_ring.base_ring().mul_int_ref(c, i as i32), i - 1))
    )
}

pub fn distinct_degree_factorization<P>(poly_ring: P, mut f: El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field
{
    let p = poly_ring.base_ring().modulus();
    let ZZ = poly_ring.base_ring().integer_ring();
    assert!(!poly_ring.is_zero(&f));

    let mut result = Vec::new();
    result.push(poly_ring.one());
    let mut x_power_q_mod_f = poly_ring.indeterminate();
    while poly_ring.degree(&f) != Some(0) {
        // technically, we could just compute gcd(f, X^(p^i) - X), however p^i might be
        // really large and eea will be very slow. Hence, we do the first modulo operation
        // X^(p^i) mod f using square-and-multiply in the ring F[X]/(f)
        x_power_q_mod_f = pow_mod_f(&poly_ring, x_power_q_mod_f, &f, p, ZZ);
        let fq_defining_poly_mod_f = poly_ring.sub_ref_fst(&x_power_q_mod_f, poly_ring.indeterminate());
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
/// The algorithm relies on the fact that for some monic polynomial T over Fp have
/// ```text
/// T^q - T = T (T^((q - 1)/2) + 1) (T^((q - 1)/2) - 1)
/// ```
/// where `q = p^d`. Furthermore, the three factors are pairwise coprime.
/// Since `X^q - X` divides `T^q - T`, and f is squarefree, we see that
/// f divides `T^q - T` and so
/// ```text
/// f = gcd(T, f) gcd((T^((q - 1)/2) + 1, f) gcd(T^((q - 1)/2) - 1, f)
/// ```
/// The idea is now to choose a random T and check whether `gcd(T^((q - 1)/2) - 1, f)`
/// gives a nontrivial factor of f. When f has two irreducible factors, with roots a, b
/// in Fq, then this works if exactly one of them maps to zero under the polynomial
/// `T^((q - 1)/2) - 1`. Now observe that this is the case if and only if `T(a)` resp.
/// `T(b)` is a square in Fq. Now apparently, for a polynomial chosen uniformly at random
/// among all monic polynomials of degree 2d in Fp\[X\], the values T(a) and T(b) are close
/// to independent and uniform on Fq, and thus the probability that one is a square and
/// the other is not is approximately 1/2.
/// 
/// ## Why is the degree of T equal to 2d ?
/// 
/// Pick an Fp-vector space basis of Fq and write a, b as dxs matrices A, B over Fp, where the
/// i-th column is the representation of `a^i` resp. `b^i` w.r.t. that basis. Then the
/// evaluation of `T(a)` resp. `T(b)` is the matrix-vector product `w^T A` resp. `w^T B` where
/// w is the coefficient vector of T (a vector over Fp). We want that `w^T A` and `w^T B` are
/// uniform and independent. Hence, we want all `w^T c` to be uniform and independent, where
/// c runs through the columns of A resp. B. There are 2d such columns in total, thus s = 2d
/// will do (note that all columns are different, as `1, a, ..., a^(d - 1)` is a basis of Fq
/// and similarly for b). 
///
#[allow(non_snake_case)]
pub fn cantor_zassenhaus<P>(poly_ring: P, f: El<P>, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field,
        <<<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as ZnRing>::IntegerRingBase: CanonicalIso<Base<BigIntRing>>
{
    let ZZ = BigIntRing::RING;
    let p = poly_ring.base_ring().integer_ring().cast(&ZZ, poly_ring.base_ring().integer_ring().clone_el(poly_ring.base_ring().modulus()));
    assert!(ZZ.is_odd(&p));
    assert!(poly_ring.degree(&f).unwrap() % d == 0);
    assert!(poly_ring.degree(&f).unwrap() > d);
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&p) as u128);
    let exp = ZZ.half_exact(ZZ.sub(ZZ.pow(p, d), ZZ.one()));

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
        <P::Type as RingExtension>::BaseRing: ZnRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{
    assert!(!poly_ring.is_zero(&poly));
    let derivate = derive_poly(&poly_ring, &poly);
    if poly_ring.is_zero(&derivate) {
        let p = poly_ring.base_ring().integer_ring().cast(&StaticRing::<i32>::RING, poly_ring.base_ring().integer_ring().clone_el(poly_ring.base_ring().modulus()));
        let base_poly = poly_ring.from_terms(poly_ring.terms(&poly).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i / p as usize)));
        return poly_squarefree_part(poly_ring, base_poly);
    } else {
        let square_part = algorithms::eea::gcd(poly_ring.clone_el(&poly), derivate, &poly_ring);
        return poly_ring.checked_div(&poly, &square_part).unwrap();
    }
}

pub fn factor_complete<'a, P>(poly_ring: P, mut el: El<P>) -> Vec<(El<P>, usize)> 
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field,
        <<<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as ZnRing>::IntegerRingBase: CanonicalIso<Base<BigIntRing>>
{
    assert!(!poly_ring.is_zero(&el));

    let mut result = Vec::new();
    let mut unit = poly_ring.base_ring().one();

    // we repeatedly remove the square-free part
    while !poly_ring.is_unit(&el) {
        let sqrfree_part = poly_squarefree_part(&poly_ring, poly_ring.clone_el(&el));
        assert!(!poly_ring.is_unit(&sqrfree_part));

        // factor the square-free part into distinct-degree factors
        for (d, factor_d) in distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part)).into_iter().enumerate() {
            let mut stack = Vec::new();
            stack.push(factor_d);
            
            // and finally extract each individual factor
            while let Some(mut current) = stack.pop() {
                // normalize current
                let lc = poly_ring.lc(&current).unwrap();
                poly_ring.base_ring().mul_assign_ref(&mut unit, lc);
                let lc_inv = poly_ring.base_ring().div(&poly_ring.base_ring().one(), lc);
                poly_ring.mul_assign_base(&mut current, &lc_inv);

                if poly_ring.is_one(&current) {
                    continue;
                } else if poly_ring.degree(&current) == Some(d) {
                    // add to result
                    let mut found = false;
                    for (factor, power) in &mut result {
                        if poly_ring.eq_el(factor, &current) {
                            *power += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        result.push((current, 1));
                    }
                } else {
                    let factor = cantor_zassenhaus(&poly_ring, poly_ring.clone_el(&current), d);
                    stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                    stack.push(factor);
                }
            }
        }
        el = poly_ring.checked_div(&el, &sqrfree_part).unwrap();
    }
    poly_ring.base_ring().mul_assign_ref(&mut unit, poly_ring.coefficient_at(&el, 0));
    debug_assert!(!poly_ring.base_ring().is_zero(&unit));
    if !poly_ring.base_ring().is_one(&unit) {
        result.push((poly_ring.from(unit), 1));
    }
    return result;
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::rings::zn::zn_42;

#[test]
fn test_poly_squarefree_part() {
    let ring = DensePolyRing::new(Zn::<257>::RING, "X");
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
    let f = ring.from_terms([(ring.base_ring().from_int(3), 0), (ring.base_ring().from_int(1), 10)].into_iter());
    let g = ring.from_terms([(ring.base_ring().from_int(3), 0), (ring.base_ring().from_int(1), 2)].into_iter());
    let mut actual = poly_squarefree_part(&ring, f);
    normalize_poly(&ring, &mut actual);
    assert_el_eq!(&ring, &g, &actual);
}

#[test]
fn test_distinct_degree_factorization() {
    let field = Zn::<2>::RING;
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
    let ring = DensePolyRing::new(Zn::<7>::RING, "X");
    let f = ring.from_terms([(1, 0), (1, 2)].into_iter());
    let g = ring.from_terms([(3, 0), (1, 1), (1, 2)].into_iter());
    let p = ring.mul_ref(&f, &g);
    let mut factor = cantor_zassenhaus(&ring, p, 2);
    normalize_poly(&ring, &mut factor);
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
}
