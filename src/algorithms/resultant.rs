
use crate::reduce_lift::poly_eval::{EvaluatePolyLocallyRing, EvaluatePolyLocallyReductionMap};
use crate::divisibility::{DivisibilityRingStore, Domain};
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::rings::poly::*;
use crate::algorithms;
use crate::homomorphism::*;
use crate::ring::*;
use crate::rings::poly::dense_poly::DensePolyRing;

///
/// Computes the resultant of `f` and `g` over the base ring.
/// 
/// The resultant is the determinant of the linear map
/// ```text
///   R[X]_deg(g)  x  R[X]_deg(f)  ->  R[X]_deg(fg),
///        a       ,       b       ->    af + bg
/// ```
/// where `R[X]_d` refers to the vector space of polynomials in `R[X]` of degree
/// less than `d`.
/// 
/// # Example
/// ```
/// use feanor_math::ring::*;
/// use feanor_math::primitive_int::*;
/// use feanor_math::rings::poly::dense_poly::DensePolyRing;
/// use feanor_math::rings::poly::*;
/// use feanor_math::algorithms::resultant::*;
/// let ZZ = StaticRing::<i64>::RING;
/// let ZZX = DensePolyRing::new(ZZ, "X");
/// let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X + 1]);
/// // the discrimiant is the resultant of f and f'
/// let discriminant = resultant_global(&ZZX, ZZX.clone_el(&f), derive_poly(&ZZX, &f));
/// assert_eq!(0, discriminant);
/// ```
/// 
pub fn resultant_global<P>(ring: P, mut f: El<P>, mut g: El<P>) -> El<<P::Type as RingExtension>::BaseRing>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Domain + PrincipalIdealRing
{
    let base_ring = ring.base_ring();
    if ring.is_zero(&g) || ring.is_zero(&f) {
        return base_ring.zero();
    }
    let mut scale_den = base_ring.one();
    let mut scale_num = base_ring.one();

    if ring.degree(&g).unwrap() < ring.degree(&f).unwrap() {
        if (ring.degree(&g).unwrap() + ring.degree(&f).unwrap()) % 2 != 0 {
            base_ring.negate_inplace(&mut scale_num);
        }
        std::mem::swap(&mut f, &mut g);
    }

    while ring.degree(&f).unwrap_or(0) >= 1 {

        let balance_factor = ring.get_ring().balance_poly(&mut f);
        if let Some(balance_factor) = balance_factor {
            base_ring.mul_assign(&mut scale_num, base_ring.pow(balance_factor, ring.degree(&g).unwrap()));
        }

        // use here that `res(f, g) = a^(-deg(f)) lc(f)^(deg(g) - deg(ag - fh)) res(f, ag - fh)` if `deg(fh) <= deg(g)`
        let deg_g = ring.degree(&g).unwrap();
        let (_q, r, a) = algorithms::poly_div::poly_div_rem_domain(ring, g, &f);
        let deg_r = ring.degree(&r).unwrap_or(0);

        // adjust the scaling factor - we cancel out gcd's to prevent excessive number growth
        base_ring.mul_assign(&mut scale_den, base_ring.pow(a, ring.degree(&f).unwrap()));
        base_ring.mul_assign(&mut scale_num, base_ring.pow(base_ring.clone_el(ring.lc(&f).unwrap()), deg_g - deg_r));
        let gcd = base_ring.ideal_gen(&scale_den, &scale_num);
        scale_den = base_ring.checked_div(&scale_den, &gcd).unwrap();
        scale_num = base_ring.checked_div(&scale_num, &gcd).unwrap();

        g = f;
        f = r;
    }

    if ring.is_zero(&f) {
        return base_ring.zero();
    } else {
        let mut result = base_ring.clone_el(&ring.coefficient_at(&f, 0));
        result = base_ring.pow(result, ring.degree(&g).unwrap());
        base_ring.mul_assign(&mut result, scale_num);
        return base_ring.checked_div(&result, &scale_den).unwrap();
    }
}

///
/// Computes the resultant of `f` and `g` over the base ring.
/// 
/// The resultant is the determinant of the linear map
/// ```text
///   R[X]_deg(g)  x  R[X]_deg(f)  ->  R[X]_deg(fg),
///        a       ,       b       ->    af + bg
/// ```
/// where `R[X]_d` refers to the vector space of polynomials in `R[X]` of degree
/// less than `d`.
/// 
/// As opposed to [`resultant_global()`], this function does so by first
/// computing the resultant modulo multiple prime ideals, and then reconstructing
/// the full resultant from this. For infinite rings, this is usually much faster.
/// 
/// # Example
/// ```
/// use feanor_math::assert_el_eq;
/// use feanor_math::ring::*;
/// use feanor_math::integer::*;
/// use feanor_math::rings::poly::dense_poly::DensePolyRing;
/// use feanor_math::rings::poly::*;
/// use feanor_math::algorithms::resultant::*;
/// let ZZ = BigIntRing::RING;
/// let ZZX = DensePolyRing::new(ZZ, "X");
/// let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X + 1]);
/// // the discrimiant is the resultant of f and f'
/// let discriminant = resultant_local(&ZZX, ZZX.clone_el(&f), derive_poly(&ZZX, &f));
/// assert_el_eq!(ZZ, ZZ.zero(), discriminant);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn resultant_local<'a, P>(ring: P, f: El<P>, g: El<P>) -> El<<P::Type as RingExtension>::BaseRing>
    where P: 'a + RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: EvaluatePolyLocallyRing
{
    let base_ring = ring.base_ring();
    if ring.is_zero(&f) || ring.is_zero(&g) {
        return base_ring.zero();
    }
    let ln_max_norm = ring.terms(&f).map(|(c, _)| base_ring.get_ring().ln_pseudo_norm(c)).max_by(f64::total_cmp).unwrap() * ring.degree(&g).unwrap() as f64 +
        ring.terms(&g).map(|(c, _)| base_ring.get_ring().ln_pseudo_norm(c)).max_by(f64::total_cmp).unwrap() * ring.degree(&f).unwrap() as f64;
    let work_locally = base_ring.get_ring().local_computation(ln_max_norm);
    let mut resultants = Vec::new();
    for i in 0..base_ring.get_ring().local_ring_count(&work_locally) {
        let embedding = EvaluatePolyLocallyReductionMap::new(base_ring.get_ring(), &work_locally, i);
        let new_poly_ring = DensePolyRing::new(embedding.codomain(), "X");
        let poly_ring_embedding = new_poly_ring.lifted_hom(ring, &embedding);
        let local_f = poly_ring_embedding.map_ref(&f);
        let local_g = poly_ring_embedding.map_ref(&g);
        resultants.push(resultant_global(&new_poly_ring, local_f, local_g));
    }
    return base_ring.get_ring().lift_combine(&work_locally, &resultants);
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::rational::*;
#[cfg(test)]
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;
#[cfg(test)]
use crate::rings::multivariate::*;
#[cfg(test)]
use crate::algorithms::buchberger::buchberger_simple;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::algorithms::poly_gcd::PolyTFracGCDRing;

#[test]
fn test_resultant() {
    let ZZ = StaticRing::<i64>::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    // a quadratic polynomial and its derivative - the resultant should be the discriminant
    let f = ZZX.from_terms([(3, 0), (-5, 1), (1, 2)].into_iter());
    let g = ZZX.from_terms([(-5, 0), (2, 1)].into_iter());

    assert_el_eq!(ZZ, 13, resultant_global(&ZZX, ZZX.clone_el(&f), ZZX.clone_el(&g)));
    assert_el_eq!(ZZ, -13, resultant_global(&ZZX, g, f));

    // if f and g have common factors, this should be zero
    let f = ZZX.from_terms([(1, 0), (-2, 1), (1, 2)].into_iter());
    let g = ZZX.from_terms([(-1, 0), (1, 2)].into_iter());
    assert_el_eq!(ZZ, 0, resultant_global(&ZZX, f, g));

    // a slightly larger example
    let f = ZZX.from_terms([(5, 0), (-1, 1), (3, 2), (1, 4)].into_iter());
    let g = ZZX.from_terms([(-1, 0), (-1, 2), (1, 3), (4, 5)].into_iter());
    assert_el_eq!(ZZ, 642632, resultant_global(&ZZX, f, g));
}

#[test]
fn test_resultant_local_polynomial() {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);
    static_assert_impls!(RationalFieldBase<BigIntRing>: PolyTFracGCDRing);
    // we eliminate `Y`, so add it as the outer indeterminate
    let QQX = DensePolyRing::new(&QQ, "X");
    let QQXY = DensePolyRing::new(&QQX, "Y");
    let ZZ_to_QQ = QQ.int_hom();

    // 1 + X^2 + 2 Y + (1 + X) Y^2
    let f= QQXY.from_terms([
        (vec![(1, 0), (1, 2)], 0),
        (vec![(2, 0)], 1),
        (vec![(1, 0), (1, 1)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    // 3 + X + (2 + X) Y + (1 + X + X^2) Y^2
    let g = QQXY.from_terms([
        (vec![(3, 0), (1, 1)], 0),
        (vec![(2, 0), (1, 1)], 1),
        (vec![(1, 0), (1, 1), (1, 2)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    let actual = QQX.normalize(resultant_global(&QQXY, QQXY.clone_el(&f), QQXY.clone_el(&g)));
    let actual_local = resultant_local(&QQXY, QQXY.clone_el(&f), QQXY.clone_el(&g));
    assert_el_eq!(&QQX, &actual, &actual_local);

    let QQYX = MultivariatePolyRingImpl::new(&QQ, 2);
    // reverse the order of indeterminates, so that we indeed eliminate `Y`
    let [f, g] = QQYX.with_wrapped_indeterminates(|[Y, X]| [ 1 + X.pow_ref(2) + 2 * Y + (1 + X) * Y.pow_ref(2), 3 + X + (2 + X) * Y + (1 + X + X.pow_ref(2)) * Y.pow_ref(2) ]);

    let gb = buchberger_simple::<_, _>(&QQYX, vec![f, g], Lex);
    let expected = gb.into_iter().filter(|poly| QQYX.appearing_indeterminates(&poly).len() == 1).collect::<Vec<_>>();
    assert!(expected.len() == 1);
    let expected = QQX.normalize(QQX.from_terms(QQYX.terms(&expected[0]).map(|(c, m)| (QQ.clone_el(c), QQYX.exponent_at(m, 1)))));

    assert_el_eq!(QQX, expected, actual);
}

#[test]
fn test_resultant_local_integer() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [
        X.pow_ref(32) + 1,
        X.pow_ref(2) - X - 1
    ]);
    assert_el_eq!(ZZ, ZZ.int_hom().map(4870849), resultant_local(&ZZX, f, g));
}