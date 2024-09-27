use finite_field::poly_squarefree_part_if_finite_field;
use integer::rational_poly_squarefree_part_local;

use crate::compute_locally::InterpolationBaseRing;
use crate::divisibility::*;
use crate::field::*;
use crate::rings::zn::*;
use crate::integer::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::field::AsFieldBase;
use crate::rings::poly::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::pid::*;
use crate::field::FieldStore;
use crate::rings::rational::RationalFieldBase;
use crate::specialization::*;

pub mod finite_field;
pub mod integer;

///
/// Trait for fields over which we can efficiently compute the square-free part of a polynomial.
/// 
pub trait PolySquarefreePartField: Field {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;
}

impl<R> PolySquarefreePartField for R
    where R: FreeAlgebra + Field + SpecializeToFiniteField + SpecializeToFiniteRing + PerfectField + SpecializeToFiniteField,
        <R::BaseRing as RingStore>::Type: PerfectField + FactorPolyField + InterpolationBaseRing + SpecializeToFiniteField
{
    default fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        // TODO: Find a good way to do this over number fields efficiently, i.e. locally
        poly_squarefree_part_global(&poly_ring, poly)
    }
}

///
/// Unfortunately, `AsFieldBase<R> where R: RingStore<Type = zn_64::ZnBase>` leads to
/// a conflicting impl with the one for field extensions 
///
impl PolySquarefreePartField for AsFieldBase<zn_64::Zn> {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_global(&poly_ring, poly)
    }
}

impl<'a> PolySquarefreePartField for AsFieldBase<RingRef<'a, zn_64::ZnBase>> {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_global(&poly_ring, poly)
    }
}

///
/// Unfortunately, `AsFieldBase<R> where R: RingStore<Type = zn_big::ZnBase<I>>` leads to
/// a conflicting impl with the one for field extensions 
///
impl<I> PolySquarefreePartField for AsFieldBase<zn_big::Zn<I>>
where I: IntegerRingStore,
    I::Type: IntegerRing
{
fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
{
    poly_squarefree_part_global(&poly_ring, poly)
}
}

impl<'a, I> PolySquarefreePartField for AsFieldBase<RingRef<'a, zn_big::ZnBase<I>>>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_global(&poly_ring, poly)
    }
}

impl<const N: u64> PolySquarefreePartField for zn_static::ZnBase<N, true> {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_global(&poly_ring, poly)
    }
}

impl<I> PolySquarefreePartField for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + InterpolationBaseRing,
        zn_64::ZnBase: CanHomFrom<I::Type>
{
    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        rational_poly_squarefree_part_local(&poly_ring, poly)
    }
}

///
/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_global<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + PrincipalIdealRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + SpecializeToFiniteField
{
    assert!(!poly_ring.is_zero(&poly));
    let squarefree_part = poly_squarefree_part_global(poly_ring, poly);
    if poly_ring.degree(&squarefree_part).unwrap() == poly_ring.degree(&poly).unwrap() {
        return vec![(squarefree_part, 1)];
    } else {
        let square_part = poly_ring.checked_div(&poly, &squarefree_part).unwrap();
        let square_part_decomposition = poly_power_decomposition_global(poly_ring, &square_part);
        let mut result = square_part_decomposition;
        let mut degree = 0;
        for (g, k) in &mut result {
            *k += 1;
            degree += poly_ring.degree(g).unwrap() * *k;
        }
        if degree != poly_ring.degree(&poly).unwrap() {
            let remaining_part = poly_ring.checked_div(&poly, &poly_ring.prod(result.iter().map(|(g, e)| poly_ring.pow(poly_ring.clone_el(g), *e)))).unwrap();
            result.push((poly_ring.normalize(remaining_part), 1));
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
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::algorithms::poly_squarefree::poly_squarefree_part_global;
/// let Fp = Zn::new(3).as_field().ok().unwrap();
/// let FpX = DensePolyRing::new(Fp, "X");
/// // f = (X^2 + 1)^2 (X^3 + 2 X + 1)
/// let [f] = FpX.with_wrapped_indeterminate(|X| [(X.pow_ref(2) + 1).pow(2) * (X.pow_ref(3) + 2 * X + 1)]);
/// let squarefree_part = poly_squarefree_part_global(&FpX, &f);
/// let [expected] = FpX.with_wrapped_indeterminate(|X| [(X.pow_ref(2) + 1) * (X.pow_ref(3) + 2 * X + 1)]);
/// assert_el_eq!(FpX, &expected, &squarefree_part);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_global<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + SpecializeToFiniteField
{
    assert!(!poly_ring.is_zero(&poly));
    if let Some(result) = poly_squarefree_part_if_finite_field(poly_ring, poly) {
        return result;
    }
    let derivate = derive_poly(&poly_ring, poly);
    if poly_ring.degree(&poly).unwrap() == 0 {
        return poly_ring.one();
    }
    if poly_ring.is_zero(&derivate) {
        unimplemented!("infinite field with positive characteristic are currently not supported")
    } else {
        let square_part = poly_ring.ideal_gen(poly, &derivate);
        let result = poly_ring.checked_div(&poly, &square_part).unwrap();
        return poly_ring.normalize(result);
    }
}

#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::extension::galois_field::*;
#[cfg(test)]
use crate::rings::poly::dense_poly::*;
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::rings::extension::FreeAlgebraStore;

use super::poly_factor::FactorPolyField;

#[test]
fn test_poly_squarefree_part_global() {
    let ring = DensePolyRing::new(zn_static::Fp::<257>::RING, "X");
    let poly = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let expected = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let actual = poly_squarefree_part_global(&ring, &poly);
    assert_el_eq!(ring, expected, actual);

    let QQ = RationalField::new(BigIntRing::RING);
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let [mut f, mut expected] = poly_ring.with_wrapped_indeterminate(|X| [
        16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4),
        4 - 4 * X + 11 * X.pow_ref(2)
    ]);
    poly_ring.inclusion().mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    poly_ring.inclusion().mul_assign_map(&mut expected, QQ.div(&QQ.one(), &QQ.int_hom().map(11)));

    let actual = poly_squarefree_part_global(&poly_ring, &f);
    assert_el_eq!(poly_ring, expected, actual);
}

#[test]
fn test_poly_power_decomposition_global() {
    let ring = DensePolyRing::new(zn_static::Fp::<257>::RING, "X");
    let [f1, f2, f4] = ring.with_wrapped_indeterminate(|X| [
        X.pow_ref(2) + 251 * X + 3,
        X.pow_ref(3) + 6 * X + 254,
        X + 4
    ]);
    let poly = ring.prod([
        ring.clone_el(&f1),
        ring.pow(ring.clone_el(&f2), 2),
        ring.pow(ring.clone_el(&f4), 4),
    ]);
    let actual = poly_power_decomposition_global(&ring, &poly);
    assert_eq!(3, actual.len());
    for (f, k) in &actual {
        match k {
            1 => assert_el_eq!(&ring, &f1, f),
            2 => assert_el_eq!(&ring, &f2, f),
            4 => assert_el_eq!(&ring, &f4, f),
            _ => unreachable!()
        }
    }
}

#[test]
fn test_poly_squarefree_part_global_multiplicity_p() {
    let ring = DensePolyRing::new(zn_64::Zn::new(5).as_field().ok().unwrap(), "X");
    let [f] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(10)]);
    let [g] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(2)]);
    let actual = poly_squarefree_part_global(&ring, &f);
    assert_el_eq!(ring, g, actual);
}

#[test]
fn test_poly_squarefree_part_global_galois_field() {
    let ring = DensePolyRing::new(GaloisField::new(2, 3), "X");
    let f = ring.from_terms([(ring.base_ring().pow(ring.base_ring().canonical_gen(), 2), 0), (ring.base_ring().one(), 2)]);
    let g = ring.from_terms([(ring.base_ring().canonical_gen(), 0), (ring.base_ring().one(), 1)]);
    let actual = poly_squarefree_part_global(&ring, &f);
    assert_el_eq!(ring, g, actual);
}