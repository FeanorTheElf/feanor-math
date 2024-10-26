use crate::compute_locally::InterpolationBaseRing;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::field::*;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::rings::zn::zn_64::*;
use crate::specialization::*;
use crate::rings::zn::*;

use extension::poly_factor_extension;
use finite::*;
use rational::*;

use super::poly_gcd::PolyGCDRing;

pub mod cantor_zassenhaus;
pub mod extension;
pub mod rational;
pub mod finite;

///
/// Trait for fields over which we can efficiently factor polynomials.
/// For details, see the only associated function [`FactorPolyField::factor_poly()`].
/// 
pub trait FactorPolyField: Field {

    ///
    /// Factors a univariate polynomial with coefficients in this field into its irreducible factors.
    /// 
    /// All factors must be monic and but may be returned in any order (with multiplicities). The
    /// unit `poly / prod_i factor[i]^multiplicity[i]` (which is a unit in the base ring) is returned
    /// as second tuple element.
    /// 
    /// # Example - factorization over `QQ`
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::rings::poly::*;
    /// # use feanor_math::rings::rational::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::field::*;
    /// # use feanor_math::algorithms::poly_factor::*;
    /// // Unfortunately, the internal gcd computations will *extremely* blow up coefficients;
    /// // If you are unsure, use BigIntRing::RING as underlying implementation of ZZ
    /// let ZZ = StaticRing::<i128>::RING;
    /// let QQ = RationalField::new(ZZ);
    /// let P = dense_poly::DensePolyRing::new(QQ, "X");
    /// let ZZ_to_QQ = QQ.can_hom(&ZZ).unwrap();
    /// let fraction = |nom: i128, den: i128| QQ.div(&ZZ_to_QQ.map(nom), &ZZ_to_QQ.map(den));
    /// 
    /// // f is X^2 + 3/2
    /// let f = P.from_terms([(fraction(3, 2), 0), (fraction(1, 1), 2)].into_iter());
    /// 
    /// // g is X^2 + 2/3 X + 1
    /// let g = P.from_terms([(fraction(1, 1), 0), (fraction(2, 3), 1), (fraction(1, 1), 2)].into_iter());
    /// 
    /// let fgg = P.prod([&f, &g, &g, &P.int_hom().map(6)].iter().map(|poly| P.clone_el(poly)));
    /// let (factorization, unit) = <RationalFieldBase<_> as FactorPolyField>::factor_poly(&P, &fgg);
    /// assert_eq!(2, factorization.len());
    /// if P.eq_el(&f, &factorization[0].0) {
    ///     assert_eq!(1, factorization[0].1);
    ///     assert_eq!(2, factorization[1].1);
    ///     assert_el_eq!(P, g, factorization[1].0);
    /// } else {
    ///     assert_eq!(2, factorization[0].1);
    ///     assert_eq!(1, factorization[1].1);
    ///     assert_el_eq!(P, g, factorization[0].0);
    ///     assert_el_eq!(P, f, factorization[1].0);
    /// }
    /// assert_el_eq!(QQ, ZZ_to_QQ.map(6), unit);
    /// ```
    /// 
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;
}

impl<R> FactorPolyField for R
    where R: FreeAlgebra + PerfectField + FiniteRingSpecializable + PolyGCDRing,
        <R::BaseRing as RingStore>::Type: PerfectField + FactorPolyField + InterpolationBaseRing + FiniteRingSpecializable + PolyGCDRing
{
    default fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        if let Some(result) = poly_factor_if_finite_field(&poly_ring, poly) {
            return result;
        } else {
            return poly_factor_extension(poly_ring, poly);
        }
    }
}

///
/// Unfortunately, `AsFieldBase<R> where R: RingStore<Type = zn_64::ZnBase>` leads to
/// a conflicting impl with the one for field extensions 
///
impl FactorPolyField for AsFieldBase<Zn> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

impl<'a> FactorPolyField for AsFieldBase<RingRef<'a, ZnBase>> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

///
/// Unfortunately, `AsFieldBase<R> where R: RingStore<Type = zn_big::ZnBase<I>>` leads to
/// a conflicting impl with the one for field extensions 
///
impl<I> FactorPolyField for AsFieldBase<zn_big::Zn<I>>
where I: IntegerRingStore,
    I::Type: IntegerRing
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

impl<'a, I> FactorPolyField for AsFieldBase<RingRef<'a, zn_big::ZnBase<I>>>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

impl<const N: u64> FactorPolyField for zn_static::ZnBase<N, true> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

impl<I> FactorPolyField for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing,
        ZnBase: CanHomFrom<I::Type>
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_rational(poly_ring, poly)
    }
}
