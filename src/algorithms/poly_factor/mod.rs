use crate::computation::*;
use super::poly_gcd::PolyTFracGCDRing;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::finite::FiniteRing;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::rings::zn::zn_64::*;

use finite::*;
use rational::*;

///
/// Contains algorithms for computing the factorization of polynomials.
/// 
pub mod factor_locally;
///
/// Contains an implementation of the Cantor-Zassenhaus algorithm for
/// finding factors of univariate polynomials over finite fields.
/// 
/// Additionally, a distinct-degree factorization and variants of Cantor-
/// Zassenhaus are also implemented.
/// 
pub mod cantor_zassenhaus;

///
/// Contains an an algorithm to factor univariate polynomials over 
/// field extensions.
/// 
pub mod extension;

///
/// Contains an algorithm to factor univariate polynomials over the integers
/// and the rational numbers.
/// 
pub mod rational;

///
/// Contains an algorithm to factor univariate polynomials over finite fields,
/// based on the more basic functionality of [`cantor_zassenhaus`].
/// 
pub mod finite;

///
/// Trait for fields over which we can efficiently factor polynomials.
/// For details, see the only associated function [`FactorPolyField::factor_poly()`].
/// 
pub trait FactorPolyField: Field + PolyTFracGCDRing {

    ///
    /// Factors a univariate polynomial with coefficients in this field into its irreducible factors.
    /// 
    /// All factors must be monic and but may be returned in any order (with multiplicities). The
    /// unit `poly / prod_i factor[i]^multiplicity[i]` (which is a unit in the base ring) is returned
    /// as second tuple element.
    /// 
    /// # Example - factorization over `QQ`
    /// ```rust
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
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;

    ///
    /// As [`FactorPolyField::factor_poly()`], this computes the factorization of
    /// a polynomial. However, it additionally accepts a [`ComputationController`]
    /// to customize the performed computation.
    /// 
    fn factor_poly_with_controller<P, Controller>(poly_ring: P, poly: &El<P>, _: Controller) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        Self::factor_poly(poly_ring, poly)
    }

    ///
    /// Returns whether the given polynomial is irreducible over the base field.
    /// 
    /// This is functionally equivalent to checking whether the output of [`FactorPolyField::factor_poly()`]
    /// has only a single factor, but may be faster.
    /// 
    fn is_irred<P>(poly_ring: P, poly: &El<P>) -> bool
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let factorization = Self::factor_poly(poly_ring, poly).0;
        return factorization.len() == 1 && factorization[0].1 == 1;
    }
}

impl<R: ?Sized> FactorPolyField for R
    where R: FiniteRing + Field + SelfIso
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        Self::factor_poly_with_controller(poly_ring, poly, DontObserve)
    }

    fn factor_poly_with_controller<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        poly_factor_finite_field(poly_ring, poly, controller)
    }
}

impl<I> FactorPolyField for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing,
        Zn64BBase: CanHomFrom<I::Type>
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        Self::factor_poly_with_controller(poly_ring, poly, DontObserve)
    }

    fn factor_poly_with_controller<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        poly_factor_rational(poly_ring, poly, controller)
    }
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::*;

#[test]
fn test_factor_rational_poly() {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f = poly_ring.from_terms([(incl.map(2), 0), (incl.map(1), 3)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f), poly_ring.clone_el(&f), poly_ring.clone_el(&g)].into_iter()));
    assert_eq!(2, actual.len());
    assert_el_eq!(poly_ring, f, actual[0].0);
    assert_eq!(2, actual[0].1);
    assert_el_eq!(poly_ring, g, actual[1].0);
    assert_eq!(1, actual[1].1);
    assert_el_eq!(QQ, QQ.one(), unit);

    let f = poly_ring.from_terms([(incl.map(3), 0), (incl.map(1), 1)]);
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
    assert_eq!(1, actual.len());
    assert_eq!(1, actual[0].1);
    assert_el_eq!(&poly_ring, f, &actual[0].0);
    assert_el_eq!(QQ, QQ.one(), unit);

    let [mut f] = poly_ring.with_wrapped_indeterminate(|X| [16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4)]);
    poly_ring.inclusion().mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
    assert_eq!(1, actual.len());
    assert_eq!(2, actual[0].1);
    assert_el_eq!(QQ, QQ.one(), unit);
}

#[test]
fn test_factor_nonmonic_poly() {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f = poly_ring.from_terms([(QQ.div(&incl.map(3), &incl.map(5)), 0), (incl.map(1), 4)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f), poly_ring.clone_el(&f), poly_ring.clone_el(&g), poly_ring.int_hom().map(100)].into_iter()));
    assert_eq!(2, actual.len());

    assert_el_eq!(poly_ring, g, actual[0].0);
    assert_eq!(1, actual[0].1);
    assert_el_eq!(poly_ring, f, actual[1].0);
    assert_eq!(2, actual[1].1);
    assert_el_eq!(QQ, incl.map(100), unit);
}

#[test]
fn test_factor_fp() {
    let Fp = zn_static::Fp::<5>::RING;
    let poly_ring = DensePolyRing::new(Fp, "X");
    let f = poly_ring.from_terms([(1, 0), (2, 1), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (1, 1)].into_iter());
    let h = poly_ring.from_terms([(2, 0), (1, 2)].into_iter());
    let fgghhh = poly_ring.prod([&f, &g, &g, &h, &h, &h].iter().map(|poly| poly_ring.clone_el(poly)));
    let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &fgghhh);
    assert_el_eq!(Fp, Fp.one(), unit);
    
    assert_eq!(2, factorization[0].1);
    assert_el_eq!(poly_ring, g, factorization[0].0);
    assert_eq!(3, factorization[1].1);
    assert_el_eq!(poly_ring, h, factorization[1].0);
    assert_eq!(1, factorization[2].1);
    assert_el_eq!(poly_ring, f, factorization[2].0);
}
