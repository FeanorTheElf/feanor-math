use crate::algorithms::poly_factor::finite::poly_factor_finite_field;
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::prelude::*;
use crate::ring_impls::poly::*;
use crate::ring_properties::finite::FiniteRing;

/// Contains an implementation of the Cantor-Zassenhaus algorithm for
/// finding factors of univariate polynomials over finite fields.
///
/// Additionally, a distinct-degree factorization and variants of Cantor-
/// Zassenhaus are also implemented.
pub mod cantor_zassenhaus;
/// Contains an implementation of polynomial factorization over extension fields.
pub mod extension;
/// Contains an implementation of polynomial factorization over finite fields.
pub mod finite;
/// Contains an implementation of polynomial factorization over the integers.
pub mod integer;

/// Trait for fields over which we can efficiently factor polynomials.
/// For details, see the only associated function [`FactorPolyField::factor_poly()`].
pub trait FactorPolyField: Field + PolyTFracGCDRing {
    /// Factors a univariate polynomial with coefficients in this field into its irreducible
    /// factors.
    ///
    /// All factors must be monic and but may be returned in any order (with multiplicities). The
    /// unit `poly / prod_i factor[i]^multiplicity[i]` (which is a unit in the base ring) is
    /// returned as second tuple element.
    ///
    /// # Example - factorization over `QQ`
    /// ```rust
    /// # use feanor_math::prelude::*;
    /// # use feanor_math::ring_impls::poly::*;
    /// # use feanor_math::ring_impls::rational::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::algorithms::poly_factor::*;
    /// // Unfortunately, the internal gcd computations will *extremely* blow up coefficients;
    /// // If you are unsure, use ZZbig as underlying implementation of ZZ
    /// let ZZ = ZZbig;
    /// let QQ = RationalField::new(ZZ);
    /// let P = dense_poly::DensePolyRing::new(&QQ, "X");
    /// let ZZ_to_QQ = QQ.int_hom();
    /// let fraction = |nom, den| QQ.div(&ZZ_to_QQ.map(nom), &ZZ_to_QQ.map(den));
    /// let [f, g] = P.with_wrapped_indeterminate(|X| {
    ///     [
    ///         X.pow_ref(2) + 3 * X.pow_ref(0) / 2,
    ///         X.pow_ref(2) + 2 * X / 3 + 1,
    ///     ]
    /// });
    ///
    /// let fgg = P.prod([&f, &g, &g, &P.int_hom().map(6)].into_iter().cloned());
    /// let (factorization, unit) = FactorPolyField::factor_poly(&P, &fgg);
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
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>;

    /// Returns whether the given polynomial is irreducible over the base field.
    ///
    /// This is functionally equivalent to checking whether the output of
    /// [`FactorPolyField::factor_poly()`] has only a single factor, but may be faster.
    fn is_irred<P>(poly_ring: P, poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        let factorization = Self::factor_poly(poly_ring, poly).0;
        return factorization.len() == 1 && factorization[0].1 == 1;
    }
}

impl<R> FactorPolyField for R
where
    R: ?Sized + FiniteRing + Field,
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        poly_factor_finite_field(poly_ring, poly)
    }
}

#[cfg(test)]
use crate::ring_impls::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::ring_impls::zn::*;

#[test]
fn test_factor_fp() {
    feanor_tracing::DelayedLogger::init_test();
    let Fp = zn_static::Fp::<5>::RING;
    let poly_ring = DensePolyRing::new(Fp, "X");
    let f = poly_ring.from_terms([(1, 0), (2, 1), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (1, 1)].into_iter());
    let h = poly_ring.from_terms([(2, 0), (1, 2)].into_iter());
    let fgghhh = poly_ring.prod([&f, &g, &g, &h, &h, &h].iter().map(|poly| (*poly).clone()));
    let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &fgghhh);
    assert_el_eq!(Fp, Fp.one(), unit);

    assert_eq!(2, factorization[0].1);
    assert_el_eq!(poly_ring, g, factorization[0].0);
    assert_eq!(3, factorization[1].1);
    assert_el_eq!(poly_ring, h, factorization[1].0);
    assert_eq!(1, factorization[2].1);
    assert_el_eq!(poly_ring, f, factorization[2].0);
}
