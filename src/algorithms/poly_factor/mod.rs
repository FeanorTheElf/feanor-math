use crate::compute_locally::InterpolationBaseRing;
use crate::divisibility::*;
use crate::field::{Field, FieldStore};
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::field::*;
use crate::rings::poly::{derive_poly, PolyRing, PolyRingStore};
use crate::rings::rational::*;
use crate::rings::zn::zn_64::*;
use crate::rings::extension::FreeAlgebraStore;
use crate::specialization::*;
use finite_field::{factor_if_finite_field, factor_over_finite_field};
use integer::factor_rational_poly;
use crate::rings::zn::*;

use extension::factor_over_extension;

pub mod cantor_zassenhaus;
pub mod extension;
pub mod integer;
pub mod finite_field;

///
/// Trait for fields over which we can efficiently factor polynomials.
/// For details, see the only associated function [`FactorPolyField::factor_poly()`].
/// 
pub trait FactorPolyField: Field {

    ///
    /// Factors a univariate polynomial with coefficients in this ring into its irreducible factors.
    /// This requires that this ring is a UFD, otherwise a unique factorization does not exist in 
    /// the corresponding polynomial ring.
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

    ///
    /// Factors the monic polynomial `f` over `R`, given as an implementation of the ring `R[X]/(f)`.
    /// 
    /// If possible, this will use arithmetic in the given ring to compute the factorization, instead
    /// of using a default implementation of the ring `R[X]/(f)` (if necessary for factoring).
    /// 
    fn find_factor_by_extension<P, S>(poly_ring: P, mod_f_ring: S) -> Option<El<P>>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            S: RingStore,
            S::Type: FreeAlgebra,
            <S::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        assert!(poly_ring.base_ring().get_ring() == mod_f_ring.base_ring().get_ring());
        let poly = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());
        let (factorization, _unit) = Self::factor_poly(poly_ring, &poly);
        if factorization.len() > 1 || factorization[0].1 > 1 {
            return Some(factorization.into_iter().next().unwrap().0);
        } else {
            return None;
        }
    }
}

///
/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
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
/// # use feanor_math::algorithms::poly_factor::poly_squarefree_part;
/// let Fp = Zn::new(3).as_field().ok().unwrap();
/// let FpX = DensePolyRing::new(Fp, "X");
/// // f = (X^2 + 1)^2 (X^3 + 2 X + 1)
/// let f = FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (Fp.one(), 3)].into_iter())
/// ].into_iter());
/// let mut squarefree_part = poly_squarefree_part(&FpX, f);
/// // the result is not necessarily monic, so normalize
/// let normalization_factor = Fp.invert(FpX.lc(&squarefree_part).unwrap()).unwrap();
/// FpX.inclusion().mul_assign_map(&mut squarefree_part, normalization_factor);
/// assert_el_eq!(FpX, FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (Fp.one(), 3)].into_iter())
/// ].into_iter()), &squarefree_part);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part<P>(poly_ring: P, poly: El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + PrincipalIdealRing,
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
        let square_part = poly_ring.ideal_gen(&poly, &derivate);
        return poly_ring.checked_div(&poly, &square_part).unwrap();
    }
}


impl<R> FactorPolyField for R
    where R: FreeAlgebra + Field + SpecializeToFiniteField,
        <R::BaseRing as RingStore>::Type: FactorPolyField + InterpolationBaseRing
{
    default fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        if let Some(result) = factor_if_finite_field(&poly_ring, poly) {
            return result;
        } else {
            return factor_over_extension(poly_ring, poly);
        }
    }
}

impl FactorPolyField for AsFieldBase<Zn> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        factor_over_finite_field(poly_ring, poly)
    }
}

impl<const N: u64> FactorPolyField for zn_static::ZnBase<N, true> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        factor_over_finite_field(poly_ring, poly)
    }
}

impl<I> FactorPolyField for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        ZnBase: CanHomFrom<I::Type>
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        factor_rational_poly(poly_ring, poly)
    }
}

#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::rings::extension::extension_impl::*;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::finite::*;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;

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

#[test]
fn test_poly_squarefree_part() {
    let ring = DensePolyRing::new(zn_static::Fp::<257>::RING, "X");
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
    let squarefree_part = ring.normalize(poly_squarefree_part(&ring, a));
    assert_el_eq!(ring, b, squarefree_part);

    let QQ = RationalField::new(BigIntRing::RING);
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let [mut f] = poly_ring.with_wrapped_indeterminate(|X| [16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4)]);
    poly_ring.inclusion().mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    let actual = poly_squarefree_part(&poly_ring, poly_ring.clone_el(&f));
    poly_ring.println(&f);
    poly_ring.println(&derive_poly(&poly_ring, &f));
    poly_ring.println(&poly_ring.ideal_gen(&f, &derive_poly(&poly_ring, &f)));
    assert_eq!(2, poly_ring.degree(&actual).unwrap());
}

#[test]
fn test_poly_squarefree_part_multiplicity_p() {
    let ring = DensePolyRing::new(zn_64::Zn::new(5).as_field().ok().unwrap(), "X");
    let f = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 10)].into_iter());
    let g = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 2)].into_iter());
    let actual = ring.normalize(poly_squarefree_part(&ring, f));
    assert_el_eq!(ring, g, actual);
}

#[test]
fn test_find_factor_by_extension_finite_field() {
    let field = Zn::new(2).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    assert!(<_ as FactorPolyField>::find_factor_by_extension(&poly_ring, FreeAlgebraImpl::new(field, 2, [field.one(), field.one()])).is_none());

    let poly = poly_ring.mul(
        poly_ring.from_terms([(field.one(), 0), (field.one(), 1), (field.one(), 3)].into_iter()),
        poly_ring.from_terms([(field.one(), 0), (field.one(), 2), (field.one(), 3)].into_iter()),
    );
    let factor = <_ as FactorPolyField>::find_factor_by_extension(&poly_ring, FreeAlgebraImpl::new(field, 6, [field.one(), field.one(), field.one(), field.one(), field.one(), field.one()])).unwrap();
    assert_eq!(3, poly_ring.degree(&factor).unwrap());
    assert!(poly_ring.checked_div(&poly, &factor).is_some());
}

#[bench]
fn bench_factor_random_poly(bencher: &mut Bencher) {
    let degree = 128;
    let Fp = zn_64::Zn::new(17).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(Fp, "X");
    let mut rng = oorandom::Rand64::new(1);
    let f = poly_ring.from_terms((0..degree).map(|d| (Fp.random_element(|| rng.rand_u64()), d)).chain([(Fp.one(), degree)].into_iter()));
    bencher.iter(|| {
        let (factors, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
        assert_el_eq!(&Fp, unit, Fp.one());
        assert_el_eq!(poly_ring, &f, poly_ring.prod(factors.into_iter().map(|(g, e)| poly_ring.pow(g, e))));
    });
}

#[bench]
fn bench_factor_rational_poly(bencher: &mut Bencher) {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f1 = poly_ring.checked_div(&poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 2), (incl.map(1), 4), (incl.map(3), 8)].into_iter()), &poly_ring.int_hom().map(3)).unwrap();
    let f2 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4), (incl.map(1), 5), (incl.map(1), 10)].into_iter());
    let f3 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 1), (incl.map(-2), 5), (incl.map(1), 17)].into_iter());
    bencher.iter(|| {
        let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f1), poly_ring.clone_el(&f1), poly_ring.clone_el(&f2), poly_ring.clone_el(&f3), poly_ring.int_hom().map(9)].into_iter()));
        assert_eq!(3, actual.len());
        assert_el_eq!(QQ, QQ.int_hom().map(9), unit);
        for (f, e) in actual.iter() {
            if poly_ring.eq_el(f, &f1) {
                assert_el_eq!(poly_ring, f1, f);
                assert_eq!(2, *e);
            } else if poly_ring.eq_el(f, &f2) {
                assert_el_eq!(poly_ring, f2, f);
                assert_eq!(1, *e);
           } else if poly_ring.eq_el(f, &f3) {
               assert_el_eq!(poly_ring, f3, f);
               assert_eq!(1, *e);
            } else {
                panic!("Factorization returned wrong factor {} of ({})^2 * {} * {}", poly_ring.format(f), poly_ring.format(&f1), poly_ring.format(&f2), poly_ring.format(&f3));
            }
        }
    });
}
