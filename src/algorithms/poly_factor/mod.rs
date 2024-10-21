use crate::compute_locally::InterpolationBaseRing;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::field::*;
use crate::divisibility::*;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::rings::zn::zn_64::*;
use crate::rings::extension::FreeAlgebraStore;
use crate::specialization::*;
use crate::rings::zn::*;

use extension::factor_over_extension;
use finite_field::*;
use integer::*;

pub mod cantor_zassenhaus;
pub mod extension;
pub mod integer;
pub mod finite_field;

///
/// Trait for fields over which we can efficiently factor polynomials.
/// For details, see the only associated function [`FactorPolyField::factor_poly()`].
/// 
pub trait FactorPolyField: Field {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;

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

impl<R> FactorPolyField for R
    where R: FreeAlgebra + PerfectField + FiniteRingSpecializable,
        <R::BaseRing as RingStore>::Type: PerfectField + FactorPolyField + InterpolationBaseRing + FiniteRingSpecializable
{
    default fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        if let Some(result) = poly_factor_if_finite_field(&poly_ring, poly) {
            return result;
        } else {
            return factor_over_extension(poly_ring, poly);
        }
    }
    
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
impl FactorPolyField for AsFieldBase<Zn> {

    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_finite_field(poly_ring, poly)
    }
    
    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_finite_field(&poly_ring, poly)
    }
}

impl<'a> FactorPolyField for AsFieldBase<RingRef<'a, ZnBase>> {

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_finite_field(&poly_ring, poly)
    }

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

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_finite_field(&poly_ring, poly)
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

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_finite_field(&poly_ring, poly)
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

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_finite_field(&poly_ring, poly)
    }
}

impl<I> FactorPolyField for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + InterpolationBaseRing,
        ZnBase: CanHomFrom<I::Type>
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_factor_rational(poly_ring, poly)
    }

    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_squarefree_part_rational(&poly_ring, poly)
    }
}

#[stability::unstable(feature = "enable")]
pub fn make_primitive<R>(ZZX: R, f: El<R>) -> El<R>
    where R: RingStore,
        R::Type: PolyRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
{
    let content = ZZX.terms(&f).map(|(c, _)| c).fold(ZZX.base_ring().zero(), |a, b| ZZX.base_ring().ideal_gen(&a, b));
    return ZZX.from_terms(ZZX.terms(&f).map(|(c, i)| (ZZX.base_ring().checked_div(c, &content).unwrap(), i)));
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
/// # use feanor_math::algorithms::poly_factor::poly_squarefree_part_global;
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
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FiniteRingSpecializable
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
use test::Bencher;
#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::finite::*;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;

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
fn random_test_factor_poly_galois_field() {
    let mut rng = oorandom::Rand64::new(1);
    let Fq = GaloisField::new(17, 5);
    let ring = DensePolyRing::new(&Fq, "X");

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f1 = ring.from_terms((0..8).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f2 = ring.from_terms((0..10).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f3 = ring.from_terms((0..10).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f = ring.mul_ref_fst(&f1, ring.mul_ref(&f2, &f3));

        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&ring, &f);

        let product = ring.inclusion().mul_map(ring.prod(factorization.iter().map(|(g, e)| ring.pow(ring.clone_el(g), *e))), unit);
        assert_el_eq!(&ring, &f, &product);
        assert!(factorization.iter().map(|(_, e)| *e).sum::<usize>() >= 3);
        for (g, _) in &factorization {
            assert!(ring.checked_div(&f1, g).is_some() || ring.checked_div(&f2, g).is_some() || ring.checked_div(&f3, g).is_some());
        }
    }
}

#[test]
fn random_test_factor_rational_poly() {
    let mut rng = oorandom::Rand64::new(1);
    let ZZbig = BigIntRing::RING;
    let QQ = RationalField::new(ZZbig);
    let ring = DensePolyRing::new(&QQ, "X");
    let coeff_bound = ZZbig.int_hom().map(10);

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f1 = ring.from_terms((0..8).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f2 = ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f3 = ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f = ring.mul_ref_fst(&f1, ring.mul_ref(&f2, &f3));

        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&ring, &f);

        let product = ring.inclusion().mul_map(ring.prod(factorization.iter().map(|(g, e)| ring.pow(ring.clone_el(g), *e))), unit);
        assert_el_eq!(&ring, &f, &product);
        assert!(factorization.iter().map(|(_, e)| *e).sum::<usize>() >= 3);
        for (g, _) in &factorization {
            assert!(ring.checked_div(&f1, g).is_some() || ring.checked_div(&f2, g).is_some() || ring.checked_div(&f3, g).is_some());
        }
    }
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
fn bench_factor_rational_poly_old(bencher: &mut Bencher) {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f1 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 2), (incl.map(3), 4), (incl.map(1), 8)].into_iter());
    let f2 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4), (incl.map(1), 5), (incl.map(1), 10)].into_iter());
    let f3 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 1), (incl.map(-2), 5), (incl.map(1), 17)].into_iter());
    bencher.iter(|| {
        let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f1), poly_ring.clone_el(&f1), poly_ring.clone_el(&f2), poly_ring.clone_el(&f3), poly_ring.int_hom().map(9)].into_iter()));
        assert_eq!(3, actual.len());
        assert_el_eq!(QQ, QQ.int_hom().map(9), unit);
        for (f, e) in actual.into_iter() {
            if poly_ring.eq_el(&f, &f1) {
                assert_el_eq!(poly_ring, f1, f);
                assert_eq!(2, e);
            } else if poly_ring.eq_el(&f, &f2) {
                assert_el_eq!(poly_ring, f2, f);
                assert_eq!(1, e);
           } else if poly_ring.eq_el(&f, &f3) {
               assert_el_eq!(poly_ring, f3, f);
               assert_eq!(1, e);
            } else {
                panic!("Factorization returned wrong factor {} of ({})^2 * ({}) * ({})", poly_ring.format(&f), poly_ring.format(&f1), poly_ring.format(&f2), poly_ring.format(&f3));
            }
        }
    });
}
