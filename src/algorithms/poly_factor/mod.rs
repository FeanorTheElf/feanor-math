use std::cmp::min;
use std::alloc::Allocator;

use crate::divisibility::*;
use crate::field::{Field, FieldStore};
use crate::homomorphism::Homomorphism;
use crate::integer::{int_cast, BigIntRing, IntegerRing, IntegerRingStore};
use crate::ordered::*;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::extension::extension_impl::{FreeAlgebraImpl, FreeAlgebraImplBase};
use crate::rings::extension::FreeAlgebra;
use crate::rings::field::AsFieldBase;
use crate::rings::finite::FiniteRing;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{derive_poly, PolyRing, PolyRingStore};
use crate::algorithms::{self, int_bisect};
use crate::rings::rational::*;
use crate::rings::zn::zn_64::*;
use crate::rings::zn::{choose_zn_impl, ZnOperation, ZnRing, ZnRingStore};
use crate::vector::VectorView;
use crate::rings::fieldextension::*;

use super::erathostenes;
use super::hensel::hensel_lift_factorization;

pub mod cantor_zassenhaus;
pub mod number_field;

fn binomial(n: usize, mut k: usize) -> El<BigIntRing> {
    if k > n {
        BigIntRing::RING.zero()
    } else {
        k = min(k, n - k);
        let to_ZZbig = BigIntRing::RING.can_hom(&StaticRing::<i64>::RING).unwrap();
        BigIntRing::RING.checked_div(&BigIntRing::RING.prod(((n - k + 1)..=n).map(|k| to_ZZbig.map(k as i64))), &BigIntRing::RING.prod((1..=k).map(|k| to_ZZbig.map(k as i64)))).unwrap()
    }
}

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
    ///     assert_el_eq!(&P, &g, &factorization[1].0);
    /// } else {
    ///     assert_eq!(2, factorization[0].1);
    ///     assert_eq!(1, factorization[1].1);
    ///     assert_el_eq!(&P, &g, &factorization[0].0);
    ///     assert_el_eq!(&P, &f, &factorization[1].0);
    /// }
    /// assert_el_eq!(&QQ, &ZZ_to_QQ.map(6), &unit);
    /// ```
    /// 
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;
}

///
/// Local struct that implements [`ZnOperation`] to factor a polynomial over the integers,
/// by factoring it over `Fp`, lifting the factorization to `Z/p^eZ` and then extracting
/// integral factors. Used only in [`factor_integer_poly()`].
/// 
struct FactorizeMonicIntegerPolynomialUsingHenselLifting<'a, P, R>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        R: PolyRingStore,
        R::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Field + ZnRing
{
    FpX: R,
    ZZX: P,
    poly_mod_p: El<R>,
    poly: &'a El<P>,
    bound: El<BigIntRing>
}

impl<'a, P, R> ZnOperation<Vec<El<P>>> for FactorizeMonicIntegerPolynomialUsingHenselLifting<'a, P, R>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        R: PolyRingStore,
        R::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Field + ZnRing
{
    fn call<S: ZnRingStore>(self, Zpe: S) -> Vec<El<P>>
        where S::Type: ZnRing
    {
        let ZZ = Zpe.integer_ring();
        let bound = int_cast(self.bound, ZZ, &BigIntRing::RING);
        let mod_pe = Zpe.can_hom(ZZ).unwrap();
        let reduce = |x: El<<P::Type as RingExtension>::BaseRing>| mod_pe.map(int_cast(x, ZZ, self.ZZX.base_ring()));
        let ZpeX = DensePolyRing::new(&Zpe, "X");
        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&self.FpX, &self.poly_mod_p);
        debug_assert!(self.FpX.base_ring().is_one(&unit));
        debug_assert!(factorization.iter().all(|(_, e)| *e == 1));
        debug_assert!(factorization.iter().map(|(f, _)| self.FpX.degree(f).unwrap()).sum::<usize>() == self.ZZX.degree(&self.poly).unwrap());
        
        let lifted_factorization = hensel_lift_factorization(
            &ZpeX, 
            &self.FpX, 
            &self.FpX, 
            &ZpeX.from_terms(self.ZZX.terms(self.poly).map(|(c, i)| (reduce(self.ZZX.base_ring().clone_el(c)), i))), 
            &factorization.into_iter().map(|(f, _)| f).collect::<Vec<_>>()
        );

        let mut current = self.ZZX.clone_el(self.poly);
        let mut ungrouped_factors = (0..lifted_factorization.len()).collect::<Vec<_>>();
        let mut result = Vec::new();
        while !self.ZZX.is_unit(&current) {

            // Here we use the naive approach to group the factors in the p-adic numbers such that the product of each group
            // is integral - just try all combinations. It might be worth using LLL for this instead (as soon as LLL is implemented
            // in this library).
            let (factor, new_poly, factor_group) = crate::iters::basic_powerset(ungrouped_factors.iter().copied())
                // skip the empty set
                .skip(1)
                // compute the product of a subset of factors
                .map(|slice| (ZpeX.prod(slice.iter().copied().map(|i| ZpeX.clone_el(&lifted_factorization[i]))), slice))
                // if this is not bounded by `bound`, there is no chance it gives a factor over ZZ
                .filter(|(f, _)| ZpeX.terms(f).all(|(c, _)| ZZ.is_lt(&ZZ.abs(Zpe.smallest_lift(Zpe.clone_el(c))), &bound)))
                // lift it to ZZ
                .map(|(f, slice)| (self.ZZX.from_terms(ZpeX.terms(&f).map(|(c, i)| (int_cast(Zpe.smallest_lift(Zpe.clone_el(c)), self.ZZX.base_ring(), ZZ), i))), slice))
                // check if it is indeed a factor
                .filter_map(|(f, slice)| self.ZZX.checked_div(&current, &f).map(|quo| (f, quo, slice)))
                .next().unwrap();

            ungrouped_factors.retain(|j| !factor_group.contains(j));
            current = new_poly;
            result.push(factor);
        }
        assert!(self.ZZX.is_one(&current));
        return result;
    }
}

///
/// Computes the square-free part of a polynomial `f`, i.e. the greatest (w.r.t.
/// divisibility) polynomial `g | f` that is square-free.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::algorithms::poly_factor::cantor_zassenhaus::*;
/// let Fp = Zn::new(3).as_field().unwrap();
/// let FpX = DensePolyRing::new(Fp, "X");
/// // f = (X^2 + 1)^2 (X^3 + 2 X + 1)
/// let f = FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)]),
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)]),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (QQ.one(), 3)])
/// ].into_iter());
/// let squarefree_part = poly_squarefree_part(&FpX, f);
/// assert_el_eq!(&FpX, &FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)]),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (QQ.one(), 3)])
/// ].into_iter()), &squarefree_part);
/// ```
/// 
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

///
/// The given polynomial must be square-free, monic and should have integral
/// coefficients. In this case, all factors are also monic integral polynomials.
/// 
fn factor_integer_poly<'a, P>(ZZX: &'a P, f: &El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    let ZZ = StaticRing::<i64>::RING;
    let d = ZZX.degree(f).unwrap();
    assert!(ZZX.base_ring().is_one(ZZX.lc(f).unwrap()));

    // Cantor-Zassenhaus does not directly work for p = 2, so skip the first prime
    for p in erathostenes::enumerate_primes(&ZZ, &1000).into_iter().skip(1) {

        // check whether `f mod p` is also square-free, there are only finitely many primes
        // where this would not be the case
        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let mod_p = Fp.can_hom(&ZZ).unwrap();
        let reduce = |x: El<<P::Type as RingExtension>::BaseRing>| mod_p.map(int_cast(x, &ZZ, ZZX.base_ring()));
        let FpX = DensePolyRing::new(Fp, "X");
        let f_mod_p = FpX.from_terms(ZZX.terms(&f).map(|(c, i)| (reduce(ZZX.base_ring().clone_el(c)), i)));
        let mut squarefree_part = poly_squarefree_part(&FpX, FpX.clone_el(&f_mod_p));
        let lc_inv = Fp.div(&Fp.one(), FpX.lc(&squarefree_part).unwrap());
        FpX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);

        if FpX.eq_el(&squarefree_part, &f_mod_p) {

            // we found a prime such that f remains square-free mod p;
            // now we can use the factorization of `f mod p` to derive a factorization of f
            let ZZbig = BigIntRing::RING;

            // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen
            let poly_norm = int_bisect::root_floor(
                &ZZbig, 
                <_ as RingStore>::sum(&ZZbig, ZZX.terms(f)
                    .map(|(c, _)| ZZbig.pow(int_cast(ZZX.base_ring().clone_el(c), &ZZbig, ZZX.base_ring()), 2))
                ), 
                2
            );
            let bound = ZZbig.add(
                ZZbig.mul(poly_norm, binomial(d, d / 2)),
                ZZbig.mul(
                    int_cast(ZZX.base_ring().clone_el(ZZX.lc(f).unwrap()), ZZbig, ZZX.base_ring()), 
                    binomial(d, d / 2)
                )
            );
            let exponent = ZZbig.abs_log2_ceil(&bound).unwrap() / (ZZ.abs_log2_ceil(&(p + 1)).unwrap() - 1) + 1;
            let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZ), exponent);

            return choose_zn_impl(ZZbig, modulus, FactorizeMonicIntegerPolynomialUsingHenselLifting {
                poly: f, ZZX: ZZX, poly_mod_p: f_mod_p, FpX: FpX, bound
            });
        }
    }
    unreachable!()
}

impl<I> FactorPolyField for RationalFieldBase<I>
    where I: IntegerRingStore, I::Type: IntegerRing
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        assert!(!poly_ring.is_zero(poly));
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
        let mut result = Vec::new();
        let mut current = QQX.clone_el(poly);
        while !QQX.is_unit(&current) {
            let mut squarefree_part = poly_squarefree_part(&poly_ring, QQX.clone_el(&current));
            let lc_inv = QQ.div(&QQ.one(), &QQX.lc(&squarefree_part).unwrap());
            QQX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);
            current = QQX.checked_div(&current, &squarefree_part).unwrap();

            // we switch from `f(X)` to `c^d f(X/c)`, where c is the lcm of all denominators;
            // this will make the polynomial integral
            let mut den_lcm = ZZ.one();
            for (c, _) in QQX.terms(&squarefree_part) {
                den_lcm = algorithms::eea::signed_lcm(den_lcm, ZZ.clone_el(&c.1), ZZ);
            }
            let poly_d = QQX.degree(&squarefree_part).unwrap();
            let ZZX = DensePolyRing::new(ZZ, "X");
            let integral_poly = ZZX.from_terms(QQX.terms(&squarefree_part).map(|(c, i)|
                (ZZ.checked_div(&ZZ.mul_ref_fst(&c.0, ZZ.pow(ZZ.clone_el(&den_lcm), poly_d - i)), &c.1).unwrap(), i)
            ));
            for factor in factor_integer_poly(&ZZX, &integral_poly) {
                let factor_d = ZZX.degree(&factor).unwrap();
                let inclusion = QQ.inclusion();

                // go back from `c^d f(X/c)` to `f(X)` - as the degrees of the factors must add up to the total degree,
                // we can do this individually for each factor
                let factor_rational = QQX.from_terms(ZZX.terms(&factor).map(|(c, i)| 
                    (QQ.div(&inclusion.map_ref(c), &QQ.pow(inclusion.map_ref(&den_lcm), factor_d - i)), i)
                ));
                if let Some((i, _)) = result.iter().enumerate().filter(|(_, (f, _))| QQX.eq_el(f, &factor_rational)).next() {
                    result[i].1 += 1;
                } else {
                    result.push((factor_rational, 1));
                }
            }
        }
        let unit = QQ.clone_el(QQX.coefficient_at(&current, 0));
        assert_el_eq!(&QQX, &QQX.inclusion().map_ref(&unit), &current);
        return (result, unit);
    }
}

macro_rules! impl_factor_poly_number_fields {
    ($number_field_type:ty, $($lifetimes:lifetime),*) => {

        impl<$($lifetimes,)* I, V, A> FactorPolyField for $number_field_type
            where I: IntegerRingStore,
                I::Type: IntegerRing,
                RationalFieldBase<I>: FactorPolyField,
                V: VectorView<El<RationalField<I>>>,
                A: Allocator + Clone
        {
            fn factor_poly<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
                where P: PolyRingStore,
                    P::Type: PolyRing + EuclideanRing,
                    <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
            {
                number_field::factor_over_number_field(poly_ring, f)
            }
        }

        impl<$($lifetimes,)* I, V, A> ExtensionField for $number_field_type
            where I: IntegerRingStore,
                I::Type: IntegerRing,
                RationalFieldBase<I>: FactorPolyField,
                V: VectorView<El<RationalField<I>>>,
                A: Allocator + Clone
        {}
    };
}

// unfortunately, any blanket impl conflicts with the one for finite fields...
impl_factor_poly_number_fields!{ AsFieldBase<FreeAlgebraImpl<RationalField<I>, V, A>>, }
impl_factor_poly_number_fields!{ AsFieldBase<RingRef<'a, FreeAlgebraImplBase<RationalField<I>, V, A>>>, 'a }
impl_factor_poly_number_fields!{ AsFieldBase<FreeAlgebraImpl<RingRef<'a, RationalFieldBase<I>>, V, A>>, 'a }
impl_factor_poly_number_fields!{ AsFieldBase<RingRef<'a, FreeAlgebraImplBase<RingRef<'b, RationalFieldBase<I>>, V, A>>>, 'a, 'b }
impl_factor_poly_number_fields!{ AsFieldBase<&'a FreeAlgebraImpl<RationalField<I>, V, A>>, 'a }
impl_factor_poly_number_fields!{ AsFieldBase<FreeAlgebraImpl<&'a RationalField<I>, V, A>>, 'a }
impl_factor_poly_number_fields!{ AsFieldBase<&'a FreeAlgebraImpl<&'b RationalField<I>, V, A>>, 'a, 'b }

impl<R> ExtensionField for R
    where R: ?Sized + FiniteRing + Field + FreeAlgebra
{}

impl<R> FactorPolyField for R
    where R: ?Sized + FiniteRing + Field
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self> 
    {
        assert!(!poly_ring.is_zero(poly));

        let mut result = Vec::new();
        let mut unit = poly_ring.base_ring().one();
        let mut el = poly_ring.clone_el(poly);

        // we repeatedly remove the square-free part
        while !poly_ring.is_unit(&el) {
            let sqrfree_part = poly_squarefree_part(&poly_ring, poly_ring.clone_el(&el));
            assert!(!poly_ring.is_unit(&sqrfree_part));

            // factor the square-free part into distinct-degree factors
            for (d, factor_d) in cantor_zassenhaus::distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part)).into_iter().enumerate() {
                let mut stack = Vec::new();
                stack.push(factor_d);
                
                // and finally extract each individual factor
                while let Some(mut current) = stack.pop() {
                    // normalize current
                    let lc = poly_ring.lc(&current).unwrap();
                    poly_ring.base_ring().mul_assign_ref(&mut unit, lc);
                    let lc_inv = poly_ring.base_ring().div(&poly_ring.base_ring().one(), lc);
                    poly_ring.inclusion().mul_assign_map_ref(&mut current, &lc_inv);

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
                        let factor = cantor_zassenhaus::cantor_zassenhaus(&poly_ring, poly_ring.clone_el(&current), d);
                        stack.push(poly_ring.checked_div(&current, &factor).unwrap());
                        stack.push(factor);
                    }
                }
            }
            el = poly_ring.checked_div(&el, &sqrfree_part).unwrap();
        }
        poly_ring.base_ring().mul_assign_ref(&mut unit, poly_ring.coefficient_at(&el, 0));
        debug_assert!(poly_ring.base_ring().is_unit(&unit));
        return (result, unit);
    }
}

#[cfg(test)]
use crate::rings::zn::{zn_static, zn_64};
#[cfg(test)]
use test::Bencher;

#[cfg(test)]
fn normalize_poly<P>(poly_ring: P, poly: &mut El<P>)
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    let inv_lc = poly_ring.base_ring().div(&poly_ring.base_ring().one(), poly_ring.lc(poly).unwrap());
    poly_ring.inclusion().mul_assign_map_ref(poly, &inv_lc);
}

#[test]
fn test_factor_int_poly() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let f = poly_ring.from_terms([(2, 0), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (2, 1), (1, 2), (1, 4)].into_iter());
    let actual = factor_integer_poly(&poly_ring, &poly_ring.mul_ref(&f, &g));
    assert_eq!(2, actual.len());
    assert_el_eq!(&poly_ring, &f, &actual[0]);
    assert_el_eq!(&poly_ring, &g, &actual[1]);
}

#[test]
fn test_factor_rational_poly() {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(QQ, "X");
    let f = poly_ring.from_terms([(incl.map(2), 0), (incl.map(1), 3)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f), poly_ring.clone_el(&f), poly_ring.clone_el(&g)].into_iter()));
    assert_eq!(2, actual.len());
    assert_el_eq!(&poly_ring, &f, &actual[0].0);
    assert_eq!(2, actual[0].1);
    assert_el_eq!(&poly_ring, &g, &actual[1].0);
    assert_eq!(1, actual[1].1);
    assert_el_eq!(&QQ, &QQ.one(), &unit);
}

#[test]
fn test_factor_nonmonic_poly() {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(QQ, "X");
    let f = poly_ring.from_terms([(QQ.div(&incl.map(3), &incl.map(5)), 0), (incl.map(1), 4)].into_iter());
    let g = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4)].into_iter());
    let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f), poly_ring.clone_el(&f), poly_ring.clone_el(&g), poly_ring.int_hom().map(100)].into_iter()));
    assert_eq!(2, actual.len());

    assert_el_eq!(&poly_ring, &g, &actual[0].0);
    assert_eq!(1, actual[0].1);
    assert_el_eq!(&poly_ring, &f, &actual[1].0);
    assert_eq!(2, actual[1].1);
    assert_el_eq!(&QQ, &incl.map(100), &unit);
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
    assert_el_eq!(&Fp, &Fp.one(), &unit);
    
    assert_eq!(2, factorization[0].1);
    assert_el_eq!(&poly_ring, &g, &factorization[0].0);
    assert_eq!(3, factorization[1].1);
    assert_el_eq!(&poly_ring, &h, &factorization[1].0);
    assert_eq!(1, factorization[2].1);
    assert_el_eq!(&poly_ring, &f, &factorization[2].0);
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
    let mut squarefree_part = poly_squarefree_part(&ring, a);
    normalize_poly(&ring, &mut squarefree_part);
    assert_el_eq!(&ring, &b, &squarefree_part);
}

#[test]
fn test_poly_squarefree_part_multiplicity_p() {
    let ring = DensePolyRing::new(zn_64::Zn::new(5).as_field().ok().unwrap(), "X");
    let f = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 10)].into_iter());
    let g = ring.from_terms([(ring.base_ring().int_hom().map(3), 0), (ring.base_ring().int_hom().map(1), 2)].into_iter());
    let mut actual = poly_squarefree_part(&ring, f);
    normalize_poly(&ring, &mut actual);
    assert_el_eq!(&ring, &g, &actual);
}

#[bench]
fn bench_factor_rational_poly(bencher: &mut Bencher) {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(QQ, "X");
    let f1 = poly_ring.checked_div(&poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 2), (incl.map(1), 4), (incl.map(3), 8)].into_iter()), &poly_ring.int_hom().map(3)).unwrap();
    let f2 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4), (incl.map(1), 5), (incl.map(1), 10)].into_iter());
    let f3 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 1), (incl.map(-2), 5), (incl.map(1), 17)].into_iter());
    bencher.iter(|| {
        let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f1), poly_ring.clone_el(&f1), poly_ring.clone_el(&f2), poly_ring.clone_el(&f3), poly_ring.int_hom().map(9)].into_iter()));
        assert_eq!(3, actual.len());
        assert_el_eq!(&QQ, &QQ.int_hom().map(9), &unit);
        for (f, e) in actual.iter() {
            if poly_ring.eq_el(f, &f1) {
                assert_el_eq!(&poly_ring, &f1, f);
                assert_eq!(2, *e);
            } else if poly_ring.eq_el(f, &f2) {
                assert_el_eq!(&poly_ring, &f2, f);
                assert_eq!(1, *e);
           } else if poly_ring.eq_el(f, &f3) {
               assert_el_eq!(&poly_ring, &f3, f);
               assert_eq!(1, *e);
            } else {
                panic!("Factorization returned wrong factor {} of ({})^2 * {} * {}", poly_ring.format(f), poly_ring.format(&f1), poly_ring.format(&f2), poly_ring.format(&f3));
            }
        }
    });
}
