use std::cmp::min;

use crate::divisibility::*;
use crate::field::{Field, FieldStore};
use crate::homomorphism::Homomorphism;
use crate::integer::{int_cast, BigIntRing, IntegerRing, IntegerRingStore};
use crate::ordered::*;
use crate::pid::EuclideanRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::finite::FiniteRing;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::algorithms::{self, int_bisect};
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::{choose_zn_impl, ZnOperation, ZnRing, ZnRingStore};

use super::cantor_zassenhaus::poly_squarefree_part;
use super::erathostenes;
use super::hensel::hensel_lift_factorization;

pub fn binomial(n: usize, mut k: usize) -> usize {
    if k > n {
        0
    } else {
        k = min(k, n - k);
        ((n - k + 1)..=n).product::<usize>() / (1..=k).product::<usize>()
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

struct FactorizeMonicIntegerPolynomialUsingHenselLifting<'a, P, R>
    where P: PolyRingStore,
        P::Type: PolyRing,
        R: PolyRingStore,
        R::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Field + ZnRing
{
    prime_poly_ring: R,
    poly_ring: P,
    poly_mod_p: El<R>,
    poly: &'a El<P>,
    bound: El<BigIntRing>
}

impl<'a, P, R> ZnOperation<Vec<El<P>>> for FactorizeMonicIntegerPolynomialUsingHenselLifting<'a, P, R>
    where P: PolyRingStore,
        P::Type: PolyRing,
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
        let reduce = |x: El<<P::Type as RingExtension>::BaseRing>| mod_pe.map(int_cast(x, ZZ, self.poly_ring.base_ring()));
        let prime_power_poly_ring = DensePolyRing::new(&Zpe, "X");
        let factorization = <_ as FactorPolyField>::factor_poly(&self.prime_poly_ring, &self.poly_mod_p).0.into_iter().inspect(|(_, e)| assert!(*e == 1)).map(|(f, _)| f).collect::<Vec<_>>();
        let mut poly = prime_power_poly_ring.from_terms(self.poly_ring.terms(&self.poly).map(|(c, i)| (reduce(self.poly_ring.base_ring().clone_el(c)), i)));
        let lifted_factorization = hensel_lift_factorization(&prime_power_poly_ring, &self.prime_poly_ring, &self.prime_poly_ring, &poly, &factorization);

        let mut ungrouped_factors = (0..lifted_factorization.len()).collect::<Vec<_>>();
        let mut result = Vec::new();
        while !prime_power_poly_ring.is_unit(&poly) {
            // Here we use the naive approach to group the factors in the p-adic numbers such that the product of each group
            // is integral - just try all combinations. It might be worth using LLL for this instead (as soon as LLL is implemented
            // in this library).
            let (factor, factor_group) = crate::iters::basic_powerset(ungrouped_factors.iter().copied())
                .skip(1)
                .map(|slice| (prime_power_poly_ring.prod(slice.iter().copied().map(|i| prime_power_poly_ring.clone_el(&lifted_factorization[i]))), slice))
                .filter(|(f, _)| prime_power_poly_ring.terms(f).all(|(c, _)| ZZ.is_lt(&ZZ.abs(Zpe.smallest_lift(Zpe.clone_el(c))), &bound)))
                .next().unwrap();
            ungrouped_factors.retain(|j| !factor_group.contains(j));
            poly = prime_power_poly_ring.checked_div(&poly, &factor).unwrap();
            result.push(self.poly_ring.from_terms(prime_power_poly_ring.terms(&factor).map(|(c, i)| (int_cast(Zpe.smallest_lift(Zpe.clone_el(c)), self.poly_ring.base_ring(), ZZ), i))));
        }
        assert!(prime_power_poly_ring.is_one(&poly));
        return result;
    }
}

///
/// The given polynomial must be square-free, monic and should have integral
/// coefficients. In this case, all factors are also monic integral polynomials.
/// 
fn factor_monic_int_poly<'a, P>(poly_ring: &'a P, poly: &El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    let d = poly_ring.degree(poly).unwrap();
    let poly_lc = poly_ring.lc(poly).unwrap();
    assert!(poly_ring.base_ring().is_one(poly_lc));
    let ZZ = StaticRing::<i64>::RING;

    // Cantor-Zassenhaus does not directly work for p = 2, so skip the first prime
    for p in erathostenes::enumerate_primes(&ZZ, &1000).into_iter().skip(1) {

        // check whether f mod p is square-free
        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let mod_p = Fp.can_hom(&ZZ).unwrap();
        let reduce = |x: El<<P::Type as RingExtension>::BaseRing>| mod_p.map(int_cast(x, &ZZ, poly_ring.base_ring()));
        let prime_poly_ring = DensePolyRing::new(Fp, "X");
        let poly_mod_p = prime_poly_ring.from_terms(poly_ring.terms(&poly).map(|(c, i)| (reduce(poly_ring.base_ring().clone_el(c)), i)));
        let mut squarefree_part = poly_squarefree_part(&prime_poly_ring, prime_poly_ring.clone_el(&poly_mod_p));
        let inv_lc = Fp.div(&Fp.one(), prime_poly_ring.lc(&squarefree_part).unwrap());
        prime_poly_ring.inclusion().mul_assign_map(&mut squarefree_part, inv_lc);

        if prime_poly_ring.eq_el(&squarefree_part, &poly_mod_p) {

            // we found a prime such that f remains square-free mod p;
            // now we can use the factorization of f mod p to derive a factorization of f
            let ZZbig = BigIntRing::RING;

            // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen
            let poly_norm = int_bisect::root_floor(&ZZbig, ZZbig.sum(poly_ring.terms(poly)
                .map(|(c, _)| ZZbig.pow(int_cast(poly_ring.base_ring().clone_el(c), &ZZbig, poly_ring.base_ring()), 2))
            ), 2);
            let bound = ZZbig.add(
                ZZbig.mul(poly_norm, ZZbig.coerce(&ZZ, binomial(d, d / 2) as i64)),
                ZZbig.mul(
                    int_cast(poly_ring.base_ring().clone_el(poly_ring.lc(poly).unwrap()), ZZbig, poly_ring.base_ring()), 
                    ZZbig.coerce(&ZZ, binomial(d, d / 2) as i64)
                )
            );
            let exponent = ZZbig.abs_log2_ceil(&bound).unwrap() / (ZZ.abs_log2_ceil(&(p + 1)).unwrap() - 1) + 1;
            let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZ), exponent);

            return choose_zn_impl(ZZbig, modulus, FactorizeMonicIntegerPolynomialUsingHenselLifting {
                poly, poly_ring, poly_mod_p, prime_poly_ring, bound
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
        let QQ = poly_ring.base_ring();
        let ZZ = QQ.base_ring();
        let mut result = Vec::new();
        let mut current = poly_ring.clone_el(poly);
        while !poly_ring.is_unit(&current) {
            let mut squarefree_part = algorithms::cantor_zassenhaus::poly_squarefree_part(&poly_ring, poly_ring.clone_el(&current));
            current = poly_ring.checked_div(&current, &squarefree_part).unwrap();

            let lc = QQ.clone_el(poly_ring.lc(&squarefree_part).unwrap());
            let inclusion = poly_ring.inclusion();
            inclusion.mul_assign_map(&mut squarefree_part, QQ.checked_div(&QQ.one(), &lc).unwrap());
            inclusion.mul_assign_map(&mut current, lc);

            // we switch from `f(X)` to `c^d f(X/c)`, where c is the lcm of all denominators;
            // this will make the polynomial integral
            let mut den_lcm = ZZ.one();
            for (c, _) in poly_ring.terms(&squarefree_part) {
                den_lcm = algorithms::eea::signed_lcm(den_lcm, ZZ.clone_el(&c.1), ZZ);
            }
            let poly_d = poly_ring.degree(&squarefree_part).unwrap();
            let int_poly_ring = DensePolyRing::new(ZZ, "X");
            let integral_poly = int_poly_ring.from_terms(poly_ring.terms(&squarefree_part).map(|(c, i)|
                (ZZ.checked_div(&ZZ.mul_ref_fst(&c.0, ZZ.pow(ZZ.clone_el(&den_lcm), poly_d - i)), &c.1).unwrap(), i)
            ));
            for factor in factor_monic_int_poly(&int_poly_ring, &integral_poly) {
                let factor_d = int_poly_ring.degree(&factor).unwrap();
                let inclusion = QQ.inclusion();

                // go back from `c^d f(X/c)` to `f(X)` - as the degrees of the factors must add up to the total degree,
                // we can do this individually for each factor
                let factor_rational = poly_ring.from_terms(int_poly_ring.terms(&factor).map(|(c, i)| 
                    (QQ.div(&inclusion.map_ref(c), &QQ.pow(inclusion.map_ref(&den_lcm), factor_d - i)), i)
                ));
                if let Some((i, _)) = result.iter().enumerate().filter(|(_, (f, _))| poly_ring.eq_el(f, &factor_rational)).next() {
                    result[i].1 += 1;
                } else {
                    result.push((factor_rational, 1));
                }
            }
        }
        let unit = QQ.clone_el(poly_ring.coefficient_at(&current, 0));
        assert_el_eq!(&poly_ring, &poly_ring.inclusion().map_ref(&unit), &current);
        return (result, unit);
    }
}

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
            for (d, factor_d) in algorithms::cantor_zassenhaus::distinct_degree_factorization(&poly_ring, poly_ring.clone_el(&sqrfree_part)).into_iter().enumerate() {
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
                        let factor = algorithms::cantor_zassenhaus::cantor_zassenhaus(&poly_ring, poly_ring.clone_el(&current), d);
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
use crate::rings::rational::*;
#[cfg(test)]
use crate::rings::zn::zn_static;

#[test]
fn test_factor_int_poly() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let f = poly_ring.from_terms([(2, 0), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (2, 1), (1, 2), (1, 4)].into_iter());
    let actual = factor_monic_int_poly(&poly_ring, &poly_ring.mul_ref(&f, &g));
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