use crate::algorithms::int_bisect;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::algorithms::erathostenes::enumerate_primes;
use crate::ring::*;
use crate::integer::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::primitive_int::*;
use crate::divisibility::*;
use crate::field::*;
use crate::rings::zn::*;
use crate::rings::zn::zn_64::*;
use crate::ordered::*;
use crate::pid::EuclideanRing;

use crate::algorithms::hensel::hensel_lift_factorization;
use super::poly_squarefree_part;
use super::FactorPolyField;

///
/// The given polynomial must be square-free, monic and should have integral
/// coefficients. In this case, all factors are also monic integral polynomials.
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_integer_poly<'a, P>(ZZX: &'a P, f: &El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        ZnBase: CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let d = ZZX.degree(f).unwrap();
    assert!(ZZX.base_ring().is_one(ZZX.lc(f).unwrap()));

    // Cantor-Zassenhaus does not directly work for p = 2, so skip the first prime
    for p in enumerate_primes(&StaticRing::<i64>::RING, &1000).into_iter().skip(1) {

        // check whether `f mod p` is also square-free, there are only finitely many primes
        // where this would not be the case
        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let mod_p = Fp.can_hom(ZZX.base_ring()).unwrap();
        let FpX = DensePolyRing::new(Fp, "X");
        let f_mod_p = FpX.from_terms(ZZX.terms(&f).map(|(c, i)| (mod_p.map(ZZX.base_ring().clone_el(c)), i)));
        let mut squarefree_part = poly_squarefree_part(&FpX, FpX.clone_el(&f_mod_p));
        let lc_inv = Fp.div(&Fp.one(), FpX.lc(&squarefree_part).unwrap());
        FpX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);

        if FpX.eq_el(&squarefree_part, &f_mod_p) {

            // we found a prime such that f remains square-free mod p;
            // now we can use the factorization of `f mod p` to derive a factorization of f
            let ZZbig = BigIntRing::RING;
            let ZZ = StaticRing::<i64>::RING;

            // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen
            let poly_norm = int_bisect::root_floor(
                &ZZbig, 
                <_ as RingStore>::sum(&ZZbig, ZZX.terms(f)
                    .map(|(c, _)| ZZbig.pow(int_cast(ZZX.base_ring().clone_el(c), &ZZbig, ZZX.base_ring()), 2))
                ), 
                2
            );
            let bound = ZZbig.add(
                ZZbig.mul(poly_norm, binomial(int_cast(d as i64, ZZbig, ZZ), &int_cast(d as i64 / 2, ZZbig, ZZ), ZZbig)),
                ZZbig.mul(
                    int_cast(ZZX.base_ring().clone_el(ZZX.lc(f).unwrap()), ZZbig, ZZX.base_ring()), 
                    binomial(int_cast(d as i64, ZZbig, ZZ), &int_cast(d as i64 / 2, ZZbig, ZZ), ZZbig)
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
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField + ZnRing
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
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField + ZnRing
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

#[test]
fn test_factor_int_poly() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let f = poly_ring.from_terms([(2, 0), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (2, 1), (1, 2), (1, 4)].into_iter());
    let actual = factor_integer_poly(&poly_ring, &poly_ring.mul_ref(&f, &g));
    assert_eq!(2, actual.len());
    assert_el_eq!(poly_ring, f, actual[0]);
    assert_el_eq!(poly_ring, g, actual[1]);
}
