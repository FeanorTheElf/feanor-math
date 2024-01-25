use std::cmp::min;

use crate::divisibility::*;
use crate::field::{Field, FieldStore};
use crate::homomorphism::Homomorphism;
use crate::integer::{int_cast, BigIntRing, IntegerRing, IntegerRingStore};
use crate::ordered::*;
use crate::pid::EuclideanRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::algorithms::{self, int_bisect};
use crate::rings::rational::RationalField;
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::{choose_zn_impl, ZnOperation, ZnRing, ZnRingStore};

use super::cantor_zassenhaus::{factor_complete, poly_squarefree_part};
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

struct ComputeFactorizationUsingHenselLifting<'a, P, R>
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

impl<'a, P, R> ZnOperation<Vec<El<P>>> for ComputeFactorizationUsingHenselLifting<'a, P, R>
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
        let factorization = factor_complete(&self.prime_poly_ring, self.poly_mod_p).into_iter().inspect(|(_, e)| assert!(*e == 1)).map(|(f, _)| f).collect::<Vec<_>>();
        let mut poly = prime_power_poly_ring.from_terms(self.poly_ring.terms(&self.poly).map(|(c, i)| (reduce(self.poly_ring.base_ring().clone_el(c)), i)));
        let lifted_factorization = hensel_lift_factorization(&prime_power_poly_ring, &self.prime_poly_ring, &self.prime_poly_ring, &poly, &factorization);
        println!();
        for f in &lifted_factorization {
            prime_power_poly_ring.println(f);
        }

        let mut ungrouped_factors = (0..lifted_factorization.len()).collect::<Vec<_>>();
        let mut result = Vec::new();
        while !prime_power_poly_ring.is_unit(&poly) {
            println!("{:?}", crate::iters::basic_powerset(ungrouped_factors.iter().copied()).next().unwrap());
            let (factor, factor_group) = crate::iters::basic_powerset(ungrouped_factors.iter().copied())
                .skip(1)
                .map(|slice| (prime_power_poly_ring.prod(slice.iter().copied().map(|i| prime_power_poly_ring.clone_el(&lifted_factorization[i]))), slice))
                .inspect(|(_, slice)| println!("{:?}", slice))
                .inspect(|(f, _)| prime_power_poly_ring.println(f))
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

fn factor_squarefree_int_poly<'a, P>(poly_ring: &'a P, poly: &El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    let d = poly_ring.degree(poly).unwrap();
    let poly_lc = poly_ring.base_ring().clone_el(poly_ring.lc(poly).unwrap());
    let ZZ = StaticRing::<i64>::RING;

    // Cantor-Zassenhaus does not directly work for p = 2, so skip the first prime
    for p in erathostenes::enumerate_primes(&ZZ, &1000).into_iter().skip(1) {

        if poly_ring.base_ring().checked_div(&poly_lc, &int_cast(p, poly_ring.base_ring(), &ZZ)).is_some() {
            continue;
        }

        // check whether f mod p is square-free
        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let mod_p = Fp.can_hom(&ZZ).unwrap();
        let reduce = |x: El<<P::Type as RingExtension>::BaseRing>| mod_p.map(int_cast(x, &ZZ, poly_ring.base_ring()));
        let prime_poly_ring = DensePolyRing::new(Fp, "X");
        let poly_mod_p = prime_poly_ring.from_terms(poly_ring.terms(&poly).map(|(c, i)| (reduce(poly_ring.base_ring().clone_el(c)), i)));
        let squarefree_part = poly_squarefree_part(&prime_poly_ring, prime_poly_ring.clone_el(&poly_mod_p));

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
                ZZbig.mul(int_cast(poly_ring.base_ring().clone_el(poly_ring.lc(poly).unwrap()), ZZbig, poly_ring.base_ring()), ZZbig.coerce(&ZZ, binomial(d, d / 2) as i64))
            );
            let exponent = ZZbig.abs_log2_ceil(&bound).unwrap() / (ZZ.abs_log2_ceil(&(p + 1)).unwrap() - 1) + 1;
            let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZ), exponent);

            return choose_zn_impl(ZZbig, modulus, ComputeFactorizationUsingHenselLifting {
                poly, poly_ring, poly_mod_p, prime_poly_ring, bound
            });
        }
    }
    unreachable!()
}

pub fn factor_rational_poly<P, I>(poly_ring: &P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing + RingExtension<BaseRing = RationalField<I>>,
        I: IntegerRingStore,
        I::Type: IntegerRing
{
    let mut result = Vec::new();
    let mut current = poly_ring.clone_el(poly);
    let mut unit = poly_ring.base_ring().one();
    while !poly_ring.is_unit(&current) {
        let squarefree_part = algorithms::cantor_zassenhaus::poly_squarefree_part(poly_ring, poly_ring.clone_el(&current));
        current = poly_ring.checked_div(&current, &squarefree_part).unwrap();

        let mut nom_gcd = poly_ring.base_ring().base_ring().one();
        let mut den_lcm = poly_ring.base_ring().base_ring().one();
        for ((nom, den), _) in poly_ring.terms(&squarefree_part) {
            nom_gcd = algorithms::eea::signed_gcd(nom_gcd, poly_ring.base_ring().base_ring().clone_el(nom), poly_ring.base_ring().base_ring());
            den_lcm = algorithms::eea::signed_lcm(den_lcm, poly_ring.base_ring().base_ring().clone_el(den), poly_ring.base_ring().base_ring());
        }
        let inclusion = poly_ring.base_ring().inclusion();
        let factor = poly_ring.base_ring().div(&inclusion.map(den_lcm), &inclusion.map(nom_gcd));
        unit = poly_ring.base_ring().div(&unit, &factor);

        let int_poly_ring = DensePolyRing::new(poly_ring.base_ring().base_ring(), "X");
        let squarefree_part_int = int_poly_ring.from_terms(poly_ring.terms(&squarefree_part).map(|(c, i)| {
            let (nom, den) = poly_ring.base_ring().mul_ref(c, &factor);
            assert!(poly_ring.base_ring().base_ring().is_one(&den));
            (nom, i)
        }));
        for factor in factor_squarefree_int_poly(&int_poly_ring, &squarefree_part_int) {
            let factor_rational = poly_ring.from_terms(int_poly_ring.terms(&factor).map(|(c, i)| (inclusion.map_ref(c), i)));
            if let Some((i, _)) = result.iter().enumerate().filter(|(_, (f, _))| poly_ring.eq_el(f, &factor_rational)).next() {
                result[i].1 += 1;
            } else {
                result.push((factor_rational, 1));
            }
        }
    }
    return result;
}

#[test]
fn test_squarefree_factor_int_poly() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let f = poly_ring.from_terms([(2, 0), (1, 3)].into_iter());
    let g = poly_ring.from_terms([(1, 0), (2, 1), (1, 2), (1, 4)].into_iter());
    let actual = factor_squarefree_int_poly(&poly_ring, &poly_ring.mul_ref(&f, &g));
    assert_eq!(2, actual.len());
    assert_el_eq!(&poly_ring, &f, &actual[0]);
    assert_el_eq!(&poly_ring, &g, &actual[1]);
}