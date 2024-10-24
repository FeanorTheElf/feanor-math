use std::cmp::max;
use std::alloc::Global;

use crate::algorithms::interpolate::interpolate;
use crate::algorithms::poly_factor::make_primitive;
use crate::algorithms::int_bisect;
use crate::algorithms::poly_factor::poly_squarefree_part_finite_field;
use crate::compute_locally::InterpolationBaseRing;
use crate::compute_locally::ToExtRingMap;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::algorithms::erathostenes::enumerate_primes;
use crate::ring::*;
use crate::integer::*;
use crate::seq::VectorView;
use crate::rings::poly::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::primitive_int::*;
use crate::specialization::*;
use crate::divisibility::*;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::*;
use crate::rings::zn::zn_64::*;
use crate::ordered::*;
use crate::algorithms::eea::signed_lcm;
use crate::pid::EuclideanRing;
use crate::algorithms::hensel::hensel_lift_factorization;

use super::FactorPolyField;

///
/// Computes a bound on the largest coefficient of any factor of `c f`.
/// 
#[stability::unstable(feature = "enable")]
pub fn max_coeff_of_factor<P>(ZZX: P, f: &El<P>, c: &El<<P::Type as RingExtension>::BaseRing>) -> El<BigIntRing>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    let ZZbig = BigIntRing::RING;
    let ZZ = StaticRing::<i64>::RING;

    if ZZX.is_zero(f) {
        return ZZbig.zero();
    }

    let d = ZZX.degree(f).unwrap();

    // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen,
    // or equivalently Ex. 20 from Chapter 4.6.2 in Knuth's Art
    let c = int_cast(ZZX.base_ring().clone_el(c), ZZbig, ZZX.base_ring());
    let poly_norm = ZZbig.mul_ref_snd(ZZbig.add(int_bisect::root_floor(
        &ZZbig, 
        <_ as RingStore>::sum(&ZZbig, ZZX.terms(f)
            .map(|(c, _)| ZZbig.pow(int_cast(ZZX.base_ring().clone_el(c), &ZZbig, ZZX.base_ring()), 2))
        ), 
        2
    ), ZZbig.one()), &c);
    let bound = ZZbig.add(
        ZZbig.mul(poly_norm, binomial(int_cast(d as i64, ZZbig, ZZ), &int_cast(d as i64 / 2, ZZbig, ZZ), ZZbig)),
        ZZbig.mul(
            ZZbig.mul_ref_snd(int_cast(ZZX.base_ring().clone_el(ZZX.lc(f).unwrap()), ZZbig, ZZX.base_ring()), &c), 
            binomial(int_cast(d as i64, ZZbig, ZZ), &int_cast(d as i64 / 2, ZZbig, ZZ), ZZbig)
        )
    );
    return bound;
}

fn factor_squarefree_primitive_integer_poly_local<P>(ZZX: P, f: &El<P>) -> Vec<El<P>>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        ZnBase: CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    // we have to switch to `c^(d - 1) f(X/c)` to make `f` monic
    let ZZ = ZZX.base_ring();
    let scale = ZZ.clone_el(ZZX.lc(&f).unwrap());
    let deg_f = ZZX.degree(f).unwrap();
    let f = ZZX.from_terms(ZZX.terms(f).map(|(c, i)| if i == deg_f {
        (ZZ.one(), i)
    } else {
        (ZZ.mul_ref_fst(c, ZZ.pow(ZZ.clone_el(&scale), deg_f - i - 1)), i)
    }));

    let undo_scale = |g: &El<P>| make_primitive(ZZX, ZZX.from_terms(ZZX.terms(g).map(
        |(c, i)| (ZZ.mul_ref_fst(c, ZZ.pow(ZZ.clone_el(&scale), i)), i))
    ));

    // very small primes have a lower probability of working
    for p in enumerate_primes(&StaticRing::<i64>::RING, &1000).into_iter().skip(100) {

        // check whether `f mod p` is also square-free, there are only finitely many primes
        // where this would not be the case
        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let mod_p = Fp.can_hom(ZZX.base_ring()).unwrap();
        let FpX = DensePolyRing::new(Fp, "X");
        let f_mod_p = FpX.from_terms(ZZX.terms(&f).map(|(c, i)| (mod_p.map(ZZX.base_ring().clone_el(c)), i)));
        let squarefree_part = poly_squarefree_part_finite_field(&FpX, &f_mod_p);

        if FpX.eq_el(&squarefree_part, &f_mod_p) {

            // we found a prime such that f remains square-free mod p;
            // now we can use the factorization of `f mod p` to derive a factorization of f

            let ZZbig = BigIntRing::RING;
            let ZZ = StaticRing::<i64>::RING;
            let bound = max_coeff_of_factor(ZZX, &f, &ZZX.base_ring().one());
            let exponent = max(2, ZZbig.abs_log2_ceil(&bound).unwrap() / ZZ.abs_log2_floor(&p).unwrap() + 1);
            let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZ), exponent);
            
            return choose_zn_impl(ZZbig, modulus, FactorizeMonicIntegerPolynomialUsingHenselLifting {
                poly: &f, ZZX: ZZX, poly_mod_p: f_mod_p, FpX: FpX, bound
            }).into_iter().map(|g| undo_scale(&g)).collect();
        }
    }
    unreachable!()
}

///
/// Factors the given polynomial over the integers.
/// 
/// Its factors are returned as primitive polynomials, thus their
/// product is `f` only up to multiplication by a nonzero integer. 
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_factor_integer<P>(ZZX: P, f: El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + InterpolationBaseRing,
        ZnBase: CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    assert!(!ZZX.is_zero(&f));
    let mut current = make_primitive(ZZX, f);
    let mut result = Vec::new();

    while ZZX.degree(&current).unwrap() > 0 {
        let squarefree_part = poly_squarefree_part_integer(ZZX, &current);
        current = ZZX.checked_div(&current, &squarefree_part).unwrap();

        if result.len() == 0 {
            result.extend(factor_squarefree_primitive_integer_poly_local(ZZX, &squarefree_part).into_iter().map(|factor| (factor, 1)));
        } else {
            for factor in factor_squarefree_primitive_integer_poly_local(ZZX, &squarefree_part) {
                let idx = result.iter().enumerate().filter(|(_, f)| ZZX.eq_el(&f.0, &factor)).next().unwrap().0;
                result[idx].1 += 1;
            }
        }
    }
    return result;
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

impl<'a, P, R> ZnOperation for FactorizeMonicIntegerPolynomialUsingHenselLifting<'a, P, R>
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        R: PolyRingStore,
        R::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField + ZnRing
{
    type Output<'b> = Vec<El<P>>
        where Self: 'b;

    fn call<'b, S>(self, Zpe: S) -> Vec<El<P>>
        where S: 'b + ZnRingStore, S::Type: ZnRing
    {
        assert!(self.ZZX.base_ring().is_one(self.ZZX.lc(self.poly).unwrap()));
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

#[stability::unstable(feature = "enable")]
pub fn poly_factor_rational<'a, P, I>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing + InterpolationBaseRing,
        ZnBase: CanHomFrom<I::Type>
{
    assert!(!poly_ring.is_zero(poly));
    let QQX = &poly_ring;
    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();

    let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
    
    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
    let factorization = poly_factor_integer(&ZZX, f);
    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());

    return (
        factorization.into_iter().map(|(f, e)| (QQX.normalize(ZZX_to_QQX.map(f)), e)).collect(),
        QQ.clone_el(QQX.lc(poly).unwrap())
    );
}

fn poly_root<P>(poly_ring: P, f: &El<P>, k: usize) -> Option<El<P>>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + InterpolationBaseRing
{
    assert!((poly_ring.degree(f).unwrap() % k) == 0);
    let result_degree = poly_ring.degree(f).unwrap() / k;
    let (ext_ring_hom, points) = ToExtRingMap::for_interpolation(poly_ring.base_ring().get_ring(), result_degree + 1);
    let mut values = Vec::new();
    let mut points_in_base_ring = Vec::new();
    for x in &points {
        points_in_base_ring.push(ext_ring_hom.as_base_ring_el(ext_ring_hom.codomain().clone_el(x)));
        let y = poly_ring.evaluate(f, Option::unwrap(points_in_base_ring.last()), &poly_ring.base_ring().identity());
        let value = int_bisect::root_floor(poly_ring.base_ring(), poly_ring.base_ring().clone_el(&y), k);
        if !poly_ring.base_ring().eq_el(&poly_ring.base_ring().pow(poly_ring.base_ring().clone_el(&value), k), &y) {
            return None;
        }
        values.push(value);
    }
    let result = interpolate(&poly_ring, points_in_base_ring.as_ring_el_fn(poly_ring.base_ring()), values.as_ring_el_fn(poly_ring.base_ring()), Global).ok()?;
    if poly_ring.eq_el(f, &poly_ring.pow(poly_ring.clone_el(&result), k)) {
        return Some(result);
    } else {
        return None;
    }
}

struct IntegerPolyPowerDecompositionUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + InterpolationBaseRing,
        Q: RingStore,
        Q::Type: PolyRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    ZZX: P,
    FpX: Q,
    f: &'a El<P>
}

impl<'a, P, Q> ZnOperation for IntegerPolyPowerDecompositionUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + InterpolationBaseRing,
        Q: RingStore,
        Q::Type: PolyRing + EuclideanRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + FactorPolyField + PerfectField + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> + FiniteRingSpecializable
{
    type Output<'b> = Option<Vec<(El<P>, usize)>>
        where Self: 'b;

    fn call<'b, R>(self, Zpe: R) -> Option<Vec<(El<P>, usize)>>
        where R: 'b + RingStore, R::Type: ZnRing
    {
        let Fp = self.FpX.base_ring();
        let ZZ = self.ZZX.base_ring();
        let ZZ_to_Fp = Fp.can_hom(ZZ).unwrap();
        let ZZX_to_FpX = self.FpX.lifted_hom(&self.ZZX, &ZZ_to_Fp);

        let f_mod_p = self.FpX.normalize(ZZX_to_FpX.map_ref(&self.f));
        let power_decomposition = poly_power_decomposition_global(&self.FpX, &f_mod_p);
        let factors = power_decomposition.iter().map(|(f, k)| self.FpX.pow(self.FpX.clone_el(f), *k)).collect::<Vec<_>>();

        let ZpeX = DensePolyRing::new(&Zpe, "X");
        let ZZ_to_Zpe = Zpe.can_hom(Zpe.integer_ring()).unwrap();
        let reduce_pe = |c: &El<<P::Type as RingExtension>::BaseRing>| ZZ_to_Zpe.map(int_cast(ZZ.clone_el(c), Zpe.integer_ring(), ZZ));
        let lc_f = reduce_pe(self.ZZX.lc(self.f).unwrap());
        let lc_f_inv = Zpe.invert(&lc_f).unwrap();
        let f_mod_pe = ZpeX.inclusion().mul_map(ZpeX.from_terms(self.ZZX.terms(&self.f).map(|(c, i)| (reduce_pe(c), i))), lc_f_inv);

        let lifted_factors = hensel_lift_factorization(
            &ZpeX,
            &self.FpX,
            &self.FpX,
            &f_mod_pe,
            &factors
        );

        let mut result = Vec::new();
        for (i, mut factor) in lifted_factors.into_iter().enumerate() {
            let k = power_decomposition[i].1;
            ZpeX.inclusion().mul_assign_ref_map(&mut factor, &lc_f);
            let factor_over_Z = self.ZZX.from_terms(ZpeX.terms(&factor).map(|(c, i)| (int_cast(Zpe.smallest_lift(Zpe.clone_el(c)), ZZ, Zpe.integer_ring()), i)));
        
            let factor_over_Z = make_primitive(self.ZZX, factor_over_Z);
            if self.ZZX.checked_div(self.f, &factor_over_Z).is_none() {
                return None;
            }
            if let Some(root_of_factor) = poly_root(self.ZZX, &factor_over_Z, k) {
                result.push((root_of_factor, k));
            } else {
                return None;
            }
        }
        
        return Some(result);
    }
}

///
/// Computes the square-free part of a polynomial `f` over the integers, 
/// i.e. the greatest (w.r.t. divisibility) polynomial `g | f` that is square-free, 
/// up to multiplication by nonzero integers.
/// 
/// The returned polynomial is always primitive, and with this restriction, it
/// is unique. This function does the computations locally, i.e. modulo suitable
/// primes. This is usually much faster than working over `Z` or `Q`.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_integer<P>(ZZX: P, f: &El<P>) -> El<P>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + InterpolationBaseRing,
        ZnBase: CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let f = make_primitive(ZZX, ZZX.clone_el(f));

    // very small primes have a lower probability of working (we require that no two coprime irreducible factors of `f`
    // are the same modulo `p`, and no irreducible factor of `f` contains a square modulo `p`)
    for p in enumerate_primes(&StaticRing::<i64>::RING, &1000).into_iter().skip(100) {

        let Fp = Zn::new(p as u64).as_field().ok().unwrap();
        let FpX = DensePolyRing::new(Fp, "X");

        let ZZbig = BigIntRing::RING;
        let ZZ = StaticRing::<i64>::RING;
        let bound = max_coeff_of_factor(ZZX, &f, &ZZX.base_ring().one());
        let exponent = max(2, ZZbig.abs_log2_ceil(&bound).unwrap() / ZZ.abs_log2_floor(&p).unwrap() + 1);
        let modulus = ZZbig.pow(int_cast(p, &ZZbig, &ZZ), exponent);

        if let Some(result) = choose_zn_impl(ZZbig, modulus, IntegerPolyPowerDecompositionUsingHenselLifting { ZZX: &ZZX, FpX: &FpX, f: &f }) {
            
            return ZZX.prod(result.into_iter().map(|(f, _)| f));
        }
    }
    unreachable!()
}

///
/// Computes the square-free part of a polynomial `f` over the rational numbers, 
/// i.e. the greatest (w.r.t. divisibility) polynomial `g | f` that is square-free, 
/// up to multiplication by nonzero integers.
/// 
/// The returned polynomial is always monic, and with this restriction, it
/// is unique. This function does the computations locally, i.e. modulo suitable
/// primes. This is usually much faster than working over `Z` or `Q`.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_rational<P, I>(QQX: P, f: &El<P>) -> El<P>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing + InterpolationBaseRing,
        ZnBase: CanHomFrom<I::Type>
{
    assert!(!QQX.is_zero(f));
    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();

    let den_lcm = QQX.terms(f).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
    
    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(f).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
    let squarefree_part = poly_squarefree_part_integer(&ZZX, &f);
    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());

    return QQX.normalize(ZZX_to_QQX.map(squarefree_part));
}

///
/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_global<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FactorPolyField
{
    assert!(!poly_ring.is_zero(&poly));
    let squarefree_part = <_ as FactorPolyField>::squarefree_part(poly_ring, poly);
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


#[test]
fn test_integer_poly_squarefree_part_local() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");

    let [f, g, h] = ZZX.with_wrapped_indeterminate(|X| [
        X.pow_ref(2) + 1,
        X.pow_ref(2) + 2,
        X + 1
    ]);

    let input = ZZX.prod([ZZX.clone_el(&f), ZZX.clone_el(&f), ZZX.clone_el(&g), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h)]);
    let actual = poly_squarefree_part_integer(&ZZX, &input);
    let expected = ZZX.prod([f, g, h]);
    assert_el_eq!(&ZZX, expected, actual);

    let [f, g, h] = ZZX.with_wrapped_indeterminate(|X| [
        15 * X.pow_ref(2) + 1,
        6 * X.pow_ref(2) + 2,
        X + 3
    ]);

    let input = ZZX.prod([ZZX.clone_el(&f), ZZX.clone_el(&f), ZZX.clone_el(&g), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.int_hom().map(30)]);
    let actual = poly_squarefree_part_integer(&ZZX, &input);
    let expected = make_primitive(&ZZX, ZZX.prod([f, g, h]));
    assert_el_eq!(&ZZX, expected, actual);
}

#[test]
fn test_factor_int_poly() {
    let ZZX = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1, X + 1]);
    let input = ZZX.mul_ref(&f, &g);
    let actual = poly_factor_integer(&ZZX, input);
    assert_eq!(2, actual.len());
    for (factor, e) in &actual {
        assert_eq!(1, *e);
        assert!(ZZX.eq_el(&f, factor) || ZZX.eq_el(&g, factor), "Got unexpected factor {}", ZZX.format(&factor));
    }

    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [5 * X.pow_ref(2) + 1, 3 * X.pow_ref(2) + 2]);
    let input = ZZX.mul_ref(&f, &g);
    let actual = poly_factor_integer(&ZZX, input);
    assert_eq!(2, actual.len());
    for (factor, e) in &actual {
        assert_eq!(1, *e);
        assert!(ZZX.eq_el(&f, factor) || ZZX.eq_el(&g, factor), "Got unexpected factor {}", ZZX.format(&factor));
    }

    let [f] = ZZX.with_wrapped_indeterminate(|X| [5 * X.pow_ref(2) + 1]);
    let input = ZZX.mul_ref(&f, &f);
    let actual = poly_factor_integer(&ZZX, input);
    assert_eq!(1, actual.len());
    assert_eq!(2, actual[0].1);
    assert_el_eq!(&ZZX, &f, &actual[0].0);
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
