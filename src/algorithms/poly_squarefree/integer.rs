use std::alloc::Global;
use std::cmp::max;

use zn_64::Zn;
use zn_64::ZnBase;

use crate::algorithms::eea::poly::make_primitive;
use crate::algorithms::eea::signed_lcm;
use crate::algorithms::erathostenes::enumerate_primes;
use crate::algorithms::hensel::hensel_lift_factorization;
use crate::algorithms::int_bisect;
use crate::algorithms::interpolate::interpolate;
use crate::algorithms::poly_factor::integer::max_coeff_of_factor;
use crate::compute_locally::InterpolationBaseRing;
use crate::compute_locally::ToExtRingMap;
use crate::field::Field;
use crate::homomorphism::CanHomFrom;
use crate::homomorphism::Homomorphism;
use crate::divisibility::*;
use crate::integer::IntegerRing;
use crate::field::*;
use crate::primitive_int::StaticRing;
use crate::rings::poly::*;
use crate::ring::*;
use crate::integer::*;
use crate::pid::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::*;
use crate::seq::VectorView;

use super::poly_power_decomposition_global;
use super::FiniteRingSpecializable;

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
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + PerfectField + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> + FiniteRingSpecializable
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
pub fn integer_poly_squarefree_part_local<P>(ZZX: P, f: &El<P>) -> El<P>
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
pub fn rational_poly_squarefree_part_local<P, I>(QQX: P, f: &El<P>) -> El<P>
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
    let squarefree_part = integer_poly_squarefree_part_local(&ZZX, &f);
    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());

    return QQX.normalize(ZZX_to_QQX.map(squarefree_part));
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
    let actual = integer_poly_squarefree_part_local(&ZZX, &input);
    let expected = ZZX.prod([f, g, h]);
    assert_el_eq!(&ZZX, expected, actual);

    let [f, g, h] = ZZX.with_wrapped_indeterminate(|X| [
        15 * X.pow_ref(2) + 1,
        6 * X.pow_ref(2) + 2,
        X + 3
    ]);

    let input = ZZX.prod([ZZX.clone_el(&f), ZZX.clone_el(&f), ZZX.clone_el(&g), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.clone_el(&h), ZZX.int_hom().map(30)]);
    let actual = integer_poly_squarefree_part_local(&ZZX, &input);
    let expected = make_primitive(&ZZX, ZZX.prod([f, g, h]));
    assert_el_eq!(&ZZX, expected, actual);
}