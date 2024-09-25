use finite_field::poly_squarefree_part_if_finite_field;

use crate::algorithms::hensel::hensel_lift_factorization;
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
use crate::rings::zn::*;
use crate::field::FieldStore;
use crate::specialization::*;

pub mod finite_field;

struct IntegerPolyPowerDecompositionUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Q: RingStore,
        Q::Type: PolyRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    ZZX: P,
    FpX: Q,
    f: &'a El<P>
}

impl<'a, P, Q> ZnOperation<Vec<El<P>>> for IntegerPolyPowerDecompositionUsingHenselLifting<'a, P, Q>
    where P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing,
        Q: RingStore,
        Q::Type: PolyRing + EuclideanRing,
        <<Q::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + PerfectField + CanHomFrom<<<P::Type as RingExtension>::BaseRing as RingStore>::Type> + SpecializeToFiniteField
{
    fn call<R: ZnRingStore>(self, Zpe: R) -> Vec<El<P>>
        where R::Type: ZnRing
    {
        let Fp = self.FpX.base_ring();
        let ZZ = self.ZZX.base_ring();
        let ZZ_to_Fp = Fp.can_hom(ZZ).unwrap();
        let ZZX_to_FpX = self.FpX.lifted_hom(&self.ZZX, &ZZ_to_Fp);

        let f_mod_p = ZZX_to_FpX.map_ref(&self.f);
        let power_decomposition = poly_power_decomposition_global(&self.FpX, self.FpX.clone_el(&f_mod_p));
        let factors = power_decomposition.iter().map(|(f, k)| self.FpX.pow(self.FpX.clone_el(f), *k)).collect::<Vec<_>>();

        let ZpeX = DensePolyRing::new(&Zpe, "X");
        let ZZ_to_Zpe = Zpe.can_hom(Zpe.integer_ring()).unwrap();
        let reduce_pe = |c: &El<<P::Type as RingExtension>::BaseRing>| ZZ_to_Zpe.map(int_cast(ZZ.clone_el(c), Zpe.integer_ring(), ZZ));
        let f_mod_pe = ZpeX.from_terms(self.ZZX.terms(&self.f).map(|(c, i)| (reduce_pe(c), i)));

        let lifted_factors = hensel_lift_factorization(
            &ZpeX,
            &self.FpX,
            &self.FpX,
            &f_mod_pe,
            &factors
        );

        
        unimplemented!()
    }
}

///
/// Returns a list of `(fi, ki)` such that the `fi` are monic, square-free and pairwise coprime, and
/// `f = a prod_i fi^ki` for a unit `a` of the base field.
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_global<P>(poly_ring: P, poly: El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + PrincipalIdealRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + SpecializeToFiniteField
{
    assert!(!poly_ring.is_zero(&poly));
    let derivate = derive_poly(&poly_ring, &poly);
    if poly_ring.is_zero(&derivate) {
        let p = poly_ring.base_ring().characteristic(&StaticRing::<i64>::RING).unwrap() as usize;
        if poly_ring.degree(&poly).unwrap() == 0 {
            return Vec::new();
        } else {
            assert!(p > 0);
        }
        let base_poly = poly_ring.from_terms(poly_ring.terms(&poly).map(|(c, i)| {
            debug_assert!(i % p == 0);
            (poly_ring.base_ring().clone_el(c), i / p)
        }));
        let result: Vec<(El<P>, usize)> = poly_power_decomposition_global(poly_ring, base_poly).into_iter()
            .map(|(f, k)| (f, k * p))
            .collect();
        
        debug_assert!(poly_ring.eq_el(&poly_squarefree_part_global(&poly_ring, poly), &poly_ring.prod(result.iter().map(|(f, _k)| poly_ring.clone_el(f)))));
        return result;
    } else {
        let square_part = poly_ring.ideal_gen(&poly, &derivate);
        let mut result = poly_power_decomposition_global(poly_ring, square_part);
        let mut power_part = poly_ring.one();
        let mut degree = 0;
        for (f, k) in &mut result {
            *k += 1;
            degree += poly_ring.degree(f).unwrap() * *k;
            poly_ring.mul_assign(&mut power_part, poly_ring.pow(poly_ring.clone_el(f), *k));
        }
        if degree < poly_ring.degree(&poly).unwrap() {
            let mut new_part = poly_ring.checked_div(&poly, &power_part).unwrap();
            let lc_inv = poly_ring.base_ring().invert(poly_ring.lc(&new_part).unwrap()).unwrap();
            poly_ring.inclusion().mul_assign_map(&mut new_part, lc_inv);
            result.push((new_part, 1));
        }

        debug_assert!(poly_ring.eq_el(&poly_squarefree_part_global(&poly_ring, poly), &poly_ring.prod(result.iter().map(|(f, _k)| poly_ring.clone_el(f)))));
        return result;
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
/// let [f] = FpX.with_wrapped_indeterminate(|X| [(X.pow_ref(2) + 1).pow(2) * (X.pow_ref(3) + 2 * X + 1)]);
/// let squarefree_part = poly_squarefree_part(&FpX, f);
/// let [expected] = FpX.with_wrapped_indeterminate(|X| [(X.pow_ref(2) + 1) * (X.pow_ref(3) + 2 * X + 1)]);
/// assert_el_eq!(FpX, &expected, &squarefree_part);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_squarefree_part_global<P>(poly_ring: P, poly: El<P>) -> El<P>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing + PrincipalIdealRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + SpecializeToFiniteField
{
    assert!(!poly_ring.is_zero(&poly));
    if let Some(result) = poly_squarefree_part_if_finite_field(poly_ring, &poly) {
        return result;
    }
    let derivate = derive_poly(&poly_ring, &poly);
    if poly_ring.degree(&poly).unwrap() == 0 {
        return poly_ring.one();
    }
    if poly_ring.is_zero(&derivate) {
        unimplemented!("infinite field with positive characteristic are currently not supported")
    } else {
        let square_part = poly_ring.ideal_gen(&poly, &derivate);
        let mut result = poly_ring.checked_div(&poly, &square_part).unwrap();
        let lc_inv = poly_ring.base_ring().invert(poly_ring.lc(&result).unwrap()).unwrap();
        poly_ring.inclusion().mul_assign_map(&mut result, lc_inv);
        return result;
    }
}

#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::extension::galois_field::*;
#[cfg(test)]
use crate::rings::extension::*;
#[cfg(test)]
use crate::rings::rational::RationalField;

#[test]
fn test_poly_squarefree_part_global() {
    let ring = DensePolyRing::new(zn_static::Fp::<257>::RING, "X");
    let poly = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let expected = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let actual = poly_squarefree_part_global(&ring, poly);
    assert_el_eq!(ring, expected, actual);

    let QQ = RationalField::new(BigIntRing::RING);
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let [mut f, mut expected] = poly_ring.with_wrapped_indeterminate(|X| [
        16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4),
        4 - 4 * X + 11 * X.pow_ref(2)
    ]);
    poly_ring.inclusion().mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    poly_ring.inclusion().mul_assign_map(&mut expected, QQ.div(&QQ.one(), &QQ.int_hom().map(11)));

    let actual = poly_squarefree_part_global(&poly_ring, poly_ring.clone_el(&f));
    assert_el_eq!(poly_ring, expected, actual);
}

#[test]
fn test_poly_squarefree_part_global_multiplicity_p() {
    let ring = DensePolyRing::new(zn_64::Zn::new(5).as_field().ok().unwrap(), "X");
    let [f] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(10)]);
    let [g] = ring.with_wrapped_indeterminate(|X| [3 + X.pow_ref(2)]);
    let actual = poly_squarefree_part_global(&ring, f);
    assert_el_eq!(ring, g, actual);
}

#[test]
fn test_poly_squarefree_part_global_galois_field() {
    let ring = DensePolyRing::new(GaloisField::new(2, 3), "X");
    let f = ring.from_terms([(ring.base_ring().pow(ring.base_ring().canonical_gen(), 2), 0), (ring.base_ring().one(), 2)]);
    let g = ring.from_terms([(ring.base_ring().canonical_gen(), 0), (ring.base_ring().one(), 1)]);
    let actual = poly_squarefree_part_global(&ring, f);
    assert_el_eq!(ring, g, actual);
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
    let actual = poly_power_decomposition_global(&ring, poly);
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
