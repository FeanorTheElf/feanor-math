use crate::ring::*;
use crate::algorithms::int_factor;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::primitive_int::StaticRing;
use crate::rings::extension::RingStore;
use crate::rings::field::AsField;
use crate::rings::finite::FiniteRingStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::PolyRingStore;
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::ZnRingStore;

use super::extension_impl::FreeAlgebraImpl;

pub type GaloisField<const DEGREE: usize> = AsField<FreeAlgebraImpl<AsField<Zn>, [El<AsField<Zn>>; DEGREE]>>;
pub type GaloisFieldDyn = AsField<FreeAlgebraImpl<AsField<Zn>, Box<[El<AsField<Zn>>]>>>;

///
/// Creates a finite/galois field of degree known at compile time. The
/// given p must be a prime and will be the characteristic of the returned
/// field.
/// 
/// See also [`GFdyn()`] if the degree of the field is not a compile-time constant.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let F25 = GF::<2>(5);
/// let generator = F25.canonical_gen();
/// let norm = F25.mul_ref_fst(&generator, F25.pow(F25.clone_el(&generator), 5));
/// let inclusion = F25.inclusion();
/// // the norm must be an element of the prime field
/// assert!(F25.base_ring().elements().any(|x| {
///     F25.eq_el(&norm, &inclusion.map(x))
/// }));
/// ```
/// 
pub fn GF<const DEGREE: usize>(p: u64) -> GaloisField<DEGREE> {
    assert!(p > 2, "binary galois fields are currently not supported");
    assert!(DEGREE > 1);
    let Fp = Zn::new(p).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(Fp, "X");
    let mut rng = oorandom::Rand64::new(p as u128);
    loop {
        let random_poly = poly_ring.from_terms((0..DEGREE).map(|i| (Fp.random_element(|| rng.rand_u64()), i)).chain([(Fp.one(), DEGREE)].into_iter()));
        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &random_poly);
        assert_el_eq!(&Fp, &Fp.one(), &unit);
        if factorization.len() == 1 && factorization[0].1 == 1 {
            return FreeAlgebraImpl::new(Fp, std::array::from_fn(|i| Fp.negate(Fp.clone_el(poly_ring.coefficient_at(&random_poly, i))))).as_field().ok().unwrap();
        }
    }
}

///
/// Creates a finite/galois field of degree not known at compile time. The
/// given p must be a prime and will be the characteristic of the returned
/// field.
/// 
/// See also [`GF()`] if the degree of the field is a compile-time constant.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let F25 = GFdyn(25);
/// let generator = F25.canonical_gen();
/// let norm = F25.mul_ref_fst(&generator, F25.pow(F25.clone_el(&generator), 5));
/// let inclusion = F25.inclusion();
/// // the norm must be an element of the prime field
/// assert!(F25.base_ring().elements().any(|x| {
///     F25.eq_el(&norm, &inclusion.map(x))
/// }));
/// ```
/// 
pub fn GFdyn(power_of_p: u64) -> GaloisFieldDyn {
    let (p, e) = int_factor::is_prime_power(&StaticRing::<i64>::RING, &(power_of_p as i64)).unwrap();
    assert!(p > 2, "binary galois fields are currently not supported");
    assert!(e > 1);
    let Fp = Zn::new(p as u64).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(Fp, "X");
    let mut rng = oorandom::Rand64::new(p as u128);
    loop {
        let random_poly = poly_ring.from_terms((0..e).map(|i| (Fp.random_element(|| rng.rand_u64()), i)).chain([(Fp.one(), e)].into_iter()));
        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &random_poly);
        assert_el_eq!(&Fp, &Fp.one(), &unit);
        if factorization.len() == 1 && factorization[0].1 == 1 {
            return FreeAlgebraImpl::new(Fp, (0..e).map(|i| Fp.negate(Fp.clone_el(poly_ring.coefficient_at(&random_poly, i)))).collect::<Vec<_>>().into_boxed_slice()).as_field().ok().unwrap();
        }
    }
}

#[cfg(test)]
use crate::field::FieldStore;

#[test]
fn test_GF() {
    let F27 = GF::<3>(3);
    assert_eq!(27, F27.elements().count());
    for (i, a) in F27.elements().enumerate() {
        for (j, b) in F27.elements().enumerate() {
            assert!(i == j || !F27.eq_el(&a, &b));
            if !F27.is_zero(&b) {
                assert_el_eq!(&F27, &a, &F27.mul_ref_fst(&b, F27.div(&a, &b)));
            }
        }
    }
}

#[test]
fn test_GFdyn() {
    let F27 = GFdyn(27);
    assert_eq!(27, F27.elements().count());
    for (i, a) in F27.elements().enumerate() {
        for (j, b) in F27.elements().enumerate() {
            assert!(i == j || !F27.eq_el(&a, &b));
            if !F27.is_zero(&b) {
                assert_el_eq!(&F27, &a, &F27.mul_ref_fst(&b, F27.div(&a, &b)));
            }
        }
    }
}