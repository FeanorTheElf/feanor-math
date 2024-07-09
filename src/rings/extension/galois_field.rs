use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::algorithms::int_factor;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::primitive_int::StaticRing;
use crate::rings::extension::{Homomorphism, RingStore};
use crate::rings::field::AsField;
use crate::rings::finite::FiniteRingStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::PolyRingStore;
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::ZnRingStore;

use super::conway::*;
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

///
/// Creates a finite/galois field, using a generating polynomial from a table of Conway polynomials.
/// Since Conway polynomials are unique, this is useful for comparison with other Computer Algebra systems
/// that support Conway polynomials, e.g. SAGE.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// # use feanor_math::rings::poly::dense_poly::DensePolyRing;
/// # use feanor_math::rings::poly::*;
/// let F32 = GF_conway(32);
/// let F2 = F32.base_ring();
/// let poly_ring = DensePolyRing::new(F2, "x");
/// let gen_poly = F32.generating_poly(&poly_ring, F2.identity());
/// // SAGE: GF(32).gen().minpoly() == x^5 + x^2 + 1
/// assert_el_eq!(&poly_ring, &poly_ring.from_terms([(F2.one(), 0), (F2.one(), 2), (F2.one(), 5)].into_iter()), &gen_poly);
/// ```
/// 
pub fn GF_conway(power_of_p: u64) -> GaloisFieldDyn {
    if power_of_p % 2 == 0 {
        let log_q = StaticRing::<i64>::RING.abs_log2_ceil(&(power_of_p as i64)).unwrap();
        assert!(power_of_p == 1 << log_q);
        assert!(log_q >= 2);
        if log_q - 2 >= EVEN_CONWAY_POLYNOMIALS.len() {
            panic!("{} not in table of Conway polynomials", power_of_p);
        }
        let Fp = Zn::new(2).as_field().ok().unwrap();
        let int_hom = Fp.int_hom();
        return FreeAlgebraImpl::new(Fp, EVEN_CONWAY_POLYNOMIALS[log_q - 2].iter().take(log_q).map(|c| int_hom.map(*c)).collect::<Vec<_>>().into_boxed_slice()).as_field().ok().unwrap();
    } else {
        let (p, e) = int_factor::is_prime_power(&StaticRing::<i64>::RING, &(power_of_p as i64)).unwrap();
        assert!(e > 1);
        match ODD_CONWAY_POLYNOMIALS.binary_search_by_key(&power_of_p, |(q, _)| *q) {
            Ok(idx) => {
                let Fp = Zn::new(p as u64).as_field().ok().unwrap();
                let int_hom = Fp.int_hom();
                return FreeAlgebraImpl::new(Fp, ODD_CONWAY_POLYNOMIALS[idx].1.iter().take(e).map(|c| int_hom.map(*c)).collect::<Vec<_>>().into_boxed_slice()).as_field().ok().unwrap();
            },
            Err(_) => panic!("{} not in table of Conway polynomials", power_of_p)
        }
    }
}

#[test]
fn test_GF() {
    let F27 = GF::<3>(3);
    assert_eq!(27, F27.elements().count());
    crate::field::generic_tests::test_field_axioms(F27, F27.elements());
}

#[test]
fn test_GFdyn() {
    let F27 = GFdyn(27);
    assert_eq!(27, F27.elements().count());
    crate::field::generic_tests::test_field_axioms(&F27, F27.elements());
}

#[test]
fn test_GFdyn_even() {
    let F16 = GFdyn(16);
    assert_eq!(16, F16.elements().count());
    crate::field::generic_tests::test_field_axioms(&F16, F16.elements());
}

#[test]
fn test_GF_conway() {
    let F16 = GF_conway(16);
    assert_eq!(16, F16.elements().count());
    crate::field::generic_tests::test_field_axioms(&F16, F16.elements());
}