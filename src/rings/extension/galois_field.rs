use crate::algorithms::convolution::fftconv::FFTBasedConvolutionZn;
use crate::field::Field;
use crate::integer::IntegerRingStore;
use crate::pid::EuclideanRing;
use crate::ring::*;
use crate::algorithms::int_factor;
use crate::algorithms::poly_factor::{cantor_zassenhaus, poly_squarefree_part};
use crate::primitive_int::*;
use crate::rings::extension::*;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::finite::FiniteRingStore;
use crate::rings::local::{AsLocalPIR, AsLocalPIRBase};
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::{ReductionMap, ZnRing, ZnRingStore};
use crate::local::PrincipalLocalRingStore;

use super::conway::*;
use super::impl_new::FreeAlgebraImpl;

pub type GaloisField<const DEGREE: usize> = AsField<FreeAlgebraImpl<AsField<Zn>, [El<AsField<Zn>>; DEGREE]>>;
pub type GaloisFieldDyn = AsField<FreeAlgebraImpl<AsField<Zn>, Box<[El<AsField<Zn>>]>>>;

#[stability::unstable(feature = "enable")]
pub type GaloisRingDyn = AsLocalPIR<FreeAlgebraImpl<AsLocalPIR<Zn>, Box<[El<AsLocalPIR<Zn>>]>>>;

fn test_is_irreducible_base<R, P>(poly_ring: P, mod_f_ring: R, degree: usize) -> Option<El<P>>
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field,
        R: RingStore,
        R::Type: FreeAlgebra,
        <R::Type as RingExtension>::BaseRing: RingStore<Type = <<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());
    let squarefree_part = poly_squarefree_part(&poly_ring, f);
    if poly_ring.degree(&squarefree_part) != Some(degree) {
        return None;
    }
    let distinct_degree_factorization = cantor_zassenhaus::distinct_degree_factorization_base(&poly_ring, &mod_f_ring);
    if distinct_degree_factorization.len() <= degree || poly_ring.degree(&distinct_degree_factorization[degree]) != Some(degree) {
        return None;
    }
    return Some(mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity()));
}

fn random_low_body_deg_irreducible_polynomial<P>(poly_ring: P, degree: usize) -> El<P>
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field + CanHomFrom<StaticRingBase<i64>>
{
    let mut body_deg = 1;
    let mut rng = oorandom::Rand64::new(poly_ring.base_ring().integer_ring().default_hash(poly_ring.base_ring().modulus()) as u128);
    let fft_convolution = FFTBasedConvolutionZn::new_with(Global);
    if fft_convolution.can_compute(StaticRing::<i64>::RING.abs_log2_ceil(&(degree as i64)).unwrap() + 1, poly_ring.base_ring().get_ring()) {
        loop {
            for _ in 0..8 {
                let f_body = (0..body_deg).map(|_| poly_ring.base_ring().random_element(|| rng.rand_u64())).collect::<Vec<_>>();
                let mod_f_ring = FreeAlgebraImpl::new_with(poly_ring.base_ring(), degree, &f_body, Global, &fft_convolution);
                if let Some(result) = test_is_irreducible_base(&poly_ring, mod_f_ring, degree) {
                    return result;
                }
            }
            if body_deg < degree {
                body_deg += 1;
            }
        }
    } else {
        loop {
            for _ in 0..8 {
                let f_body = (0..body_deg).map(|_| poly_ring.base_ring().random_element(|| rng.rand_u64())).collect::<Vec<_>>();
                let mod_f_ring = FreeAlgebraImpl::new(poly_ring.base_ring(), degree, &f_body);
                if let Some(result) = test_is_irreducible_base(&poly_ring, mod_f_ring, degree) {
                    return result;
                }
            }
            if body_deg < degree {
                body_deg += 1;
            }
        }
    }
}

///
/// Creates a finite/galois field of degree known at compile time. The
/// given p must be a prime and will be the characteristic of the returned
/// field.
/// 
/// Prefer [`galois_field_dyn()`], which allows ring degrees only known at runtime and will
/// usually be much faster, since it can choose a sparse modulus polynomial.
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
#[deprecated]
pub fn GF<const DEGREE: usize>(p: u64) -> GaloisField<DEGREE> {
    assert!(DEGREE >= 1);
    let Fp = Zn::new(p).as_field().ok().unwrap();
    if DEGREE == 1 {
        return FreeAlgebraImpl::new(Fp, DEGREE, std::array::from_fn(|_| Fp.one())).as_field().ok().unwrap();
    }
    let poly_ring = DensePolyRing::new(Fp, "X");
    let random_poly = random_low_body_deg_irreducible_polynomial(&poly_ring, DEGREE);
    return FreeAlgebraImpl::new(Fp, DEGREE, std::array::from_fn(|i| Fp.negate(Fp.clone_el(poly_ring.coefficient_at(&random_poly, i))))).as_field().ok().unwrap();
}

///
/// Creates a finite/galois field of degree not known at compile time. The
/// given p must be a prime and will be the characteristic of the returned
/// field.
/// 
/// See also [`GF()`] if the degree of the field is a compile-time constant.
/// 
/// This is deprecated in favor of [`galois_field_dyn()`], which allows creating
/// galois fields of size that exceeds `i64`.
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
#[deprecated]
pub fn GFdyn(power_of_p: u64) -> GaloisFieldDyn {
    let (p, e) = int_factor::is_prime_power(&StaticRing::<i64>::RING, &(power_of_p as i64)).unwrap();
    return galois_field_dyn(p, e);
}

///
/// Creates a finite/galois field of degree not known at compile time. The
/// given p must be a prime and will be the characteristic of the returned
/// field.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let F25 = galois_field_dyn(5, 2);
/// let generator = F25.canonical_gen();
/// let norm = F25.mul_ref_fst(&generator, F25.pow(F25.clone_el(&generator), 5));
/// let inclusion = F25.inclusion();
/// // the norm must be an element of the prime field
/// assert!(F25.base_ring().elements().any(|x| {
///     F25.eq_el(&norm, &inclusion.map(x))
/// }));
/// ```
/// 
pub fn galois_field_dyn(p: i64, degree: usize) -> GaloisFieldDyn {
    assert!(degree >= 1);
    let Fp = Zn::new(p as u64).as_field().ok().unwrap();
    if degree == 1 {
        return FreeAlgebraImpl::new(Fp, degree, vec![Fp.one()].into_boxed_slice()).as_field().ok().unwrap();
    }
    let poly_ring = DensePolyRing::new(Fp, "X");
    let random_poly = random_low_body_deg_irreducible_polynomial(&poly_ring, degree);
    let mut coefficients = (0..degree).map(|i| Fp.negate(Fp.clone_el(poly_ring.coefficient_at(&random_poly, i)))).collect::<Vec<_>>();
    let nonzero_coeff_count = (0..degree).rev().filter(|i| !Fp.is_zero(&coefficients[*i])).next().unwrap();
    coefficients.truncate(nonzero_coeff_count + 1);
    return AsField::from(AsFieldBase::promise_is_field(FreeAlgebraImpl::new(Fp, degree, coefficients.into_boxed_slice())));
}

///
/// Creates the galois ring of given degree and characteristic `p^e`.
/// 
/// The galois ring is the generalization of the galois field to an extension of `Z/p^eZ`.
/// In other words, it is a local ring and free module of given rank over `Z/p^eZ`.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// // sometimes also denoted GR(5^2, 3)
/// let R = galois_ring_dyn(5, 2, 3);
/// let generator = R.canonical_gen();
/// assert_eq!(25, R.characteristic(&StaticRing::<i64>::RING).unwrap());
/// assert_eq!(25 * 25 * 25, R.size(&StaticRing::<i64>::RING).unwrap());
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn galois_ring_dyn(p: i64, e: usize, degree: usize) -> GaloisRingDyn {
    assert!(degree >= 1);
    let Zpe = AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(p, e) as u64)).unwrap();
    if degree == 1 {
        let result = FreeAlgebraImpl::new(Zpe, degree, vec![Zpe.one()].into_boxed_slice());
        let max_ideal_gen = result.inclusion().map_ref(Zpe.max_ideal_gen());
        let nilpotent_power = Zpe.nilpotent_power();
        return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(result, max_ideal_gen, nilpotent_power));
    }
    let FpX = DensePolyRing::new(Zn::new(p as u64).as_field().ok().unwrap(), "X");
    let random_poly = random_low_body_deg_irreducible_polynomial(&FpX, degree);
    let red_map = ReductionMap::new(&Zpe, FpX.base_ring()).unwrap();
    let mut coefficients = (0..degree).map(|i| Zpe.negate(red_map.smallest_lift_ref(FpX.coefficient_at(&random_poly, i)))).collect::<Vec<_>>();
    let nonzero_coeff_count = (0..degree).rev().filter(|i| !Zpe.is_zero(&coefficients[*i])).next().unwrap();
    coefficients.truncate(nonzero_coeff_count + 1);
    let result = FreeAlgebraImpl::new(Zpe, degree, coefficients.into_boxed_slice());
    let max_ideal_gen = result.inclusion().map_ref(Zpe.max_ideal_gen());
    let nilpotent_power = Zpe.nilpotent_power();
    return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(result, max_ideal_gen, nilpotent_power));
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
/// assert_el_eq!(poly_ring, poly_ring.from_terms([(F2.one(), 0), (F2.one(), 2), (F2.one(), 5)].into_iter()), &gen_poly);
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
        return FreeAlgebraImpl::new(Fp, log_q, EVEN_CONWAY_POLYNOMIALS[log_q - 2].iter().take(log_q).map(|c| int_hom.map(*c)).collect::<Vec<_>>().into_boxed_slice()).as_field().ok().unwrap();
    } else {
        let (p, e) = int_factor::is_prime_power(&StaticRing::<i64>::RING, &(power_of_p as i64)).unwrap();
        assert!(e > 1);
        match ODD_CONWAY_POLYNOMIALS.binary_search_by_key(&power_of_p, |(q, _)| *q) {
            Ok(idx) => {
                let Fp = Zn::new(p as u64).as_field().ok().unwrap();
                let int_hom = Fp.int_hom();
                return FreeAlgebraImpl::new(Fp, e, ODD_CONWAY_POLYNOMIALS[idx].1.iter().take(e).map(|c| int_hom.map(*c)).collect::<Vec<_>>().into_boxed_slice()).as_field().ok().unwrap();
            },
            Err(_) => panic!("{} not in table of Conway polynomials", power_of_p)
        }
    }
}

use std::alloc::Global;
#[cfg(test)]
use std::time::Instant;

#[test]
#[allow(deprecated)]
fn test_GF() {
    let F7 = GF::<1>(7);
    assert_eq!(7, F7.elements().count());
    crate::field::generic_tests::test_field_axioms(F7, F7.elements());

    let F27 = GF::<3>(3);
    assert_eq!(27, F27.elements().count());
    crate::field::generic_tests::test_field_axioms(F27, F27.elements());
}

#[test]
#[allow(deprecated)]
fn test_GFdyn() {
    let F7 = GFdyn(7);
    assert_eq!(7, F7.elements().count());
    crate::field::generic_tests::test_field_axioms(&F7, F7.elements());

    let F27 = GFdyn(27);
    assert_eq!(27, F27.elements().count());
    crate::field::generic_tests::test_field_axioms(&F27, F27.elements());
}

#[test]
#[allow(deprecated)]
fn test_GFdyn_even() {
    let F16 = GFdyn(16);
    assert_eq!(16, F16.elements().count());
    crate::field::generic_tests::test_field_axioms(&F16, F16.elements());
    let F32 = GFdyn(32);
    assert_eq!(32, F32.elements().count());
    crate::field::generic_tests::test_field_axioms(&F32, F32.elements());
}

#[test]
fn test_GF_conway() {
    let F16 = GF_conway(16);
    assert_eq!(16, F16.elements().count());
    crate::field::generic_tests::test_field_axioms(&F16, F16.elements());
}

#[test]
#[ignore]
fn test_galois_ring_large() {
    let start = Instant::now();
    let ring = galois_ring_dyn(17, 5, 4096);
    let end = Instant::now();
    println!("Computed GR(17, 5, 4096) in {} ms", (end - start).as_millis());
    std::hint::black_box(ring);
}
