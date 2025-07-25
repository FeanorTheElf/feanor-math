
///
/// This module contains [`rust_bigint::RustBigintRing`], a pure Rust implementation of
/// arbitrary precision integers. It is not very optimized, however
///  
pub mod rust_bigint;

///
/// This module contains the trait [`zn::ZnRing`] for all rings that represent a
/// quotient `Z/nZ` of the integers `Z`. Furthermore, it provides four different
/// implementions, in [`zn::zn_static`], [`zn::zn_big`], [`zn::zn_64`] and [`zn::zn_rns`].
/// 
pub mod zn;

///
/// This module contains the trait [`poly::PolyRing`] for all rings that represent a
/// univariate polynomial ring `R[X]` over any base ring. Furthermore, it provides
/// two different implementations, in [`poly::dense_poly`] and [`poly::sparse_poly`].
/// 
pub mod poly;

///
/// This module contains the wrapper [`field::AsField`] that can be used to create a ring
/// implementing [`crate::field::Field`] from rings that are fields, but do not implement
/// the trait (e.g. because being a field for them might be only determinable at runtime). 
/// 
pub mod field;

///
/// An approximate implementation of the field of complex numbers `C`, using 64-bit floating point
/// numbers.
/// 
pub mod float_complex;

///
/// This module contains the trait [`extension::FreeAlgebra`] for rings that are free modules of finite
/// rank over a base ring. It also provides one implementation in [`extension::extension_impl`] based
/// on polynomial division.
/// 
pub mod extension;

///
/// This module contains the trait [`finite::FiniteRing`] for all rings with finitely many elements.
/// 
pub mod finite;

///
/// This module contains the trait [`multivariate::MultivariatePolyRing`] for all multivariate polynomial
/// rings.
/// 
pub mod multivariate;

///
/// This module contains the trait [`approx_real::ApproxRealField`] for rings that implement
/// approximations to the real numbers, usually using floating-point numbers.
/// 
pub mod approx_real;

///
/// This module contains [`rational::RationalField`], which provides an implementation of the field of 
/// rational numbers `Q`.
/// 
pub mod rational;

///
/// This module contains the ring [`mpir::MPZ`] that represents the integers `Z` and uses the heavily
/// optimized arbitrary-precision integer library mpir as implementation.
/// 
/// Note that to use it, you have to activate the feature "mpir" and provide the compiler with
/// the location of the mpir library file - e.g. by setting `RUSTFLAGS="-L /location/to/mpir/dir"`.
/// 
#[doc(cfg(feature = "mpir"))]
#[cfg(feature = "mpir")]
pub mod mpir;

///
/// This module contains the wrapper [`local::AsLocalPIR`] that wraps a ring to signal that it is a
/// local principal ideal ring [`crate::local::PrincipalLocalRing`]. 
/// 
pub mod local;

///
/// This module contains the trait [`fraction::FractionField`] for rings that are the field of fractions
/// of an underlying domain.
/// 
pub mod fraction;

pub mod direct_power;