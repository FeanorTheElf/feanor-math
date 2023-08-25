
///
/// This module contains [`RustBigintRing`], a pure Rust implementation of
/// arbitrary precision integers. It is not very optimized, however
///  
pub mod rust_bigint;

///
/// This module contains the trait [`ZnRing`] for all rings that represent a
/// quotient `Z/nZ` of the integers `Z`. Furthermore, it provides four different
/// implementions, in [`zn_static`], [`zn_barett`], [`zn_42`] and [`zn_rns`].
/// 
pub mod zn;

///
/// This module contains the trait [`PolyRing`] for all rings that represent a
/// univariate polynomial ring `R[X]` over any base ring. Furthermore, it provides
/// two different implementations, in [`dense_poly`] and [`sparse_poly`].
/// 
pub mod poly;

///
/// This module contains the wrapper [`AsField`] that can be used to create a ring
/// implementing [`crate::field::Field`] from rings that are fields, but do not implement
/// the trait (e.g. because being a field for them might be only determinable at runtime). 
/// 
pub mod field;

///
/// An implementation of the field of complex numbers `C`, using 64-bit floating point
/// numbers.
/// 
pub mod float_complex;

///
/// This module contains the trait [`FreeAlgebra`] for rings that are free modules of finite
/// rank over a base ring. It also provides one implementation in [`extension_impl`] based
/// on polynomial division.
/// 
pub mod extension;

///
/// This module contains the trait [`FiniteRing`] for all rings with finitely many elements.
/// 
pub mod finite;

///
/// This module contains the ring [`MPZ`] that represents the integers `Z` and uses the heavily
/// optimized arbitrary-precision integer library mpir as implementation.
/// 
/// Note that to use it, you have to activate the feature "mpir" and provide the compiler with
/// the location of the mpir library file - e.g. by setting `RUSTFLAGS="-L /location/to/mpir/dir"`.
/// 
#[cfg(feature = "mpir")]
pub mod mpir;