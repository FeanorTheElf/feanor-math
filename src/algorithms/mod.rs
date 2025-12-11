///
/// Contains [`convolution::ConvolutionAlgorithm`], an abstraction for algorithms
/// for computing convolutions, together with various implementations.
/// 
pub mod convolution;
///
/// Contains [`matmul::MatmulAlgorithm`], an abstraction for algorithms
/// for computing matrix multiplication, together with various implementations.
/// 
pub mod matmul;
///
/// Contains [`sqr_mul::generic_abs_square_and_multiply()`] and other functions
/// for computing a power of an element in a generic monoid.
/// 
pub mod sqr_mul;
///
/// Contains multiple variants of the Extended Euclidean Algorithm.
/// 
pub mod eea;
///
/// Contains algorithms to find roots of unity in finite fields.
/// 
pub mod unity_root;
///
/// Contains [`fft::FFTAlgorithm`], an abstraction for algorithms
/// for computing FFTs over various rings, together with different implementations.
/// 
pub mod fft;
///
/// Contains basic algorithms for implementing operations on arbitrary-precision
/// integers. Unless you are implementing your own big integer type, you should use
/// [`crate::integer::BigIntRing`] instead.
/// 
pub mod bigint_ops;
///
/// Contains basic algorithms for implementing operations on ring extensions. Unless
/// you are implementing your own extension ring type, you should use the operations
/// through [`crate::rings::extension::FreeAlgebra`] instead.
/// 
pub mod extension_ops;
///
/// Contains an implementation of the Miller-Rabin probabilistic primality test.
/// 
pub mod miller_rabin;
///
/// Contains an implementation of Lenstra's ECM factoring algorithm.
/// 
pub mod ec_factor;
///
/// Contains an implementation of the Sieve of Erathostenes, for enumerating
/// prime number up to a certain bound.
/// 
pub mod erathostenes;
///
/// Contains an implementation of the bisection method for computing roots, but
/// working with integers only.
/// 
pub mod int_bisect;
///
/// Contains an implementation of integer factoring and related utilities, delegating
/// to [`ec_factor`].
/// 
pub mod int_factor;
///
/// Contains an implementation of the Lenstra-Lenstra-Lovasz algorithm for lattice basis
/// reduction.
/// 
pub mod lll;
///
/// Contains [`cyclotomic::cyclotomic_polynomial()`] for computing cyclotomic polynomials.
/// 
pub mod cyclotomic;
///
/// Contains [`poly_div::poly_div_rem()`] for computing polynomial division. In most cases,
/// you will instead use this functionality through [`crate::pid::EuclideanRing::euclidean_div_rem()`].
/// 
pub mod poly_div;
///
/// Contains [`poly_factor::FactorPolyField`] for fields over which we can factor polynomials.
/// 
/// Additionally contains most of the algorithms for factoring polynomials over various fields
/// and rings.
/// 
pub mod poly_factor;
///
/// Contains various algorithms for computing discrete logarithms over generic monoids.
/// 
pub mod discrete_log;
///
/// Contains [`linsolve::LinSolveRing`] for rings over which we can solve linear systems.
/// 
/// Additionally contains most of the algorithms for actually solving linear systems over
/// various rings, including partial smith normal forms.
/// 
pub mod linsolve;
///
/// Contains algorithms for computing resultants.
/// 
pub mod resultant;
///
/// Contains algorithms for rational reconstruction, i.e. find a small rational number `x`
/// from its reduction modulo some `n` (coprime to the denominator of `x`).
/// 
pub mod rational_reconstruction;
///
/// Contains algorithms for polynomial interpolation.
/// 
pub mod interpolate;
///
/// Contains Buchberger's algorithm for computing Groebner basis.
/// 
pub mod buchberger;
///
/// Contains [`poly_gcd::PolyTFracGCDRing`] for rings over which we can compute polynomial gcds and
/// related operations, modulo multiplication by non-zero divisors.
/// 
pub mod poly_gcd;
///
/// Contains implementations to extend [`crate::rings::extension::number_field::NumberField`]s by adjoining
/// additional roots of polynomials.
/// 
// pub mod splitting_field;
///
/// Contains algorithms for computing divisions in ring extensions for which standard methods
/// are not sufficient.
/// 
pub mod zpe_extension;
///
/// Contains algorithms for computing the Galois group and Galois closure of a
/// [`crate::rings::extension::number_field::NumberField`].
/// 
// pub mod galois;
///
/// Contains an implementation of the Newton-Raphson method for approximating roots of
/// polynomials (and more generally, "well-behaved" functions).
/// 
pub mod newton;
///
/// Contains an implementation of the Fincke-Pohst lattice point enumeration algorithm.
/// 
pub mod fincke_pohst;
pub mod qr;