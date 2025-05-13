
///
/// Contains the trait [`poly_eval::EvalPolyLocallyRing`] that formalizes the assumptions
/// required when we want to perform some algebraic computation (i.e. given by a polynomial)
/// modulo prime ideals, and reconstruct the result over the original ring from that. 
/// 
pub mod poly_eval;
///
/// Contains the trait [`poly_factor_gcd::PolyGCDLocallyDomain`] that formalizes the assumptions
/// required when we want to compute the factorization or gcd of polynomials modulo a power of
/// a maximal ideal (using Hensel's lemma), and reconstruct the result over the original ring
/// from that.
/// 
pub mod poly_factor_gcd;