
pub mod primelist;
///
/// Contains the trait [`poly_eval::LiftPolyEvalRing`] that formalizes the assumptions
/// required when we want to perform some algebraic computation (i.e. given by a polynomial)
/// modulo prime ideals, and reconstruct the result over the original ring from that. 
/// 
pub mod lift_poly_eval;
///
/// Contains the trait [`poly_factor_gcd::PolyLiftFactorsDomain`] that formalizes the assumptions
/// required when we want to compute the factorization or gcd of polynomials modulo a power of
/// a maximal ideal (using Hensel's lemma), and reconstruct the result over the original ring
/// from that.
/// 
pub mod lift_poly_factors;