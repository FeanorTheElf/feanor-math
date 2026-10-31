/// Contains Buchberger's algorithm for computing Groebner basis.
pub mod buchberger;
/// Contains multivariate polynomial division, i.e. reduction of polynomials modulo a
/// set of polynomials.
mod reduction;
/// Contains critical pairs, the Gebauer-Moeller criteria and the basis data structure
/// that manages both.
mod spoly;

pub use buchberger::buchberger;
#[allow(unused_imports)]
#[stability::unstable(feature = "enable")]
pub use buchberger::buchberger_with_strategy;
#[allow(unused_imports)]
#[stability::unstable(feature = "enable")]
pub use reduction::multivariate_division;
