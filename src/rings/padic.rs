use crate::algorithms::int_factor::is_prime_power;
use crate::local::PrincipalLocalRing;
use crate::ring::*;
use crate::rings::zn::*;

///
/// A partial ring that tries to approximate the p-adic numbers.
/// Since the p-adic numbers are uncountable, this can clearly only be an approximation.
/// 
/// As usually, we store the first `max_precision` summands in the sum `sum_i a_i p^i` of a p-adic number `a`.
/// However, we still want to provide equality, thus we consider p-adic numbers "modulo `p^min_precision`".
/// This works, until (because of many division by `p`) we cannot compute `min_precision` terms of an element
/// anymore. In such a case, we panic.
/// 
/// In other words, elements are all of the form `sum_i a_i p^i` where `i` ranges from some `i_start <= 0` to 
/// `min_precision <= i_end < max_precision`.
/// 
#[stability::unstable(feature = "enable")]
pub struct PAdicModPEBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    p: El<<R::Type as ZnRing>::Integers>,
    max_precision: usize,
    min_precision: usize,
    ring: R
}

#[stability::unstable(feature = "enable")]
pub struct PAdicModPEEl<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    exponent: i64,
    el: El<R>
}

impl<R> PAdicModPEBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(min_precision: usize, computations_in: R) -> PAdicModPEBase<R> {
        let (p, e) = is_prime_power(computations_in.integer_ring(), computations_in.modulus()).unwrap();
        PAdicModPEBase {
            p: p,
            max_precision: e,
            min_precision: min_precision,
            ring: computations_in
        }
    }

}