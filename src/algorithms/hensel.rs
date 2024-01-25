use crate::pid::EuclideanRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::zn::{ZnRing, ZnRingStore};
use crate::ring::*;

///
/// Given a monic polynomial `f` modulo `p^r` and a factorization `f = gh mod p^e`
/// into monic and coprime polynomials `g, h` modulo `p^e`, `r > e`, computes a factorization
/// `f = g' h'` with `g', h'` monic polynomials modulo `p^r` that reduce to `g, h`
/// modulo `p^e`.
/// 
pub fn lift_factorization<P, R>(target_ring: &P, prime_ring: &R, f: &El<P>, factors: (&El<P>, &El<P>))
    where P: PolyRingStore,
        P::Type: PolyRing,
        R: PolyRingStore,
        R::Type: PolyRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        <<R as RingStore>::Type as RingExtension>::BaseRing: ZnRingStore,
        <<<R as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{

}
