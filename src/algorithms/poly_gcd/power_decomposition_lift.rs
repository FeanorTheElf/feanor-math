use std::iter::repeat;

use crate::algorithms::poly_gcd::gcd_lift::LiftUnsuccessful;
use crate::prelude::*;
use crate::ring_impls::poly::*;

const FRACTION_TO_ATTEMPT_LIFT: f64 = 0.51;
const RAMP_UP_LIFT_TO: f64 = 4.0;

/// The polynomial behaves badly modulo the current prime. One example is that the prime splits into
/// multiple prime ideals and the power decomposition looks different modulo different prime ideals.
#[stability::unstable(feature = "enable")]
#[derive(Debug)]
pub struct BadPrime;

#[stability::unstable(feature = "enable")]
pub type PolyPowerDecompositionResult<T> = Result<T, LiftUnsuccessful>;

/// The `i`-th entry is the degree of the multiple-`i` factor of the initial polynomial
#[stability::unstable(feature = "enable")]
pub struct PolyPowerDecompositionSignature {
    degrees: Vec<usize>,
}

impl PolyPowerDecompositionSignature {
    #[stability::unstable(feature = "enable")]
    pub fn from_decomposition<P>(poly_ring: P, decomposition: &[(El<P>, usize)]) -> Self
    where
        P: RingStore,
        P::Ring: PolyRing,
    {
        let mut signature_vector = Vec::new();
        let mut total_deg = 0;
        for (poly, i) in decomposition {
            signature_vector.resize(usize::max(signature_vector.len(), *i + 1), 0);
            signature_vector[*i] = poly_ring.degree(poly).unwrap();
            total_deg += i * poly_ring.degree(poly).unwrap();
        }
        return Self::new(signature_vector, total_deg);
    }

    #[stability::unstable(feature = "enable")]
    pub fn new(degrees: Vec<usize>, total_degree: usize) -> Self {
        assert_eq!(0, degrees[0]);
        assert_eq!(
            total_degree,
            degrees.iter().enumerate().map(|(i, di)| i * di).sum::<usize>()
        );
        Self { degrees }
    }

    /// Generally, the "refinement" order of these signatures is hard to work with
    /// (i.e. the order: "x <= y" if there could be a way for factors in y to split
    /// further in a way that we reach power decomposition with signature x). However,
    /// it is clear that `x <= y` implies `x.signature_sum() <= y.signature_sum()`,
    /// so let's use this as proxy.
    #[stability::unstable(feature = "enable")]
    pub fn signature_sum(&self) -> usize { self.degrees.iter().copied().sum() }
}

impl PartialEq for PolyPowerDecompositionSignature {
    fn eq(&self, other: &Self) -> bool {
        self.degrees
            .iter()
            .copied()
            .map(Some)
            .chain(repeat(None))
            .zip(other.degrees.iter().copied().map(Some).chain(repeat(None)))
            .take_while(|(self_, other_)| self_.is_some() || other_.is_some())
            .all(|(self_, other_)| self_.unwrap_or(0) == other_.unwrap_or(0))
    }
}

/// High-level approach of deriving the power decomposition of a polynomial by
/// computing it in quotients, and trying to lift the result.
///
/// This function doesn't handle any arithmetic, but encodes the
/// high-level strategy:
///  - The iterator yields `(signature(power_decomposition(poly mod p)), state_of(p))` for various
///    different prime ideals p.
///  - If the results seem to indicate that some of the power decompositions in the quotient
///    actually represent the global power decomposition, `start_lift` is called for the state
///    associated to a suitable prime `p`.
///  - afterwards, `proceed_with_lift(state_of(p), e)` is called, and should attempt to lift the
///    gcd-factorization to `R/p^e`; if that gives rise to a global power decomposition, it returns
///    "success", otherwise it returns "lift unsucessful".
///  - `proceed_with_lift` may later be called on the same state for larger values of `e`, if this
///    is deemed sensible, so it can be advantageous to store the current lift and continue from
///    there.
#[stability::unstable(feature = "enable")]
pub fn poly_power_decomposition_from_quotients<I, F_start, F_proc, State, OngoingLift, R>(
    gcd_in_quotients: I,
    mut start_lift: F_start,
    mut proceed_with_lift: F_proc,
) -> PolyPowerDecompositionResult<R>
where
    I: Iterator<Item = (PolyPowerDecompositionSignature, State)>,
    F_start: FnMut(State) -> Result<OngoingLift, BadPrime>,
    F_proc: FnMut(&mut OngoingLift, usize) -> Result<R, LiftUnsuccessful>,
{
    let mut expected_signature = PolyPowerDecompositionSignature::new(vec![0, usize::MAX], usize::MAX);
    let mut queued_lifts = Vec::new();
    let mut ongoing_lifts = Vec::new();
    let mut attempts = 0;
    for (signature, state) in gcd_in_quotients {
        attempts += 1;
        if signature.signature_sum() < expected_signature.signature_sum() {
            attempts = 1;
            expected_signature = signature;
            queued_lifts.clear();
            ongoing_lifts.clear();
        } else if signature.signature_sum() > expected_signature.signature_sum() {
            continue;
        } else if signature != expected_signature {
            let signature_sum = signature.signature_sum().checked_sub(1).unwrap();
            expected_signature = PolyPowerDecompositionSignature::new(vec![0, signature_sum], signature_sum);
        }
        queued_lifts.push(state);
        if ongoing_lifts.len() + queued_lifts.len() >= (attempts as f64 * FRACTION_TO_ATTEMPT_LIFT).ceil() as usize {
            if let Some(state) = queued_lifts.pop() {
                if let Ok(lift) = start_lift(state) {
                    ongoing_lifts.push((0, lift));
                }
            }
            for (lift_attempt, state) in &mut ongoing_lifts {
                *lift_attempt += 1;
                if let Ok(res) = proceed_with_lift(state, RAMP_UP_LIFT_TO.powi(*lift_attempt).ceil() as usize) {
                    return Ok(res);
                }
            }
        }
    }
    return Err(LiftUnsuccessful);
}
