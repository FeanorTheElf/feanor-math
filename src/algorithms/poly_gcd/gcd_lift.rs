#[stability::unstable(feature = "enable")]
pub enum PolyGCDResult<T> {
    FoundGCD(T),
    NotSquarefree,
    LiftUnsuccessful,
    TrivialGCD,
}

#[stability::unstable(feature = "enable")]
pub struct NotSquarefree;

#[stability::unstable(feature = "enable")]
pub enum NotLiftable {
    NotSquarefree,
    BadPrime,
}

#[stability::unstable(feature = "enable")]
#[derive(Debug)]
pub struct LiftUnsuccessful;

#[stability::unstable(feature = "enable")]
pub struct PolyGCDSignature {
    pub gcd_deg: usize,
}

const FRACTION_TO_ATTEMPT_LIFT: f64 = 0.51;
const NON_SQUAREFREE_COUNT_ABORT: usize = 3;
const RAMP_UP_LIFT_TO: f64 = 2.0;

/// High-level approach of deriving the gcd of two polynomials by
/// computing it in quotients, and trying to lift the result.
///
/// This function doesn't handle any arithmetic, but encodes the
/// high-level strategy:
///  - The iterator yields `(deg(gcd_mod_p), state_of(p))` for various different prime ideals p
///  - If the results seem to indicate that some of the gcds in the quotient actually represent the
///    real gcd, `start_lift` is called for the state associated to a suitable prime `p`
///  - afterwards, `proceed_with_lift(state_of(p), e)` is called, and should attempt to lift the
///    gcd-factorization to `R/p^e`; if that gives rise to a gcd-factorization in `R`, it returns
///    "success", otherwise it returns "lift unsucessful"
///  - `proceed_with_lift` may later be called on the same state for larger values of `e`, if this
///    is deemed sensible, so it can be advantageous to store the current lift and continue from
///    there
#[stability::unstable(feature = "enable")]
pub fn poly_gcd_from_quotients<I, F_start, F_proc, State, OngoingLift, R>(
    gcd_in_quotients: I,
    mut start_lift: F_start,
    mut proceed_with_lift: F_proc,
) -> PolyGCDResult<R>
where
    I: Iterator<Item = (PolyGCDSignature, State)>,
    F_start: FnMut(State) -> Result<OngoingLift, NotLiftable>,
    F_proc: FnMut(&mut OngoingLift, usize) -> Result<R, LiftUnsuccessful>,
{
    let mut expected_gcd_deg = usize::MAX;
    let mut queued_lifts = Vec::new();
    let mut ongoing_lifts = Vec::new();
    let mut found_non_squarefree = 0;
    let mut attempts = 0;
    for (signature, state) in gcd_in_quotients {
        attempts += 1;
        if signature.gcd_deg == 0 {
            return PolyGCDResult::TrivialGCD;
        } else if signature.gcd_deg < expected_gcd_deg {
            attempts = 1;
            expected_gcd_deg = signature.gcd_deg;
            queued_lifts.clear();
            ongoing_lifts.clear();
        } else if signature.gcd_deg > expected_gcd_deg {
            continue;
        }
        queued_lifts.push(state);
        if ongoing_lifts.len() + queued_lifts.len() >= (attempts as f64 * FRACTION_TO_ATTEMPT_LIFT).ceil() as usize {
            if let Some(state) = queued_lifts.pop() {
                if let Ok(started_lift) = start_lift(state) {
                    ongoing_lifts.push((0, started_lift));
                } else if found_non_squarefree + 1 >= NON_SQUAREFREE_COUNT_ABORT {
                    return PolyGCDResult::NotSquarefree;
                } else {
                    found_non_squarefree += 1;
                };
            }
            for (lift_attempt, state) in &mut ongoing_lifts {
                *lift_attempt += 1;
                if let Ok(res) = proceed_with_lift(state, RAMP_UP_LIFT_TO.powi(*lift_attempt).ceil() as usize) {
                    return PolyGCDResult::FoundGCD(res);
                }
            }
        }
    }
    return PolyGCDResult::LiftUnsuccessful;
}
