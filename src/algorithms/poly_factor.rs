use crate::divisibility::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::ring::*;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::algorithms;

pub fn factor_int_poly<P>(poly_ring: &P, poly: &El<P>)
    where P: PolyRingStore,
        P::Type: PolyRing + DivisibilityRing,
        <P::Type as RingExtension>::BaseRing: IntegerRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
{
    unimplemented!()
}