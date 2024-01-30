use crate::divisibility::Domain;
use crate::field::Field;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::rings::poly::*;
use crate::ring::*;

pub fn resultant<P>(ring: P, mut f: El<P>, mut g: El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Domain
{
    let base_ring = ring.base_ring();
    let mut unit_num = base_ring.one();
    let mut unit_den = base_ring.one();
    while !ring.is_zero(&f) {
        // use here that `res(f, g) = lc(f)^(deg(g) - deg(g - fh)) res(f, g - fh)` if `deg(fh) <= deg(g)`
        
    }
    unimplemented!()
}