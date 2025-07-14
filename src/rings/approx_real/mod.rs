use crate::field::Field;
use crate::integer::IntegerRing;
use crate::ordered::OrderedRing;
use crate::ring::*;

pub mod float;

#[derive(Debug, PartialEq, Eq)]
#[stability::unstable(feature = "enable")]
pub struct NotEnoughPrecision;

#[stability::unstable(feature = "enable")]
pub trait ApproxRealField: Field + OrderedRing {

    fn round_to_integer<I>(&self, ZZ: I, x: Self::Element) -> Option<El<I>>
        where I: RingStore, I::Type: IntegerRing;
        
    fn epsilon(&self) -> &Self::Element;

    fn infinity(&self) -> Self::Element;
}

#[stability::unstable(feature = "enable")]
pub trait SqrtRing: RingBase + OrderedRing {

    fn sqrt(&self, x: Self::Element) -> Self::Element;
}