use crate::field::Field;
use crate::integer::IntegerRing;
use crate::ordered::OrderedRing;
use crate::ring::*;

///
/// Contains [`float::Real64`] as implementation of [`ApproxRealField`]
/// based on the primitive type [`f64`].
/// 
pub mod float;

///
/// Zero-sized struct that can be used as error value to indicate that
/// the currently used [`ApproxRealField`] does not have sufficient precision
/// to perform the demanded computation.
/// 
#[derive(Debug, PartialEq, Eq)]
#[stability::unstable(feature = "enable")]
pub struct NotEnoughPrecision;

///
/// Trait for rings that represent subfields of the reals, possibly
/// being only approximate.
/// 
/// This should only be implemented for rings that behave in a sense
/// like floating point approximations to the reals, with fixed precision.
/// However, it is allowed to have the precision to be infinite, and
/// possibly only represent subfields of the reals (e.g. it would be
/// allowed to implement this for exact implementations of the rationals,
/// but I don't think this would serve any purpose).
/// 
#[stability::unstable(feature = "enable")]
pub trait ApproxRealField: Field + OrderedRing {

    ///
    /// Returns the closest integer to the given number.
    /// 
    /// If the integer is not representable by `ZZ`, or the given
    /// number is infinite, returns `None`.
    /// 
    fn round_to_integer<I>(&self, ZZ: I, x: Self::Element) -> Option<El<I>>
        where I: RingStore, I::Type: IntegerRing;
    
    ///
    /// Returns the difference between one and the next larger
    /// representable number.
    /// 
    /// If the precision of the ring is infinite, this should return 0.
    /// 
    fn epsilon(&self) -> &Self::Element;

    ///
    /// Returns a value representing positive infinity.
    /// 
    fn infinity(&self) -> Self::Element;
}

///
/// Trait for rings that allow taking positive square roots of
/// positive numbers.
/// 
/// The most common case are likely to be approximations to the
/// real numbers, although it could also be implemented for certain
/// infinite-degree number fields with embeddings into the reals.
/// 
#[stability::unstable(feature = "enable")]
pub trait SqrtRing: RingBase + OrderedRing {

    ///
    /// Computes (possibly an approximation to) the unique real
    /// number `y >= 0` such that `y^2 = 0`.
    /// 
    /// If `x` is negative, this can either panic or return an
    /// equivalent of "NaN" in the ring.
    /// 
    fn sqrt(&self, x: Self::Element) -> Self::Element;
}