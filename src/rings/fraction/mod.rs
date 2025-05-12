use crate::field::Field;
use crate::ring::*;

///
/// Contains [`fraction_impl::FractionFieldImpl`], an implementation of the fraction
/// fields of an arbitrary principal ideal domain.
/// 
pub mod fraction_impl;

///
/// Trait for fields that are the field of fractions over a base ring.
/// 
/// Note that a field of fractions is usually the field of fractions of
/// many rings - in particular, every field is technically its own field
/// of fractions. However, such cases don't add any value, and this trait
/// is mainly designed and implemented for fields that have a "canonical" or
/// "natural" subring whose field of fractions they represent. In many
/// cases, this is the smallest subring whose fractions generate the whole
/// field, but this trait can also be implemented in other cases where it
/// makes sense.
/// 
#[stability::unstable(feature = "enable")]
pub trait FractionField: Field + RingExtension {

    ///
    /// Returns `a, b` such that the given element is `a/b`.
    /// 
    fn as_fraction(&self, el: Self::Element) -> (El<Self::BaseRing>, El<Self::BaseRing>);

    ///
    /// Computes `num / den`.
    /// 
    /// This is functionally equivalent, but may be faster than combining
    /// [`RingExtension::from()`] and [`Field::div()`].
    /// 
    fn from_fraction(&self, num: El<Self::BaseRing>, den: El<Self::BaseRing>) -> Self::Element {
        self.div(&self.from(num), &self.from(den))
    }
}

///
/// [`RingStore`] corresponding to [`FractionField`]
/// 
#[stability::unstable(feature = "enable")]
pub trait FractionFieldStore: RingStore
    where Self::Type: FractionField
{
    delegate!{ FractionField, fn as_fraction(&self, el: El<Self>) -> (El<<Self::Type as RingExtension>::BaseRing>, El<<Self::Type as RingExtension>::BaseRing>) }
    delegate!{ FractionField, fn from_fraction(&self, num: El<<Self::Type as RingExtension>::BaseRing>, den: El<<Self::Type as RingExtension>::BaseRing>) -> El<Self> }
}

impl<R: RingStore> FractionFieldStore for R
    where R::Type: FractionField
{}