use crate::{divisibility::DivisibilityRing, integer::{IntegerRing}, ring::*};

pub mod zn_dyn;
pub mod zn_static;

pub trait ZnRing: DivisibilityRing {

    // there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingWrapper`
    // or `Integers: RingWrapper<Type: IntegerRing>`
    type IntegerRingBase: IntegerRing;
    type Integers: RingWrapper<Type = Self::IntegerRingBase>;

    fn integer_ring(&self) -> &Self::Integers;
    fn modulus(&self) -> &El<Self::Integers>;
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers>;
}

pub trait ZnRingWrapper: RingWrapper<Type: ZnRing> {
    
    delegate!{ fn integer_ring(&self) -> &<Self::Type as ZnRing>::Integers }
    delegate!{ fn modulus(&self) -> &El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn smallest_positive_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::Integers> }
}

impl<R: RingWrapper<Type: ZnRing>> ZnRingWrapper for R {}
