use crate::{divisibility::DivisibilityRing, integer::{IntegerRing, IntegerRingWrapper}, ring::*, algorithms};

pub mod zn_dyn;
pub mod zn_static;
pub mod zn_rns;

pub trait ZnRing: DivisibilityRing + CanonicalHom<Self::IntegerRingBase> {

    // there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingWrapper`
    // or `Integers: RingWrapper<Type: IntegerRing>`
    type IntegerRingBase: IntegerRing;
    type Integers: RingWrapper<Type = Self::IntegerRingBase>;

    fn integer_ring(&self) -> &Self::Integers;
    fn modulus(&self) -> &El<Self::Integers>;
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers>;

    fn is_field(&self) -> bool {
        algorithms::miller_rabin::is_prime(self.integer_ring(), self.modulus(), 6)
    }
}

pub trait ZnRingWrapper: RingWrapper<Type: ZnRing> {
    
    delegate!{ fn integer_ring(&self) -> &<Self::Type as ZnRing>::Integers }
    delegate!{ fn modulus(&self) -> &El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn smallest_positive_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn is_field(&self) -> bool }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> {
        self.map_in(self.integer_ring(), self.integer_ring().get_uniformly_random(self.modulus(), rng))
    }
}

impl<R: RingWrapper<Type: ZnRing>> ZnRingWrapper for R {}
