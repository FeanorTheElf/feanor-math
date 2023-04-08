use crate::{divisibility::DivisibilityRing, integer::{IntegerRing, IntegerRingStore}, ring::*, algorithms};

pub mod zn_dyn;
pub mod zn_static;
pub mod zn_rns;

pub trait ZnRing: DivisibilityRing + CanonicalHom<Self::IntegerRingBase> {

    // there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingStore`
    // or `Integers: RingStore<Type: IntegerRing>`
    type IntegerRingBase: IntegerRing + ?Sized;
    type Integers: RingStore<Type = Self::IntegerRingBase>;

    fn integer_ring(&self) -> &Self::Integers;
    fn modulus(&self) -> &El<Self::Integers>;
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers>;

    fn is_field(&self) -> bool {
        algorithms::miller_rabin::is_prime(self.integer_ring(), self.modulus(), 6)
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        self.map_in(
            self.integer_ring().get_ring(), 
            self.integer_ring().get_uniformly_random(self.modulus(), rng), 
            &self.has_canonical_hom(self.integer_ring().get_ring()).unwrap()
        )
    }
}

pub trait ZnRingStore: RingStore<Type: ZnRing> {
    
    delegate!{ fn integer_ring(&self) -> &<Self::Type as ZnRing>::Integers }
    delegate!{ fn modulus(&self) -> &El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn smallest_positive_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn is_field(&self) -> bool }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> {
        self.get_ring().random_element(rng)
    }
}

impl<R: RingStore<Type: ZnRing>> ZnRingStore for R {}
