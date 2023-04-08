use crate::{divisibility::DivisibilityRing, integer::{IntegerRing, IntegerRingStore}, ring::*, algorithms};

pub mod zn_dyn;
pub mod zn_static;
pub mod zn_rns;

pub trait ZnRing: DivisibilityRing + CanonicalHom<Self::IntegerRingBase> {

    // there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingStore`
    // or `Integers: RingStore<Type: IntegerRing>`
    type IntegerRingBase: IntegerRing + ?Sized;
    type Integers: RingStore<Type = Self::IntegerRingBase>;
    type IteratorState;

    fn integer_ring(&self) -> &Self::Integers;
    fn modulus(&self) -> &El<Self::Integers>;
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers>;
    fn elements<'a>(&'a self) -> ZnElementsIterator<'a, Self>;

    ///
    /// This is a workaround that enables us to "return" an iterator depending
    /// on the lifetimes of self; You should never have to call these functions.
    /// The drawback of this method is that the iterator state cannot depend on the
    /// element or self.
    /// 
    fn elements_iterator_next<'a>(iter: &mut ZnElementsIterator<'a, Self>) -> Option<Self::Element>;

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

pub struct ZnElementsIterator<'a, R: ?Sized + ZnRing> {
    ring: &'a R,
    state: R::IteratorState
}

impl<'a, R: ?Sized + ZnRing> Clone for ZnElementsIterator<'a, R>
    where R::IteratorState: Clone
{
    fn clone(&self) -> Self {
        Self::new(self.ring, self.state.clone())
    }
}

impl<'a, R: ?Sized + ZnRing> Copy for ZnElementsIterator<'a, R>
    where R::IteratorState: Copy
{}

impl<'a, R: ?Sized + ZnRing> ZnElementsIterator<'a, R> {

    pub fn new(ring: &'a R, state: R::IteratorState) -> Self {
        ZnElementsIterator { ring: ring, state: state }
    }

    pub fn ring(&self) -> &R {
        self.ring
    }

    pub fn state(&mut self) -> &mut R::IteratorState {
        &mut self.state
    }
}

impl<'a, R: ?Sized + ZnRing> Iterator for ZnElementsIterator<'a, R> {

    type Item = R::Element;

    fn next(&mut self) -> Option<R::Element> {
        R::elements_iterator_next(self)
    }
}
