use crate::{ring::*, integer::{IntegerRingStore, IntegerRing}};

pub trait FiniteRing: SelfIso {

    type ElementsIter<'a>: Iterator<Item = Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a>;
    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element;
    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> El<I>
        where I::Type: IntegerRing;
}

pub trait FiniteRingStore: RingStore
    where Self::Type: FiniteRing
{
    fn elements<'a>(&'a self) -> <Self::Type as FiniteRing>::ElementsIter<'a> {
        self.get_ring().elements()
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> {
        self.get_ring().random_element(rng)
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> El<I>
        where I::Type: IntegerRing
    {
        self.get_ring().size(ZZ)
    }
}

impl<R: RingStore> FiniteRingStore for R
    where R::Type: FiniteRing
{}