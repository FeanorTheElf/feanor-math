use crate::ring::*;

pub struct PolyRingCoefficientIterator<'a, R: ?Sized + PolyRing> {
    ring: &'a R,
    element: &'a R::Element,
    state: R::IteratorState
}

impl<'a, R: ?Sized + PolyRing> Clone for PolyRingCoefficientIterator<'a, R>
    where R::IteratorState: Clone
{
    fn clone(&self) -> Self {
        Self::new(self.ring, self.element, self.state.clone())
    }
}

impl<'a, R: ?Sized + PolyRing> Copy for PolyRingCoefficientIterator<'a, R>
    where R::IteratorState: Copy
{}

impl<'a, R: ?Sized + PolyRing> PolyRingCoefficientIterator<'a, R> {

    pub fn new(ring: &'a R, element: &'a R::Element, state: R::IteratorState) -> Self {
        PolyRingCoefficientIterator { ring: ring, element: element, state: state }
    }

    pub fn ring(&self) -> &R {
        self.ring
    }

    pub fn element(&self) -> &R::Element {
        self.element
    }

    pub fn state(&mut self) -> &mut R::IteratorState {
        &mut self.state
    }
}

impl<'a, R: ?Sized + PolyRing> Iterator for PolyRingCoefficientIterator<'a, R> {

    type Item = (&'a El<R::BaseRing>, usize);

    fn next(&mut self) -> Option<(&'a El<R::BaseRing>, usize)> {
        R::coefficient_iterator_next(self)
    }
}

pub trait PolyRing: RingExtension + CanonicalIso<Self> {

    type IteratorState;

    fn indeterminate(&self) -> Self::Element;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> PolyRingCoefficientIterator<'a, Self>;

    ///
    /// This is a workaround that enables us to "return" an iterator depending
    /// on the lifetimes of self; You should never have to call these functions.
    /// The drawback of this method is that the iterator state cannot depend on the
    /// element or self.
    /// 
    fn coefficient_iterator_next<'a>(iter: &mut PolyRingCoefficientIterator<'a, Self>) -> Option<(&'a El<Self::BaseRing>, usize)>;
    
    fn from_terms<'a, I>(&self, iter: I) -> Self::Element
        where I: Iterator<Item = (&'a El<Self::BaseRing>, usize)>, El<Self::BaseRing>: 'a
    {
        let x = self.indeterminate();
        let self_ring = RingRef::new(self);
        self_ring.sum(
            iter.map(|(c, i)| self.mul(self.from_ref(c), self_ring.pow(&x, i)))
        )
    }

    fn coefficient_at(&self, f: &Self::Element, i: usize) -> &El<Self::BaseRing>;

    fn degree(&self, f: &Self::Element) -> Option<usize>;
}

pub trait PolyRingStore: RingStore<Type: PolyRing> {

    delegate!{ fn indeterminate(&self) -> El<Self> }
    delegate!{ fn coefficient_at(&self, f: &El<Self>, i: usize) -> &El<<Self::Type as RingExtension>::BaseRing> }
    delegate!{ fn degree(&self, f: &El<Self>) -> Option<usize> }

    fn terms<'a>(&'a self, f: &'a El<Self>) -> PolyRingCoefficientIterator<'a, Self::Type> {
        self.get_ring().terms(f)
    }

    fn from_terms<'a, I>(&self, iter: I) -> El<Self>
        where I: Iterator<Item = (&'a El<<Self::Type as RingExtension>::BaseRing>, usize)>, 
        El<<Self::Type as RingExtension>::BaseRing>: 'a
    {
        self.get_ring().from_terms(iter)
    }
}