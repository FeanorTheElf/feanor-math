use crate::ring::*;
pub mod dense_poly;

///
/// Trait for all rings that represent the polynomial ring `R[X]` with
/// any base ring R.
/// 
pub trait PolyRing: RingExtension + CanonicalIso<Self> {

    type TermsIterator<'a>: Iterator<Item = (&'a El<Self::BaseRing>, usize)>
        where Self: 'a;

    fn indeterminate(&self) -> Self::Element;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a>;
    
    fn from_terms<I>(&self, iter: I) -> Self::Element
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        let self_ring = RingRef::new(self);
        self_ring.sum(
            iter.map(|(c, i)| self.mul(self.from(c), self_ring.pow(self.indeterminate(), i)))
        )
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing>;

    fn degree(&self, f: &Self::Element) -> Option<usize>;
}

pub trait PolyRingStore: RingStore
    where Self::Type: PolyRing
{
    delegate!{ fn indeterminate(&self) -> El<Self> }
    delegate!{ fn degree(&self, f: &El<Self>) -> Option<usize> }

    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, i: usize) -> &'a El<<Self::Type as RingExtension>::BaseRing> {
        self.get_ring().coefficient_at(f, i)
    }

    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as PolyRing>::TermsIterator<'a> {
        self.get_ring().terms(f)
    }

    fn from_terms<I>(&self, iter: I) -> El<Self>
        where I: Iterator<Item = (El<<Self::Type as RingExtension>::BaseRing>, usize)>,
    {
        self.get_ring().from_terms(iter)
    }
}

impl<R: RingStore> PolyRingStore for R
    where R::Type: PolyRing
{}

#[cfg(test)]
pub fn generic_test_poly_ring_axioms<R: PolyRingStore, I: Iterator<Item = El<<R::Type as RingExtension>::BaseRing>>>(ring: R, interesting_base_ring_elements: I)
    where R::Type: PolyRing
{    
    let x = ring.indeterminate();
    let elements = interesting_base_ring_elements.collect::<Vec<_>>();
    
    // test linear independence of X
    for a in &elements {
        for b in &elements {
            for c in &elements {
                for d in &elements {
                    let a_bx = ring.add(ring.from_ref(a), ring.mul_ref_snd(ring.from_ref(b), &x));
                    let c_dx = ring.add(ring.from_ref(c), ring.mul_ref_snd(ring.from_ref(d), &x));
                    assert!(ring.eq_el(&a_bx, &c_dx) == (ring.base_ring().eq_el(a, c) && ring.base_ring().eq_el(b, d)));
                }
            }
        }
    }
    
    // elementwise addition follows trivially from the ring axioms

    // technically, convoluted multiplication follows from the ring axioms too, but test it anyway
    for a in &elements {
        for b in &elements {
            for c in &elements {
                for d in &elements {
                    let a_bx = ring.add(ring.from_ref(a), ring.mul_ref_snd(ring.from_ref(b), &x));
                    let c_dx = ring.add(ring.from_ref(c), ring.mul_ref_snd(ring.from_ref(d), &x));
                    let result = ring.sum([
                        ring.mul(ring.from_ref(a), ring.from_ref(c)),
                        ring.mul(ring.from_ref(a), ring.mul_ref_snd(ring.from_ref(d), &x)),
                        ring.mul(ring.from_ref(b), ring.mul_ref_snd(ring.from_ref(c), &x)),
                        ring.mul(ring.from_ref(b), ring.mul(ring.from_ref(d), ring.pow(ring.clone_el(&x), 2)))
                    ].into_iter());
                    assert!(ring.eq_el(&result, &ring.mul(a_bx, c_dx)));
                }
            }
        }
    }

    // test terms(), from_terms()
    for a in &elements {
        for b in &elements {
            for c in &elements {
                let f = ring.sum([
                    ring.from_ref(a),
                    ring.mul_ref_snd(ring.from_ref(b), &x),
                    ring.mul(ring.from_ref(c), ring.pow(ring.clone_el(&x), 3))
                ].into_iter());
                let actual = ring.from_terms([(ring.base_ring().clone_el(a), 0), (ring.base_ring().clone_el(c), 3), (ring.base_ring().clone_el(b), 1)].into_iter());
                assert!(ring.eq_el(&f, &actual));
                assert!(ring.eq_el(&f, &ring.from_terms(ring.terms(&f).map(|(c, i)| (ring.base_ring().clone_el(c), i)))));
            }
        }
    }
}