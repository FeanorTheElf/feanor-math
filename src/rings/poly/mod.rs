use crate::ring::*;

pub mod dense_poly;
pub mod sparse_poly;

///
/// Trait for all rings that represent the polynomial ring `R[X]` with
/// any base ring R.
/// 
pub trait PolyRing: RingExtension + SelfIso {

    type TermsIterator<'a>: Iterator<Item = (&'a El<Self::BaseRing>, usize)>
        where Self: 'a;

    fn indeterminate(&self) -> Self::Element;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a>;
    
    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.map(|(c, i)| self.mul(self.from(c), self_ring.pow(self.indeterminate(), i)))
        ));
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing>;

    fn degree(&self, f: &Self::Element) -> Option<usize>;

    fn div_rem_monic(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);
}

pub trait PolyRingStore: RingStore
    where Self::Type: PolyRing
{
    delegate!{ fn indeterminate(&self) -> El<Self> }
    delegate!{ fn degree(&self, f: &El<Self>) -> Option<usize> }
    delegate!{ fn div_rem_monic(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }

    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, i: usize) -> &'a El<<Self::Type as RingExtension>::BaseRing> {
        self.get_ring().coefficient_at(f, i)
    }

    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as PolyRing>::TermsIterator<'a> {
        self.get_ring().terms(f)
    }

    fn from_terms<I>(&self, iter: I) -> El<Self>
        where I: Iterator<Item = (El<<Self::Type as RingExtension>::BaseRing>, usize)>,
    {
        let mut result = self.zero();
        self.get_ring().add_assign_from_terms(&mut result, iter);
        return result;
    }

    fn lc<'a>(&'a self, f: &'a El<Self>) -> Option<&'a El<<Self::Type as RingExtension>::BaseRing>> {
        Some(self.coefficient_at(f, self.degree(f)?))
    }
}

impl<R: RingStore> PolyRingStore for R
    where R::Type: PolyRing
{}

pub mod generic_impls {
    use crate::ring::*;
    use super::PolyRing;

    #[allow(type_alias_bounds)]
    pub type GenericCanonicalHom<P1: PolyRing, P2: PolyRing> = <<P2::BaseRing as RingStore>::Type as CanonicalHom<<P1::BaseRing as RingStore>::Type>>::Homomorphism;

    pub fn generic_map_in<P1: PolyRing, P2: PolyRing>(from: &P1, to: &P2, el: P1::Element, hom: &GenericCanonicalHom<P1, P2>) -> P2::Element
        where <P2::BaseRing as RingStore>::Type: CanonicalHom<<P1::BaseRing as RingStore>::Type>
    {
        let mut result = to.zero();
        to.add_assign_from_terms(&mut result, from.terms(&el).map(|(c, i)| (to.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(c), hom), i)));
        return result;
    }

    #[allow(type_alias_bounds)]
    pub type GenericCanonicalIso<P1: PolyRing, P2: PolyRing> = <<P2::BaseRing as RingStore>::Type as CanonicalIso<<P1::BaseRing as RingStore>::Type>>::Isomorphism;

    pub fn generic_map_out<P1: PolyRing, P2: PolyRing>(from: &P1, to: &P2, el: P2::Element, iso: &GenericCanonicalIso<P1, P2>) -> P1::Element
        where <P2::BaseRing as RingStore>::Type: CanonicalIso<<P1::BaseRing as RingStore>::Type>
    {
        let mut result = from.zero();
        from.add_assign_from_terms(&mut result, to.terms(&el).map(|(c, i)| (to.base_ring().get_ring().map_out(from.base_ring().get_ring(), to.base_ring().clone_el(c), iso), i)));
        return result;
    }

    pub fn dbg_poly<P: PolyRing>(ring: &P, el: &P::Element, out: &mut std::fmt::Formatter, unknown_name: &str) -> std::fmt::Result {
        let mut terms = ring.terms(el);
        let print_unknown = |i: usize, out: &mut std::fmt::Formatter| {
            if i == 0 {
                // print nothing
                Ok(())
            } else if i == 1 {
                write!(out, "{}", unknown_name)
            } else {
                write!(out, "{}^{}", unknown_name, i)
            }
        };
        if let Some((c, i)) = terms.next() {
            ring.base_ring().get_ring().dbg(c, out)?;
            print_unknown(i, out)?;
        } else {
            write!(out, "0")?;
        }
        while let Some((c, i)) = terms.next() {
            write!(out, " + ")?;
            ring.base_ring().get_ring().dbg(c, out)?;
            print_unknown(i, out)?;
        }
        return Ok(());
    }
}

#[cfg(any(test, feature = "generic_tests"))]
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
                    let result = <_ as RingStore>::sum(&ring, [
                        ring.mul(ring.from_ref(a), ring.from_ref(c)),
                        ring.mul(ring.from_ref(a), ring.mul_ref_snd(ring.from_ref(d), &x)),
                        ring.mul(ring.from_ref(b), ring.mul_ref_snd(ring.from_ref(c), &x)),
                        ring.mul(ring.from_ref(b), ring.mul(ring.from_ref(d), ring.pow(ring.clone_el(&x), 2)))
                    ].into_iter());
                    assert_el_eq!(&ring, &result, &ring.mul(a_bx, c_dx));
                }
            }
        }
    }

    // test terms(), from_terms()
    for a in &elements {
        for b in &elements {
            for c in &elements {
                let f = <_ as RingStore>::sum(&ring, [
                    ring.from_ref(a),
                    ring.mul_ref_snd(ring.from_ref(b), &x),
                    ring.mul(ring.from_ref(c), ring.pow(ring.clone_el(&x), 3))
                ].into_iter());
                let actual = ring.from_terms([(ring.base_ring().clone_el(a), 0), (ring.base_ring().clone_el(c), 3), (ring.base_ring().clone_el(b), 1)].into_iter());
                assert_el_eq!(&ring, &f, &actual);
                assert_el_eq!(&ring, &f, &ring.from_terms(ring.terms(&f).map(|(c, i)| (ring.base_ring().clone_el(c), i))));
            }
        }
    }

    // test div_rem_monic()
    for a in &elements {
        for b in &elements {
            for c in &elements {
                let f = ring.from_terms([(ring.base_ring().clone_el(a), 0), (ring.base_ring().clone_el(b), 3)].into_iter());
                let g = ring.from_terms([(ring.base_ring().negate(ring.base_ring().clone_el(c)), 0), (ring.base_ring().one(), 1)].into_iter());

                let (quo, rem) = ring.div_rem_monic(ring.clone_el(&f), &g);
                assert_el_eq!(
                    &ring,
                    &ring.from_terms([(ring.base_ring().add_ref_fst(a, ring.base_ring().mul_ref_fst(b, ring.base_ring().pow(ring.base_ring().clone_el(c), 3))), 0)].into_iter()),
                    &rem
                );
                assert_el_eq!(
                    &ring,
                    &f,
                    &ring.add(rem, ring.mul(quo, g))
                );
            }
        }
    }
}