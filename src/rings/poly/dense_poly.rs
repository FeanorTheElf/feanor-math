use crate::mempool::{AllocatingMemoryProvider, GrowableMemoryProvider};
use crate::vector::VectorViewMut;
use crate::{ring::*, algorithms};
use crate::rings::poly::*;

use std::cmp::min;

pub struct DensePolyRingBase<R: RingStore, M: GrowableMemoryProvider<El<R>> = AllocatingMemoryProvider> {
    base_ring: R,
    unknown_name: &'static str,
    zero: El<R>,
    memory_provider: M
}

impl<R: RingStore + Clone, M: GrowableMemoryProvider<El<R>> + Clone> Clone for DensePolyRingBase<R, M> {
    
    fn clone(&self) -> Self {
        DensePolyRingBase {
            base_ring: <R as Clone>::clone(&self.base_ring), 
            unknown_name: self.unknown_name, 
            zero: self.base_ring.zero() ,
            memory_provider: self.memory_provider.clone()
        }
    }
}

#[allow(type_alias_bounds)]
pub type DensePolyRing<R: RingStore, M: GrowableMemoryProvider<El<R>> = AllocatingMemoryProvider> = RingValue<DensePolyRingBase<R, M>>;

impl<R: RingStore> DensePolyRing<R> {

    pub fn new(base_ring: R, unknown_name: &'static str) -> Self {
        Self::from(DensePolyRingBase::new(base_ring, unknown_name, AllocatingMemoryProvider::default()))
    }
}

impl<R: RingStore, M: GrowableMemoryProvider<El<R>>> DensePolyRingBase<R, M> {

    pub fn new(base_ring: R, unknown_name: &'static str, memory_provider: M) -> Self {
        let zero = base_ring.zero();
        DensePolyRingBase { base_ring, unknown_name, zero, memory_provider }
    }

    fn grow(&self, vector: &mut M::Object, size: usize) {
        if vector.len() < size {
           self.memory_provider.grow_init(vector, size, |_| self.base_ring.zero());
        }
    }
}

impl<R: RingStore, M: GrowableMemoryProvider<El<R>>> RingBase for DensePolyRingBase<R, M> {
    
    type Element = M::Object;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        self.memory_provider.get_new_init(self.degree(val).map(|d| d + 1).unwrap_or(0), |i| self.base_ring.clone_el(&val[i]))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.grow(lhs, rhs.len());
        for i in 0..rhs.len() {
            self.base_ring.add_assign_ref(&mut lhs[i], &rhs[i])
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.grow(lhs, rhs.len());
        for (i, x) in rhs.into_iter().enumerate() {
            self.base_ring.add_assign_ref(&mut lhs[i], x)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.grow(lhs, rhs.len());
        for i in 0..rhs.len() {
            self.base_ring.sub_assign_ref(&mut lhs[i], &rhs[i])
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..lhs.len() {
            self.base_ring.negate_inplace(&mut lhs[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        self.memory_provider.get_new_init(0, |_| self.base_ring.zero())
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.memory_provider.get_new_init(1, |_| self.base_ring.from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        for i in 0..min(lhs.len(), rhs.len()) {
            if !self.base_ring.eq_el(&lhs[i], &rhs[i]) {
                return false;
            }
        }
        let longer = if lhs.len() > rhs.len() { lhs } else { rhs };
        for i in min(lhs.len(), rhs.len())..longer.len() {
            if !self.base_ring.is_zero(&longer[i]) {
                return false;
            }
        }
        return true;
    }

    fn is_commutative(&self) -> bool {
        self.base_ring.is_commutative()
    }

    fn is_noetherian(&self) -> bool {
        // by Hilbert's basis theorem
        self.base_ring.is_noetherian()
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let mut terms = self.terms(value);
        let print_unknown = |i: usize, out: &mut std::fmt::Formatter| {
            if i == 0 {
                // print nothing
                Ok(())
            } else if i == 1 {
                write!(out, "{}", self.unknown_name)
            } else {
                write!(out, "{}^{}", self.unknown_name, i)
            }
        };
        if let Some((c, i)) = terms.next() {
            self.base_ring.get_ring().dbg(c, out)?;
            print_unknown(i, out)?;
        } else {
            write!(out, "0")?;
        }
        while let Some((c, i)) = terms.next() {
            write!(out, " + ")?;
            self.base_ring.get_ring().dbg(c, out)?;
            print_unknown(i, out)?;
        }
        return Ok(());
    }

    fn square(&self, value: &mut Self::Element) {
        *value = self.mul_ref(&value, &value);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let lhs_len = self.degree(lhs).map(|i| i + 1).unwrap_or(0);
        let rhs_len = self.degree(rhs).map(|i| i + 1).unwrap_or(0);
        let mut result = self.memory_provider.get_new_init(lhs_len + rhs_len, |_| self.base_ring.zero());
        algorithms::conv_mul::add_assign_convoluted_mul(
            &mut result[..], 
            &lhs[0..lhs_len], 
            &rhs[0..rhs_len], 
            &self.base_ring
        );
        return result;
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..lhs.len() {
            self.base_ring().mul_assign_int(lhs.at_mut(i), rhs);
        }
    }
}

impl<R, P, M> CanonicalHom<P> for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: CanonicalHom<<P::BaseRing as RingStore>::Type>, P: PolyRing, M: GrowableMemoryProvider<El<R>>
{
    type Homomorphism = <R::Type as CanonicalHom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &P, el: P::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from_terms(from.terms(&el).map(|(c, i)| (self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(c), hom), i)))
    }
}

impl<R, P, M> CanonicalIso<P> for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: CanonicalIso<<P::BaseRing as RingStore>::Type>, P: PolyRing, M: GrowableMemoryProvider<El<R>>
{
    type Isomorphism = <R::Type as CanonicalIso<<P::BaseRing as RingStore>::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &P, el: Self::Element, hom: &Self::Isomorphism) -> P::Element {
        from.from_terms(self.terms(&el).map(|(c, i)| (self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(c), hom), i)))
    }
}

impl<R: RingStore, M: GrowableMemoryProvider<El<R>>> RingExtension for DensePolyRingBase<R, M> {
    
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut value = Some(x);
        self.memory_provider.get_new_init(1, |_| std::mem::replace(&mut value, None).unwrap())
    }
}

pub struct TermIterator<'a, R>
    where R: RingStore
{
    iter: std::iter::Enumerate<std::slice::Iter<'a, El<R>>>,
    ring: &'a R
}

impl<'a, R> Clone for TermIterator<'a, R>
    where R: RingStore
{
    fn clone(&self) -> Self {
        TermIterator {
            iter: self.iter.clone(),
            ring: self.ring
        }
    }
}

impl<'a, R> Iterator for TermIterator<'a, R>
    where R: RingStore
{
    type Item = (&'a El<R>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, c)) = self.iter.next() {
            if !self.ring.is_zero(c) {
                return Some((c, i));
            }
        }
        return None;
    }
}

impl<R, M: GrowableMemoryProvider<El<R>>> PolyRing for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: CanonicalIso<R::Type>
{
    type TermsIterator<'a> = TermIterator<'a, R>
        where Self: 'a;

    fn indeterminate(&self) -> Self::Element {
        self.memory_provider.get_new_init(2, |i| if i == 0 { self.base_ring().zero() } else { self.base_ring().one() })
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> TermIterator<'a, R> {
        TermIterator {
            iter: f.iter().enumerate(), 
            ring: self.base_ring()
        }
    }

    fn from_terms<I>(&self, iter: I) -> Self::Element
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        let mut result = self.memory_provider.get_new_init(iter.size_hint().0, |_| self.base_ring.zero());
        for (c, i) in iter {
            self.grow(&mut result, i + 1);
            result[i] = c;
        }
        return result;
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing> {
        if i < f.len() {
            return &f[i];
        } else {
            return &self.zero;
        }
    }

    fn degree(&self, f: &Self::Element) -> Option<usize> {
        for i in (0..f.len()).rev() {
            if !self.base_ring().is_zero(&f[i]) {
                return Some(i);
            }
        }
        return None;
    }
}

#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[cfg(test)]
fn edge_case_elements(poly_ring: &DensePolyRing<Zn<7>>) -> impl Iterator<Item = El<DensePolyRing<Zn<7>>>> {
    vec![ 
        poly_ring.from_terms([].into_iter()),
        poly_ring.from_terms([(1, 0)].into_iter()),
        poly_ring.from_terms([(1, 1)].into_iter()),
        poly_ring.from_terms([(1, 0), (1, 1)].into_iter()),
        poly_ring.from_terms([(6, 0)].into_iter()),
        poly_ring.from_terms([(6, 1)].into_iter()),
        poly_ring.from_terms([(6, 0), (1, 1)].into_iter()),
        poly_ring.from_terms([(1, 0), (6, 1)].into_iter())
    ].into_iter()
}

#[test]
fn test_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    generic_test_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_poly_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    generic_test_poly_ring_axioms(poly_ring, Zn::<7>::RING.elements());
}