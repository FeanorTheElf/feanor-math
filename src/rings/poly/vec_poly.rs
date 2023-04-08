use crate::{ring::*, algorithms};
use crate::rings::poly::*;

use std::cmp::min;

pub struct VecPolyRing<R: RingStore> {
    base_ring: R,
    unknown_name: &'static str,
    zero: El<R>
}

impl<R: RingStore> VecPolyRing<R> {

    fn grow(&self, vector: &mut Vec<El<R>>, size: usize) {
        if vector.len() < size {
            vector.resize(size, self.base_ring.zero());
        }
    }
}

impl<R: RingStore> RingBase for VecPolyRing<R> {
    
    type Element = Vec<El<R>>;

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.grow(lhs, rhs.len());
        for i in 0..rhs.len() {
            self.base_ring.add_assign_ref(&mut lhs[i], &rhs[i])
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.grow(lhs, rhs.len());
        for (i, x) in rhs.into_iter().enumerate() {
            self.base_ring.add_assign(&mut lhs[i], x)
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
        vec![]
    }
    
    fn from_z(&self, value: i32) -> Self::Element {
        vec![ self.base_ring.from_z(value) ]
    }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        for i in 0..min(lhs.len(), rhs.len()) {
            if !self.base_ring.eq(&lhs[i], &rhs[i]) {
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
        }
        while let Some((c, i)) = terms.next() {
            write!(out, " + ")?;
            self.base_ring.get_ring().dbg(c, out)?;
            print_unknown(i, out)?;
        }
        return Ok(());
    }

    fn square(&self, value: Self::Element) -> Self::Element {
        self.mul_ref(&value, &value)
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = Vec::new();
        let lhs_len = self.degree(lhs).unwrap_or(0) + 1;
        let rhs_len = self.degree(rhs).unwrap_or(0) + 1;
        self.grow(&mut result, lhs_len + rhs_len);
        algorithms::conv_mul::add_assign_convoluted_mul(
            &mut result[..], 
            &lhs[0..lhs_len], 
            &rhs[0..rhs_len], 
            &self.base_ring
        );
        return result;
    }
}

impl<R, P> CanonicalHom<P> for VecPolyRing<R> 
    where R: RingStore, R::Type: CanonicalHom<<P::BaseRing as RingStore>::Type>, P: PolyRing
{
    type Homomorphism = <R::Type as CanonicalHom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &P, el: P::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from_terms(from.terms(&el).map(|(c, i)| (self.base_ring().get_ring().map_in(from.base_ring().get_ring(), c.clone(), hom), i)))
    }
}

impl<R, P> CanonicalIso<P> for VecPolyRing<R> 
    where R: RingStore, R::Type: CanonicalIso<<P::BaseRing as RingStore>::Type>, P: PolyRing
{
    type Isomorphism = <R::Type as CanonicalIso<<P::BaseRing as RingStore>::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &P, el: Self::Element, hom: &Self::Isomorphism) -> P::Element {
        from.from_terms(self.terms(&el).map(|(c, i)| (self.base_ring().get_ring().map_out(from.base_ring().get_ring(), c.clone(), hom), i)))
    }
}

impl<R: RingStore> RingExtension for VecPolyRing<R> {
    
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        vec![ x ]
    }
}

impl<R> PolyRing for VecPolyRing<R> 
    where R: RingStore, R::Type: CanonicalIso<R::Type>
{
    type IteratorState = usize;

    fn indeterminate(&self) -> Self::Element {
        vec![ self.base_ring().zero(), self.base_ring().one() ]
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> PolyRingCoefficientIterator<'a, Self> {
        PolyRingCoefficientIterator::new(self, f, 0)
    }

    fn coefficient_iterator_next<'a>(iter: &mut PolyRingCoefficientIterator<'a, Self>) -> Option<(&'a El<Self::BaseRing>, usize)> {
        while iter.ring.base_ring().is_zero(&iter.element[iter.state]) && iter.state < iter.element.len() {
            iter.state += 1;
        }
        if iter.state == iter.element.len() {
            return None;
        } else {
            iter.state += 1;
            return Some((&iter.element[iter.state - 1], iter.state - 1));
        }
    }
    
    fn from_terms<I>(&self, iter: I) -> Self::Element
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        let mut result = Vec::new();
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