use crate::divisibility::*;
use crate::euclidean::*;
use crate::field::Field;
use crate::default_memory_provider;
use crate::mempool::{DefaultMemoryProvider, GrowableMemoryProvider};
use crate::vector::VectorViewMut;
use crate::ring::*;
use crate::algorithms;
use crate::rings::poly::*;

use std::cmp::min;

///
/// The univariate polynomial ring `R[X]`. Polynomials are stored as dense vectors of
/// coefficients, allocated by the given memory provider.
/// 
/// If most of the coefficients will be zero, consider using [`sparse_poly::SparsePolyRingBase`]
/// instead for improved performance.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// 
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let x_plus_1 = P.add(P.indeterminate(), P.from_int(1));
/// let binomial_coefficients = P.pow(x_plus_1, 10);
/// assert_eq!(10 * 9 * 8 * 7 * 6 / 120, *P.coefficient_at(&binomial_coefficients, 5));
/// ```
/// To create a ring with a custom memory provider, use
/// ```
/// # use feanor_math::default_memory_provider;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// 
/// let ZZ = StaticRing::<i32>::RING;
/// let P = RingValue::from(DensePolyRingBase::new(ZZ, "X", default_memory_provider!()));
/// ```
/// This ring has a [`CanonicalIso`] to [`sparse_poly::SparsePolyRingBase`].
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::poly::sparse_poly::*;
/// # use feanor_math::primitive_int::*;
/// 
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let P2 = SparsePolyRing::new(ZZ, "X");
/// let high_power_of_x = P.pow(P.indeterminate(), 10);
/// assert_el_eq!(&P2, &P2.pow(P2.indeterminate(), 10), &P.cast(&P2, high_power_of_x));
/// ```
/// 
pub struct DensePolyRingBase<R: RingStore, M: GrowableMemoryProvider<El<R>> = DefaultMemoryProvider> {
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

///
/// The univariate polynomial ring `R[X]`, with polynomials being stored as dense vectors of coefficients.
/// For details, see [`DensePolyRingBase`].
/// 
#[allow(type_alias_bounds)]
pub type DensePolyRing<R: RingStore, M: GrowableMemoryProvider<El<R>> = DefaultMemoryProvider> = RingValue<DensePolyRingBase<R, M>>;

impl<R: RingStore> DensePolyRing<R> {

    pub fn new(base_ring: R, unknown_name: &'static str) -> Self {
        Self::from(DensePolyRingBase::new(base_ring, unknown_name, default_memory_provider!()))
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

    fn poly_div<F>(&self, lhs: &mut M::Object, rhs: &M::Object, mut left_div_lc: F) -> Option<M::Object>
        where F: FnMut(El<R>) -> Option<El<R>>
    {
        let lhs_val = std::mem::replace(lhs, self.zero());
        let (quo, rem) = algorithms::poly_div::sparse_poly_div(
            lhs_val, 
            rhs, 
            RingRef::new(self), 
            RingRef::new(self), 
            |x| left_div_lc(self.base_ring().clone_el(x)).ok_or(())
        ).ok()?;
        *lhs = rem;
        return Some(quo);
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
        self.add_assign_ref(lhs, &rhs);
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
        super::generic_impls::dbg_poly(self, value, out, self.unknown_name)
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
            &self.base_ring,
            &self.memory_provider
        );
        return result;
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..lhs.len() {
            self.base_ring().mul_assign_int(lhs.at_mut(i), rhs);
        }
    }
}

impl<R, M> PartialEq for DensePolyRingBase<R, M> 
    where R: RingStore, M: GrowableMemoryProvider<El<R>>
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

pub trait CanonicalIsoToDensePolyRing: PolyRing {}

impl<R> CanonicalIsoToDensePolyRing for sparse_poly::SparsePolyRingBase<R> 
    where R: RingStore
{}

impl<R, P, M> CanonicalHom<P> for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: CanonicalHom<<P::BaseRing as RingStore>::Type>, P: CanonicalIsoToDensePolyRing, M: GrowableMemoryProvider<El<R>>
{
    type Homomorphism = super::generic_impls::GenericCanonicalHom<P, DensePolyRingBase<R, M>>;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        super::generic_impls::generic_has_canonical_hom(from, self)
    }

    fn map_in(&self, from: &P, el: P::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::generic_map_in(from, self, el, hom)
    }
}

impl<R1, M1, R2, M2> CanonicalHom<DensePolyRingBase<R1, M1>> for DensePolyRingBase<R2, M2> 
    where R1: RingStore, M1: GrowableMemoryProvider<El<R1>>, 
        R2: RingStore, M2: GrowableMemoryProvider<El<R2>>,
        R2::Type: CanonicalHom<R1::Type>
{
    type Homomorphism = <R2::Type as CanonicalHom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DensePolyRingBase<R1, M1>) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in_ref(&self, from: &DensePolyRingBase<R1, M1>, el: &M1::Object, hom: &Self::Homomorphism) -> Self::Element {
        self.memory_provider.get_new_init(el.len(), |i| self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), &el[i], hom))
    }

    fn map_in(&self, from: &DensePolyRingBase<R1, M1>, el: M1::Object, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)    
    }
}

impl<R1, M1, R2, M2> CanonicalIso<DensePolyRingBase<R1, M1>> for DensePolyRingBase<R2, M2> 
    where R1: RingStore, M1: GrowableMemoryProvider<El<R1>>, 
        R2: RingStore, M2: GrowableMemoryProvider<El<R2>>,
        R2::Type: CanonicalIso<R1::Type>
{
    type Isomorphism = <R2::Type as CanonicalIso<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DensePolyRingBase<R1, M1>) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &DensePolyRingBase<R1, M1>, el: M2::Object, hom: &Self::Isomorphism) -> M1::Object {
        from.memory_provider.get_new_init(el.len(), |i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el[i]), hom))
    }
}

impl<R, P, M> CanonicalIso<P> for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: CanonicalIso<<P::BaseRing as RingStore>::Type>, P: CanonicalIsoToDensePolyRing, M: GrowableMemoryProvider<El<R>>
{
    type Isomorphism = super::generic_impls::GenericCanonicalIso<P, Self>;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &P, el: Self::Element, iso: &Self::Isomorphism) -> P::Element {
        super::generic_impls::generic_map_out(from, self, el, iso)
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

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        for (c, i) in rhs {
            self.grow(lhs, i + 1);
            self.base_ring().add_assign(&mut lhs[i], c);
        }
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

    fn div_rem_monic(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(self.base_ring().is_one(self.coefficient_at(rhs, self.degree(rhs).unwrap())));
        let quo = self.poly_div(&mut lhs, rhs, |x| Some(x)).unwrap();
        return (quo, lhs);
    }
}

impl<R, M: GrowableMemoryProvider<El<R>>> DivisibilityRing for DensePolyRingBase<R, M> 
    where R: DivisibilityRingStore, R::Type: DivisibilityRing + CanonicalIso<R::Type>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if let Some(d) = self.degree(rhs) {
            let lc = &rhs[d];
            let mut lhs_copy = self.memory_provider.get_new_init(lhs.len(), |i| self.base_ring.clone_el(&lhs[i]));
            let quo = self.poly_div(&mut lhs_copy, rhs, |x| self.base_ring().checked_left_div(&x, lc))?;
            if self.is_zero(&lhs_copy) {
                Some(quo)
            } else {
                None
            }
        } else if self.is_zero(lhs) {
            Some(self.zero())
        } else {
            None
        }
    }
}

impl<R, M: GrowableMemoryProvider<El<R>>> EuclideanRing for DensePolyRingBase<R, M> 
    where R: RingStore, R::Type: Field + CanonicalIso<R::Type>
{
    fn euclidean_div_rem(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        let lc_inv = self.base_ring.invert(&rhs[self.degree(rhs).unwrap()]).unwrap();
        let quo = self.poly_div(&mut lhs, rhs, |x| Some(self.base_ring().mul_ref_snd(x, &lc_inv))).unwrap();
        return (quo, lhs);
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        return Some(self.degree(val).map(|x| x + 1).unwrap_or(0));
    }
}

#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use super::sparse_poly::SparsePolyRing;

#[cfg(test)]
fn edge_case_elements<P: PolyRingStore>(poly_ring: P) -> impl Iterator<Item = El<P>>
    where P::Type: PolyRing
{
    let base_ring = poly_ring.base_ring();
    vec![ 
        poly_ring.from_terms([].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(1), 0), (base_ring.from_int(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(-1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(-1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(-1), 0), (base_ring.from_int(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.from_int(1), 0), (base_ring.from_int(-1), 1)].into_iter())
    ].into_iter()
}

#[test]
fn test_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    crate::ring::generic_tests::test_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_poly_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    super::generic_tests::test_poly_ring_axioms(poly_ring, Zn::<7>::RING.elements());
}

#[test]
fn test_divisibility_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    crate::divisibility::generic_tests::test_divisibility_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_euclidean_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    generic_test_euclidean_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_canonical_iso_axioms_different_base_ring() {
    let poly_ring1 = DensePolyRing::new(zn_barett::Zn::new(StaticRing::<i128>::RING, 7), "X");
    let poly_ring2 = DensePolyRing::new(zn_42::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}

#[test]
fn test_canonical_iso_sparse_poly_ring() {
    let poly_ring1 = SparsePolyRing::new(zn_42::Zn::new(7), "X");
    let poly_ring2 = DensePolyRing::new(zn_42::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}