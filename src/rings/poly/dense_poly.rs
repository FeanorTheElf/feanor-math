use crate::algorithms::eea::poly_pid_fractionfield_gcd;
use crate::algorithms::convolution::*;
use crate::divisibility::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::pid::*;
use crate::field::Field;
use crate::rings::rational::RationalFieldBase;
use crate::ring::*;
use crate::algorithms;
use crate::rings::poly::*;
use crate::seq::{VectorView, VectorViewMut};

use std::alloc::{Allocator, Global};
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
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let x_plus_1 = P.add(P.indeterminate(), P.int_hom().map(1));
/// let binomial_coefficients = P.pow(x_plus_1, 10);
/// assert_eq!(10 * 9 * 8 * 7 * 6 / 120, *P.coefficient_at(&binomial_coefficients, 5));
/// ```
/// This ring has a [`CanIsoFromTo`] to [`sparse_poly::SparsePolyRingBase`].
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::poly::sparse_poly::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let P2 = SparsePolyRing::new(ZZ, "X");
/// let high_power_of_x = P.pow(P.indeterminate(), 10);
/// assert_el_eq!(&P2, &P2.pow(P2.indeterminate(), 10), &P.can_iso(&P2).unwrap().map(high_power_of_x));
/// ```
/// 
pub struct DensePolyRingBase<R: RingStore, A: Allocator + Clone = Global> {
    base_ring: R,
    unknown_name: &'static str,
    zero: El<R>,
    element_allocator: A
}

impl<R: RingStore + Clone, A: Allocator + Clone> Clone for DensePolyRingBase<R, A> {
    
    fn clone(&self) -> Self {
        DensePolyRingBase {
            base_ring: <R as Clone>::clone(&self.base_ring), 
            unknown_name: self.unknown_name, 
            zero: self.base_ring.zero() ,
            element_allocator: self.element_allocator.clone()
        }
    }
}

///
/// The univariate polynomial ring `R[X]`, with polynomials being stored as dense vectors of coefficients.
/// For details, see [`DensePolyRingBase`].
/// 
pub type DensePolyRing<R, A = Global> = RingValue<DensePolyRingBase<R, A>>;

impl<R: RingStore> DensePolyRing<R> {

    pub fn new(base_ring: R, unknown_name: &'static str) -> Self {
        Self::new_with(base_ring, unknown_name, Global)
    }
}

impl<R: RingStore, A: Allocator + Clone> DensePolyRing<R, A> {

    #[stability::unstable(feature = "enable")]
    pub fn new_with(base_ring: R, unknown_name: &'static str, element_allocator: A) -> Self {
        let zero = base_ring.zero();
        RingValue::from(DensePolyRingBase {
            base_ring, 
            unknown_name, 
            zero, 
            element_allocator
        })
    }
}

impl<R: RingStore, A: Allocator + Clone> DensePolyRingBase<R, A> {

    fn poly_div<F>(&self, lhs: &mut <Self as RingBase>::Element, rhs: &<Self as RingBase>::Element, mut left_div_lc: F) -> Option<<Self as RingBase>::Element>
        where F: FnMut(El<R>) -> Option<El<R>>
    {
        let lhs_val = std::mem::replace(lhs, self.zero());
        let (quo, rem) = algorithms::poly_div::poly_div(
            lhs_val, 
            rhs, 
            RingRef::new(self), 
            RingRef::new(self), 
            |x| left_div_lc(self.base_ring().clone_el(x)).ok_or(()),
            &self.base_ring().identity()
        ).ok()?;
        *lhs = rem;
        return Some(quo);
    }
}

///
/// An element of [`DensePolyRing`].
/// 
pub struct DensePolyRingEl<R: RingStore, A: Allocator + Clone = Global> {
    data: Vec<El<R>, A>
}

impl<R: RingStore, A: Allocator + Clone> RingBase for DensePolyRingBase<R, A> {
    
    type Element = DensePolyRingEl<R, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(val.data.len(), self.element_allocator.clone());
        data.extend((0..val.data.len()).map(|i| (self.base_ring.clone_el(&val.data.at(i)))));
        DensePolyRingEl { data }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            self.base_ring.add_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        for i in min(lhs.data.len(), rhs.data.len())..rhs.data.len() {
            lhs.data.push(self.base_ring().clone_el(&rhs.data[i]));
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            self.base_ring.sub_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        for i in min(lhs.data.len(), rhs.data.len())..rhs.data.len() {
            lhs.data.push(self.base_ring().negate(self.base_ring().clone_el(&rhs.data[i])));
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..lhs.data.len() {
            self.base_ring.negate_inplace(&mut lhs.data[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        DensePolyRingEl {
            data: Vec::new_in(self.element_allocator.clone())
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        let mut result = self.zero();
        result.data.push(self.base_ring().get_ring().from_int(value)); 
        return result;
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            if !self.base_ring.eq_el(&lhs.data[i], &rhs.data[i]) {
                return false;
            }
        }
        let longer = if lhs.data.len() > rhs.data.len() { lhs } else { rhs };
        for i in min(lhs.data.len(), rhs.data.len())..longer.data.len() {
            if !self.base_ring.is_zero(&longer.data[i]) {
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
        let mut result = Vec::with_capacity_in(lhs_len + rhs_len, self.element_allocator.clone());
        result.extend((0..(lhs_len + rhs_len)).map(|_| self.base_ring().zero()));
        STANDARD_CONVOLUTION.compute_convolution(
            &lhs.data[0..lhs_len], 
            &rhs.data[0..rhs_len],
            &mut result[..], 
            self.base_ring.get_ring()
        );
        return DensePolyRingEl {
            data: result
        };
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..lhs.data.len() {
            self.base_ring().int_hom().mul_assign_map(lhs.data.at_mut(i), rhs);
        }
    }
    
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)
    }
}

impl<R, A> PartialEq for DensePolyRingBase<R, A> 
    where R: RingStore, A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

///
/// Marker trait to signal that for this polynomial ring `P`, we want to use the
/// default implementation of the potential isomorphism `P <-> DensePolyRing` when
/// applicable.
/// 
/// This is currently necessary, since we want to provide a specialized implementation
/// of `DensePolyRingBase<R1, A1>: CanHomFrom<DensePolyRingBase<R2, A2>>`, but we cannot
/// currently specialize on types that still have generic parameters.
/// 
pub trait ImplGenericCanIsoFromToMarker: PolyRing {}

impl<R> ImplGenericCanIsoFromToMarker for sparse_poly::SparsePolyRingBase<R> 
    where R: RingStore
{}

impl<R, P, A> CanHomFrom<P> for DensePolyRingBase<R, A> 
    where R: RingStore, R::Type: CanHomFrom<<P::BaseRing as RingStore>::Type>, P: ImplGenericCanIsoFromToMarker, A: Allocator + Clone
{
    type Homomorphism = super::generic_impls::Homomorphism<P, DensePolyRingBase<R, A>>;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom(from, self)
    }

    fn map_in(&self, from: &P, el: P::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in(from, self, el, hom)
    }
}

impl<R1, A1, R2, A2> CanHomFrom<DensePolyRingBase<R1, A1>> for DensePolyRingBase<R2, A2> 
    where R1: RingStore, A1: Allocator + Clone, 
        R2: RingStore, A2: Allocator + Clone,
        R2::Type: CanHomFrom<R1::Type>
{
    type Homomorphism = <R2::Type as CanHomFrom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DensePolyRingBase<R1, A1>) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in_ref(&self, from: &DensePolyRingBase<R1, A1>, el: &DensePolyRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        RingRef::new(self).from_terms((0..el.data.len()).map(|i| (self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), &el.data[i], hom), i)))
    }

    fn map_in(&self, from: &DensePolyRingBase<R1, A1>, el: DensePolyRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)    
    }
}

impl<R1, A1, R2, A2> CanIsoFromTo<DensePolyRingBase<R1, A1>> for DensePolyRingBase<R2, A2> 
    where R1: RingStore, A1: Allocator + Clone, 
        R2: RingStore, A2: Allocator + Clone,
        R2::Type: CanIsoFromTo<R1::Type>
{
    type Isomorphism = <R2::Type as CanIsoFromTo<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DensePolyRingBase<R1, A1>) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &DensePolyRingBase<R1, A1>, el: DensePolyRingEl<R2, A2>, hom: &Self::Isomorphism) -> DensePolyRingEl<R1, A1> {
        RingRef::new(from).from_terms((0..el.data.len()).map(|i| (self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), hom), i)))
    }
}

impl<R, P, A> CanIsoFromTo<P> for DensePolyRingBase<R, A> 
    where R: RingStore, R::Type: CanIsoFromTo<<P::BaseRing as RingStore>::Type>, P: ImplGenericCanIsoFromToMarker, A: Allocator + Clone
{
    type Isomorphism = super::generic_impls::Isomorphism<P, Self>;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &P, el: Self::Element, iso: &Self::Isomorphism) -> P::Element {
        super::generic_impls::map_out(from, self, el, iso)
    }
}

impl<R: RingStore, A: Allocator + Clone> RingExtension for DensePolyRingBase<R, A> {
    
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data.push(x);
        return result;
    }
}

///
/// Iterator over all terms of an element of [`DensePolyRing`].
/// 
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

impl<R, A: Allocator + Clone> PolyRing for DensePolyRingBase<R, A> 
    where R: RingStore
{
    type TermsIterator<'a> = TermIterator<'a, R>
        where Self: 'a;

    fn indeterminate(&self) -> Self::Element {
        let mut result = self.zero();
        result.data.extend([self.base_ring().zero(), self.base_ring().one()].into_iter());
        return result;
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> TermIterator<'a, R> {
        TermIterator {
            iter: f.data.iter().enumerate(), 
            ring: self.base_ring()
        }
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, usize)>
    {
        for (c, i) in rhs {
            if lhs.data.len() <= i {
                lhs.data.resize_with(i + 1, || self.base_ring().zero());
            }
            self.base_ring().add_assign(&mut lhs.data[i], c);
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing> {
        if i < f.data.len() {
            return &f.data[i];
        } else {
            return &self.zero;
        }
    }

    fn degree(&self, f: &Self::Element) -> Option<usize> {
        for i in (0..f.data.len()).rev() {
            if !self.base_ring().is_zero(&f.data[i]) {
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

    fn evaluate<S, H>(&self, f: &Self::Element, value: &S::Element, hom: &H) -> S::Element
        where S: ?Sized + RingBase,
            H: Homomorphism<R::Type, S>
    {
        if self.is_zero(f) {
            return hom.codomain().zero();
        }
        let d = self.degree(f).unwrap();
        let mut current = hom.map_ref(self.coefficient_at(f, d));
        for i in (0..d).rev() {
            hom.codomain().mul_assign_ref(&mut current, value);
            hom.codomain().add_assign(&mut current, hom.map_ref(self.coefficient_at(f, i)));
        }
        return current;
    }
}

impl<R, A: Allocator + Clone> Domain for DensePolyRingBase<R, A> 
    where R: RingStore, R::Type: Domain
{}

impl<R, A: Allocator + Clone> DivisibilityRing for DensePolyRingBase<R, A> 
    where R: DivisibilityRingStore, R::Type: DivisibilityRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if let Some(d) = self.degree(rhs) {
            let lc = &rhs.data[d];
            let mut lhs_copy = self.clone_el(lhs);
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

trait ImplPrincipalIdealRing: Field {

    fn extended_ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> (El<P>, El<P>, El<P>)
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone;

    fn ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone;
}

impl<F: ?Sized + Field> ImplPrincipalIdealRing for F {

    default fn extended_ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> (El<P>, El<P>, El<P>)
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone
    {
        algorithms::eea::eea(poly_ring.clone_el(lhs), poly_ring.clone_el(rhs), poly_ring)
    }

    default fn ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone
    {
        <Self as ImplPrincipalIdealRing>::extended_ideal_gen::<P, R, A>(poly_ring, lhs, rhs).2
    }
}

impl<I: IntegerRingStore> ImplPrincipalIdealRing for RationalFieldBase<I>
    where I::Type: IntegerRing
{
    fn extended_ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> (El<P>, El<P>, El<P>)
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone
    {
        algorithms::eea::eea(poly_ring.clone_el(lhs), poly_ring.clone_el(rhs), poly_ring)
    }

    fn ideal_gen<P, R, A>(poly_ring: &P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: PolyRingStore<Type = DensePolyRingBase<R, A>>,
            R: RingStore<Type = Self>,
            A: Allocator + Clone
    {
        let QQ = poly_ring.base_ring();
        let ZZX = DensePolyRing::new(QQ.base_ring(), "X");
        let lhs_factor = poly_ring.terms(lhs).map(|(c, _)| c).fold(QQ.one(), |x, y| QQ.lcm(&x, y));
        let lhs = ZZX.from_terms(poly_ring.terms(lhs).map(|(c, d)| (QQ.mul_ref(c, &lhs_factor).0, d)));
        let rhs_factor = poly_ring.terms(rhs).map(|(c, _)| c).fold(QQ.one(), |x, y| QQ.lcm(&x, y));
        let rhs = ZZX.from_terms(poly_ring.terms(rhs).map(|(c, d)| (QQ.mul_ref(c, &rhs_factor).0, d)));
        return poly_ring.lifted_hom(&ZZX, QQ.inclusion()).map(poly_pid_fractionfield_gcd(&ZZX, &lhs, &rhs));
    }
}

impl<R, A: Allocator + Clone> PrincipalIdealRing for DensePolyRingBase<R, A>
    where R: RingStore, R::Type: Field
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        <R::Type as ImplPrincipalIdealRing>::extended_ideal_gen(&RingRef::new(self), lhs, rhs)
    }

    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        <R::Type as ImplPrincipalIdealRing>::ideal_gen(&RingRef::new(self), lhs, rhs)
    }
}

impl<R, A: Allocator + Clone> EuclideanRing for DensePolyRingBase<R, A> 
    where R: RingStore, R::Type: Field
{
    fn euclidean_div_rem(&self, mut lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        let lc_inv = self.base_ring.invert(&rhs.data[self.degree(rhs).unwrap()]).unwrap();
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
use crate::rings::zn::zn_static::{Zn, Fp};
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
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0), (base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 0), (base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0), (base_ring.int_hom().map(-1), 1)].into_iter())
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
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    crate::pid::generic_tests::test_euclidean_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_canonical_iso_axioms_different_base_ring() {
    let poly_ring1 = DensePolyRing::new(zn_big::Zn::new(StaticRing::<i128>::RING, 7), "X");
    let poly_ring2 = DensePolyRing::new(zn_64::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}

#[test]
fn test_canonical_iso_sparse_poly_ring() {
    let poly_ring1 = SparsePolyRing::new(zn_64::Zn::new(7), "X");
    let poly_ring2 = DensePolyRing::new(zn_64::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}