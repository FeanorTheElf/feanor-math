use std::hash::Hash;
use std::ops::{Index, RangeTo, RangeFrom};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::cmp::{min, max};

use crate::vector::subvector::{Subvector, SelfSubvectorView};
use crate::ring::*;
use crate::generic_cast::generic_cast;
use crate::vector::{VectorView, VectorViewMut};

///
/// This module contains [`ordered::MultivariatePolyRingImpl`], an implementation
/// of multivariate polynomials using a sparse representation.
/// 
pub mod ordered;

type MonomialExponent = u16;

///
/// Trait for rings that are multivariate polynomial rings in finitely many indeterminates
/// over a base ring, i.e. `R[X0, X1, X2, ..., XN]`.
/// 
/// Currently, the only implementation of such rings is [`ordered::MultivariatePolyRingImpl`],
/// which is stores all monomials in an ordered vector.
/// 
pub trait MultivariatePolyRing: RingExtension + SelfIso {

    type MonomialVector: VectorViewMut<MonomialExponent> + Clone;
    type TermsIterator<'a>: Iterator<Item = (&'a El<Self::BaseRing>, &'a Monomial<Self::MonomialVector>)>
        where Self: 'a;

    ///
    /// Returns the number of indeterminates, i.e. the transcendence degree
    /// of this ring over its base ring.
    /// 
    fn indeterminate_len(&self) -> usize;

    ///
    /// Returns the i-th indeterminate/variable/unknown as a ring element
    /// 
    fn indeterminate(&self, i: usize) -> Self::Element;
    
    ///
    /// Returns the given monomial as a ring element
    /// 
    fn monomial(&self, m: &Monomial<Self::MonomialVector>) -> Self::Element {
        RingRef::new(self).prod((0..m.len()).flat_map(|i| std::iter::repeat_with(move || self.indeterminate(i)).take(m[i] as usize)))
    }

    ///
    /// Returns all terms of the given polynomial. A term is a product
    /// `c * m` with a nonzero coefficient `c` (i.e. element of the base ring)
    /// and a monomial `m`.
    /// 
    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a>;

    ///
    /// Multiplies the given polynomial with a monomial.
    /// 
    fn mul_monomial(&self, el: &mut Self::Element, m: &Monomial<Self::MonomialVector>);
    
    ///
    /// Add-assigns to the given polynomial the polynomial implicitly given by the
    /// iterator over terms.
    /// 
    fn add_assign_from_terms<'a, I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, &'a Monomial<Self::MonomialVector>)>,
            Self: 'a
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.map(|(c, m)| self.mul(self.from(c), self.monomial(m)))
        ));
    }

    ///
    /// Returns the coefficient of the polynomial of the given monomial. If
    /// the monomial does not appear in the polynomial, this returns zero.
    /// 
    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Monomial<Self::MonomialVector>) -> &'a El<Self::BaseRing>;

    ///
    /// Returns the leading monomial of the given polynomial.
    /// 
    /// The leading monomial is the monomial with nonzero coefficient that is
    /// largest w.r.t. the given monomial ordering.
    /// 
    fn lm<'a, O>(&'a self, f: &'a Self::Element, order: O) -> Option<&'a Monomial<Self::MonomialVector>>
        where O: MonomialOrder;

    ///
    /// Replaces the given indeterminate in the given polynomial by the value `val`.
    /// The implementation is optimized using the facts that only one indeterminate is
    /// replaced, and the new value is in the same ring (it can be a polynomial however).
    /// 
    fn specialize(&self, f: &Self::Element, var: usize, val: &Self::Element) -> Self::Element {
        let mut parts = Vec::new();
        for (c, m) in self.terms(f) {
            while m[var] as usize >= parts.len() {
                parts.push(self.zero());
            }
            let mut new_m = m.clone();
            *new_m.exponents.at_mut(var) = 0;
            self.add_assign_from_terms(&mut parts[m[var] as usize], Some((self.base_ring().clone_el(c), &new_m)).into_iter());
        }
        let mut current = parts.pop().unwrap();
        while let Some(new) = parts.pop() {
            self.mul_assign_ref(&mut current, val);
            self.add_assign(&mut current, new);
        }
        return current;
    }
}

pub trait MultivariatePolyRingStore: RingStore
    where Self::Type: MultivariatePolyRing
{
    delegate!{ fn indeterminate_len(&self) -> usize }
    delegate!{ fn indeterminate(&self, i: usize) -> El<Self> }
    delegate!{ fn monomial(&self, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> El<Self> }
    delegate!{ fn mul_monomial(&self, el: &mut El<Self>, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> () }
    delegate!{ fn specialize(&self, f: &El<Self>, var: usize, val: &El<Self>) -> El<Self> }

    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> &'a El<<Self::Type as RingExtension>::BaseRing> {
        self.get_ring().coefficient_at(f, m)
    }

    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as MultivariatePolyRing>::TermsIterator<'a> {
        self.get_ring().terms(f)
    }

    fn from_terms<'a, I>(&self, iter: I) -> El<Self>
        where I: Iterator<Item = (El<<Self::Type as RingExtension>::BaseRing>, &'a Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>)>,
            Self: 'a
    {
        let mut result = self.zero();
        self.get_ring().add_assign_from_terms(&mut result, iter);
        return result;
    }

    fn lm<'a, O>(&'a self, f: &'a El<Self>, order: O) -> Option<&'a Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>>
        where O: MonomialOrder
    {
        self.get_ring().lm(f, order)
    }
}

impl<R> MultivariatePolyRingStore for R
    where R: RingStore,
        R::Type: MultivariatePolyRing
{}

static ZERO: MonomialExponent = 0;

///
/// A monomial, i.e. a power-product of indeterminates.
/// 
/// From a low-level point of view, this is just a wrapper around some
/// short sequence of nonnegative numbers. Note that monomials for the
/// same multivariate polynomial ring should always contain all variables,
/// i.e. contain (possibly trailing) zeros for variables that do not occur
/// in the monomial.
/// 
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Monomial<V: VectorView<MonomialExponent>> {
    exponents: V
}

impl<V: VectorView<MonomialExponent>> PartialEq<Monomial<V>> for Monomial<V> {

    fn eq(&self, other: &Monomial<V>) -> bool {
        self.exponents.len() == other.exponents.len() && (0..self.exponents.len()).all(|i| self.exponents.at(i) == other.exponents.at(i))
    }
}

impl<V: VectorView<MonomialExponent>> Hash for Monomial<V> {

    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.exponents.len());
        for e in self.exponents.iter() {
            state.write_u16(*e);
        }
    }
}

impl<V: VectorView<MonomialExponent>> Monomial<V> {

    pub fn from_vector_ref<'a>(vector: &'a V) -> &'a Self {
        unsafe { std::mem::transmute(vector) }
    }
    
    pub fn new(exponents: V) -> Self {
        Self { exponents }
    }

    pub fn deg(&self) -> MonomialExponent {
        self.exponents.iter().sum()
    }

    pub fn len(&self) -> usize {
        self.exponents.len()
    }

    pub fn unwrap(self) -> V {
        self.exponents
    }

    pub fn divides(&self, rhs: &Self) -> bool {
        assert_eq!(self.exponents.len(), rhs.exponents.len());
        (0..self.exponents.len()).all(|i| *self.exponents.at(i) <= *rhs.exponents.at(i))
    }

    pub fn is_coprime(&self, rhs: &Self) -> bool {
        assert_eq!(self.exponents.len(), rhs.exponents.len());
        (0..self.exponents.len()).all(|i| *self.exponents.at(i) == 0 || *rhs.exponents.at(i) == 0)
    }
}

impl<V: VectorViewMut<MonomialExponent>> Monomial<V> {

    pub fn gcd(mut self, rhs: &Self) -> Self {
        self.gcd_assign(rhs);
        self
    }

    pub fn lcm(mut self, rhs: &Self) -> Self {
        self.lcm_assign(rhs);
        self
    }

    pub fn mul(mut self, rhs: &Self) -> Self {
        self.mul_assign(rhs);
        self
    }

    pub fn div(mut self, rhs: &Self) -> Self {
        self.div_assign(rhs);
        self
    }

    pub fn gcd_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            *self.exponents.at_mut(i) = min(*self.exponents.at(i), *rhs.exponents.at(i));
        }
    }

    pub fn lcm_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            *self.exponents.at_mut(i) = max(*self.exponents.at(i), *rhs.exponents.at(i));
        }
    }

    pub fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            *self.exponents.at_mut(i) = *self.exponents.at(i) + *rhs.exponents.at(i);
        }
    }

    pub fn div_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            *self.exponents.at_mut(i) = *self.exponents.at(i) - *rhs.exponents.at(i);
        }
    }
}

impl<V: VectorViewMut<MonomialExponent> + Clone> Monomial<V> {

    ///
    /// Returns all monomials that divide this one, i.e. all sequences
    /// of same length such that `result[i] <= self[i]`.
    /// 
    pub fn dividing_monomials(self) -> DividingMonomialIter<V> {
        let mut start = self.clone();
        for i in 0..self.len() {
            *start.exponents.at_mut(i) = 0;
        }
        DividingMonomialIter { monomial: self, current: Some(start) }
    }
}

pub struct DividingMonomialIter<V: VectorViewMut<MonomialExponent>> {
    monomial: Monomial<V>,
    current: Option<Monomial<V>>
}

impl<V: VectorViewMut<MonomialExponent> + Clone> Iterator for DividingMonomialIter<V> {

    type Item = Monomial<V>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = &mut self.current {
            let result = current.clone();
            let mut refill = 0;
            for i in 1..self.monomial.len() {
                if current[i - 1] > 0 && current[i] < self.monomial[i] {
                    *current.exponents.at_mut(i - 1) -= 1;
                    *current.exponents.at_mut(i) += 1;
                    let mut j = 0;
                    while refill > 0 {
                        *current.exponents.at_mut(j) = min(refill, self.monomial[j]);
                        refill -= current[j];
                        j += 1;
                    }
                    return Some(result); 
                } else {
                    refill += current[i - 1];
                    *current.exponents.at_mut(i - 1) = 0;
                }
            }
            refill += current[self.monomial.len() - 1];
            *current.exponents.at_mut(self.monomial.len() - 1) = 0;

            // we did all combinations with the current number of elements, so increase element number by one
            refill += 1;
            let mut j = 0;

            while refill > 0 && j < self.monomial.len() {
                *current.exponents.at_mut(j) = min(refill, self.monomial[j]);
                refill -= current[j];
                j += 1;
            }
            if j == self.monomial.len() && refill != 0 {
                self.current = None;
            }
            return Some(result);
        } else {
            return None;
        }
    }
}

impl<V: VectorView<MonomialExponent>> Debug for Monomial<V> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.exponents.len() == 0 {
            return write!(f, "()");
        }
        write!(f, "(")?;
        for i in 0..(self.exponents.len() - 1) {
            write!(f, "{}, ", self.exponents.at(i))?;
        }
        write!(f, "{})", self.exponents.at(self.exponents.len() - 1))
    }
}

impl<V: VectorView<MonomialExponent>> Index<usize> for Monomial<V> {

    type Output = MonomialExponent;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.exponents.len() {
            &ZERO
        } else {
            self.exponents.at(index)
        }
    }
}

///
/// Trait for term orders, that is an ordering on all monomials with
/// a fixed number of variables subject to the constraints
///  - `mp < np` whenever `m < n`
///  - `m <= mp`
/// for all monomials `m, n, p`.
/// 
/// Monomial/Term orders are particularly important in the context of
/// Groebner basis (see e.g. [`crate::algorithms::f4::f4()`]).
/// 
pub trait MonomialOrder: Clone + Sized + 'static {

    fn is_graded(&self) -> bool;
    fn compare<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering;

    fn is_same<O: MonomialOrder>(&self, other: O) -> bool {
        assert!(std::mem::size_of::<Self>() == 0);
        match generic_cast::<_, Self>(other) {
            Some(_) => true,
            None => false
        }
    }
}

#[derive(Clone, Copy)]
pub struct FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{
    m: Monomial<V>,
    order: O
}

impl<V, O> FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{
    pub fn new(m: Monomial<V>, order: O) -> Self {
        Self { m, order }
    }

    pub fn into(self) -> Monomial<V> {
        self.m
    }

    pub fn order(&self) -> &O {
        &self.order
    }
}

impl<V, O> PartialEq for FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<V, O> Eq for FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{}

impl<V, O> PartialOrd for FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<V, O> Ord for FixedOrderMonomial<V, O>
    where V: VectorView<MonomialExponent>, O: MonomialOrder
{
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(self.order().is_same(other.order().clone()));
        self.order.compare(&self.m, &other.m)
    }
}

///
/// Graded reverse lexicographic order.
/// 
/// It is defined by first comparing the degree of monomials, and
/// in case of equality, reverse the result of a lexicographic comparison,
/// using reversed variable order.
/// 
/// # Example
/// 
/// The smallest example where this differs from the graded lexicographic order
/// is as follows.
/// ```
/// # use feanor_math::rings::multivariate::*;
/// 
/// let a = Monomial::new([1, 0, 1]);
/// let b = Monomial::new([0, 2, 0]);
/// assert_eq!(std::cmp::Ordering::Less, DegRevLex.compare(&a, &b));
/// assert_eq!(std::cmp::Ordering::Greater, Lex.compare(&a, &b));
/// ```
/// 
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DegRevLex;

///
/// Block-wise graded reverse lexicographic order.
/// 
/// This order is defined by first comparing the monomials induced by the
/// first variables via degrevlex, and in the case of equality compare the
/// monomials induced by the rest of the variables. Its main use is the fact
/// that it can serve as an elimination order.
/// 
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlockLexDegRevLex {
    larger_block: usize
}

impl BlockLexDegRevLex {

    pub fn new(larger_block: RangeTo<usize>, smaller_block: RangeFrom<usize>) -> Self {
        assert_eq!(larger_block.end, smaller_block.start);
        Self {
            larger_block: larger_block.end
        }
    }
}

impl MonomialOrder for BlockLexDegRevLex {

    fn is_graded(&self) -> bool {
        false
    }

    fn compare<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering {
        match DegRevLex.compare(
            &Monomial::new(Subvector::new(&lhs.exponents).subvector(..self.larger_block)), 
            &Monomial::new(Subvector::new(&rhs.exponents).subvector(..self.larger_block))
        ) {
            Ordering::Equal => DegRevLex.compare(
                &Monomial::new(Subvector::new(&lhs.exponents).subvector(self.larger_block..)), 
                &Monomial::new(Subvector::new(&rhs.exponents).subvector(self.larger_block..))
            ),
            ordering => ordering
        }
    }

    fn is_same<O: ?Sized + MonomialOrder>(&self, other: O) -> bool {
        match generic_cast::<_, Self>(other) {
            Some(other) => self.larger_block == other.larger_block,
            None => false
        }
    }
}

impl MonomialOrder for DegRevLex {

    fn is_graded(&self) -> bool {
        true
    }

    fn compare<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering {
        let lhs_deg = lhs.deg();
        let rhs_deg = rhs.deg();
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..max(lhs.len(), rhs.len())).rev() {
                if lhs[i] > rhs[i] {
                    return Ordering::Less
                } else if lhs[i] < rhs[i] {
                    return Ordering::Greater;
                }
            }
            return Ordering::Equal;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Lex;

impl MonomialOrder for Lex {

    fn is_graded(&self) -> bool {
        false
    }

    fn compare<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering {
        for i in 0..max(lhs.len(), rhs.len()) {
            if lhs[i] < rhs[i] {
                return Ordering::Less;
            } else if lhs[i] > rhs[i] {
                return Ordering::Greater;
            }
        }
        return Ordering::Equal;
    }
}

#[cfg(test)]
use crate::default_memory_provider;
#[cfg(test)]
use super::zn::zn_static::Zn;

#[test]
fn test_lex() {
    let mut monomials: Vec<Monomial<[u16; 3]>> = vec![
        Monomial::new([0, 0, 0]),
        Monomial::new([0, 0, 1]),
        Monomial::new([0, 0, 2]),
        Monomial::new([0, 1, 0]),
        Monomial::new([0, 1, 1]),
        Monomial::new([0, 2, 0]),
        Monomial::new([1, 0, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([1, 1, 0]),
        Monomial::new([2, 0, 0])
    ];
    monomials.sort_by(|l, r| Lex.compare(l, r).reverse());
    assert_eq!(vec![
        Monomial::new([2, 0, 0]),
        Monomial::new([1, 1, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([1, 0, 0]),
        Monomial::new([0, 2, 0]),
        Monomial::new([0, 1, 1]),
        Monomial::new([0, 1, 0]),
        Monomial::new([0, 0, 2]),
        Monomial::new([0, 0, 1]),
        Monomial::new([0, 0, 0])
    ], monomials);
}

#[test]
fn test_degrevlex() {
    let mut monomials: Vec<Monomial<[u16; 3]>> = vec![
        Monomial::new([0, 0, 0]),
        Monomial::new([0, 0, 1]),
        Monomial::new([0, 0, 2]),
        Monomial::new([0, 1, 0]),
        Monomial::new([0, 1, 1]),
        Monomial::new([0, 2, 0]),
        Monomial::new([1, 0, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([1, 1, 0]),
        Monomial::new([2, 0, 0])
    ];
    monomials.sort_by(|l, r| DegRevLex.compare(l, r).reverse());
    assert_eq!(vec![
        Monomial::new([2, 0, 0]),
        Monomial::new([1, 1, 0]),
        Monomial::new([0, 2, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([0, 1, 1]),
        Monomial::new([0, 0, 2]),
        Monomial::new([1, 0, 0]),
        Monomial::new([0, 1, 0]),
        Monomial::new([0, 0, 1]),
        Monomial::new([0, 0, 0])
    ], monomials);
}

#[test]
fn test_dividing_monomials() {
    let m = Monomial::new([2, 1, 3]);
    assert_eq!(vec![
        Monomial::new([0, 0, 0]), Monomial::new([1, 0, 0]), Monomial::new([0, 1, 0]), Monomial::new([0, 0, 1]),
        Monomial::new([2, 0, 0]), Monomial::new([1, 1, 0]), Monomial::new([1, 0, 1]), Monomial::new([0, 1, 1]), Monomial::new([0, 0, 2]),
        Monomial::new([2, 1, 0]), Monomial::new([2, 0, 1]), Monomial::new([1, 1, 1]), Monomial::new([1, 0, 2]), Monomial::new([0, 1, 2]), Monomial::new([0, 0, 3]),
        Monomial::new([2, 1, 1]), Monomial::new([2, 0, 2]), Monomial::new([1, 1, 2]), Monomial::new([1, 0, 3]), Monomial::new([0, 1, 3]),
        Monomial::new([2, 1, 2]), Monomial::new([2, 0, 3]), Monomial::new([1, 1, 3]), 
        Monomial::new([2, 1, 3])
    ], m.dividing_monomials().collect::<Vec<_>>());
}

#[test]
fn test_specialize() {
    let ring: ordered::MultivariatePolyRingImpl<_, _, _, 1> = ordered::MultivariatePolyRingImpl::new(Zn::<17>::RING, DegRevLex, default_memory_provider!());
    let f = ring.from_terms([(1, &Monomial::new([2])), (1, &Monomial::new([1]))].into_iter());
    let g = ring.from_terms([(1, &Monomial::new([2])), (16, &Monomial::new([1]))].into_iter());

    assert_el_eq!(&ring, &ring.add_ref_snd(ring.mul_ref(&g, &g), &g), &ring.specialize(&f, 0, &g));

    let ring: ordered::MultivariatePolyRingImpl<_, _, _, 2> = ordered::MultivariatePolyRingImpl::new(Zn::<17>::RING, DegRevLex, default_memory_provider!());
    let f = ring.from_terms([(1, &Monomial::new([2, 0])), (1, &Monomial::new([0, 1]))].into_iter());
    let g = ring.from_terms([(1, &Monomial::new([0, 2])), (16, &Monomial::new([0, 1]))].into_iter());

    assert_el_eq!(&ring, &ring.add(ring.mul_ref(&g, &g), ring.indeterminate(1)), &ring.specialize(&f, 0, &g));
}