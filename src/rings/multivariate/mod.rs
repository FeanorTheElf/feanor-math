use std::hash::Hash;
use std::ops::Index;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::cmp::{min, max};

use crate::{ring::*, type_eq};
use crate::vector::{VectorView, VectorViewMut};

pub mod vec_based;

type MonomialExponent = u16;

pub trait MultivariatePolyRing: RingExtension + SelfIso {

    type MonomialVector: VectorViewMut<MonomialExponent> + Clone;
    type TermsIterator<'a>: Iterator<Item = (&'a El<Self::BaseRing>, &'a Monomial<Self::MonomialVector>)>
        where Self: 'a;

    fn indeterminate_len(&self) -> usize;

    fn indeterminate(&self, i: usize) -> Self::Element;
        
    fn monomial(&self, m: &Monomial<Self::MonomialVector>) -> Self::Element {
        RingRef::new(self).prod((0..m.len()).flat_map(|i| std::iter::repeat_with(move || self.indeterminate(i)).take(m[i] as usize)))
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a>;

    fn mul_monomial(&self, el: &mut Self::Element, m: &Monomial<Self::MonomialVector>);
    
    fn add_assign_from_terms<'a, I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, &'a Monomial<Self::MonomialVector>)>,
            Self: 'a
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.map(|(c, m)| self.mul(self.from(c), self.monomial(m)))
        ));
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Monomial<Self::MonomialVector>) -> &'a El<Self::BaseRing>;

    fn lm<'a, O>(&'a self, f: &'a Self::Element, order: O) -> Option<&'a Monomial<Self::MonomialVector>>
        where O: MonomialOrder;
}

pub trait MultivariatePolyRingStore: RingStore
    where Self::Type: MultivariatePolyRing
{
    delegate!{ fn indeterminate_len(&self) -> usize }
    delegate!{ fn indeterminate(&self, i: usize) -> El<Self> }
    delegate!{ fn monomial(&self, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> El<Self> }
    delegate!{ fn mul_monomial(&self, el: &mut El<Self>, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> () }

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

    pub fn is_coprime(&self, rhs: &Self) -> bool {
        assert_eq!(self.exponents.len(), rhs.exponents.len());
        (0..self.exponents.len()).all(|i| *self.exponents.at(i) == 0 || *rhs.exponents.at(i) == 0)
    }
}

impl<V: VectorViewMut<MonomialExponent>> Monomial<V> {

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
            refill += 1;let mut j = 0;

            while refill > 0 && j < self.monomial.len() {
                *current.exponents.at_mut(j) = min(refill, self.monomial[j]);
                refill -= current[j];
                j += 1;
            }
            if j == self.monomial.len() {
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

pub trait MonomialOrder {

    fn is_graded(&self) -> bool;
    fn cmp<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering;
    fn is_same<O: ?Sized + MonomialOrder>(&self, other: &O) -> bool;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DegRevLex;

impl MonomialOrder for DegRevLex {

    fn is_graded(&self) -> bool {
        true
    }

    fn cmp<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering {
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

    fn is_same<O: ?Sized + MonomialOrder>(&self, _: &O) -> bool {
        type_eq::<Self, O>()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Lex;

impl MonomialOrder for Lex {

    fn is_graded(&self) -> bool {
        false
    }

    fn cmp<V: VectorView<MonomialExponent>>(&self, lhs: &Monomial<V>, rhs: &Monomial<V>) -> Ordering {
        for i in 0..max(lhs.len(), rhs.len()) {
            if lhs[i] < rhs[i] {
                return Ordering::Less;
            } else if lhs[i] > rhs[i] {
                return Ordering::Greater;
            }
        }
        return Ordering::Equal;
    }

    fn is_same<O: ?Sized + MonomialOrder>(&self, _: &O) -> bool {
        type_eq::<Self, O>()
    }
}

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
    monomials.sort_by(|l, r| Lex.cmp(l, r).reverse());
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
    monomials.sort_by(|l, r| DegRevLex.cmp(l, r).reverse());
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