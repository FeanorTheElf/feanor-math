use std::ops::Index;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::cmp::{min, max};

use crate::ring::*;
use crate::vector::{VectorView, VectorViewMut};

pub mod vec_based;

type MonomialExponent = u16;

pub trait MultivariatePolyRing: RingExtension + SelfIso {

    type MonomialVector: VectorView<MonomialExponent>;
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
}

pub trait MultivarPolyRingStore: RingStore
    where Self::Type: MultivariatePolyRing
{
    delegate!{ fn indeterminate_len(&self) -> usize }
    delegate!{ fn indeterminate(&self, i: usize) -> El<Self> }
    delegate!{ fn monomial(&self, m: &Monomial<<Self::Type as MultivariatePolyRing>::MonomialVector>) -> El<Self> }

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
}

impl<R> MultivarPolyRingStore for R
    where R: RingStore,
        R::Type: MultivariatePolyRing
{}

static ZERO: MonomialExponent = 0;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Monomial<V: VectorView<MonomialExponent>> {
    exponents: V
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
