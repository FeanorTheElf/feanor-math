use std::{ops::Index, cmp::Ordering, fmt::Debug};

type MonomialExponent = u16;

static ZERO: MonomialExponent = 0;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Monomial<const N: usize> {
    exponents: [MonomialExponent; N]
}

impl<const N: usize> Monomial<N> {
    
    pub fn new<const M: usize>(exponents: [MonomialExponent; M]) -> Self {
        assert!(M <= N);
        Self {
            exponents: std::array::from_fn(|i| if i < M { exponents[i] } else { 0 })
        }
    }

    pub fn deg(&self) -> MonomialExponent {
        self.exponents.iter().sum()
    }
}

impl<const N: usize> Debug for Monomial<N> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.exponents)
    }
}

impl<const N: usize> Index<usize> for Monomial<N> {

    type Output = MonomialExponent;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= N {
            &ZERO
        } else {
            &self.exponents[index]
        }
    }
}

pub trait MonomialOrder {

    fn is_graded(&self) -> bool;
    fn cmp<const N: usize>(&self, lhs: &Monomial<N>, rhs: &Monomial<N>) -> Ordering;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DegRevLex;

impl MonomialOrder for DegRevLex {

    fn is_graded(&self) -> bool {
        true
    }

    fn cmp<const N: usize>(&self, lhs: &Monomial<N>, rhs: &Monomial<N>) -> Ordering {
        let lhs_deg = lhs.deg();
        let rhs_deg = rhs.deg();
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..N).rev() {
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

    fn cmp<const N: usize>(&self, lhs: &Monomial<N>, rhs: &Monomial<N>) -> Ordering {
        for i in 0..N {
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
    let mut monomials: Vec<Monomial<3>> = vec![
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
    let mut monomials: Vec<Monomial<3>> = vec![
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