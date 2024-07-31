use crate::extcmpmap::CompareFnFamily;
use crate::ring::*;
use crate::wrapper::RingElementWrapper;
use crate::homomorphism::*;

pub mod multivariate_impl;

use std::any::Any;
use std::cmp::{max, Ordering};
use std::marker::PhantomData;

pub type PolyCoeff<P> = El<<<P as RingStore>::Type as RingExtension>::BaseRing>;
pub type PolyMonomial<P> = <<P as RingStore>::Type as MultivariatePolyRing>::Monomial;

///
/// A new try at creating a trait for multivariate polynomial rings.
/// 
/// Currently [`crate::rings::multivariate::MultivariatePolyRing`] is still the main
/// trait for multivariate polynomial rings in feanor-math, but this here might someday replace
/// it, assuming I am happy with how its design turns out.
/// 
#[stability::unstable(feature = "enable")]
pub trait MultivariatePolyRing: RingExtension {

    type Monomial;
    type TermIter<'a>: Iterator<Item = (&'a El<Self::BaseRing>, &'a Self::Monomial)>
        where Self: 'a;

    fn variable_count(&self) -> usize;

    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator;

    fn mul_assign_monomial(&self, f: &mut Self::Element, monomial: Self::Monomial);

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing>;

    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a>;

    fn create_term(&self, coeff: El<Self::BaseRing>, monomial: Self::Monomial) -> Self::Element {
        let mut result = self.from(coeff);
        self.mul_assign_monomial(&mut result, monomial);
        return result;
    }

    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        self.terms(f).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
    }

    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O, lt_than: &Self::Monomial) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.into_iter().map(|(c, m)| self.create_term(c, m))
        ));
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.variable_count()).map(|i| self.exponent_at(mon, i)))
    }

    fn monomial_mul(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.variable_count()).map(|i| self.exponent_at(&lhs, i) + self.exponent_at(rhs, i)))
    }

    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        (0..self.variable_count()).map(|i| self.exponent_at(mon, i)).sum()
    }

    fn monomial_lcm(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.variable_count()).map(|i| max(self.exponent_at(&lhs, i), self.exponent_at(rhs, i))))
    }

    ///
    /// if rhs divides lhs, returns the quotient. Otherwise returns `Result::Err` with the monomial
    /// `lhs / gcd(rhs, lhs)`.
    /// 
    fn monomial_div(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Result<Self::Monomial, Self::Monomial> {
        let mut failed = false;
        let result = self.create_monomial((0..self.variable_count()).map(|i| {
            if let Some(res) = self.exponent_at(&lhs, i).checked_sub(self.exponent_at(rhs, i)) {
                res
            } else {
                failed = true;
                0
            }
        }));
        if failed {
            Err(result)
        } else {
            Ok(result)
        }
    }
}

#[stability::unstable(feature = "enable")]
pub trait MultivariatePolyRingStore: RingStore
    where Self::Type: MultivariatePolyRing
{
    delegate!{ MultivariatePolyRing, fn variable_count(&self) -> usize }
    delegate!{ MultivariatePolyRing, fn create_term(&self, coeff: PolyCoeff<Self>, monomial: PolyMonomial<Self>) -> El<Self> }
    delegate!{ MultivariatePolyRing, fn exponent_at(&self, m: &PolyMonomial<Self>, var_index: usize) -> usize }
    delegate!{ MultivariatePolyRing, fn monomial_mul(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> PolyMonomial<Self> }
    delegate!{ MultivariatePolyRing, fn monomial_lcm(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> PolyMonomial<Self> }
    delegate!{ MultivariatePolyRing, fn monomial_div(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> Result<PolyMonomial<Self>, PolyMonomial<Self>> }
    delegate!{ MultivariatePolyRing, fn monomial_deg(&self, val: &PolyMonomial<Self>) -> usize }
    delegate!{ MultivariatePolyRing, fn mul_assign_monomial(&self, f: &mut El<Self>, monomial: PolyMonomial<Self>) -> () }

    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a El<Self>, order: O, lt_than: &PolyMonomial<Self>) -> Option<(&'a PolyCoeff<Self>, &'a PolyMonomial<Self>)> {
        self.get_ring().largest_term_lt(f, order, lt_than)
    }
    
    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a El<Self>, order: O) -> Option<(&'a PolyCoeff<Self>, &'a PolyMonomial<Self>)> {
        self.get_ring().LT(f, order)
    }

    fn create_monomial<I>(&self, exponents: I) -> PolyMonomial<Self>
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator
    {
        self.get_ring().create_monomial(exponents)
    }

    fn clone_monomial(&self, mon: &PolyMonomial<Self>) -> PolyMonomial<Self> {
        self.get_ring().clone_monomial(mon)
    }

    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, m: &PolyMonomial<Self>) -> &'a PolyCoeff<Self> {
        self.get_ring().coefficient_at(f, m)
    }

    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as MultivariatePolyRing>::TermIter<'a> {
        self.get_ring().terms(f)
    }

    fn from_terms<I>(&self, terms: I) -> El<Self>
        where I: IntoIterator<Item = (PolyCoeff<Self>, PolyMonomial<Self>)>
    {
        let mut result = self.zero();
        self.get_ring().add_assign_from_terms(&mut result, terms);
        return result;
    }
    
    #[stability::unstable(feature = "enable")]
    fn with_wrapped_indeterminates<'a, F, T, const N: usize>(&'a self, f: F) -> Vec<El<Self>>
        where F: FnOnce([&RingElementWrapper<&'a Self>; N]) -> T,
            T: IntoIterator<Item = RingElementWrapper<&'a Self>>
    {
        assert_eq!(self.variable_count(), N);
        let wrapped_indets: [_; N] = std::array::from_fn(|i| RingElementWrapper::new(self, self.create_term(self.base_ring().one(), self.create_monomial((0..N).map(|j| if i == j { 1 } else { 0 })))));
        f(std::array::from_fn(|i| &wrapped_indets[i])).into_iter().map(|f| f.unwrap()).collect()
    }
}

impl<P> MultivariatePolyRingStore for P
    where P: RingStore,
        P::Type: MultivariatePolyRing {}

#[stability::unstable(feature = "enable")]
pub trait MonomialOrder: Clone {

    fn compare<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> Ordering
        where P: RingStore,
            P::Type: MultivariatePolyRing;

    fn eq_mon<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> bool
        where P: RingStore,
            P::Type: MultivariatePolyRing
    {
        self.compare(ring, lhs, rhs) == Ordering::Equal
    }

    fn is_same<O>(&self, rhs: &O) -> bool
        where O: MonomialOrder
    {
        assert!(self.as_any().is_some());
        assert!(std::mem::size_of::<Self>() == 0);
        if let Some(rhs_as_any) = rhs.as_any() {
            self.as_any().unwrap().type_id() == rhs_as_any.type_id()
        } else {
            false
        }
    }

    fn as_any(&self) -> Option<&dyn Any>;
}

#[stability::unstable(feature = "enable")]
pub trait GradedMonomialOrder: MonomialOrder {}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DegRevLex;

impl MonomialOrder for DegRevLex {

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self as &dyn Any)
    }

    fn compare<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> Ordering
        where P:RingStore,
            P::Type:MultivariatePolyRing
    {
        let lhs_deg = ring.monomial_deg(lhs);
        let rhs_deg = ring.monomial_deg(rhs);
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..ring.variable_count()).rev() {
                if ring.exponent_at(lhs, i) > ring.exponent_at(rhs, i) {
                    return Ordering::Less
                } else if ring.exponent_at(lhs, i) < ring.exponent_at(rhs, i) {
                    return Ordering::Greater;
                }
            }
            return Ordering::Equal;
        }
    }
}

pub struct CompareMonomialFamily<P, O>
    where P: ?Sized + MultivariatePolyRing,
        O: MonomialOrder
{
    poly_ring: PhantomData<P>,
    order: PhantomData<O>
}

impl<P, O> CompareFnFamily<P::Monomial> for CompareMonomialFamily<P, O>
    where P: ?Sized + MultivariatePolyRing,
        O: MonomialOrder
{
    type CompareFn<'a> = CompareMonomial<RingRef<'a, P>, O>
        where Self: 'a;
}

#[derive(Copy, Clone)]
pub struct CompareMonomial<P, O>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder
{
    pub poly_ring: P,
    pub order: O
}
impl<'a, P, O> FnOnce<(&'a PolyMonomial<P>, &'a PolyMonomial<P>)> for CompareMonomial<P, O>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder
{
    type Output = Ordering;
    
    extern "rust-call" fn call_once(self, args: (&'a PolyMonomial<P>, &'a PolyMonomial<P>)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, P, O> FnMut<(&'a PolyMonomial<P>, &'a PolyMonomial<P>)> for CompareMonomial<P, O>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder
{
    extern "rust-call" fn call_mut(&mut self, args: (&'a PolyMonomial<P>, &'a PolyMonomial<P>)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, P, O> Fn<(&'a PolyMonomial<P>, &'a PolyMonomial<P>)> for CompareMonomial<P, O>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder
{
    extern "rust-call" fn call(&self, args: (&'a PolyMonomial<P>, &'a PolyMonomial<P>)) -> Self::Output {
        self.order.compare(self.poly_ring, args.0, args.1)
    }
}

#[stability::unstable(feature = "enable")]
pub mod generic_impls {
    use std::fmt::{Formatter, Result};
    use super::*;

    #[stability::unstable(feature = "enable")]
    pub fn print<P>(ring: P, poly: &El<P>, out: &mut Formatter, env: EnvBindingStrength) -> Result
        where P: RingStore,
            P::Type: MultivariatePolyRing
    {
        if ring.is_zero(poly) {
            ring.base_ring().get_ring().dbg_within(&ring.base_ring().zero(), out, env)?;
        } else {
            if env >= EnvBindingStrength::Product {
                write!(out, "(")?;
            }
            
            let mut print_term = |c: &PolyCoeff<P>, m: &PolyMonomial<P>, with_plus: bool| {
                if with_plus {
                    write!(out, " + ")?;
                }
                let is_constant_term = ring.monomial_deg(m) == 0;
                if !ring.base_ring().is_one(c) || is_constant_term {
                    ring.base_ring().get_ring().dbg_within(c, out, if is_constant_term { EnvBindingStrength::Sum } else { EnvBindingStrength::Product })?;
                    if !is_constant_term {
                        write!(out, " * ")?;
                    }
                }
                let mut needs_space = false;
                for i in 0..ring.variable_count() {
                    if ring.exponent_at(m, i) > 0 {
                        if needs_space {
                            write!(out, " * ")?;
                        }
                        write!(out, "X{}", i)?;
                        needs_space = true;
                    }
                    if ring.exponent_at(m, i) > 1 {
                        write!(out, "^{}", ring.exponent_at(m, i))?;
                    }
                }
                return Ok::<(), std::fmt::Error>(());
            };
            
            for (i, (c, m)) in ring.terms(poly).enumerate() {
                print_term(c, m, i != 0)?;
            }
            if env >= EnvBindingStrength::Product {
                write!(out, ")")?;
            }
        }

        return Ok(());
    }
}

#[stability::unstable(feature = "enable")]
#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use super::*;

    #[stability::unstable(feature = "enable")]
    pub fn test_poly_ring_axioms<P: RingStore, I: Iterator<Item = PolyCoeff<P>>>(ring: P, interesting_base_ring_elements: I)
        where P::Type: MultivariatePolyRing
    {
        let elements = interesting_base_ring_elements.collect::<Vec<_>>();
        let n = ring.variable_count();

        // test multiplication of variables
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                let xj = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 })));
                let xixj = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == i && k == j { 2 } else if k == j || k == i { 1 } else { 0 })));
                assert_el_eq!(ring, xixj, ring.mul(xi, xj));
            }
        }

        // test monomial operations
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 }));
                let xj = ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 }));
                let xixj_lcm = ring.create_monomial((0..n).map(|k| if k == j || k == i { 1 } else { 0 }));
                assert_el_eq!(ring, ring.create_term(ring.base_ring().one(), xixj_lcm), ring.create_term(ring.base_ring().one(), ring.monomial_lcm(xi, &xj)));
            }
        }

        // all monomials should be different
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                let xj = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 })));
                assert!((i == j) == ring.eq_el(&xi, &xj));
            }
        }

        // monomials shouldn't be zero divisors
        for i in 0..n {
            for a in &elements {
                let xi = ring.create_term(ring.base_ring().one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                assert!(ring.base_ring().is_zero(a) == ring.is_zero(&ring.inclusion().mul_ref_map(&xi, a)));
            }
        }

        if n >= 2 {
            let one = ring.create_monomial((0..n).map(|_| 0));
            let x0 = ring.create_monomial((0..n).map(|k| if k == 0 { 1 } else { 0 }));
            let x1 = ring.create_monomial((0..n).map(|k| if k == 1 { 1 } else { 0 }));
            let x0_v = ring.create_term(ring.base_ring().one(), ring.clone_monomial(&x0));
            let x1_v = ring.create_term(ring.base_ring().one(), ring.clone_monomial(&x1));
            let x0_2 = ring.create_monomial((0..n).map(|k| if k == 0 { 2 } else { 0 }));
            let x0_3 = ring.create_monomial((0..n).map(|k| if k == 0 { 3 } else { 0 }));
            let x0_4 = ring.create_monomial((0..n).map(|k| if k == 0 { 4 } else { 0 }));
            let x0x1 = ring.create_monomial((0..n).map(|k| if k == 0 || k == 1 { 1 } else { 0 }));
            let x0_3x1 = ring.create_monomial((0..n).map(|k| if k == 0 { 3 } else if k == 1 { 1 } else { 0 }));
            let x1_2 = ring.create_monomial((0..n).map(|k| if k == 1 { 2 } else { 0 }));

            // test product
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let f = ring.add(ring.inclusion().mul_ref_map(&x0_v, a), ring.inclusion().mul_ref_map(&x1_v, b));
                        let g = ring.add(ring.inclusion().mul_ref_map(&x0_v, c), ring.clone_el(&x1_v));
                        let h = ring.from_terms([
                            (ring.base_ring().mul_ref(a, c), ring.clone_monomial(&x0_2)),
                            (ring.base_ring().add_ref_snd(ring.base_ring().mul_ref(b, c), a), ring.clone_monomial(&x0x1)),
                            (ring.base_ring().clone_el(b), ring.clone_monomial(&x1_2)),
                        ].into_iter());
                        assert_el_eq!(ring, h, ring.mul(f, g));
                    }
                }
            }

            // test sum
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let f = ring.from_terms([
                            (ring.base_ring().clone_el(a), ring.clone_monomial(&one)),
                            (ring.base_ring().clone_el(c), ring.clone_monomial(&x0_2)),
                            (ring.base_ring().one(), ring.clone_monomial(&x0x1))
                        ]);
                        let g = ring.from_terms([
                            (ring.base_ring().clone_el(b), ring.clone_monomial(&x0)),
                            (ring.base_ring().one(), ring.clone_monomial(&x0_2)),
                        ]);
                        let h = ring.from_terms([
                            (ring.base_ring().clone_el(a), ring.clone_monomial(&one)),
                            (ring.base_ring().clone_el(b), ring.clone_monomial(&x0)),
                            (ring.base_ring().add_ref_fst(c, ring.base_ring().one()), ring.clone_monomial(&x0_2)),
                            (ring.base_ring().one(), ring.clone_monomial(&x0x1))
                        ]);
                        assert_el_eq!(ring, h, ring.add(f, g));
                    }
                }
            }

            // test mul_assign_monomial
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let mut f = ring.from_terms([
                            (ring.base_ring().clone_el(a), ring.clone_monomial(&one)),
                            (ring.base_ring().clone_el(b), ring.clone_monomial(&x0)),
                            (ring.base_ring().clone_el(c), ring.clone_monomial(&x0_2)),
                            (ring.base_ring().one(), ring.clone_monomial(&x0x1))
                        ]);
                        let h = ring.from_terms([
                            (ring.base_ring().clone_el(a), ring.clone_monomial(&x0_2)),
                            (ring.base_ring().clone_el(b), ring.clone_monomial(&x0_3)),
                            (ring.base_ring().clone_el(c), ring.clone_monomial(&x0_4)),
                            (ring.base_ring().one(), ring.clone_monomial(&x0_3x1))
                        ]);
                        let m = ring.clone_monomial(&x0_2);
                        ring.mul_assign_monomial(&mut f, m);
                        assert_el_eq!(ring, h, f);
                    }
                }
            }
        }
    }
}