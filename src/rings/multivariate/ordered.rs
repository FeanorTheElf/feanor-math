use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;

use crate::integer::IntegerRing;
use crate::integer::IntegerRingStore;
use crate::ring::*;
use crate::homomorphism::*;
use crate::seq::VectorViewMut;

use super::*;

///
/// Represents the multivariate polynomial ring `R[X1, ..., XN]` in `N` unknowns.
/// Polynomials are stored as lists of their terms, ordered by the given monomial
/// ordering.
/// 
/// Note that the specific ordering does not matter in principle, but when frequently
/// accessing the leading monomial (as e.g. done by Groebner basis algorithms, see
/// [`crate::algorithms::f4::f4()`]), choosing the ordering to match the ordering
/// used for the GB algorithm can significantly improve performance.
/// 
/// Note that there is currently no implementation of multivariate polynomial rings
/// with a number of unknowns chosen at runtime.
/// 
pub struct MultivariatePolyRingImplBase<R, O, const N: usize, A = Global>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    base_ring: R,
    element_allocator: A,
    order: O,
    zero: El<R>
}

///
/// [`RingStore`] corresponding to [`MultivariatePolyRingImplBase`].
/// 
pub type MultivariatePolyRingImpl<R, O, const N: usize, A = Global> = RingValue<MultivariatePolyRingImplBase<R, O, N, A>>;

impl<R, O, const N: usize> MultivariatePolyRingImpl<R, O, N>
    where R: RingStore,
        O: MonomialOrder
{
    ///
    /// Creates a new [`MultivariatePolyRingImpl`]
    /// 
    pub fn new(base_ring: R, monomial_order: O) -> Self {
        Self::new_with(base_ring, monomial_order, Global)
    }
}

impl<R, O, const N: usize, A> MultivariatePolyRingImpl<R, O, N, A>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    ///
    /// Creates a new [`MultivariatePolyRingImpl`], using the given allocator to allocate
    /// space for ring elements.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with(base_ring: R, monomial_order: O, element_allocator: A) -> Self {
        RingValue::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring,
            element_allocator: element_allocator,
            order: monomial_order
        })
    }
}

impl<R, O, const N: usize, A> Clone for MultivariatePolyRingImplBase<R, O, N, A>
    where R: RingStore + Clone,
        O: MonomialOrder + Clone,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self {
            base_ring: self.base_ring().clone(),
            element_allocator: self.element_allocator.clone(),
            order: self.order.clone(),
            zero: self.base_ring().zero()
        }
    }
}

impl<R, O, const N: usize, A> MultivariatePolyRingImplBase<R, O, N, A>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    fn is_valid(&self, el: &[(El<R>, Monomial<[MonomialExponent; N]>)]) -> bool {
        for i in 1..el.len() {
            if self.order.compare(&el.at(i - 1).1, &el.at(i).1) != Ordering::Less {
                return false;
            }
        }
        return true;
    }

    fn remove_zeros(&self, el: &mut Vec<(El<R>, Monomial<[MonomialExponent; N]>), A>) {
        let mut i = 0;
        for j in 0..el.len() {
            if !self.base_ring.is_zero(&el[j].0) {
                if i != j {
                    let tmp = std::mem::replace(el.at_mut(j), (self.base_ring.zero(), Monomial::new([0; N])));
                    *el.at_mut(i) = tmp;
                }
                i += 1;
            }
        }
        el.truncate(i);
    }

    ///
    /// Computes the sum of two elements, where the latter one does not have to fulfill
    /// all the contracts that we have for a ring element.
    ///
    #[inline]
    fn add_invalid(&self, lhs: <Self as RingBase>::Element, rhs_sorted: &[(El<R>, Monomial<[MonomialExponent; N]>)]) -> <Self as RingBase>::Element {
        debug_assert!(self.is_valid(&lhs.data));
        
        let mut result = Vec::with_capacity_in(lhs.data.len() + rhs_sorted.len(), self.element_allocator.clone());
        
        let mut i_l = 0;
        let mut i_r = 0;

        if lhs.data.len() == 0 && rhs_sorted.len() == 0 {
            return lhs;
        } else if lhs.data.len() > 0 && self.order.compare(&lhs.data.at(0).1, &rhs_sorted.at(0).1) != Ordering::Greater {
            result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1));
            i_l += 1;
        } else {
            result.push((self.base_ring.clone_el(&rhs_sorted.at(i_r).0), rhs_sorted.at(i_r).1));
            i_r += 1;
        }

        while i_r < rhs_sorted.len() {
            match self.order.compare(&result.last().unwrap().1, &rhs_sorted.at(i_r).1) {
                Ordering::Equal => {
                    self.base_ring.add_assign_ref(&mut result.last_mut().unwrap().0, &rhs_sorted.at(i_r).0);
                    i_r += 1;
                },
                Ordering::Greater => unreachable!(),
                Ordering::Less => if i_l < lhs.data.len() && self.order.compare(&lhs.data.at(i_l).1, &rhs_sorted.at(i_r).1) != Ordering::Greater {
                    result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1));
                    i_l += 1;
                } else {
                    result.push((self.base_ring.clone_el(&rhs_sorted.at(i_r).0), rhs_sorted.at(i_r).1));
                    i_r += 1;
                }
            }
        }
        for i in i_l..lhs.data.len() {
            result.push((self.base_ring.clone_el(&lhs.data.at(i).0), lhs.data.at(i).1));
        }
        self.remove_zeros(&mut result);
        return MultivariatePolyRingImplEl {
            data: result,
            ordering: PhantomData
        };
    }

    #[inline]
    fn add_scaled<const SCALED: bool>(&self, lhs: &<Self as RingBase>::Element, rhs: &<Self as RingBase>::Element, m: &Monomial<[MonomialExponent; N]>, factor: &El<R>) -> <Self as RingBase>::Element {
        debug_assert!(self.is_valid(&lhs.data));
        debug_assert!(self.is_valid(&rhs.data));
        
        let mut result = Vec::with_capacity_in(lhs.data.len() + rhs.data.len(), self.element_allocator.clone());
        
        let mut i_l = 0;
        let mut i_r = 0;
        while i_l < lhs.data.len() && i_r < rhs.data.len() {
            let mut rhs_monomial = rhs.data.at(i_r).1;
            rhs_monomial.mul_assign(m);
            match self.order.compare(&lhs.data.at(i_l).1, &rhs_monomial) {
                Ordering::Equal => {
                    result.push((self.base_ring.add_ref_fst(&lhs.data.at(i_l).0, self.base_ring.mul_ref(&rhs.data.at(i_r).0, factor)), lhs.data.at(i_l).1));
                    i_l += 1;
                    i_r += 1;
                },
                Ordering::Greater => {
                    if SCALED {
                        result.push((self.base_ring.mul_ref(&rhs.data.at(i_r).0, factor), rhs_monomial));
                    } else {
                        result.push((self.base_ring.clone_el(&rhs.data.at(i_r).0), rhs_monomial));
                    }
                    i_r += 1;
                },
                Ordering::Less => {
                    result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1));
                    i_l += 1;
                }
            }
        }
        if i_l == lhs.data.len() {
            for i in i_r..rhs.data.len() {
                let mut rhs_monomial = rhs.data.at(i).1;
                rhs_monomial.mul_assign(m);
                if SCALED {
                    result.push((self.base_ring.mul_ref(&rhs.data.at(i).0, factor), rhs_monomial));
                } else {
                    result.push((self.base_ring.clone_el(&rhs.data.at(i).0), rhs_monomial));
                }
            }
        } else {
            for i in i_l..lhs.data.len() {
                result.push((self.base_ring.clone_el(&lhs.data.at(i).0), lhs.data.at(i).1));
            }
        }
        self.remove_zeros(&mut result);
        return MultivariatePolyRingImplEl {
            data: result,
            ordering: PhantomData
        };
    }
}

pub struct MultivariatePolyRingImplEl<R: RingStore, O, const N: usize, A: Allocator + Clone> {
    data: Vec<(El<R>, Monomial<[MonomialExponent; N]>), A>,
    ordering: PhantomData<O>
}

impl<R, O, const N: usize, A> PartialEq for MultivariatePolyRingImplBase<R, O, N, A>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<R, O, const N: usize, A> RingBase for MultivariatePolyRingImplBase<R, O, N, A>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    type Element = MultivariatePolyRingImplEl<R, O, N, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(val.data.len(), self.element_allocator.clone());
        data.extend((0..val.data.len()).map(|i| (self.base_ring.clone_el(&val.data.at(i).0), val.data.at(i).1)));
        MultivariatePolyRingImplEl {
            data: data,
            ordering: PhantomData
        }
    }
    
    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.add_scaled::<false>(lhs, rhs, &Monomial::new([0; N]), &self.base_ring().one())
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        if lhs.data.len() > rhs.data.len() {
            (0..rhs.data.len()).fold(self.zero(), |current, i| self.add_scaled::<true>(&current, lhs, &rhs.data.at(i).1, &rhs.data.at(i).0))
        } else {
            (0..lhs.data.len()).fold(self.zero(), |current, i| self.add_scaled::<true>(&current, rhs, &lhs.data.at(i).1, &lhs.data.at(i).0))
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.add_ref(lhs, rhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.add_ref(lhs, &rhs);
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..lhs.data.len() {
            self.base_ring.negate_inplace(&mut lhs.data.at_mut(i).0);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.mul_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring.int_hom().map(value))
    }

    fn zero(&self) -> Self::Element {
        MultivariatePolyRingImplEl {
            data: Vec::new_in(self.element_allocator.clone()),
            ordering: PhantomData
        }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.data.len() != rhs.data.len() {
            return false;
        }
        for i in 0..lhs.data.len() {
            if lhs.data.at(i).1 != rhs.data.at(i).1 || !self.base_ring.eq_el(&lhs.data.at(i).0, &rhs.data.at(i).0) {
                return false
            }
        }
        return true;
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.data.len() == 0
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.data.len() == 1 && value.data.at(0).1 == Monomial::new([0; N]) && self.base_ring.is_one(&value.data.at(0).0)
    }
    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.data.len() == 1 && value.data.at(0).1 == Monomial::new([0; N]) && self.base_ring.is_neg_one(&value.data.at(0).0)
    }

    fn is_commutative(&self) -> bool { self.base_ring.is_commutative() }
    fn is_noetherian(&self) -> bool { self.base_ring.is_commutative() }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let mut print_term = |c: &El<R>, m: &Monomial<[MonomialExponent; N]>, with_plus: bool| {
            if with_plus {
                write!(out, " + ")?;
            }
            if !self.base_ring.is_one(c) || self.order.compare(m, &Monomial::new([0; N])) == Ordering::Equal {
                write!(out, "{}", self.base_ring.format(c))?;
                if self.order.compare(m, &Monomial::new([0; N])) != Ordering::Equal {
                    write!(out, " * ")?;
                }
            }
            let mut needs_space = false;
            for i in 0..N {
                if m[i] > 0 {
                    if needs_space {
                        write!(out, " * ")?;
                    }
                    write!(out, "X{}", i)?;
                    needs_space = true;
                }
                if m[i] > 1 {
                    write!(out, "^{}", m[i])?;
                }
            }
            return Ok::<(), std::fmt::Error>(());
        };

        if value.data.len() == 0 {
            write!(out, "{}", self.base_ring.format(&self.base_ring.zero()))?;
        } else {
            for i in 0..value.data.len() {
                print_term(&value.data.at(i).0, &value.data.at(i).1, i != 0)?;
            }
        }

        return Ok(());
    }
    
    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)
    }
}


impl<R, O, const N: usize, A> RingExtension for MultivariatePolyRingImplBase<R, O, N, A>
    where R: RingStore,
        O: MonomialOrder,
        A: Allocator + Clone
{
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        if self.base_ring.is_zero(&x) {
            self.zero()
        } else {
            let mut result = Vec::with_capacity_in(1, self.element_allocator.clone());
            result.push((x, Monomial::new([0; N])));
            return MultivariatePolyRingImplEl {
                data: result,
                ordering: PhantomData
            };
        }
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        for i in 0..lhs.data.len() {
            self.base_ring.mul_assign_ref(&mut lhs.data.at_mut(i).0, rhs)
        }
        self.remove_zeros(&mut lhs.data);
    }
}

impl<R1, O1, R2, O2, const N1: usize, const N2: usize, A1, A2> CanHomFrom<MultivariatePolyRingImplBase<R2, O2, N2, A2>> for MultivariatePolyRingImplBase<R1, O1, N1, A1>
    where R1: RingStore,
        O1: MonomialOrder,
        A1: Allocator + Clone,
        R2: RingStore,
        O2: MonomialOrder,
        A2: Allocator + Clone,
        R1::Type: CanHomFrom<R2::Type>
{
    type Homomorphism = <R1::Type as CanHomFrom<R2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &MultivariatePolyRingImplBase<R2, O2, N2, A2>) -> Option<Self::Homomorphism> {
        if N1 >= N2 {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in_ref(&self, from: &MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: &<MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = Vec::with_capacity_in(el.data.len(), self.element_allocator.clone());
        result.extend((0..el.data.len()).map(|i| (
            self.base_ring.get_ring().map_in_ref(from.base_ring().get_ring(), &el.data.at(i).0, hom), 
            Monomial::new(std::array::from_fn(|j| el.data.at(i).1[j] ))
        )));
        if !self.order.is_same(from.order.clone()) {
            result.sort_by(|l, r| self.order.compare(&l.1, &r.1));
        }
        return MultivariatePolyRingImplEl {
            data: result,
            ordering: PhantomData
        };
    }

    fn map_in(&self, from: &MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: <MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<R1, O1, R2, O2, const N1: usize, const N2: usize, A1, A2> CanIsoFromTo<MultivariatePolyRingImplBase<R2, O2, N2, A2>> for MultivariatePolyRingImplBase<R1, O1, N1, A1>
    where R1: RingStore,
        O1: MonomialOrder,
        A1: Allocator + Clone,
        R2: RingStore,
        O2: MonomialOrder,
        A2: Allocator + Clone,
        R1::Type: CanIsoFromTo<R2::Type>
{
    type Isomorphism = <R1::Type as CanIsoFromTo<R2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &MultivariatePolyRingImplBase<R2, O2, N2, A2>) -> Option<Self::Isomorphism> {
        if N1 == N2 {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element {
        let mut result = Vec::with_capacity_in(el.data.len(), from.element_allocator.clone());
        result.extend((0..el.data.len()).map(|i| (
            self.base_ring.get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data.at(i).0), iso), 
            Monomial::new(std::array::from_fn(|j| el.data.at(i).1[j] ))
        )));
        if !self.order.is_same(from.order.clone()) {
            result.sort_by(|l, r| self.order.compare(&l.1, &r.1));
        }
        return MultivariatePolyRingImplEl {
            data: result,
            ordering: PhantomData
        };
    }
}

///
/// Iterator over all terms of an element of [`MultivariatePolyRingImpl`].
/// 
pub struct MultivariatePolyRingBaseTermsIter<'a, R, O, const N: usize>
    where R: RingStore,
        O: MonomialOrder
{
    base_iter: std::slice::Iter<'a, (El<R>, Monomial<[MonomialExponent; N]>)>,
    order: PhantomData<O>
}

impl<'a, R, O, const N: usize> Iterator for MultivariatePolyRingBaseTermsIter<'a, R, O, N>
    where R: RingStore,
        O: MonomialOrder
{
    type Item = (&'a El<R>, &'a Monomial<[MonomialExponent; N]>);

    fn next(&mut self) -> Option<Self::Item> {
        let (c, m) = self.base_iter.next()?;
        return Some((c, m));
    }
}

impl<R, O, const N: usize> MultivariatePolyRing for MultivariatePolyRingImplBase<R, O, N>
    where R: RingStore,
        O: MonomialOrder
{
    type MonomialVector = [MonomialExponent; N];
    type TermsIterator<'a> = MultivariatePolyRingBaseTermsIter<'a, R, O, N>
        where Self: 'a;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a> {
        MultivariatePolyRingBaseTermsIter {
            base_iter: (&f.data).into_iter(),
            order: PhantomData
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Monomial<Self::MonomialVector>) -> &'a El<Self::BaseRing> {
        match f.data.binary_search_by(|x| self.order.compare(&x.1, m)) {
            Ok(i) => &f.data.at(i).0,
            Err(_) => &self.zero
        }
    }

    fn indeterminate_len(&self) -> usize {
        N
    }

    fn indeterminate(&self, i: usize) -> Self::Element {
        assert!(i < self.indeterminate_len());
        let mut result = Vec::with_capacity_in(1, self.element_allocator.clone());
        result.push((
            self.base_ring.one(),
            Monomial::new(std::array::from_fn(|j| if i == j { 1 } else { 0 }))
        ));
        return MultivariatePolyRingImplEl {
            data: result,
            ordering: PhantomData
        };
    }

    fn mul_monomial(&self, el: &mut Self::Element, m: &Monomial<Self::MonomialVector>) {
        for i in 0..el.data.len() {
            el.data.at_mut(i).1.mul_assign(m);
        }
    }

    fn lm<'a, O2>(&'a self, f: &'a Self::Element, order: O2) -> Option<&'a Monomial<Self::MonomialVector>>
        where O2: MonomialOrder
    {
        if f.data.len() == 0 {
            return None;
        } else if self.order.is_same(order.clone()) {
            return Some(&f.data.at(f.data.len() - 1).1);
        } else {
            return Some(&f.data.iter().max_by(|(_, ml), (_, mr)| order.compare(ml, mr)).unwrap().1);
        }
    }

    fn create_monomial<I: ExactSizeIterator<Item = MonomialExponent>>(&self, mut exponents: I) -> Monomial<Self::MonomialVector> {
        assert!(exponents.len() == self.indeterminate_len());
        Monomial::new(std::array::from_fn(|_| exponents.next().unwrap()))
    }

    fn evaluate<S, V, H>(&self, f: &Self::Element, values: V, hom: &H) -> S::Element
        where S: ?Sized + RingBase,
            H: Homomorphism<R::Type, S>,
            V: VectorView<S::Element> 
    {
        assert_eq!(values.len(), self.indeterminate_len());
        let new_ring: MultivariatePolyRingImpl<_, _, N> = MultivariatePolyRingImpl::new(hom.codomain(), self.order.clone());
        let mut result = new_ring.lifted_hom(&RingRef::new(self), hom).map_ref(f);
        for i in 0..self.indeterminate_len() {
            result = new_ring.specialize(&result, i, &new_ring.inclusion().map_ref(values.at(i)));
        }
        debug_assert!(result.data.len() <= 1);
        if result.data.len() == 0 {
            return hom.codomain().zero();
        } else {
            debug_assert!(result.data[0].1.deg() == 0);
            return result.data.into_iter().next().unwrap().0;
        }
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: Iterator<Item = (El<Self::BaseRing>, Monomial<Self::MonomialVector>)>
    {
        let mut to_add = Vec::new_in(self.element_allocator.clone());
        to_add.extend(rhs);
        to_add.sort_unstable_by(|(_, l), (_, r)| self.order.compare(l, r));
        *lhs = self.add_invalid(std::mem::replace(lhs, self.zero()), &to_add);
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::zn::zn_static;

#[test]
fn test_add() {
    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, Lex);
    let lhs = ring.from_terms([
        (1, Monomial::new([1, 0, 0]))
    ].into_iter());

    let rhs = ring.from_terms([
        (1, Monomial::new([1, 0, 0])),
        (1, Monomial::new([1, 0, 1])),
    ].into_iter());

    let expected = ring.from_terms([
        (2, Monomial::new([1, 0, 0])),
        (1, Monomial::new([1, 0, 1]))
    ].into_iter());

    let actual = ring.add_ref(&lhs, &rhs);
    assert_el_eq!(ring, expected, actual);

    let lhs = ring.from_terms([
        (1, Monomial::new([1, 0, 0])),
        (2, Monomial::new([1, 0, 1])),
        (4, Monomial::new([0, 2, 1])),
        (8, Monomial::new([0, 0, 0]))
    ].into_iter());

    let rhs = ring.from_terms([
        (-1, Monomial::new([1, 0, 0])),
        (3, Monomial::new([1, 0, 1])),
        (9, Monomial::new([0, 0, 1]))
    ].into_iter());

    let expected = ring.from_terms([
        (5, Monomial::new([1, 0, 1])),
        (4, Monomial::new([0, 2, 1])),
        (8, Monomial::new([0, 0, 0])),
        (9, Monomial::new([0, 0, 1]))
    ].into_iter());

    let actual = ring.add_ref(&lhs, &rhs);
    assert_el_eq!(ring, expected, actual);
}

#[cfg(test)]
fn edge_case_elements<'a>(ring: &'a RingValue<MultivariatePolyRingImplBase<StaticRing<i64>, DegRevLex, 3>>) -> impl 'a + Iterator<Item = <MultivariatePolyRingImplBase<StaticRing<i64>, DegRevLex, 3> as RingBase>::Element> {
    let mut result = vec![];
    let monomials = [
        Monomial::new([1, 0, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([2, 0, 1]),
        Monomial::new([4, 2, 0])

    ];
    for m1 in &monomials {
        result.push(ring.from_terms([(1, m1.clone())].into_iter()));
    }
    for m1 in &monomials {
        for m2 in &monomials {
            result.push(ring.from_terms([(1, m1.clone()), (-2, m2.clone())].into_iter()));
        }
    }
    return result.into_iter();
}

#[test]
fn test_ring_axioms() {
    let ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex);
    crate::ring::generic_tests::test_ring_axioms(&ring, edge_case_elements(&ring));
}

#[test]
fn test_add_assign_from_terms() {
    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, Lex);

    let mut lhs = ring.from_terms([
        (0, Monomial::new([0, 2, 0])),
        (1, Monomial::new([1, 0, 0])),
        (1, Monomial::new([1, 1, 1]))
    ].into_iter());

    ring.get_ring().add_assign_from_terms(&mut lhs, [
        (1, Monomial::new([0, 0, 0])),
        (1, Monomial::new([0, 0, 0])),
        (0, Monomial::new([1, 0, 0])),
        (1, Monomial::new([1, 0, 0])),
        (0, Monomial::new([1, 1, 0])),
        (1, Monomial::new([0, 1, 1])),
        (-1, Monomial::new([1, 1, 1]))
    ].into_iter());

    let expected = [
        (2, Monomial::new([0, 0, 0])),
        (1, Monomial::new([0, 1, 1])),
        (2, Monomial::new([1, 0, 0]))
    ];

    assert_eq!(expected.len(), ring.terms(&lhs).count());
    for (e, a) in expected.iter().zip(ring.terms(&lhs)) {
        assert_eq!(e.1, *a.1);
        assert_el_eq!(ring.base_ring(), e.0, a.0);
    }

    let lhs = ring.from_terms([(1, Monomial::new([0, 0, 0]))].into_iter().filter(|_| std::hint::black_box(true)));

    assert_el_eq!(ring, ring.one(), lhs);

    let value = ring.from_terms((0..100).map(|i| (1, Monomial::new([i, 0, 0]))).chain((0..100).map(|i| (0, Monomial::new([0, i, 0])))).filter(|_| true));

    assert_eq!(100, ring.terms(&value).count());

    assert_el_eq!(ring, ring.zero(), ring.from_terms([].into_iter()));
}

#[test]
fn test_evaluate() {
    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, Lex);
    let poly = ring.from_terms([
        (1, Monomial::new([0, 2, 0])),
        (2, Monomial::new([1, 0, 0])),
        (3, Monomial::new([1, 1, 1]))
    ].into_iter());
    assert_eq!(3, ring.evaluate(&poly, [1, 1, 0], &ring.base_ring().identity()));
    assert_eq!(0, ring.evaluate(&poly, [-2, 2, 0], &ring.base_ring().identity()));

    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(zn_static::Zn::<8>::RING, DegRevLex);
    let poly = ring.from_terms([
        (2, Monomial::new([0, 0, 0])),
        (1, Monomial::new([0, 1, 0])),
        (3, Monomial::new([0, 2, 0]))
    ].into_iter());
    assert_eq!(6, ring.evaluate(&poly, [1, 1, 0], &ring.base_ring().identity()));
    assert_eq!(0, ring.evaluate(&poly, [2, 2, 0], &ring.base_ring().identity()));
}