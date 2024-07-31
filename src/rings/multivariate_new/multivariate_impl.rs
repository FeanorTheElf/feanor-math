use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

use append_only_vec::AppendOnlyVec;
use thread_local::ThreadLocal;

use crate::extcmpmap::ExtCmpBTreeMap;
use crate::ring::*;
use crate::rings::multivariate_new::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::seq::{VectorView, VectorViewMut};

type Exponent = u16;

#[stability::unstable(feature = "enable")]
#[derive(Clone)]
pub enum MonomialIdentifier {
    Permanent { 
        deg: Exponent,
        index: usize
    },
    Temporary {
        deg: Exponent
    }
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingEl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    data: Vec<(El<R>, MonomialIdentifier)>,
    order: PhantomData<O>
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    base_ring: R,
    order: O,
    variable_count: usize,
    allocated_monomials: AppendOnlyVec<Box<[Exponent]>>,
    tmp_monomials: ThreadLocal<Box<[Cell<Exponent>]>>,
    monomial_table: RefCell<ExtCmpBTreeMap<MonomialIdentifier, usize, CompareMonomialFamily<Self, O>>>,
    zero: El<R>
}

impl<R, O> MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    fn is_valid(&self, el: &[(El<R>, MonomialIdentifier)]) -> bool {
        for (_, m) in el {
            if let MonomialIdentifier::Permanent { deg, index } = m {
                assert_eq!(*deg, self.allocated_monomials[*index].iter().copied().sum::<Exponent>());
            } else {
                assert!(false);
            }
        }
        for i in 1..el.len() {
            if self.order.compare(RingRef::new(self), &el[i - 1].1, &el[i].1) != Ordering::Less {
                return false;
            }
        }
        for i in 0..el.len() {
            if self.base_ring().is_zero(&el[i].0) {
                return false;
            }
        }
        return true;
    }

    fn remove_zeros(&self, el: &mut Vec<(El<R>, MonomialIdentifier)>) {
        let mut i = 0;
        for j in 0..el.len() {
            if !self.base_ring.is_zero(&el[j].0) {
                if i != j {
                    let tmp = std::mem::replace(el.at_mut(j), (self.base_ring.zero(), MonomialIdentifier::Temporary { deg: 0 }));
                    *el.at_mut(i) = tmp;
                }
                i += 1;
            }
        }
        el.truncate(i);
        debug_assert!(self.is_valid(&el));
    }

    ///
    /// Computes the sum of two elements, where the latter one does not have to fulfill
    /// all the contracts that we have for a ring element. However, the latter one must be 
    /// sorted.
    ///
    fn add_terms<I>(&self, lhs: &<Self as RingBase>::Element, mut rhs_sorted: I, out: Vec<(El<R>, MonomialIdentifier)>) -> <Self as RingBase>::Element
        where I: Iterator<Item = (El<R>, MonomialIdentifier)>
    {
        debug_assert!(self.is_valid(&lhs.data));
        
        let mut result = out;
        result.clear();
        result.reserve(lhs.data.len() + rhs_sorted.size_hint().0);
        
        let mut i_l = 0;

        if let Some((c_r, m_r)) = rhs_sorted.next() {
            while i_l < lhs.data.len() && self.order.compare(RingRef::new(self), &lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            if result.len() == 0 {
                result.push((c_r, m_r));
            } else {
                match self.order.compare(RingRef::new(self), &result.last().unwrap().1, &m_r) {
                    Ordering::Equal => {
                        self.base_ring.add_assign_ref(&mut result.last_mut().unwrap().0, &c_r);
                    },
                    Ordering::Less => {
                        result.push((c_r, m_r));
                    },
                    Ordering::Greater => unreachable!(),
                }
            }
        }
        while let Some((c_r, m_r)) = rhs_sorted.next() {
            while i_l < lhs.data.len() && self.order.compare(RingRef::new(self), &lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            match self.order.compare(RingRef::new(self), &result.last().unwrap().1, &m_r) {
                Ordering::Equal => {
                    self.base_ring.add_assign_ref(&mut result.last_mut().unwrap().0, &c_r);
                },
                Ordering::Less => {
                    result.push((c_r, m_r));
                },
                Ordering::Greater => unreachable!(),
            }
        }
        for i in i_l..lhs.data.len() {
            result.push((self.base_ring.clone_el(&lhs.data.at(i).0), lhs.data.at(i).1.clone()));
        }
        self.remove_zeros(&mut result);
        debug_assert!(self.is_valid(&result));
        return MultivariatePolyRingEl {
            data: result,
            order: PhantomData
        };
    }

    fn compare_monomial<'a>(&'a self) -> CompareMonomial<RingRef<'a, Self>, O> {
        CompareMonomial { poly_ring: RingRef::new(self), order: self.order.clone() }
    }
}

impl<R, O> PartialEq for MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    fn eq(&self, other: &Self) -> bool {
        // it is not sufficient if base_ring and variable_count match (the rings are isomorphic then),
        // since the monomial indices of elements could point to different values
        std::ptr::eq(self, other)
    }
}

impl<R, O> RingBase for MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    type Element = MultivariatePolyRingEl<R, O>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        MultivariatePolyRingEl {
            data: val.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), self.clone_monomial(m))).collect(),
            order: PhantomData
        }
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        debug_assert!(self.is_valid(&rhs.data));
        self.add_terms(lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), m.clone())), Vec::new())
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.add_ref(lhs, rhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.into_iter(), Vec::new());
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().negate(self.base_ring.clone_el(c)), m.clone())), Vec::new());
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(self.is_valid(&lhs.data));
        for (c, _) in &mut lhs.data {
            self.base_ring().negate_inplace(c);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        MultivariatePolyRingEl {
            data: Vec::new(),
            order: PhantomData
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&lhs.data));
        debug_assert!(self.is_valid(&rhs.data));
        if lhs.data.len() != rhs.data.len() {
            return false;
        }
        for i in 0..lhs.data.len() {
            match (&lhs.data.at(i).1, &rhs.data.at(i).1) {
                (MonomialIdentifier::Permanent { deg: _, index: lhs_i }, MonomialIdentifier::Permanent { deg: _, index: rhs_i }) => if lhs_i != rhs_i || !self.base_ring.eq_el(&lhs.data.at(i).0, &rhs.data.at(i).0) {
                    return false;
                },
                _ => unreachable!()
            }
        }
        return true;
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.data.len() == 0
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&value.data));
        if value.data.len() != 1 {
            return false;
        }
        if let MonomialIdentifier::Permanent { deg, index: _ } = &value.data[0].1 {
            return *deg == 0 && self.base_ring().is_one(&value.data[0].0)
        } else {
            unreachable!()
        }
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&value.data));
        if value.data.len() != 1 {
            return false;
        }
        if let MonomialIdentifier::Permanent { deg, index: _ } = &value.data[0].1 {
            return *deg == 0 && self.base_ring().is_neg_one(&value.data[0].0)
        } else {
            unreachable!()
        }
    }

    fn is_commutative(&self) -> bool { self.base_ring().is_commutative() }
    fn is_noetherian(&self) -> bool { self.base_ring().is_noetherian() }

    fn dbg(&self, value: &Self::Element, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        unimplemented!()
    }

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
         where I::Type: IntegerRing 
    {
        self.base_ring().characteristic(ZZ)
    }
}

#[stability::unstable(feature = "enable")]
pub struct TermIterImpl<'a, R, O>
    where R: RingStore,
        O: MonomialOrder
{
    base_iter: std::slice::Iter<'a, (El<R>, MonomialIdentifier)>,
    order: PhantomData<O>
}

impl<'a, R, O> Iterator for TermIterImpl<'a, R, O>
    where R: RingStore,
        O: MonomialOrder
{
    type Item = (&'a El<R>, &'a MonomialIdentifier);

    fn next(&mut self) -> Option<Self::Item> {
        self.base_iter.next().map(|(c, m)| (c, m))
    }
}

impl<R, O> RingExtension for MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    type BaseRing = R;

    fn base_ring<'b>(&'b self) -> &'b Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        if self.base_ring().is_zero(&x) {
            return self.zero();
        } else {
            return MultivariatePolyRingEl {
                data: vec![(x, self.create_monomial((0..self.variable_count).map(|_| 0)))],
                order: PhantomData
            }
        }
    }
}

impl<R, O> MultivariatePolyRing for MultivariatePolyRingImpl<R, O>
    where R: RingStore,
        O: MonomialOrder
{
    type Monomial = MonomialIdentifier;
    type TermIter<'a> = TermIterImpl<'a, R, O>
        where Self: 'a;
        
    fn variable_count(&self) -> usize {
        self.variable_count
    }

    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: ExactSizeIterator<Item = usize>
    {
        assert_eq!(exponents.len(), self.variable_count());

        let tmp_monomial = self.tmp_monomials.get_or(|| (0..self.variable_count).map(|_| Cell::new(0)).collect::<Vec<_>>().into_boxed_slice());
        let mut deg = 0;
        for (i, e) in exponents.enumerate() {
            deg += e as Exponent;
            tmp_monomial[i].set(e as Exponent);
        }
        {
            let monomial_table = self.monomial_table.borrow();
            let entry = monomial_table.get(&MonomialIdentifier::Temporary { deg: deg }, self.compare_monomial());
            // do we have the monomial already allocated?
            if let Some(idx) = entry {
                return MonomialIdentifier::Permanent { deg: deg, index: *idx };
            }
        }
        {
            let mut monomial_table = self.monomial_table.borrow_mut();
            // if not, allocate it!
            let idx = self.allocated_monomials.push(tmp_monomial.iter().map(|e| e.get()).collect::<Vec<_>>().into_boxed_slice());
            monomial_table.insert(MonomialIdentifier::Permanent { deg: deg, index: idx  }, idx, self.compare_monomial());
            return MonomialIdentifier::Permanent { deg: deg, index: idx };
        }
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        mon.clone()
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I:Iterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        unimplemented!()
    }

    fn mul_assign_monomial(&self, f: &mut Self::Element, monomial: Self::Monomial) {
        unimplemented!()
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing> {
        match f.data.binary_search_by(|(_, fm)| self.order.compare(RingRef::new(self), fm, m)) {
            Ok(i) => &f.data.at(i).0,
            Err(_) => &self.zero
        }
    }

    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize {
        unimplemented!()
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a> {
        TermIterImpl {
            base_iter: f.data.iter(),
            order: PhantomData
        }
    }

    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        match mon {
            MonomialIdentifier::Permanent { deg, index: _ } => *deg as usize,
            MonomialIdentifier::Temporary { deg } => *deg as usize
        }
    }
}