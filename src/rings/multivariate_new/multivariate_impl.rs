use std::cell::{Cell, RefCell};

use append_only_vec::AppendOnlyVec;
use thread_local::ThreadLocal;

use crate::extcmpmap::ExtCmpBTreeMap;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate_new::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::seq::{VectorView, VectorViewMut};

type Exponent = u16;
type OrderIdx = u32;

thread_local!{
    static BINOMIAL_COEFF_LOOKUP_TABLE: RefCell<Vec<Vec<i64>>> = RefCell::new(Vec::new());
    static CUM_BINOMIAL_COEFF_LOOKUP_TABLE: RefCell<Vec<Vec<u32>>> = RefCell::new(Vec::new());
}

fn binomial(n: usize, k: usize) -> i64 {

    fn compute_binomial(n: i64, k: i64) -> i64 {
        assert!(k <= n);
        StaticRing::<i64>::RING.prod((n - k + 1)..(n + 1)) / StaticRing::<i64>::RING.prod(1..(k + 1))
    }

    assert!(k <= n);
    BINOMIAL_COEFF_LOOKUP_TABLE.with_borrow_mut(|table| {
        if table.len() <= n {
            table.resize_with(n + 1, || Vec::new());
        }
        let table_for_n = &mut table[n];
        while table_for_n.len() <= k {
            table_for_n.push(compute_binomial(n as i64, table_for_n.len() as i64));
        }
        return table_for_n[k];
    })
}

///
/// Computes `sum_(0 <= l <= k) binomial(n + l, n)`
/// 
fn cum_binomial(n: usize, k: i64) -> u32 {

    if k < 0 {
        return 0;
    }

    fn compute_cum_binomial(n: usize, k: usize) -> i64 {
        StaticRing::<i64>::RING.sum((0..(k + 1)).map(|l| binomial(n + l, n)))
    }

    CUM_BINOMIAL_COEFF_LOOKUP_TABLE.with_borrow_mut(|table| {
        if table.len() <= n {
            table.resize_with(n + 1, || Vec::new());
        }
        let table_for_n = &mut table[n];
        while table_for_n.len() <= k as usize {
            table_for_n.push(u32::try_from(compute_cum_binomial(n, table_for_n.len())).unwrap());
        }
        return table_for_n[k as usize];
    })
}

///
/// Returns the index of the given monomial within the list of all degree-d monomials, ordered by DegRevLex
/// 
fn enumeration_index_degrevlex(d: Exponent, mon: &[Exponent]) -> OrderIdx {
    debug_assert!(d == mon.iter().copied().sum());
    let n = mon.len();
    let mut remaining_degree: i64 = d as i64 - 1;
    let mut result = 0;
    for i in 0..(n - 1) {
        remaining_degree -= mon[n - 1 - i] as i64;
        result += cum_binomial(n - i - 2, remaining_degree);
    }
    return result;
}

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
pub struct MultivariatePolyRingEl<R>
    where R: RingStore
{
    data: Vec<(El<R>, MonomialIdentifier)>
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    base_ring: R,
    variable_count: usize,
    allocated_monomials: AppendOnlyVec<Box<[Exponent]>>,
    tmp_monomials: ThreadLocal<Box<[Cell<Exponent>]>>,
    monomial_table: RefCell<ExtCmpBTreeMap<MonomialIdentifier, usize, CompareMonomialFamily<Self, DegRevLex>>>,
    zero: El<R>
}

pub type MultivariatePolyRingImpl<R> = RingValue<MultivariatePolyRingImplBase<R>>;

impl<R> MultivariatePolyRingImpl<R>
    where R: RingStore
{
    #[stability::unstable(feature = "enable")]
    pub fn new(base_ring: R, variable_count: usize) -> Self {
        Self::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring: base_ring,
            variable_count: variable_count,
            allocated_monomials: AppendOnlyVec::new(),
            tmp_monomials: ThreadLocal::new(),
            monomial_table: RefCell::new(ExtCmpBTreeMap::new()),
        })
    }
}

impl<R> MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    fn tmp_monomial(&self) -> &[Cell<u16>] {
        self.tmp_monomials.get_or(|| (0..self.variable_count).map(|_| Cell::new(0)).collect::<Vec<_>>().into_boxed_slice())
    }

    fn is_valid(&self, el: &[(El<R>, MonomialIdentifier)]) -> bool {
        for (_, m) in el {
            if let MonomialIdentifier::Permanent { deg, index } = m {
                assert_eq!(*deg, self.allocated_monomials[*index].iter().copied().sum::<Exponent>());
            } else {
                assert!(false);
            }
        }
        for i in 1..el.len() {
            if DegRevLex.compare(RingRef::new(self), &el[i - 1].1, &el[i].1) != Ordering::Less {
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
            while i_l < lhs.data.len() && DegRevLex.compare(RingRef::new(self), &lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            if result.len() == 0 {
                result.push((c_r, m_r));
            } else {
                match DegRevLex.compare(RingRef::new(self), &result.last().unwrap().1, &m_r) {
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
            while i_l < lhs.data.len() && DegRevLex.compare(RingRef::new(self), &lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            match DegRevLex.compare(RingRef::new(self), &result.last().unwrap().1, &m_r) {
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
            data: result
        };
    }

    fn compare_monomial<'a>(&'a self) -> CompareMonomial<RingRef<'a, Self>, DegRevLex> {
        CompareMonomial { poly_ring: RingRef::new(self), order: DegRevLex.clone() }
    }
}

impl<R> PartialEq for MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    fn eq(&self, other: &Self) -> bool {
        // it is not sufficient if base_ring and variable_count match (the rings are isomorphic then),
        // since the monomial indices of elements could point to different values
        std::ptr::eq(self, other)
    }
}

impl<R> RingBase for MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    type Element = MultivariatePolyRingEl<R>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        MultivariatePolyRingEl {
            data: val.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), self.clone_monomial(m))).collect()
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

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut tmp = Vec::new();
        if lhs.data.len() > rhs.data.len() {
            rhs.data.iter().fold(self.zero(), |mut current, (r_c, r_m)| {
                let mut new = self.add_terms(&current, lhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(c, r_c), self.monomial_mul(m.clone(), r_m))), std::mem::replace(&mut tmp, Vec::new()));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        } else {
            // we duplicate it to work better with noncommutative rings (not that this is currently of relevance...)
            lhs.data.iter().fold(self.zero(), |mut current, (l_c, l_m)| {
                let mut new = self.add_terms(&current, rhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(l_c, c), self.monomial_mul(m.clone(), l_m))), std::mem::replace(&mut tmp, Vec::new()));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        }
    }

    fn zero(&self) -> Self::Element {
        MultivariatePolyRingEl {
            data: Vec::new()
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
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        super::generic_impls::print(RingRef::new(self), value, out, env)
    }

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
         where I::Type: IntegerRing 
    {
        self.base_ring().characteristic(ZZ)
    }
}

#[stability::unstable(feature = "enable")]
pub struct TermIterImpl<'a, R>
    where R: RingStore
{
    base_iter: std::slice::Iter<'a, (El<R>, MonomialIdentifier)>
}

impl<'a, R> Iterator for TermIterImpl<'a, R>
    where R: RingStore
{
    type Item = (&'a El<R>, &'a MonomialIdentifier);

    fn next(&mut self) -> Option<Self::Item> {
        self.base_iter.next().map(|(c, m)| (c, m))
    }
}

impl<R> RingExtension for MultivariatePolyRingImplBase<R>
    where R: RingStore
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
                data: vec![(x, self.create_monomial((0..self.variable_count).map(|_| 0)))]
            }
        }
    }
}

impl<R> MultivariatePolyRing for MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    type Monomial = MonomialIdentifier;
    type TermIter<'a> = TermIterImpl<'a, R>
        where Self: 'a;
        
    fn variable_count(&self) -> usize {
        self.variable_count
    }

    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator
    {
        let exponents = exponents.into_iter();
        assert_eq!(exponents.len(), self.variable_count());

        let tmp_monomial = self.tmp_monomial();
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
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let mut rhs = rhs.into_iter().collect::<Vec<_>>();
        rhs.sort_unstable_by(|l, r| DegRevLex.compare(RingRef::new(self), &l.1, &r.1));
        *lhs = self.add_terms(&lhs, rhs.into_iter(), Vec::new());
    }

    fn mul_assign_monomial(&self, f: &mut Self::Element, monomial: Self::Monomial) {
        for (_, m) in &mut f.data {
            *m = self.monomial_mul(m.clone(), &monomial);
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing> {
        match f.data.binary_search_by(|(_, fm)| DegRevLex.compare(RingRef::new(self), fm, m)) {
            Ok(i) => &f.data.at(i).0,
            Err(_) => &self.zero
        }
    }

    #[inline(always)]
    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize {
        match m {
            MonomialIdentifier::Temporary { deg: _ } => self.tmp_monomial()[var_index].get() as usize,
            MonomialIdentifier::Permanent { deg: _, index } => self.allocated_monomials[*index][var_index] as usize
        }
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a> {
        TermIterImpl {
            base_iter: f.data.iter()
        }
    }

    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        match mon {
            MonomialIdentifier::Permanent { deg, index: _ } => *deg as usize,
            MonomialIdentifier::Temporary { deg } => *deg as usize
        }
    }

    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        if order.is_same(&DegRevLex) {
            f.data.last().map(|(c, m)| (c, m))
        } else {
            self.terms(f).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
        }
    }

    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O, lt_than: &Self::Monomial) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        if order.is_same(&DegRevLex) {
            match f.data.binary_search_by(|(_, mon)| DegRevLex.compare(RingRef::new(self), mon, lt_than)) {
                Ok(0) => None,
                Ok(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1)),
                Err(0) => None,
                Err(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1))
            }
        } else {
            self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
        }
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::rings::zn::zn_static::F17;
#[cfg(test)]
use crate::iters::multiset_combinations;

#[cfg(test)]
fn ring_and_elements() -> (MultivariatePolyRingImpl<zn_static::Fp<17>>, Vec<MultivariatePolyRingEl<zn_static::Fp<17>>>) {
    let ring = MultivariatePolyRingImpl::new(F17, 3);
    let els = vec![
        ring.from_terms([]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([0, 0, 0]))]),
        ring.from_terms([(ring.base_ring().neg_one(), ring.create_monomial([0, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().neg_one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0])), (ring.base_ring().one(), ring.create_monomial([0, 1, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([2, 0, 0])), (ring.base_ring().neg_one(), ring.create_monomial([1, 0, 0]))]),
        ring.from_terms([(ring.base_ring().one(), ring.create_monomial([1, 0, 0])), (ring.base_ring().neg_one(), ring.create_monomial([0, 1, 1])), (ring.base_ring().one(), ring.create_monomial([0, 0, 2]))]),
    ];
    return (ring, els);
}

#[test]
fn test_ring_axioms() {
    let (ring, els) = ring_and_elements();
    crate::ring::generic_tests::test_ring_axioms(&ring, els.into_iter());
}

#[test]
fn test_multivariate_axioms() {
    let (ring, _els) = ring_and_elements();
    crate::rings::multivariate_new::generic_tests::test_poly_ring_axioms(&ring, [F17.one(), F17.zero(), F17.int_hom().map(2), F17.neg_one()].into_iter());
}

#[test]
fn test_enumeration_index_degrevlex() {

    fn degrevlex_cmp(lhs: &[u16; 4], rhs: &[u16; 4]) -> Ordering {
        let lhs_deg = lhs[0] + lhs[1] + lhs[2] + lhs[3];
        let rhs_deg = rhs[0] + rhs[1] + rhs[2] + rhs[3];
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..4).rev() {
                if lhs[i] > rhs[i] {
                    return Ordering::Less
                } else if lhs[i] < rhs[i] {
                    return Ordering::Greater;
                }
            }
            return Ordering::Equal;
        }
    }

    let mut all_monomials = multiset_combinations(&[6, 6, 6, 6], 6, |slice| std::array::from_fn::<_, 4, _>(|i| slice[i] as u16)).collect::<Vec<_>>();
    all_monomials.sort_unstable_by(|l, r| degrevlex_cmp(l, r));

    for i in 0..all_monomials.len() {
        assert_eq!(i as u32, enumeration_index_degrevlex(6, &all_monomials[i][..]));
    }
}