use std::cell::{Cell, RefCell};
use std::num::NonZeroU32;

use append_only_vec::AppendOnlyVec;
use thread_local::ThreadLocal;

use crate::algorithms::int_bisect;
use crate::extcmpmap::ExtCmpBTreeMap;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate_new::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::seq::{VectorFn, VectorView, VectorViewMut};

type Exponent = u16;
type OrderIdx = u64;
type Index = NonZeroU32;

thread_local!{
    static BINOMIAL_COEFF_LOOKUP_TABLE: RefCell<Vec<Vec<u64>>> = RefCell::new(Vec::new());
    static CUM_BINOMIAL_COEFF_LOOKUP_TABLE: RefCell<Vec<Vec<u64>>> = RefCell::new(Vec::new());
}

fn compute_binomial(n: i64, k: i64) -> i128 {
    assert!(k <= n);
    let n = n as i128;
    let k = k as i128;
    StaticRing::<i128>::RING.prod((n - k + 1)..(n + 1)) / StaticRing::<i128>::RING.prod(1..(k + 1))
}

fn binomial(n: usize, k: usize) -> u64 {
    assert!(k <= n);
    BINOMIAL_COEFF_LOOKUP_TABLE.with_borrow_mut(|table| {
        if table.len() <= n {
            table.resize_with(n + 1, || Vec::new());
        }
        let table_for_n = &mut table[n];
        while table_for_n.len() <= k {
            table_for_n.push(u64::try_from(compute_binomial(n as i64, table_for_n.len() as i64)).unwrap());
        }
        return table_for_n[k];
    })
}

///
/// Computes `sum_(0 <= l <= k) binomial(n + l, n)`
/// 
fn cum_binomial(n: usize, k: i64) -> u64 {

    if k < 0 {
        return 0;
    }

    fn compute_cum_binomial(n: usize, k: usize) -> u64 {
        StaticRing::<i64>::RING.sum((0..(k + 1)).map(|l| binomial(n + l, n) as i64)) as u64
    }

    CUM_BINOMIAL_COEFF_LOOKUP_TABLE.with_borrow_mut(|table| {
        if table.len() <= n {
            table.resize_with(n + 1, || Vec::new());
        }
        let table_for_n = &mut table[n];
        while table_for_n.len() <= k as usize {
            table_for_n.push(compute_cum_binomial(n, table_for_n.len()));
        }
        return table_for_n[k as usize];
    })
}

///
/// Returns the index of the given monomial within the list of all degree-d monomials, ordered by DegRevLex
/// 
fn enumeration_index_degrevlex<V>(d: Exponent, mon: V) -> u64
    where V: VectorFn<Exponent>
{
    debug_assert!(d == mon.iter().sum());
    let n = mon.len();
    let mut remaining_degree: i64 = d as i64 - 1;
    let mut result = 0;
    for i in 0..(n - 1) {
        remaining_degree -= mon.at(n - 1 - i) as i64;
        result += cum_binomial(n - i - 2, remaining_degree);
    }
    return result;
}

#[stability::unstable(feature = "enable")]
#[derive(Clone)]
pub struct MonomialIdentifier {
    deg: Exponent,
    order: OrderIdx,
    index: Option<Index>
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingEl<R>
    where R: RingStore
{
    data: Vec<(El<R>, MonomialIdentifier)>
}

struct CompareMonomialDegRevLexFamily<R: RingStore> {
    base_ring: PhantomData<R>
}

impl<R: RingStore> CompareFnFamily<MonomialIdentifier> for CompareMonomialDegRevLexFamily<R> {
    type CompareFn<'a> = CompareMonomialDegRevLex<'a, R>
        where Self: 'a;
}

struct CompareMonomialDegRevLex<'a, R: RingStore> {
    poly_ring: &'a MultivariatePolyRingImplBase<R>
}

impl<'a, 'b, R: RingStore> FnOnce<(&'b MonomialIdentifier, &'b MonomialIdentifier)> for CompareMonomialDegRevLex<'a, R> {
    type Output = Ordering;

    extern "rust-call" fn call_once(self, args: (&'b MonomialIdentifier, &'b MonomialIdentifier)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R: RingStore> FnMut<(&'b MonomialIdentifier, &'b MonomialIdentifier)> for CompareMonomialDegRevLex<'a, R> {
    extern "rust-call" fn call_mut(&mut self, args: (&'b MonomialIdentifier, &'b MonomialIdentifier)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R: RingStore> Fn<(&'b MonomialIdentifier, &'b MonomialIdentifier)> for CompareMonomialDegRevLex<'a, R> {
    extern "rust-call" fn call(&self, args: (&'b MonomialIdentifier, &'b MonomialIdentifier)) -> Self::Output {
        self.poly_ring.compare_degrevlex(args.0, args.1)
    }
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    base_ring: R,
    variable_count: usize,
    allocated_monomials: AppendOnlyVec<Box<[Exponent]>>,
    // for each thread, the values of the temporary monomial; these may change and are usually set shortly before
    // searching monomial_table; after the search, either the existing index is retrieved, or a new monomial allocated,
    // thus the temporary monomial is not required anymore
    tmp_monomials: ThreadLocal<Box<[Cell<Exponent>]>>,
    // maps monomials to the corresponding indics; usually access with a temporary monomial (one without index)
    monomial_table: RefCell<ExtCmpBTreeMap<MonomialIdentifier, Index, CompareMonomialDegRevLexFamily<R>>>,
    zero: El<R>,
    max_degree_for_orderidx: usize
}

pub type MultivariatePolyRingImpl<R> = RingValue<MultivariatePolyRingImplBase<R>>;

impl<R> MultivariatePolyRingImpl<R>
    where R: RingStore
{
    #[stability::unstable(feature = "enable")]
    pub fn new(base_ring: R, variable_count: usize) -> Self {
        assert!(variable_count >= 1);
        let max_degree_for_orderidx = if variable_count == 1 || variable_count == 2 {
            usize::MAX
        } else {
            int_bisect::find_root_floor(StaticRing::<i64>::RING, 0, |d| if compute_binomial(d + variable_count as i64 - 1, variable_count as i64 - 1) < u64::MAX as i128 { -1 } else { 1 }) as usize
        };
        let allocated_monomials = AppendOnlyVec::new();
        // add dummy element so that the index is nonzero
        allocated_monomials.push(Vec::new().into_boxed_slice());
        Self::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring: base_ring,
            variable_count: variable_count,
            allocated_monomials: allocated_monomials,
            tmp_monomials: ThreadLocal::new(),
            monomial_table: RefCell::new(ExtCmpBTreeMap::new()),
            max_degree_for_orderidx: max_degree_for_orderidx
        })
    }
}

impl<R> MultivariatePolyRingImplBase<R>
    where R: RingStore
{
    fn tmp_monomial(&self) -> &[Cell<u16>] {
        self.tmp_monomials.get_or(|| (0..self.variable_count).map(|_| Cell::new(0)).collect::<Vec<_>>().into_boxed_slice())
    }

    fn compare_degrevlex(&self, lhs: &MonomialIdentifier, rhs: &MonomialIdentifier) -> Ordering {
        let res = lhs.deg.cmp(&rhs.deg).then_with(|| lhs.order.cmp(&rhs.order));
        debug_assert!(res == DegRevLex.compare(RingRef::new(self), lhs, rhs));
        return res;
    }

    fn is_valid(&self, el: &[(El<R>, MonomialIdentifier)]) -> bool {
        for (_, m) in el {
            assert!(m.index.is_some());
            assert_eq!(m.deg, self.allocated_monomials[u32::from(m.index.unwrap()) as usize].iter().copied().sum::<Exponent>());
        }
        for i in 1..el.len() {
            if self.compare_degrevlex(&el[i - 1].1, &el[i].1) != Ordering::Less {
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
                    let tmp = std::mem::replace(el.at_mut(j), (self.base_ring.zero(), MonomialIdentifier { deg: 0, order: 0, index: None }));
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
            while i_l < lhs.data.len() && self.compare_degrevlex(&lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            if result.len() == 0 {
                result.push((c_r, m_r));
            } else {
                match self.compare_degrevlex(&result.last().unwrap().1, &m_r) {
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
            while i_l < lhs.data.len() && self.compare_degrevlex(&lhs.data.at(i_l).1, &m_r) != Ordering::Greater {
                result.push((self.base_ring.clone_el(&lhs.data.at(i_l).0), lhs.data.at(i_l).1.clone()));
                i_l += 1;
            }
            match self.compare_degrevlex(&result.last().unwrap().1, &m_r) {
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

    fn compare_monomial<'a>(&'a self) -> CompareMonomialDegRevLex<'a, R> {
        CompareMonomialDegRevLex { poly_ring: self }
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
            if lhs.data.at(i).1.index != rhs.data.at(i).1.index || !self.base_ring.eq_el(&lhs.data.at(i).0, &rhs.data.at(i).0) {
                return false;
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
        value.data[0].1.deg == 0 && self.base_ring().is_one(&value.data[0].0)
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.is_valid(&value.data));
        if value.data.len() != 1 {
            return false;
        }
        value.data[0].1.deg == 0 && self.base_ring().is_neg_one(&value.data[0].0)
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
        if deg as usize > self.max_degree_for_orderidx {
            unimplemented!("Currently we only support degrees such that the number of monomials of this degree fits into a u64");
        }
        let order_idx = enumeration_index_degrevlex(deg, tmp_monomial.as_fn().map_fn(|e| e.get()));
        {
            let monomial_table = self.monomial_table.borrow();
            let entry = monomial_table.get(&MonomialIdentifier { deg: deg, order: order_idx, index: None }, self.compare_monomial());
            // do we have the monomial already allocated?
            if let Some(idx) = entry {
                return MonomialIdentifier { deg: deg, index: Some(*idx), order: order_idx };
            }
        }
        {
            let mut monomial_table = self.monomial_table.borrow_mut();
            // if not, allocate it!
            let idx = self.allocated_monomials.push(tmp_monomial.iter().map(|e| e.get()).collect::<Vec<_>>().into_boxed_slice());
            let idx = NonZeroU32::new(u32::try_from(idx).unwrap()).unwrap();
            monomial_table.insert(MonomialIdentifier { deg: deg, index: Some(idx), order: order_idx }, idx, self.compare_monomial());
            return MonomialIdentifier { deg: deg, index: Some(idx), order: order_idx };
        }
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        mon.clone()
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let mut rhs = rhs.into_iter().collect::<Vec<_>>();
        rhs.sort_unstable_by(|l, r| self.compare_degrevlex(&l.1, &r.1));
        *lhs = self.add_terms(&lhs, rhs.into_iter(), Vec::new());
    }

    fn mul_assign_monomial(&self, f: &mut Self::Element, monomial: Self::Monomial) {
        for (_, m) in &mut f.data {
            *m = self.monomial_mul(m.clone(), &monomial);
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing> {
        match f.data.binary_search_by(|(_, fm)| self.compare_degrevlex(fm, m)) {
            Ok(i) => &f.data.at(i).0,
            Err(_) => &self.zero
        }
    }

    #[inline(always)]
    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize {
        match m.index {
            None => self.tmp_monomial()[var_index].get() as usize,
            Some(index) => self.allocated_monomials[u32::from(index) as usize][var_index] as usize
        }
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a> {
        TermIterImpl {
            base_iter: f.data.iter()
        }
    }

    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        mon.deg as usize
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
            match f.data.binary_search_by(|(_, mon)| self.compare_degrevlex(mon, lt_than)) {
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
use crate::homomorphism::*;

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
        assert_eq!(i as u64, enumeration_index_degrevlex(6, (&all_monomials[i]).into_fn(|x| *x)));
    }
}

#[test]
fn test_monomial_small() {
    assert_eq!(16, std::mem::size_of::<MonomialIdentifier>());
}