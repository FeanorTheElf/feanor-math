use std::alloc::{Allocator, Global};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::num::NonZeroU32;

use append_only_vec::AppendOnlyVec;
use atomicbox::AtomicOptionBox;
use thread_local::ThreadLocal;

use crate::algorithms::int_bisect;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate;
use crate::rings::multivariate_new::*;
use crate::integer::{binomial, int_cast, BigIntRing, IntegerRing, IntegerRingStore};
use crate::seq::{VectorFn, VectorView, VectorViewMut};
use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;

type Exponent = u16;
type OrderIdx = u64;
type Index = NonZeroU32;

///
/// Computes `sum_(0 <= l <= k) binomial(n + l, n)`
/// 
fn compute_cum_binomial(n: usize, k: usize) -> u64 {
    StaticRing::<i64>::RING.sum((0..(k + 1)).map(|l| binomial((n + l) as i128, &(n as i128), StaticRing::<i128>::RING) as i64)) as u64
}

///
/// Returns the index of the given monomial within the list of all degree-d monomials, ordered by DegRevLex
/// 
fn enumeration_index_degrevlex<V>(d: Exponent, mon: V, cum_binomial_lookup_table: &[Vec<u64>]) -> u64
    where V: VectorFn<Exponent>
{
    debug_assert!(d == mon.iter().sum());
    let n = mon.len();
    let mut remaining_degree: i64 = d as i64 - 1;
    let mut result = 0;
    for i in 0..(n - 1) {
        remaining_degree -= mon.at(n - 1 - i) as i64;
        if remaining_degree < 0 {
            return result;
        }
        result += cum_binomial_lookup_table[n - i - 2][remaining_degree as usize];
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

impl PartialEq for MonomialIdentifier {
    fn eq(&self, other: &Self) -> bool {
        let res = self.deg == other.deg && self.order == other.order;
        debug_assert!(self.index.is_none() || other.index.is_none() || res == (self.index == other.index));
        return res;
    }
}

impl Eq for MonomialIdentifier {}

impl Ord for MonomialIdentifier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deg.cmp(&other.deg).then_with(|| self.order.cmp(&other.order))
    }
}

impl PartialOrd for MonomialIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingEl<R, A = Global>
    where R: RingStore,
        A: Allocator + Clone
{
    data: Vec<(El<R>, MonomialIdentifier), A>
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingImplBase<R, A = Global>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    base_ring: R,
    variable_count: usize,
    allocated_monomials: AppendOnlyVec<Box<[Exponent]>>,
    // for each thread, the values of the temporary monomial; these may change and are usually set shortly before
    // searching monomial_table; after the search, either the existing index is retrieved, or a new monomial allocated,
    // thus the temporary monomial is not required anymore
    tmp_monomials: ThreadLocal<Box<[Cell<Exponent>]>>,
    // maps monomials to the corresponding indics; usually access with a temporary monomial (one without index)
    monomial_table: RefCell<BTreeMap<MonomialIdentifier, Index>>,
    zero: El<R>,
    max_degree_for_orderidx: usize,
    cum_binomial_lookup_table: Vec<Vec<u64>>,
    allocator: A,
    tmp_poly: AtomicOptionBox<Vec<(El<R>, MonomialIdentifier)>>
}

#[stability::unstable(feature = "enable")]
pub type MultivariatePolyRingImpl<R, A = Global> = RingValue<MultivariatePolyRingImplBase<R, A>>;

impl<R> MultivariatePolyRingImpl<R>
    where R: RingStore
{
    #[stability::unstable(feature = "enable")]
    pub fn new(base_ring: R, variable_count: usize) -> Self {
        Self::new_with(base_ring, variable_count, 64, Global)
    }
}

impl<R, A> MultivariatePolyRingImpl<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(base_ring: R, variable_count: usize, max_supported_deg: Exponent, allocator: A) -> Self {
        assert!(variable_count >= 1);
        // the largest degree for which we have an order-preserving embedding of same-degree monomials into OrderIdx
        let max_degree_for_orderidx = if variable_count == 1 || variable_count == 2 {
            usize::MAX
        } else {
            let k = int_cast(variable_count as i64 - 1, BigIntRing::RING, StaticRing::<i64>::RING);
            // ensure that cum_binomial() always fits within an u64
            int_bisect::find_root_floor(StaticRing::<i64>::RING, 0, |d| if BigIntRing::RING.is_lt(&BigIntRing::RING.mul(
                binomial(int_cast(d + variable_count as i64 - 1, BigIntRing::RING, StaticRing::<i64>::RING), &k, BigIntRing::RING),
                int_cast(*d as i64, BigIntRing::RING, StaticRing::<i64>::RING)
            ), &BigIntRing::RING.power_of_two(u64::BITS as usize)) { -1 } else { 1 }) as usize
        };
        assert!(max_degree_for_orderidx >= max_supported_deg as usize);

        let allocated_monomials = AppendOnlyVec::new();
        // add dummy element so that the index is nonzero, this allows keeping MonomialIdentifier within 16 bytes
        allocated_monomials.push(Vec::new().into_boxed_slice());
        let cum_binomial_lookup_table = (0..(variable_count - 1)).map(|n| (0..=max_supported_deg).map(|k| compute_cum_binomial(n, k as usize)).collect()).collect();
        Self::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring: base_ring,
            variable_count: variable_count,
            allocated_monomials: allocated_monomials,
            tmp_monomials: ThreadLocal::new(),
            monomial_table: RefCell::new(BTreeMap::new()),
            max_degree_for_orderidx: max_degree_for_orderidx,
            cum_binomial_lookup_table: cum_binomial_lookup_table,
            tmp_poly: AtomicOptionBox::none(),
            allocator: allocator,
        })
    }
}

impl<R, A> MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
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

    fn remove_zeros(&self, el: &mut Vec<(El<R>, MonomialIdentifier), A>) {
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
    /// Computes the sum of two elements; rhs may contain zero elements, but must be sorted and not contain equal monomials
    ///
    fn add_terms<I>(&self, lhs: &<Self as RingBase>::Element, rhs_sorted: I, out: Vec<(El<R>, MonomialIdentifier), A>) -> <Self as RingBase>::Element
        where I: Iterator<Item = (El<R>, MonomialIdentifier)>
    {
        debug_assert!(self.is_valid(&lhs.data));
        
        let mut result = out;
        result.clear();
        result.reserve(lhs.data.len() + rhs_sorted.size_hint().0);
        
        let mut lhs_it = lhs.data.iter().peekable();
        let mut rhs_it = rhs_sorted.peekable();

        while let (Some((_, l_m)), Some((_, r_m))) = (lhs_it.peek(), rhs_it.peek()) {
            let next_element = match self.compare_degrevlex(l_m, r_m) {
                Ordering::Equal => {
                    let (l_c, _l_m) = lhs_it.next().unwrap();
                    let (r_c, r_m) = rhs_it.next().unwrap();
                    (self.base_ring().add_ref_fst(l_c, r_c), r_m)
                },
                Ordering::Less => {
                    let (l_c, l_m) = lhs_it.next().unwrap();
                    (self.base_ring().clone_el(l_c), l_m.clone())
                },
                Ordering::Greater => {
                    let (r_c, r_m) = rhs_it.next().unwrap();
                    (r_c, r_m)
                }
            };
            result.push(next_element);
        }
        result.extend(lhs_it.map(|(c, m)| (self.base_ring().clone_el(c), m.clone())));
        result.extend(rhs_it);
        self.remove_zeros(&mut result);
        debug_assert!(self.is_valid(&result));
        return MultivariatePolyRingEl {
            data: result
        };
    }

    fn allocate_tmp_monomial(&self, deg: u16) -> MonomialIdentifier {
        let tmp_monomial = self.tmp_monomial();
        if deg as usize > self.max_degree_for_orderidx {
            unimplemented!("Currently we only support degrees such that the number of monomials of this degree fits into a u64");
        }
        let order_idx = enumeration_index_degrevlex(deg, tmp_monomial.as_fn().map_fn(|e| e.get()), &self.cum_binomial_lookup_table);
        {
            let monomial_table = self.monomial_table.borrow();
            let entry = monomial_table.get(&MonomialIdentifier { deg: deg, order: order_idx, index: None });
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
            monomial_table.insert(MonomialIdentifier { deg: deg, order: order_idx, index: Some(idx) }, idx);
            return MonomialIdentifier { deg: deg, index: Some(idx), order: order_idx };
        }
    }
}

impl<R, A> PartialEq for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    fn eq(&self, other: &Self) -> bool {
        // it is not sufficient if base_ring and variable_count match (the rings are isomorphic then),
        // since the monomial indices of elements could point to different values
        std::ptr::eq(&self.allocated_monomials, &other.allocated_monomials)
    }
}

impl<R, A> RingBase for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    type Element = MultivariatePolyRingEl<R, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(val.data.len(), self.allocator.clone());
        data.extend(val.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), self.clone_monomial(m))));
        MultivariatePolyRingEl {
            data: data
        }
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        debug_assert!(self.is_valid(&rhs.data));
        self.add_terms(lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().clone_el(c), m.clone())), Vec::new_in(self.allocator.clone()))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.add_ref(lhs, rhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.into_iter(), Vec::new_in(self.allocator.clone()));
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.is_valid(&rhs.data));
        *lhs = self.add_terms(&lhs, rhs.data.iter().map(|(c, m)| (self.base_ring().negate(self.base_ring.clone_el(c)), m.clone())), Vec::new_in(self.allocator.clone()));
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
        let mut tmp = Vec::new_in(self.allocator.clone());
        if lhs.data.len() > rhs.data.len() {
            rhs.data.iter().fold(self.zero(), |mut current, (r_c, r_m)| {
                let mut new = self.add_terms(&current, lhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(c, r_c), self.monomial_mul(m.clone(), r_m))), std::mem::replace(&mut tmp, Vec::new_in(self.allocator.clone())));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        } else {
            // we duplicate it to work better with noncommutative rings (not that this is currently of relevance...)
            lhs.data.iter().fold(self.zero(), |mut current, (l_c, l_m)| {
                let mut new = self.add_terms(&current, rhs.data.iter().map(|(c, m)| (self.base_ring().mul_ref(l_c, c), self.monomial_mul(m.clone(), l_m))), std::mem::replace(&mut tmp, Vec::new_in(self.allocator.clone())));
                std::mem::swap(&mut new, &mut current);
                std::mem::swap(&mut new.data, &mut tmp);
                current
            })
        }
    }

    fn zero(&self) -> Self::Element {
        MultivariatePolyRingEl {
            data: Vec::new_in(self.allocator.clone())
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

impl<R, A> RingExtension for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
{
    type BaseRing = R;

    fn base_ring<'b>(&'b self) -> &'b Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        if self.base_ring().is_zero(&x) {
            return self.zero();
        } else {
            let mut data = Vec::with_capacity_in(1, self.allocator.clone());
            data.push((x, self.create_monomial((0..self.variable_count).map(|_| 0))));
            return MultivariatePolyRingEl { data };
        }
    }
}

impl<R, A> MultivariatePolyRing for MultivariatePolyRingImplBase<R, A>
    where R: RingStore,
        A: Clone + Allocator + Send
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
        return self.allocate_tmp_monomial(deg);
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        mon.clone()
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, terms: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let terms = terms.into_iter();
        let mut rhs = self.tmp_poly.swap(None, std::sync::atomic::Ordering::AcqRel).map(|b| *b).unwrap_or(Vec::new());
        debug_assert!(rhs.len() == 0);
        rhs.extend(terms);
        rhs.sort_unstable_by(|l, r| self.compare_degrevlex(&l.1, &r.1));
        rhs.dedup_by(|(snd_c, snd_m), (fst_c, fst_m)| {
            if self.compare_degrevlex(&fst_m, &snd_m) == Ordering::Equal {
                self.base_ring().add_assign_ref(fst_c, snd_c);
                return true;
            } else {
                return false;
            }
        });
        *lhs = self.add_terms(&lhs, rhs.drain(..), Vec::new_in(self.allocator.clone()));
        self.tmp_poly.swap(Some(Box::new(rhs)), std::sync::atomic::Ordering::AcqRel);
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
            let res = match f.data.binary_search_by(|(_, mon)| self.compare_degrevlex(mon, lt_than)) {
                Ok(0) => None,
                Ok(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1)),
                Err(0) => None,
                Err(i) => Some((&f.data[i - 1].0, &f.data[i - 1].1))
            };
            assert!({
                let expected = self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1));
                (res.is_none() && expected.is_none()) || std::ptr::eq(res.unwrap().0, expected.unwrap().0)
            });
            return res;
        } else {
            self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
        }
    }

    fn monomial_mul(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        let l_i = u32::from(lhs.index.unwrap()) as usize;
        let r_i = u32::from(rhs.index.unwrap()) as usize;
        let tmp_monomial = self.tmp_monomial();
        for i in 0..self.variable_count {
            tmp_monomial[i].set(self.allocated_monomials[l_i][i] + self.allocated_monomials[r_i][i]);
        }
        return self.allocate_tmp_monomial(lhs.deg + rhs.deg);
    }
}

impl<P, R, A> CanHomFrom<P> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        P: MultivariatePolyRing,
        R::Type: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        if self.variable_count() >= from.variable_count() {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &<P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        RingRef::new(self).from_terms(from.terms(el).map(|(c, m)| (
            self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), c, hom),
            self.create_monomial((0..self.variable_count()).map(|i| if i < from.variable_count() { from.exponent_at(m, i) } else { 0 }))
        )))
    }
}

impl<R2, A2, O2, R, A, const N2: usize> CanHomFrom<multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        R2: RingStore,
        O2: multivariate::MonomialOrder,
        A2: Allocator + Clone,
        R::Type: CanHomFrom<R2::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<R2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>) -> Option<Self::Homomorphism> {
        if self.variable_count() >= <_ as multivariate::MultivariatePolyRing>::indeterminate_len(from) {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: <multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: &<multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        RingRef::new(self).from_terms(<_ as multivariate::MultivariatePolyRing>::terms(from, el).map(|(c, m)| (
            self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), c, hom),
            self.create_monomial((0..self.variable_count()).map(|i| if i < <_ as multivariate::MultivariatePolyRing>::indeterminate_len(from) { m[i] as usize } else { 0 }))
        )))
    }
}

impl<P, R, A> CanIsoFromTo<P> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        P: MultivariatePolyRing,
        R::Type: CanIsoFromTo<<P::BaseRing as RingStore>::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<<P::BaseRing as RingStore>::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        if self.variable_count() == from.variable_count() {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &P, el: Self::Element, iso: &Self::Isomorphism) -> <P as RingBase>::Element {
        RingRef::new(from).from_terms(self.terms(&el).map(|(c, m)| (
            self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(c), iso),
            from.create_monomial((0..self.variable_count()).map(|i| self.exponent_at(m, i)))
        )))
    }
}

impl<R2, A2, O2, R, A, const N2: usize> CanIsoFromTo<multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>> for MultivariatePolyRingImplBase<R, A> 
    where R: RingStore,
        A: Clone + Allocator + Send,
        R2: RingStore,
        O2: multivariate::MonomialOrder,
        A2: Allocator + Clone,
        R::Type: CanIsoFromTo<R2::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<R2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>) -> Option<Self::Isomorphism> {
        if self.variable_count() == <_ as multivariate::MultivariatePolyRing>::indeterminate_len(from) {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <multivariate::ordered::MultivariatePolyRingImplBase<R2, O2, N2, A2> as RingBase>::Element {
        <_ as multivariate::MultivariatePolyRingStore>::from_terms(&RingRef::new(from), self.terms(&el).map(|(c, m)| (
            self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(c), iso),
            <_ as multivariate::MultivariatePolyRing>::create_monomial(from, (0..self.variable_count()).map(|i| self.exponent_at(m, i) as u16))
        )))
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

    let cum_binomial_lookup_table = (0..4).map(|n| (0..7).map(|k| compute_cum_binomial(n, k)).collect::<Vec<_>>()).collect::<Vec<_>>();

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
        assert_eq!(i as u64, enumeration_index_degrevlex(6, (&all_monomials[i]).into_fn(|x| *x), &cum_binomial_lookup_table));
    }
}

#[test]
fn test_monomial_small() {
    assert_eq!(16, std::mem::size_of::<MonomialIdentifier>());
}

#[test]
fn test_new_many_variables() {
    for m in 1..32 {
        println!("{}", m);
        let ring = MultivariatePolyRingImpl::new_with(StaticRing::<i64>::RING, m, 32, Global);
        assert_eq!(m, ring.variable_count());
    }
}