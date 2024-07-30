use std::cell::RefCell;
use std::ops::Deref;
use std::{collections::BTreeMap, marker::PhantomData, pin::Pin};

use logic::MultivariatePolyRingCoreData;
use xselfref::{opaque, Holder, NonSelfrefPart, Opaque, OperateIn};

mod logic;

use super::*;
use crate::integer::{IntegerRing, IntegerRingStore};

#[stability::unstable(feature = "enable")]
pub enum MonomialIdentifier {
    Permanent { 
        deg: u16,
        index: usize
    },
    Temporary {
        deg: u16
    }
}

struct MonomialKey<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    monomial_ref: MonomialIdentifier,
    ring: MultivariatePolyRingImpl<'outer, R, O, &'ring MultivariatePolyRingCore<'ring, 'outer, R, O>>
}

impl<'outer, 'ring, R, O> PartialOrd for MonomialKey<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'outer, 'ring, R, O> PartialEq for MonomialKey<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<'outer, 'ring, R, O> Ord for MonomialKey<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.ring.order().compare(RingRef::new(&self.ring), &self.monomial_ref, &other.monomial_ref)
    }
}

impl<'outer, 'ring, R, O> Eq for MonomialKey<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingCore<'this, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    main_data: MultivariatePolyRingCoreData<R, O>,
    monomial_table: RefCell<BTreeMap<MonomialKey<'this, 'outer, R, O>, usize>>
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingKey<'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    base_ring: PhantomData<&'outer R>,
    order: PhantomData<&'outer O>
}

impl<'outer, R, O> NonSelfrefPart for MultivariatePolyRingKey<'outer, R, O> 
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    type Part = MultivariatePolyRingCoreData<R, O>;

    fn deref_part<'b, 'c>(data: &'b Self::Kind<'c>) -> &'b Self::Part
        where 'c: 'b
    {
        &data.main_data
    }
}

opaque!{
    impl['outer, R: 'outer + RingStore, O: 'outer + MonomialOrder] Opaque for MultivariatePolyRingKey<'outer, R, O> {
        type Kind<'this> = MultivariatePolyRingCore<'this, 'outer, R, O>;
    }
}

#[stability::unstable(feature = "enable")]
pub trait MultivariatePolyRingCoreContainer<'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn with_core<F, T>(&self, f: F) -> T
        where F: for<'b> FnOnce(&'b MultivariatePolyRingCore<'b, 'outer, R, O>) -> T;

    fn get_core_data<'b>(&'b self) -> &'b MultivariatePolyRingCoreData<R, O>;
}

impl<'outer, R, O> MultivariatePolyRingCoreContainer<'outer, R, O> for Pin<Box<Holder<'outer, MultivariatePolyRingKey<'outer, R, O>>>>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn get_core_data<'b>(&'b self) -> &'b MultivariatePolyRingCoreData<R, O> {
        self.as_ref().deref_part()
    }

    fn with_core<F, T>(&self, f: F) -> T
        where F: for<'b> FnOnce(&'b MultivariatePolyRingCore<'b, 'outer, R, O>) -> T
    {
        fn call_f<'outer, 'b, R, O, F, T>(f: F, arg: OperateIn<'b, MultivariatePolyRingKey<'outer, R, O>>) -> T
            where F: for<'c> FnOnce(&'c MultivariatePolyRingCore<'c, 'outer, R, O>) -> T,
                R: 'outer + RingStore,
                O: 'outer + MonomialOrder
        {
            let ring_ref: &Pin<&'b <MultivariatePolyRingKey<'outer, R, O> as Opaque>::Kind<'b>> = arg.deref();
            let converted_ring_ref: &Pin<&'b MultivariatePolyRingCore<'b, 'outer, R, O>> = ring_ref;
            f(converted_ring_ref.get_ref())
        }
        self.as_ref().operate_in(|ring| call_f(f, ring))
    }
}

impl<'ring, 'outer, R, O> MultivariatePolyRingCoreContainer<'outer, R, O> for &'ring MultivariatePolyRingCore<'ring, 'outer, R, O>
    where R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn get_core_data<'b>(&'b self) -> &'b MultivariatePolyRingCoreData<R, O> {
        &self.main_data
    }

    fn with_core<F, T>(&self, f: F) -> T
        where F: for<'b> FnOnce(&'b MultivariatePolyRingCore<'b, 'outer, R, O>) -> T
    {
        f(self)
    }
}

#[stability::unstable(feature = "enable")]
pub struct MultivariatePolyRingImpl<'outer, R, O, Internal = Pin<Box<Holder<'outer, MultivariatePolyRingKey<'outer, R, O>>>>>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    core_container: Internal,
    core: PhantomData<MultivariatePolyRingCore<'outer, 'outer, R, O>>
}

impl<'outer, R, O, Internal> MultivariatePolyRingImpl<'outer, R, O, Internal>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn order<'b>(&'b self) -> &'b O {
        self.core_container.get_core_data().order()
    }
}

impl<'outer, R, O, Internal> PartialEq for MultivariatePolyRingImpl<'outer, R, O, Internal>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    fn eq(&self, other: &Self) -> bool {
        self.core_container.with_core(|self_ring| other.core_container.with_core(|other_ring|
            self_ring.main_data == other_ring.main_data
        ))
    }
}

impl<'outer, R, O, Internal> RingBase for MultivariatePolyRingImpl<'outer, R, O, Internal>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    type Element = ();

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        unimplemented!()
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        unimplemented!()
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        unimplemented!()
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        unimplemented!()
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        unimplemented!()
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        unimplemented!()
    }

    fn from_int(&self, value: i32) -> Self::Element {
        unimplemented!()
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        unimplemented!()
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        unimplemented!()
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        unimplemented!()
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        unimplemented!()
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

impl<'outer, R, O, Internal> RingExtension for MultivariatePolyRingImpl<'outer, R, O, Internal>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    type BaseRing = R;

    fn base_ring<'b>(&'b self) -> &'b Self::BaseRing {
        self.core_container.get_core_data().base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        unimplemented!()
    }
}

impl<'outer, R, O, Internal> MultivariatePolyRing for MultivariatePolyRingImpl<'outer, R, O, Internal>
    where Internal: MultivariatePolyRingCoreContainer<'outer, R, O>,
        R: 'outer + RingStore,
        O: 'outer + MonomialOrder
{
    type Monomial = MonomialIdentifier;
    type TermIter<'a> = TermIterImpl<'a, R, O>
        where Self: 'a;
        
    fn variable_count(&self) -> usize {
        self.core_container.get_core_data().variable_count()
    }

    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: ExactSizeIterator<Item = usize>
    {
        assert_eq!(exponents.len(), self.variable_count());

        fn compute_inner<'outer, 'ring, R, O, I>(exponents: I, ring_ptr: &'ring MultivariatePolyRingCore<'ring, 'outer, R, O>) -> MonomialIdentifier
            where R: 'outer + RingStore,
                O: 'outer + MonomialOrder,
                I: ExactSizeIterator<Item = usize>
        {
            let tmp_monomial = ring_ptr.main_data.tmp_monomial();
            let mut deg = 0;
            for (i, e) in exponents.enumerate() {
                deg += e as u16;
                tmp_monomial[i].set(e as u16);
            }
            {
                let monomial_table = ring_ptr.monomial_table.borrow();
                let entry = monomial_table.get(&MonomialKey {
                    ring: MultivariatePolyRingImpl { core_container: ring_ptr, core: PhantomData },
                    monomial_ref: MonomialIdentifier::Temporary { deg: deg }
                });
                // do we have the monomial already allocated?
                if let Some(idx) = entry {
                    return MonomialIdentifier::Permanent { deg: deg, index: *idx };
                }
            }
            {
                let mut monomial_table = ring_ptr.monomial_table.borrow_mut();
                // if not, allocate it!
                let idx = ring_ptr.main_data.create_monomial(tmp_monomial.iter().map(|e| e.get()));
                monomial_table.insert(MonomialKey {
                    ring: MultivariatePolyRingImpl { core_container: ring_ptr, core: PhantomData },
                    monomial_ref: MonomialIdentifier::Permanent { deg: deg, index: idx  }
                }, idx);
                return MonomialIdentifier::Permanent { deg: deg, index: idx };
            }
        }

        self.core_container.with_core(|ring_ptr| compute_inner(exponents, ring_ptr))
    }

    fn create_term(&self, coeff: El<Self::BaseRing>, monomial: Self::Monomial) -> Self::Element {
        unimplemented!()
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing> {
        unimplemented!()
    }

    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize {
        unimplemented!()
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a> {
        unimplemented!()
    }
}