use std::cell::Cell;

use append_only_vec::AppendOnlyVec;
use thread_local::ThreadLocal;

use super::*;

pub struct MultivariatePolyRingCoreData<R: RingStore, O: MonomialOrder> {
    base_ring: R,
    order: O,
    variable_count: usize,
    allocated_monomials: AppendOnlyVec<Box<[u16]>>,
    tmp_monomials: ThreadLocal<Box<[Cell<u16>]>>
}

impl<R: RingStore, O: MonomialOrder> MultivariatePolyRingCoreData<R, O> {

    pub fn order(&self) -> &O {
        &self.order
    }

    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    pub fn variable_count(&self) -> usize {
        self.variable_count
    }

    pub fn tmp_monomial(&self) -> &[Cell<u16>] {
        self.tmp_monomials.get_or(|| (0..self.variable_count).map(|_| Cell::new(0)).collect::<Vec<_>>().into_boxed_slice())
    }

    pub fn create_monomial<I>(&self, exponents: I) -> usize
        where I: Iterator<Item = u16>
    {
        self.allocated_monomials.push(exponents.collect::<Vec<_>>().into_boxed_slice())
    }
}

impl<R: RingStore, O: MonomialOrder> PartialEq for MultivariatePolyRingCoreData<R, O> {
    
    fn eq(&self, other: &Self) -> bool {
        self.variable_count == other.variable_count && self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}
