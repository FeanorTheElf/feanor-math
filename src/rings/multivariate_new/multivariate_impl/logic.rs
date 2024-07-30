use super::*;

pub struct MonomialData {
    exponents: Box<[u16]>
}

pub struct MultivariatePolyRingCoreData<R: RingStore, O: MonomialOrder> {
    base_ring: R,
    order: O,
    variable_count: usize,
    monomial_data: Vec<MonomialData>
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
}

impl<R: RingStore, O: MonomialOrder> PartialEq for MultivariatePolyRingCoreData<R, O> {
    
    fn eq(&self, other: &Self) -> bool {
        self.variable_count == other.variable_count && self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}
