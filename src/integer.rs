use crate::primitive::StaticRingBase;
use crate::ring::*;
use crate::euclidean::*;
use crate::ordered::*;

pub trait IntegerRing: EuclideanRing + CanonicalIso<StaticRingBase<i128>> + OrderedRing {

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool;
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize>;
    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize>;
    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize);
    fn mul_pow_2(&self, value: &mut Self::Element, power: usize);
    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, rng: G) -> Self::Element;
}

pub trait IntegerRingWrapper: RingWrapper<Type: IntegerRing> {

    delegate!{ fn abs_is_bit_set(&self, value: &El<Self>, i: usize) -> bool }
    delegate!{ fn abs_highest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ fn abs_lowest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ fn euclidean_div_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ fn mul_pow_2(&self, value: &mut El<Self>, power: usize) -> () }

    fn get_uniformly_random<G: FnMut() -> u64>(&self, bound_exclusive: &El<Self>, mut rng: G) -> El<Self> {
        assert!(self.is_gt(bound_exclusive, &self.zero()));
        let log2_ceil_bound = self.abs_highest_set_bit(bound_exclusive).unwrap() + 1;
        let mut result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        while self.is_geq(&result, bound_exclusive) {
            result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        }
        return result;
    }
}

impl<R> IntegerRingWrapper for R
    where R: RingWrapper<Type: IntegerRing>
{}

#[cfg(test)]
pub fn test_integer_uniformly_random<R: IntegerRingWrapper>(ring: R) {
    for b in [15, 16] {
        let bound = ring.from_z(b);
        let mut rng = oorandom::Rand64::new(0);
        let elements: Vec<El<R>> = (0..1000).map(|_| ring.get_uniformly_random(&bound, || rng.rand_u64())).collect();
        for i in 0..b {
            assert!(elements.iter().any(|x| ring.eq(x, &ring.from_z(i))))
        }
        for x in &elements {
            assert!(ring.is_lt(x, &bound));
        }
    }
}

#[cfg(test)]
pub fn test_integer_axioms<R: IntegerRingWrapper, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) {
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        let mut ceil_pow_2 = ring.from_z(2);
        ring.mul_pow_2(&mut ceil_pow_2, ring.abs_highest_set_bit(a).unwrap_or(0));
        assert!(ring.is_lt(a, &ceil_pow_2));
        assert!(ring.is_lt(&ring.negate(a.clone()), &ceil_pow_2));
        
        for i in 0..ring.abs_highest_set_bit(a).unwrap_or(0) {
            let mut pow_2 = ring.one();
            ring.mul_pow_2(&mut pow_2, i);
            let mut b = a.clone();
            ring.mul_pow_2(&mut b, i);
            assert!(ring.eq(&b, &ring.mul(a.clone(), pow_2.clone())));
            ring.euclidean_div_pow_2(&mut b, i);
            assert!(ring.eq(&b, a));
            ring.euclidean_div_pow_2(&mut b, i);
            ring.println(&b);
            ring.println(&a);
            ring.println(&pow_2);
            println!("{}", i);
            assert!(ring.eq(&b, &ring.euclidean_div(a.clone(), &pow_2)));
        }
    }
}