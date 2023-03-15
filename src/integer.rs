use crate::primitive::StaticRingBase;
use crate::ring::*;
use crate::euclidean::*;
use crate::ordered::*;

pub trait IntegerRing: EuclideanRing + CanonicalIso<StaticRingBase<i128>> + OrderedRing {

    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool;
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize>;
    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize);
    fn mul_pow_2(&self, value: &mut Self::Element, power: usize);
}

pub trait IntegerRingWrapper: RingWrapper<Type: IntegerRing> {

    delegate!{ fn abs_is_bit_set(&self, value: &El<Self>, i: usize) -> bool }
    delegate!{ fn abs_highest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ fn euclidean_div_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ fn mul_pow_2(&self, value: &mut El<Self>, power: usize) -> () }

}

impl<R> IntegerRingWrapper for R
    where R: RingWrapper<Type: IntegerRing>
{}

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