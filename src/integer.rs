use crate::ring::*;
use crate::euclidean::*;
use crate::ordered::*;
use crate::primitive_int::*;
use crate::algorithms;

///
/// Trait for rings that are isomorphic to the ring of integers `ZZ = { ..., -2, -1, 0, 1, 2, ... }`.
/// 
/// Some of the functionality in this trait refers to the binary expansion of
/// a positive integer. While this is not really general, it is often required
/// for fast operations with integers.
/// 
pub trait IntegerRing: EuclideanRing + OrderedRing + HashableElRing + SelfIso + CanonicalIso<StaticRingBase<i32>> {

    fn to_float_approx(&self, value: &Self::Element) -> f64;
    fn from_float_approx(&self, value: f64) -> Option<Self::Element>;
    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool;
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize>;
    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize>;
    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize);
    fn mul_pow_2(&self, value: &mut Self::Element, power: usize);
    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, rng: G) -> Self::Element;

    fn power_of_two(&self, power: usize) -> Self::Element {
        let mut result = self.one();
        self.mul_pow_2(&mut result, power);
        return result;
    }
}

// impl<I: IntegerRing + CanonicalIso<I> + ?Sized, J: IntegerRing + ?Sized> CanonicalHom<I> for J {

//     type Homomorphism = ();
    
//     default fn has_canonical_hom(&self, _: &I) -> Option<()> { Some(()) }

//     default fn map_in(&self, from: &I, el: I::Element, _: &()) -> Self::Element {
//         let result = algorithms::sqr_mul::generic_abs_square_and_multiply(self.one(), &el, RingRef::new(from), |a, b| self.add(a, b), |a, b| self.add_ref(a, b), self.zero());
//         if from.is_neg(&el) {
//             return self.negate(result);
//         } else {
//             return result;
//         }
//     }
// }

// impl<I: IntegerRing + CanonicalIso<I> + ?Sized, J: IntegerRing + CanonicalIso<J> + ?Sized> CanonicalIso<I> for J {

//     type Isomorphism = ();
    
//     default fn has_canonical_iso(&self, _: &I) -> Option<()> { Some(()) }

//     default fn map_out(&self, from: &I, el: Self::Element, _: &()) -> I::Element {
//         let result = algorithms::sqr_mul::generic_abs_square_and_multiply(from.one(), &el, RingRef::new(self), |a, b| from.add(a, b), |a, b| from.add_ref(a, b), from.zero());
//         if self.is_neg(&el) {
//             return from.negate(result);
//         } else {
//             return result;
//         }
//     }
// }

pub trait IntegerRingStore: RingStore
    where Self::Type: IntegerRing
{
    delegate!{ fn to_float_approx(&self, value: &El<Self>) -> f64 }
    delegate!{ fn from_float_approx(&self, value: f64) -> Option<El<Self>> }
    delegate!{ fn abs_is_bit_set(&self, value: &El<Self>, i: usize) -> bool }
    delegate!{ fn abs_highest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ fn abs_lowest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ fn euclidean_div_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ fn mul_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ fn power_of_two(&self, power: usize) -> El<Self> }

    fn get_uniformly_random<G: FnMut() -> u64>(&self, bound_exclusive: &El<Self>, mut rng: G) -> El<Self> {
        assert!(self.is_gt(bound_exclusive, &self.zero()));
        let log2_ceil_bound = self.abs_highest_set_bit(bound_exclusive).unwrap() + 1;
        let mut result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        while self.is_geq(&result, bound_exclusive) {
            result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        }
        return result;
    }

    fn abs_log2_ceil(&self, value: &El<Self>) -> Option<usize> {
        let highest_bit = self.abs_highest_set_bit(value)?;
        if self.abs_lowest_set_bit(value).unwrap() == highest_bit {
            return Some(highest_bit);
        } else {
            return Some(highest_bit + 1);
        }
    }
}

impl<R> IntegerRingStore for R
    where R: RingStore,
        R::Type: IntegerRing
{}

#[cfg(test)]
pub fn generic_test_integer_uniformly_random<R: IntegerRingStore>(ring: R) 
    where R::Type: IntegerRing
{
    for b in [15, 16] {
        let bound = ring.from_int(b);
        let mut rng = oorandom::Rand64::new(0);
        let elements: Vec<El<R>> = (0..1000).map(|_| ring.get_uniformly_random(&bound, || rng.rand_u64())).collect();
        for i in 0..b {
            assert!(elements.iter().any(|x| ring.eq(x, &ring.from_int(i))))
        }
        for x in &elements {
            assert!(ring.is_lt(x, &bound));
        }
    }
}

#[cfg(test)]
pub fn generic_test_integer_axioms<R: IntegerRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) 
    where R::Type: IntegerRing
{
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        let mut ceil_pow_2 = ring.from_int(2);
        ring.mul_pow_2(&mut ceil_pow_2, ring.abs_highest_set_bit(a).unwrap_or(0));
        assert!(ring.is_lt(a, &ceil_pow_2));
        assert!(ring.is_lt(&ring.negate(ring.clone(a)), &ceil_pow_2));
        
        for i in 0..ring.abs_highest_set_bit(a).unwrap_or(0) {
            let mut pow_2 = ring.one();
            ring.mul_pow_2(&mut pow_2, i);
            let mut b = ring.clone(a);
            ring.mul_pow_2(&mut b, i);
            assert!(ring.eq(&b, &ring.mul(ring.clone(a), ring.clone(&pow_2))));
            ring.euclidean_div_pow_2(&mut b, i);
            assert!(ring.eq(&b, a));
            ring.euclidean_div_pow_2(&mut b, i);
            assert!(ring.eq(&b, &ring.euclidean_div(ring.clone(a), &pow_2)));
        }
    }
}