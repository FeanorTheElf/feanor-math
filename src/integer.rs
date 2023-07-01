use crate::ring::*;
use crate::euclidean::*;
use crate::ordered::*;
use crate::primitive_int::*;

///
/// Trait for rings that are isomorphic to the ring of integers `ZZ = { ..., -2, -1, 0, 1, 2, ... }`.
/// 
/// Some of the functionality in this trait refers to the binary expansion of
/// a positive integer. While this is not really general, it is often required
/// for fast operations with integers.
/// 
/// As an additional requirement, the euclidean division (i.e. [`EuclideanRing::euclidean_div_rem()`] and
/// [`IntegerRing::euclidean_div_pow_2()`]) are expected to round towards zero.
/// 
pub trait IntegerRing: EuclideanRing + OrderedRing + HashableElRing + SelfIso + CanonicalIso<StaticRingBase<i32>> + CanonicalIntegerHom {

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

pub trait CanonicalIntegerHom: RingBase {
    
    fn map_in_int<R: IntegerRing>(&self, from: &R, el: R::Element) -> Self::Element {
        let result = crate::algorithms::sqr_mul::generic_abs_square_and_multiply(
            self.one(), 
            &el, 
            RingRef::new(from), 
            |a| self.add_ref(&a, &a), 
            |a, b| self.add_ref_fst(a, b), 
            self.zero()
        );
        if from.is_neg(&el) {
            return self.negate(result);
        } else {
            return result;
        }
    }

    fn map_in_int_ref<R: IntegerRing>(&self, from: &R, el: &R::Element) -> Self::Element {
        self.map_in_int(from, from.clone_el(el))
    }

    fn mul_assign_map_in_int<R: IntegerRing>(&self, from: &R, lhs: &mut Self::Element, rhs: R::Element) {
        self.mul_assign(lhs, self.map_in_int(from, rhs));
    }

    fn mul_assign_map_in_int_ref<R: IntegerRing>(&self, from: &R, lhs: &mut Self::Element, rhs: &R::Element) {
        self.mul_assign(lhs, self.map_in_int_ref(from, rhs));
    }
}

macro_rules! impl_from_integer_canonical_hom {
    ($from:ty => $to:ty; $($constraints:tt)*) => {
        impl<$($constraints)*> CanonicalHom<$from> for $to {

            type Homomorphism = ();

            fn has_canonical_hom(&self, _from: &$from) -> Option<Self::Homomorphism> { Some(()) }
    
            fn map_in(&self, from: &$from, el: <$from as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
                <Self as CanonicalIntegerHom>::map_in_int::<$from>(self, from, el)
            }
        
            fn map_in_ref(&self, from: &$from, el: &<$from as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
                <Self as CanonicalIntegerHom>::map_in_int_ref::<$from>(self, from, el)
            }
        
            fn mul_assign_map_in(&self, from: &$from, lhs: &mut Self::Element, rhs: <$from as RingBase>::Element, _hom: &Self::Homomorphism) {
                <Self as CanonicalIntegerHom>::mul_assign_map_in_int::<$from>(self, from, lhs, rhs)
            }
        
            fn mul_assign_map_in_ref(&self, from: &$from, lhs: &mut Self::Element, rhs: &<$from as RingBase>::Element, _hom: &Self::Homomorphism) {
                <Self as CanonicalIntegerHom>::mul_assign_map_in_int_ref::<$from>(self, from, lhs, rhs)
            }
        }
    };
    ($from:ty => $to:ty) => {
        impl CanonicalHom<$from> for $to {

            type Homomorphism = ();

            fn has_canonical_hom(&self, _from: &$from) -> Option<Self::Homomorphism> { Some(()) }
    
            fn map_in(&self, from: &$from, el: <$from as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
                <Self as CanonicalIntegerHom>::map_in_int::<$from>(self, from, el)
            }
        
            fn map_in_ref(&self, from: &$from, el: &<$from as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
                <Self as CanonicalIntegerHom>::map_in_int_ref::<$from>(self, from, el)
            }
        
            fn mul_assign_map_in(&self, from: &$from, lhs: &mut Self::Element, rhs: <$from as RingBase>::Element, _hom: &Self::Homomorphism) {
                <Self as CanonicalIntegerHom>::mul_assign_map_in_int::<$from>(self, from, lhs, rhs)
            }
        
            fn mul_assign_map_in_ref(&self, from: &$from, lhs: &mut Self::Element, rhs: &<$from as RingBase>::Element, _hom: &Self::Homomorphism) {
                <Self as CanonicalIntegerHom>::mul_assign_map_in_int_ref::<$from>(self, from, lhs, rhs)
            }
        }
    };
}

macro_rules! impl_from_integer_canonical_iso {
    ($from:ty => $to:ty; $($constraints:tt)*) => {
        impl<$($constraints)*> CanonicalIso<$from> for $to {

            type Isomorphism = ();

            fn has_canonical_iso(&self, _from: &$from) -> Option<Self::Isomorphism> { Some(()) }
    
            fn map_out(&self, from: &$from, el: Self::Element, _hom: &Self::Isomorphism) -> <$from as RingBase>::Element {
                <$from as CanonicalIntegerHom>::map_in_int::<Self>(from, self, el)
            }
        }
    };
    ($from:ty => $to:ty) => {
        impl CanonicalIso<$from> for $to {

            type Isomorphism = ();

            fn has_canonical_iso(&self, _from: &$from) -> Option<Self::Isomorphism> { Some(()) }
    
            fn map_out(&self, from: &$from, el: Self::Element, _hom: &Self::Isomorphism) -> <$from as RingBase>::Element {
                <$from as CanonicalIntegerHom>::map_in_int::<Self>(from, self, el)
            }
        }
    };
}

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

    fn is_even(&self, value: &El<Self>) -> bool {
        !self.abs_is_bit_set(value, 0)
    }

    fn is_odd(&self, value: &El<Self>) -> bool {
        !self.is_even(value)
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
            assert!(elements.iter().any(|x| ring.eq_el(x, &ring.from_int(i))))
        }
        for x in &elements {
            assert!(ring.is_lt(x, &bound));
        }
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_integer_axioms<R: IntegerRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) 
    where R::Type: IntegerRing
{
    let elements = edge_case_elements.collect::<Vec<_>>();
    for a in &elements {
        let mut ceil_pow_2 = ring.from_int(2);
        ring.mul_pow_2(&mut ceil_pow_2, ring.abs_highest_set_bit(a).unwrap_or(0));
        assert!(ring.is_lt(a, &ceil_pow_2));
        assert!(ring.is_lt(&ring.negate(ring.clone_el(a)), &ceil_pow_2));
        
        for i in 0..ring.abs_highest_set_bit(a).unwrap_or(0) {
            let mut pow_2 = ring.one();
            ring.mul_pow_2(&mut pow_2, i);
            let mut b = ring.clone_el(a);
            ring.mul_pow_2(&mut b, i);
            assert_el_eq!(&ring, &b, &ring.mul(ring.clone_el(a), ring.clone_el(&pow_2)));
            ring.euclidean_div_pow_2(&mut b, i);
            assert_el_eq!(&ring, &b, a);
            ring.euclidean_div_pow_2(&mut b, i);
            assert_el_eq!(&ring, &b, &ring.euclidean_div(ring.clone_el(a), &pow_2));
        }
    }

    // test euclidean div round to zero
    let d = ring.from_int(8);
    for k in -10..=10 {
        let mut a = ring.from_int(k);
        assert_el_eq!(&ring, &ring.from_int(k / 8), &ring.euclidean_div(ring.clone_el(&a), &d));
        ring.euclidean_div_pow_2(&mut a, 3);
        assert_el_eq!(&ring, &ring.from_int(k / 8), &a);
    }
    let d = ring.from_int(-8);
    for k in -10..=10 {
        let a = ring.from_int(k);
        assert_el_eq!(&ring, &ring.from_int(k / -8), &ring.euclidean_div(ring.clone_el(&a), &d));
    }
}

#[test]
fn test_int_div_assumption() {
    assert_eq!(-1, -10 / 8);
    assert_eq!(-1, 10 / -8);
    assert_eq!(1, 10 / 8);
    assert_eq!(1, -10 / -8);
}