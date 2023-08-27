use crate::{ring::{RingBase, CanonicalHom, CanonicalIso, RingStore}, algorithms::{fft::cooley_tuckey::CooleyTuckeyButterfly, self}, rings::finite::FiniteRing, divisibility::DivisibilityRing, primitive_int::{StaticRingBase, StaticRing}, integer::IntegerRingStore};

use super::{ZnRing, zn_42::usigned_as_signed_ref};

const BITSHIFT: u32 = 60;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastFFTZn {
    modulus: u64,
    two_modulus: u64
}

impl FastFFTZn {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus <= (1 << BITSHIFT));
        let delta = (1 << BITSHIFT) - modulus;
        assert!(2 * delta + 4 * delta * delta < (1 << BITSHIFT));
        FastFFTZn {
            modulus: modulus,
            two_modulus: 2 * modulus
        }
    }

    fn bounded_reduce(&self, value: u128) -> u64 {
        let quo_approx1 = (value >> BITSHIFT) as u64;
        let red1 = value - self.modulus as u128 * quo_approx1 as u128;
        let quo_approx2 = (red1 >> BITSHIFT) as u64;
        let red2 = red1 - self.modulus as u128 * quo_approx2 as u128;
        return red2 as u64;
    }

    fn additive_reduce(&self, value: &mut u64) {
        if std::intrinsics::unlikely(*value > self.two_modulus) {
            *value -= self.two_modulus;
        }
    }
}

impl RingBase for FastFFTZn {

    type Element = u64;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs += rhs;
        if *lhs >= self.two_modulus {
            *lhs -= self.two_modulus
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        *lhs = self.two_modulus - *lhs;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.bounded_reduce(*lhs as u128 * rhs as u128)
    }

    fn from_int(&self, value: i32) -> Self::Element {
        if value < 0 {
            self.negate(value.unsigned_abs() as u64)
        } else {
            value as u64
        }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if *lhs > *rhs {
            self.is_zero(&(*lhs - *rhs))
        } else {
            self.is_zero(&(*rhs - *lhs))
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value % self.modulus == 0
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", *value % self.modulus)
    }
}

impl CanonicalHom<StaticRingBase<i64>> for FastFFTZn {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &StaticRingBase<i64>) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &StaticRingBase<i64>, el: <StaticRingBase<i64> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        if el < 0 {
            self.negate(self.bounded_reduce(el.unsigned_abs() as u128))
        } else {
            self.bounded_reduce(el.unsigned_abs() as u128)
        }
    }
}

impl CanonicalHom<FastFFTZn> for FastFFTZn {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &FastFFTZn) -> Option<Self::Homomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &FastFFTZn, el: <FastFFTZn as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<FastFFTZn> for FastFFTZn {

    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &FastFFTZn) -> Option<Self::Isomorphism> {
        if self.modulus == from.modulus {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, from: &FastFFTZn, el: Self::Element, iso: &Self::Isomorphism) -> <FastFFTZn as RingBase>::Element {
        el
    }
}

impl CooleyTuckeyButterfly<FastFFTZn> for FastFFTZn {

    fn butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, from: &FastFFTZn, hom: &<Self as CanonicalHom<FastFFTZn>>::Homomorphism, values: &mut V, twiddle: &<FastFFTZn as RingBase>::Element, i1: usize, i2: usize) {
        let lhs_val = *values.at(i1);
        let mut rhs_val = *values.at(i2);
        rhs_val = self.bounded_reduce(rhs_val as u128 * *twiddle as u128);
        *values.at_mut(i1) = lhs_val + rhs_val;
        *values.at_mut(i2) = lhs_val + self.two_modulus - rhs_val;
        self.additive_reduce(values.at_mut(i1));
        self.additive_reduce(values.at_mut(i2));
    }

    fn inv_butterfly<V: crate::vector::VectorViewMut<Self::Element>>(&self, from: &FastFFTZn, hom: &<Self as CanonicalHom<FastFFTZn>>::Homomorphism, values: &mut V, twiddle: &<FastFFTZn as RingBase>::Element, i1: usize, i2: usize) {
        let lhs_val = *values.at(i1);
        let rhs_val = *values.at(i2);
        *values.at_mut(i1) = lhs_val + rhs_val;
        *values.at_mut(i2) = self.bounded_reduce((lhs_val + self.two_modulus - rhs_val) as u128 * *twiddle as u128); 
        self.additive_reduce(values.at_mut(i1));
        self.additive_reduce(values.at_mut(i2));
    }
}

impl DivisibilityRing for FastFFTZn {
    
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ZZ = StaticRing::<i64>::RING;
        let d = algorithms::eea::gcd(*lhs as i64, *rhs as i64, &ZZ);
        if d == 0 {
            return Some(self.zero());
        } else {
            let (s, _, d) = algorithms::eea::eea(*rhs as i64 / d, *self.modulus(), &ZZ);
            if d == 1 || d == -1 {
                return Some(self.mul(
                    self.map_in(ZZ.get_ring(), s, &()), 
                    self.map_in(ZZ.get_ring(), *rhs as i64 / d, &())
                ));
            } else {
                return None;
            }
        }
    }
}

impl FiniteRing for FastFFTZn {

    type ElementsIter<'a> = std::vec::IntoIter<u64>;
    
    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        unimplemented!()
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        self.map_in(StaticRing::<i64>::RING.get_ring(), StaticRing::<i64>::RING.get_uniformly_random(self.modulus(), rng), &())
    }

    fn size<I: crate::integer::IntegerRingStore>(&self, ZZ: &I) -> crate::ring::El<I>
            where I::Type: crate::integer::IntegerRing
    {
        unimplemented!()
    }
}

impl ZnRing for FastFFTZn {
    
    type IntegerRingBase = StaticRingBase<i64>;
    type Integers = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i64>::RING
    }

    fn modulus(&self) -> &crate::ring::El<Self::Integers> {
        usigned_as_signed_ref(&self.modulus)
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> crate::ring::El<Self::Integers> {
        (el % self.modulus) as i64
    }
}