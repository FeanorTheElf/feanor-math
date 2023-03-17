use crate::delegate::DelegateRing;
use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingWrapper;
use crate::euclidean::EuclideanRing;
use crate::euclidean::EuclideanRingWrapper;
use crate::field::Field;
use crate::integer::IntegerRing;
use crate::integer::IntegerRingWrapper;
use crate::ordered::OrderedRingWrapper;
use crate::ring::*;
use crate::algorithms;

use std::cmp::Ordering;

#[derive(Clone)]
pub struct Zn<I: IntegerRingWrapper> {
    integer_ring: I,
    modulus: El<I>,
    inverse_modulus: El<I>,
    inverse_modulus_bitshift: usize,
}

impl<I: IntegerRingWrapper> Zn<I> {

    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        assert!(integer_ring.is_geq(&modulus, &integer_ring.from_z(2)));
        // have k such that 2^k > modulus^2
        // then (2^k / modulus) * x >> k differs at most 1 from floor(x / modulus)
        // if x < n^2, which is the case after multiplication
        let k = integer_ring.abs_highest_set_bit(&modulus).unwrap() * 2 + 2;
        let inverse_modulus = integer_ring.euclidean_div(
            integer_ring.pow(&integer_ring.from_z(2), k as usize), 
            &modulus
        );
        return Zn {
            integer_ring: integer_ring,
            modulus: modulus,
            inverse_modulus: inverse_modulus,
            inverse_modulus_bitshift: k
        };
    }

    fn project_leq_n_square(&self, n: &mut El<I>) {
        assert!(self.integer_ring.cmp(&n, &self.integer_ring.zero()) != Ordering::Less);
        let mut subtract = self.integer_ring.mul_ref(&n, &self.inverse_modulus);
        self.integer_ring.euclidean_div_pow_2(&mut subtract, self.inverse_modulus_bitshift);
        self.integer_ring.mul_assign_ref(&mut subtract, &self.modulus);
        self.integer_ring.sub_assign(n, subtract);
        if self.integer_ring.is_geq(&n, &self.modulus) {
            self.integer_ring.sub_assign_ref(n, &self.modulus);
        }
        assert!(self.integer_ring.is_lt(&n, &self.modulus), "The input is not smaller than {}^2", self.integer_ring.format(&self.modulus));
    }

    pub fn project(&self, n: El<I>) -> <Self as RingBase>::Element {
        let mut red_n = n;
        let negated = self.integer_ring.is_neg(&red_n);
        if negated {
            self.integer_ring.negate_inplace(&mut red_n);
        }
        if self.integer_ring.is_lt(&red_n, &self.modulus) {
            // already in the interval [0, self.modulus[
        } else if self.integer_ring.abs_highest_set_bit(&red_n).unwrap_or(0) + 1 < self.integer_ring.abs_highest_set_bit(&self.modulus).unwrap() * 2 {
            self.project_leq_n_square(&mut red_n);
        } else {
            red_n = self.integer_ring.euclidean_rem(red_n, &self.modulus);
        };
        if negated {
            red_n = self.integer_ring.sub_ref_fst(&self.modulus, red_n);
        }
        debug_assert!(self.integer_ring.is_lt(&red_n, &self.modulus));
        return ZnEl(red_n);
    }

    ///
    /// Returns either the inverse of x (as Ok()) or a nontrivial 
    /// factor of the modulus (as Err())
    /// 
    pub fn invert(&self, x: El<I>) -> Result<El<I>, El<I>> {
        let (s, _, d) = algorithms::eea::eea(x.clone(), self.modulus.clone(), &self.integer_ring);
        if self.integer_ring.is_neg_one(&d) || self.integer_ring.is_one(&d) {
            Ok(s)
        } else {
            Err(d)
        }
    }

    pub fn is_field(self) -> Result<Fp<I>, Zn<I>> 
        where I: HashableElRingWrapper
    {
        if algorithms::miller_rabin::is_prime(self.integer_ring(), &self.modulus, 6) {
            Ok(Fp { base: self })
        } else {
            Err(self)
        }
    }
}

pub struct ZnEl<I: IntegerRingWrapper>(El<I>);

impl<I: IntegerRingWrapper> Clone for ZnEl<I> {

    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingWrapper> Copy for ZnEl<I> 
    where El<I>: Copy
{}

impl<I: IntegerRingWrapper> RingBase for Zn<I> {

    type Element = ZnEl<I>;

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.integer_ring.add_assign_ref(&mut lhs.0, &rhs.0);
        if self.integer_ring.is_geq(&lhs.0, &self.modulus) {
            self.integer_ring.sub_assign_ref(&mut lhs.0, &self.modulus);
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.integer_ring.add_assign(&mut lhs.0, rhs.0);
        if self.integer_ring.is_geq(&lhs.0, &self.modulus) {
            self.integer_ring.sub_assign_ref(&mut lhs.0, &self.modulus);
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.integer_ring.sub_assign_ref(&mut lhs.0, &rhs.0);
        if self.integer_ring.is_neg(&lhs.0) {
            self.integer_ring.add_assign_ref(&mut lhs.0, &self.modulus);
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        if !self.integer_ring.is_zero(&lhs.0) {
            self.integer_ring.negate_inplace(&mut lhs.0);
            self.integer_ring.add_assign_ref(&mut lhs.0, &self.modulus);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.integer_ring.mul_assign(&mut lhs.0, rhs.0);
        self.project_leq_n_square(&mut lhs.0);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.integer_ring.mul_assign_ref(&mut lhs.0, &rhs.0);
        self.project_leq_n_square(&mut lhs.0);
    }

    fn from_z(&self, value: i32) -> Self::Element {
        self.project(self.integer_ring.from_z(value))
    }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.integer_ring.eq(&lhs.0, &rhs.0)
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.integer_ring.is_zero(&value.0)
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        self.integer_ring.is_one(&value.0)
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.integer_ring.get_ring().dbg(&value.0, out)
    }

}

impl<I: IntegerRingWrapper> DivisibilityRing for Zn<I> {
    
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let d = algorithms::eea::gcd(lhs.0.clone(), rhs.0.clone(), &self.integer_ring);
        if let Ok(inv) = self.invert(self.integer_ring.checked_div(&rhs.0, &d).unwrap()) {
            return Some(self.project(self.integer_ring.mul(inv, self.integer_ring.checked_div(&lhs.0, &d).unwrap())));
        } else {
            return None;
        }
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalHom<Zn<J>> for Zn<I> 
    where I::Type: CanonicalHom<J::Type>
{
    fn has_canonical_hom(&self, from: &Zn<J>) -> bool {
        self.integer_ring.get_ring().has_canonical_hom(from.integer_ring.get_ring()) && 
            self.integer_ring.eq(&self.modulus, &self.integer_ring.map_in(&from.integer_ring, from.modulus.clone()))
    }

    fn map_in(&self, from: &Zn<J>, el: <Zn<J> as RingBase>::Element) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        ZnEl(self.integer_ring.map_in(&from.integer_ring, el.0))
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalIso<Zn<J>> for Zn<I> 
    where I::Type: CanonicalIso<J::Type>
{
    fn has_canonical_iso(&self, from: &Zn<J>) -> bool {
        self.integer_ring.get_ring().has_canonical_iso(from.integer_ring.get_ring()) && 
            self.integer_ring.eq(&self.modulus, &self.integer_ring.map_in(&from.integer_ring, from.modulus.clone()))
    }

    fn map_out(&self, from: &Zn<J>, el: Self::Element) -> <Zn<J> as RingBase>::Element {
        debug_assert!(self.has_canonical_iso(from));
        ZnEl(self.integer_ring.map_out(&from.integer_ring, el.0))
    }
}

impl<I: IntegerRingWrapper, J: IntegerRing> CanonicalHom<J> for Zn<I> 
    where I::Type: CanonicalHom<J>
{
    fn has_canonical_hom(&self, from: &J) -> bool {
        self.integer_ring().get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &J, el: J::Element) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        self.project(self.integer_ring().get_ring().map_in(from, el))
    }
}

impl<I: IntegerRingWrapper> ZnRing for Zn<I> {
    
    type IntegerRingBase = I::Type;
    type Integers = I;

    fn integer_ring(&self) -> &Self::Integers {
        &self.integer_ring
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        el.0
    }
}

#[derive(Clone)]
pub struct Fp<I: IntegerRingWrapper> {
    base: Zn<I>
}

pub struct FpEl<I: IntegerRingWrapper>(ZnEl<I>);

impl<I: IntegerRingWrapper> Clone for FpEl<I> {

    fn clone(&self) -> Self {
        FpEl(self.0.clone())
    }
}

impl<I: IntegerRingWrapper> Copy for FpEl<I> 
    where El<I>: Copy
{}

impl<I: IntegerRingWrapper> Fp<I> {

    pub fn get_base(&self) -> &Zn<I> {
        &self.base
    }
}

impl<I: IntegerRingWrapper> DelegateRing for Fp<I> {

    type Element = FpEl<I>;
    type Base = Zn<I>;

    fn get_delegate(&self) -> &Self::Base {
        &self.base
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.0
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.0
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.0
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        FpEl(el)
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalHom<Fp<J>> for Fp<I> 
    where I::Type: CanonicalHom<J::Type>
{
    fn has_canonical_hom(&self, from: &Fp<J>) -> bool {
        self.get_base().has_canonical_hom(from.get_base())
    }

    fn map_in(&self, from: &Fp<J>, el: FpEl<J>) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        FpEl(self.get_base().map_in(from.get_base(), el.0))
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalIso<Fp<J>> for Fp<I> 
    where I::Type: CanonicalIso<J::Type>
{
    fn has_canonical_iso(&self, from: &Fp<J>) -> bool {
        self.get_base().has_canonical_iso(from.get_base())
    }

    fn map_out(&self, from: &Fp<J>, el: Self::Element) -> FpEl<J> {
        debug_assert!(self.has_canonical_iso(from));
        FpEl(self.get_base().map_out(from.get_base(), el.0))
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalHom<Zn<J>> for Fp<I> 
    where I::Type: CanonicalHom<J::Type>
{
    fn has_canonical_hom(&self, from: &Zn<J>) -> bool {
        self.get_base().has_canonical_hom(from)
    }

    fn map_in(&self, from: &Zn<J>, el: ZnEl<J>) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        FpEl(self.get_base().map_in(from, el))
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> CanonicalIso<Zn<J>> for Fp<I> 
    where I::Type: CanonicalIso<J::Type>
{
    fn has_canonical_iso(&self, from: &Zn<J>) -> bool {
        self.get_base().has_canonical_iso(from)
    }

    fn map_out(&self, from: &Zn<J>, el: Self::Element) -> <Zn<J> as RingBase>::Element {
        debug_assert!(self.has_canonical_iso(from));
        self.get_base().map_out(from, el.0)
    }
}

impl<I: IntegerRingWrapper, J: IntegerRing> CanonicalHom<J> for Fp<I> 
    where I::Type: CanonicalHom<J>
{
    fn has_canonical_hom(&self, from: &J) -> bool {
        self.get_base().has_canonical_hom(from)
    }

    fn map_in(&self, from: &J, el: J::Element) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        FpEl(self.get_base().map_in(from, el))
    }
}

impl<I: IntegerRingWrapper> DivisibilityRing for Fp<I> {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(rhs) {
            None
        } else {
            Some(self.mul_ref_fst(
                lhs, 
                FpEl(self.get_base().project(
                    self.get_base().invert(rhs.0.clone().0).ok().unwrap()
                ))
            ))
        }
    }
}

impl<I: IntegerRingWrapper> EuclideanRing for Fp<I> {

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }

    fn euclidean_rem(&self, _: Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        self.zero()
    }
}

impl<I: IntegerRingWrapper> Field for Fp<I> {}

impl<I: IntegerRingWrapper> ZnRing for Fp<I> {
    
    type IntegerRingBase = I::Type;
    type Integers = I;

    fn integer_ring(&self) -> &Self::Integers {
        self.get_base().integer_ring()
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.get_base().modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.get_base().smallest_positive_lift(el.0)
    }
}


#[cfg(test)]
use crate::rings::bigint::*;

use super::ZnRing;

#[test]
fn test_mul() {
    const ZZ: RingValue<DefaultBigIntRing> = DefaultBigIntRing::RING;
    let z257 = Zn::new(ZZ, ZZ.from_z(257));
    let x = z257.project(ZZ.from_z(256));
    assert!(z257.eq(&z257.one(), &z257.mul_ref(&x, &x)));
}
