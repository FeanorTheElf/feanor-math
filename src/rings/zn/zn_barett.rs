use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::euclidean::EuclideanRingStore;
use crate::integer::IntegerRing;
use crate::integer::IntegerRingStore;
use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::rings::zn::*;
use crate::algorithms;
use crate::primitive_int::*;

use std::cmp::Ordering;

///
/// Ring representing `Z/nZ`, computing the modular reductions
/// via a Barett-reduction algorithm. This is a fast general-purpose
/// method, but note that it is required that `n^4` fits into the
/// supplied integer type.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_barett::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i64>::RING, 257);
/// let a = R.from_z(16);
/// assert!(R.eq(&R.from_z(-1), &R.mul_ref(&a, &a)));
/// assert!(R.is_one(&R.pow(&a, 4)));
/// ```
/// However, this will panic as `257^4 > i32::MAX`.
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_barett::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i32>::RING, 257);
/// ```
/// 
/// # Canonical mappings
/// This ring has a canonical homomorphism from any integer ring
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_barett::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i16>::RING, 7);
/// let S = DefaultBigIntRing::RING;
/// assert!(R.eq(&R.from_z(120493), &R.coerce(&S, S.from_z(120493))));
/// ```
/// 
#[derive(Clone)]
pub struct ZnBase<I: IntegerRingStore> {
    integer_ring: I,
    modulus: El<I>,
    inverse_modulus: El<I>,
    inverse_modulus_bitshift: usize,
}

pub type Zn<I> = RingValue<ZnBase<I>>;

impl<I: IntegerRingStore> Zn<I> {

    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        RingValue::from(ZnBase::new(integer_ring, modulus))
    }
}

impl<I: IntegerRingStore> ZnBase<I> {

    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        assert!(integer_ring.is_geq(&modulus, &integer_ring.from_z(2)));

        // have k such that `2^k >= modulus^2`
        // then `floor(2^k / modulus) * x >> k` differs at most 1 from `floor(x / modulus)`
        // if `x < 2^k`, which is the case after multiplication
        let k = integer_ring.abs_log2_ceil(&modulus).unwrap() * 2;
        let mut mod_square_bound = integer_ring.one();
        integer_ring.mul_pow_2(&mut mod_square_bound, k);

        // check that this expression does not overflow
        integer_ring.println(&modulus);
        integer_ring.mul_ref_snd(integer_ring.pow(&modulus, 2), &mod_square_bound);

        let inverse_modulus = integer_ring.euclidean_div(mod_square_bound, &modulus);
        return ZnBase {
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
        self.project_gen(n, &self.integer_ring)
    }

    pub fn project_gen<J: IntegerRingStore>(&self, n: El<J>, ZZ: &J) -> <Self as RingBase>::Element {
        let mut red_n = n;
        let negated = ZZ.is_neg(&red_n);
        if negated {
            ZZ.negate_inplace(&mut red_n);
        }
        let result = if ZZ.abs_highest_set_bit(&red_n).unwrap_or(0) + 1 < self.integer_ring.abs_highest_set_bit(&self.modulus).unwrap() * 2 {
            let mut result = self.integer_ring.coerce::<J>(ZZ, red_n); 
            if !self.integer_ring.is_lt(&result, &self.modulus) {
                self.project_leq_n_square(&mut result);
            }
            result
        } else {
            let modulus = ZZ.coerce::<I>(&self.integer_ring, self.modulus.clone());
            red_n = ZZ.euclidean_rem(red_n, &modulus);
            self.integer_ring.coerce::<J>(ZZ, red_n)
        };
        if negated {
            return self.negate(ZnEl(result));
        } else {
            return ZnEl(result);
        }
    }

    ///
    /// Returns either the inverse of x (as Ok()) or a nontrivial 
    /// factor of the modulus (as Err())
    /// 
    pub fn invert(&self, x: ZnEl<I>) -> Result<ZnEl<I>, El<I>> {
        let (s, _, d) = algorithms::eea::eea(x.0.clone(), self.modulus.clone(), &self.integer_ring);
        if self.integer_ring.is_neg_one(&d) || self.integer_ring.is_one(&d) {
            Ok(self.project(s))
        } else {
            Err(d)
        }
    }
}

pub struct ZnEl<I: IntegerRingStore>(El<I>);

impl<I: IntegerRingStore> Clone for ZnEl<I> {

    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingStore> Copy for ZnEl<I> 
    where El<I>: Copy
{}

impl<I: IntegerRingStore> RingBase for ZnBase<I> {

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
        self.project_gen(value, &StaticRing::<i32>::RING)
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

impl<I: IntegerRingStore> DivisibilityRing for ZnBase<I> {
    
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let d = algorithms::eea::gcd(lhs.0.clone(), rhs.0.clone(), &self.integer_ring);
        if let Ok(inv) = self.invert(self.project(self.integer_ring.checked_div(&rhs.0, &d).unwrap())) {
            return Some(self.mul(inv, self.project(self.integer_ring.checked_div(&lhs.0, &d).unwrap())));
        } else {
            return None;
        }
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanonicalHom<ZnBase<J>> for ZnBase<I> {

    type Homomorphism =  <I::Type as CanonicalHom<J::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ZnBase<J>) -> Option<Self::Homomorphism> {
        let base_hom = <I::Type as CanonicalHom<J::Type>>::has_canonical_hom(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if self.integer_ring.eq(
            &self.modulus, 
            &<I::Type as CanonicalHom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), from.modulus.clone(), &base_hom)
        ) {
            Some(base_hom)
        } else {
            None
        }
    }

    fn map_in(&self, from: &ZnBase<J>, el: <ZnBase<J> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(<I::Type as CanonicalHom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), el.0, hom))
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanonicalIso<ZnBase<J>> for ZnBase<I> {

    type Isomorphism = <I::Type as CanonicalIso<J::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ZnBase<J>) -> Option<Self::Isomorphism> {
        let base_iso = <I::Type as CanonicalIso<J::Type>>::has_canonical_iso(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if self.integer_ring.eq(
             &self.modulus, 
            &<I::Type as CanonicalHom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), from.modulus.clone(), &base_iso)
        ) {
            Some(base_iso)
        } else {
            None
        }
    }

    fn map_out(&self, from: &ZnBase<J>, el: Self::Element, iso: &Self::Isomorphism) -> <ZnBase<J> as RingBase>::Element {
        ZnEl(<I::Type as CanonicalIso<J::Type>>::map_out(self.integer_ring.get_ring(), from.integer_ring.get_ring(), el.0, iso))
    }
}

impl<I: IntegerRingStore, J: IntegerRing + ?Sized> CanonicalHom<J> for ZnBase<I> 
    where J: CanonicalIso<J>
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &J) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &J, el: J::Element, _hom: &Self::Homomorphism) -> Self::Element {
        self.project_gen(el, &RingRef::new(from))
    }
}

pub struct ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore
{
    ring: &'a ZnBase<I>,
    current: El<I>
}

impl<'a, I> Iterator for ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore
{
    type Item = ZnEl<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ring.integer_ring().is_lt(&self.current, self.ring.modulus()) {
            let result = self.current.clone();
            self.ring.integer_ring().add_assign(&mut self.current, self.ring.integer_ring().one());
            return Some(ZnEl(result));
        } else {
            return None;
        }
    }
}

impl<I: IntegerRingStore> ZnRing for ZnBase<I> {
    
    type IntegerRingBase = I::Type;
    type Integers = I;
    type ElementsIter<'a> = ZnBaseElementsIter<'a, I>
        where Self: 'a;

    fn integer_ring(&self) -> &Self::Integers {
        &self.integer_ring
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        el.0
    }

    fn elements<'a>(&'a self) -> ZnBaseElementsIter<'a, I> {
        ZnBaseElementsIter {
            ring: self,
            current: self.integer_ring().zero()
        }
    }

}

#[cfg(test)]
use crate::rings::bigint::*;

#[test]
fn test_mul() {
    const ZZ: RingValue<DefaultBigIntRing> = DefaultBigIntRing::RING;
    let Z257 = ZnBase::new(ZZ, ZZ.from_z(257));
    let x = Z257.project(ZZ.from_z(256));
    assert!(Z257.eq(&Z257.one(), &Z257.mul_ref(&x, &x)));
}

#[test]
fn test_project() {
    const ZZ: StaticRing<i64> = StaticRing::RING;
    let Z17 = Zn::new(ZZ, 17);
    for k in 0..289 {
        assert!(Z17.eq(&Z17.from_z((289 - k) % 17), &Z17.get_ring().project(-k as i64)));
    }
}

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 10] = [0, 1, 3, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms_znbase() {
    let ZZ = Zn::new(StaticRing::<i64>::RING, 63);
    generic_test_ring_axioms(&ZZ, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ZZ.from_z(x)))
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    let from = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::new(DefaultBigIntRing::RING, DefaultBigIntRing::RING.from_z(7 * 11));
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_z(x)));
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_z(x)));
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i32>::RING;
    let to = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_z(x)));
}

#[test]
fn test_zn_ring_axioms_znbase() {
    generic_test_zn_ring_axioms(Zn::new(StaticRing::<i64>::RING, 17));
    generic_test_zn_ring_axioms(Zn::new(StaticRing::<i64>::RING, 63));
}

#[test]
fn test_zn_map_in_large_int_znbase() {
    generic_test_map_in_large_int(Zn::new(StaticRing::<i64>::RING, 63));
}