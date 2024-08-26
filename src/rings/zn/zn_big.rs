use std::fmt::Debug;

use serde::de;
use serde::{Deserializer, Serializer}; 

use crate::divisibility::DivisibilityRing;
use crate::impl_localpir_wrap_unwrap_homs;
use crate::impl_localpir_wrap_unwrap_isos;
use crate::impl_wrap_unwrap_homs;
use crate::impl_wrap_unwrap_isos;
use crate::pid::*;
use crate::integer::IntegerRing;
use crate::integer::IntegerRingStore;
use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::homomorphism::*;
use crate::delegate::DelegateRing;
use crate::rings::zn::*;

///
/// Ring representing `Z/nZ`, computing the modular reductions
/// via a Barett-reduction algorithm. This is a general-purpose
/// method, but note that it is required that `n^3` fits into the
/// supplied integer type.]
/// 
/// # Performance
/// 
/// This implementation is optimized for use with large integer
/// rings. If the moduli are small, consider using specialized implementations 
/// (like [`crate::rings::zn::zn_64::Zn`]), which will be much faster.
///
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_big::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i64>::RING, 257);
/// let a = R.int_hom().map(16);
/// assert!(R.eq_el(&R.int_hom().map(-1), &R.mul_ref(&a, &a)));
/// assert!(R.is_one(&R.pow(a, 4)));
/// ```
/// However, this will panic as `2053^3 > i32::MAX`.
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_big::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i32>::RING, 2053);
/// ```
///
/// # Canonical mappings
/// This ring has a canonical homomorphism from any integer ring
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_big::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i16>::RING, 7);
/// let S = BigIntRing::RING;
/// assert!(R.eq_el(&R.int_hom().map(120493), &R.coerce(&S, S.int_hom().map(120493))));
/// ```
///
pub struct ZnBase<I: IntegerRingStore> 
    where I::Type: IntegerRing
{
    integer_ring: I,
    modulus: El<I>,
    twice_modulus: El<I>,
    inverse_modulus: El<I>,
    inverse_modulus_bitshift: usize,
}

///
/// Ring representing `Z/nZ`, computing the modular reductions
/// via a Barett-reduction algorithm. For details, see [`ZnBase`].
/// 
pub type Zn<I> = RingValue<ZnBase<I>>;

impl<I: IntegerRingStore> Zn<I>
    where I::Type: IntegerRing
{
    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        RingValue::from(ZnBase::new(integer_ring, modulus))
    }
}

impl<I: IntegerRingStore> ZnBase<I> 
    where I::Type: IntegerRing
{
    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        assert!(integer_ring.is_geq(&modulus, &integer_ring.int_hom().map(2)));

        // have k such that `2^k >= (2 * modulus)^2`
        // then `floor(2^k / modulus) * x >> k` differs at most 1 from `floor(x / modulus)`
        // if `x <= 2^k`, which is the case after multiplication
        let k = integer_ring.abs_log2_ceil(&integer_ring.mul_ref(&modulus, &modulus)).unwrap() + 2;
        let mod_square_bound = integer_ring.power_of_two(k);
        let inverse_modulus = integer_ring.euclidean_div(mod_square_bound, &modulus);

        // check that this expression does not overflow
        integer_ring.mul_ref_snd(integer_ring.pow(integer_ring.clone_el(&modulus), 2), &inverse_modulus);

        return ZnBase {
            twice_modulus: integer_ring.add_ref(&modulus, &modulus),
            integer_ring: integer_ring,
            modulus: modulus,
            inverse_modulus: inverse_modulus,
            inverse_modulus_bitshift: k
        };
    }

    fn bounded_reduce(&self, n: &mut El<I>) {
        debug_assert!(self.integer_ring.is_leq(&n, &self.integer_ring.mul_ref(&self.twice_modulus, &self.twice_modulus)));
        debug_assert!(!self.integer_ring.is_neg(&n));

        let mut subtract = self.integer_ring.mul_ref(&n, &self.inverse_modulus);
        self.integer_ring.euclidean_div_pow_2(&mut subtract, self.inverse_modulus_bitshift);
        self.integer_ring.mul_assign_ref(&mut subtract, &self.modulus);
        self.integer_ring.sub_assign(n, subtract);

        debug_assert!(self.integer_ring.is_lt(&n, &self.twice_modulus));
    }
}

pub struct ZnEl<I: IntegerRingStore>(/* allow it to grow up to 2 * modulus(), inclusively */ El<I>)
    where I::Type: IntegerRing;

impl<I: IntegerRingStore> Debug for ZnEl<I> 
    where El<I>: Clone + Debug,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZnEl({:?})", self.0)
    }
}

impl<I: IntegerRingStore> Clone for ZnEl<I> 
    where El<I>: Clone,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingStore> Copy for ZnEl<I>
    where El<I>: Copy,
        I::Type: IntegerRing
{}

impl<I: IntegerRingStore> RingBase for ZnBase<I> 
    where I::Type: IntegerRing
{
    type Element = ZnEl<I>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        ZnEl(self.integer_ring().clone_el(&val.0))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        self.integer_ring.add_assign_ref(&mut lhs.0, &rhs.0);
        if self.integer_ring.is_geq(&lhs.0, &self.twice_modulus) {
            self.integer_ring.sub_assign_ref(&mut lhs.0, &self.twice_modulus);
        }

        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        self.integer_ring.add_assign(&mut lhs.0, rhs.0);
        if self.integer_ring.is_geq(&lhs.0, &self.twice_modulus) {
            self.integer_ring.sub_assign_ref(&mut lhs.0, &self.twice_modulus);
        }

        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        self.integer_ring.sub_assign_ref(&mut lhs.0, &rhs.0);
        if self.integer_ring.is_neg(&lhs.0) {
            self.integer_ring.add_assign_ref(&mut lhs.0, &self.twice_modulus);
        }

        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(!self.integer_ring.is_neg(&lhs.0));
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));

        self.integer_ring.negate_inplace(&mut lhs.0);
        self.integer_ring.add_assign_ref(&mut lhs.0, &self.twice_modulus);

        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(!self.integer_ring.is_neg(&lhs.0));
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        self.integer_ring.mul_assign(&mut lhs.0, rhs.0);
        self.bounded_reduce(&mut lhs.0);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        self.integer_ring.mul_assign_ref(&mut lhs.0, &rhs.0);
        self.bounded_reduce(&mut lhs.0);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(self.integer_ring.is_leq(&lhs.0, &self.twice_modulus));
        debug_assert!(self.integer_ring.is_leq(&rhs.0, &self.twice_modulus));

        if self.integer_ring.eq_el(&lhs.0, &rhs.0) {
            return true;
        }
        let difference = self.integer_ring.abs(self.integer_ring.sub_ref(&lhs.0, &rhs.0));
        return self.integer_ring.eq_el(&difference, &self.modulus) || self.integer_ring.eq_el(&difference, &self.twice_modulus);
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        debug_assert!(self.integer_ring.is_leq(&value.0, &self.twice_modulus));

        self.integer_ring.is_zero(&value.0) || self.integer_ring.eq_el(&value.0, &self.modulus) || self.integer_ring.eq_el(&value.0, &self.twice_modulus)
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        debug_assert!(self.integer_ring.is_leq(&value.0, &self.twice_modulus));

        self.integer_ring.is_one(&value.0) || self.integer_ring.eq_el(&value.0, &self.integer_ring.add_ref_fst(&self.modulus, self.integer_ring.one()))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.integer_ring.get_ring().dbg(&value.0, out)
    }
    
    fn characteristic<J: IntegerRingStore>(&self, ZZ: &J) -> Option<El<J>>
        where J::Type: IntegerRing
    {
        self.size(ZZ)
    }
}

impl<I: IntegerRingStore> Clone for ZnBase<I> 
    where I: Clone,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        ZnBase {
            integer_ring: self.integer_ring.clone(),
            modulus: self.integer_ring.clone_el(&self.modulus),
            inverse_modulus: self.integer_ring.clone_el(&self.inverse_modulus),
            inverse_modulus_bitshift: self.inverse_modulus_bitshift,
            twice_modulus: self.integer_ring.clone_el(&self.twice_modulus)
        }
    }
}

impl<I: IntegerRingStore> Copy for ZnBase<I> 
    where I: Copy,
        El<I>: Copy,
        I::Type: IntegerRing
{}

impl<I: IntegerRingStore> HashableElRing for ZnBase<I>
    where I::Type: IntegerRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.integer_ring().hash(&self.smallest_positive_lift(self.clone_el(el)), h)
    }
}

impl<I: IntegerRingStore + Default> FromModulusCreateableZnRing for ZnBase<I>
    where I::Type: IntegerRing
{
    fn create<F, E>(create_modulus: F) -> Result<Self, E>
        where F: FnOnce(&Self::IntegerRingBase) -> Result<El<Self::Integers>, E>
    {
        let ZZ = I::default();
        let modulus = create_modulus(ZZ.get_ring())?;
        Ok(Self::new(ZZ, modulus))
    }
}

impl<I: IntegerRingStore> DivisibilityRing for ZnBase<I> 
    where I::Type: IntegerRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs)
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanHomFrom<ZnBase<J>> for ZnBase<I> 
    where I::Type: IntegerRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing
{
    type Homomorphism =  <I::Type as CanHomFrom<J::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ZnBase<J>) -> Option<Self::Homomorphism> {
        let base_hom = <I::Type as CanHomFrom<J::Type>>::has_canonical_hom(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if self.integer_ring.eq_el(
            &self.modulus,
            &<I::Type as CanHomFrom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), from.integer_ring().clone_el(&from.modulus), &base_hom)
        ) {
            Some(base_hom)
        } else {
            None
        }
    }

    fn map_in(&self, from: &ZnBase<J>, el: <ZnBase<J> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(<I::Type as CanHomFrom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), el.0, hom))
    }
}

impl<I: IntegerRingStore> CanHomFrom<zn_64::ZnBase> for ZnBase<I>
    where I::Type: IntegerRing
{
    type Homomorphism = <zn_64::ZnBase as CanIsoFromTo<ZnBase<I>>>::Isomorphism;

    fn has_canonical_hom(&self, from: &zn_64::ZnBase) -> Option<Self::Homomorphism> {
        from.has_canonical_iso(self)
    }

    fn map_in(&self, from: &zn_64::ZnBase, el: <zn_64::ZnBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        from.map_out(self, el, hom)
    }
}

impl<I: IntegerRingStore> CanIsoFromTo<zn_64::ZnBase> for ZnBase<I>
    where I::Type: IntegerRing
{
    type Isomorphism = <zn_64::ZnBase as CanHomFrom<ZnBase<I>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_64::ZnBase) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &zn_64::ZnBase, el: Self::Element, iso: &Self::Isomorphism) -> <zn_64::ZnBase as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

impl<I: IntegerRingStore> PartialEq for ZnBase<I>
    where I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.integer_ring.get_ring() == other.integer_ring.get_ring() && self.integer_ring.eq_el(&self.modulus, &other.modulus)
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanIsoFromTo<ZnBase<J>> for ZnBase<I>
    where I::Type: IntegerRing + CanIsoFromTo<J::Type>,
        J::Type: IntegerRing
{
    type Isomorphism = <I::Type as CanIsoFromTo<J::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ZnBase<J>) -> Option<Self::Isomorphism> {
        let base_iso = <I::Type as CanIsoFromTo<J::Type>>::has_canonical_iso(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if from.integer_ring().eq_el(
            from.modulus(),
            &<I::Type as CanIsoFromTo<J::Type>>::map_out(self.integer_ring.get_ring(), from.integer_ring.get_ring(), self.integer_ring().clone_el(self.modulus()), &base_iso)
        ) {
            Some(base_iso)
        } else {
            None
        }
    }

    fn map_out(&self, from: &ZnBase<J>, el: Self::Element, iso: &Self::Isomorphism) -> <ZnBase<J> as RingBase>::Element {
        ZnEl(<I::Type as CanIsoFromTo<J::Type>>::map_out(self.integer_ring.get_ring(), from.integer_ring.get_ring(), el.0, iso))
    }
}

impl<I: IntegerRingStore, J: IntegerRing + ?Sized> CanHomFrom<J> for ZnBase<I>
    where I::Type: IntegerRing, 
        J: CanIsoFromTo<I::Type>
{
    type Homomorphism = generic_impls::BigIntToZnHom<J, I::Type, ZnBase<I>>;

    fn has_canonical_hom(&self, from: &J) -> Option<Self::Homomorphism> {
        generic_impls::has_canonical_hom_from_bigint(from, self, self.integer_ring.get_ring(), Some(&self.integer_ring.mul_ref(&self.twice_modulus, &self.twice_modulus)))
    }

    fn map_in(&self, from: &J, el: J::Element, hom: &Self::Homomorphism) -> Self::Element {
        generic_impls::map_in_from_bigint(from, self, self.integer_ring.get_ring(), el, hom, |n| {
            debug_assert!(self.integer_ring.is_lt(&n, &self.modulus));
            ZnEl(n)
        }, |mut n| {
            debug_assert!(self.integer_ring.is_lt(&n, &self.integer_ring.mul_ref(&self.twice_modulus, &self.twice_modulus)));
            self.bounded_reduce(&mut n);
            ZnEl(n)
        })
    }
}

pub struct ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    ring: &'a ZnBase<I>,
    current: El<I>
}

impl<'a, I> Clone for ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        Self { ring: self.ring, current: self.ring.integer_ring().clone_el(&self.current) }
    }
}

impl<'a, I> Iterator for ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing
{
    type Item = ZnEl<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ring.integer_ring().is_lt(&self.current, self.ring.modulus()) {
            let result = self.ring.integer_ring().clone_el(&self.current);
            self.ring.integer_ring().add_assign(&mut self.current, self.ring.integer_ring().one());
            return Some(ZnEl(result));
        } else {
            return None;
        }
    }
}

impl<I: IntegerRingStore> SerializableElementRing for ZnBase<I>
    where I::Type: IntegerRing + SerializableElementRing
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        self.integer_ring().get_ring().deserialize(deserializer)
            .and_then(|x| if self.integer_ring().is_neg(&x) || self.integer_ring().is_geq(&x, self.modulus()) { Err(de::Error::custom("ring element value out of bounds for ring Z/nZ")) } else { Ok(x) })
            .map(|x| self.from_int_promise_reduced(x))
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        self.integer_ring().get_ring().serialize(&self.smallest_positive_lift(self.clone_el(el)), serializer)
    }
}

impl<I: IntegerRingStore> FiniteRing for ZnBase<I>
    where I::Type: IntegerRing
{
    type ElementsIter<'a> = ZnBaseElementsIter<'a, I>
        where Self: 'a;

    fn elements<'a>(&'a self) -> ZnBaseElementsIter<'a, I> {
        ZnBaseElementsIter {
            ring: self,
            current: self.integer_ring().zero()
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        generic_impls::random_element(self, rng)
    }
    
    fn size<J: IntegerRingStore>(&self, ZZ: &J) -> Option<El<J>>
        where J::Type: IntegerRing
    {
        if ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(self.modulus()) < ZZ.get_ring().representable_bits() {
            Some(int_cast(self.integer_ring().clone_el(self.modulus()), ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl<I: IntegerRingStore> PrincipalIdealRing for ZnBase<I>
    where I::Type: IntegerRing
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self.integer_ring().extended_ideal_gen(&lhs.0, &rhs.0);
        let quo = RingRef::new(self).into_can_hom(self.integer_ring()).ok().unwrap();
        (quo.map(s), quo.map(t), quo.map(d))
    }
}

impl<I: IntegerRingStore> ZnRing for ZnBase<I>
    where I::Type: IntegerRing
{
    type IntegerRingBase = I::Type;
    type Integers = I;

    fn integer_ring(&self) -> &Self::Integers {
        &self.integer_ring
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus
    }

    fn smallest_positive_lift(&self, mut el: Self::Element) -> El<Self::Integers> {
        if self.integer_ring.eq_el(&el.0, &self.twice_modulus) {
            return self.integer_ring.zero();
        }
        if self.integer_ring.is_geq(&el.0, &self.modulus) {
            self.integer_ring.sub_assign_ref(&mut el.0, &self.modulus);
        }
        debug_assert!(self.integer_ring.is_lt(&el.0, &self.modulus));
        return el.0;
    }

    fn from_int_promise_reduced(&self, x: El<Self::Integers>) -> Self::Element {
        debug_assert!(!self.integer_ring().is_neg(&x));
        debug_assert!(self.integer_ring().is_lt(&x, self.modulus()));
        ZnEl(x)
    }
}

impl_wrap_unwrap_homs!{ <{I, J}> ZnBase<I>, ZnBase<J> where I: IntegerRingStore, I::Type: IntegerRing, J: IntegerRingStore, J::Type: IntegerRing }
impl_wrap_unwrap_isos!{ <{I, J}> ZnBase<I>, ZnBase<J> where I: IntegerRingStore, I::Type: IntegerRing, J: IntegerRingStore, J::Type: IntegerRing }
impl_localpir_wrap_unwrap_homs!{ <{I, J}> ZnBase<I>, ZnBase<J> where I: IntegerRingStore, I::Type: IntegerRing, J: IntegerRingStore, J::Type: IntegerRing }
impl_localpir_wrap_unwrap_isos!{ <{I, J}> ZnBase<I>, ZnBase<J> where I: IntegerRingStore, I::Type: IntegerRing, J: IntegerRingStore, J::Type: IntegerRing }

#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use crate::integer::BigIntRing;
use crate::serialization::SerializableElementRing;

#[test]
fn test_mul() {
    const ZZ: BigIntRing = BigIntRing::RING;
    let Z257 = Zn::new(ZZ, ZZ.int_hom().map(257));
    let x = Z257.coerce(&ZZ, ZZ.int_hom().map(256));
    assert_el_eq!(Z257, Z257.one(), Z257.mul_ref(&x, &x));
}

#[test]
fn test_project() {
    const ZZ: StaticRing<i64> = StaticRing::RING;
    let Z17 = Zn::new(ZZ, 17);
    for k in 0..289 {
        assert_el_eq!(Z17, Z17.int_hom().map((289 - k) % 17), Z17.coerce(&ZZ, -k as i64));
    }
}

#[test]
fn test_ring_axioms_znbase() {
    let ring = Zn::new(StaticRing::<i64>::RING, 63);
    crate::ring::generic_tests::test_ring_axioms(&ring, ring.elements())
}

#[test]
fn test_hash_axioms() {
    let ring = Zn::new(StaticRing::<i64>::RING, 63);
    crate::ring::generic_tests::test_hash_axioms(&ring, ring.elements())
}

#[test]
fn test_canonical_iso_axioms_zn_big() {
    let from = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::new(BigIntRing::RING, BigIntRing::RING.int_hom().map(7 * 11));
    crate::ring::generic_tests::test_hom_axioms(&from, &to, from.elements());
    crate::ring::generic_tests::test_iso_axioms(&from, &to, from.elements());
    assert!(from.can_hom(&Zn::new(StaticRing::<i64>::RING, 19)).is_none());
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i32>::RING;
    let to = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    crate::ring::generic_tests::test_hom_axioms(&from, to, 0..=(2 * 7 * 11));
}

#[test]
fn test_zn_ring_axioms_znbase() {
    super::generic_tests::test_zn_axioms(Zn::new(StaticRing::<i64>::RING, 17));
    super::generic_tests::test_zn_axioms(Zn::new(StaticRing::<i64>::RING, 63));
}

#[test]
fn test_zn_map_in_large_int_znbase() {
    super::generic_tests::test_map_in_large_int(Zn::new(StaticRing::<i64>::RING, 63));
}

#[test]
fn test_zn_map_in_small_int() {
    let ring = Zn::new(StaticRing::<i64>::RING, 257);
    assert_el_eq!(ring, ring.one(), ring.coerce(&StaticRing::<i8>::RING, 1));
}

#[test]
fn test_divisibility_axioms() {
    let R = Zn::new(StaticRing::<i64>::RING, 17);
    crate::divisibility::generic_tests::test_divisibility_axioms(&R, R.elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let R = Zn::new(StaticRing::<i64>::RING, 17);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
    let R = Zn::new(StaticRing::<i64>::RING, 63);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
}

#[test]
fn test_canonical_iso_axioms_as_field() {
    let R = Zn::new(StaticRing::<i128>::RING, 17);
    let R2 = R.clone().as_field().ok().unwrap();
    crate::ring::generic_tests::test_hom_axioms(&R, &R2, R.elements());
    crate::ring::generic_tests::test_iso_axioms(&R, &R2, R.elements());
    crate::ring::generic_tests::test_hom_axioms(&R2, &R, R2.elements());
    crate::ring::generic_tests::test_iso_axioms(&R2, &R, R2.elements());
}

#[test]
fn test_canonical_iso_axioms_zn_64() {
    let R = Zn::new(StaticRing::<i128>::RING, 17);
    let R2 = zn_64::Zn::new(17);
    crate::ring::generic_tests::test_hom_axioms(&R, &R2, R.elements());
    crate::ring::generic_tests::test_iso_axioms(&R, &R2, R.elements());
    crate::ring::generic_tests::test_hom_axioms(&R2, &R, R2.elements());
    crate::ring::generic_tests::test_iso_axioms(&R2, &R, R2.elements());
}

#[test]
fn test_finite_field_axioms() {
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(&StaticRing::<i64>::RING, 128));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(&StaticRing::<i64>::RING, 15));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(&StaticRing::<i128>::RING, 1 << 32));
}

#[test]
fn test_serialize() {
    let ring = Zn::new(&StaticRing::<i64>::RING, 128);
    crate::serialization::generic_tests::test_serialization(ring, ring.elements())
}