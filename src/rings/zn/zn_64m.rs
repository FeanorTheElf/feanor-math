use crate::algorithms::eea::const_eea;
use crate::{impl_field_wrap_unwrap_homs, impl_field_wrap_unwrap_isos, impl_localpir_wrap_unwrap_homs, impl_localpir_wrap_unwrap_isos};
use crate::reduce_lift::lift_poly_eval::InterpolationBaseRing;
use crate::divisibility::*;
use crate::impl_eq_based_self_iso;
use crate::ordered::OrderedRingStore;
use crate::primitive_int::*;
use crate::integer::*;
use crate::pid::*;
use crate::rings::extension::FreeAlgebraStore;
use crate::ring::*;
use crate::rings::extension::galois_field::*;
use crate::seq::*;
use crate::serialization::*;
use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::matmul::StrassenHint;
use crate::specialization::*;
use crate::homomorphism::*;

use std::marker::PhantomData;
use std::fmt::Debug;

use feanor_serde::newtype_struct::*;
use serde::de::{DeserializeSeed, Error};
use serde::{Deserialize, Deserializer, Serialize, Serializer}; 

use super::*;
use super::zn_big;

///
/// Represents the ring `Z/nZ` for odd `n` somewhat below 64 bits in size.
/// 
/// The `64M` in the name stands for "64-bit Montgomery", which is the
/// of internally used reduction algorithm. `R` is chosen as `2^64`.
/// 
#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
pub struct Zn64MBase {
    modulus: i64,
    modulus_times_three: u64,
    n_inv_mod_R: u64,
    R_sqr_mod_n: u64
}

impl Zn64MBase {

    #[stability::unstable(feature = "enable")]
    pub const fn new(modulus: u64) -> Self {
        assert!(modulus % 2 == 1);
        assert!(modulus as u128 * 9 <= u64::MAX as u128);
        Self {
            modulus: modulus as i64,
            modulus_times_three: modulus * 3,
            n_inv_mod_R: const_eea(modulus as i128, 1 << 64).0 as u64,
            R_sqr_mod_n: ((((1 << 64) % modulus as i128) * (1 << 64) % modulus as i128) % modulus as i128) as u64, 
        }
    }

    fn modulus_u64(&self) -> u64 {
        self.modulus as u64
    }

    ///
    /// All representatives should be within `{ 0, ..., repr_bound }`
    /// 
    fn repr_bound(&self) -> u64 {
        self.modulus_times_three
    }

    ///
    /// Assumes that `x` is `<= R * n`, then the result is `< 2 * n` and
    /// congruent to `x * R^-1` modulo `n`.
    /// 
    fn montgomery_reduce(&self, x: u128) -> u64 {
        let x_low = (x % (1 << 64)) as u64;
        let a = x_low.wrapping_mul(self.n_inv_mod_R);
        let b = a as i128 * self.modulus as i128;
        let res = (x as i128 - b) / (1 << 64);
        let res = ((res as i64) + self.modulus) as u64;
        debug_assert!(res < 2 * self.modulus_u64());
        return res;
    }

    ///
    /// Assumes that `x` is `<= 6 * n`, then the result is
    /// `<= 3 * n` and congruent to `x` modulo `n`.
    /// 
    fn half_reduce(&self, x: u64) -> u64 {
        if x >= self.modulus_times_three {
            x - self.modulus_times_three
        } else {
            x
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn from_primitive_int<T: PrimitiveInt>(&self, el: T) -> Zn64MEl {
        let is_neg = StaticRing::<T>::RING.is_neg(&el);
        let el_abs = <T as Into<i128>>::into(el).unsigned_abs();
        if el_abs <= self.modulus_u64() as u128 {
            if is_neg {
                self.negate(self.from_int_promise_reduced(el_abs as i64))
            } else {
                self.from_int_promise_reduced(el_abs as i64)
            }
        } else {
            if is_neg {
                self.negate(self.from_int_promise_reduced((el_abs % self.modulus as u128) as i64))
            } else {
                self.from_int_promise_reduced((el_abs % self.modulus as u128) as i64)
            }
        }
    }
}

/// 
/// Represents the ring `Z/nZ`, using a variant of Barett reduction
/// for moduli somewhat smaller than 64 bits. For details, see [`ZnBase`].
/// 
#[stability::unstable(feature = "enable")]
pub type Zn64M = RingValue<Zn64MBase>;

impl Zn64M {

    #[stability::unstable(feature = "enable")]
    pub const fn new(modulus: u64) -> Self {
        RingValue::from(Zn64MBase::new(modulus))
    }
}

#[derive(Clone, Copy, Debug)]
#[stability::unstable(feature = "enable")]
pub struct Zn64MEl(/* a representative within `[0, ring.repr_bound()]` */ u64);

impl PartialEq for Zn64MBase {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl Debug for Zn64MBase {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Z/{}Z", self.modulus)
    }
}

impl RingBase for Zn64MBase {

    type Element = Zn64MEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.half_reduce(lhs.0 + rhs.0);
        debug_assert!(lhs.0 <= self.repr_bound());
    }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.half_reduce(lhs.0 + self.modulus_times_three - rhs.0);
        debug_assert!(lhs.0 <= self.repr_bound());
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        lhs.0 = self.modulus_times_three - lhs.0;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.montgomery_reduce(lhs.0 as u128 * rhs.0 as u128);
        debug_assert!(lhs.0 <= self.repr_bound());
    }

    fn fma(&self, lhs: &Self::Element, rhs: &Self::Element, summand: Self::Element) -> Self::Element {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        debug_assert!(summand.0 <= self.repr_bound());
        let result = self.montgomery_reduce(lhs.0 as u128 * rhs.0 as u128) + summand.0;
        debug_assert!(result <= self.repr_bound());
        return Zn64MEl(result);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from_primitive_int(value)
    }

    fn zero(&self) -> Self::Element {
        Zn64MEl(0)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        self.is_zero(&self.sub_ref(lhs, rhs))
    }

    fn is_zero(&self, val: &Self::Element) -> bool {
        debug_assert!(val.0 <= self.repr_bound());
        let mut reduced = val.0;
        if reduced >= self.modulus_u64() {
            reduced -= self.modulus_u64();
        }
        return reduced == 0 || reduced == self.modulus_u64();
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _: EnvBindingStrength) -> std::fmt::Result {
        write!(out, "{}", self.smallest_positive_lift(*value))
    }
    
    fn characteristic<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.size(other_ZZ)
    }

    fn is_approximate(&self) -> bool { false }
}

impl SerializableElementRing for Zn64MBase {

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        <i64 as Deserialize>::deserialize(deserializer)
            .and_then(|x| if x < 0 || x >= *self.modulus() { Err(Error::custom("ring element value out of bounds for ring Z/nZ")) } else { Ok(x) })
            .map(|x| self.from_int_promise_reduced(x))
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        <i64 as Serialize>::serialize(&self.smallest_positive_lift(*el), serializer)
    }
}

impl FromModulusCreateableZnRing for Zn64MBase {

    fn from_modulus<F, E>(create_modulus: F) -> Result<Self, E>
        where F: FnOnce(&Self::IntegerRingBase) -> Result<El<Self::IntegerRing>, E>
    {
        create_modulus(StaticRing::<i64>::RING.get_ring()).map(|n| Self::new(n as u64))
    }
}

impl InterpolationBaseRing for AsFieldBase<Zn64M> {

    type ExtendedRingBase<'a> = GaloisFieldBaseOver<RingRef<'a, Self>>
        where Self: 'a;

    type ExtendedRing<'a> = GaloisFieldOver<RingRef<'a, Self>>
        where Self: 'a;

    fn in_base<'a, S>(&self, ext_ring: S, el: El<S>) -> Option<Self::Element>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        let wrt_basis = ext_ring.wrt_canonical_basis(&el);
        if wrt_basis.iter().skip(1).all(|x| self.is_zero(&x)) {
            return Some(wrt_basis.at(0));
        } else {
            return None;
        }
    }

    fn in_extension<'a, S>(&self, ext_ring: S, el: Self::Element) -> El<S>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
    {
        ext_ring.inclusion().map(el)
    }

    fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>) {
        let ring = super::generic_impls::interpolation_ring(RingRef::new(self), count);
        let points = ring.elements().take(count).collect();
        return (ring, points);
    }
}

impl_eq_based_self_iso!{ Zn64MBase }

impl<I: RingStore> CanHomFrom<zn_big::ZnGBBase<I>> for Zn64MBase
    where I::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Homomorphism> {
        if from.integer_ring().get_ring().representable_bits().is_none() || from.integer_ring().get_ring().representable_bits().unwrap() >= self.integer_ring().abs_log2_ceil(self.modulus()).unwrap() {
            if from.integer_ring().eq_el(from.modulus(), &int_cast(*self.modulus(), from.integer_ring(), self.integer_ring())) {
                Some(())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_big::ZnGBBase<I>, el: <zn_big::ZnGBBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        self.from_int_promise_reduced(int_cast(from.smallest_positive_lift(el), self.integer_ring(), from.integer_ring()))
    }
}

impl<I: RingStore> CanIsoFromTo<zn_big::ZnGBBase<I>> for Zn64MBase
    where I::Type: IntegerRing
{
    type Isomorphism = <zn_big::ZnGBBase<I> as CanHomFrom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_big::ZnGBBase<I>) -> Option<Self::Isomorphism> {
        if from.integer_ring().get_ring().representable_bits().is_none() || from.integer_ring().get_ring().representable_bits().unwrap() >= self.integer_ring().abs_log2_ceil(self.modulus()).unwrap() {
            if from.integer_ring().eq_el(from.modulus(), &int_cast(*self.modulus(), from.integer_ring(), self.integer_ring())) {
                from.has_canonical_hom(self.integer_ring().get_ring())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_big::ZnGBBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_big::ZnGBBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0.try_into().unwrap(), iso)
    }
}

impl CanHomFrom<zn_64::Zn64BBase> for Zn64MBase {
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_64::Zn64BBase) -> Option<Self::Homomorphism> {
        if self.modulus() == from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_64::Zn64BBase, el: <zn_64::Zn64BBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        self.from_int_promise_reduced(from.smallest_positive_lift(el))
    }
}

impl CanIsoFromTo<zn_64::Zn64BBase> for Zn64MBase {
    type Isomorphism = <zn_64::Zn64BBase as CanHomFrom<Self>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_64::Zn64BBase) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &zn_64::Zn64BBase, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_64::Zn64BBase as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

impl DivisibilityRing for Zn64MBase {

    type PreparedDivisorData = ();

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs)
    }

    fn prepare_divisor(&self, _: &Self::Element) -> Self::PreparedDivisorData {
        ()
    }
}

impl<I: ?Sized + IntegerRing> CanHomFrom<I> for Zn64MBase {

    type Homomorphism = super::generic_impls::BigIntToZnHom<I, StaticRingBase<i64>, Self>;

    default fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_bigint(from, self, StaticRing::<i64>::RING.get_ring(), None)
    }

    default fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in_from_bigint(from, self, StaticRing::<i64>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as i64) < self.modulus);
            self.from_int_promise_reduced(n as i64)
        }, |_| unreachable!())
    }
}

impl Serialize for Zn64MBase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("Zn", *self.modulus()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Zn64MBase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("Zn", PhantomData::<i64>).deserialize(deserializer).map(|n| Zn64MBase::new(n as u64))
    }
}

macro_rules! impl_static_int_to_zn {
    ($($int:ident),*) => {
        $(
            impl CanHomFrom<StaticRingBase<$int>> for Zn64MBase {
                fn map_in(&self, _from: &StaticRingBase<$int>, el: $int, _hom: &Self::Homomorphism) -> Self::Element {
                    self.from_primitive_int(el)
                }
            }
        )*
    };
}

impl_static_int_to_zn!{ i8, i16, i32, i64, i128 }

impl_field_wrap_unwrap_homs!{ Zn64MBase, Zn64MBase }
impl_field_wrap_unwrap_isos!{ Zn64MBase, Zn64MBase }
impl_localpir_wrap_unwrap_homs!{ Zn64MBase, Zn64MBase }
impl_localpir_wrap_unwrap_isos!{ Zn64MBase, Zn64MBase }

#[derive(Clone, Copy)]
#[stability::unstable(feature = "enable")]
pub struct ZnBaseElementsIter<'a> {
    ring: &'a Zn64MBase,
    current: i64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

    type Item = Zn64MEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus {
            let result = self.current;
            self.current += 1;
            return Some(self.ring.from_int_promise_reduced(result));
        } else {
            return None;
        }
    }
}

impl FiniteRingSpecializable for Zn64MBase {
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.execute()
    }
}

impl FiniteRing for Zn64MBase {

    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        super::generic_impls::random_element(self, rng)
    }

    fn size<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        if other_ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(&(self.modulus() + 1)) <= other_ZZ.get_ring().representable_bits() {
            Some(int_cast(*self.modulus(), other_ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl PrincipalIdealRing for Zn64MBase {

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_div_min(RingRef::new(self), lhs, rhs)
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        // we can actually work in Montgomery form, since R is a unit modulo n
        let (s, t, d) = StaticRing::<i64>::RING.extended_ideal_gen(&lhs.0.try_into().unwrap(), &rhs.0.try_into().unwrap());
        let quo = RingRef::new(self).into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        (quo.map(s), quo.map(t), Zn64MEl(d as u64))
    }
}

impl StrassenHint for Zn64MBase {
    default fn strassen_threshold(&self) -> usize {
        6
    }
}

impl KaratsubaHint for Zn64MBase {
    default fn karatsuba_threshold(&self) -> usize {
        6
    }
}

impl ZnRing for Zn64MBase {

    type IntegerRingBase = StaticRingBase<i64>;
    type IntegerRing = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::IntegerRing {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        let mut result = self.montgomery_reduce(el.0 as u128) as i64;
        if result >= self.modulus {
            result -= self.modulus;
        }
        return result;
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        let result = self.smallest_positive_lift(el);
        if 2 * result >= self.modulus {
            return result - self.modulus;
        } else {
            return result;
        }
    }

    fn modulus(&self) -> &El<Self::IntegerRing> {
        &self.modulus
    }

    fn any_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        el.0 as i64
    }

    ///
    /// If the given integer is within `{ 0, ..., 9 * n }`, returns the corresponding
    /// element in `Z/nZ`. Any other input is considered a logic error.
    /// 
    /// This function follows [`ZnRing::from_int_promise_reduced()`], but is guaranteed
    /// to work on elements `{ 0, ..., 9 * n }` instead of only `{ 0, ..., n - 1 }`.
    /// 
    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        debug_assert!(x >= 0);
        debug_assert!(x as u64 <= 3 * self.repr_bound());
        Zn64MEl(self.montgomery_reduce(x as u128 * self.R_sqr_mod_n as u128))
    }
}

impl HashableElRing for Zn64MBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.integer_ring().hash(&self.smallest_positive_lift(*el), h)
    }
}

#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[cfg(test)]
fn elements<'a>(ring: &'a Zn64M) -> impl 'a + Iterator<Item = El<Zn64M>> {
    (0..63).map(|i| ring.coerce(&StaticRing::<i64>::RING, 1 << i))
}

#[cfg(test)]
const TEST_MODULI: [u64; 12] = [1, 3, 5, 7, 9, 11, 13, 15, 17, (1 << 60) - 3, (1 << 60) - 1, (1 << 60) + 1];

#[test]
fn test_montgomery_reduce() {
    LogAlgorithmSubscriber::init_test();
    let Zn = Zn64M::new(19);
    assert_eq!(9, Zn.get_ring().montgomery_reduce(1));
    assert_eq!(18, Zn.get_ring().montgomery_reduce(2));
    assert_eq!(18, Zn.get_ring().montgomery_reduce(40));
    assert_eq!(8, Zn.get_ring().montgomery_reduce(364));
    assert_eq!(17, Zn.get_ring().montgomery_reduce(289));
}

#[test]
fn test_from_int_promise_reduced() {
    LogAlgorithmSubscriber::init_test();
    let Zn = Zn64M::new(19);
    assert_eq!(17, Zn.get_ring().from_int_promise_reduced(1).0);
}

#[test]
fn test_sum() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        let Zn = Zn64M::new(n);
        assert_el_eq!(Zn, Zn.int_hom().map(10001 * 5000), Zn.sum((0..=10000).map(|x| Zn.int_hom().map(x))));
    }
}

#[test]
fn test_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        let ring = Zn64M::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, elements(&ring));
    }
}

#[test]
fn test_hash_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        if n != 1 {
            let ring = Zn64M::new(n);
            crate::ring::generic_tests::test_hash_axioms(&ring, elements(&ring));
        }
    }
}

#[test]
fn test_divisibility_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        let Zn = Zn64M::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, elements(&Zn));
    }
}

#[test]
fn test_zn_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        let Zn = Zn64M::new(n);
        super::generic_tests::test_zn_axioms(&Zn);
    }
}

#[test]
fn test_principal_ideal_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        if n < 100 {
            let R = Zn64M::new(n);
            crate::pid::generic_tests::test_principal_ideal_ring_axioms(R, R.elements());
        }
    }
}

#[test]
fn test_finite_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        if n < 100 {
            crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn64M::new(n));
        }
    }
}

#[test]
fn test_from_int_hom() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        let Zn = Zn64M::new(n);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i8>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i16>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i32>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i64>::RING, Zn, -8..8);
        crate::ring::generic_tests::test_hom_axioms(StaticRing::<i128>::RING, Zn, -8..8);
    }
    let Zn = Zn64M::new(5);
    assert_el_eq!(Zn, Zn.int_hom().map(3), Zn.can_hom(&StaticRing::<i64>::RING).unwrap().map(-1596802));
}

#[test]
fn test_smallest_positive_lift() {
    LogAlgorithmSubscriber::init_test();
    for n in TEST_MODULI {
        if n < 100 {
            let ring = Zn64M::new(n);
            for k in 0..=ring.get_ring().repr_bound() {
                assert_eq!(k as i64 % n as i64, ((ring.smallest_positive_lift(Zn64MEl(k)) as i128 * (1 << 64)) % n as i128) as i64);
            }
        }
    }
}

#[bench]
fn bench_hom_from_i64_large_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn64M::new(36028797018963971 /* = 2^55 + 3 */);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(Zn, Zn.int_hom().map(-1300), Zn.sum((0..100).flat_map(|_| (0..=56).map(|k| 1 << k)).map(|x| hom.map(x))))
    });
}

#[bench]
fn bench_hom_from_i64_small_modulus(bencher: &mut Bencher) {
    // the case that the modulus is large
    let Zn = Zn64M::new(17);
    bencher.iter(|| {
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        assert_el_eq!(Zn, Zn.int_hom().map(2850 * 5699), Zn.sum((0..5700).map(|x| hom.map(x))))
    });
}

#[test]
fn test_serialize() {
    LogAlgorithmSubscriber::init_test();
    let ring = Zn64M::new(129);
    crate::serialization::generic_tests::test_serialization(ring, ring.elements())
}