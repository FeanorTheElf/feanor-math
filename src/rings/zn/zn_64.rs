use std::marker::PhantomData;

use crate::ordered::OrderedRingStore;
use crate::primitive_int::*;
use crate::integer::*;
use crate::euclidean::EuclideanRingStore;
use crate::ring::*;
use crate::rings::rust_bigint::*;

use super::*;
use super::zn_barett;

///
/// Represents the ring `Z/nZ`.
/// A variant of Barett reduction is used to perform fast modular
/// arithmetic for `n` slightly smaller than 64 bit.
/// 
/// More concretely, the currently maximal supported modulus is `floor(2^62 / 9)`.
/// Note that the exact value might change in the future.
/// 
/// Standard arithmetic in this ring is slightly slower than in [`super::zn_42::ZnBase`],
/// which supports moduli up to 41 bits. However, this ring is perfectly suited for the
/// number theoretic transform together with [`ZnFastmulBase`], where it achieves basically 
/// the same speed as the 42-bit ring.
/// 
/// # Examples
/// 
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// let zn = Zn::new(7);
/// assert_el_eq!(&zn, &zn.one(), &zn.mul(zn.from_int(3), zn.from_int(5)));
/// ```
/// We have natural isomorphisms to [`super::zn_42::ZnBase`] that are extremely fast.
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// let R1 = zn_42::Zn::new(17);
/// let R2 = zn_64::Zn::new(17);
/// assert_el_eq!(&R2, &R2.from_int(6), &R2.coerce(&R1, R1.from_int(6)));
/// assert_el_eq!(&R1, &R1.from_int(16), &R2.cast(&R1, R2.from_int(16)));
/// ```
/// Too large moduli will give an error.
/// ```should_panic
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// Zn::new((1 << 62) / 9 + 1);
/// ```
/// 
pub struct ZnBase {
    modulus: i64,
    inv_modulus: u128
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
#[allow(non_upper_case_globals)]
const ZZbig: BigIntRing = BigIntRing::RING;

impl ZnBase {

    pub fn new(modulus: u64) -> Self {
        assert!(modulus > 1);
        // this should imply the statement we need later
        assert!(modulus <= ((1 << 62) / 9));
        // we want representatives to grow up to 6 * modulus
        assert!(modulus as u128 * 6 <= u64::MAX as u128);
        let inv_modulus = ZZbig.euclidean_div(ZZbig.power_of_two(128), &ZZbig.coerce(&ZZ, modulus as i64));
        // we need the product `inv_modulus * (6 * modulus)^2` to fit into 192 bit, should be implied by `modulus < ((1 << 62) / 9)`
        debug_assert!(ZZbig.is_lt(&ZZbig.mul_ref_fst(&inv_modulus, ZZbig.pow(ZZbig.mul_int(ZZbig.coerce(&ZZ, modulus as i64), 6), 2)), &ZZbig.power_of_two(192)));
        let inv_modulus = if ZZbig.eq_el(&inv_modulus, &ZZbig.power_of_two(127)) {
            1u128 << 127
        } else {
            ZZbig.cast(&StaticRing::<i128>::RING, inv_modulus) as u128
        };
        Self {
            modulus: modulus as i64,
            inv_modulus: inv_modulus
        }
    }

    fn modulus_u64(&self) -> u64 {
        self.modulus as u64
    }

    fn repr_bound(&self) -> u64 {
        self.modulus_u64() * 3
    }

    ///
    /// If input is bounded by `4 * self.repr_bound() * self.repr_bound()`, then the output
    /// is `< 3 * modulus` and congruent to the input.
    /// 
    fn bounded_reduce(&self, value: u128) -> u64 {
        debug_assert!(value <= 4 * self.repr_bound() as u128 * self.repr_bound() as u128);
        let (in_low, in_high) = ((value & ((1 << 64) - 1)) as u64, (value >> 64) as u64);
        let (invmod_low, invmod_high) = ((self.inv_modulus & ((1 << 64) - 1)) as u64, (self.inv_modulus >> 64) as u64);
        // we ignore the lowest part of the sum, causing an error of at most 1;
        // we also assume that `repr_bound * repr_bound * inv_modulus` fits into 192 bit
        let approx_quotient = ((in_low as u128 * invmod_high as u128 + in_high as u128 * invmod_low as u128) >> 64) as u64 + in_high * invmod_high;
        let result = (value - approx_quotient as u128 * self.modulus_u64() as u128) as u64;
        debug_assert!(result < 3 * self.modulus_u64());
        return result;
    }

    fn potential_reduce(&self, value: u64) -> u64 {
        if value > self.repr_bound() {
            value - self.repr_bound()
        } else {
            value
        }
    }

    fn complete_reduce(&self, mut value: u64) -> u64 {
        debug_assert!(value <= self.repr_bound());
        if value >= 2 * self.modulus_u64() {
            value -= 2 * self.modulus_u64();
        }
        if value >= self.modulus_u64() {
            value -= self.modulus_u64();
        }
        debug_assert!(value < self.modulus_u64());
        return value;
    }
}

/// 
/// Represents the ring `Z/nZ`, using a variant of Barett reduction
/// for moduli somewhat smaller than 64 bits. For details, see [`ZnBase`].
/// 
pub type Zn = RingValue<ZnBase>;

impl Zn {

    pub fn new(modulus: u64) -> Self {
        RingValue::from(ZnBase::new(modulus))
    }
}

#[derive(Clone, Copy)]
pub struct ZnEl(u64);

impl PartialEq for ZnBase {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl RingBase for ZnBase {

    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.potential_reduce(lhs.0 + rhs.0);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        lhs.0 = self.repr_bound() - lhs.0;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        lhs.0 = self.bounded_reduce(lhs.0 as u128 * rhs.0 as u128);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(lhs.0 <= self.repr_bound());
        debug_assert!(rhs.0 <= self.repr_bound());
        if lhs.0 > rhs.0 {
            self.is_zero(&ZnEl(lhs.0 - rhs.0))
        } else {
            self.is_zero(&ZnEl(rhs.0 - lhs.0))
        }
    }

    fn is_zero(&self, val: &Self::Element) -> bool {
        debug_assert!(val.0 <= self.repr_bound());
        self.complete_reduce(val.0) == 0
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", self.complete_reduce(value.0))
    }
}

impl_eq_based_self_iso!{ ZnBase }

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalHom<zn_barett::ZnBase<I>> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Homomorphism> {
        if self.modulus as i128 == *from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_barett::ZnBase<I>, el: <zn_barett::ZnBase<I> as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        ZnEl(from.smallest_positive_lift(el) as u64)
    }
}

impl<I: IntegerRingStore<Type = StaticRingBase<i128>>> CanonicalIso<zn_barett::ZnBase<I>> for ZnBase {

    type Isomorphism = <zn_barett::ZnBase<I> as CanonicalHom<StaticRingBase<i64>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<I>) -> Option<Self::Isomorphism> {
        if self.modulus as i128 == *from.modulus() {
            from.has_canonical_hom(self.integer_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_barett::ZnBase<I>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_barett::ZnBase<I> as RingBase>::Element {
        from.map_in(self.integer_ring().get_ring(), el.0 as i64, iso)
    }
}

impl CanonicalHom<zn_42::ZnBase> for ZnBase {

    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_42::ZnBase) -> Option<Self::Homomorphism> {
        if *self.modulus() == *from.modulus() {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_42::ZnBase, el: <zn_42::ZnBase as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        // we usually require much smaller representatives as zn_42 (except for very large moduli), so do not
        // specialize this
        ZnEl(from.smallest_positive_lift(el) as u64)
    }
}

pub enum ToZn42Iso {
    Trivial, ReduceRequired(<zn_42::ZnBase as CanonicalHom<StaticRingBase<i64>>>::Homomorphism)
}

impl CanonicalIso<zn_42::ZnBase> for ZnBase {

    type Isomorphism = ToZn42Iso;

    fn has_canonical_iso(&self, from: &zn_42::ZnBase) -> Option<Self::Isomorphism> {
        if *self.modulus() == *from.modulus() {
            if from.repr_bound() >= self.repr_bound() {
                Some(ToZn42Iso::Trivial)
            } else {
                Some(ToZn42Iso::ReduceRequired(from.has_canonical_hom(self.integer_ring().get_ring())?))
            }
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_42::ZnBase, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <zn_42::ZnBase as RingBase>::Element {
        match iso {
            ToZn42Iso::Trivial => from.from_bounded(el.0),
            ToZn42Iso::ReduceRequired(reduce_hom) => from.map_in(self.integer_ring().get_ring(), el.0 as i64, reduce_hom)
        }
    }
}

impl DivisibilityRing for ZnBase {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs, self.modulus())
    }
}
trait GenericMapInFromInt: IntegerRing + CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i64>> {}

impl GenericMapInFromInt for StaticRingBase<i64> {}
impl GenericMapInFromInt for StaticRingBase<i128> {}
impl GenericMapInFromInt for RustBigintRingBase {}

#[cfg(feature = "mpir")]
impl GenericMapInFromInt for crate::rings::mpir::MPZBase {}

impl<I: ?Sized + GenericMapInFromInt> CanonicalHom<I> for ZnBase {

    type Homomorphism = super::generic_impls::IntegerToZnHom<I, StaticRingBase<i128>, Self>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_int(from, self, StaticRing::<i128>::RING.get_ring(), Some(&(self.repr_bound() as i128 * self.repr_bound() as i128)))
    }

    fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in_from_int(from, self, StaticRing::<i128>::RING.get_ring(), el, hom, |n| {
            debug_assert!((n as u64) < self.modulus_u64());
            ZnEl(n as u64)
        }, |n| {
            debug_assert!(n <= (self.repr_bound() as i128 * self.repr_bound() as i128));
            ZnEl(self.bounded_reduce(n as u128))
        })
    }
}

pub struct IntToZnHom<T: PrimitiveInt> {
    reduction_is_trivial: bool,
    int: PhantomData<T>
}

macro_rules! impl_static_int_to_zn {
    ($($int:ident),*) => {
        $(
            impl CanonicalHom<StaticRingBase<$int>> for ZnBase {

                type Homomorphism = IntToZnHom<$int>;

                fn has_canonical_hom(&self, _from: &StaticRingBase<$int>) -> Option<Self::Homomorphism> {
                    if self.repr_bound() > $int::MAX as u64 {
                        Some(IntToZnHom { reduction_is_trivial: true, int: PhantomData })
                    } else {
                        Some(IntToZnHom { reduction_is_trivial: false, int: PhantomData })
                    }
                }

                fn map_in(&self, _from: &StaticRingBase<$int>, el: $int, hom: &IntToZnHom<$int>) -> Self::Element {
                    if std::intrinsics::likely(hom.reduction_is_trivial) {
                        if el < 0 {
                            self.negate(ZnEl(-(el as i128) as u64))
                        } else {
                            ZnEl(el as u64)
                        }
                    } else {
                        if el < 0 {
                            self.negate(ZnEl(self.bounded_reduce(-(el as i128) as u128)))
                        } else {
                            ZnEl(self.bounded_reduce(el as u128))
                        }
                    }
                }
            }
        )*
    };
}

impl_static_int_to_zn!{ i8, i16, i32 }

pub struct ZnBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u64
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {

    type Item = ZnEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus_u64() {
            let result = self.current;
            self.current += 1;
            return Some(ZnEl(result));
        } else {
            return None;
        }
    }
}

impl FiniteRing for ZnBase {

    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: 0
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        super::generic_impls::random_element(self, rng)
    }

    fn size<I: IntegerRingStore>(&self, other_ZZ: &I) -> El<I>
        where I::Type: IntegerRing
    {
        int_cast(*self.modulus(), other_ZZ, self.integer_ring())
    }
}

impl ZnRing for ZnBase {

    type IntegerRingBase = StaticRingBase<i64>;
    type Integers = StaticRing<i64>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<i64>::RING
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.complete_reduce(el.0) as i64
    }

    fn modulus(&self) -> &El<Self::Integers> {
        &self.modulus
    }
}

impl HashableElRing for ZnBase {

    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        self.integer_ring().hash(&self.smallest_positive_lift(*el), h)
    }
}

#[cfg(test)]
fn elements<'a>(ring: &'a Zn) -> impl 'a + Iterator<Item = El<Zn>> {
    (0..63).map(|i| ring.coerce(&ZZ, 1 << i))
}

#[test]
fn test_ring_axioms() {
    for n in [2, 5, 7, 17] {
        let Zn = Zn::new(n);
        crate::ring::generic_tests::test_ring_axioms(&Zn, Zn.elements());
    }
    for n in [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1] {
        let Zn = Zn::new(n);
        crate::ring::generic_tests::test_ring_axioms(&Zn, elements(&Zn));
    }
}

#[test]
fn test_iso_zn_42() {
    for n in [2, 5, 17, (1 << 41) - 1] {
        let R1 = Zn::new(n);
        let R2 = zn_42::Zn::new(n);
        let elements = (1..42).map(|i| 1 << i).map(|x| R2.coerce(&ZZ, x));
        crate::ring::generic_tests::test_hom_axioms(&R2, &R1, elements.clone());
        crate::ring::generic_tests::test_iso_axioms(&R2, &R1, elements);
    }
}

#[test]
fn test_divisibility_axioms() {
    for n in [2, 5, 7, 17] {
        let Zn = Zn::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, Zn.elements());
    }
    for n in [(1 << 41) - 1, (1 << 42) - 1, (1 << 58) - 1, (1 << 58) + 1, (3 << 57) - 1, (3 << 57) + 1] {
        let Zn = Zn::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&Zn, elements(&Zn));
    }
}

#[test]
fn test_finite_ring_axioms() {
    for n in [2, 5, 7, 17] {
        let Zn = Zn::new(n);
        super::generic_tests::test_zn_axioms(&Zn);
    }
}