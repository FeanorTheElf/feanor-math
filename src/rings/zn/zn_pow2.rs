use smallvec::SmallVec;

use crate::delegate::DelegateRing;
use crate::divisibility::DivisibilityRing;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::primitive_int::{StaticRing, StaticRingBase};
use crate::ring::*;
use crate::rings::rust_bigint::{RustBigintRing, RustBigintRingBase};
use crate::rings::zn::*;
use crate::specialization::*;
use crate::{
    impl_field_wrap_unwrap_homs, impl_field_wrap_unwrap_isos, impl_localpir_wrap_unwrap_homs,
    impl_localpir_wrap_unwrap_isos,
};

const DEFAULT_SMALLVEC_SIZE: usize = 4;

/// Ring representing `Z/nZ`, where `n = 2^k` for some `k`, computing the modular reductions
/// via standard bit masking.
///
/// # Performance
///
/// This implementation uses schoolbook multiplication.  As such, it is only optimized for a
/// small enough `k`.
/// Elements are internally stored using `SmallVec<[u64; N]>`, where `N` is a compile-time
/// constant, and is the threshold for when dynamic allocation is used.
/// It is recommended to set N to be exactly div_ceil(k, 64).
///
///
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_pow2::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::assert_el_eq;
/// let R = Z2k::<3>::new(130);
/// let a = R
///     .get_ring()
///     .from_base_u64_repr([u64::MAX, u64::MAX, 3].into_iter());
/// assert_el_eq!(R, R.int_hom().map(-1), a);
/// ```
#[derive(Clone)]
pub struct Z2kBase<const N: usize = DEFAULT_SMALLVEC_SIZE> {
    k: usize,
    n_limbs: usize,
    last_limb_mask: u64,
    modulus: El<RustBigintRing>,
}
impl<const N: usize> Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    pub fn new(k: usize) -> Self {
        assert!(k >= 1);
        let n_limbs = k.div_ceil(64);
        assert!(
            n_limbs <= 16,
            "This implementation is not optimized for such a large modulus."
        );
        let modulus = RustBigintRing::RING.power_of_two(k);

        let last_limb_mask = if k % 64 == 0 {
            0xFFFFFFFFFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF >> (64 - (k % 64))
        };
        Self {
            k,
            n_limbs,
            last_limb_mask,
            modulus,
        }
    }

    fn mask_el(&self, el: &mut Z2kEl<N>) {
        el.0[self.n_limbs - 1] &= self.last_limb_mask;
        el.0.truncate(self.n_limbs);
    }

    fn bigint_to_el(&self, ZZ: &RustBigintRing, x: &El<RustBigintRing>) -> Z2kEl<N> {
        let mut rem = ZZ.euclidean_rem(ZZ.clone_el(x), self.modulus());

        // normalize to `[0, 2^k)`.
        if ZZ.is_neg(&rem) {
            ZZ.add_assign_ref(&mut rem, self.modulus());
        }

        let mut limbs = SmallVec::from_elem(0, self.n_limbs);
        for (i, d) in ZZ.get_ring().abs_base_u64_repr(&rem).take(self.n_limbs).enumerate() {
            limbs[i] = d;
        }
        Z2kEl(limbs)
    }

    fn el_to_bigint(&self, ZZ: &RustBigintRing, mut el: Z2kEl<N>) -> El<RustBigintRing> {
        self.mask_el(&mut el);
        let mut acc = ZZ.zero();
        let i128_ring = StaticRing::<i128>::RING;
        for i in 0..self.n_limbs {
            let limb = int_cast(el.0[i] as i128, ZZ, &i128_ring);
            if !ZZ.is_zero(&limb) {
                let shifted = ZZ.mul(limb, ZZ.power_of_two(64 * i));
                ZZ.add_assign(&mut acc, shifted);
            }
        }
        acc
    }

    /// Returns an iterator over the digits of the `2^64`-adic digit
    /// representation of the absolute value of the given element.
    pub fn abs_base_u64_repr(&self, el: &Z2kEl<N>) -> impl Iterator<Item = u64> {
        el.0.iter().take(self.n_limbs).enumerate().map(|(i, &x)| {
            if i == self.n_limbs - 1 {
                x & self.last_limb_mask
            } else {
                x
            }
        })
    }

    /// Interprets the elements of the iterator as digits in a `2^64`-adic
    /// digit representation, and returns the big integer represented by it.
    pub fn from_base_u64_repr<I>(&self, data: I) -> Z2kEl<N>
    where
        I: Iterator<Item = u64>,
    {
        let data = data.take(self.n_limbs).enumerate().map(|(i, x)| {
            if i == self.n_limbs - 1 {
                x & self.last_limb_mask
            } else {
                x
            }
        });
        Z2kEl(SmallVec::from_iter(data))
    }
}

impl<const N: usize> PartialEq for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    fn eq(&self, other: &Self) -> bool { self.k == other.k }
}

impl<const N: usize> Eq for Z2kBase<N> where [u64; N]: smallvec::Array<Item = u64> {}

pub type Z2k<const N: usize = DEFAULT_SMALLVEC_SIZE> = RingValue<Z2kBase<N>>;
impl<const N: usize> Z2k<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    pub fn new(log_modulus: usize) -> Self { RingValue::from(Z2kBase::<N>::new(log_modulus)) }
}

#[derive(Clone)]
pub struct Z2kEl<const N: usize>(SmallVec<[u64; N]>)
where
    [u64; N]: smallvec::Array<Item = u64>;
impl<const N: usize> RingBase for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type Element = Z2kEl<N>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element { val.clone() }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let mut carry = 0u64;
        for i in 0..self.n_limbs {
            let (sum, o0) = lhs.0[i].overflowing_add(rhs.0[i]);
            let (sum, o1) = sum.overflowing_add(carry);
            lhs.0[i] = sum;
            carry = (o0 || o1) as u64;
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { self.add_assign_ref(lhs, &rhs); }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let mut borrow = 0u64;
        for i in 0..self.n_limbs {
            let (diff, o0) = lhs.0[i].overflowing_sub(rhs.0[i]);
            let (diff, o1) = diff.overflowing_sub(borrow);
            lhs.0[i] = diff;
            borrow = (o0 || o1) as u64;
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        let mut z = self.zero();
        self.sub_assign_ref(&mut z, lhs);
        *lhs = z;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { self.mul_assign_ref(lhs, &rhs); }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { *lhs = mul_z2k(lhs, rhs, self.n_limbs); }

    fn from_int(&self, value: i32) -> Self::Element {
        let ZZ = &RustBigintRing::RING;
        let x = ZZ.int_hom().map(value);
        self.bigint_to_el(ZZ, &x)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        let mut a = lhs.clone();
        let mut b = rhs.clone();
        self.mask_el(&mut a);
        self.mask_el(&mut b);
        a.0 == b.0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        let mut v = value.clone();
        self.mask_el(&mut v);
        v.0.iter().all(|x| *x == 0)
    }

    fn is_one(&self, value: &Self::Element) -> bool { self.eq_el(value, &self.one()) }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg_within<'a>(
        &self,
        value: &Self::Element,
        out: &mut std::fmt::Formatter<'a>,
        _: EnvBindingStrength,
    ) -> std::fmt::Result {
        let ZZ = RustBigintRing::RING;
        write!(out, "{}", ZZ.format(&self.el_to_bigint(&ZZ, value.clone())))
    }

    fn characteristic<J: RingStore + Copy>(&self, ZZ: J) -> Option<El<J>>
    where
        J::Type: IntegerRing,
    {
        self.size(ZZ)
    }

    fn is_approximate(&self) -> bool { false }
}

// Schoolbook multiplication, only keeps the first `n_limbs` limbs.
fn mul_z2k<const N: usize>(Z2kEl(a): &Z2kEl<N>, Z2kEl(b): &Z2kEl<N>, n_limbs: usize) -> Z2kEl<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    let mut res = SmallVec::from_elem(0, n_limbs);
    for i in 0..n_limbs {
        let mut carry = 0u128;
        for j in 0..n_limbs - i {
            let idx = i + j;
            let sum = a[i] as u128 * b[j] as u128 + res[idx] as u128 + carry;
            res[idx] = sum as u64;
            carry = sum >> 64;
        }
    }
    Z2kEl(res)
}

impl<const N: usize> DivisibilityRing for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs)
    }
}

impl<const N: usize> CanHomFrom<Z2kBase<N>> for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &Z2kBase<N>) -> Option<Self::Homomorphism> {
        // Canonical homomorphism is expected to be unital. As such, characteristics must match.
        (from.k == self.k).then_some(())
    }

    fn map_in(
        &self,
        _from: &Z2kBase<N>,
        el: <Z2kBase<N> as RingBase>::Element,
        _: &Self::Homomorphism,
    ) -> Self::Element {
        el
    }
}

impl<const N: usize> CanIsoFromTo<Z2kBase<N>> for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &Z2kBase<N>) -> Option<Self::Isomorphism> { (self.k == from.k).then_some(()) }

    fn map_out(
        &self,
        _from: &Z2kBase<N>,
        el: Self::Element,
        _: &Self::Isomorphism,
    ) -> <Z2kBase<N> as RingBase>::Element {
        el
    }
}

#[derive(Clone)]
pub struct Z2KBaseElementsIter<'a, const N: usize>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    ring: &'a Z2kBase<N>,
    next_index: Z2kEl<N>,
    wrap_around: bool,
}

impl<'a, const N: usize> Iterator for Z2KBaseElementsIter<'a, N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type Item = Z2kEl<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.wrap_around {
            return None;
        }

        let out = self.next_index.clone();
        self.ring.add_assign(&mut self.next_index, self.ring.one());
        if self.ring.is_zero(&self.next_index) {
            self.wrap_around = true;
        }
        Some(out)
    }
}

impl<const N: usize> FiniteRingSpecializable for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output { op.execute() }
}

impl<const N: usize> FiniteRing for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type ElementsIter<'a>
        = Z2KBaseElementsIter<'a, N>
    where
        Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        Z2KBaseElementsIter {
            ring: self,
            next_index: self.zero(),
            wrap_around: false,
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        super::generic_impls::random_element(self, rng)
    }

    fn size<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
    where
        I::Type: IntegerRing,
    {
        {
            if other_ZZ.get_ring().representable_bits().is_none()
                || self.k < other_ZZ.get_ring().representable_bits().unwrap()
            {
                Some(int_cast(
                    RustBigintRing::RING.clone_el(&self.modulus),
                    other_ZZ,
                    &RustBigintRing::RING,
                ))
            } else {
                None
            }
        }
    }
}

impl<const N: usize> PrincipalIdealRing for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_div_min(RingRef::new(self), lhs, rhs)
    }

    fn extended_ideal_gen(
        &self,
        lhs: &Self::Element,
        rhs: &Self::Element,
    ) -> (Self::Element, Self::Element, Self::Element) {
        let ZZ = RustBigintRing::RING;
        let l = self.el_to_bigint(&ZZ, lhs.clone());
        let r = self.el_to_bigint(&ZZ, rhs.clone());
        let (s, t, d) = ZZ.extended_ideal_gen(&l, &r);
        let [s, t, d] = [s, t, d].map(|x| self.bigint_to_el(&ZZ, &x));
        (s, t, d)
    }
}

impl<const N: usize, I: ?Sized + IntegerRing> CanHomFrom<I> for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type Homomorphism = super::generic_impls::BigIntToZnHom<I, RustBigintRingBase, Self>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_bigint(from, self, RustBigintRing::RING.get_ring(), None)
    }

    default fn map_in(&self, from: &I, el: I::Element, hom: &Self::Homomorphism) -> Self::Element {
        let ZZ = &RustBigintRing::RING;
        super::generic_impls::map_in_from_bigint(
            from,
            self,
            ZZ.get_ring(),
            el,
            hom,
            |n| self.bigint_to_el(ZZ, &n),
            |n| self.bigint_to_el(ZZ, &n),
        )
    }
}

impl<const N: usize> CanHomFrom<RustBigintRingBase> for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    fn map_in(
        &self,
        _from: &RustBigintRingBase,
        el: <RustBigintRingBase as RingBase>::Element,
        _: &Self::Homomorphism,
    ) -> Self::Element {
        self.bigint_to_el(&RustBigintRing::RING, &el)
    }
}

macro_rules! impl_static_int_to_z2k {
    ($($int:ident),*) => {
        $(
            impl<const N: usize> CanHomFrom<StaticRingBase<$int>> for Z2kBase<N>
            where
                [u64; N]: smallvec::Array<Item = u64>,
            {
                fn map_in(
                    &self,
                    from: &StaticRingBase<$int>,
                    el: $int,
                    _: &Self::Homomorphism,
                ) -> Self::Element {
                    let ZZ = &RustBigintRing::RING;
                    let x = int_cast(el, ZZ, RingRef::new(from));
                    self.bigint_to_el(ZZ, &x)
                }
            }
        )*
    };
}

impl_static_int_to_z2k! { i8, i16, i32, i64, i128 }

impl<const N: usize> ZnRing for Z2kBase<N>
where
    [u64; N]: smallvec::Array<Item = u64>,
{
    type IntegerRingBase = RustBigintRingBase;
    type IntegerRing = RustBigintRing;

    fn integer_ring(&self) -> &Self::IntegerRing { &RustBigintRing::RING }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        let ZZ = RustBigintRing::RING;
        self.el_to_bigint(&ZZ, el)
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        let result = self.smallest_positive_lift(el);
        let mut mod_half = self.integer_ring().clone_el(self.modulus());
        self.integer_ring().euclidean_div_pow_2(&mut mod_half, 1);
        if self.integer_ring().is_gt(&result, &mod_half) {
            self.integer_ring().sub_ref_snd(result, self.modulus())
        } else {
            result
        }
    }

    fn modulus(&self) -> &El<Self::IntegerRing> { &self.modulus }

    fn any_lift(&self, el: Self::Element) -> El<Self::IntegerRing> { self.smallest_positive_lift(el) }

    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        debug_assert!({
            let ZZ = RustBigintRing::RING;
            !ZZ.is_neg(&x) && ZZ.is_lt(&x, &self.modulus)
        });
        self.bigint_to_el(&RustBigintRing::RING, &x)
    }
}

impl_field_wrap_unwrap_homs! { Z2kBase, Z2kBase }
impl_field_wrap_unwrap_isos! { Z2kBase, Z2kBase }
impl_localpir_wrap_unwrap_homs! { Z2kBase, Z2kBase }
impl_localpir_wrap_unwrap_isos! { Z2kBase, Z2kBase }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive_int::StaticRing;
    use crate::ring::generic_tests as ring_generic_tests;
    use crate::rings::zn::generic_tests as zn_generic_tests;

    const SMALL_Ks: [usize; 2] = [3, 8];
    const ZZ: BigIntRing = BigIntRing::RING;

    #[test]
    fn test_z2k_can_hom_map_in_large_power() {
        let r: Z2k = Z2k::new(256);
        zn_generic_tests::test_map_in_large_int(r);
    }

    #[test]
    fn test_z2k_can_hom_axioms_static_i32() {
        let to: Z2k = Z2k::new(14);
        ring_generic_tests::test_hom_axioms(StaticRing::<i32>::RING, to, -12i32..=12);
    }

    #[test]
    fn test_z2k_can_hom_axioms_bigint_ring() {
        let to: Z2k = Z2k::new(20);
        let edge = [
            ZZ.zero(),
            ZZ.one(),
            ZZ.negate(ZZ.one()),
            ZZ.int_hom().map(17),
            ZZ.int_hom().map(-42),
            ZZ.pow(ZZ.int_hom().map(2), 200),
        ];
        ring_generic_tests::test_hom_axioms(&ZZ, to, edge.into_iter());
    }

    #[test]
    fn test_z2k_pow2_64_add_mul_edge_cases() {
        let r: Z2k = Z2k::new(64);

        let almost_mod = ZZ.sub(ZZ.power_of_two(64), ZZ.int_hom().map(5));
        let a = r.coerce(&ZZ, almost_mod);
        let b = r.int_hom().map(5);
        assert_el_eq!(r, r.zero(), r.add_ref(&a, &b));

        // `2^32 * 2^32` = 0 mod `2^64`.
        let pow32 = ZZ.power_of_two(32);
        let u = r.coerce(&ZZ, pow32);
        assert_el_eq!(r, r.zero(), r.mul_ref(&u, &u));
    }

    #[test]
    fn test_z2k_pow2_130_add_mul_edge_cases() {
        let r: Z2k = Z2k::new(130);

        // `k` is not a multiple of 64, so the last limb is partially masked.
        let almost_mod = ZZ.sub(ZZ.power_of_two(130), ZZ.int_hom().map(11));
        let a = r.coerce(&ZZ, almost_mod);
        let b = r.int_hom().map(11);
        assert_el_eq!(r, r.zero(), r.add_ref(&a, &b));

        // `2^100 + 2^129` = 0 mod `2^130`.
        let hi = ZZ.add(ZZ.power_of_two(100), ZZ.power_of_two(129));
        let expected = ZZ.euclidean_rem(ZZ.mul_ref(&hi, &hi), &r.modulus());
        let x = r.coerce(&ZZ, hi);
        let y = r.mul_ref(&x, &x);
        assert!(ZZ.eq_el(&expected, &r.smallest_positive_lift(y)));
    }

    #[test]
    fn test_ring_axioms_z2kbase() {
        for k in SMALL_Ks {
            let ring: Z2k = Z2k::new(k);
            crate::ring::generic_tests::test_ring_axioms(&ring, ring.elements())
        }
    }

    #[test]
    fn test_zn_ring_axioms_znbase() {
        for k in SMALL_Ks {
            let ring: Z2k = Z2k::new(k);
            crate::rings::zn::generic_tests::test_zn_axioms(ring);
        }
    }

    #[test]
    fn test_divisibility_axioms() {
        for k in SMALL_Ks {
            let R: Z2k = Z2k::new(k);
            crate::divisibility::generic_tests::test_divisibility_axioms(&R, R.elements());
        }
    }

    #[test]
    fn test_principal_ideal_ring_axioms() {
        for k in SMALL_Ks {
            let R: Z2k = Z2k::new(k);
            crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
        }
    }

    #[test]
    fn test_finite_field_axioms() {
        for k in SMALL_Ks {
            let R: Z2k = Z2k::new(k);
            crate::rings::finite::generic_tests::test_finite_ring_axioms(&R);
        }
    }
}
