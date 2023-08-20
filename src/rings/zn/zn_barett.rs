use crate::divisibility::DivisibilityRing;
use crate::divisibility::DivisibilityRingStore;
use crate::euclidean::EuclideanRingStore;
use crate::integer::IntegerRing;
use crate::integer::IntegerRingStore;
use crate::ordered::OrderedRingStore;
use crate::ring::*;
use crate::rings::field::AssumeFieldDivision;
use crate::rings::zn::*;
use crate::algorithms;
use crate::primitive_int::*;

///
/// Ring representing `Z/nZ`, computing the modular reductions
/// via a Barett-reduction algorithm. This is a fast general-purpose
/// method, but note that it is required that `n^4` fits into the
/// supplied integer type.
/// 
/// # Performance
/// 
/// For small moduli, this implementation cannot use all the bounds
/// on the representative sizes, and hence often has to compute with
/// more precision than necessary.
/// In these cases, specialized implementations (like [`crate::rings::zn::zn_42::Zn`])
/// can be much faster.
///
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_barett::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i64>::RING, 257);
/// let a = R.from_int(16);
/// assert!(R.eq_el(&R.from_int(-1), &R.mul_ref(&a, &a)));
/// assert!(R.is_one(&R.pow(a, 4)));
/// ```
/// However, this will panic as `2053^3 > i32::MAX`.
/// ```should_panic
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_barett::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::new(StaticRing::<i32>::RING, 2053);
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
/// assert!(R.eq_el(&R.from_int(120493), &R.coerce(&S, S.from_int(120493))));
/// ```
///
pub struct ZnBase<I: IntegerRingStore> 
    where I::Type: IntegerRing
{
    integer_ring: I,
    modulus: El<I>,
    inverse_modulus: El<I>,
    inverse_modulus_bitshift: usize,
}

pub type Zn<I> = RingValue<ZnBase<I>>;

impl<I: IntegerRingStore> Zn<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        RingValue::from(ZnBase::new(integer_ring, modulus))
    }
}

impl<I: IntegerRingStore> ZnBase<I> 
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    pub fn new(integer_ring: I, modulus: El<I>) -> Self {
        assert!(integer_ring.is_geq(&modulus, &integer_ring.from_int(2)));

        // have k such that `2^k >= modulus^2`
        // then `floor(2^k / modulus) * x >> k` differs at most 1 from `floor(x / modulus)`
        // if `x < 2^k`, which is the case after multiplication
        let k = integer_ring.abs_log2_ceil(&modulus).unwrap() * 2;
        let mut mod_square_bound = integer_ring.one();
        integer_ring.mul_pow_2(&mut mod_square_bound, k);
        let inverse_modulus = integer_ring.euclidean_div(mod_square_bound, &modulus);

        // check that this expression does not overflow
        integer_ring.mul_ref_snd(integer_ring.pow(integer_ring.clone_el(&modulus), 2), &inverse_modulus);

        return ZnBase {
            integer_ring: integer_ring,
            modulus: modulus,
            inverse_modulus: inverse_modulus,
            inverse_modulus_bitshift: k
        };
    }

    fn project_leq_n_square(&self, n: &mut El<I>) {
        assert!(!self.integer_ring.is_neg(&n));
        let mut subtract = self.integer_ring.mul_ref(&n, &self.inverse_modulus);
        self.integer_ring.euclidean_div_pow_2(&mut subtract, self.inverse_modulus_bitshift);
        self.integer_ring.mul_assign_ref(&mut subtract, &self.modulus);
        self.integer_ring.sub_assign(n, subtract);
        if self.integer_ring.is_geq(&n, &self.modulus) {
            self.integer_ring.sub_assign_ref(n, &self.modulus);
        }
        assert!(self.integer_ring.is_lt(&n, &self.modulus), "The input is not smaller than {}^2", self.integer_ring.format(&self.modulus));
    }

    fn from_exact(&self, mut n: El<I>) -> ZnEl<I> {
        let negative = self.integer_ring.is_neg(&n);
        n = self.integer_ring.abs(n);
        assert!(self.integer_ring.is_lt(&n, &self.modulus));
        if negative {
            self.negate(ZnEl(n))
        } else {
            ZnEl(n)
        }
    }

    ///
    /// Returns either the inverse of x (as Ok()) or a nontrivial
    /// factor of the modulus (as Err())
    ///
    pub fn invert(&self, x: ZnEl<I>) -> Result<ZnEl<I>, El<I>> {
        let (s, _, d) = algorithms::eea::eea(self.integer_ring().clone_el(&x.0), self.integer_ring().clone_el(self.modulus()), &self.integer_ring);
        if self.integer_ring.is_neg_one(&d) || self.integer_ring.is_one(&d) {
            Ok(self.from_exact(s))
        } else {
            Err(d)
        }
    }
}

pub struct ZnEl<I: IntegerRingStore>(El<I>)
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>;

impl<I: IntegerRingStore> Clone for ZnEl<I> 
    where El<I>: Clone,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingStore> Copy for ZnEl<I>
    where El<I>: Copy,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{}

impl<I: IntegerRingStore> RingBase for ZnBase<I> 
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Element = ZnEl<I>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        ZnEl(self.integer_ring().clone_el(&val.0))
    }

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

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.integer_ring.eq_el(&lhs.0, &rhs.0)
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

impl<I: IntegerRingStore> Clone for ZnBase<I> 
    where I: Clone,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    fn clone(&self) -> Self {
        ZnBase {
            integer_ring: self.integer_ring.clone(),
            modulus: self.integer_ring.clone_el(&self.modulus),
            inverse_modulus: self.integer_ring.clone_el(&self.inverse_modulus),
            inverse_modulus_bitshift: self.inverse_modulus_bitshift
        }
    }
}

impl<I: IntegerRingStore> DivisibilityRing for ZnBase<I> 
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let d = algorithms::eea::gcd(self.integer_ring().clone_el(&lhs.0), self.integer_ring().clone_el(&rhs.0), &self.integer_ring);
        if self.integer_ring.is_zero(&d) {
            return Some(self.zero());
        } else if let Ok(inv) = self.invert(self.from_exact(self.integer_ring.checked_div(&rhs.0, &d).unwrap())) {
            return Some(self.mul(inv, self.from_exact(self.integer_ring.checked_div(&lhs.0, &d).unwrap())));
        } else {
            return None;
        }
    }
}

impl<I: IntegerRingStore> AssumeFieldDivision for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    fn assume_field_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        return self.mul_ref_fst(lhs, self.invert(self.clone_el(rhs)).ok().unwrap());
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanonicalHom<ZnBase<J>> for ZnBase<I> 
    where I::Type: IntegerRing + CanonicalHom<J::Type> + CanonicalIso<StaticRingBase<i32>>,
        J::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Homomorphism =  <I::Type as CanonicalHom<J::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ZnBase<J>) -> Option<Self::Homomorphism> {
        let base_hom = <I::Type as CanonicalHom<J::Type>>::has_canonical_hom(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if self.integer_ring.eq_el(
            &self.modulus,
            &<I::Type as CanonicalHom<J::Type>>::map_in(self.integer_ring.get_ring(), from.integer_ring.get_ring(), from.integer_ring().clone_el(&from.modulus), &base_hom)
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

impl<I: IntegerRingStore> PartialEq for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    fn eq(&self, other: &Self) -> bool {
        self.integer_ring.get_ring() == other.integer_ring.get_ring() && self.integer_ring.eq_el(&self.modulus, &other.modulus)
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> CanonicalIso<ZnBase<J>> for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<J::Type> + CanonicalIso<StaticRingBase<i32>>,
        J::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Isomorphism = <I::Type as CanonicalIso<J::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ZnBase<J>) -> Option<Self::Isomorphism> {
        let base_iso = <I::Type as CanonicalIso<J::Type>>::has_canonical_iso(self.integer_ring.get_ring(), from.integer_ring.get_ring())?;
        if from.integer_ring().eq_el(
            from.modulus(),
            &<I::Type as CanonicalIso<J::Type>>::map_out(self.integer_ring.get_ring(), from.integer_ring.get_ring(), self.integer_ring().clone_el(self.modulus()), &base_iso)
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
    where I::Type: IntegerRing, 
        J: CanonicalIso<I::Type>
{
    type Homomorphism = generic_impls::GenericIntegerToZnHom<J, I::Type, ZnBase<I>>;

    fn has_canonical_hom(&self, from: &J) -> Option<Self::Homomorphism> {
        generic_impls::generic_has_canonical_hom_from_int(from, self, self.integer_ring.get_ring(), Some(&self.integer_ring.mul_ref(self.modulus(), self.modulus())))
    }

    fn map_in(&self, from: &J, el: J::Element, hom: &Self::Homomorphism) -> Self::Element {
        generic_impls::generic_map_in_from_int(from, self, self.integer_ring.get_ring(), el, hom, |n| {
            debug_assert!(self.integer_ring.is_lt(&n, &self.modulus));
            ZnEl(n)
        }, |mut n| {
            debug_assert!(self.integer_ring.is_lt(&n, &self.integer_ring.mul_ref(&self.modulus, &self.modulus)));
            self.project_leq_n_square(&mut n);
            ZnEl(n)
        })
    }
}

pub struct ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    ring: &'a ZnBase<I>,
    current: El<I>
}

impl<'a, I> Iterator for ZnBaseElementsIter<'a, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
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

impl<I: IntegerRingStore> FiniteRing for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type ElementsIter<'a> = ZnBaseElementsIter<'a, I>
        where Self: 'a;

    fn elements<'a>(&'a self) -> ZnBaseElementsIter<'a, I> {
        ZnBaseElementsIter {
            ring: self,
            current: self.integer_ring().zero()
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        generic_impls::generic_random_element(self, rng)
    }
    
    fn size<J: IntegerRingStore>(&self, ZZ: &J) -> El<J>
        where J::Type: IntegerRing
    {
        int_cast(self.integer_ring().clone_el(self.modulus()), ZZ, self.integer_ring())
    }
}

impl<I: IntegerRingStore> ZnRing for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
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

impl<R: ZnRingStore<Type = ZnBase<I>>, I: IntegerRingStore> CanonicalHom<ZnBase<I>> for AsFieldBase<R>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Homomorphism = <ZnBase<I> as CanonicalHom<ZnBase<I>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ZnBase<I>) -> Option<Self::Homomorphism> {
        <ZnBase<I> as CanonicalHom<ZnBase<I>>>::has_canonical_hom(self.base_ring().get_ring(), from)
    }

    fn map_in(&self, from: &ZnBase<I>, el: <ZnBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from(<ZnBase<I> as CanonicalHom<ZnBase<I>>>::map_in(self.base_ring().get_ring(), from, el, hom))
    }
}

impl<R: ZnRingStore<Type = ZnBase<I>>, I: IntegerRingStore> CanonicalIso<ZnBase<I>> for AsFieldBase<R>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Isomorphism = <ZnBase<I> as CanonicalIso<ZnBase<I>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ZnBase<I>) -> Option<Self::Isomorphism> {
        <ZnBase<I> as CanonicalIso<ZnBase<I>>>::has_canonical_iso(self.base_ring().get_ring(), from)
    }

    fn map_out(&self, from: &ZnBase<I>, el: <AsFieldBase<R> as RingBase>::Element, iso: &Self::Isomorphism) -> <ZnBase<I> as RingBase>::Element {
        <ZnBase<I> as CanonicalIso<ZnBase<I>>>::map_out(self.base_ring().get_ring(), from, self.unwrap_element(el), iso)
    }
}

impl<R: ZnRingStore<Type = ZnBase<I>>, I: IntegerRingStore> CanonicalHom<AsFieldBase<R>> for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Homomorphism = <ZnBase<I> as CanonicalHom<ZnBase<I>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &AsFieldBase<R>) -> Option<Self::Homomorphism> {
        <ZnBase<I> as CanonicalHom<ZnBase<I>>>::has_canonical_hom(self, from.base_ring().get_ring())
    }

    fn map_in(&self, from: &AsFieldBase<R>, el: <AsFieldBase<R> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        <ZnBase<I> as CanonicalHom<ZnBase<I>>>::map_in(self, from.base_ring().get_ring(), from.unwrap_element(el), hom)
    }
}

impl<R: ZnRingStore<Type = ZnBase<I>>, I: IntegerRingStore> CanonicalIso<AsFieldBase<R>> for ZnBase<I>
    where I::Type: IntegerRing + CanonicalIso<StaticRingBase<i32>>
{
    type Isomorphism = <ZnBase<I> as CanonicalIso<ZnBase<I>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &AsFieldBase<R>) -> Option<Self::Isomorphism> {
        <ZnBase<I> as CanonicalIso<ZnBase<I>>>::has_canonical_iso(self, from.base_ring().get_ring())
    }

    fn map_out(&self, from: &AsFieldBase<R>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <AsFieldBase<R> as RingBase>::Element {
        from.from(<ZnBase<I> as CanonicalIso<ZnBase<I>>>::map_out(self, from.base_ring().get_ring(), el, iso))
    }
}

#[cfg(test)]
use crate::rings::bigint::*;
#[cfg(test)]
use crate::divisibility::generic_test_divisibility_axioms;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_mul() {
    const ZZ: RingValue<DefaultBigIntRing> = DefaultBigIntRing::RING;
    let Z257 = Zn::new(ZZ, ZZ.from_int(257));
    let x = Z257.coerce(&ZZ, ZZ.from_int(256));
    assert_el_eq!(&Z257, &Z257.one(), &Z257.mul_ref(&x, &x));
}

#[test]
fn test_project() {
    const ZZ: StaticRing<i64> = StaticRing::RING;
    let Z17 = Zn::new(ZZ, 17);
    for k in 0..289 {
        assert_el_eq!(&Z17, &Z17.from_int((289 - k) % 17), &Z17.coerce(&ZZ, -k as i64));
    }
}

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 10] = [0, 1, 3, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms_znbase() {
    let ring = Zn::new(StaticRing::<i64>::RING, 63);
    generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)))
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    let from = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::new(DefaultBigIntRing::RING, DefaultBigIntRing::RING.from_int(7 * 11));
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    generic_test_canonical_iso_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    assert!(from.can_hom(&Zn::new(StaticRing::<i64>::RING, 19)).is_none());
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i32>::RING;
    let to = Zn::new(StaticRing::<i128>::RING, 7 * 11);
    generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
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

#[test]
fn test_zn_map_in_small_int() {
    let ring = Zn::new(StaticRing::<i64>::RING, 257);
    assert_el_eq!(&ring, &ring.one(), &ring.coerce(&StaticRing::<i8>::RING, 1));
}

#[test]
fn test_divisibility_axioms() {
    let R = Zn::new(StaticRing::<i64>::RING, 17);
    generic_test_divisibility_axioms(&R, R.elements());
}

#[test]
fn test_canonical_iso_axioms_as_field() {
    let R = Zn::new(StaticRing::<i128>::RING, 17);
    let R2 = R.clone().as_field().ok().unwrap();
    generic_test_canonical_hom_axioms(&R, &R2, R.elements());
    generic_test_canonical_iso_axioms(&R, &R2, R.elements());
    generic_test_canonical_hom_axioms(&R2, &R, R2.elements());
    generic_test_canonical_iso_axioms(&R2, &R, R2.elements());
}