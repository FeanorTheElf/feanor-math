use std::array::{from_fn, try_from_fn};
use std::marker::PhantomData;

use crate::divisibility::*;
use crate::serialization::*;
use crate::specialization::{FiniteRingSpecializable, FiniteRingOperation};
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyButterfly;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::rings::finite::{FiniteRing, FiniteRingStore};
use crate::ring::*;
use crate::iters::*;
use crate::rings::zn::zn_64::*;
use crate::homomorphism::*;
use crate::seq::CloneRingEl;

use feanor_serde::newtype_struct::{DeserializeSeedNewtypeStruct, SerializableNewtypeStruct};
use feanor_serde::seq::{DeserializeSeedSeq, SerializableSeq};
use serde::{Serializer, Deserializer, Serialize};
use serde::de::DeserializeSeed;

///
/// The `N`-fold direct product ring `R x ... x R`.
/// 
/// Currently, this is a quite naive implementation, which just
/// repeats operations along each component. In the future, this
/// might become an entrypoint for vectorization or similar. Hence,
/// it might remain unstable for a while.
/// 
#[stability::unstable(feature = "enable")]
pub struct DirectPowerRingBase<R: RingStore, const N: usize> {
    base: R
}

#[stability::unstable(feature = "enable")]
pub type DirectPowerRing<R, const N: usize> = RingValue<DirectPowerRingBase<R, N>>;

impl<R: RingStore, const N: usize> DirectPowerRing<R, N> {

    #[stability::unstable(feature = "enable")]
    pub fn new(base: R) -> Self {
        Self::from(DirectPowerRingBase { base })
    }
}

#[stability::unstable(feature = "enable")]
pub struct DirectPowerRingElCreator<'a, R: RingStore, const N: usize> {
    ring: &'a DirectPowerRingBase<R, N>
}

impl<R: RingStore, const N: usize> Clone for DirectPowerRingBase<R, N>
    where R: Clone
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<R: RingStore, const N: usize> Copy for DirectPowerRingBase<R, N>
    where R: Copy, El<R>: Copy
{}

impl<R: RingStore, const N: usize> PartialEq for DirectPowerRingBase<R, N> {
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<R: RingStore, const N: usize> RingBase for DirectPowerRingBase<R, N> {

    type Element = [El<R>; N];
    
    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        from_fn(|i| self.base.clone_el(&val[i]))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.add_assign_ref(tgt, src)
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.add_assign(tgt, src)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.sub_assign_ref(tgt, src)
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for val in lhs.into_iter() {
            self.base.negate_inplace(val);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.mul_assign(tgt, src)
        }
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.mul_assign_ref(tgt, src)
        }
    }

    fn zero(&self) -> Self::Element {
        from_fn(|_| self.base.zero())
    }

    fn one(&self) -> Self::Element { 
        from_fn(|_| self.base.one())
    }

    fn neg_one(&self) -> Self::Element {
        from_fn(|_| self.base.neg_one())
    }

    fn from_int(&self, value: i32) -> Self::Element {
        let val = self.base.get_ring().from_int(value);
        from_fn(|_| self.base.clone_el(&val))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        lhs.into_iter().zip(rhs.into_iter()).all(|(l, r)| self.base.eq_el(l, r))
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.into_iter().all(|v| self.base.is_zero(v))
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.into_iter().all(|v| self.base.is_one(v))
    }
    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.into_iter().all(|v| self.base.is_neg_one(v))
    }

    fn is_commutative(&self) -> bool {
        self.base.is_commutative()
    }

    fn is_noetherian(&self) -> bool {
        self.base.is_noetherian()
    }

    fn is_approximate(&self) -> bool {
        self.base.get_ring().is_approximate()
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        write!(out, "(")?;
        for i in 0..N {
            self.base.get_ring().dbg_within(&value[i], out, EnvBindingStrength::Weakest)?;
            if i + 1 != N {
                write!(out, ", ")?;
            }
        }
        write!(out, ")")?;
        return Ok(());
    }

    fn square(&self, value: &mut Self::Element) {
        for val in value.into_iter() {
            self.base.square(val)
        }
    }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.sub_assign(tgt, src)
        }
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for tgt in lhs.into_iter() {
            self.base.get_ring().mul_assign_int(tgt, rhs)
        }
    }

    fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.sub_self_assign(tgt, src)
        }
    }

    fn sub_self_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for (tgt, src) in lhs.into_iter().zip(rhs.into_iter()) {
            self.base.sub_self_assign_ref(tgt, src)
        }
    }

    fn sum<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        els.into_iter().fold(self.zero(), |a, b| self.add(a, b))
    }

    fn prod<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        els.into_iter().fold(self.one(), |a, b| self.mul(a, b))
    }
    
    fn characteristic<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base.get_ring().characteristic(ZZ)
    }
}

impl<R: RingStore, const N: usize> RingExtension for DirectPowerRingBase<R, N> {

    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.from_ref(&x)
    }

    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        from_fn(|_| self.base.clone_el(x))
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        for tgt in lhs.into_iter() {
            self.base.mul_assign_ref(tgt, rhs);
        }
    }

    fn mul_assign_base_through_hom<S: ?Sized + RingBase, H: Homomorphism<S, R::Type>>(&self, lhs: &mut Self::Element, rhs: &S::Element, hom: H) {
        for tgt in lhs.into_iter() {
            hom.mul_assign_ref_map(tgt, rhs);
        }
    }
}

impl<S: RingStore, R: RingStore, const N: usize> CanHomFrom<DirectPowerRingBase<S, N>> for DirectPowerRingBase<R, N>
    where R::Type: CanHomFrom<S::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DirectPowerRingBase<S, N>) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from.base.get_ring())
    }

    fn map_in(&self, from: &DirectPowerRingBase<S, N>, el: <DirectPowerRingBase<S, N> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut el_it = el.into_iter();
        from_fn(|_| self.base.get_ring().map_in(from.base.get_ring(), el_it.next().unwrap(), hom))
    }

    fn map_in_ref(&self, from: &DirectPowerRingBase<S, N>, el: &<DirectPowerRingBase<S, N> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        from_fn(|i| self.base.get_ring().map_in_ref(from.base.get_ring(), &el[i], hom))
    }
}

impl<S: RingStore, R: RingStore, const N: usize> CanIsoFromTo<DirectPowerRingBase<S, N>> for DirectPowerRingBase<R, N>
    where R::Type: CanIsoFromTo<S::Type>
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DirectPowerRingBase<S, N>) -> Option<Self::Isomorphism> {
        self.base.get_ring().has_canonical_iso(from.base.get_ring())
    }

    fn map_out(&self, from: &DirectPowerRingBase<S, N>, el: Self::Element, iso: &Self::Isomorphism) -> <DirectPowerRingBase<S, N> as RingBase>::Element {
        let mut el_it = el.into_iter();
        from_fn(|_| self.base.get_ring().map_out(from.base.get_ring(), el_it.next().unwrap(), iso))
    }
}

impl<R: RingStore, const N: usize> DivisibilityRing for DirectPowerRingBase<R, N>
    where R::Type: DivisibilityRing
{
    type PreparedDivisorData = [<R::Type as DivisibilityRing>::PreparedDivisorData; N];

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        try_from_fn(|i| self.base.checked_left_div(&lhs[i], &rhs[i]))
    }

    fn prepare_divisor(&self, el: &Self::Element) -> Self::PreparedDivisorData {
        from_fn(|i| self.base.get_ring().prepare_divisor(&el[i]))
    }

    fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> Option<Self::Element> {
        try_from_fn(|i| self.base.get_ring().checked_left_div_prepared(&lhs[i], &rhs[i], &rhs_prep[i]))
    }
    
    fn divides_left_prepared(&self, lhs: &Self::Element, rhs: &Self::Element, rhs_prep: &Self::PreparedDivisorData) -> bool {
        (0..N).all(|i| self.base.get_ring().divides_left_prepared(&lhs[i], &rhs[i], &rhs_prep[i]))
    }
}

impl<R: RingStore, const N: usize> FiniteRingSpecializable for DirectPowerRingBase<R, N>
    where R::Type: FiniteRingSpecializable
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        struct BaseRingCase<R: RingStore, O: FiniteRingOperation<DirectPowerRingBase<R, N>>, const N: usize> {
            op: O,
            ring: PhantomData<R>
        }
        impl<R: RingStore, O: FiniteRingOperation<DirectPowerRingBase<R, N>>, const N: usize> FiniteRingOperation<R::Type> for BaseRingCase<R, O, N> {
            type Output = O::Output;
            fn execute(self) -> Self::Output where R::Type: FiniteRing {
                self.op.execute()
            }
            fn fallback(self) -> Self::Output {
                self.op.fallback()
            }
        }
        <R::Type as FiniteRingSpecializable>::specialize(BaseRingCase {
            op: op,
            ring: PhantomData
        })
    }
}

impl<'a, 'b, R: RingStore, const N: usize> Copy for DirectPowerRingElCreator<'a, R, N> {}

impl<'a, 'b, R: RingStore, const N: usize> Clone for DirectPowerRingElCreator<'a, R, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, R: RingStore, const N: usize> FnOnce<(&'b [El<R>],)> for DirectPowerRingElCreator<'a, R, N> {

    type Output = [El<R>; N];

    extern "rust-call" fn call_once(self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R: RingStore, const N: usize> FnMut<(&'b [El<R>],)> for DirectPowerRingElCreator<'a, R, N> {

    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R: RingStore, const N: usize> Fn<(&'b [El<R>],)> for DirectPowerRingElCreator<'a, R, N> {

    extern "rust-call" fn call(&self, args: (&'b [El<R>],)) -> Self::Output {
        assert_eq!(N, args.0.len());
        from_fn(|i| self.ring.base.clone_el(&args.0[i]))
    }
}

impl<R: RingStore, const N: usize> FiniteRing for DirectPowerRingBase<R, N>
    where R::Type: FiniteRing
{
    type ElementsIter<'a> = MultiProduct<<R::Type as FiniteRing>::ElementsIter<'a>, DirectPowerRingElCreator<'a, R, N>, CloneRingEl<&'a R>, [El<R>; N]>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..N).map(|_| self.base.elements()), DirectPowerRingElCreator { ring: self }, CloneRingEl(&self.base))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> <Self as RingBase>::Element {
        from_fn(|_| self.base.random_element(&mut rng))
    }

    fn size<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let base_size = self.base.size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= ZZ.abs_log2_ceil(&base_size).unwrap_or(0) * N {
            return Some(ZZ.pow(base_size, N));
        } else {
            return None;
        }
    }
}

macro_rules! specialize_butterfly {
    ($($num:literal),*) => { $(
                
        impl CooleyTuckeyButterfly<ZnFastmulBase> for DirectPowerRingBase<Zn, $num> {

            #[inline(always)]
            fn butterfly_new<H: Homomorphism<ZnFastmulBase, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &ZnFastmulEl) {
                for (x, y) in x.into_iter().zip(y.into_iter()) {
                    // the only homomorphism this can be is the `CanHom` composed with an `Inclusion`
                    <ZnBase as CooleyTuckeyButterfly<ZnFastmulBase>>::butterfly_new(CanHom::from_raw_parts(hom.domain(), hom.codomain().base_ring(), ()), x, y, twiddle);
                }
            }

            #[inline(always)]
            fn inv_butterfly_new<H: Homomorphism<ZnFastmulBase, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &ZnFastmulEl) {
                for (x, y) in x.into_iter().zip(y.into_iter()) {
                    // the only homomorphism this can be is the `CanHom` composed with an `Inclusion`
                    <ZnBase as CooleyTuckeyButterfly<ZnFastmulBase>>::inv_butterfly_new(CanHom::from_raw_parts(hom.domain(), hom.codomain().base_ring(), ()), x, y, twiddle);
                }
            }
            
            #[inline(always)]
            fn prepare_for_fft(&self, value: &mut [ZnEl; $num]) {
                for x in value.into_iter() {
                    <ZnBase as CooleyTuckeyButterfly<ZnFastmulBase>>::prepare_for_fft(self.base_ring().get_ring(), x)
                }
            }
            
            #[inline(always)]
            fn prepare_for_inv_fft(&self, value: &mut [ZnEl; $num]) {
                for x in value.into_iter() {
                    <ZnBase as CooleyTuckeyButterfly<ZnFastmulBase>>::prepare_for_inv_fft(self.base_ring().get_ring(), x)
                }
            }
        }
    )* }
}
specialize_butterfly!{ 1, 2, 3, 4, 5, 6, 7, 8, 16 }

impl<R: RingStore, const N: usize> HashableElRing for DirectPowerRingBase<R, N>
    where R::Type: HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        for val in el.into_iter() {
            self.base.get_ring().hash(val, h);
        }
    }
}

impl<R: RingStore, const N: usize> SerializableElementRing for DirectPowerRingBase<R, N>
    where R::Type: SerializableElementRing
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("DirectPowerRingEl", DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeWithRing::new(self.base_ring())).take(N + 1),
            (from_fn(|_| None), 0),
            |(mut current, mut current_idx), next| {
                current[current_idx] = Some(next);
                current_idx += 1;
                (current, current_idx)
            }
        )).deserialize(deserializer).map(|(res, len): ([Option<_>; N], usize)| {
            assert_eq!(N, len);
            let mut res_it = res.into_iter();
            from_fn(|_| res_it.next().unwrap().unwrap())
        })
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("DirectPowerRingEl", SerializableSeq::new_with_len(
            (0..N).map(|i| SerializeWithRing::new(&el[i], self.base_ring())), N
        )).serialize(serializer)
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring: DirectPowerRing<_, 3> = DirectPowerRing::new(Zn64B::new(3));
    crate::ring::generic_tests::test_ring_axioms(&ring, ring.elements());
}

#[test]
fn test_can_hom_axioms() {
    LogAlgorithmSubscriber::init_test();
    let from: DirectPowerRing<_, 3> = DirectPowerRing::new(StaticRing::<i64>::RING);
    let ring: DirectPowerRing<_, 3> = DirectPowerRing::new(Zn64B::new(3));
    crate::ring::generic_tests::test_hom_axioms(&from, &ring, multi_cartesian_product((0..3).map(|_| -4..5), clone_array::<_, 3>, |_, x| *x));
}

#[test]
fn test_hash_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring: DirectPowerRing<_, 3> = DirectPowerRing::new(Zn64B::new(3));
    crate::ring::generic_tests::test_hash_axioms(&ring, ring.elements());
}

#[test]
fn test_serialization() {
    LogAlgorithmSubscriber::init_test();
    let ring: DirectPowerRing<_, 3> = DirectPowerRing::new(Zn64B::new(3));
    crate::serialization::generic_tests::test_serialization(&ring, ring.elements());
}