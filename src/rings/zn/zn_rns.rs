use std::alloc::Allocator;
use std::alloc::Global;

use algorithms::matmul::ComputeInnerProduct;

use crate::iters::multi_cartesian_product;
use crate::iters::MultiProduct;
use crate::seq::VectorView;
use crate::integer::IntegerRingStore;
use crate::divisibility::DivisibilityRingStore;
use crate::rings::zn::*;
use crate::primitive_int::*;

///
/// A ring representing `Z/nZ` for composite n by storing the
/// values modulo `m1, ..., mr` for `n = m1 * ... * mr`.
/// Generally, the advantage is improved performance in cases
/// where `m1`, ..., `mr` are sufficiently small, and can e.g.
/// by implemented without large integers.
/// 
/// Note that the component rings `Z/miZ` of this ring can be
/// accessed via the [`crate::seq::VectorView`]-functions.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::seq::*;
/// 
/// let R = Zn::create_from_primes(vec![17, 19], StaticRing::<i64>::RING);
/// let x = R.get_ring().from_congruence([R.get_ring().at(0).int_hom().map(1), R.get_ring().at(1).int_hom().map(16)].into_iter());
/// assert_eq!(35, R.smallest_lift(R.clone_el(&x)));
/// let y = R.mul_ref(&x, &x);
/// let z = R.get_ring().from_congruence([R.get_ring().at(0).int_hom().map(1 * 1), R.get_ring().at(1).int_hom().map(16 * 16)].into_iter());
/// assert!(R.eq_el(&z, &y));
/// ```
/// 
/// # Canonical mappings
/// This ring has a canonical isomorphism to Barett-reduction based Zn
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::create_from_primes(vec![17, 19], BigIntRing::RING);
/// let S = zn_big::Zn::new(StaticRing::<i64>::RING, 17 * 19);
/// assert!(R.eq_el(&R.int_hom().map(12), &R.coerce(&S, S.int_hom().map(12))));
/// assert!(S.eq_el(&S.int_hom().map(12), &R.can_iso(&S).unwrap().map(R.int_hom().map(12))));
/// ```
/// and a canonical homomorphism from any integer ring
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::create_from_primes(vec![3, 5, 7], BigIntRing::RING);
/// let S = BigIntRing::RING;
/// assert!(R.eq_el(&R.int_hom().map(120493), &R.coerce(&S, S.int_hom().map(120493))));
/// ```
/// 
pub struct ZnBase<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone = Global> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    components: Vec<C>,
    total_ring: zn_big::Zn<J>,
    unit_vectors: Vec<El<zn_big::Zn<J>>>,
    element_allocator: A
}

///
/// The ring `Z/nZ` for composite `n` implemented using the residue number system (RNS), 
/// i.e. storing values by storing their value modulo every factor of `n`.
/// For details, see [`ZnBase`].
/// 
pub type Zn<C, J, A = Global> = RingValue<ZnBase<C, J, A>>;

impl<C: ZnRingStore, J: IntegerRingStore> Zn<C, J, Global> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ///
    /// Creates a new ring for `Z/nZ` with `n = m1 ... mr` where the `mi` are the moduli
    /// of the given component rings. Furthermore, the corresponding large integer ring must be
    /// provided, which has to be able to store values of size at least `n^3`.
    /// 
    pub fn new(summands: Vec<C>, large_integers: J) -> Self {
        Self::new_with(summands, large_integers, Global)
    }
}

impl<J: IntegerRingStore> Zn<zn_64::Zn, J, Global> 
    where zn_64::ZnBase: CanHomFrom<J::Type>,
        J::Type: IntegerRing
{
    pub fn create_from_primes(primes: Vec<i64>, large_integers: J) -> Self {
        Self::new_with(primes.into_iter().map(|p| zn_64::Zn::new(p as u64)).collect(), large_integers, Global)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Zn<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ///
    /// Creates a new ring for `Z/nZ` with `n = m1 ... mr` where the `mi` are the moduli
    /// of the given component rings. Furthermore, the corresponding large integer ring must be
    /// provided, which has to be able to store values of size at least `n^3`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with(summands: Vec<C>, large_integers: J, element_allocator: A) -> Self {
        assert!(summands.len() > 0);
        let total_modulus = large_integers.prod(
            summands.iter().map(|R| R.integer_ring().can_iso(&large_integers).unwrap().map_ref(R.modulus()))
        );
        let total_ring = zn_big::Zn::new(large_integers, total_modulus);
        let ZZ = total_ring.integer_ring();
        for R in &summands {
            let R_modulus = R.integer_ring().can_iso(ZZ).unwrap().map_ref(R.modulus());
            assert!(
                ZZ.is_one(&algorithms::eea::signed_gcd(ZZ.checked_div(total_ring.modulus(), &R_modulus).unwrap(), R_modulus, ZZ)),
                "all moduli must be coprime"
            );
            // makes things much easier, e.g. during CanIsoFromTo implementation
            assert!(R.integer_ring().get_ring() == summands[0].integer_ring().get_ring());
        }
        let unit_vectors = summands.iter()
            .map(|R: &C| (R, ZZ.checked_div(total_ring.modulus(), &R.integer_ring().can_iso(ZZ).unwrap().map_ref(R.modulus())).unwrap()))
            .map(|(R, n)| (int_cast(R.any_lift(R.invert(&R.coerce(&ZZ, ZZ.clone_el(&n))).unwrap()), ZZ, R.integer_ring()), n))
            .map(|(n_mod_inv, n)| total_ring.mul(total_ring.coerce(&ZZ, n_mod_inv), total_ring.coerce(&ZZ, n)))
            .collect();
        RingValue::from(ZnBase {
            components: summands,
            total_ring: total_ring,
            unit_vectors: unit_vectors,
            element_allocator: element_allocator
        })
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Zn<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ///
    /// Given values `ai` for each component ring `Z/miZ`, computes the unique element in this
    /// ring `Z/nZ` that is congruent to `ai` modulo `mi`. The "opposite" function is [`Zn::get_congruence()`].
    /// 
    pub fn from_congruence<I>(&self, el: I) -> ZnEl<C, A>
        where I: IntoIterator<Item = El<C>>
    {
        self.get_ring().from_congruence(el)
    }

    ///
    /// Given `a` in `Z/nZ`, returns the vector whose `i`-th entry is `a mod mi`, where the `mi` are the
    /// moduli of the component rings of this ring.
    /// 
    pub fn get_congruence<'a>(&self, el: &'a ZnEl<C, A>) -> impl 'a + VectorView<El<C>> {
        self.get_ring().get_congruence(el)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ///
    /// Given values `ai` for each component ring `Z/miZ`, computes the unique element in this
    /// ring `Z/nZ` that is congruent to `ai` modulo `mi`. The "opposite" function is [`ZnBase::get_congruence()`].
    /// 
    pub fn from_congruence<I>(&self, el: I) -> ZnEl<C, A>
        where I: IntoIterator<Item = El<C>>
    {
        let mut data = Vec::with_capacity_in(self.len(), self.element_allocator.clone());
        data.extend(el);
        assert_eq!(self.len(), data.len());
        ZnEl { data }
    }

    ///
    /// Given `a` in `Z/nZ`, returns the vector whose `i`-th entry is `a mod mi`, where the `mi` are the
    /// moduli of the component rings of this ring.
    /// 
    pub fn get_congruence<'a>(&self, el: &'a ZnEl<C, A>) -> impl 'a + VectorView<El<C>> {
        &el.data as &[El<C>]
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> VectorView<C> for Zn<C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn len(&self) -> usize {
        self.get_ring().len()
    }

    fn at(&self, index: usize) -> &C {
        &self.get_ring().at(index)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> VectorView<C> for ZnBase<C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn len(&self) -> usize {
        self.components.len()
    }

    fn at(&self, index: usize) -> &C {
        &self.components[index]
    }
}

pub struct ZnEl<C: ZnRingStore, A: Allocator + Clone>
    where C::Type: ZnRing
{
    data: Vec<El<C>, A>
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> RingBase for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type Element = ZnEl<C, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(self.len(), self.element_allocator.clone());
        data.extend((0..self.len()).map(|i| self.at(i).clone_el(val.data.at(i))));
        ZnEl { data }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].add_assign_ref(&mut lhs.data[i], &rhs.data[i])
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (i, el) in (0..self.components.len()).zip(rhs.data.into_iter()) {
            self.components[i].add_assign_ref(&mut lhs.data[i], &el)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].sub_assign_ref(&mut lhs.data[i], &rhs.data[i])
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].negate_inplace(&mut lhs.data[i])
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (i, el) in (0..self.components.len()).zip(rhs.data.into_iter()) {
            self.components[i].mul_assign_ref(&mut lhs.data[i], &el)
        }
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].mul_assign_ref(&mut lhs.data[i], &rhs.data[i])
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from_congruence((0..self.len()).map(|i| self.components[i].get_ring().from_int(value)))
    }
    
    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..self.components.len() {
            self.components[i].int_hom().mul_assign_map(&mut lhs.data[i], rhs)
        }

    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        (0..self.components.len()).zip(lhs.data.iter()).zip(rhs.data.iter()).all(|((i, l), r)| self.components[i].eq_el(l, r))
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.data.iter()).all(|(i, x)| self.components[i].is_zero(x))
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.data.iter()).all(|(i, x)| self.components[i].is_one(x))
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.data.iter()).all(|(i, x)| self.components[i].is_neg_one(x))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.total_ring.get_ring().dbg(&RingRef::new(self).can_iso(&self.total_ring).unwrap().map_ref(value), out)
    }
    
    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.size(ZZ)
    }
    
    fn is_approximate(&self) -> bool { false }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Clone for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>,
        C: Clone,
        J: Clone
{
    fn clone(&self) -> Self {
        ZnBase {
            components: self.components.clone(),
            total_ring: self.total_ring.clone(),
            unit_vectors: self.unit_vectors.iter().map(|e| self.total_ring.clone_el(e)).collect(),
            element_allocator: self.element_allocator.clone()
        }
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore, A1: Allocator + Clone, A2: Allocator + Clone> CanHomFrom<ZnBase<C2, J2, A2>> for ZnBase<C1, J1, A1> 
    where C1::Type: ZnRing + CanHomFrom<C2::Type> + CanHomFrom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J1::Type>,
        C2::Type: ZnRing + CanHomFrom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J2::Type>,
        J1::Type: IntegerRing,
        J2::Type: IntegerRing
{
    type Homomorphism = Vec<<C1::Type as CanHomFrom<C2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &ZnBase<C2, J2, A2>) -> Option<Self::Homomorphism> {
        if self.components.len() == from.components.len() {
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f): (&C1, &C2)| s.get_ring().has_canonical_hom(f.get_ring()).ok_or(()))
                .collect::<Result<Self::Homomorphism, ()>>()
                .ok()
        } else {
            None
        }
    }

    fn map_in_ref(&self, from: &ZnBase<C2, J2, A2>, el: &ZnEl<C2, A2>, hom: &Self::Homomorphism) -> Self::Element {
        assert_eq!(from.len(), el.data.len());
        self.from_congruence((0..self.len()).map(|i| 
            self.at(i).get_ring().map_in_ref(from.at(i).get_ring(), el.data.at(i), &hom[i])
        ))
    }

    fn map_in(&self, from: &ZnBase<C2, J2, A2>, el: ZnEl<C2, A2>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> PartialEq for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>,
        J::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.components.len() == other.components.len() && self.components.iter().zip(other.components.iter()).all(|(R1, R2)| R1.get_ring() == R2.get_ring())
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore, A1: Allocator + Clone, A2: Allocator + Clone> CanIsoFromTo<ZnBase<C2, J2, A2>> for ZnBase<C1, J1, A1> 
    where C1::Type: ZnRing + CanIsoFromTo<C2::Type> + CanHomFrom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J1::Type>,
        C2::Type: ZnRing + CanHomFrom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J2::Type>,
        J1::Type: IntegerRing,
        J2::Type: IntegerRing
{
    type Isomorphism = Vec<<C1::Type as CanIsoFromTo<C2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &ZnBase<C2, J2, A2>) -> Option<Self::Isomorphism> {
        if self.components.len() == from.components.len() {
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f): (&C1, &C2)| s.get_ring().has_canonical_iso(f.get_ring()).ok_or(()))
                .collect::<Result<Self::Isomorphism, ()>>()
                .ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &ZnBase<C2, J2, A2>, el: ZnEl<C1, A1>, iso: &Self::Isomorphism) -> ZnEl<C2, A2> {
        assert_eq!(self.len(), el.data.len());
        from.from_congruence((0..from.len()).map(|i|
            self.at(i).get_ring().map_out(from.at(i).get_ring(), self.at(i).clone_el(el.data.at(i)), &iso[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore, A: Allocator + Clone> CanHomFrom<zn_big::ZnBase<K>> for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing + CanIsoFromTo<K::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>,
        K::Type: IntegerRing
{
    type Homomorphism = (<J::Type as CanHomFrom<K::Type>>::Homomorphism, Vec<<C::Type as CanHomFrom<J::Type>>::Homomorphism>);

    fn has_canonical_hom(&self, from: &zn_big::ZnBase<K>) -> Option<Self::Homomorphism> {
        if self.total_ring.get_ring().has_canonical_hom(from).is_some() {
            Some((
                self.total_ring.get_ring().has_canonical_hom(from)?,
                self.components.iter()
                    .map(|s| s.get_ring())
                    .map(|s| s.has_canonical_hom(self.integer_ring().get_ring()).ok_or(()))
                    .collect::<Result<Vec<_>, ()>>()
                    .ok()?
            ))
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_big::ZnBase<K>, el: zn_big::ZnEl<K>, hom: &Self::Homomorphism) -> ZnEl<C, A> {
        let lift = from.smallest_positive_lift(el);
        let mapped_lift = <J::Type as CanHomFrom<K::Type>>::map_in(
            self.integer_ring().get_ring(), 
            from.integer_ring().get_ring(), 
            lift, 
            &hom.0
        );
        self.from_congruence((0..self.len()).map(|i|
            self.at(i).get_ring().map_in_ref(self.integer_ring().get_ring(), &mapped_lift, &hom.1[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore, A: Allocator + Clone> CanIsoFromTo<zn_big::ZnBase<K>> for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing + CanIsoFromTo<K::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>,
        K::Type: IntegerRing
{
    // we first map each `lift(x[i]) into `self.total_ring.integer_ring(): J`, then reduce it to 
    // `self.total_ring: Zn<J>`, then compute the value `sum_i lift(x[i]) * unit_vectors[i]` 
    // in `self.total_ring: Zn<J>` and then map this to `from: Zn<K>`.
    type Isomorphism = (
        <zn_big::ZnBase<J> as CanIsoFromTo<zn_big::ZnBase<K>>>::Isomorphism, 
        <zn_big::ZnBase<J> as CanHomFrom<J::Type>>::Homomorphism
    );

    fn has_canonical_iso(&self, from: &zn_big::ZnBase<K>) -> Option<Self::Isomorphism> {
        Some((
            <zn_big::ZnBase<J> as CanIsoFromTo<zn_big::ZnBase<K>>>::has_canonical_iso(self.total_ring.get_ring(), from)?,
            self.total_ring.get_ring().has_canonical_hom(self.total_ring.integer_ring().get_ring())?,
        ))
    }

    fn map_out(&self, from: &zn_big::ZnBase<K>, el: Self::Element, (final_iso, red): &Self::Isomorphism) -> zn_big::ZnEl<K> {
        assert_eq!(self.len(), el.data.len());
        let small_integer_ring = self.at(0).integer_ring();
        let result = <_ as ComputeInnerProduct>::inner_product_ref_fst(self.total_ring.get_ring(),
            self.components.iter()
                .zip(el.data.into_iter())
                .map(|(R, x): (&C, El<C>)| R.smallest_positive_lift(x))
                .zip(self.unit_vectors.iter())
                .map(|(x, u)| 
                    (
                        u,
                        self.total_ring.get_ring().map_in(
                            self.total_ring.integer_ring().get_ring(),
                            int_cast(x, self.total_ring.integer_ring(), small_integer_ring),
                            red
                        )
                    )
                )
        );
        return <zn_big::ZnBase<J> as CanIsoFromTo<zn_big::ZnBase<K>>>::map_out(self.total_ring.get_ring(), from, result, final_iso);
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> CanHomFrom<zn_64::ZnBase> for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type Homomorphism = (<Self as CanHomFrom<zn_big::ZnBase<J>>>::Homomorphism, <zn_big::ZnBase<J> as CanHomFrom<zn_64::ZnBase>>::Homomorphism);

    fn has_canonical_hom(&self, from: &zn_64::ZnBase) -> Option<Self::Homomorphism> {
        Some((self.has_canonical_hom(self.total_ring.get_ring())?, self.total_ring.get_ring().has_canonical_hom(from)?))
    }
    
    fn map_in(&self, from: &zn_64::ZnBase, el: zn_64::ZnEl, hom: &Self::Homomorphism) -> ZnEl<C, A> {
        self.map_in(self.total_ring.get_ring(), self.total_ring.get_ring().map_in(from, el, &hom.1), &hom.0)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRing, A: Allocator + Clone> CanHomFrom<K> for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type> + CanHomFrom<K>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>,
        K: ?Sized
{
    type Homomorphism = Vec<<C::Type as CanHomFrom<K>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &K) -> Option<Self::Homomorphism> {
        Some(self.components.iter()
            .map(|R| <C::Type as CanHomFrom<K>>::has_canonical_hom(R.get_ring(), from).ok_or(()))
            .collect::<Result<Vec<<C::Type as CanHomFrom<K>>::Homomorphism>, ()>>()
            .ok()?
        )
    }

    fn map_in(&self, from: &K, el: K::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from_congruence((0..self.len()).map(|i|
            <C::Type as CanHomFrom<K>>::map_in_ref(self.at(i).get_ring(), from, &el, &hom[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> DivisibilityRing for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut data = Vec::with_capacity_in(self.len(), self.element_allocator.clone());
        for i in 0..self.len() {
            data.push(self.at(i).checked_div(lhs.data.at(i), rhs.data.at(i))?);
        }
        return Some(ZnEl { data });
    }
}

pub struct FromCongruenceElementCreator<'a, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ring: &'a ZnBase<C, J, A>
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Clone for FromCongruenceElementCreator<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Copy for FromCongruenceElementCreator<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> FnOnce<(&'b [El<C>],)> for FromCongruenceElementCreator<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type Output = <ZnBase<C, J, A> as RingBase>::Element;

    extern "rust-call" fn call_once(mut self, args: (&'b [El<C>],)) -> Self::Output {
        self.call_mut(args)
    }
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> FnMut<(&'b [El<C>],)> for FromCongruenceElementCreator<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<C>],)) -> Self::Output {
        self.ring.from_congruence(args.0.into_iter().enumerate().map(|(i, x)| self.ring.at(i).clone_el(x)))
    }
}

pub struct CloneComponentElement<'a, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    ring: &'a ZnBase<C, J, A>
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Clone for CloneComponentElement<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Copy for CloneComponentElement<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> FnOnce<(usize, &'b El<C>)> for CloneComponentElement<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type Output = El<C>;

    extern "rust-call" fn call_once(mut self, args: (usize, &'b El<C>)) -> Self::Output {
        self.call_mut(args)
    }
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> FnMut<(usize, &'b El<C>)> for CloneComponentElement<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    extern "rust-call" fn call_mut(&mut self, args: (usize, &'b El<C>)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> Fn<(usize, &'b El<C>)> for CloneComponentElement<'a, C, J, A>
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    extern "rust-call" fn call(&self, args: (usize, &'b El<C>)) -> Self::Output {
        self.ring.at(args.0).clone_el(args.1)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> HashableElRing for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type> + HashableElRing,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        for (i, el) in (0..self.components.len()).zip(el.data.iter()) {
            self.components[i].hash(el, h);
        }
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> FiniteRing for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type ElementsIter<'a> = MultiProduct<<C::Type as FiniteRing>::ElementsIter<'a>, FromCongruenceElementCreator<'a, C, J, A>, CloneComponentElement<'a, C, J, A>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.len()).map(|i| self.at(i).elements()), FromCongruenceElementCreator { ring: self }, CloneComponentElement { ring: self })
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> ZnEl<C, A> {
        self.from_congruence((0..self.len()).map(|i| self.at(i).random_element(&mut rng)))
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        if ZZ.get_ring().representable_bits().is_none() || self.integer_ring().abs_log2_ceil(self.modulus()) < ZZ.get_ring().representable_bits() {
            Some(int_cast(self.integer_ring().clone_el(self.modulus()), ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> PrincipalIdealRing for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut data = Vec::with_capacity_in(self.len(), self.element_allocator.clone());
        for i in 0..self.len() {
            data.push(self.at(i).checked_div_min(lhs.data.at(i), rhs.data.at(i))?);
        }
        return Some(ZnEl { data });
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let mut result = (self.zero(), self.zero(), self.zero());
        for (i, Zn) in self.as_iter().enumerate() {
            (result.0.data[i], result.1.data[i], result.2.data[i]) = Zn.extended_ideal_gen(&lhs.data[i], &rhs.data[i]);
        }
        return result;
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, A: Allocator + Clone> ZnRing for ZnBase<C, J, A> 
    where C::Type: ZnRing + CanHomFrom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanIsoFromTo<J::Type>
{
    type IntegerRingBase = J::Type;
    type IntegerRing = J;

    fn integer_ring(&self) -> &Self::IntegerRing {
        self.total_ring.integer_ring()
    }

    fn modulus(&self) -> &El<Self::IntegerRing> {
        self.total_ring.modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.total_ring.smallest_positive_lift(
            <Self as CanIsoFromTo<zn_big::ZnBase<J>>>::map_out(
                self, 
                self.total_ring.get_ring(), 
                el, 
                &<Self as CanIsoFromTo<zn_big::ZnBase<J>>>::has_canonical_iso(self, self.total_ring.get_ring()).unwrap()
            )
        )
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.total_ring.smallest_lift(
            <Self as CanIsoFromTo<zn_big::ZnBase<J>>>::map_out(
                self, 
                self.total_ring.get_ring(), 
                el, 
                &<Self as CanIsoFromTo<zn_big::ZnBase<J>>>::has_canonical_iso(self, self.total_ring.get_ring()).unwrap()
            )
        )
    }

    fn is_field(&self) -> bool {
        self.components.len() == 1 && self.components[0].is_field()
    }

    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        debug_assert!(!self.integer_ring().is_neg(&x));
        debug_assert!(self.integer_ring().is_lt(&x, self.modulus()));
        RingRef::new(self).can_hom(self.integer_ring()).unwrap().map(x)
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 9] = [0, 1, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms() {
    let ring = Zn::create_from_primes(vec![7, 11], StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.int_hom().map(x)))
}

#[test]
fn test_hash_axioms() {
    let ring = Zn::create_from_primes(vec![7, 11], StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hash_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.int_hom().map(x)))
}

#[test]
fn test_map_in_map_out() {
    let ring1 = Zn::create_from_primes(vec![7, 11, 17], StaticRing::<i64>::RING);
    let ring2 = zn_big::Zn::new(StaticRing::<i64>::RING, 7 * 11 * 17);
    for x in [0, 1, 7, 8, 9, 10, 11, 17, 7 * 17, 11 * 8, 11 * 17, 7 * 11 * 17 - 1] {
        let value = ring2.int_hom().map(x);
        assert!(ring2.eq_el(&value, &ring1.can_iso(&ring2).unwrap().map(ring1.coerce(&ring2, value.clone()))));
    }
}

#[test]
fn test_canonical_iso_axioms_zn_big() {
    let from = zn_big::Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::create_from_primes(vec![7, 11], StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.int_hom().map(x)));
    crate::ring::generic_tests::test_iso_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.int_hom().map(x)));

    let from = zn_big::Zn::new(StaticRing::<i128>::RING, 7 * 11 * 65537);
    let to = Zn::create_from_primes(vec![7, 11, 65537], StaticRing::<i128>::RING);
    crate::ring::generic_tests::test_hom_axioms(&from, &to, from.elements().step_by(65536));
    crate::ring::generic_tests::test_iso_axioms(&from, &to, from.elements().step_by(65536));
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i32>::RING;
    let to = Zn::create_from_primes(vec![7, 11], StaticRing::<i64>::RING);
    crate::ring::generic_tests::test_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.int_hom().map(x)));
}

#[test]
fn test_zn_ring_axioms() {
    let ring = Zn::create_from_primes(vec![7, 11], StaticRing::<i64>::RING);
    super::generic_tests::test_zn_axioms(ring);
}

#[test]
fn test_zn_map_in_large_int() {
    let ring = Zn::create_from_primes(vec![7, 11], BigIntRing::RING);
    super::generic_tests::test_map_in_large_int(ring);

    let R = Zn::create_from_primes(vec![3, 5, 7], BigIntRing::RING);
    let S = BigIntRing::RING;
    assert!(R.eq_el(&R.int_hom().map(120493), &R.coerce(&S, S.int_hom().map(120493))));
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let R = Zn::create_from_primes(vec![5], BigIntRing::RING);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
    
    let R = Zn::create_from_primes(vec![3, 5], BigIntRing::RING);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());
    
    let R = Zn::create_from_primes(vec![2, 3, 5], BigIntRing::RING);
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&R, R.elements());

    let R = Zn::create_from_primes(vec![3, 5, 2], BigIntRing::RING);
    let modulo = R.int_hom();
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(
        &R,
        [-1, 0, 1, 3, 2, 4, 5, 9, 18, 15, 30].into_iter().map(|x| modulo.map(x))
    );
}

#[test]
fn test_finite_ring_axioms() {
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::create_from_primes(vec![3, 5, 7, 11], StaticRing::<i64>::RING));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::create_from_primes(vec![3, 5], StaticRing::<i64>::RING));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::create_from_primes(vec![3], StaticRing::<i64>::RING));
    crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::create_from_primes(vec![2], StaticRing::<i64>::RING));
}

#[test]
fn test_not_prime() {
    let ring = Zn::new(vec![zn_64::Zn::new(15), zn_64::Zn::new(7)], StaticRing::<i64>::RING);
    let equivalent_ring = zn_big::Zn::new(StaticRing::<i64>::RING, 15 * 7);
    crate::ring::generic_tests::test_ring_axioms(&ring, ring.elements());
    crate::divisibility::generic_tests::test_divisibility_axioms(&ring, ring.elements());
    crate::homomorphism::generic_tests::test_homomorphism_axioms(ring.can_hom(&equivalent_ring).unwrap(), equivalent_ring.elements());
    crate::homomorphism::generic_tests::test_homomorphism_axioms(ring.can_iso(&equivalent_ring).unwrap(), ring.elements());
}

#[test]
#[should_panic]
fn test_not_coprime() {
    Zn::new(vec![zn_64::Zn::new(15), zn_64::Zn::new(35)], StaticRing::<i64>::RING);
}

#[test]
fn test_format() {
    let ring = Zn::new([72057594035352641, 72057594035418113, 72057594036334721, 72057594036945793, ].iter().map(|p| zn_64::Zn::new(*p)).collect(), BigIntRing::RING);
    assert_eq!("1", format!("{}", ring.format(&ring.int_hom().map(1))));
}