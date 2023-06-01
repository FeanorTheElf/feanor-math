use crate::mempool::{MemoryProvider, AllocatingMemoryProvider};
use crate::vector::VectorView;
use crate::{integer::IntegerRingStore, divisibility::DivisibilityRingStore};
use crate::rings::zn::*;

///
/// A ring representing `Z/nZ` for composite n by storing the
/// values modulo `m1, ..., mr` for `n = m1 * ... * mr`.
/// Generally, the advantage is improved performance in cases
/// where `m1`, ..., `mr` are sufficiently small, and can e.g.
/// by implemented without large integers.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::vector::*;
/// 
/// let R = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![17, 19]);
/// let x = R.get_ring().from_congruence([R.get_ring().at(0).from_int(1), R.get_ring().at(1).from_int(16)]);
/// assert_eq!(35, R.smallest_lift(R.clone_el(&x)));
/// let y = R.mul_ref(&x, &x);
/// let z = R.get_ring().from_congruence([R.get_ring().at(0).from_int(1 * 1), R.get_ring().at(1).from_int(16 * 16)]);
/// assert!(R.eq_el(&z, &y));
/// ```
/// 
/// # Canonical mappings
/// This ring has a canonical isomorphism to Barett-reduction based Zn
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::from_primes(StaticRing::<i64>::RING, DefaultBigIntRing::RING, vec![17, 19]);
/// let S = zn_barett::Zn::new(StaticRing::<i64>::RING, 17 * 19);
/// assert!(R.eq_el(&R.from_int(12), &R.coerce(&S, S.from_int(12))));
/// assert!(S.eq_el(&S.from_int(12), &R.cast(&S, R.from_int(12))));
/// ```
/// and a canonical homomorphism from any integer ring
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_rns::*;
/// # use feanor_math::rings::bigint::*;
/// # use feanor_math::primitive_int::*;
/// let R = Zn::from_primes(StaticRing::<i16>::RING, DefaultBigIntRing::RING, vec![3, 5, 7]);
/// let S = DefaultBigIntRing::RING;
/// assert!(R.eq_el(&R.from_int(120493), &R.coerce(&S, S.from_int(120493))));
/// ```
/// 
pub struct ZnBase<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>> = AllocatingMemoryProvider> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    components: Vec<C>,
    total_ring: zn_barett::Zn<J>,
    unit_vectors: Vec<El<zn_barett::Zn<J>>>,
    memory_provider: M
}

pub type Zn<C, J> = RingValue<ZnBase<C, J>>;

impl<C: ZnRingStore + Clone, J: IntegerRingStore> Zn<C, J> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    pub fn new(component_rings: Vec<C>, large_integers: J) -> Self {
        Self::from(ZnBase::new(component_rings, large_integers, AllocatingMemoryProvider))
    }
}

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> Zn<zn_barett::Zn<I>, J> 
    where I::Type: IntegerRing + CanonicalIso<J::Type>, 
        J::Type: IntegerRing
{
    pub fn from_primes(integers: I, large_integers: J, primes: Vec<El<I>>) -> Self {
        Self::from(ZnBase::from_primes(integers, large_integers, primes))
    }
}

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> ZnBase<zn_barett::Zn<I>, J> 
    where I::Type: IntegerRing + CanonicalIso<J::Type>, 
        J::Type: IntegerRing
{
    pub fn from_primes(integers: I, large_integers: J, primes: Vec<El<I>>) -> Self {
        Self::new(
            primes.into_iter().map(|n| zn_barett::Zn::new(<I as Clone>::clone(&integers), n)).collect(),
            large_integers,
            AllocatingMemoryProvider
        )
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    pub fn new(component_rings: Vec<C>, large_integers: J, memory_provider: M) -> Self {
        assert!(component_rings.len() > 0);
        let total_modulus = large_integers.prod(
            component_rings.iter().map(|R| R.integer_ring().cast::<J>(&large_integers, R.integer_ring().clone_el(R.modulus())))
        );
        let total_ring = zn_barett::Zn::new(large_integers, total_modulus);
        let ZZ = total_ring.integer_ring();
        for R in &component_rings {
            let R_modulus = R.integer_ring().cast::<J>(ZZ, R.integer_ring().clone_el(R.modulus()));
            assert!(
                ZZ.is_one(&algorithms::eea::signed_gcd(ZZ.checked_div(total_ring.modulus(), &R_modulus).unwrap(), R_modulus, ZZ)),
                "all moduli must be coprime"
            );
        }
        let unit_vectors = component_rings.iter()
            .map(|R| ZZ.checked_div(total_ring.modulus(), &R.integer_ring().cast::<J>(ZZ, R.integer_ring().clone_el(R.modulus()))))
            .map(Option::unwrap)
            .map(|n| total_ring.coerce(&ZZ, n))
            .zip(component_rings.iter())
            .map(|(n, R)| total_ring.pow_gen(n, &R.integer_ring().sub_ref_fst(R.modulus(), R.integer_ring().one()), R.integer_ring()))
            .collect();
        ZnBase {
            components: component_rings,
            total_ring: total_ring,
            unit_vectors: unit_vectors,
            memory_provider: memory_provider
        }
    }

    fn ZZ(&self) -> &J {
        self.total_ring.integer_ring()
    }

    pub fn from_congruence<V: VectorView<El<C>>>(&self, el: V) -> ZnEl<C, M> {
        assert_eq!(self.components.len(), el.len());
        ZnEl(self.memory_provider.get_new_init(self.len(), |i| self.at(i).clone_el(el.at(i))))
    }

    pub fn get_congruence<'a>(&self, el: &'a ZnEl<C, M>) -> impl 'a + VectorView<El<C>> {
        &el.0 as &[El<C>]
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> VectorView<C> for ZnBase<C, J, M>
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    fn len(&self) -> usize {
        self.components.len()
    }

    fn at(&self, index: usize) -> &C {
        &self.components[index]
    }
}

pub struct ZnEl<C: ZnRingStore, M: MemoryProvider<El<C>>>(M::Object)
    where C::Type: ZnRing;

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> RingBase for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    type Element = ZnEl<C, M>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        ZnEl(self.memory_provider.get_new_init(self.len(), |i| self.at(i).clone_el(val.0.at(i))))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].add_assign_ref(&mut lhs.0[i], &rhs.0[i])
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (i, el) in (0..self.components.len()).zip(rhs.0.into_iter()) {
            self.components[i].add_assign_ref(&mut lhs.0[i], el)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].sub_assign_ref(&mut lhs.0[i], &rhs.0[i])
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].negate_inplace(&mut lhs.0[i])
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (i, el) in (0..self.components.len()).zip(rhs.0.into_iter()) {
            self.components[i].mul_assign_ref(&mut lhs.0[i], el)
        }
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].mul_assign_ref(&mut lhs.0[i], &rhs.0[i])
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        ZnEl(self.memory_provider.get_new_init(self.len(), |i| self.components[i].from_int(value)))
    }
    
    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..self.components.len() {
            self.components[i].mul_assign_int(&mut lhs.0[i], rhs)
        }

    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        (0..self.components.len()).zip(lhs.0.iter()).zip(rhs.0.iter()).all(|((i, l), r)| self.components[i].eq_el(l, r))
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.0.iter()).all(|(i, x)| self.components[i].is_zero(x))
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.0.iter()).all(|(i, x)| self.components[i].is_one(x))
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        (0..self.components.len()).zip(value.0.iter()).all(|(i, x)| self.components[i].is_neg_one(x))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.total_ring.get_ring().dbg(&RingRef::new(self).cast::<zn_barett::Zn<_>>(&self.total_ring, self.clone_el(value)), out)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> Clone for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>,
        C: Clone,
        J: Clone,
        M: Clone
{
    fn clone(&self) -> Self {
        ZnBase {
            components: self.components.clone(),
            total_ring: self.total_ring.clone(),
            unit_vectors: self.unit_vectors.iter().map(|e| self.total_ring.clone_el(e)).collect(),
            memory_provider: self.memory_provider.clone()
        }
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore, M1: MemoryProvider<El<C1>>, M2: MemoryProvider<El<C2>>> CanonicalHom<ZnBase<C2, J2, M2>> for ZnBase<C1, J1, M1> 
    where C1::Type: ZnRing + CanonicalHom<C2::Type> + CanonicalHom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J1::Type>,
        C2::Type: ZnRing + CanonicalHom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J2::Type>,
        J1::Type: IntegerRing,
        J2::Type: IntegerRing
{
    type Homomorphism = Vec<<C1::Type as CanonicalHom<C2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &ZnBase<C2, J2, M2>) -> Option<Self::Homomorphism> {
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

    fn map_in_ref(&self, from: &ZnBase<C2, J2, M2>, el: &ZnEl<C2, M2>, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(self.memory_provider.get_new_init(
            self.len(), 
            |i| self.at(i).get_ring().map_in_ref(from.at(i).get_ring(), el.0.at(i), &hom[i])
        ))
    }

    fn map_in(&self, from: &ZnBase<C2, J2, M2>, el: ZnEl<C2, M2>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> PartialEq for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>,
        J::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.components.len() == other.components.len() && self.components.iter().zip(other.components.iter()).all(|(R1, R2)| R1.get_ring() == R2.get_ring())
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore, M1: MemoryProvider<El<C1>>, M2: MemoryProvider<El<C2>>> CanonicalIso<ZnBase<C2, J2, M2>> for ZnBase<C1, J1, M1> 
    where C1::Type: ZnRing + CanonicalIso<C2::Type> + CanonicalHom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J1::Type>,
        C2::Type: ZnRing + CanonicalHom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J2::Type>,
        J1::Type: IntegerRing,
        J2::Type: IntegerRing
{
    type Isomorphism = Vec<<C1::Type as CanonicalIso<C2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &ZnBase<C2, J2, M2>) -> Option<Self::Isomorphism> {
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

    fn map_out(&self, from: &ZnBase<C2, J2, M2>, el: ZnEl<C1, M1>, iso: &Self::Isomorphism) -> ZnEl<C2, M2> {
        ZnEl(from.memory_provider.get_new_init(
            self.len(), 
            |i| self.at(i).get_ring().map_out(from.at(i).get_ring(), self.at(i).clone_el(el.0.at(i)), &iso[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore, M: MemoryProvider<El<C>>> CanonicalHom<zn_barett::ZnBase<K>> for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing + CanonicalIso<K::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>,
        K::Type: IntegerRing
{
    type Homomorphism = (<J::Type as CanonicalHom<K::Type>>::Homomorphism, Vec<<C::Type as CanonicalHom<J::Type>>::Homomorphism>);

    fn has_canonical_hom(&self, from: &zn_barett::ZnBase<K>) -> Option<Self::Homomorphism> {
        if self.total_ring.get_ring().has_canonical_hom(from).is_some() {
            Some((
                self.total_ring.get_ring().has_canonical_hom(from)?,
                self.components.iter()
                    .map(|s| s.get_ring())
                    .map(|s| s.has_canonical_hom(self.ZZ().get_ring()).ok_or(()))
                    .collect::<Result<Vec<_>, ()>>()
                    .ok()?
            ))
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_barett::ZnBase<K>, el: zn_barett::ZnEl<K>, hom: &Self::Homomorphism) -> ZnEl<C, M> {
        let lift = from.smallest_positive_lift(el);
        let mapped_lift = <J::Type as CanonicalHom<K::Type>>::map_in(
            self.ZZ().get_ring(), 
            from.integer_ring().get_ring(), 
            lift, 
            &hom.0
        );
        ZnEl(self.memory_provider.get_new_init(
            self.len(),
            |i| self.at(i).get_ring().map_in_ref(self.ZZ().get_ring(), &mapped_lift, &hom.1[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore, M: MemoryProvider<El<C>>> CanonicalIso<zn_barett::ZnBase<K>> for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing + CanonicalIso<K::Type>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>,
        K::Type: IntegerRing
{
    type Isomorphism = (
        <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::Isomorphism, 
        <zn_barett::ZnBase<J> as CanonicalHom<J::Type>>::Homomorphism,
        Vec<<<C::Type as ZnRing>::IntegerRingBase as CanonicalIso<J::Type>>::Isomorphism>
    );

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<K>) -> Option<Self::Isomorphism> {
        Some((
            <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::has_canonical_iso(self.total_ring.get_ring(), from)?,
            self.total_ring.get_ring().has_canonical_hom(self.total_ring.integer_ring().get_ring())?,
            self.components.iter()
                .map(|s| s.integer_ring().get_ring())
                .map(|s| s.has_canonical_iso(self.total_ring.integer_ring().get_ring()).unwrap())
                .collect()
        ))
    }

    fn map_out(&self, from: &zn_barett::ZnBase<K>, el: Self::Element, (final_iso, red, homs): &Self::Isomorphism) -> zn_barett::ZnEl<K> {
        let result = <_ as RingStore>::sum(&self.total_ring,
            self.components.iter()
                .zip(el.0.into_iter())
                .map(|(fp, x)| (fp.integer_ring().get_ring(), fp.smallest_positive_lift(fp.clone_el(x))))
                .zip(self.unit_vectors.iter())
                .zip(homs.iter())
                .map(|(((integers, x), u), hom)| (integers, x, u, hom))
                .map(|(integers, x, u, hom)| 
                    self.total_ring.mul_ref_snd(self.total_ring.get_ring().map_in(
                        self.total_ring.integer_ring().get_ring(), 
                        <_ as CanonicalIso<_>>::map_out(integers, self.total_ring.integer_ring().get_ring(), x, hom), 
                        red
                    ), u)
                )
        );
        return <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::map_out(self.total_ring.get_ring(), from, result, final_iso);
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRing, M: MemoryProvider<El<C>>> CanonicalHom<K> for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing + CanonicalIso<K>,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>,
        K: ?Sized + SelfIso
{
    type Homomorphism = (<J::Type as CanonicalHom<K>>::Homomorphism, Vec<<C::Type as CanonicalHom<J::Type>>::Homomorphism>);

    fn has_canonical_hom(&self, from: &K) -> Option<Self::Homomorphism> {
        Some((
            <J::Type as CanonicalHom<K>>::has_canonical_hom(self.ZZ().get_ring(), from)?,
            self.components.iter()
                .map(|R| <C::Type as CanonicalHom<J::Type>>::has_canonical_hom(R.get_ring(), self.ZZ().get_ring()).ok_or(()))
                .collect::<Result<Vec<_>, ()>>()
                .ok()?
        ))
    }

    fn map_in(&self, from: &K, el: K::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mapped_el = <J::Type as CanonicalHom<K>>::map_in(self.ZZ().get_ring(), from, el, &hom.0);
        ZnEl(self.memory_provider.get_new_init(
            self.len(),
            |i| <C::Type as CanonicalHom<J::Type>>::map_in_ref(self.at(i).get_ring(), self.ZZ().get_ring(), &mapped_el, &hom.1[i])
        ))
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> DivisibilityRing for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        Some(ZnEl(self.memory_provider.try_get_new_init(self.len(), |i| self.at(i).checked_div(lhs.0.at(i), rhs.0.at(i)).ok_or(())).ok()?))
    }
}

pub struct ZnBaseElementsIterator<'a, C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>>
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    ring: &'a ZnBase<C, J, M>,
    part_iters: Option<Vec<std::iter::Peekable<<C::Type as ZnRing>::ElementsIter<'a>>>>
}

impl<'a, C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> Iterator for ZnBaseElementsIterator<'a, C, J, M>
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    type Item = ZnEl<C, M>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(part_iters) = &mut self.part_iters {
            while part_iters.len() < self.ring.components.len() {
                part_iters.push(self.ring.components[part_iters.len()].elements().peekable());
            }

            let result = self.ring.memory_provider.get_new_init(
                self.ring.len(),
                |i: usize| self.ring.at(i).clone_el(part_iters[i].peek().unwrap())
            );
            part_iters.last_mut().unwrap().next();
            while part_iters.last_mut().unwrap().peek().is_none() {
                part_iters.pop();
                if part_iters.len() > 0 {
                    part_iters.last_mut().unwrap().next();
                } else {
                    self.part_iters = None;
                    return Some(ZnEl(result));
                }
            }
            return Some(ZnEl(result));
        } else {
            return None;
        }
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, M: MemoryProvider<El<C>>> ZnRing for ZnBase<C, J, M> 
    where C::Type: ZnRing + CanonicalHom<J::Type>,
        J::Type: IntegerRing + SelfIso,
        <C::Type as ZnRing>::IntegerRingBase: IntegerRing + CanonicalIso<J::Type>
{
    type IntegerRingBase = J::Type;
    type Integers = J;
    type ElementsIter<'a> = ZnBaseElementsIterator<'a, C, J, M>
        where Self: 'a;

    fn integer_ring(&self) -> &Self::Integers {
        self.ZZ()
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.total_ring.modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.total_ring.smallest_positive_lift(
            <Self as CanonicalIso<zn_barett::ZnBase<J>>>::map_out(
                self, 
                self.total_ring.get_ring(), 
                el, 
                &<Self as CanonicalIso<zn_barett::ZnBase<J>>>::has_canonical_iso(self, self.total_ring.get_ring()).unwrap()
            )
        )
    }

    fn elements<'a>(&'a self) -> ZnBaseElementsIterator<'a, C, J, M> {
        ZnBaseElementsIterator {
            ring: self,
            part_iters: Some(Vec::new())
        }
    }

    fn is_field(&self) -> bool {
        self.components.len() == 1
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> ZnEl<C, M> {
        ZnEl(self.memory_provider.get_new_init(self.len(), |i| self.at(i).random_element(&mut rng)))
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[cfg(test)]
const EDGE_CASE_ELEMENTS: [i32; 9] = [0, 1, 7, 9, 62, 8, 10, 11, 12];

#[test]
fn test_ring_axioms() {
    let ring = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    generic_test_ring_axioms(&ring, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| ring.from_int(x)))
}

#[test]
fn test_map_in_map_out() {
    let ring1 = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11, 17]);
    let ring2 = zn_barett::Zn::new(StaticRing::<i64>::RING, 7 * 11 * 17);
    for x in [0, 1, 7, 8, 9, 10, 11, 17, 7 * 17, 11 * 8, 11 * 17, 7 * 11 * 17 - 1] {
        let value = ring2.from_int(x);
        assert!(ring2.eq_el(&value, &ring1.cast(&ring2, ring1.coerce(&ring2, value.clone()))));
    }
}

#[test]
fn test_canonical_iso_axioms_zn_barett() {
    let from = zn_barett::Zn::new(StaticRing::<i128>::RING, 7 * 11);
    let to = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
    generic_test_canonical_hom_axioms(&from, &to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
}

#[test]
fn test_canonical_hom_axioms_static_int() {
    let from = StaticRing::<i32>::RING;
    let to = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    generic_test_canonical_hom_axioms(&from, to, EDGE_CASE_ELEMENTS.iter().cloned().map(|x| from.from_int(x)));
}

#[test]
fn test_zn_ring_axioms() {
    let ring = Zn::from_primes(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    generic_test_zn_ring_axioms(ring);
}

#[test]
fn test_zn_map_in_large_int() {
    let ring = Zn::from_primes(StaticRing::<i64>::RING, DefaultBigIntRing::RING, vec![7, 11]);
    generic_test_map_in_large_int(ring);

    let R = Zn::from_primes(StaticRing::<i16>::RING, DefaultBigIntRing::RING, vec![3, 5, 7]);
    let S = DefaultBigIntRing::RING;
    assert!(R.eq_el(&R.from_int(120493), &R.coerce(&S, S.from_int(120493))));
}