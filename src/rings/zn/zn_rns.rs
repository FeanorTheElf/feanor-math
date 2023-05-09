use crate::algorithms::cooley_tuckey::FFTTableCooleyTuckey;
use crate::vector::{VectorViewMut, VectorView};
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
/// assert_eq!(35, R.smallest_lift(<_ as RingStore>::clone(&R, &x)));
/// let y = R.mul_ref(&x, &x);
/// let z = R.get_ring().from_congruence([R.get_ring().at(0).from_int(1 * 1), R.get_ring().at(1).from_int(16 * 16)]);
/// assert!(R.eq(&z, &y));
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
/// assert!(R.eq(&R.from_int(12), &R.coerce(&S, S.from_int(12))));
/// assert!(S.eq(&S.from_int(12), &R.cast(&S, R.from_int(12))));
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
/// assert!(R.eq(&R.from_int(120493), &R.coerce(&S, S.from_int(120493))));
/// ```
///
pub struct ZnBase<C: ZnRingStore, J: IntegerRingStore> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    components: Vec<C>,
    total_ring: zn_barett::Zn<J>,
    unit_vectors: Vec<El<zn_barett::Zn<J>>>
}

pub type Zn<C, J> = RingValue<ZnBase<C, J>>;

impl<C: ZnRingStore + Clone, J: IntegerRingStore> Zn<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    pub fn new(component_rings: Vec<C>, large_integers: J) -> Self {
        Self::from(ZnBase::new(component_rings, large_integers))
    }
}

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> Zn<zn_barett::Zn<I>, J> {

    pub fn from_primes(integers: I, large_integers: J, primes: Vec<El<I>>) -> Self {
        Self::from(ZnBase::from_primes(integers, large_integers, primes))
    }
}

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> ZnBase<zn_barett::Zn<I>, J> {

    pub fn from_primes(integers: I, large_integers: J, primes: Vec<El<I>>) -> Self {
        Self::new(
            primes.into_iter().map(|n| zn_barett::Zn::new(<I as Clone>::clone(&integers), n)).collect(),
            large_integers
        )
    }
}

impl<C: ZnRingStore, J: IntegerRingStore> ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    pub fn new(component_rings: Vec<C>, large_integers: J) -> Self {
        assert!(component_rings.len() > 0);
        let total_modulus = large_integers.prod(
            component_rings.iter().map(|R| large_integers.coerce::<<C::Type as ZnRing>::Integers>(R.integer_ring(), R.integer_ring().clone(R.modulus())))
        );
        let total_ring = zn_barett::Zn::new(large_integers, total_modulus);
        let ZZ = total_ring.integer_ring();
        for R in &component_rings {
            let R_modulus = ZZ.coerce::<<C::Type as ZnRing>::Integers>(R.integer_ring(), R.integer_ring().clone(R.modulus()));
            assert!(
                ZZ.is_one(&algorithms::eea::signed_gcd(ZZ.checked_div(total_ring.modulus(), &R_modulus).unwrap(), R_modulus, ZZ)),
                "all moduli must be coprime"
            );
        }
        let unit_vectors = component_rings.iter()
            .map(|R| ZZ.checked_div(total_ring.modulus(), &ZZ.coerce::<<C::Type as ZnRing>::Integers>(R.integer_ring(), R.integer_ring().clone(R.modulus()))))
            .map(Option::unwrap)
            .map(|n| total_ring.coerce(&ZZ, n))
            .zip(component_rings.iter())
            .map(|(n, R)| total_ring.pow_gen(n, &R.integer_ring().sub_ref_fst(R.modulus(), R.integer_ring().one()), R.integer_ring()))
            .collect();
        ZnBase {
            components: component_rings,
            total_ring: total_ring,
            unit_vectors: unit_vectors
        }
    }

    fn ZZ(&self) -> &J {
        self.total_ring.integer_ring()
    }

    pub fn from_congruence<V: VectorView<El<C>>>(&self, el: V) -> ZnEl<C> {
        assert_eq!(self.components.len(), el.len());
        ZnEl((0..el.len()).map(|i| self.at(i).clone(el.at(i))).collect())
    }

    pub fn get_congruence<'a>(&self, el: &'a ZnEl<C>) -> impl 'a + VectorView<El<C>> {
        &el.0
    }

}

impl<C: ZnRingStore, J: IntegerRingStore> VectorView<C> for ZnBase<C, J>
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    fn len(&self) -> usize {
        self.components.len()
    }

    fn at(&self, index: usize) -> &C {
        &self.components[index]
    }
}

pub struct ZnEl<C: ZnRingStore>(Vec<El<C>>);

impl<C: ZnRingStore, J: IntegerRingStore> RingBase for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type Element = ZnEl<C>;

    fn clone(&self, val: &Self::Element) -> Self::Element {
        ZnEl(val.0.iter().enumerate().map(|(i, x)| self.at(i).clone(x)).collect())
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].add_assign_ref(&mut lhs.0[i], &rhs.0[i])
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for (i, el) in (0..self.components.len()).zip(rhs.0.into_iter()) {
            self.components[i].add_assign(&mut lhs.0[i], el)
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
            self.components[i].mul_assign(&mut lhs.0[i], el)
        }
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..self.components.len() {
            self.components[i].mul_assign_ref(&mut lhs.0[i], &rhs.0[i])
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        ZnEl((0..self.components.len()).map(|i| self.components[i].from_int(value)).collect())
    }
    
    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..self.components.len() {
            self.components[i].mul_assign_int(&mut lhs.0[i], rhs)
        }
    }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        (0..self.components.len()).zip(lhs.0.iter()).zip(rhs.0.iter()).all(|((i, l), r)| self.components[i].eq(l, r))
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
        self.total_ring.get_ring().dbg(&RingRef::new(self).cast(&self.total_ring, self.clone(value)), out)
    }
}

impl<C: ZnRingStore, J: IntegerRingStore> Clone for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso,
        C: Clone,
        J: Clone
{
    fn clone(&self) -> Self {
        ZnBase {
            components: <_ as Clone>::clone(&self.components),
            total_ring: <_ as Clone>::clone(&self.total_ring),
            unit_vectors: self.unit_vectors.iter().map(|e| <_ as RingStore>::clone(&self.total_ring, e)).collect()
        }
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore> CanonicalHom<ZnBase<C2, J2>> for ZnBase<C1, J1> 
    where C1::Type: CanonicalHom<C2::Type>,
        C1::Type: CanonicalHom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: SelfIso,
        C2::Type: CanonicalHom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type Homomorphism = Vec<<C1::Type as CanonicalHom<C2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &ZnBase<C2, J2>) -> Option<Self::Homomorphism> {
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

    fn map_in(&self, from: &ZnBase<C2, J2>, el: ZnEl<C2>, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f)| (s.get_ring(), f.get_ring()))
                .zip(el.0.into_iter())
                .zip(hom.iter())
                .map(|(((s, f), x), hom)| (s, f, x, hom))
                .map(|(s, f, x, hom)| s.map_in(f, x, hom))
                .collect()
        )
    }
}

impl<C1: ZnRingStore, J1: IntegerRingStore, C2: ZnRingStore, J2: IntegerRingStore> CanonicalIso<ZnBase<C2, J2>> for ZnBase<C1, J1> 
    where C1::Type: CanonicalIso<C2::Type>,
        C1::Type: CanonicalHom<J1::Type>,
        <C1::Type as ZnRing>::IntegerRingBase: SelfIso,
        C2::Type: CanonicalHom<J2::Type>,
        <C2::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type Isomorphism = Vec<<C1::Type as CanonicalIso<C2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &ZnBase<C2, J2>) -> Option<Self::Isomorphism> {
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

    fn map_out(&self, from: &ZnBase<C2, J2>, el: Self::Element, iso: &Self::Isomorphism) -> ZnEl<C2> {
        ZnEl(
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f)| (s.get_ring(), f.get_ring()))
                .zip(el.0.into_iter())
                .zip(iso.iter())
                .map(|(((s, f), x), hom)| (s, f, x, hom))
                .map(|(s, f, x, hom)| s.map_out(f, x, hom))
                .collect()
        )
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore> CanonicalHom<zn_barett::ZnBase<K>> for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
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

    fn map_in(&self, from: &zn_barett::ZnBase<K>, el: zn_barett::ZnEl<K>, hom: &Self::Homomorphism) -> ZnEl<C> {
        let lift = from.smallest_positive_lift(el);
        let mapped_lift = <J::Type as CanonicalHom<K::Type>>::map_in(
            self.ZZ().get_ring(), 
            from.integer_ring().get_ring(), 
            lift, 
            &hom.0
        );
        ZnEl(
            self.components.iter()
                .map(|R| R.get_ring())
                .zip(hom.1.iter())
                .map(|(R, hom)| R.map_in_ref(self.ZZ().get_ring(), &mapped_lift, hom))
                .collect()
        )
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRingStore> CanonicalIso<zn_barett::ZnBase<K>> for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type Isomorphism = (
        <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::Isomorphism, 
        Vec<<J::Type as CanonicalHom<<C::Type as ZnRing>::IntegerRingBase>>::Homomorphism>
    );

    fn has_canonical_iso(&self, from: &zn_barett::ZnBase<K>) -> Option<Self::Isomorphism> {
        Some((
            <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::has_canonical_iso(self.total_ring.get_ring(), from)?,
            self.components.iter()
                .map(|s| s.integer_ring().get_ring())
                .map(|s| self.total_ring.get_ring().has_canonical_hom(s).unwrap())
                .collect()
        ))
    }

    fn map_out(&self, from: &zn_barett::ZnBase<K>, el: Self::Element, (final_iso, homs): &Self::Isomorphism) -> zn_barett::ZnEl<K> {
        let result = <_ as RingStore>::sum(&self.total_ring,
            self.components.iter()
                .zip(el.0.into_iter())
                .map(|(fp, x)| (fp.integer_ring().get_ring(), fp.smallest_positive_lift(x)))
                .zip(self.unit_vectors.iter())
                .zip(homs.iter())
                .map(|(((integers, x), u), hom)| (integers, x, u, hom))
                .map(|(integers, x, u, hom)| 
                    self.total_ring.mul_ref_snd(<zn_barett::ZnBase<J> as CanonicalHom<_>>::map_in(self.total_ring.get_ring(), integers, x, hom), u)
                )
        );
        return <zn_barett::ZnBase<J> as CanonicalIso<zn_barett::ZnBase<K>>>::map_out(self.total_ring.get_ring(), from, result, final_iso);
    }
}

impl<C: ZnRingStore, J: IntegerRingStore, K: IntegerRing> CanonicalHom<K> for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso,
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
        ZnEl(
            self.components.iter()
                .zip(hom.1.iter())
                .map(|(R, hom)| <C::Type as CanonicalHom<J::Type>>::map_in_ref(R.get_ring(), self.ZZ().get_ring(), &mapped_el, hom))
                .collect()
        )
    }
}

impl<C: ZnRingStore, J: IntegerRingStore> DivisibilityRing for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        Some(ZnEl(self.components.iter()
            .zip(lhs.0.iter())
            .zip(rhs.0.iter())
            .map(|((r, x), y)| (r, x, y))
            .map(|(r, x, y)| r.checked_left_div(x, y).ok_or(()))
            .collect::<Result<Vec<El<C>>, ()>>().ok()?))
    }
}

pub struct ZnBaseElementsIterator<'a, C: ZnRingStore, J: IntegerRingStore>
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    ring: &'a ZnBase<C, J>,
    part_iters: Option<Vec<std::iter::Peekable<<C::Type as ZnRing>::ElementsIter<'a>>>>
}

impl<'a, C: ZnRingStore, J: IntegerRingStore> Iterator for ZnBaseElementsIterator<'a, C, J>
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type Item = ZnEl<C>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(part_iters) = &mut self.part_iters {
            while part_iters.len() < self.ring.components.len() {
                part_iters.push(self.ring.components[part_iters.len()].elements().peekable());
            }
            let result = part_iters.iter_mut().enumerate().map(|(i, it)| self.ring.at(i).clone(it.peek().unwrap())).collect::<Vec<_>>();
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

impl<C: ZnRingStore, J: IntegerRingStore> ZnRing for ZnBase<C, J> 
    where C::Type: CanonicalHom<J::Type>,
        <C::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type IntegerRingBase = J::Type;
    type Integers = J;
    type ElementsIter<'a> = ZnBaseElementsIterator<'a, C, J>
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

    fn elements<'a>(&'a self) -> ZnBaseElementsIterator<'a, C, J> {
        ZnBaseElementsIterator {
            ring: self,
            part_iters: Some(Vec::new())
        }
    }

    fn is_field(&self) -> bool {
        self.components.len() == 1
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> ZnEl<C> {
        ZnEl(self.components.iter()
            .map(|r| r.random_element(&mut rng))
            .collect::<Vec<_>>())
    }
}

pub struct RNSFFTTable<'a, C: ZnRingStore> {
    part_tables: Vec<FFTTableCooleyTuckey<&'a C>>
}

impl<'a, C: ZnRingStore> RNSFFTTable<'a, C> {

    pub fn new<J: IntegerRingStore>(ring: &'a ZnBase<C, J>, log2_n: usize) -> Option<Self> 
        where C::Type: CanonicalHom<J::Type>,
            <C::Type as ZnRing>::IntegerRingBase: SelfIso
    {
        Some(RNSFFTTable {
            part_tables: ring.components.iter()
                .map(|r| FFTTableCooleyTuckey::for_zn(r, log2_n).ok_or(()))
                .collect::<Result<Vec<_>, ()>>()
                .ok()?
        })
    }
}

impl<'a, C: ZnRingStore> RNSFFTTable<'a, C> {

    pub fn bitreverse_fft_inplace<V: VectorViewMut<ZnEl<C>>>(&self, mut values: V) {
        for i in 0..self.part_tables.len() {
            self.part_tables[i].bitreverse_fft_inplace((&mut values).map_mut(|x| &x.0[i], |x| &mut x.0[i]));
        }
    }

    pub fn bitreverse_inv_fft_inplace<V: VectorViewMut<ZnEl<C>>>(&self, mut values: V) {
        for i in 0..self.part_tables.len() {
            self.part_tables[i].bitreverse_inv_fft_inplace((&mut values).map_mut(|x| &x.0[i], |x| &mut x.0[i]));
        }
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
        assert!(ring2.eq(&value, &ring1.cast(&ring2, ring1.coerce(&ring2, value.clone()))));
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
    assert!(R.eq(&R.from_int(120493), &R.coerce(&S, S.from_int(120493))));
}