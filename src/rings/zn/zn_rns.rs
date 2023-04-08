use crate::{integer::IntegerRingStore, divisibility::DivisibilityRingStore};
use crate::ordered::OrderedRingStore;
use crate::rings::zn::*;

use super::zn_dyn::{Fp, FpEl, FpBase};

#[derive(Clone)]
pub struct ZnBase<I: IntegerRingStore, J: IntegerRingStore> {
    components: Vec<Fp<I>>,
    total_ring: zn_dyn::Zn<J>,
    unit_vectors: Vec<El<zn_dyn::Zn<J>>>
}

pub type Zn<I, J> = RingValue<ZnBase<I, J>>;

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> Zn<I, J> {
    
    pub fn new(ring: I, large_ring: J, primes: Vec<El<I>>) -> Self {
        Self::from(ZnBase::new(ring, large_ring, primes))
    }
}

impl<I: IntegerRingStore + Clone, J: IntegerRingStore> ZnBase<I, J> {

    #[allow(non_snake_case)]
    pub fn new(ring: I, large_ring: J, primes: Vec<El<I>>) -> Self {
        assert!(primes.len() > 0);
        for i in 1..primes.len() {
            assert!(ring.is_gt(&primes[i], &primes[i - 1]));
        }
        let total_modulus = large_ring.prod(
            primes.iter().map(|p| large_ring.coerce::<I>(&ring, p.clone()))
        );
        let total_ring = zn_dyn::Zn::new(large_ring, total_modulus);
        let ZZ = total_ring.integer_ring();
        let components: Vec<_> = primes.into_iter()
            .map(|p| zn_dyn::ZnBase::new(ring.clone(), p))
            .map(|r| r.is_field().ok().unwrap())
            .map(|r| RingValue::from(r))
            .collect();
        let unit_vectors = (0..components.len())
            .map(|i| ZZ.checked_div(total_ring.modulus(), &ZZ.coerce::<I>(&ring, components[i].modulus().clone())))
            .map(|n| n.unwrap())
            .map(|n| total_ring.coerce(&ZZ, n))
            .enumerate()
            .map(|(i, n)| total_ring.pow_gen(&n, &ring.sub_ref_fst(components[i].modulus(), ring.one()), &ring))
            .collect();
        ZnBase { components, total_ring, unit_vectors }
    }
}

pub struct ZnEl<I: IntegerRingStore>(Vec<FpEl<I>>);

impl<I: IntegerRingStore> Clone for ZnEl<I> {

    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> RingBase for ZnBase<I, J> {

    type Element = ZnEl<I>;

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
    
    fn from_z(&self, value: i32) -> Self::Element {
        ZnEl((0..self.components.len()).map(|i| self.components[i].from_z(value)).collect())
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
        self.total_ring.get_ring().dbg(&RingRef::new(self).cast(&self.total_ring, value.clone()), out)
    }
}

impl<I1: IntegerRingStore, J1: IntegerRingStore, I2: IntegerRingStore, J2: IntegerRingStore> CanonicalHom<ZnBase<I2, J2>> for ZnBase<I1, J1> {

    type Homomorphism = Vec<<FpBase<I1> as CanonicalHom<FpBase<I2>>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &ZnBase<I2, J2>) -> Option<Self::Homomorphism> {
        if self.components.len() == from.components.len() {
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f): (&Fp<I1>, &Fp<I2>)| s.get_ring().has_canonical_hom(f.get_ring()).ok_or(()))
                .collect::<Result<Self::Homomorphism, ()>>()
                .ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &ZnBase<I2, J2>, el: ZnEl<I2>, hom: &Self::Homomorphism) -> Self::Element {
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

impl<I1: IntegerRingStore, J1: IntegerRingStore, I2: IntegerRingStore, J2: IntegerRingStore> CanonicalIso<ZnBase<I2, J2>> for ZnBase<I1, J1> {

    type Isomorphism = Vec<<FpBase<I1> as CanonicalIso<FpBase<I2>>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &ZnBase<I2, J2>) -> Option<Self::Isomorphism> {
        if self.components.len() == from.components.len() {
            self.components.iter()
                .zip(from.components.iter())
                .map(|(s, f): (&Fp<I1>, &Fp<I2>)| s.get_ring().has_canonical_iso(f.get_ring()).ok_or(()))
                .collect::<Result<Self::Homomorphism, ()>>()
                .ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &ZnBase<I2, J2>, el: Self::Element, iso: &Self::Isomorphism) -> ZnEl<I2> {
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

impl<I: IntegerRingStore, J: IntegerRingStore, K: IntegerRingStore> CanonicalHom<zn_dyn::ZnBase<K>> for ZnBase<I, J> {

    type Homomorphism = Vec<<zn_dyn::ZnBase<J> as CanonicalHom<K::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &zn_dyn::ZnBase<K>) -> Option<Self::Homomorphism> {
        if self.total_ring.get_ring().has_canonical_hom(from).is_some() {
            self.components.iter()
                .map(|s| s.get_ring())
                .map(|s| s.has_canonical_hom(from.integer_ring().get_ring()).ok_or(()))
                .collect::<Result<Self::Homomorphism, ()>>()
                .ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_dyn::ZnBase<K>, el: zn_dyn::ZnEl<K>, hom: &Self::Homomorphism) -> ZnEl<I> {
        ZnEl(
            self.components.iter()
                .map(|s| s.get_ring())
                .zip(hom.iter())
                .map(|(r, hom)| r.map_in(from.integer_ring().get_ring(), from.smallest_positive_lift(el.clone()), hom))
                .collect()
        )
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore, K: IntegerRingStore> CanonicalIso<zn_dyn::ZnBase<K>> for ZnBase<I, J> {

    type Isomorphism = (
        <zn_dyn::ZnBase<J> as CanonicalIso<zn_dyn::ZnBase<K>>>::Isomorphism, 
        Vec<<zn_dyn::ZnBase<J> as CanonicalHom<I::Type>>::Homomorphism>
    );

    fn has_canonical_iso(&self, from: &zn_dyn::ZnBase<K>) -> Option<Self::Isomorphism> {
        Some((
            <zn_dyn::ZnBase<J> as CanonicalIso<zn_dyn::ZnBase<K>>>::has_canonical_iso(self.total_ring.get_ring(), from)?,
            self.components.iter()
                .map(|s| s.integer_ring().get_ring())
                .map(|s| self.total_ring.get_ring().has_canonical_hom(s).unwrap())
                .collect()
        ))
    }

    fn map_out(&self, from: &zn_dyn::ZnBase<K>, el: Self::Element, (final_iso, homs): &Self::Isomorphism) -> zn_dyn::ZnEl<K> {
        let result = self.total_ring.sum(
            self.components.iter()
                .zip(el.0.into_iter())
                .map(|(fp, x)| (fp.integer_ring().get_ring(), fp.smallest_positive_lift(x)))
                .zip(self.unit_vectors.iter())
                .zip(homs.iter())
                .map(|(((integers, x), u), hom)| (integers, x, u, hom))
                .map(|(integers, x, u, hom)| 
                    self.total_ring.mul(<zn_dyn::ZnBase<J> as CanonicalHom<I::Type>>::map_in(self.total_ring.get_ring(), integers, x, hom), u.clone())
                )
        );
        return <zn_dyn::ZnBase<J> as CanonicalIso<zn_dyn::ZnBase<K>>>::map_out(self.total_ring.get_ring(), from, result, final_iso);
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore, K: IntegerRing> CanonicalHom<K> for ZnBase<I, J> 
    where K: CanonicalIso<K> + ?Sized
{
    type Homomorphism = Vec<<I::Type as CanonicalHom<K>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &K) -> Option<Self::Homomorphism> {
        self.components.iter()
            .map(|r| r.get_ring().has_canonical_hom(from).ok_or(()))
            .collect::<Result<Self::Homomorphism, ()>>()
            .ok()
    }

    fn map_in(&self, from: &K, el: K::Element, hom: &Self::Homomorphism) -> Self::Element {
        ZnEl(
            self.components.iter()
                .zip(hom)
                .map(|(r, hom)| r.get_ring().map_in(from, el.clone(), hom))
                .collect()
        )
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> DivisibilityRing for ZnBase<I, J> {
    
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        unimplemented!()
    }
}

impl<I: IntegerRingStore, J: IntegerRingStore> ZnRing for ZnBase<I, J> {
    
    type IntegerRingBase = J::Type;
    type Integers = J;
    type IteratorState = Vec<<FpBase<I> as ZnRing>::IteratorState>;

    fn integer_ring(&self) -> &Self::Integers {
        self.total_ring.integer_ring()
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.total_ring.modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.total_ring.smallest_positive_lift(
            <Self as CanonicalIso<zn_dyn::ZnBase<J>>>::map_out(
                self, 
                self.total_ring.get_ring(), 
                el, 
                &<Self as CanonicalIso<zn_dyn::ZnBase<J>>>::has_canonical_iso(self, self.total_ring.get_ring()).unwrap()
            )
        )
    }

    fn elements<'a>(&'a self) -> ZnElementsIterator<'a, Self> {
        unimplemented!()
    }

    fn elements_iterator_next<'a>(iter: &mut ZnElementsIterator<'a, Self>) -> Option<Self::Element> {
        unimplemented!()
    }

    fn is_field(&self) -> bool {
        self.components.len() == 1
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> ZnEl<I> {
        ZnEl::<I>(self.components.iter()
            .map(|r| r.random_element(&mut rng))
            .collect::<Vec<_>>())
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_zn_ring_axioms() {
    let ring = Zn::new(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    test_ring_axioms(&ring, [0, 1, 7, 9, 62, 8, 10, 11, 12].iter().cloned().map(|x| ring.from_z(x)))
}

#[test]
fn test_map_in_map_out() {
    let ring1 = Zn::new(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11, 17]);
    let ring2 = zn_dyn::Zn::new(StaticRing::<i32>::RING, 7 * 11 * 17);
    for x in [0, 1, 7, 8, 9, 10, 11, 17, 7 * 17, 11 * 8, 11 * 17, 7 * 11 * 17 - 1] {
        let value = ring2.from_z(x);
        assert!(ring2.eq(&value, &ring1.cast(&ring2, ring1.coerce(&ring2, value.clone()))));
    }
}