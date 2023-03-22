use crate::{integer::IntegerRingWrapper, divisibility::DivisibilityRingWrapper};
use crate::ordered::OrderedRingWrapper;
use crate::rings::zn::*;

use super::zn_dyn::{Fp, FpEl};

#[derive(Clone)]
pub struct ZnBase<I: IntegerRingWrapper, J: IntegerRingWrapper> {
    components: Vec<Fp<I>>,
    total_ring: zn_dyn::Zn<J>,
    unit_vectors: Vec<El<zn_dyn::Zn<J>>>
}

pub type Zn<I, J> = RingValue<ZnBase<I, J>>;

impl<I: IntegerRingWrapper + Clone, J: IntegerRingWrapper> ZnBase<I, J> {

    #[allow(non_snake_case)]
    pub fn new(ring: I, large_ring: J, primes: Vec<El<I>>) -> Self {
        assert!(primes.len() > 0);
        for i in 1..primes.len() {
            assert!(ring.is_gt(&primes[i], &primes[i - 1]));
        }
        let total_modulus = large_ring.prod(
            primes.iter().map(|p| large_ring.map_in::<I>(&ring, p.clone()))
        );
        let total_ring = RingValue::new(zn_dyn::ZnBase::new(large_ring, total_modulus));
        let ZZ = total_ring.integer_ring();
        let components: Vec<_> = primes.into_iter()
            .map(|p| zn_dyn::ZnBase::new(ring.clone(), p))
            .map(|r| r.is_field().ok().unwrap())
            .map(|r| RingValue::new(r))
            .collect();
        let unit_vectors = (0..components.len())
            .map(|i| ZZ.checked_div(total_ring.modulus(), &ZZ.map_in::<I>(&ring, components[i].modulus().clone())))
            .map(|n| n.unwrap())
            .map(|n| total_ring.map_in(&ZZ, n))
            .enumerate()
            .map(|(i, n)| total_ring.pow_gen(&n, &ring.sub_ref_fst(components[i].modulus(), ring.one()), &ring))
            .collect();
        ZnBase { components, total_ring, unit_vectors }
    }
}

pub struct ZnEl<I: IntegerRingWrapper>(Vec<FpEl<I>>);

impl<I: IntegerRingWrapper> Clone for ZnEl<I> {

    fn clone(&self) -> Self {
        ZnEl(self.0.clone())
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper> RingBase for ZnBase<I, J> {

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
        self.total_ring.get_ring().dbg(&self.map_out(self.total_ring.get_ring(), value.clone()), out)
    }
}

impl<I1: IntegerRingWrapper, J1: IntegerRingWrapper, I2: IntegerRingWrapper, J2: IntegerRingWrapper> CanonicalHom<ZnBase<I2, J2>> for ZnBase<I1, J1> {

    fn has_canonical_hom(&self, from: &ZnBase<I2, J2>) -> bool {
        self.components.len() == from.components.len() && 
            self.components.iter()
                .zip(from.components.iter())
                .all(|(s, f)| s.get_ring().has_canonical_hom(f.get_ring()))
    }

    fn map_in(&self, from: &ZnBase<I2, J2>, el: ZnEl<I2>) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        ZnEl(
            self.components.iter()
                .zip(from.components.iter())
                .zip(el.0.into_iter())
                .map(|((s, f), x)| s.map_in(f, x))
                .collect()
        )
    }
}

impl<I1: IntegerRingWrapper, J1: IntegerRingWrapper, I2: IntegerRingWrapper, J2: IntegerRingWrapper> CanonicalIso<ZnBase<I2, J2>> for ZnBase<I1, J1> {

    fn has_canonical_iso(&self, from: &ZnBase<I2, J2>) -> bool {
        self.components.len() == from.components.len() && 
            self.components.iter()
                .zip(from.components.iter())
                .all(|(s, f)| s.get_ring().has_canonical_iso(f.get_ring()))
    }

    fn map_out(&self, from: &ZnBase<I2, J2>, el: Self::Element) -> ZnEl<I2> {
        debug_assert!(self.has_canonical_iso(from));
        ZnEl(
            self.components.iter()
                .zip(from.components.iter())
                .zip(el.0.into_iter())
                .map(|((s, f), x)| s.map_out(f, x))
                .collect()
        )
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper, K: IntegerRingWrapper> CanonicalHom<zn_dyn::ZnBase<K>> for ZnBase<I, J> {

    fn has_canonical_hom(&self, from: &zn_dyn::ZnBase<K>) -> bool {
        self.total_ring.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &zn_dyn::ZnBase<K>, el: zn_dyn::ZnEl<K>) -> ZnEl<I> {
        debug_assert!(self.has_canonical_hom(from));
        ZnEl(
            self.components.iter()
                .map(|r| r.map_in(from.integer_ring(), from.smallest_positive_lift(el.clone())))
                .collect()
        )
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper, K: IntegerRingWrapper> CanonicalIso<zn_dyn::ZnBase<K>> for ZnBase<I, J> {

    fn has_canonical_iso(&self, from: &zn_dyn::ZnBase<K>) -> bool {
        self.total_ring.get_ring().has_canonical_iso(from)
    }

    fn map_out(&self, from: &zn_dyn::ZnBase<K>, el: Self::Element) -> zn_dyn::ZnEl<K> {
        debug_assert!(self.has_canonical_iso(from));
        RingRef::new(from).sum(
            self.components.iter().zip(el.0.into_iter())
                .map(|(fp, x)| (fp.integer_ring(), fp.smallest_positive_lift(x)))
                .zip(self.unit_vectors.iter())
                .map(|((z, x), u)| from.mul(
                    from.map_in(self.total_ring.get_ring(), u.clone()), 
                    from.map_in(z.get_ring(), x)
                ))
        )
    }
}

impl<I: IntegerRingWrapper, J: IntegerRingWrapper, K: IntegerRing> CanonicalHom<K> for ZnBase<I, J> 
    where K: CanonicalIso<K>
{

    fn has_canonical_hom(&self, from: &K) -> bool {
        self.components.iter().all(|r| r.get_ring().has_canonical_hom(from))
    }

    fn map_in(&self, from: &K, el: K::Element) -> Self::Element {
        debug_assert!(self.has_canonical_hom(from));
        ZnEl(
            self.components.iter()
                .map(|r| r.get_ring().map_in(from, el.clone()))
                .collect()
        )
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_zn_ring_axioms() {
    let ring = ZnBase::new(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11]);
    test_ring_axioms(RingValue::new(ring.clone()), [0, 1, 7, 9, 62, 8, 10, 11, 12].iter().cloned().map(|x| ring.from_z(x)))
}

#[test]
fn test_map_in_map_out() {
    let ring1 = ZnBase::new(StaticRing::<i64>::RING, StaticRing::<i64>::RING, vec![7, 11, 17]);
    let ring2 = zn_dyn::ZnBase::new(StaticRing::<i32>::RING, 1309);
    for x in [0, 1, 7, 8, 9, 10, 11, 17, 7 * 17, 11 * 8, 11 * 17, 7 * 11 * 17 - 1] {
        let value = ring2.from_z(x);
        assert!(ring2.eq(&value, &ring1.map_out(&ring2, ring1.map_in(&ring2, value.clone()))));
    }
}