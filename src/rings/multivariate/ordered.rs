use std::marker::PhantomData;

use crate::ring::*;
use crate::mempool::*;
use crate::vector::VectorViewIter;
use crate::vector::VectorViewMut;

use super::*;

pub struct MultivariatePolyRingImplBase<R, O, M, const N: usize>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    base_ring: R,
    memory_provider: M,
    order: O,
    zero: El<R>
}

pub type MultivariatePolyRingImpl<R, O, M, const N: usize> = RingValue<MultivariatePolyRingImplBase<R, O, M, N>>;

impl<R, O, M, const N: usize> MultivariatePolyRingImpl<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    pub fn new(base_ring: R, monomial_order: O, memory_provider: M) -> Self {
        RingValue::from(MultivariatePolyRingImplBase {
            zero: base_ring.zero(),
            base_ring,
            memory_provider: memory_provider,
            order: monomial_order
        })
    }
}

impl<R, O, M, const N: usize> MultivariatePolyRingImplBase<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    fn is_valid(&self, el: &<Self as RingBase>::Element) -> bool {
        for i in 1..el.len() {
            if self.order.compare(&el.at(i - 1).1, &el.at(i).1) != Ordering::Less {
                return false;
            }
        }
        return true;
    }

    fn remove_zeros(&self, el: &mut M::Object) {
        let mut i = 0;
        for j in 0..el.len() {
            if !self.base_ring.is_zero(&el.at(j).0) {
                if i != j {
                    let tmp = std::mem::replace(el.at_mut(j), (self.base_ring.zero(), Monomial::new([0; N])));
                    *el.at_mut(i) = tmp;
                }
                i += 1;
            }
        }
        self.memory_provider.shrink(el, i);
    }

    #[inline]
    fn add_scaled<const SCALED: bool>(&self, lhs: &<Self as RingBase>::Element, rhs: &<Self as RingBase>::Element, m: &Monomial<[MonomialExponent; N]>, factor: &El<R>) -> <Self as RingBase>::Element {
        debug_assert!(self.is_valid(lhs));
        debug_assert!(self.is_valid(rhs));
        
        let mut result = self.memory_provider.get_new_init(lhs.len() + rhs.len(), |_| (self.base_ring.zero(), Monomial::new([0; N])));
        
        let mut i_l = 0;
        let mut i_r = 0;
        let mut i_o = 0;
        while i_l < lhs.len() && i_r < rhs.len() {
            let mut rhs_monomial = rhs.at(i_r).1;
            rhs_monomial.mul_assign(m);
            match self.order.compare(&lhs.at(i_l).1, &rhs_monomial) {
                Ordering::Equal => {
                    *result.at_mut(i_o) = (self.base_ring.add_ref_fst(&lhs.at(i_l).0, self.base_ring.mul_ref(&rhs.at(i_r).0, factor)), lhs.at(i_l).1);
                    i_l += 1;
                    i_r += 1;
                },
                Ordering::Greater => {
                    if SCALED {
                        *result.at_mut(i_o) = (self.base_ring.mul_ref(&rhs.at(i_r).0, factor), rhs_monomial);
                    } else {
                        *result.at_mut(i_o) = (self.base_ring.clone_el(&rhs.at(i_r).0), rhs_monomial);
                    }
                    i_r += 1;
                },
                Ordering::Less => {
                    *result.at_mut(i_o) = (self.base_ring.clone_el(&lhs.at(i_l).0), lhs.at(i_l).1);
                    i_l += 1;
                }
            }
            i_o += 1;
        }
        if i_l == lhs.len() {
            for i in i_r..rhs.len() {
                let mut rhs_monomial = rhs.at(i).1;
                rhs_monomial.mul_assign(m);
                if SCALED {
                    *result.at_mut(i_o) = (self.base_ring.mul_ref(&rhs.at(i).0, factor), rhs_monomial);
                } else {
                    *result.at_mut(i_o) = (self.base_ring.clone_el(&rhs.at(i).0), rhs_monomial);
                }
                i_o += 1;
            }
        } else {
            for i in i_l..lhs.len() {
                *result.at_mut(i_o) = (self.base_ring.clone_el(&lhs.at(i).0), lhs.at(i).1);
                i_o += 1;
            }
        }
        self.remove_zeros(&mut result);
        return result;
    }
}

impl<R, O, M, const N: usize> PartialEq for MultivariatePolyRingImplBase<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<R, O, M, const N: usize> RingBase for MultivariatePolyRingImplBase<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    type Element = M::Object;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        self.memory_provider.get_new_init(val.len(), |i| (self.base_ring.clone_el(&val.at(i).0), val.at(i).1))
    }
    
    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.add_scaled::<false>(lhs, rhs, &Monomial::new([0; N]), &self.base_ring().one())
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        if lhs.len() > rhs.len() {
            (0..rhs.len()).fold(self.zero(), |current, i| self.add_scaled::<true>(&current, lhs, &rhs.at(i).1, &rhs.at(i).0))
        } else {
            (0..lhs.len()).fold(self.zero(), |current, i| self.add_scaled::<true>(&current, rhs, &lhs.at(i).1, &lhs.at(i).0))
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.add_ref(lhs, rhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.add_ref(lhs, &rhs);
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..lhs.len() {
            self.base_ring.negate_inplace(&mut lhs.at_mut(i).0);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.mul_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring.from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.len() != rhs.len() {
            return false;
        }
        for i in 0..lhs.len() {
            if lhs.at(i).1 != rhs.at(i).1 || !self.base_ring.eq_el(&lhs.at(i).0, &rhs.at(i).0) {
                return false
            }
        }
        return true;
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.len() == 0
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        value.len() == 1 && value.at(0).1 == Monomial::new([0; N]) && self.base_ring.is_one(&value.at(0).0)
    }
    fn is_neg_one(&self, value: &Self::Element) -> bool {
        value.len() == 1 && value.at(0).1 == Monomial::new([0; N]) && self.base_ring.is_neg_one(&value.at(0).0)
    }

    fn is_commutative(&self) -> bool { self.base_ring.is_commutative() }
    fn is_noetherian(&self) -> bool { self.base_ring.is_commutative() }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let mut print_term = |c: &El<R>, m: &Monomial<[MonomialExponent; N]>, with_plus: bool| {
            if with_plus {
                write!(out, " + ")?;
            }
            if !self.base_ring.is_one(c) || self.order.compare(m, &Monomial::new([0; N])) == Ordering::Equal {
                write!(out, "{}", self.base_ring.format(c))?;
                if self.order.compare(m, &Monomial::new([0; N])) != Ordering::Equal {
                    write!(out, " * ")?;
                }
            }
            let mut needs_space = false;
            for i in 0..N {
                if m[i] > 0 {
                    if needs_space {
                        write!(out, " * ")?;
                    }
                    write!(out, "X{}", i)?;
                    needs_space = true;
                }
                if m[i] > 1 {
                    write!(out, "^{}", m[i])?;
                }
            }
            return Ok::<(), std::fmt::Error>(());
        };

        if value.len() == 0 {
            write!(out, "{}", self.base_ring.format(&self.base_ring.zero()))?;
        } else {
            for i in 0..value.len() {
                print_term(&value.at(i).0, &value.at(i).1, i != 0)?;
            }
        }

        return Ok(());
    }
}


impl<R, O, M, const N: usize> RingExtension for MultivariatePolyRingImplBase<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        if self.base_ring.is_zero(&x) {
            self.memory_provider.get_new_init(0, |_| unreachable!())
        } else {
            let mut x_opt = Some(x);
            self.memory_provider.get_new_init(1, |_| (x_opt.take().unwrap(), Monomial::new([0; N])))
        }
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        for i in 0..lhs.len() {
            self.base_ring.mul_assign_ref(&mut lhs.at_mut(i).0, rhs)
        }
        self.remove_zeros(lhs);
    }
}

impl<R1, O1, M1, R2, O2, M2, const N1: usize, const N2: usize> CanonicalHom<MultivariatePolyRingImplBase<R2, O2, M2, N2>> for MultivariatePolyRingImplBase<R1, O1, M1, N1>
    where R1: RingStore,
        O1: MonomialOrder,
        M1: GrowableMemoryProvider<(El<R1>, Monomial<[MonomialExponent; N1]>)>,
        R2: RingStore,
        O2: MonomialOrder,
        M2: GrowableMemoryProvider<(El<R2>, Monomial<[MonomialExponent; N2]>)>,
        R1::Type: CanonicalHom<R2::Type>
{
    type Homomorphism = <R1::Type as CanonicalHom<R2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &MultivariatePolyRingImplBase<R2, O2, M2, N2>) -> Option<Self::Homomorphism> {
        if N1 >= N2 {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    default fn map_in_ref(&self, from: &MultivariatePolyRingImplBase<R2, O2, M2, N2>, el: &<MultivariatePolyRingImplBase<R2, O2, M2, N2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = self.memory_provider.get_new_init(el.len(), |i| (
            self.base_ring.get_ring().map_in_ref(from.base_ring().get_ring(), &el.at(i).0, hom), 
            Monomial::new(std::array::from_fn(|j| el.at(i).1[j] ))
        ));
        result.sort_by(|l, r| self.order.compare(&l.1, &r.1));
        return result;
    }

    default fn map_in(&self, from: &MultivariatePolyRingImplBase<R2, O2, M2, N2>, el: <MultivariatePolyRingImplBase<R2, O2, M2, N2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<R1, O, M1, R2, M2, const N: usize> CanonicalHom<MultivariatePolyRingImplBase<R2, O, M2, N>> for MultivariatePolyRingImplBase<R1, O, M1, N>
    where R1: RingStore,
        O: MonomialOrder + PartialEq,
        M1: GrowableMemoryProvider<(El<R1>, Monomial<[MonomialExponent; N]>)>,
        R2: RingStore,
        M2: GrowableMemoryProvider<(El<R2>, Monomial<[MonomialExponent; N]>)>,
        R1::Type: CanonicalHom<R2::Type>
{
    fn map_in_ref(&self, from: &MultivariatePolyRingImplBase<R2, O, M2, N>, el: &<MultivariatePolyRingImplBase<R2, O, M2, N> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = self.memory_provider.get_new_init(el.len(), |i| (self.base_ring.get_ring().map_in_ref(from.base_ring().get_ring(), &el.at(i).0, hom), el.at(i).1));
        if self.order != from.order {
            result.sort_by(|l, r| self.order.compare(&l.1, &r.1));
        }
        return result;
    }

    fn map_in(&self, from: &MultivariatePolyRingImplBase<R2, O, M2, N>, el: <MultivariatePolyRingImplBase<R2, O, M2, N> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<R1, O1, M1, R2, O2, M2, const N1: usize, const N2: usize> CanonicalIso<MultivariatePolyRingImplBase<R2, O2, M2, N2>> for MultivariatePolyRingImplBase<R1, O1, M1, N1>
    where R1: RingStore,
        O1: MonomialOrder,
        M1: GrowableMemoryProvider<(El<R1>, Monomial<[MonomialExponent; N1]>)>,
        R2: RingStore,
        O2: MonomialOrder,
        M2: GrowableMemoryProvider<(El<R2>, Monomial<[MonomialExponent; N2]>)>,
        R1::Type: CanonicalIso<R2::Type>
{
    type Isomorphism = <R1::Type as CanonicalIso<R2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &MultivariatePolyRingImplBase<R2, O2, M2, N2>) -> Option<Self::Isomorphism> {
        if N1 == N2 {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    default fn map_out(&self, from: &MultivariatePolyRingImplBase<R2, O2, M2, N2>, el: Self::Element, iso: &Self::Isomorphism) -> <MultivariatePolyRingImplBase<R2, O2, M2, N2> as RingBase>::Element {
        let mut result = from.memory_provider.get_new_init(el.len(), |i| (
            self.base_ring.get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.at(i).0), iso), 
            Monomial::new(std::array::from_fn(|j| el.at(i).1[j] ))
        ));
        result.sort_by(|l, r| from.order.compare(&l.1, &r.1));
        return result;
    }
}

impl<R1, O, M1, R2, M2, const N: usize> CanonicalIso<MultivariatePolyRingImplBase<R2, O, M2, N>> for MultivariatePolyRingImplBase<R1, O, M1, N>
    where R1: RingStore,
        O: MonomialOrder + PartialEq,
        M1: GrowableMemoryProvider<(El<R1>, Monomial<[MonomialExponent; N]>)>,
        R2: RingStore,
        M2: GrowableMemoryProvider<(El<R2>, Monomial<[MonomialExponent; N]>)>,
        R1::Type: CanonicalIso<R2::Type>
{
    fn map_out(&self, from: &MultivariatePolyRingImplBase<R2, O, M2, N>, el: Self::Element, iso: &Self::Isomorphism) -> <MultivariatePolyRingImplBase<R2, O, M2, N> as RingBase>::Element {
        let mut result = from.memory_provider.get_new_init(el.len(), |i| (
            self.base_ring.get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.at(i).0), iso), 
            el.at(i).1
        ));
        if self.order != from.order {
            result.sort_by(|l, r| self.order.compare(&l.1, &r.1));
        }
        return result;
    }
}

pub struct MultivariatePolyRingBaseTermsIter<'a, R, O, M, const N: usize>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    base_iter: VectorViewIter<'a, M::Object, (El<R>, Monomial<[MonomialExponent; N]>)>,
    order: PhantomData<O>
}

impl<'a, R, O, M, const N: usize> Iterator for MultivariatePolyRingBaseTermsIter<'a, R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    type Item = (&'a El<R>, &'a Monomial<[MonomialExponent; N]>);

    fn next(&mut self) -> Option<Self::Item> {
        let (c, m) = self.base_iter.next()?;
        return Some((c, m));
    }
}

impl<R, O, M, const N: usize> MultivariatePolyRing for MultivariatePolyRingImplBase<R, O, M, N>
    where R: RingStore,
        O: MonomialOrder,
        M: GrowableMemoryProvider<(El<R>, Monomial<[MonomialExponent; N]>)>
{
    type MonomialVector = [MonomialExponent; N];
    type TermsIterator<'a> = MultivariatePolyRingBaseTermsIter<'a, R, O, M, N>
        where Self: 'a;

    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a> {
        MultivariatePolyRingBaseTermsIter {
            base_iter: f.iter(),
            order: PhantomData
        }
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Monomial<Self::MonomialVector>) -> &'a El<Self::BaseRing> {
        match f.binary_search_by(|x| self.order.compare(&x.1, m)) {
            Ok(i) => &f.at(i).0,
            Err(_) => &self.zero
        }
    }

    fn indeterminate_len(&self) -> usize {
        N
    }

    fn indeterminate(&self, i: usize) -> Self::Element {
        self.memory_provider.get_new_init(1, |_| (
            self.base_ring.one(),
            Monomial::new(std::array::from_fn(|j| if i == j { 1 } else { 0 }))
        ))
    }

    fn mul_monomial(&self, el: &mut Self::Element, m: &Monomial<Self::MonomialVector>) {
        for i in 0..el.len() {
            el.at_mut(i).1.mul_assign(m);
        }
    }

    fn lm<'a, O2>(&'a self, f: &'a Self::Element, order: O2) -> Option<&'a Monomial<Self::MonomialVector>>
        where O2: MonomialOrder
    {
        if f.len() == 0 {
            return None;
        } else if self.order.is_same(order.clone()) {
            return Some(&f.at(f.len() - 1).1);
        } else {
            return Some(&f.iter().max_by(|(_, ml), (_, mr)| order.compare(ml, mr)).unwrap().1);
        }
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::default_memory_provider;

#[test]
fn test_add() {
    let ring: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, Lex, default_memory_provider!());
    let lhs = ring.from_terms([
        (1, &Monomial::new([1, 0, 0]))
    ].into_iter());
    ring.println(&lhs);

    let rhs = ring.from_terms([
        (1, &Monomial::new([1, 0, 0])),
        (1, &Monomial::new([1, 0, 1])),
    ].into_iter());

    let expected = ring.from_terms([
        (2, &Monomial::new([1, 0, 0])),
        (1, &Monomial::new([1, 0, 1]))
    ].into_iter());

    let actual = ring.add_ref(&lhs, &rhs);
    assert_el_eq!(&ring, &expected, &actual);

    let lhs = ring.from_terms([
        (1, &Monomial::new([1, 0, 0])),
        (2, &Monomial::new([1, 0, 1])),
        (4, &Monomial::new([0, 2, 1])),
        (8, &Monomial::new([0, 0, 0]))
    ].into_iter());

    let rhs = ring.from_terms([
        (-1, &Monomial::new([1, 0, 0])),
        (3, &Monomial::new([1, 0, 1])),
        (9, &Monomial::new([0, 0, 1]))
    ].into_iter());

    let expected = ring.from_terms([
        (5, &Monomial::new([1, 0, 1])),
        (4, &Monomial::new([0, 2, 1])),
        (8, &Monomial::new([0, 0, 0])),
        (9, &Monomial::new([0, 0, 1]))
    ].into_iter());

    let actual = ring.add_ref(&lhs, &rhs);
    assert_el_eq!(&ring, &expected, &actual);
}

#[cfg(test)]
fn edge_case_elements<'a, M: GrowableMemoryProvider<(i64, Monomial<[MonomialExponent; 3]>)>>(ring: &'a RingValue<MultivariatePolyRingImplBase<StaticRing<i64>, DegRevLex, M, 3>>) -> impl 'a + Iterator<Item = <MultivariatePolyRingImplBase<StaticRing<i64>, DegRevLex, M, 3> as RingBase>::Element> {
    let mut result = vec![];
    let monomials = [
        Monomial::new([1, 0, 0]),
        Monomial::new([1, 0, 1]),
        Monomial::new([2, 0, 1]),
        Monomial::new([4, 2, 0])

    ];
    for m1 in &monomials {
        result.push(ring.from_terms([(1, m1)].into_iter()));
    }
    for m1 in &monomials {
        for m2 in &monomials {
            result.push(ring.from_terms([(1, m1), (-2, m2)].into_iter()));
        }
    }
    return result.into_iter();
}

#[test]
fn test_ring_axioms() {
    let ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    generic_tests::test_ring_axioms(&ring, edge_case_elements(&ring));
}