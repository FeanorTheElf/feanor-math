use crate::delegate::DelegateRing;
use crate::euclidean::EuclideanRing;
use crate::field::Field;
use crate::ring::*;
use crate::rings::zn::*;

#[derive(Clone, Copy)]
pub struct FpBase<R: ZnRingStore> {
    base: R
}

#[allow(type_alias_bounds)]
pub type Fp<R: ZnRingStore> = RingValue<FpBase<R>>;

pub struct FpEl<R: ZnRingStore>(El<R>);

impl<R: ZnRingStore> Clone for FpEl<R> {

    fn clone(&self) -> Self {
        FpEl(self.0.clone())
    }
}

impl<R: ZnRingStore> Copy for FpEl<R> 
    where El<R>: Copy
{}

impl<R: ZnRingStore> Fp<R> {

    pub fn new(base: R) -> Self {
        Self::from(FpBase::new(base))
    }
}

impl<R: ZnRingStore> FpBase<R> {

    pub fn new(base: R) -> Self {
        assert!(algorithms::miller_rabin::is_prime(base.integer_ring(), base.modulus(), 10));
        Self { base }
    }
}

impl<R: ZnRingStore> DelegateRing for FpBase<R> {

    type Element = FpEl<R>;
    type Base = R::Type;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.0
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.0
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.0
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        FpEl(el)
    }
}

impl<R: ZnRingStore, S: ZnRingStore> CanonicalHom<FpBase<S>> for FpBase<R> 
    where R::Type: CanonicalHom<S::Type>
{
    type Homomorphism = <R::Type as CanonicalHom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &FpBase<S>) -> Option<Self::Homomorphism> {
        <R::Type as CanonicalHom<S::Type>>::has_canonical_hom(self.base_ring().get_ring(), from.base_ring().get_ring())
    }

    fn map_in(&self, from: &FpBase<S>, el: FpEl<S>, hom: &Self::Homomorphism) -> Self::Element {
        FpEl(<R::Type as CanonicalHom<S::Type>>::map_in(self.base_ring().get_ring(), from.base_ring().get_ring(), el.0, hom))
    }
}

impl<R: ZnRingStore, S: ZnRingStore> CanonicalIso<FpBase<S>> for FpBase<R> 
    where R::Type: CanonicalIso<S::Type>
{
    type Isomorphism = <R::Type as CanonicalIso<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &FpBase<S>) -> Option<Self::Isomorphism> {
        <R::Type as CanonicalIso<S::Type>>::has_canonical_iso(self.base_ring().get_ring(), from.base_ring().get_ring())
    }

    fn map_out(&self, from: &FpBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> FpEl<S> {
        FpEl(<R::Type as CanonicalIso<S::Type>>::map_out(self.base_ring().get_ring(), from.base_ring().get_ring(), el.0, iso))
    }
}

impl<R: ZnRingStore> RingExtension for FpBase<R> {

    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        FpEl(x)
    }
}

impl<R: ZnRingStore, J: IntegerRing + ?Sized> CanonicalHom<J> for FpBase<R> 
    where J: SelfIso, R::Type: CanonicalHom<J>
{
    type Homomorphism = <R::Type as CanonicalHom<J>>::Homomorphism;

    fn has_canonical_hom(&self, from: &J) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &J, el: J::Element, hom: &Self::Homomorphism) -> Self::Element {
        FpEl(<R::Type as CanonicalHom<J>>::map_in(self.base_ring().get_ring(), from, el, hom))
    }
}

impl<R: ZnRingStore> DivisibilityRing for FpBase<R> 
    where Self: ZnRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(rhs) && self.is_zero(lhs) {
            Some(self.zero())
        } else if self.is_zero(rhs) {
            None
        } else {
            let (s, _, d) = algorithms::eea::eea(self.smallest_positive_lift(rhs.clone()), self.modulus().clone(), self.integer_ring());
            debug_assert!(self.integer_ring().is_one(&d) || self.integer_ring().is_neg_one(&d));
            let inverse = self.map_in(self.integer_ring().get_ring(), s, &self.has_canonical_hom(self.integer_ring().get_ring()).unwrap());
            return Some(self.mul_ref_snd(inverse, lhs));
        }
    }
}

impl<R: ZnRingStore> EuclideanRing for FpBase<R> 
    where Self: ZnRing
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }

    fn euclidean_rem(&self, _: Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        self.zero()
    }
}

impl<R: ZnRingStore> Field for FpBase<R> 
    where Self: ZnRing
{}

pub struct FpBaseElementsIter<'a, R>
    where R: ZnRingStore, R::Type: 'a
{
    iter: <R::Type as ZnRing>::ElementsIter<'a>
}

impl<'a, R> Iterator for FpBaseElementsIter<'a, R>
    where R: ZnRingStore, R::Type: 'a
{
    type Item = FpEl<R>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(FpEl)
    }
}

impl<R: ZnRingStore> ZnRing for FpBase<R>
    where <R::Type as ZnRing>::IntegerRingBase: SelfIso
{
    type IntegerRingBase = <R::Type as ZnRing>::IntegerRingBase;
    type Integers = <R::Type as ZnRing>::Integers;
    type ElementsIter<'a> = FpBaseElementsIter<'a, R>
        where Self: 'a;

    fn integer_ring(&self) -> &Self::Integers {
        self.base_ring().integer_ring()
    }

    fn modulus(&self) -> &El<Self::Integers> {
        self.base_ring().modulus()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        self.base_ring().smallest_positive_lift(el.0)
    }

    fn elements<'a>(&'a self) -> FpBaseElementsIter<'a, R> {
        FpBaseElementsIter {
            iter: self.base_ring().elements()
        }
    }
}
