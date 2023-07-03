use crate::{divisibility::DivisibilityRing, ring::*, algorithms};
use crate::integer::*;
use crate::ordered::*;
use super::field::AsFieldBase;

pub mod zn_barett;
pub mod zn_42;
pub mod zn_static;
pub mod zn_rns;

pub trait ZnRing: DivisibilityRing + CanonicalHom<Self::IntegerRingBase> + SelfIso {

    // there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingStore`
    // or `Integers: RingStore<Type: IntegerRing>`
    type IntegerRingBase: IntegerRing + ?Sized;
    type Integers: RingStore<Type = Self::IntegerRingBase>;
    type ElementsIter<'a>: Iterator<Item = Self::Element>
        where Self: 'a;

    fn integer_ring(&self) -> &Self::Integers;
    fn modulus(&self) -> &El<Self::Integers>;
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers>;
    fn elements<'a>(&'a self) -> Self::ElementsIter<'a>;

    fn smallest_lift(&self, el: Self::Element) -> El<Self::Integers> {
        let result = self.smallest_positive_lift(el);
        let mut mod_half = self.integer_ring().clone_el(self.modulus());
        self.integer_ring().euclidean_div_pow_2(&mut mod_half, 1);
        if self.integer_ring().is_gt(&result, &mod_half) {
            return self.integer_ring().sub_ref_snd(result, self.modulus());
        } else {
            return result;
        }
    }

    fn is_field(&self) -> bool {
        algorithms::miller_rabin::is_prime_base(RingRef::new(self), 10)
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        self.map_in(
            self.integer_ring().get_ring(), 
            self.integer_ring().get_uniformly_random(self.modulus(), rng), 
            &self.has_canonical_hom(self.integer_ring().get_ring()).unwrap()
        )
    }
}

pub mod generic_maps {
    use crate::{ring::*, divisibility::DivisibilityRingStore};
    use super::ZnRing;

    #[allow(type_alias_bounds)]
    pub type GenericHomomorphism<R: ZnRing, S: ZnRing> = (<S as CanonicalHom<S::IntegerRingBase>>::Homomorphism, <S::IntegerRingBase as CanonicalHom<R::IntegerRingBase>>::Homomorphism);

    pub fn generic_has_canonical_hom<R: ZnRing, S: ZnRing>(from: &R, to: &S) -> Option<GenericHomomorphism<R, S>> 
        where S::IntegerRingBase: CanonicalHom<R::IntegerRingBase>
    {
        let hom = <S::IntegerRingBase as CanonicalHom<R::IntegerRingBase>>::has_canonical_hom(to.integer_ring().get_ring(), from.integer_ring().get_ring())?;
        if to.integer_ring().checked_div(&<S::IntegerRingBase as CanonicalHom<R::IntegerRingBase>>::map_in_ref(&to.integer_ring().get_ring(), from.integer_ring().get_ring(), from.modulus(), &hom), &to.modulus()).is_some() {
            Some((to.has_canonical_hom(to.integer_ring().get_ring()).unwrap(), hom))
        } else {
            None
        }
    }

    pub fn generic_map_in<R: ZnRing, S: ZnRing>(from: &R, to: &S, el: R::Element, hom: &GenericHomomorphism<R, S>) -> S::Element 
        where S::IntegerRingBase: CanonicalHom<R::IntegerRingBase>
    {
        to.map_in(to.integer_ring().get_ring(), <S::IntegerRingBase as CanonicalHom<R::IntegerRingBase>>::map_in(to.integer_ring().get_ring(), from.integer_ring().get_ring(), from.smallest_positive_lift(el), &hom.1), &hom.0)
    }
}

pub trait ZnRingStore: RingStore
    where Self::Type: ZnRing
{    
    delegate!{ fn integer_ring(&self) -> &<Self::Type as ZnRing>::Integers }
    delegate!{ fn modulus(&self) -> &El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn smallest_positive_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn smallest_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::Integers> }
    delegate!{ fn is_field(&self) -> bool }

    fn elements<'a>(&'a self) -> <Self::Type as ZnRing>::ElementsIter<'a> {
        self.get_ring().elements()
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> El<Self> {
        self.get_ring().random_element(rng)
    }
    
    fn as_field(self) -> Result<RingValue<AsFieldBase<Self>>, Self> 
        where Self: Sized
    {
        if self.is_field() {
            Ok(RingValue::from(unsafe { AsFieldBase::unsafe_create(self) }))
        } else {
            Err(self)
        }
    }
}

impl<R: RingStore> ZnRingStore for R
    where R::Type: ZnRing
{}

#[cfg(any(test, feature = "generic_tests"))]
use crate::primitive_int::*;
#[cfg(any(test, feature = "generic_tests"))]
use super::bigint::DefaultBigIntRing;

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_zn_ring_axioms<R: ZnRingStore>(R: R)
    where R::Type: ZnRing,
        <R::Type as ZnRing>::IntegerRingBase: CanonicalIso<StaticRingBase<i128>> + CanonicalIso<StaticRingBase<i32>>
{
    let ZZ = R.integer_ring();
    let n = R.modulus();

    assert!(R.is_zero(&R.coerce(ZZ, ZZ.clone_el(n))));
    assert!(R.is_field() == algorithms::miller_rabin::is_prime(ZZ, n, 10));

    let mut k = ZZ.one();
    while ZZ.is_lt(&k, &n) {
        assert!(!R.is_zero(&R.coerce(ZZ, ZZ.clone_el(&k))));
        ZZ.add_assign(&mut k, ZZ.one());
    }

    let all_elements = R.elements().collect::<Vec<_>>();
    assert_eq!(ZZ.cast(&StaticRing::<i128>::RING, ZZ.clone_el(n)) as usize, all_elements.len());
    for (i, x) in all_elements.iter().enumerate() {
        for (j, y) in all_elements.iter().enumerate() {
            assert!(i == j || !R.eq_el(x, y));
        }
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_map_in_large_int<R: ZnRingStore>(R: R)
    where <R as RingStore>::Type: ZnRing + CanonicalHom<DefaultBigIntRing>
{
    let ZZ_big = DefaultBigIntRing::RING;
    let n = ZZ_big.power_of_two(1000);
    let x = R.coerce(&ZZ_big, n);
    assert!(R.eq_el(&R.pow(R.from_int(2), 1000), &x));
}