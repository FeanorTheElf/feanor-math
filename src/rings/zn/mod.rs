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

pub mod generic_impls {
    use std::marker::PhantomData;

    use crate::{ring::*, divisibility::DivisibilityRingStore, integer::{IntegerRing, IntegerRingStore}};
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

    pub struct GenericIntegerToZnHom<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing>
        where I: CanonicalIso<R::IntegerRingBase> + CanonicalIso<J>
    {
        highbit_mod: usize,
        highbit_bound: usize,
        int_ring: PhantomData<I>,
        to_large_int_ring: PhantomData<J>,
        hom: <I as CanonicalHom<R::IntegerRingBase>>::Homomorphism,
        iso: <I as CanonicalIso<R::IntegerRingBase>>::Isomorphism,
        iso2: <I as CanonicalIso<J>>::Isomorphism
    }

    ///
    /// See [`generic_map_in_from_int()`].
    /// This will only ever return `None` if one of the integer ring `has_canonical_hom/iso` returns `None`.
    /// 
    pub fn generic_has_canonical_hom_from_int<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing>(from: &I, to: &R, to_large_int_ring: &J, bounded_reduce_bound: Option<&J::Element>) -> Option<GenericIntegerToZnHom<I, J, R>>
        where I: CanonicalIso<R::IntegerRingBase> + CanonicalIso<J>
    {
        if let Some(bound) = bounded_reduce_bound {
            Some(GenericIntegerToZnHom {
                highbit_mod: to.integer_ring().abs_highest_set_bit(to.modulus()).unwrap(),
                highbit_bound: to_large_int_ring.abs_highest_set_bit(bound).unwrap(),
                int_ring: PhantomData,
                to_large_int_ring: PhantomData,
                hom: from.has_canonical_hom(to.integer_ring().get_ring())?,
                iso: from.has_canonical_iso(to.integer_ring().get_ring())?,
                iso2: from.has_canonical_iso(to_large_int_ring)?
            })
        } else {
            Some(GenericIntegerToZnHom {
                highbit_mod: to.integer_ring().abs_highest_set_bit(to.modulus()).unwrap(),
                highbit_bound: usize::MAX,
                int_ring: PhantomData,
                to_large_int_ring: PhantomData,
                hom: from.has_canonical_hom(to.integer_ring().get_ring())?,
                iso: from.has_canonical_iso(to.integer_ring().get_ring())?,
                iso2: from.has_canonical_iso(to_large_int_ring)?
            })
        }
    }

    ///
    /// A parameterized, generic variant of the reduction `Z -> Z/nZ`.
    /// It considers the following situations:
    ///  - the source ring `Z` might not be large enough to represent `n`
    ///  - the integer ring associated to the destination ring `Z/nZ` might not be large enough to represent the input
    ///  - the destination ring might use Barett reductions (or similar) for fast modular reduction if the input is bounded by some fixed bound `B`
    ///  - general modular reduction modulo `n` is only performed in the source ring if necessary
    /// 
    /// In particular, we use the following additional parameters:
    ///  - `to_large_int_ring`: an integer ring that can represent all integers for which we can perform fast modular reduction (i.e. those bounded by `B`)
    ///  - `from_positive_representative_exact`: a function that performs the restricted reduction `{0, ..., n - 1} -> Z/nZ`
    ///  - `from_positive_representative_bounded`: a function that performs the restricted reduction `{0, ..., B - 1} -> Z/nZ`
    /// 
    /// Note that the input size estimates consider only the bitlength of numbers, and so there is a small margin in which a reduction method for larger
    /// numbers than necessary is used. Furthermore, if the integer rings used can represent some but not all positive numbers of a certain bitlength, 
    /// there might be rare edge cases with panics/overflows. 
    /// 
    /// In particular, if the input integer ring `Z` can represent the input `x`, but not `n` AND `x` and `n` have the same bitlength, this function might
    /// decide that we have to perform generic modular reduction (even though `x < n`), and try to map `n` into `Z`. This is never a problem if the primitive
    /// integer rings `StaticRing::<ixx>::RING` are used, or if `B >= 2n`.
    /// 
    pub fn generic_map_in_from_int<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing, F, G>(from: &I, to: &R, to_large_int_ring: &J, el: I::Element, hom: &GenericIntegerToZnHom<I, J, R>, from_positive_representative_exact: F, from_positive_representative_bounded: G) -> R::Element
        where I: CanonicalIso<R::IntegerRingBase> + CanonicalIso<J>,
            F: FnOnce(El<R::Integers>) -> R::Element,
            G: FnOnce(J::Element) -> R::Element
    {
        let (neg, n) = if from.is_neg(&el) {
            (true, from.negate(el))
        } else {
            (false, el)
        };
        let ZZ = to.integer_ring().get_ring();
        let highbit_el = from.abs_highest_set_bit(&n).unwrap_or(0);

        let reduced = if highbit_el < hom.highbit_mod {
            from_positive_representative_exact(from.map_out(ZZ, n, &hom.iso))
        } else if highbit_el < hom.highbit_bound {
            from_positive_representative_bounded(from.map_out(to_large_int_ring, n, &hom.iso2))
        } else {
            from_positive_representative_exact(from.map_out(ZZ, from.euclidean_rem(n, &from.map_in_ref(ZZ, to.modulus(), &hom.hom)), &hom.iso))
        };
        if neg {
            to.negate(reduced)
        } else {
            reduced
        }
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