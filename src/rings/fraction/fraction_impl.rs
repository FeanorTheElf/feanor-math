use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::algorithms::convolution::DefaultConvolutionRing;
use crate::algorithms::convolution::DynConvolution;
use crate::algorithms::convolution::NaiveConvolution;
use crate::algorithms::convolution::TypeErasableConvolution;
use crate::algorithms::matmul::StrassenHint;
use crate::homomorphism::*;
use crate::divisibility::*;
use crate::field::Field;
use crate::pid::PrincipalIdealRingStore;
use crate::homomorphism::Homomorphism;
use crate::rings::rational::RationalFieldBase;
use crate::integer::IntegerRing;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::ring::*;

use super::*;

#[stability::unstable(feature = "enable")]
pub struct FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    base_ring: R
}

///
/// [`RingStore`] for [`FractionFieldImplBase`].
/// 
#[stability::unstable(feature = "enable")]
pub type FractionFieldImpl<R> = RingValue<FractionFieldImplBase<R>>;

pub struct FractionFieldEl<R>
    where R: RingStore,
        R::Type: Domain
{
    num: El<R>,
    den: El<R>
}

impl<R> FractionFieldImpl<R>
    where R: RingStore,
        R::Type: Domain
{
    #[stability::unstable(feature = "enable")]
    pub fn new(base_ring: R) -> Self {
        assert!(base_ring.get_ring().is_commutative());
        RingValue::from(FractionFieldImplBase { base_ring })
    }
}

impl<R> FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    ///
    /// Partially reduces the fraction; This should be considered only a performance
    /// optimization, and does not give any reducedness-guarantees (since it only uses
    /// `balance_factor()` instead of `ideal_gen()`).
    /// 
    /// # Implementation rationale
    /// 
    /// It seems likely that in many cases, fractions are actually elements of the base
    /// ring (e.g. after rescaling a polynomial with [`FractionFieldImpl::balance_factor()`]).
    /// Hence, we completely reduce elements in this case, using `checked_div()`. This should
    /// still be much faster than a general gcd computation. In all other cases, just use 
    /// `balance_factor()` of the base ring.
    /// 
    fn reduce(&self, el: &mut FractionFieldEl<R>) {
        if let Some(quo) = self.base_ring.checked_div(&el.num, &el.den) {
            el.num = quo;
            el.den = self.base_ring.one();
        } else if let Some(factor) = <_ as DivisibilityRing>::balance_factor(self.base_ring.get_ring(), [&el.num, &el.den].into_iter()) {
            el.num = self.base_ring.checked_div(&el.num, &factor).unwrap();
            el.den = self.base_ring.checked_div(&el.den, &factor).unwrap();
        }
    }
}

impl<R> Debug for FractionFieldEl<R>
    where R: RingStore,
        R::Type: Domain,
        El<R>: Debug
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractionFieldEl")
            .field("num", &self.num)
            .field("den", &self.den)
            .finish()
    }
}

impl<R> PartialEq for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<R> Debug for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frac({:?})", self.base_ring.get_ring())
    }
}

impl<R> Clone for FractionFieldImplBase<R>
    where R: RingStore + Clone,
        R::Type: Domain
{
    fn clone(&self) -> Self {
        Self {
            base_ring: self.base_ring.clone()
        }
    }
}

impl<R> Copy for FractionFieldImplBase<R>
    where R: RingStore + Copy,
        R::Type: Domain,
        El<R>: Copy
{}

impl<R> Clone for FractionFieldEl<R>
    where R: RingStore,
        R::Type: Domain,
        El<R>: Clone
{
    fn clone(&self) -> Self {
        Self {
            num: self.num.clone(),
            den: self.den.clone()
        }
    }
}

impl<R> Copy for FractionFieldEl<R>
    where R: RingStore,
        R::Type: Domain,
        El<R>: Copy
{}

impl<R> RingBase for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    type Element = FractionFieldEl<R>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        FractionFieldEl {
            num: self.base_ring.clone_el(&val.num),
            den: self.base_ring.clone_el(&val.den)
        }
    }
    
    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.base_ring.mul_assign_ref(&mut lhs.num, &rhs.den);
        self.base_ring.add_assign(&mut lhs.num, self.base_ring.mul_ref(&lhs.den, &rhs.num));
        self.base_ring.mul_assign_ref(&mut lhs.den, &rhs.den);
        self.reduce(lhs);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.base_ring.mul_assign_ref(&mut lhs.num, &rhs.den);
        self.base_ring.add_assign(&mut lhs.num, self.base_ring.mul_ref_fst(&lhs.den, rhs.num));
        self.base_ring.mul_assign(&mut lhs.den, rhs.den);
        self.reduce(lhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.base_ring.mul_assign_ref(&mut lhs.num, &rhs.num);
        self.base_ring.mul_assign_ref(&mut lhs.den, &rhs.den);
        self.reduce(lhs);
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.base_ring.mul_assign(&mut lhs.num, rhs.num);
        self.base_ring.mul_assign(&mut lhs.den, rhs.den);
        self.reduce(lhs);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.base_ring.negate_inplace(&mut lhs.num);
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.base_ring.eq_el(
            &self.base_ring.mul_ref(&lhs.num, &rhs.den), 
            &self.base_ring.mul_ref(&rhs.num, &lhs.den)
        )
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.base_ring.is_zero(&value.num)
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        self.base_ring.eq_el(&value.num, &value.den)
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        self.base_ring.eq_el(&self.base_ring.negate(self.base_ring.clone_el(&value.num)), &value.den)
    }

    fn is_approximate(&self) -> bool {
        self.base_ring.get_ring().is_approximate()
    }

    fn is_commutative(&self) -> bool {
        // we currently enforce this, see assertion in construction; I'm not
        // sure if fraction field even works for noncommutative rings
        true
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring.int_hom().map(value))
    }

    fn fmt_el_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if let Some(quo) = self.base_ring.checked_div(&value.num, &value.den) {
            self.base_ring.get_ring().fmt_el_within(&quo, out, env)
        } else {
            if env >= EnvBindingStrength::Product {
                write!(out, "(")?;
            }
            self.base_ring.get_ring().fmt_el_within(&value.num, out, EnvBindingStrength::Product)?;
            write!(out, "/")?;
            self.base_ring.get_ring().fmt_el_within(&value.den, out, EnvBindingStrength::Product)?;
            if env >= EnvBindingStrength::Product {
                write!(out, ")")?;
            }
            Ok(())
        }
    }

    fn characteristic<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring.characteristic(ZZ)
    }
}

impl<R> RingExtension for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        FractionFieldEl {
            num: x,
            den: self.base_ring.one()
        }
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.base_ring.mul_assign_ref(&mut lhs.num, rhs);
        self.reduce(lhs);
    }
}

impl<R> DivisibilityRing for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            Some(self.zero())
        } else if self.is_zero(rhs) {
            None
        } else {
            let mut result = self.clone_el(lhs);
            self.base_ring.mul_assign_ref(&mut result.num, &rhs.den);
            self.base_ring.mul_assign_ref(&mut result.den, &rhs.num);
            self.reduce(&mut result);
            Some(result)
        }
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        !self.is_zero(x)
    }

    fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        // in general it is hard to get any guarantees, since `balance_factor()` has such
        // a weak contract; hence, we focus here on the case that `balance_factor()` in the
        // base ring behaves like gcd, and hope this is reasonable in the general case.
        // this means we take a denominator such that dividing by this will clear all denominators,
        // and then remove possible joint factors
        let mut denominator_lcm = self.base_ring.one();
        let mut it = elements.map(|x| {
            let gcd = self.base_ring.get_ring().balance_factor([&denominator_lcm, &x.den].into_iter());
            self.base_ring.mul_assign_ref(&mut denominator_lcm, &x.den);
            if let Some(gcd) = gcd {
                denominator_lcm = self.base_ring.checked_div(&denominator_lcm, &gcd).unwrap();
            }
            return &x.num;
        });
        let base_balance_factor = self.base_ring.get_ring().balance_factor(it.by_ref());
        it.for_each(|_| {});

        if let Some(den) = base_balance_factor {
            return Some(FractionFieldEl {
                num: den,
                den: denominator_lcm
            });
        } else {
            return Some(FractionFieldEl {
                num: self.base_ring.one(),
                den: denominator_lcm
            });
        }
    }

    fn prepare_divisor(&self, _: &Self::Element) -> Self::PreparedDivisorData {
        ()
    }
}

impl<R> Domain for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{}

impl<R> PrincipalIdealRing for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return Some(self.one());
        }
        self.checked_left_div(lhs, rhs)
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) && self.is_zero(rhs) {
            return (self.zero(), self.zero(), self.zero());
        } else if self.is_zero(lhs) {
            return (self.zero(), self.one(), self.clone_el(rhs));
        } else {
            return (self.one(), self.zero(), self.clone_el(lhs));
        }
    }
}

impl<R> EuclideanRing for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) { Some(0) } else { Some(1) }
    }

    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.checked_left_div(&lhs, rhs).unwrap(), self.zero())
    }
}

impl<R> Field for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{}

impl<R> FractionField for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn as_fraction(&self, el: Self::Element) -> (El<Self::BaseRing>, El<Self::BaseRing>) {
        (el.num, el.den)
    }
}

///
/// We don't have a canonical representation when the base ring is not an integer ring
/// (even if it is a PID), since we can always multiply numerator/denominator by a unit.
/// 
impl<R> HashableElRing for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain + IntegerRing + HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        let gcd = self.base_ring().ideal_gen(&el.den, &el.num);
        self.base_ring.get_ring().hash(&self.base_ring.checked_div(&el.num, &gcd).unwrap(), h);
        self.base_ring.get_ring().hash(&self.base_ring.checked_div(&el.den, &gcd).unwrap(), h);
    }
}

impl<R: RingStore> StrassenHint for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    default fn strassen_threshold(&self) -> usize {
        usize::MAX
    }
}

impl<R: RingStore> DefaultConvolutionRing for FractionFieldImplBase<R>
    where R: RingStore,
        R::Type: Domain
{
    fn create_default_convolution<'conv>(&self, _max_len_hint: Option<usize>) -> DynConvolution<'conv, Self>
        where Self: 'conv
    {
        Arc::new(TypeErasableConvolution::new(NaiveConvolution))
    }
}

impl<R: RingStore, S: RingStore> CanHomFrom<FractionFieldImplBase<S>> for FractionFieldImplBase<R>
    where R: RingStore,
        S: RingStore,
        R::Type: Domain + CanHomFrom<S::Type>,
        S::Type: Domain
{
    type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &FractionFieldImplBase<S>) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &FractionFieldImplBase<S>, el: <FractionFieldImplBase<S> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.from_fraction(self.base_ring().get_ring().map_in(from.base_ring().get_ring(), el.num, hom), self.base_ring().get_ring().map_in(from.base_ring().get_ring(), el.den, hom))
    }
}

impl<R: RingStore, S: RingStore> CanIsoFromTo<FractionFieldImplBase<S>> for FractionFieldImplBase<R>
    where R: RingStore,
        S: RingStore,
        R::Type: Domain + CanIsoFromTo<S::Type>,
        S::Type: Domain
{
    type Isomorphism = <R::Type as CanIsoFromTo<S::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &FractionFieldImplBase<S>) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &FractionFieldImplBase<S>, el: Self::Element, iso: &Self::Isomorphism) -> <FractionFieldImplBase<S> as RingBase>::Element {
        from.from_fraction(self.base_ring().get_ring().map_out(from.base_ring().get_ring(), el.num, iso), self.base_ring().get_ring().map_out(from.base_ring().get_ring(), el.den, iso))
    }
}

impl<R: RingStore, I: RingStore> CanHomFrom<RationalFieldBase<I>> for FractionFieldImplBase<R>
    where R: RingStore,
        I: RingStore,
        R::Type: Domain + CanHomFrom<I::Type>,
        I::Type: IntegerRing
{
    type Homomorphism = <R::Type as CanHomFrom<I::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &RationalFieldBase<I>) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &RationalFieldBase<I>, el: <RationalFieldBase<I> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let (num, den) = from.as_fraction(el);
        self.from_fraction(self.base_ring().get_ring().map_in(from.base_ring().get_ring(), num, hom), self.base_ring().get_ring().map_in(from.base_ring().get_ring(), den, hom))
    }
}

impl<R: RingStore, I: RingStore> CanIsoFromTo<RationalFieldBase<I>> for FractionFieldImplBase<R>
    where R: RingStore,
        I: RingStore,
        R::Type: Domain + CanIsoFromTo<I::Type>,
        I::Type: IntegerRing
{
    type Isomorphism = <R::Type as CanIsoFromTo<I::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &RationalFieldBase<I>) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &RationalFieldBase<I>, el: Self::Element, iso: &Self::Isomorphism) -> <RationalFieldBase<I> as RingBase>::Element {
        from.from_fraction(self.base_ring().get_ring().map_out(from.base_ring().get_ring(), el.num, iso), self.base_ring().get_ring().map_out(from.base_ring().get_ring(), el.den, iso))
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::iters::multi_cartesian_product;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_balance_factor() {
    LogAlgorithmSubscriber::init_test();
    let ring = FractionFieldImpl::new(StaticRing::<i64>::RING);
    let elements = [
        ring.from_fraction(2 * 11, 3),
        ring.from_fraction(6 * 11, 3),
        ring.from_fraction(3 * 11, 18),
        ring.from_fraction(6 * 11, 2),
        ring.from_fraction(2 * 11, 6),
        ring.from_fraction(5 * 11, 7),
        ring.from_fraction(0, 1),
        ring.from_fraction(0, 3),
        ring.from_fraction(0, 12),
        ring.from_fraction(0, 13)
    ];
    assert_el_eq!(&ring, ring.from_fraction(11, 7 * 6), ring.get_ring().balance_factor(elements.iter()).unwrap());
}

#[test]
fn test_ring_axioms() {
    LogAlgorithmSubscriber::init_test();
    let ring = FractionFieldImpl::new(StaticRing::<i64>::RING);
    let edge_case_elements = multi_cartesian_product([&[-3, -2, -1, 0, 1, 2, 3][..], &[1, 2, 3][..]].into_iter().map(|list| list.iter().copied()), |data| ring.from_fraction(data[0], data[1]), |_, x| *x);
    crate::ring::generic_tests::test_ring_axioms(&ring, edge_case_elements);
}