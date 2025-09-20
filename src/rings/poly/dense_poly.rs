use feanor_serde::newtype_struct::*;
use serde::de::DeserializeSeed;
use serde::{Deserializer, Serialize, Serializer};
use feanor_serde::seq::*;

use crate::algorithms::convolution::*;
use crate::algorithms::interpolate::interpolate;
use crate::algorithms::poly_div::fast_poly_div_rem;
use crate::algorithms::poly_gcd::PolyTFracGCDRing;
use crate::computation::{no_error, ComputationController, DontObserve};
use crate::reduce_lift::poly_eval::{EvalPolyLocallyRing, InterpolationBaseRing, ToExtRingMap};
use crate::divisibility::*;
use crate::integer::*;
use crate::pid::*;
use crate::field::Field;
use crate::specialization::{FiniteRingSpecializable, FiniteRingOperation};
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::algorithms;
use crate::rings::poly::*;
use crate::seq::*;
use crate::serialization::*;

use std::alloc::{Allocator, Global};
use std::fmt::Debug;
use std::cmp::{min, max};

///
/// The univariate polynomial ring `R[X]`. Polynomials are stored as dense vectors of
/// coefficients, allocated by the given memory provider.
/// 
/// If most of the coefficients will be zero, consider using [`sparse_poly::SparsePolyRingBase`]
/// instead for improved performance.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let x_plus_1 = P.add(P.indeterminate(), P.int_hom().map(1));
/// let binomial_coefficients = P.pow(x_plus_1, 10);
/// assert_eq!(10 * 9 * 8 * 7 * 6 / 120, *P.coefficient_at(&binomial_coefficients, 5));
/// ```
/// This ring has a [`CanIsoFromTo`] to [`sparse_poly::SparsePolyRingBase`].
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::poly::sparse_poly::*;
/// # use feanor_math::primitive_int::*;
/// let ZZ = StaticRing::<i32>::RING;
/// let P = DensePolyRing::new(ZZ, "X");
/// let P2 = SparsePolyRing::new(ZZ, "X");
/// let high_power_of_x = P.pow(P.indeterminate(), 10);
/// assert_el_eq!(P2, P2.pow(P2.indeterminate(), 10), &P.can_iso(&P2).unwrap().map(high_power_of_x));
/// ```
/// 
pub struct DensePolyRingBase<R: RingStore, A: Allocator + Clone = Global, C: ConvolutionAlgorithm<R::Type> = KaratsubaAlgorithm<Global>> {
    base_ring: R,
    unknown_name: &'static str,
    zero: El<R>,
    element_allocator: A,
    convolution_algorithm: C
}

impl<R: RingStore + Clone, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type> + Clone> Clone for DensePolyRingBase<R, A, C> {
    
    fn clone(&self) -> Self {
        DensePolyRingBase {
            base_ring: <R as Clone>::clone(&self.base_ring), 
            unknown_name: self.unknown_name, 
            zero: self.base_ring.zero() ,
            element_allocator: self.element_allocator.clone(),
            convolution_algorithm: self.convolution_algorithm.clone()
        }
    }
}

impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> Debug for DensePolyRingBase<R, A, C>
    where R::Type: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DensePolyRing")
            .field("base_ring", &self.base_ring.get_ring())
            .finish()
    }
}

///
/// The univariate polynomial ring `R[X]`, with polynomials being stored as dense vectors of coefficients.
/// For details, see [`DensePolyRingBase`].
/// 
pub type DensePolyRing<R, A = Global, C = KaratsubaAlgorithm> = RingValue<DensePolyRingBase<R, A, C>>;

impl<R: RingStore> DensePolyRing<R> {

    pub fn new(base_ring: R, unknown_name: &'static str) -> Self {
        Self::new_with_convolution(base_ring, unknown_name, Global, STANDARD_CONVOLUTION)
    }
}

impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> DensePolyRing<R, A, C> {

    #[stability::unstable(feature = "enable")]
    pub fn new_with_convolution(base_ring: R, unknown_name: &'static str, element_allocator: A, convolution_algorithm: C) -> Self {
        let zero = base_ring.zero();
        RingValue::from(DensePolyRingBase {
            base_ring, 
            unknown_name, 
            zero, 
            element_allocator,
            convolution_algorithm
        })
    }
}


impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> DensePolyRingBase<R, A, C> {
    
    #[stability::unstable(feature = "enable")]
    pub fn into_base_ring(self) -> R {
        self.base_ring
    }
}

///
/// An element of [`DensePolyRing`].
/// 
pub struct DensePolyRingEl<R: RingStore, A: Allocator + Clone = Global> {
    data: Vec<El<R>, A>
}

impl<R, A> Debug for DensePolyRingEl<R, A> 
    where R: RingStore,
        A: Allocator + Clone,
        El<R>: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> RingBase for DensePolyRingBase<R, A, C> {
    
    type Element = DensePolyRingEl<R, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut data = Vec::with_capacity_in(val.data.len(), self.element_allocator.clone());
        data.extend((0..val.data.len()).map(|i| self.base_ring.clone_el(&val.data.at(i))));
        DensePolyRingEl { data }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            self.base_ring.add_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        for i in min(lhs.data.len(), rhs.data.len())..rhs.data.len() {
            lhs.data.push(self.base_ring().clone_el(&rhs.data[i]));
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            self.base_ring.sub_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        for i in min(lhs.data.len(), rhs.data.len())..rhs.data.len() {
            lhs.data.push(self.base_ring().negate(self.base_ring().clone_el(&rhs.data[i])));
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        for i in 0..lhs.data.len() {
            self.base_ring.negate_inplace(&mut lhs.data[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        DensePolyRingEl {
            data: Vec::new_in(self.element_allocator.clone())
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        let mut result = self.zero();
        result.data.push(self.base_ring().get_ring().from_int(value)); 
        return result;
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        for i in 0..min(lhs.data.len(), rhs.data.len()) {
            if !self.base_ring.eq_el(&lhs.data[i], &rhs.data[i]) {
                return false;
            }
        }
        let longer = if lhs.data.len() > rhs.data.len() { lhs } else { rhs };
        for i in min(lhs.data.len(), rhs.data.len())..longer.data.len() {
            if !self.base_ring.is_zero(&longer.data[i]) {
                return false;
            }
        }
        return true;
    }

    fn is_commutative(&self) -> bool {
        self.base_ring.is_commutative()
    }

    fn is_noetherian(&self) -> bool {
        // by Hilbert's basis theorem
        self.base_ring.is_noetherian()
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        super::generic_impls::dbg_poly(self, value, out, self.unknown_name, env)
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn square(&self, value: &mut Self::Element) {
        *value = self.mul_ref(&value, &value);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let lhs_len = if self.base_ring().get_ring().is_approximate() { lhs.data.len() } else { self.degree(lhs).map(|i| i + 1).unwrap_or(0) };
        let rhs_len = if self.base_ring().get_ring().is_approximate() { rhs.data.len() } else { self.degree(rhs).map(|i| i + 1).unwrap_or(0) };
        let mut result = Vec::with_capacity_in(lhs_len + rhs_len, self.element_allocator.clone());
        result.extend((0..(lhs_len + rhs_len)).map(|_| self.base_ring().zero()));
        self.convolution_algorithm.compute_convolution(
            &lhs.data[0..lhs_len], 
            &rhs.data[0..rhs_len],
            &mut result[..],
            self.base_ring()
        );
        return DensePolyRingEl {
            data: result
        };
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        for i in 0..lhs.data.len() {
            self.base_ring().int_hom().mul_assign_map(lhs.data.at_mut(i), rhs);
        }
    }
    
    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)
    }
    
    fn prod<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        let mut elements = els.into_iter().collect::<Vec<_>>();
        if elements.len() == 0 {
            return self.one();
        }
        elements.sort_unstable_by_key(|f| self.degree(f).unwrap_or(0));
        // this can make it much faster; in particular, in the (not too uncommon) special case that we compute
        // the product of degree 1 polynomials, this means we actually make use of karatsuba multiplication
        for i in 0..StaticRing::<i64>::RING.abs_log2_ceil(&elements.len().try_into().unwrap()).unwrap() {
            let step = 1 << i;
            for j in (0..(elements.len() - step)).step_by(2 * step) {
                let (a, b) = (&mut elements[j..(j + step + 1)]).split_at_mut(step);
                self.mul_assign_ref(&mut a[0], &b[0]);
            }
        }
        return elements.into_iter().next().unwrap();
    }

    fn is_approximate(&self) -> bool {
        self.base_ring().get_ring().is_approximate()
    }
}

impl<R, A, C> PartialEq for DensePolyRingBase<R, A, C> 
    where R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

///
/// Marker trait to signal that for this polynomial ring `P`, we want to use the
/// default implementation of the potential isomorphism `P <-> DensePolyRing` when
/// applicable.
/// 
/// This is currently necessary, since we want to provide a specialized implementation
/// of `DensePolyRingBase<R1, A1>: CanHomFrom<DensePolyRingBase<R2, A2>>`, but we cannot
/// currently specialize on types that still have generic parameters.
/// 
pub trait ImplGenericCanIsoFromToMarker: PolyRing {}

impl<R> ImplGenericCanIsoFromToMarker for sparse_poly::SparsePolyRingBase<R> 
    where R: RingStore
{}

impl<R, P, A, C> CanHomFrom<P> for DensePolyRingBase<R, A, C> 
    where R: RingStore, R::Type: CanHomFrom<<P::BaseRing as RingStore>::Type>, P: ImplGenericCanIsoFromToMarker, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>
{
    type Homomorphism = super::generic_impls::Homomorphism<P, DensePolyRingBase<R, A, C>>;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom(from, self)
    }

    fn map_in(&self, from: &P, el: P::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in(from, self, el, hom)
    }
}

impl<R1, A1, R2, A2, C1, C2> CanHomFrom<DensePolyRingBase<R1, A1, C1>> for DensePolyRingBase<R2, A2, C2> 
    where R1: RingStore, A1: Allocator + Clone, C1: ConvolutionAlgorithm<R1::Type>,
        R2: RingStore, A2: Allocator + Clone, C2: ConvolutionAlgorithm<R2::Type>,
        R2::Type: CanHomFrom<R1::Type>
{
    type Homomorphism = <R2::Type as CanHomFrom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DensePolyRingBase<R1, A1, C1>) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in_ref(&self, from: &DensePolyRingBase<R1, A1, C1>, el: &DensePolyRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        RingRef::new(self).from_terms((0..el.data.len()).map(|i| (self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), &el.data[i], hom), i)))
    }

    fn map_in(&self, from: &DensePolyRingBase<R1, A1, C1>, el: DensePolyRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)    
    }
}

impl<R1, A1, R2, A2, C1, C2> CanIsoFromTo<DensePolyRingBase<R1, A1, C1>> for DensePolyRingBase<R2, A2, C2> 
    where R1: RingStore, A1: Allocator + Clone, C1: ConvolutionAlgorithm<R1::Type>, 
        R2: RingStore, A2: Allocator + Clone, C2: ConvolutionAlgorithm<R2::Type>,
        R2::Type: CanIsoFromTo<R1::Type>
{
    type Isomorphism = <R2::Type as CanIsoFromTo<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DensePolyRingBase<R1, A1, C1>) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &DensePolyRingBase<R1, A1, C1>, el: DensePolyRingEl<R2, A2>, hom: &Self::Isomorphism) -> DensePolyRingEl<R1, A1> {
        RingRef::new(from).from_terms((0..el.data.len()).map(|i| (self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), hom), i)))
    }
}

impl<R, P, A, C> CanIsoFromTo<P> for DensePolyRingBase<R, A, C> 
    where R: RingStore, R::Type: CanIsoFromTo<<P::BaseRing as RingStore>::Type>, P: ImplGenericCanIsoFromToMarker, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>
{
    type Isomorphism = super::generic_impls::Isomorphism<P, Self>;

    fn has_canonical_iso(&self, from: &P) -> Option<Self::Isomorphism> {
        self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
    }

    fn map_out(&self, from: &P, el: Self::Element, iso: &Self::Isomorphism) -> P::Element {
        super::generic_impls::map_out(from, self, el, iso)
    }
}

impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> RingExtension for DensePolyRingBase<R, A, C> {
    
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data.push(x);
        return result;
    }

    fn fma_base(&self, lhs: &Self::Element, rhs: &El<Self::BaseRing>, summand: Self::Element) -> Self::Element {
        let lhs_len = self.degree(lhs).map(|d| d + 1).unwrap_or(0);
        let summand_len = self.degree(&summand).map(|d| d + 1).unwrap_or(0);
        let mut result = Vec::with_capacity_in(max(lhs_len, summand_len), self.element_allocator.clone());
        let mut summand_it = summand.data.into_iter();
        result.extend(summand_it.by_ref().take(min(summand_len, lhs_len)).enumerate().map(|(i, x)| self.base_ring().fma(&lhs.data[i], rhs, x)));
        result.extend(summand_it.take(summand_len - min(summand_len, lhs_len)));
        result.extend((min(summand_len, lhs_len)..lhs_len).map(|i| self.base_ring().mul_ref(&lhs.data[i], rhs)));
        return DensePolyRingEl {
            data: result
        };
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        for c in &mut lhs.data {
            self.base_ring().mul_assign_ref(c, rhs)
        }
    }
}

///
/// Iterator over all terms of an element of [`DensePolyRing`].
/// 
#[allow(missing_debug_implementations)]
pub struct TermIterator<'a, R>
    where R: RingStore
{
    iter: std::iter::Rev<std::iter::Enumerate<std::slice::Iter<'a, El<R>>>>,
    ring: &'a R
}

impl<'a, R> Clone for TermIterator<'a, R>
    where R: RingStore
{
    fn clone(&self) -> Self {
        TermIterator {
            iter: self.iter.clone(),
            ring: self.ring
        }
    }
}

impl<'a, R> Iterator for TermIterator<'a, R>
    where R: RingStore
{
    type Item = (&'a El<R>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, c)) = self.iter.next() {
            if self.ring.get_ring().is_approximate() || !self.ring.is_zero(c) {
                return Some((c, i));
            }
        }
        return None;
    }
}

impl<R, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> HashableElRing for DensePolyRingBase<R, A, C> 
    where R: RingStore,
        R::Type: HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, el: &Self::Element, h: &mut H) {
        let len = self.degree(el).map(|d| d + 1).unwrap_or(0);
        h.write_length_prefix(len);
        for i in 0..len {
            self.base_ring().get_ring().hash(self.coefficient_at(el, i), h);
        }
    }
}

impl<R, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> SerializableElementRing for DensePolyRingBase<R, A, C> 
    where R: RingStore,
        R::Type: SerializableElementRing
{
    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("DensePoly", DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeWithRing::new(self.base_ring())),
            Vec::new_in(self.element_allocator.clone()),
            |mut current, next| { current.push(next); current }
        )).deserialize(deserializer).map(|data| DensePolyRingEl { data })
    }

    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let len = self.degree(el).map(|d| d + 1).unwrap_or(0);
        SerializableNewtypeStruct::new(
            "DensePoly", 
            SerializableSeq::new_with_len((0..len).map(|i| SerializeWithRing::new(self.coefficient_at(el, i), self.base_ring())), len)
        ).serialize(serializer)
    }
}

impl<R, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> PolyRing for DensePolyRingBase<R, A, C> 
    where R: RingStore
{
    type TermsIterator<'a> = TermIterator<'a, R>
        where Self: 'a;

    fn indeterminate(&self) -> Self::Element {
        let mut result = self.zero();
        result.data.extend([self.base_ring().zero(), self.base_ring().one()].into_iter());
        return result;
    }

    fn terms<'a>(&'a self, f: &'a Self::Element) -> TermIterator<'a, R> {
        TermIterator {
            iter: f.data.iter().enumerate().rev(), 
            ring: self.base_ring()
        }
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, usize)>
    {
        for (c, i) in rhs {
            if lhs.data.len() <= i {
                lhs.data.resize_with(i + 1, || self.base_ring().zero());
            }
            self.base_ring().add_assign(&mut lhs.data[i], c);
        }
    }

    fn truncate_monomials(&self, lhs: &mut Self::Element, truncated_at_degree: usize) {
        lhs.data.truncate(truncated_at_degree)
    }

    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing> {
        if i < f.data.len() {
            return &f.data[i];
        } else {
            return &self.zero;
        }
    }

    fn mul_assign_monomial(&self, lhs: &mut Self::Element, rhs_power: usize) {
        _ = lhs.data.splice(0..0, (0..rhs_power).map(|_| self.base_ring().zero()));
    }

    fn degree(&self, f: &Self::Element) -> Option<usize> {
        for i in (0..f.data.len()).rev() {
            if !self.base_ring().is_zero(&f.data[i]) {
                return Some(i);
            }
        }
        return None;
    }

    fn div_rem_monic(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(self.base_ring().is_one(self.coefficient_at(rhs, self.degree(rhs).unwrap())));
        let (quo, rem) = fast_poly_div_rem(RingRef::new(self), lhs, rhs, |x| Ok(self.base_ring().clone_el(x)), DontObserve).unwrap_or_else(no_error);
        return (quo, rem);
    }

    fn evaluate<S, H>(&self, f: &Self::Element, value: &S::Element, hom: H) -> S::Element
        where S: ?Sized + RingBase,
            H: Homomorphism<R::Type, S>
    {
        let d = if self.base_ring().get_ring().is_approximate() { f.data.len().saturating_sub(1) } else { self.degree(f).unwrap_or(0) };
        let mut current = hom.map_ref(self.coefficient_at(f, d));
        for i in (0..d).rev() {
            hom.codomain().mul_assign_ref(&mut current, value);
            hom.codomain().add_assign(&mut current, hom.map_ref(self.coefficient_at(f, i)));
        }
        return current;
    }
}
impl<R: RingStore, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> FiniteRingSpecializable for DensePolyRingBase<R, A, C> {

    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<R, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> EvalPolyLocallyRing for DensePolyRingBase<R, A, C> 
    where R: RingStore,
        R::Type: InterpolationBaseRing
{
    type LocalRing<'ring> = <R::Type as InterpolationBaseRing>::ExtendedRing<'ring>
        where Self: 'ring;

    type LocalRingBase<'ring> = <R::Type as InterpolationBaseRing>::ExtendedRingBase<'ring>
        where Self: 'ring;

    type LocalComputationData<'ring> = (ToExtRingMap<'ring, R::Type>, Vec<El<<R::Type as InterpolationBaseRing>::ExtendedRing<'ring>>>)
        where Self: 'ring;

    fn ln_pseudo_norm(&self, el: &Self::Element) -> f64 {
        if let Some(d) = self.degree(el) {
            return d as f64;
        } else {
            return 0.;
        }
    }

    fn local_computation<'ring>(&'ring self, ln_pseudo_norm: f64) -> Self::LocalComputationData<'ring> {
        let required_points = ln_pseudo_norm.ceil() as usize + 1;
        ToExtRingMap::for_interpolation(self.base_ring().get_ring(), required_points)
    }

    fn local_ring_count<'ring>(&self, data: &Self::LocalComputationData<'ring>) -> usize 
        where Self: 'ring
    {
        data.1.len()
    }

    fn local_ring_at<'ring>(&self, data: &Self::LocalComputationData<'ring>, i: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        assert!(i < self.local_ring_count(data));
        data.0.codomain().clone()
    }
        
    fn reduce<'ring>(&self, data: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
        where Self: 'ring
    {
        return data.1.iter().map(|x| self.evaluate(el, x, &data.0)).collect::<Vec<_>>();
    }

    fn lift_combine<'ring>(&self, data: &Self::LocalComputationData<'ring>, els: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring
    {
        let base_ring = RingRef::new(data.0.codomain().get_ring());
        let new_ring = DensePolyRing::new(base_ring, self.unknown_name);
        let result_in_extension = interpolate(&new_ring, (&data.1).into_clone_ring_els(data.0.codomain()), els.into_clone_ring_els(data.0.codomain()), &self.element_allocator).unwrap();
        return RingRef::new(self).from_terms(new_ring.terms(&result_in_extension).map(|(c, i)| (data.0.as_base_ring_el(base_ring.clone_el(c)), i)));
    }
}

impl<R, A: Allocator + Clone, C: ConvolutionAlgorithm<R::Type>> Domain for DensePolyRingBase<R, A, C> 
    where R: RingStore, R::Type: Domain
{}

impl<R, A: Allocator + Clone, C> DivisibilityRing for DensePolyRingBase<R, A, C> 
    where R: RingStore, 
        R::Type: DivisibilityRing + Domain, 
        C: ConvolutionAlgorithm<R::Type>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if let Some(d) = self.degree(rhs) {
            if d == 0 {
                let rhs = self.coefficient_at(rhs, 0);
                return RingRef::new(self).try_from_terms(self.terms(lhs).map(|(c, i)| self.base_ring().checked_left_div(c, rhs).map(|c| (c, i)).ok_or(()))).ok();
            } else {
                let lc = &rhs.data[d];
                let (quo, rem) = fast_poly_div_rem(RingRef::new(self), self.clone_el(lhs), rhs, |x| self.base_ring().checked_left_div(&x, lc).ok_or(()), DontObserve).ok()?;
                if self.is_zero(&rem) {
                    Some(quo)
                } else {
                    None
                }
            }
        } else if self.is_zero(lhs) {
            Some(self.zero())
        } else {
            None
        }
    }

    fn divides_left(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if let Some(d) = self.degree(rhs) {
            if d == 0 {
                true
            } else {
                let lc = &rhs.data[d];
                if let Ok((_, rem)) = fast_poly_div_rem(RingRef::new(self), self.clone_el(lhs), rhs, |x| self.base_ring().checked_left_div(&x, lc).ok_or(()), DontObserve) {
                    if self.is_zero(&rem) {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        } else if self.is_zero(lhs) {
            true
        } else {
            false
        }
    }

    fn balance_factor<'a, I>(&self, elements: I) -> Option<Self::Element>
        where I: Iterator<Item =  &'a Self::Element>,
            Self: 'a
    {
        self.base_ring().get_ring().balance_factor(elements.flat_map(|f| self.terms(f).map(|(c, _)| c))).map(|c| RingRef::new(self).inclusion().map(c))
    }
}

impl<R, A, C> PrincipalIdealRing for DensePolyRingBase<R, A, C>
    where A: Allocator + Clone, R: RingStore, R::Type: Field + PolyTFracGCDRing, C: ConvolutionAlgorithm<R::Type>
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        // base ring is a field, so everything is fine
        if self.is_zero(rhs) && self.is_zero(lhs) {
            return Some(self.one());
        } else if self.is_zero(rhs) {
            return None;
        }
        let (quo, rem) = self.euclidean_div_rem(self.clone_el(lhs), rhs);
        if self.is_zero(&rem) {
            return Some(quo);
        } else {
            return None;
        }
    }

    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        algorithms::eea::eea(self.clone_el(lhs), self.clone_el(rhs), &RingRef::new(self))
    }

    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        <_ as PolyTFracGCDRing>::gcd(RingRef::new(self), lhs, rhs)
    }

    fn ideal_gen_with_controller<Controller>(&self, lhs: &Self::Element, rhs: &Self::Element, controller: Controller) -> Self::Element
        where Controller: ComputationController
    {
        <_ as PolyTFracGCDRing>::gcd_with_controller(RingRef::new(self), lhs, rhs, controller)
    }
}

impl<R, A, C> EuclideanRing for DensePolyRingBase<R, A, C> 
    where A: Allocator + Clone, R: RingStore, R::Type: Field + PolyTFracGCDRing, C: ConvolutionAlgorithm<R::Type>
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        let lc_inv = self.base_ring.invert(&rhs.data[self.degree(rhs).unwrap()]).unwrap();
        let (quo, rem) = fast_poly_div_rem(RingRef::new(self), lhs, rhs, |x| Ok(self.base_ring().mul_ref(&x, &lc_inv)), DontObserve).unwrap_or_else(no_error);
        return (quo, rem);
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        return Some(self.degree(val).map(|x| x + 1).unwrap_or(0));
    }
}

#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::rings::zn::zn_static::{Zn, Fp};
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use super::sparse_poly::SparsePolyRing;
#[cfg(test)]
use crate::rings::extension::FreeAlgebraStore;
#[cfg(test)]
use crate::iters::multiset_combinations;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use std::time::Instant;
#[cfg(test)]
use crate::rings::approx_real::float::Real64;
#[cfg(test)]
use crate::ordered::OrderedRingStore;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::algorithms::cyclotomic::cyclotomic_polynomial;

#[cfg(test)]
fn edge_case_elements<P: PolyRingStore>(poly_ring: P) -> impl Iterator<Item = El<P>>
    where P::Type: PolyRing
{
    let base_ring = poly_ring.base_ring();
    vec![ 
        poly_ring.from_terms([].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0), (base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 0)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(-1), 0), (base_ring.int_hom().map(1), 1)].into_iter()),
        poly_ring.from_terms([(base_ring.int_hom().map(1), 0), (base_ring.int_hom().map(-1), 1)].into_iter())
    ].into_iter()
}

#[test]
fn test_prod() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    assert_el_eq!(&poly_ring, poly_ring.one(), poly_ring.prod([].into_iter()));

    let product = poly_ring.prod((0..10).map(|n| poly_ring.add(poly_ring.indeterminate(), poly_ring.inclusion().map(n))));
    assert_eq!(Some(10), poly_ring.degree(&product));
    for i in 0..10 {
        let expected = multiset_combinations(&[1; 10], 10 - i, |indices| {
            indices.iter().enumerate().filter(|(_, count)| **count > 0).map(|(n, _)| n).product::<usize>()
        }).sum::<usize>();
        assert_eq!(expected as i64, *poly_ring.coefficient_at(&product, i));
    }
}

#[test]
fn test_fma_base() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    let [f, g, h] = poly_ring.with_wrapped_indeterminate(|X| [X + 3, X.pow_ref(2) - 1, X.pow_ref(2) + 2 * X + 5]);
    assert_el_eq!(&poly_ring, h, poly_ring.get_ring().fma_base(&f, &2, g));

    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    let [f, g, h] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) + X, X.pow_ref(2) - 1, 2 * X.pow_ref(3) + X.pow_ref(2) + 2 * X - 1]);
    assert_el_eq!(&poly_ring, h, poly_ring.get_ring().fma_base(&f, &2, g));

    let poly_ring = DensePolyRing::new(zn_64::Zn::new(7), "X");
    let [f, g] = poly_ring.with_wrapped_indeterminate(|X| [3 * X.pow_ref(2) + 5, 5 * X.pow_ref(2) + 6]);
    assert_el_eq!(&poly_ring, g, poly_ring.get_ring().fma_base(&f, &poly_ring.base_ring().int_hom().map(4), poly_ring.zero()));
}

#[test]
fn test_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    crate::ring::generic_tests::test_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_hash_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    crate::ring::generic_tests::test_hash_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_poly_ring_axioms() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    super::generic_tests::test_poly_ring_axioms(poly_ring, Zn::<7>::RING.elements());
}

#[test]
fn test_divisibility_ring_axioms() {
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    crate::divisibility::generic_tests::test_divisibility_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_euclidean_ring_axioms() {
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    crate::pid::generic_tests::test_euclidean_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_principal_ideal_ring_axioms() {
    let poly_ring = DensePolyRing::new(Fp::<7>::RING, "X");
    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_canonical_iso_axioms_different_base_ring() {
    let poly_ring1 = DensePolyRing::new(zn_big::Zn::new(StaticRing::<i128>::RING, 7), "X");
    let poly_ring2 = DensePolyRing::new(zn_64::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}

#[test]
fn test_canonical_iso_sparse_poly_ring() {
    let poly_ring1 = SparsePolyRing::new(zn_64::Zn::new(7), "X");
    let poly_ring2 = DensePolyRing::new(zn_64::Zn::new(7), "X");
    crate::ring::generic_tests::test_hom_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
    crate::ring::generic_tests::test_iso_axioms(&poly_ring1, &poly_ring2, edge_case_elements(&poly_ring1));
}

#[test]
fn test_print() {
    let base_poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let poly_ring = DensePolyRing::new(&base_poly_ring, "Y");

    let poly = poly_ring.from_terms([
        (base_poly_ring.from_terms([(1, 0), (2, 2)].into_iter()), 0),
        (base_poly_ring.from_terms([(3, 0), (4, 2)].into_iter()), 2)
    ].into_iter());
    assert_eq!("(4X^2 + 3)Y^2 + 2X^2 + 1", format!("{}", poly_ring.format(&poly)));

    let poly = poly_ring.from_terms([
        (base_poly_ring.zero(), 0),
        (base_poly_ring.from_terms([(4, 2)].into_iter()), 2)
    ].into_iter());
    assert_eq!("4X^2Y^2", format!("{}", poly_ring.format(&poly)));
}

#[test]
#[ignore]
fn test_expensive_prod() {
    let ring = GaloisField::new(17, 2048);
    let poly_ring = DensePolyRing::new(&ring, "X");
    let mut rng = oorandom::Rand64::new(1);
    let a = ring.random_element(|| rng.rand_u64());

    let start = Instant::now();
    let product = poly_ring.prod(
        (0..2048).scan(ring.clone_el(&a), |current, _| {
            let result = poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map_ref(&current));
            *current = ring.pow(std::mem::replace(current, ring.zero()), 17);
            return Some(result);
        })
    );
    let end = Instant::now();

    println!("Computed product in {} ms", (end - start).as_millis());
    for i in 0..2048 {
        let coeff_wrt_basis = ring.wrt_canonical_basis(poly_ring.coefficient_at(&product, i));
        assert!((1..2028).all(|j| ring.base_ring().is_zero(&coeff_wrt_basis.at(j))));
    }
}

#[test]
fn test_serialize() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    crate::serialization::generic_tests::test_serialization(&poly_ring, edge_case_elements(&poly_ring));
}

#[test]
fn test_evaluate_approximate_ring() {
    let ring = DensePolyRing::new(Real64::RING, "X");
    let [f] = ring.with_wrapped_indeterminate(|X| [X * X * X - X + 1]);
    let x = 0.47312;
    assert!(Real64::RING.abs((x * x * x - x + 1.) - ring.evaluate(&f, &x, &Real64::RING.identity())) <= 0.000000001);
}

#[bench]
fn bench_div_rem_monic(bencher: &mut Bencher) {
    let ZZ = BigIntRing::RING;
    let ring = DensePolyRing::new(ZZ, "X");
    let phi_n = 30 * 40;
    let n = 31 * 41;
    let cyclotomic_poly = cyclotomic_polynomial(&ring, n);
    assert!(ring.degree(&cyclotomic_poly).unwrap() == phi_n);
    bencher.iter(|| {
        let mut current = ring.pow(ring.indeterminate(), phi_n - 1);
        for _ in phi_n..=n {
            ring.mul_assign_monomial(&mut current, 1);
            current = ring.div_rem_monic(current, &cyclotomic_poly).1;
        }
        assert_el_eq!(&ring, &ring.one(), &current);
    });
}
