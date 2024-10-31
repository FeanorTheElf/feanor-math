
use std::alloc::Allocator;
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use factor::heuristic_factor_poly_local;
use gcd::poly_gcd_local;
use sparse::SparseMapVector;
use squarefree_part::poly_power_decomposition_local;

use crate::computation::LogProgress;
use crate::impl_interpolation_base_ring_char_zero;
use crate::compute_locally::InterpolationBaseRing;
use crate::specialization::*;
use crate::algorithms::convolution::*;
use crate::algorithms::eea::signed_lcm;
use crate::algorithms::poly_factor::extension::poly_factor_extension;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::matmul::StrassenHint;
use crate::computation::DontObserve;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::delegate::DelegateRing;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::field::AsField;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::divisibility::*;
use crate::rings::extension::*;
use crate::Never;

use super::extension_impl::FreeAlgebraImpl;
use super::Field;
use super::FreeAlgebra;
use self::implementations_for_nested_number_fields::NumberFieldOrQQ;

///
/// An algebraic number field, i.e. a finite rank field extension of the rationals.
/// 
/// Note that the design of this type is different (and more complicated) than the one
/// for Galois fields (see [`crate::rings::extension::galois_field::GaloisFieldBase`]),
/// mainly I decided that Galois fields are always a single extension of a prime field,
/// while number fields should be representable as extensions of smaller number fields.
/// 
/// # Choice of blanket implementations of [`CanHomFrom`]
/// 
/// This is done analogously to [`crate::rings::extension::galois_field::GaloisFieldBase`], see
/// the description there.
/// 
#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    base: Impl,
}

#[stability::unstable(feature = "enable")]
pub type DefaultNumberFieldImpl = AsField<FreeAlgebraImpl<RationalField<BigIntRing>, Vec<El<RationalField<BigIntRing>>>, Global, KaratsubaAlgorithm>>;
#[stability::unstable(feature = "enable")]
pub type NumberField<Impl = DefaultNumberFieldImpl> = RingValue<NumberFieldBase<Impl>>;

impl<Impl> NumberField<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    #[stability::unstable(feature = "enable")]
    pub fn create(implementation: Impl) -> Self {
        RingValue::from(NumberFieldBase { base: implementation })
    }
}

impl NumberField {

    #[stability::unstable(feature = "enable")]
    pub fn new<P>(poly_ring: P, generating_poly: &El<P>) -> Self
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<BigIntRing>>
    {
        let rank = poly_ring.degree(generating_poly).unwrap();
        let neg_inv_lc = poly_ring.base_ring().negate(poly_ring.base_ring().invert(poly_ring.lc(generating_poly).unwrap()).unwrap());
        let modulus = (0..rank).map(|i| poly_ring.base_ring().mul_ref(poly_ring.coefficient_at(generating_poly, i), &neg_inv_lc)).collect::<Vec<_>>();
        return Self::create(FreeAlgebraImpl::new_with(RingValue::from(poly_ring.base_ring().get_ring().clone()), rank, modulus, "θ", Global, STANDARD_CONVOLUTION).as_field().ok().unwrap());
    }
}

impl<Impl, R> NumberField<AsField<FreeAlgebraImpl<R, Vec<El<R>>>>>
    where R: RingStore<Type = NumberFieldBase<Impl>> + Clone,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + FactorPolyField
{
    #[stability::unstable(feature = "enable")]
    pub fn new_over<P>(poly_ring: P, generating_poly: &El<P>) -> Self
        where P: RingStore,
            P::Type: PolyRing<BaseRing = R>
    {
        let rank = poly_ring.degree(generating_poly).unwrap();
        let neg_inv_lc = poly_ring.base_ring().negate(poly_ring.base_ring().invert(poly_ring.lc(generating_poly).unwrap()).unwrap());
        let modulus = (0..rank).map(|i| poly_ring.base_ring().mul_ref(poly_ring.coefficient_at(generating_poly, i), &neg_inv_lc)).collect::<Vec<_>>();
        return Self::create(FreeAlgebraImpl::new_with(poly_ring.base_ring().clone(), rank, modulus, "θ", Global, STANDARD_CONVOLUTION).as_field().ok().unwrap());
    }
}

impl<Impl> NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn scale_poly_to_order<'ring, 'a, P1, P2>(&self, from: P1, to: P2, poly: &El<P1>) -> El<P2>
        where P1: RingStore,
            P1::Type: PolyRing,
            <P1::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            P2: RingStore,
            P2::Type: PolyRing<BaseRing = &'a <Self as NumberFieldOrQQ>::ThisFieldByOrderStore<'ring>>,
            Self: 'ring,
            'ring: 'a
    {
        debug_assert!(self == from.base_ring().get_ring());
        let den = self.from_integer(from.terms(poly).map(|(c, _)| self.denominator_lcm(c)).fold(self.integer_ring().one(), |a, b| signed_lcm(a, b, self.integer_ring())));
        debug_assert!(!self.is_zero(&den));
        return to.from_terms(from.terms(poly).map(|(c, i)| (self.integral_element_to_number_field_by_order(*to.base_ring(), self.mul_ref(c, &den)), i)));
    }
}

impl<Impl> Clone for NumberFieldBase<Impl> 
    where Self: Never,
        Impl: RingStore + Clone,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<Impl> Copy for NumberFieldBase<Impl> 
    where Self: Never,
        Impl: RingStore + Copy,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        El<Impl>: Copy
{}

impl<Impl> PartialEq for NumberFieldBase<Impl> 
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<Impl> DelegateRing for NumberFieldBase<Impl> 
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    type Base = Impl::Type;
    type Element = El<Impl>;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<Impl> PrincipalIdealRing for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }

    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div(lhs, rhs)
    }
}

impl<Impl> EuclideanRing for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.div(&lhs, &rhs), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }
}

impl<Impl> FiniteRingSpecializable for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
    }
}

impl<Impl> Field for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{}

impl<Impl> PerfectField for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{}

impl<Impl> Domain for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{}

impl_interpolation_base_ring_char_zero!{ <{Impl}> InterpolationBaseRing for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
}

impl<Impl> StrassenHint for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn strassen_threshold(&self) -> usize {
        self.base.get_ring().strassen_threshold()
    }
}

impl<Impl> KaratsubaHint for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn karatsuba_threshold(&self) -> usize {
        self.base.get_ring().karatsuba_threshold()
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, S> CanHomFrom<S> for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra + CanHomFrom<S>,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        S: ?Sized + RingBase
{
    type Homomorphism = <Impl::Type as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: <S as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.base.get_ring().map_in(from, el, hom))
    }
}

impl<Impl> PolyGCDRing for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let self_ = poly_ring.base_ring();
        let order = self_.get_ring().construct_number_field_by_order();

        let order_poly_ring = DensePolyRing::new(&order, "X");
        let lhs_order = self_.get_ring().scale_poly_to_order(&poly_ring, &order_poly_ring, lhs);
        let rhs_order = self_.get_ring().scale_poly_to_order(&poly_ring, &order_poly_ring, rhs);

        let result_order = poly_gcd_local(&order_poly_ring, lhs_order, rhs_order, LogProgress);
        let result = poly_ring.from_terms(order_poly_ring.terms(&result_order).map(|(c, i)| (self_.get_ring().element_from_field_by_order(&order, order.clone_el(c)), i)));
        return poly_ring.normalize(result);
    }

    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self> 
    {
        let self_ = poly_ring.base_ring();
        let order = self_.get_ring().construct_number_field_by_order();
        let order_poly_ring = DensePolyRing::new(&order, "X");
        let poly_order = self_.get_ring().scale_poly_to_order(&poly_ring, &order_poly_ring, &poly);

        let result_order = poly_power_decomposition_local(&order_poly_ring, poly_order, LogProgress);
        let map_back = |f| poly_ring.from_terms(order_poly_ring.terms(&f).map(|(c, i)| (self_.get_ring().element_from_field_by_order(&order, order.clone_el(c)), i)));
        return result_order.into_iter().map(|(f, k)| (poly_ring.normalize(map_back(f)), k)).collect::<Vec<_>>();
    }
}

impl<Impl> FactorPolyField for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + FactorPolyField
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let self_ = poly_ring.base_ring();
        let order = self_.get_ring().construct_number_field_by_order();
        let order_poly_ring = DensePolyRing::new(&order, "X");
        let poly_order = self_.get_ring().scale_poly_to_order(&poly_ring, &order_poly_ring, poly);

        let map_back = |f| poly_ring.from_terms(order_poly_ring.terms(&f).map(|(c, i)| (self_.get_ring().element_from_field_by_order(&order, order.clone_el(c)), i)));

        let mut result = Vec::new();
        for (factor, e1) in heuristic_factor_poly_local(&order_poly_ring, poly_order, 1., LogProgress) {
            for (irred_factor, e2) in poly_factor_extension(&poly_ring, &map_back(factor)).0 {
                result.push((irred_factor, e1 * e2));
            }
        }
        return (result, self_.clone_el(poly_ring.lc(poly).unwrap()));
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, S> CanIsoFromTo<S> for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra + CanIsoFromTo<S>,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        S: ?Sized + RingBase
{
    type Isomorphism = <Impl::Type as CanIsoFromTo<S>>::Isomorphism;

    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism> {
        self.base.get_ring().has_canonical_iso(from)
    }

    fn map_out(&self, from: &S, el: Self::Element, iso: &Self::Isomorphism) -> <S as RingBase>::Element {
        self.base.get_ring().map_out(from, self.delegate(el), iso)
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, R, A, V, C> CanHomFrom<NumberFieldBase<Impl>> for FreeAlgebraImplBase<R, V, A, C>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        R: RingStore,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        FreeAlgebraImplBase<R, V, A, C>: CanHomFrom<Impl::Type>
{
    type Homomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanHomFrom<Impl::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &NumberFieldBase<Impl>) -> Option<Self::Homomorphism> {
        self.has_canonical_hom(from.base.get_ring())
    }

    fn map_in(&self, from: &NumberFieldBase<Impl>, el: <NumberFieldBase<Impl> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(from.base.get_ring(), from.delegate(el), hom)
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, R, A, V, C> CanHomFrom<NumberFieldBase<Impl>> for AsFieldBase<FreeAlgebraImpl<R, V, A, C>>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        R: RingStore,
        R::Type: LinSolveRing,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        FreeAlgebraImplBase<R, V, A, C>: CanHomFrom<Impl::Type>
{
    type Homomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanHomFrom<Impl::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &NumberFieldBase<Impl>) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from.base.get_ring())
    }

    fn map_in(&self, from: &NumberFieldBase<Impl>, el: <NumberFieldBase<Impl> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.rev_delegate(self.get_delegate().map_in(from.base.get_ring(), from.delegate(el), hom))
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, R, A, V, C> CanIsoFromTo<NumberFieldBase<Impl>> for FreeAlgebraImplBase<R, V, A, C>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        R: RingStore,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        FreeAlgebraImplBase<R, V, A, C>: CanIsoFromTo<Impl::Type>
{
    type Isomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanIsoFromTo<Impl::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &NumberFieldBase<Impl>) -> Option<Self::Isomorphism> {
        self.has_canonical_iso(from.base.get_ring())
    }

    fn map_out(&self, from: &NumberFieldBase<Impl>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <NumberFieldBase<Impl> as RingBase>::Element {
        from.rev_delegate(self.map_out(from.base.get_ring(), el, iso))
    }
}

///
/// For the rationale which blanket implementations I chose, see the [`NumberFieldBase`].
/// 
impl<Impl, R, A, V, C> CanIsoFromTo<NumberFieldBase<Impl>> for AsFieldBase<FreeAlgebraImpl<R, V, A, C>>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ,
        R: RingStore,
        R::Type: LinSolveRing,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        FreeAlgebraImplBase<R, V, A, C>: CanIsoFromTo<Impl::Type>
{
    type Isomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanIsoFromTo<Impl::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &NumberFieldBase<Impl>) -> Option<Self::Isomorphism> {
        self.get_delegate().has_canonical_iso(from.base.get_ring())
    }

    fn map_out(&self, from: &NumberFieldBase<Impl>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <NumberFieldBase<Impl> as RingBase>::Element {
        from.rev_delegate(self.get_delegate().map_out(from.base.get_ring(), self.delegate(el), iso))
    }
}

///
/// I'm really sorry, this is probably the most unreadable code in feanor-math. 
/// Basically, I am implementing a type-level recursion, such that [`PolyGCDLocallyDomain`]
/// is implemented for all number fields, even if we nest them arbitrarily many times.
/// 
/// In other words, we want
/// ```ignore
/// number_field_by_order::NumberFieldByOrder<
///     number_field_by_order::NumberFieldByOrder< 
///         ... <
///             number_field_by_order::NumberFieldByOrder<
///                 number_field_by_order::RationalsByZZ<I>
///             >
///         >
///     >
/// >: PolyGCDLocallyDomain
/// ```
/// no matter how deep the nesting goes.
/// 
/// Note that we would like to implement [`PolyGCDLocallyDomain`] only for orders, and not for
/// number fields. However, as specified currently, this not possible, since we need to perform 
/// rational reconstruction when lifting elements - and thus don't get values in the order anymore.
/// 
/// Instead, we could implement [`PolyGCDLocallyDomain`] for the number field itself. This
/// works, but it seems not to be a good idea to expose this interface, since we usually get
/// much better results, if we run the local algorithms on polynomials that first have been
/// scaled into an order (even if the result is not in the order anymore).
/// 
/// The solution is now to implement [`PolyGCDLocallyDomain`] for a newtype, and call it from
/// the [`PolyGCDRing`] implementation for number fields.
/// 
mod implementations_for_nested_number_fields {

    use gcd_locally::IdealDisplayWrapper;

    use crate::algorithms::poly_gcd::gcd_locally::{IntegerPolyGCDRing, ReductionMap};
    use crate::impl_interpolation_base_ring_char_zero;
    use crate::rings::zn::*;

    use crate::algorithms::linsolve::SolveResult;
    use crate::algorithms::poly_gcd::gcd_locally::PolyGCDLocallyDomain;
    use crate::delegate::DelegateRingImplEuclideanRing;
    use crate::matrix::{AsPointerToSlice, SubmatrixMut};
    use crate::specialization::{FiniteRingOperation, FiniteRingSpecializable};

    use super::*;
    
    #[stability::unstable(feature = "enable")]
    pub trait NumberFieldOrQQ: RingBase + PolyGCDRing + FiniteRingSpecializable + InterpolationBaseRing + PerfectField {

        type ThisFieldByOrder<'ring>: PolyGCDLocallyDomain + NumberFieldOrQQ + LinSolveRing
            where Self: 'ring;

        type ThisFieldByOrderStore<'ring>: RingStore<Type = Self::ThisFieldByOrder<'ring>>
            where Self: 'ring;

        type IntegerRingBase: ?Sized + IntegerRing;

        type Integers: RingStore<Type = Self::IntegerRingBase>;

        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring;

        fn integral_element_to_number_field_by_order<'ring>(&self, to: &Self::ThisFieldByOrderStore<'ring>, x: Self::Element) -> El<Self::ThisFieldByOrderStore<'ring>>
            where Self: 'ring;

        fn element_from_field_by_order<'ring>(&self, from: &Self::ThisFieldByOrderStore<'ring>, x: El<Self::ThisFieldByOrderStore<'ring>>) -> Self::Element
            where Self: 'ring;
            
        fn ln_heuristic_size(&self, c: &Self::Element) -> f64;

        fn integer_ring(&self) -> &Self::Integers;

        fn denominator_lcm(&self, c: &Self::Element) -> El<Self::Integers>;

        fn from_integer(&self, x: El<Self::Integers>) -> Self::Element;

        fn absolute_rank(&self) -> usize;
    }

    impl<I> NumberFieldOrQQ for RationalFieldBase<I>
        where I: RingStore,
            I::Type: IntegerRing
    {
        type ThisFieldByOrder<'ring> = RationalsByZZ<&'ring I>
            where Self: 'ring;

        type ThisFieldByOrderStore<'ring> = RingValue<RationalsByZZ<&'ring I>>
            where Self: 'ring;

        type IntegerRingBase = I::Type;

        type Integers = I;

        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            RingValue::from(RationalsByZZ { base: RationalField::new(self.base_ring()) })
        }

        fn integral_element_to_number_field_by_order<'ring>(&self, to: &Self::ThisFieldByOrderStore<'ring>, x: Self::Element) -> El<Self::ThisFieldByOrderStore<'ring>>
            where Self: 'ring
        {
            debug_assert!(self.base_ring().is_one(self.den(&x)));
            return to.get_ring().base.coerce(&RingRef::new(self), x);
        }

        fn element_from_field_by_order<'ring>(&self, from: &Self::ThisFieldByOrderStore<'ring>, x: El<Self::ThisFieldByOrderStore<'ring>>) -> Self::Element
            where Self: 'ring
        {
            RingRef::new(self).coerce(&from.get_ring().base, x)
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            (self.base_ring().abs_log2_ceil(self.num(c)).unwrap_or(0) + self.base_ring().abs_log2_ceil(self.den(c)).unwrap()) as f64 * 2f64.ln()
        }

        fn denominator_lcm(&self, c: &Self::Element) -> El<Self::Integers> {
            self.base_ring().clone_el(self.den(c))
        }

        fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
            self.from(x)
        }

        fn integer_ring(&self) ->  &Self::Integers {
            self.base_ring()
        }

        fn absolute_rank(&self) -> usize {
            1
        }
    }

    type BaseFieldByOrder<'ring, Impl> = <<<<Impl as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::ThisFieldByOrderStore<'ring>;

    impl<Impl> NumberFieldOrQQ for NumberFieldBase<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {
        type ThisFieldByOrder<'ring> = NumberFieldByOrder<AsField<FreeAlgebraImpl<BaseFieldByOrder<'ring, Impl>, Vec<El<BaseFieldByOrder<'ring, Impl>>>>>>
            where Self: 'ring;
    
        type ThisFieldByOrderStore<'ring> = RingValue<Self::ThisFieldByOrder<'ring>>
            where Self: 'ring;
    
        type IntegerRingBase = <<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::IntegerRingBase;
    
        type Integers = <<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::Integers;
    
        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            let self_ref = RingRef::new(self);
            let poly_ring = DensePolyRing::new(self.base_ring(), "X");
            let gen_poly = self_ref.generating_poly(&poly_ring, self.base_ring().identity());
            let denominator = poly_ring.terms(&gen_poly).map(|(c, _)| self.base_ring().get_ring().denominator_lcm(c)).fold(self.integer_ring().one(), |a, b| signed_lcm(a, b, self.integer_ring()));

            let base_field_by_order = self.base_ring().get_ring().construct_number_field_by_order();
            let new_x_pow_rank = (0..self.rank()).map(|i| base_field_by_order.negate(self.base_ring().get_ring().integral_element_to_number_field_by_order(
                &base_field_by_order,
                self.base_ring().mul_ref_fst(
                    poly_ring.coefficient_at(&gen_poly, i), 
                    self.base_ring().get_ring().from_integer(self.integer_ring().clone_el(&denominator))
                )
            ))).collect::<Vec<_>>();
            return RingValue::from(NumberFieldByOrder {
                base: NumberField::create(
                    AsField::from(AsFieldBase::promise_is_perfect_field(
                        FreeAlgebraImpl::new(base_field_by_order, self.rank(), new_x_pow_rank)
                    ))
                )
            });
        }
    
        fn integral_element_to_number_field_by_order<'ring>(&self, to: &Self::ThisFieldByOrderStore<'ring>, x: Self::Element) -> El<Self::ThisFieldByOrderStore<'ring>>
            where Self: 'ring
        {
            let result = to.from_canonical_basis(self.wrt_canonical_basis(&x).iter().map(|c| self.base_ring().get_ring().integral_element_to_number_field_by_order(to.base_ring(), c)));
            return result;
        }

        fn element_from_field_by_order<'ring>(&self, from: &Self::ThisFieldByOrderStore<'ring>, x: El<Self::ThisFieldByOrderStore<'ring>>) -> Self::Element
            where Self: 'ring
        {
            self.from_canonical_basis(from.wrt_canonical_basis(&x).iter().map(|c| self.base_ring().get_ring().element_from_field_by_order(from.base_ring(), c)))
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            self.wrt_canonical_basis(c).iter().map(|c| self.base_ring().get_ring().ln_heuristic_size(&c)).max_by(f64::total_cmp).unwrap() + (self.rank() as f64).ln()
        }
    
        fn denominator_lcm(&self, c: &Self::Element) -> El<Self::Integers> {
            self.wrt_canonical_basis(c).iter().fold(self.integer_ring().one(), |a, b| {
                signed_lcm(a, self.base_ring().get_ring().denominator_lcm(&b), self.integer_ring())
            })
        }
    
        fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
            self.from(self.base_ring().get_ring().from_integer(x))
        }
    
        fn integer_ring(&self) ->  &Self::Integers {
            self.base_ring().get_ring().integer_ring()
        }

        fn absolute_rank(&self) -> usize {
            self.rank() * self.base_ring().get_ring().absolute_rank()
        }
    }
    
    ///
    /// A wrapper around the rational numbers that implements [`PolyGCDLocallyDomain`].
    /// While [`PolyGCDLocallyDomain`] allows for rational reconstruction and thus indeed,
    /// the rationals can be considered a [`PolyGCDLocallyDomain`], the resulting 
    /// implementations are far from optimal. 
    /// 
    /// Thus, this should not be used as a general way to get polynomial operations over
    /// the rationals, but should only be called if the numbers are actually integers.
    /// Hence, it is "locked away" and only used during the rational reconstruction of
    /// polynomial factors over number fields.
    /// 
    #[stability::unstable(feature = "enable")]
    pub struct RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        pub(super) base: RationalField<I>
    }

    impl<I> NumberFieldOrQQ for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        type ThisFieldByOrder<'ring> = Self
            where Self: 'ring;

        type ThisFieldByOrderStore<'ring> = &'ring RingValue<Self>
            where Self: 'ring;

        type IntegerRingBase = I::Type;
    
        type Integers = I;
        
        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            // while we could return `RingValue::from_ref(self)`, this would probably just let an infinite recursion go unchecked
            unreachable!()
        }

        fn integral_element_to_number_field_by_order<'ring>(&self, _to: &Self::ThisFieldByOrderStore<'ring>, _x: Self::Element) -> El<Self::ThisFieldByOrderStore<'ring>>
            where Self: 'ring
        {
            unreachable!()
        }

        fn element_from_field_by_order<'ring>(&self, _from: &Self::ThisFieldByOrderStore<'ring>, _x: El<Self::ThisFieldByOrderStore<'ring>>) -> Self::Element
            where Self: 'ring
        {
            unreachable!()
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            self.base.get_ring().ln_heuristic_size(c)
        }
        
        fn denominator_lcm(&self, c: &Self::Element) -> El<Self::Integers> {
            self.base.get_ring().denominator_lcm(c)
        }
    
        fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
            self.base.get_ring().from_integer(x)
        }
    
        fn integer_ring(&self) ->  &Self::Integers {
            self.base.get_ring().integer_ring()
        }

        fn absolute_rank(&self) -> usize {
            self.base.get_ring().absolute_rank()
        }
    }

    impl<I> DelegateRingImplEuclideanRing for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {}

    impl<I> PartialEq for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        fn eq(&self, other: &Self) -> bool {
            self.base.get_ring() == other.base.get_ring()
        }
    }

    impl<I> DelegateRing for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        type Base = RationalFieldBase<I>;
        type Element = El<RationalField<I>>;

        fn get_delegate(&self) -> &Self::Base {
            self.base.get_ring()
        }

        fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
        fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
        fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
        fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
    }

    impl<I> Domain for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {}

    impl<I> Field for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {}

    impl<I> PerfectField for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {}

    impl<I> FiniteRingSpecializable for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
            Err(())
        }
    }

    impl_interpolation_base_ring_char_zero!{ <{I}> InterpolationBaseRing for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    }

    impl<I> LinSolveRing for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
            where V1: AsPointerToSlice<Self::Element>,
                V2: AsPointerToSlice<Self::Element>,
                V3: AsPointerToSlice<Self::Element>,
                A: Allocator 
        {
            self.base.solve_right_with(lhs, rhs, out, allocator)
        }
    }

    impl<I> PolyGCDLocallyDomain for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        type LocalRingBase<'ring> = <I::Type as PolyGCDLocallyDomain>::LocalRingBase<'ring>
            where Self: 'ring;

        type LocalRing<'ring> = <I::Type as PolyGCDLocallyDomain>::LocalRing<'ring>
            where Self: 'ring;

        type LocalFieldBase<'ring> = <I::Type as PolyGCDLocallyDomain>::LocalFieldBase<'ring>
            where Self: 'ring;

        type LocalField<'ring> = <I::Type as PolyGCDLocallyDomain>::LocalField<'ring>
            where Self: 'ring;

        type MaximalIdeal<'ring> = <I::Type as PolyGCDLocallyDomain>::MaximalIdeal<'ring>
            where Self: 'ring;

        fn heuristic_exponent<'ring, 'a, J>(&self, p: &Self::MaximalIdeal<'ring>, poly_deg: usize, coefficients: J) -> usize
            where J: Iterator<Item = &'a Self::Element>,
                Self: 'a,
                Self: 'ring
        {
            self.base.base_ring().get_ring().heuristic_exponent(p, poly_deg, coefficients.flat_map(move |c| [self.base.num(c), self.base.den(c)].into_iter()))
        }

        fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
            where F: FnMut() -> u64
        {
            self.base.base_ring().get_ring().random_maximal_ideal(rng)
        }

        fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
            where Self: 'ring
        {
            self.base.base_ring().get_ring().local_field_at(p)
        }

        fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
            where Self: 'ring
        {
            self.base.base_ring().get_ring().local_ring_at(p, e)
        }

        fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            debug_assert!(self.base.base_ring().is_one(self.base.den(&x)));
            self.base.base_ring().get_ring().reduce_ring_el(p, to, self.base.base_ring().clone_el(self.base.num(&x)))
        }

        fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            self.base.base_ring().get_ring().reduce_partial(p, from, to, x)
        }

        fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            self.base.base_ring().get_ring().lift_partial(p, from, to, x)
        }

        fn reconstruct_ring_el<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
            where Self: 'ring
        {
            let from_as_zn = <_ as IntegerPolyGCDRing>::local_ring_as_zn(self.base.base_ring().get_ring(), from.0);
            let iso = from_as_zn.can_iso(from.0).unwrap();
            let (n, d) = crate::algorithms::rational_reconstruction::rational_reconstruction(<_ as IntegerPolyGCDRing>::local_ring_as_zn(self.base.base_ring().get_ring(), from.0), iso.inv().map(x));
            return self.base.div(&self.base.inclusion().map(int_cast(n, self.base.base_ring(), from_as_zn.integer_ring())), &self.base.inclusion().map(int_cast(d, self.base.base_ring(), from_as_zn.integer_ring())));
        }

        fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
            where Self: 'ring
        {
            self.base.base_ring().get_ring().dbg_maximal_ideal(p, out)
        }
    }

    ///
    /// A wrapper around the rational numbers that implements [`PolyGCDLocallyDomain`].
    /// It is required that the underlying number field is generated by an integral 
    /// polynomial, and thus also gives rise to an order in the number field that can
    /// be used for local computations.
    /// 
    /// While [`PolyGCDLocallyDomain`] allows for rational reconstruction and thus indeed,
    /// a number field can be considered a [`PolyGCDLocallyDomain`], the resulting 
    /// implementations are far from optimal. 
    /// 
    /// Thus, this should not be used as a general way to get polynomial operations over
    /// the rationals, but should only be called if the numbers are actually integers.
    /// Hence, it is "locked away" and only used during the rational reconstruction of
    /// polynomial factors over number fields.
    /// 
    #[stability::unstable(feature = "enable")]
    pub struct NumberFieldByOrder<Impl>
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        base: NumberField<Impl>
    }

    impl<Impl> NumberFieldOrQQ for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        type ThisFieldByOrder<'ring> = Self
            where Self: 'ring;

        type ThisFieldByOrderStore<'ring> = &'ring RingValue<Self>
            where Self: 'ring;

        type IntegerRingBase = <NumberFieldBase<Impl> as NumberFieldOrQQ>::IntegerRingBase;

        type Integers = <NumberFieldBase<Impl> as NumberFieldOrQQ>::Integers;

        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            // while we could return `RingValue::from_ref(self)`, this would probably just let an infinite recursion go unchecked
            unreachable!()
        }

        fn integral_element_to_number_field_by_order<'ring>(&self, _to: &Self::ThisFieldByOrderStore<'ring>, _x: Self::Element) -> El<Self::ThisFieldByOrderStore<'ring>>
            where Self: 'ring
        {
            unreachable!()
        }

        fn element_from_field_by_order<'ring>(&self, _from: &Self::ThisFieldByOrderStore<'ring>, _x: El<Self::ThisFieldByOrderStore<'ring>>) -> Self::Element
            where Self: 'ring
        {
            unreachable!()
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            self.base.get_ring().ln_heuristic_size(c)
        }

        fn denominator_lcm(&self, c: &Self::Element) -> El<Self::Integers> {
            self.base.get_ring().denominator_lcm(c)
        }
    
        fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
            self.base.get_ring().from_integer(x)
        }
    
        fn integer_ring(&self) ->  &Self::Integers {
            self.base.get_ring().integer_ring()
        }

        fn absolute_rank(&self) -> usize {
            self.base.get_ring().absolute_rank()
        }
    }

    impl<Impl> DelegateRingImplEuclideanRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {}

    impl<Impl> PartialEq for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        fn eq(&self, other: &Self) -> bool {
            self.base.get_ring() == other.base.get_ring()
        }
    }

    impl<Impl> DelegateRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        type Base = NumberFieldBase<Impl>;
        type Element = El<NumberField<Impl>>;

        fn get_delegate(&self) -> &Self::Base {
            self.base.get_ring()
        }

        fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
        fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
        fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
        fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
    }

    impl<Impl> Domain for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {}

    impl<Impl> PerfectField for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {}

    impl<Impl> PrincipalIdealRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
            self.base.checked_div_min(lhs, rhs)
        }

        fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
            self.base.extended_ideal_gen(lhs, rhs)
        }
    }

    impl<Impl> Field for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {}

    impl<Impl> LinSolveRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
            where V1: AsPointerToSlice<Self::Element>,
                V2: AsPointerToSlice<Self::Element>,
                V3: AsPointerToSlice<Self::Element>,
                A: Allocator 
        {
            self.base.solve_right_with(lhs, rhs, out, allocator)
        }
    }

    impl<Impl> FiniteRingSpecializable for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
            Err(())
        }
    }

    impl<Impl> InterpolationBaseRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        type ExtendedRing<'a> = RingRef<'a, Self>
                where Self: 'a;

        type ExtendedRingBase<'a> = Self
            where Self: 'a;

        fn in_base<'a, S>(&self, _ext_ring: S, el: El<S>) -> Option<Self::Element>
            where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
        {
            Some(el)
        }

        fn in_extension<'a, S>(&self, _ext_ring: S, el: Self::Element) -> El<S>
            where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
        {
            el
        }

        fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>) {
            let ring = RingRef::new(self);
            (ring, (0..count).map(|n| ring.int_hom().map(n as i32)).collect())
        }
    }

    type LocalRing<'ring, Impl> = <<<<Impl as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as PolyGCDLocallyDomain>::LocalRing<'ring>;
    type LocalField<'ring, Impl> = <<<<Impl as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as PolyGCDLocallyDomain>::LocalField<'ring>;

    impl<Impl> PolyGCDLocallyDomain for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        type LocalRingBase<'ring> = FreeAlgebraImplBase<
            LocalRing<'ring, Impl>, 
            SparseMapVector<LocalRing<'ring, Impl>>
        >
            where Self: 'ring;

        type LocalRing<'ring> = RingValue<Self::LocalRingBase<'ring>>
            where Self: 'ring;

        type LocalFieldBase<'ring> = AsFieldBase<FreeAlgebraImpl<
            LocalField<'ring, Impl>, 
            SparseMapVector<LocalField<'ring, Impl>>
        >>
            where Self: 'ring;

        type LocalField<'ring> = RingValue<Self::LocalFieldBase<'ring>>
            where Self: 'ring;

        type MaximalIdeal<'ring> = <<<<Impl as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as PolyGCDLocallyDomain>::MaximalIdeal<'ring>
            where Self: 'ring;

        fn heuristic_exponent<'ring, 'a, I>(&self, p: &Self::MaximalIdeal<'ring>, poly_deg: usize, coefficients: I) -> usize
            where I: Iterator<Item = &'a Self::Element>,
                Self: 'a,
                Self: 'ring
        {
            // this has no mathematical basis, I just try to pass the largest integers on to the base ring in some way
            let largest_c = coefficients.max_by(|l, r| self.base.get_ring().ln_heuristic_size(l).total_cmp(&self.base.get_ring().ln_heuristic_size(r))).unwrap();
            let largest_c_components = self.base.wrt_canonical_basis(largest_c).iter().collect::<Vec<_>>();
            self.base.base_ring().get_ring().heuristic_exponent(p, poly_deg, largest_c_components.iter()) + self.base.rank()
        }

        fn random_maximal_ideal<'ring, F>(&'ring self, mut rng: F) -> Self::MaximalIdeal<'ring>
            where F: FnMut() -> u64
        {
            let poly_ring = DensePolyRing::new(self.base.base_ring(), "X");
            let gen_poly = self.base.generating_poly(&poly_ring, self.base.base_ring().identity());
            for _ in 0..(MAX_PROBABILISTIC_REPETITIONS * self.absolute_rank()) {
                let p = self.base.base_ring().get_ring().random_maximal_ideal(&mut rng);
                let local_field = self.base.base_ring().get_ring().local_field_at(&p);
                let local_ring = self.base.base_ring().get_ring().local_ring_at(&p, 1);
                let local_poly_ring = DensePolyRing::new(&local_field, "X");
                let red_map = local_field.can_hom(&local_ring).unwrap().compose(ReductionMap::new(self.base.base_ring().get_ring(), &p, 1));
                let gen_poly_mod = local_poly_ring.lifted_hom(&poly_ring, red_map).map_ref(&gen_poly);
                if <_ as FactorPolyField>::is_irred(&local_poly_ring, &gen_poly_mod) {
                    return p;
                }
            }
            unreachable!()
        }

        fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
            where Self: 'ring
        {
            let poly_ring = DensePolyRing::new(self.base.base_ring(), "X");
            let gen_poly = self.base.generating_poly(&poly_ring, self.base.base_ring().identity());
            let local_field = self.base.base_ring().get_ring().local_field_at(&p);
            let red_map_base = ReductionMap::new(self.base.base_ring().get_ring(), &p, 1);
            let red_map = local_field.can_hom(red_map_base.codomain()).unwrap().compose(&red_map_base);
            let mut x_pow_rank = SparseMapVector::new(self.base.rank(), self.base.base_ring().get_ring().local_field_at(&p));
            for (c, i) in poly_ring.terms(&gen_poly) {
                if i < self.rank() {
                    *x_pow_rank.at_mut(i) = red_map.codomain().negate(red_map.map_ref(c));
                }
            }
            x_pow_rank.at_mut(0);
            return AsField::from(AsFieldBase::promise_is_perfect_field(FreeAlgebraImpl::new(local_field, self.rank(), x_pow_rank)));
        }

        fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
            where Self: 'ring
        {
            let poly_ring = DensePolyRing::new(self.base.base_ring(), "X");
            let gen_poly = self.base.generating_poly(&poly_ring, self.base.base_ring().identity());
            let local_ring = self.base.base_ring().get_ring().local_ring_at(&p, e);
            let red_map = ReductionMap::new(self.base.base_ring().get_ring(), &p, e);
            let mut x_pow_rank = SparseMapVector::new(self.base.rank(), self.base.base_ring().get_ring().local_ring_at(&p, e));
            for (c, i) in poly_ring.terms(&gen_poly) {
                if i < self.rank() {
                    *x_pow_rank.at_mut(i) = red_map.codomain().negate(red_map.map_ref(c));
                }
            }
            x_pow_rank.at_mut(0);
            return FreeAlgebraImpl::new(local_ring, self.rank(), x_pow_rank);
        }

        fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            to.0.from_canonical_basis(self.base.wrt_canonical_basis(&x).iter().map(|c| self.base.base_ring().get_ring().reduce_ring_el(p, (to.0.base_ring(), to.1), c)))
        }

        fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| self.base.base_ring().get_ring().reduce_partial(p, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), c)))
        }

        fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| self.base.base_ring().get_ring().lift_partial(p, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), c)))
        }

        fn reconstruct_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
            where Self: 'ring
        {
            self.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| self.base.base_ring().get_ring().reconstruct_ring_el(p, (from.0.base_ring(), from.1), c)))
        }

        fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
            where Self: 'ring
        {
            self.base.base_ring().get_ring().dbg_maximal_ideal(p, out)
        }
    }
}

#[test]
fn test_poly_gcd_number_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let QQX = DensePolyRing::new(QQ, "X");

    let [f] = QQX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&QQX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let i = RingElementWrapper::new(&KY, KY.inclusion().map(K.canonical_gen()));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| [
        (Y.pow_ref(3) + 1) * (Y - &i),
        (Y.pow_ref(4) + 2) * (Y.pow_ref(2) + 1),
        Y - i
    ]);
    KY.println(&g);
    KY.println(&h);
    assert_el_eq!(&KY, &expected, <_ as PolyGCDRing>::gcd(&KY, &g, &h));

    let [f] = QQX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 3]);
    let K = NumberField::new(&QQX, &f);
    let KY = DensePolyRing::new(&K, "Y");
    let [f] = KY.with_wrapped_indeterminate(|Y| [Y.pow_ref(2) - 7]);
    let L = NumberField::new_over(&KY, &f);
    let LZ = DensePolyRing::new(&L, "Z");

    let sqrt3 = RingElementWrapper::new(&LZ, LZ.inclusion().map(L.inclusion().map(K.canonical_gen())));
    let sqrt7 = RingElementWrapper::new(&LZ, LZ.inclusion().map(L.canonical_gen()));
    let half = RingElementWrapper::new(&LZ, LZ.inclusion().map(L.invert(&L.int_hom().map(2)).unwrap()));
    let [g, h, expected] = LZ.with_wrapped_indeterminate(|Y| [
        Y.pow_ref(2) - &sqrt3 * Y - 1,
        Y.pow_ref(2) + &sqrt7 * Y + 1,
        Y - (sqrt3 + sqrt7) * half
    ]);
    assert_el_eq!(&LZ, &expected, <_ as PolyGCDRing>::gcd(&LZ, &g, &h));
}