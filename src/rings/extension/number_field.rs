
use std::alloc::Allocator;
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use gcd_locally::PolyGCDLocallyDomain;
use number_field_by_order::NumberFieldByOrder;
use number_field_by_order::RationalsByZZ;
use sparse::SparseMapVector;

use crate::algorithms::convolution::*;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::matmul::StrassenHint;
use crate::delegate::DelegateRing;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::field::AsField;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::divisibility::*;
use crate::rings::extension::*;

use super::extension_impl::FreeAlgebraImpl;
use super::Field;
use super::FreeAlgebra;

#[stability::unstable(feature = "enable")]
pub trait NumberFieldOrQQ: RingBase {

    type ThisFieldByOrder<'ring>: PolyGCDLocallyDomain + NumberFieldOrQQ + LinSolveRing
        where Self: 'ring;

    type ThisFieldByOrderStore<'ring>: RingStore<Type = Self::ThisFieldByOrder<'ring>>
        where Self: 'ring;

    fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
        where Self: 'ring;
        
    fn ln_heuristic_size(&self, c: &Self::Element) -> f64;
}

impl<I> NumberFieldOrQQ for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    type ThisFieldByOrder<'ring> = RationalsByZZ<&'ring I>
        where Self: 'ring;

    type ThisFieldByOrderStore<'ring> = RingValue<RationalsByZZ<&'ring I>>
        where Self: 'ring;

    fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
        where Self: 'ring
    {
        RingValue::from(RationalsByZZ { base: RationalField::new(self.base_ring()) })
    }

    fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
        (self.base_ring().abs_log2_ceil(self.num(c)).unwrap() + self.base_ring().abs_log2_ceil(self.den(c)).unwrap()) as f64 * 2f64.ln()
    }
}

///
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
    base: Impl
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

type BaseFieldByOrder<'ring, Impl> = RingValue<<<<<Impl as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::ThisFieldByOrder<'ring>>;

impl<Impl> NumberFieldOrQQ for NumberFieldBase<Impl> 
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    type ThisFieldByOrder<'ring> = NumberFieldByOrder<AsField<FreeAlgebraImpl<BaseFieldByOrder<'ring, Impl>, Vec<El<BaseFieldByOrder<'ring, Impl>>>>>>
        where Self: 'ring;

    type ThisFieldByOrderStore<'ring> = RingValue<Self::ThisFieldByOrder<'ring>>
        where Self: 'ring;

    fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
        where Self: 'ring
    {
        unimplemented!()
    }

    fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
        self.wrt_canonical_basis(c).iter().map(|c| self.base_ring().get_ring().ln_heuristic_size(&c)).max_by(f64::total_cmp).unwrap() + (self.rank() as f64).ln()
    }
}

impl<Impl> Clone for NumberFieldBase<Impl> 
    where Impl: RingStore + Clone,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<Impl> Copy for NumberFieldBase<Impl> 
    where Impl: RingStore + Copy,
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

impl<Impl> Field for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{}

impl<Impl> Domain for NumberFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{}

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
/// I'm really sorry, this would probably win the prize for the most unreadable code in
/// feanor-math. Basically, I am implementing a type-level recursion, such that [`PolyGCDLocallyDomain`]
/// is implemented for all "order" in number fields, even if we nest them arbitrarily many times.
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
mod number_field_by_order {

    use crate::algorithms::poly_gcd::gcd_locally::{IntegerPolyGCDRing, ReductionMap};
    use crate::rings::zn::*;

    use crate::algorithms::linsolve::SolveResult;
    use crate::algorithms::poly_gcd::gcd_locally::PolyGCDLocallyDomain;
    use crate::matrix::{AsPointerToSlice, SubmatrixMut};
    use crate::specialization::{FiniteRingOperation, FiniteRingSpecializable};

    use super::*;

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

        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            // while we could return `RingValue::from_ref(self)`, this would probably just let an infinite recursion
            // go unchecked
            unreachable!()
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            self.base.get_ring().ln_heuristic_size(c)
        }
    }

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

    impl<I> FiniteRingSpecializable for RationalsByZZ<I>
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
            Err(())
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
            assert!(self.base.base_ring().is_one(self.base.den(&x)));
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
    /// Still represents a number field, but has a generating polynomial that is integral,
    /// in other words, if we restrict to elements `el` such that `self.wrt_canonical_basis(el)`
    /// has only integral elements, we get an order in the number field.
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

        fn construct_number_field_by_order<'ring>(&'ring self) -> Self::ThisFieldByOrderStore<'ring>
            where Self: 'ring
        {
            // while we could return `RingValue::from_ref(self)`, this would probably just let an infinite recursion
            // go unchecked
            unreachable!()
        }

        fn ln_heuristic_size(&self, c: &Self::Element) -> f64 {
            self.base.get_ring().ln_heuristic_size(c)
        }
    }

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

    impl<Impl> EuclideanRing for NumberFieldByOrder<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ + PolyGCDLocallyDomain
    {
        fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
            self.base.euclidean_div_rem(lhs, rhs)
        }

        fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
            self.base.euclidean_deg(val)
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
            loop {
                let p = self.base.base_ring().get_ring().random_maximal_ideal(&mut rng);
                let local_field = self.base.base_ring().get_ring().local_field_at(&p);
                let local_ring = self.base.base_ring().get_ring().local_ring_at(&p, 1);
                let local_poly_ring = DensePolyRing::new(&local_field, "X");
                let red_map = local_field.can_hom(&local_ring).unwrap().compose(ReductionMap::new(self.base.base_ring().get_ring(), &p, 1));
                if <_ as FactorPolyField>::is_irred(&local_poly_ring, &local_poly_ring.lifted_hom(&poly_ring, red_map).map_ref(&gen_poly)) {
                    return p;
                }
            }
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