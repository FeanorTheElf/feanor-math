
use std::alloc::Allocator;
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use sparse::SparseMapVector;

use crate::algorithms::convolution::*;
use crate::algorithms::poly_gcd::*;
use crate::algorithms::matmul::StrassenHint;
use crate::delegate::DelegateRing;
use crate::integer::*;
use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::rings::field::AsField;
use crate::rings::finite::FiniteRing;
use crate::rings::poly::*;
use crate::rings::rational::*;
use crate::divisibility::*;
use crate::rings::extension::*;
use crate::rings::zn::zn_64;
use crate::rings::zn::zn_big;

use super::extension_impl::FreeAlgebraImpl;
use super::Field;
use super::FreeAlgebra;

#[stability::unstable(feature = "enable")]
pub trait NumberFieldOrQQ: RingBase {

    type QuotientRing<'ring>: FiniteRing + LinSolveRing
        where Self: 'ring;
    type QuotientField<'ring>: FiniteRing + Field + CanIsoFromTo<Self::QuotientRing<'ring>>
        where Self: 'ring;
}

impl<I> NumberFieldOrQQ for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    type QuotientField<'ring> = AsFieldBase<zn_64::Zn>
        where Self: 'ring;
    type QuotientRing<'ring> = zn_big::ZnBase<I>
        where Self: 'ring;
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

impl<Impl> NumberFieldOrQQ for NumberFieldBase<Impl> 
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
{
    type QuotientField<'ring> = AsFieldBase<FreeAlgebraImpl<
            RingValue<<<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::QuotientField<'ring>>, 
            SparseMapVector<RingValue<<<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::QuotientField<'ring>>>
    >>
        where Self: 'ring;
    type QuotientRing<'ring> = FreeAlgebraImplBase<
        RingValue<<<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::QuotientRing<'ring>>, 
        SparseMapVector<RingValue<<<<Impl::Type as RingExtension>::BaseRing as RingStore>::Type as NumberFieldOrQQ>::QuotientRing<'ring>>>
    >
        where Self: 'ring;
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
/// Basically, we want to implement [`PolyGCDLocallyDomain`] for orders in number fields.
/// However, as specified currently, this not possible, since we need to perform rational
/// reconstruction when lifting elements - and thus don't get values in the order anymore.
/// 
/// Instead, we want to implement [`PolyGCDLocallyDomain`] for the number field itself. This
/// works, but it seems not to be a good idea to expose this interface, since we usually get
/// much better results, if we run the local algorithms on polynomials that first have been
/// scaled into an order (even if the result is not in the order anymore).
/// 
/// The solution is now to implement [`PolyGCDLocallyDomain`] for a newtype, and call it from
/// the [`PolyGCDRing`] implementation for number fields.
/// 
mod number_field_as_order {

    use crate::{algorithms::poly_gcd::local::PolyGCDLocallyDomain, specialization::{FiniteRingOperation, FiniteRingSpecializable}};

    use super::*;

    struct NumberFieldPolyGCDLocallyDomain<Impl>
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {
        base: NumberField<Impl>
    }
    
    impl<Impl> PartialEq for NumberFieldPolyGCDLocallyDomain<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {
        fn eq(&self, other: &Self) -> bool {
            self.base.get_ring() == other.base.get_ring()
        }
    }

    impl<Impl> DelegateRing for NumberFieldPolyGCDLocallyDomain<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
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

    impl<Impl> Domain for NumberFieldPolyGCDLocallyDomain<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {}

    impl<Impl> FiniteRingSpecializable for NumberFieldPolyGCDLocallyDomain<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {
        fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
            Err(())
        }
    }
    
    impl<Impl> PolyGCDLocallyDomain for NumberFieldPolyGCDLocallyDomain<Impl> 
        where Impl: RingStore,
            Impl::Type: Field + FreeAlgebra,
            <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: NumberFieldOrQQ
    {
        type LocalRingBase<'ring> = <NumberFieldBase<Impl> as NumberFieldOrQQ>::QuotientRing<'ring>
            where Self: 'ring;

        type LocalRing<'ring> = RingValue<<NumberFieldBase<Impl> as NumberFieldOrQQ>::QuotientRing<'ring>>
            where Self: 'ring;

        type LocalFieldBase<'ring> = <NumberFieldBase<Impl> as NumberFieldOrQQ>::QuotientField<'ring>
            where Self: 'ring;

        type LocalField<'ring> = RingValue<<NumberFieldBase<Impl> as NumberFieldOrQQ>::QuotientField<'ring>>
            where Self: 'ring;

        type MaximalIdeal<'ring> = i64
            where Self: 'ring;

        fn heuristic_exponent<'ring, 'a, I>(&self, _maximal_ideal: &Self::MaximalIdeal<'ring>, _poly_deg: usize, _coefficients: I) -> usize
            where I: Iterator<Item = &'a Self::Element>,
                Self: 'a,
                Self: 'ring
        {
            unimplemented!()
        }

        fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
            where F: FnMut() -> u64
        {
            unimplemented!()
        }

        fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
            where Self: 'ring
        {
            unimplemented!()
        }

        fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
            where Self: 'ring
        {
            unimplemented!()
        }

        fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            unimplemented!()
        }

        fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            unimplemented!()
        }

        fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
            where Self: 'ring
        {
            unimplemented!()
        }

        fn reconstruct_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
            where Self: 'ring
        {
            unimplemented!()
        }

        fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
            where Self: 'ring
        {
            unimplemented!()
        }
    }
}