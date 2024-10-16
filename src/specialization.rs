use std::alloc::Allocator;
use std::marker::PhantomData;

use crate::algorithms::convolution::ConvolutionAlgorithm;
use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::algorithms::linsolve::LinSolveRing;
use crate::field::Field;
use crate::field::PerfectField;
use crate::integer::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::rings::extension::extension_impl::*;
use crate::rings::extension::*;
use crate::rings::extension::galois_field::*;
use crate::rings::field::*;
use crate::rings::finite::*;
use crate::rings::rational::*;
use crate::seq::*;
use crate::rings::zn::*;
use crate::divisibility::*;
use crate::delegate::*;

///
/// Trait for a function that takes a single `FiniteRing + DivisibilityRing + LinSolveRing + CanIsoFromTo<OriginalField>`
/// argument of generic type. This needs to be a new trait, since there higher-order generic bounds only are supported for
/// lifetimes.
///
#[stability::unstable(feature = "enable")]
pub trait FiniteRingOperation<OriginalField>
    where OriginalField: ?Sized + RingBase
{    
    type Output<'a>
        where Self: 'a;

    fn execute<'a, F>(self, field: F) -> Self::Output<'a>
        where Self: 'a,
            F: 'a + RingStore,
            F::Type: FiniteRing + DivisibilityRing + LinSolveRing + CanIsoFromTo<OriginalField> + SpecializeToFiniteRing;
}

///
/// Rings that support compile-time specialization for finite rings. This is basically a workaround
/// for generic specialization, since currently it is not supported to specialize on traits.
/// 
/// This is such a terrible hack, and so difficult to use, that it will probably be unstable forever.
/// 
#[stability::unstable(feature = "enable")]
pub trait SpecializeToFiniteRing: RingBase + DivisibilityRing {

    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()>;

    fn is_finite_ring(&self) -> bool {
        struct NoOperation;
        impl<R: ?Sized + RingBase> FiniteRingOperation<R> for NoOperation {
            type Output<'a> = () where Self: 'a;
            fn execute<'a, F>(self, _: F) -> Self::Output<'a>
                where F: 'a + RingStore,
                    F::Type: FiniteRing + DivisibilityRing + LinSolveRing + CanIsoFromTo<R> + SpecializeToFiniteRing
            {}
        }
        return self.specialize_finite_ring(NoOperation).is_ok();
    }
}

impl<I> SpecializeToFiniteRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, _op: O) -> Result<O::Output<'a>, ()> {
        Err(())
    }
}

impl<R, V, A, C> SpecializeToFiniteRing for FreeAlgebraImplBase<R, V, A, C>
    where R: RingStore,
        V: VectorView<El<R>>,
        R::Type: LinSolveRing + SpecializeToFiniteRing,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<R::Type>
{
    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {

        struct WrapBaseRing<'a, R, V, A, C, O>
            where R: RingStore,
                V: VectorView<El<R>>,
                R::Type: LinSolveRing + SpecializeToFiniteRing,
                A: Allocator + Clone,
                C: ConvolutionAlgorithm<R::Type>,
                O: FiniteRingOperation<FreeAlgebraImplBase<R, V, A, C>>
        {
            op: O,
            ring: &'a FreeAlgebraImplBase<R, V, A, C>
        }

        impl<'a, R, V, A, C, O> FiniteRingOperation<R::Type> for WrapBaseRing<'a, R, V, A, C, O>
            where R: RingStore,
                V: VectorView<El<R>>,
                R::Type: LinSolveRing + SpecializeToFiniteRing,
                A: Allocator + Clone,
                C: ConvolutionAlgorithm<R::Type>,
                O: FiniteRingOperation<FreeAlgebraImplBase<R, V, A, C>>
        {
            type Output<'b> = O::Output<'b>
                where Self: 'b;

            fn execute<'b, F>(self, ring: F) -> Self::Output<'b>
                where Self: 'b,
                    F: 'b + RingStore,
                    F::Type: FiniteRing + DivisibilityRing + LinSolveRing + CanIsoFromTo<R::Type> + SpecializeToFiniteRing
            {
                let iso = ring.can_iso(self.ring.base_ring()).unwrap();
                let x_pow_rank = self.ring.x_pow_rank().as_iter().map(|a| iso.inv().map_ref(a)).collect::<Vec<_>>();
                let new_ring = FreeAlgebraImpl::new_with(
                    ring,
                    self.ring.rank(),
                    x_pow_rank,
                    self.ring.gen_name(),
                    self.ring.allocator(),
                    STANDARD_CONVOLUTION
                );
                return self.op.execute(new_ring);
            }
        }

        return self.base_ring().get_ring().specialize_finite_ring(WrapBaseRing { op: op, ring: self });
    }
}

impl<R, V, A, C> SpecializeToFiniteRing for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field + SelfIso,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
{
    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

impl<R> SpecializeToFiniteRing for AsFieldBase<R>
    where R: RingStore,
        R::Type: SpecializeToFiniteRing + DivisibilityRing
{
    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {

        struct WrapFiniteRing<R, O>
            where R: RingStore,
                R::Type: SpecializeToFiniteRing,
                O: FiniteRingOperation<AsFieldBase<R>>
        {
            ring: PhantomData<R>,
            op: O
        }

        impl<R, O> FiniteRingOperation<R::Type> for WrapFiniteRing<R, O>
            where R: RingStore,
                R::Type: SpecializeToFiniteRing,
                O: FiniteRingOperation<AsFieldBase<R>> 
        {
            type Output<'a> = O::Output<'a>
                where Self: 'a;

            fn execute<'a, F>(self, field: F) -> Self::Output<'a>
                where Self: 'a,
                    F: 'a + RingStore,
                    F::Type: FiniteRing + DivisibilityRing + CanIsoFromTo<R::Type> + SpecializeToFiniteRing
            {
                self.op.execute(AsField::from(AsFieldBase::promise_is_perfect_field(field)))
            }
        }

        return self.get_delegate().specialize_finite_ring(WrapFiniteRing::<R, O> { ring: PhantomData, op: op });
    }
}

impl SpecializeToFiniteRing for zn_64::ZnBase {

    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

impl<I: IntegerRingStore> SpecializeToFiniteRing for zn_big::ZnBase<I>
    where I::Type: IntegerRing
{
    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

impl<const N: u64, const IS_FIELD: bool> SpecializeToFiniteRing for zn_static::ZnBase<N, IS_FIELD> {

    fn specialize_finite_ring<'a, O: FiniteRingOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

///
/// Trait for a function that takes a single `FiniteRing + Field + LinSolveRing + CanIsoFromTo<OriginalField>`
/// argument of generic type. This needs to be a new trait, since there higher-order generic bounds only are supported for
/// lifetimes.
///
#[stability::unstable(feature = "enable")]
pub trait FiniteFieldOperation<OriginalField>
    where OriginalField: ?Sized + RingBase
{    
    type Output<'a>
        where Self: 'a;

    fn execute<'a, F>(self, field: F) -> Self::Output<'a>
        where Self: 'a,
            F: 'a + RingStore,
            F::Type: FiniteRing + Field + PerfectField + LinSolveRing + CanIsoFromTo<OriginalField> + SpecializeToFiniteField;
}

///
/// Rings that support compile-time specialization for finite fields. This is basically a workaround
/// for generic specialization, since currently it is not supported to specialize on traits.
/// 
/// This is such a terrible hack, and so difficult to use, that it will probably be unstable forever.
/// 
#[stability::unstable(feature = "enable")]
pub trait SpecializeToFiniteField: RingBase + DivisibilityRing {

    fn specialize_finite_field<'a, O: FiniteFieldOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()>;
}

impl<I> SpecializeToFiniteField for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn specialize_finite_field<'a, O: FiniteFieldOperation<Self>>(&'a self, _op: O) -> Result<O::Output<'a>, ()> {
        Err(())
    }
}

impl<R> SpecializeToFiniteField for AsFieldBase<R>
    where R: RingStore,
        R::Type: SpecializeToFiniteRing
{
    fn specialize_finite_field<'a, O: FiniteFieldOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        struct WrapFiniteRing<R, O>
            where R: RingStore,
                R::Type: SpecializeToFiniteRing,
                O: FiniteFieldOperation<AsFieldBase<R>>
        {
            ring: PhantomData<R>,
            op: O
        }

        impl<R, O> FiniteRingOperation<R::Type> for WrapFiniteRing<R, O>
            where R: RingStore,
                R::Type: SpecializeToFiniteRing,
                O: FiniteFieldOperation<AsFieldBase<R>> 
        {
            type Output<'a> = O::Output<'a>
                where Self: 'a;

            fn execute<'a, F>(self, field: F) -> Self::Output<'a>
                where Self: 'a,
                    F: 'a + RingStore,
                    F::Type: FiniteRing + DivisibilityRing + CanIsoFromTo<R::Type> + SpecializeToFiniteRing
            {
                self.op.execute(AsField::from(AsFieldBase::promise_is_perfect_field(field)))
            }
        }

        return self.get_delegate().specialize_finite_ring(WrapFiniteRing::<R, O> { ring: PhantomData, op: op });
    }
}

impl<R, V, A, C> SpecializeToFiniteField for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        V: VectorView<El<R>>,
        R::Type: SelfIso + LinSolveRing + ZnRing + FiniteRing + Field,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<R::Type>
{
    fn specialize_finite_field<'a, O: FiniteFieldOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

impl<const N: u64> SpecializeToFiniteField for zn_static::ZnBase<N, true> {

    fn specialize_finite_field<'a, O: FiniteFieldOperation<Self>>(&'a self, op: O) -> Result<O::Output<'a>, ()> {
        Ok(op.execute(RingRef::new(self)))
    }
}

#[cfg(test)]
use crate::ordered::OrderedRingStore;

#[test]
fn test_specialize_finite_field() {

    struct Verify<'a, R>(&'a R)
        where R: RingBase;

    impl<'a, R> FiniteFieldOperation<R> for Verify<'a, R>
        where R: RingBase
    {
        type Output<'b> = ()
            where Self: 'b;

        fn execute<'b, F>(self, field: F) -> ()
            where Self: 'b,
                F: 'b + RingStore,
                F::Type: FiniteRing + DivisibilityRing + CanIsoFromTo<R>
        {
            if BigIntRing::RING.is_lt(&field.size(&BigIntRing::RING).unwrap(), &BigIntRing::RING.int_hom().map(20)) {
                crate::homomorphism::generic_tests::test_homomorphism_axioms(field.can_iso(&RingRef::new(self.0)).unwrap(), field.elements());
            }
        }
    }

    let ring = zn_64::Zn::new(7).as_field().ok().unwrap();
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_ok());
    
    let ring = GaloisField::new(3, 2);
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_ok());

    let ring = GaloisField::new(3, 2).into().unwrap_self();
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_ok());
    
    let ring = RationalField::new(BigIntRing::RING);
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_err());
    
    let QQ = RationalField::new(BigIntRing::RING);
    let ring = FreeAlgebraImpl::new(&QQ, 2, [QQ.neg_one()]).as_field().ok().unwrap();
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_err());

    let base_ring = GaloisField::new(3, 2).into().unwrap_self();
    let ring = FreeAlgebraImpl::new(&base_ring, 3, [base_ring.neg_one(), base_ring.one()]).as_field().ok().unwrap();
    assert!(ring.get_ring().specialize_finite_field(Verify(ring.get_ring())).is_ok());
}