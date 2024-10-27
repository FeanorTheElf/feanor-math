use std::alloc::Allocator;
use std::marker::PhantomData;

use crate::algorithms::convolution::ConvolutionAlgorithm;
use crate::field::Field;
use crate::integer::*;
use crate::primitive_int::PrimitiveInt;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::rings::extension::*;
use crate::rings::extension::extension_impl::*;
use crate::rings::extension::galois_field::*;
use crate::rings::field::*;
use crate::rings::finite::*;
use crate::rings::rational::*;
use crate::rings::rust_bigint::RustBigintRingBase;
use crate::seq::*;
use crate::rings::zn::*;
use crate::divisibility::*;

///
/// Operation on a ring `R` that only makes sense if `R` implements
/// the trait [`crate::rings::finite::FiniteRing`].
/// 
#[stability::unstable(feature = "enable")]
pub trait FiniteRingOperation<R>
    where R: ?Sized
{    
    type Output;

    fn execute(self) -> Self::Output
        where R: FiniteRing;
}

///
/// Trait for ring types that can check (at compile time) whether they implement
/// [`crate::rings::finite::FiniteRing`].
/// 
#[stability::unstable(feature = "enable")]
pub trait FiniteRingSpecializable: RingBase {

    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()>;

    fn is_finite_ring() -> bool {
        struct CheckIsFinite;
        impl<R: ?Sized + RingBase> FiniteRingOperation<R> for CheckIsFinite {
            type Output = ();
            fn execute(self) -> Self::Output
                where R: FiniteRing { () } 
        }
        return Self::specialize(CheckIsFinite).is_ok();
    }
}

impl<I> FiniteRingSpecializable for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
    }
}

impl<I> FiniteRingSpecializable for zn_big::ZnBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<A> FiniteRingSpecializable for RustBigintRingBase<A>
    where A: Allocator + Clone
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
    }
}

impl<T> FiniteRingSpecializable for StaticRingBase<T>
    where T: PrimitiveInt
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
    }
}

impl FiniteRingSpecializable for zn_64::ZnBase {
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<R> FiniteRingSpecializable for AsFieldBase<R>
    where R: RingStore,
        R::Type: FiniteRingSpecializable + DivisibilityRing
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        struct OpWrapper<R, O>
            where R: RingStore,
                R::Type: FiniteRingSpecializable + DivisibilityRing,
                O: FiniteRingOperation<AsFieldBase<R>>
        {
            operation: O,
            ring: PhantomData<R>
        }

        impl<R, O> FiniteRingOperation<R::Type> for OpWrapper<R, O>
            where R: RingStore,
                R::Type: FiniteRingSpecializable + DivisibilityRing,
                O: FiniteRingOperation<AsFieldBase<R>>
        {
            type Output = O::Output;
            fn execute(self) -> Self::Output where R::Type:FiniteRing {
                self.operation.execute()
            }
        }

        <R::Type as FiniteRingSpecializable>::specialize(OpWrapper { operation: op, ring: PhantomData })
    }
}

impl<Impl> FiniteRingSpecializable for GaloisFieldBase<Impl>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra + FiniteRing,
        <<Impl::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + Field
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<R, V, A, C> FiniteRingSpecializable for FreeAlgebraImplBase<R, V, A, C>
    where R: RingStore,
        R::Type: FiniteRingSpecializable, 
        V: VectorView<El<R>>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<R::Type>
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        struct OpWrapper<R, V, A, C, O>
            where R: RingStore,
                R::Type: FiniteRingSpecializable, 
                V: VectorView<El<R>>,
                A: Allocator + Clone,
                C: ConvolutionAlgorithm<R::Type>,
                O: FiniteRingOperation<FreeAlgebraImplBase<R, V, A, C>>
        {
            operation: O,
            ring: PhantomData<FreeAlgebraImpl<R, V, A, C>>
        }

        impl<R, V, A, C, O> FiniteRingOperation<R::Type> for OpWrapper<R, V, A, C, O>
            where R: RingStore,
                R::Type: FiniteRingSpecializable, 
                V: VectorView<El<R>>,
                A: Allocator + Clone,
                C: ConvolutionAlgorithm<R::Type>,
                O: FiniteRingOperation<FreeAlgebraImplBase<R, V, A, C>>
        {
            type Output = O::Output;
            fn execute(self) -> Self::Output where R::Type:FiniteRing {
                self.operation.execute()
            }
        }

        <R::Type as FiniteRingSpecializable>::specialize(OpWrapper { operation: op, ring: PhantomData })
    }
}

impl<const N: u64> FiniteRingSpecializable for zn_static::ZnBase<N, true> {

    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

#[cfg(test)]
use crate::homomorphism::*;

#[test]
fn test_specialize_finite_field() {

    struct Verify<'a, R>(&'a R, i32)
        where R: ?Sized + RingBase;

    impl<'a, R: ?Sized> FiniteRingOperation<R> for Verify<'a, R>
        where R: RingBase
    {
        type Output = ();

        fn execute(self)
            where R: FiniteRing
        {
            assert_el_eq!(BigIntRing::RING, BigIntRing::RING.int_hom().map(self.1), self.0.size(&BigIntRing::RING).unwrap());
        }
    }

    let ring = zn_64::Zn::new(7).as_field().ok().unwrap();
    assert!(<AsFieldBase<zn_64::Zn>>::specialize(Verify(ring.get_ring(), 7)).is_ok());
    
    let ring = GaloisField::new(3, 2);
    assert!(GaloisFieldBase::specialize(Verify(ring.get_ring(), 9)).is_ok());

    let ring = GaloisField::new(3, 2).into().unwrap_self();
    assert!(<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 9)).is_ok());
    
    let ring = RationalField::new(BigIntRing::RING);
    assert!(<RationalFieldBase<_>>::specialize(Verify(ring.get_ring(), 0)).is_err());
    
    let QQ = RationalField::new(BigIntRing::RING);
    let ring = FreeAlgebraImpl::new(&QQ, 2, [QQ.neg_one()]).as_field().ok().unwrap();
    assert!(<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 0)).is_err());

    // let base_ring = GaloisField::new(3, 2).into().unwrap_self();
    // let ring = FreeAlgebraImpl::new(&base_ring, 3, [base_ring.neg_one(), base_ring.one()]).as_field().ok().unwrap();
    // assert!(<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 729)).is_ok());
}