use crate::ring::*;
use crate::rings::finite::*;

///
/// Operation on a ring `R` that only makes sense if `R` implements
/// the trait [`crate::rings::finite::FiniteRing`].
/// 
/// Used through the trait [`FiniteRingSpecializable`].
/// 
pub trait FiniteRingOperation<R>
    where R: ?Sized
{    
    type Output;

    ///
    /// Runs the operations, with the additional assumption that `R: FiniteRing`.
    /// 
    fn execute(self) -> Self::Output
        where R: FiniteRing;

    ///
    /// Runs the operation in case that `R` is not a [`FiniteRing`].
    /// 
    fn fallback(self) -> Self::Output;
}

///
/// Trait for ring types that can check (at compile time) whether they implement
/// [`crate::rings::finite::FiniteRing`].
/// 
/// This serves as a workaround while specialization is not properly supported. 
/// 
pub trait FiniteRingSpecializable: RingBase {

    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output;

    fn is_finite_ring() -> bool {
        struct CheckIsFinite;
        impl<R: ?Sized + RingBase> FiniteRingOperation<R> for CheckIsFinite {
            type Output = bool;
            fn execute(self) -> Self::Output
                where R: FiniteRing { true }
            fn fallback(self) -> Self::Output { false }
        }
        return Self::specialize(CheckIsFinite);
    }
}

#[cfg(test)]
use crate::homomorphism::*;
#[cfg(test)]
use crate::rings::field::*;
#[cfg(test)]
use crate::rings::extension::galois_field::*;
#[cfg(test)]
use crate::rings::rational::*;
#[cfg(test)]
use crate::rings::zn::*;
#[cfg(test)]
use crate::integer::*;
#[cfg(test)]
use crate::rings::extension::extension_impl::*;
#[cfg(test)]
use crate::rings::extension::*;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_specialize_finite_field() {
    LogAlgorithmSubscriber::init_test();

    struct Verify<'a, R>(&'a R, i32)
        where R: ?Sized + RingBase;

    impl<'a, R: ?Sized> FiniteRingOperation<R> for Verify<'a, R>
        where R: RingBase
    {
        type Output = bool;

        fn execute(self) -> bool
            where R: FiniteRing
        {
            assert_el_eq!(BigIntRing::RING, BigIntRing::RING.int_hom().map(self.1), self.0.size(&BigIntRing::RING).unwrap());
            return true;
        }

        fn fallback(self) -> Self::Output {
            return false;
        }
    }

    let ring = zn_64b::Zn64B::new(7).as_field().ok().unwrap();
    assert!(<AsFieldBase<zn_64b::Zn64B>>::specialize(Verify(ring.get_ring(), 7)));
    
    let ring = GaloisField::new(3, 2);
    assert!(GaloisFieldBase::specialize(Verify(ring.get_ring(), 9)));

    let ring = GaloisField::new(3, 2).into().unwrap_self();
    assert!(<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 9)));
    
    let ring = RationalField::new(BigIntRing::RING);
    assert!(!<RationalFieldBase<_>>::specialize(Verify(ring.get_ring(), 0)));
    
    let QQ = RationalField::new(BigIntRing::RING);
    let ring = FreeAlgebraImpl::new(&QQ, 2, [QQ.neg_one()]).as_field().ok().unwrap();
    assert!(!<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 0)));

    let base_ring = GaloisField::new(3, 2).into().unwrap_self();
    let ring = FreeAlgebraImpl::new(&base_ring, 3, [base_ring.neg_one(), base_ring.one()]).as_field().ok().unwrap();
    assert!(<AsFieldBase<FreeAlgebraImpl<_, _, _, _>>>::specialize(Verify(ring.get_ring(), 729)));
}