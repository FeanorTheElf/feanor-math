use std::alloc::Allocator;

use crate::matrix::{AsPointerToSlice, SubmatrixMut};
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::rings::extension::extension_impl::FreeAlgebraImplBase;
use crate::seq::VectorView;

pub mod smith;
pub mod extension;
pub mod poly_det;

#[stability::unstable(feature = "enable")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolveResult {
    FoundSomeSolution, FoundUniqueSolution, NoSolution
}

impl SolveResult {

    #[stability::unstable(feature = "enable")]
    pub fn is_solved(&self) -> bool {
        match self {
            Self::FoundSomeSolution | Self::FoundUniqueSolution => true,
            Self::NoSolution => false
        }
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn assert_solved(&self) {
        assert!(self.is_solved());
    }
}

///
/// Class for rings over which we can solve linear systems.
/// 
#[stability::unstable(feature = "enable")]
pub trait LinSolveRing: RingBase {

    ///
    /// Tries to find a matrix `X` such that `lhs * X = rhs`.
    /// 
    /// If a solution exists, it will be written to `out`. Otherwise, `out` will have an
    /// unspecified (but valid) value after the function returns. Similarly, `lhs` and `rhs`
    /// will be modified in an unspecified way, but its entries will always be valid ring elements.
    /// 
    fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<Self::Element>,
            V2: AsPointerToSlice<Self::Element>,
            V3: AsPointerToSlice<Self::Element>,
            A: Allocator;
}

impl<R: ?Sized + PrincipalIdealRing> LinSolveRing for R {

    default fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>, V3: AsPointerToSlice<Self::Element>,
            A: Allocator
    {
        smith::solve_right_using_pre_smith(RingRef::new(self), lhs, rhs, out, allocator)
    }
}

impl<R, V, A_ring> LinSolveRing for FreeAlgebraImplBase<R, V, A_ring>
    where R: RingStore,
        R::Type: LinSolveRing,
        V: VectorView<El<R>>,
        A_ring: Allocator + Clone
{
    fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<Self::Element>,
            V2: AsPointerToSlice<Self::Element>,
            V3: AsPointerToSlice<Self::Element>,
            A:  Allocator
    {
        extension::solve_right_over_extension(RingRef::new(self), lhs, rhs, out, allocator)
    }
}