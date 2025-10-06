use std::alloc::{Allocator, Global};

use crate::divisibility::DivisibilityRing;
use crate::matrix::{AsPointerToSlice, SubmatrixMut};
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::rings::extension::extension_impl::FreeAlgebraImplBase;
use crate::seq::VectorView;

use super::convolution::ConvolutionAlgorithm;

///
/// Contains the algorithm to compute the "pre-smith" form of a matrix, which
/// is sufficient for solving linear systems. Works over all principal ideal rings.
/// 
pub mod smith;
///
/// Contains the algorithm for solving linear systems over free ring extensions.
/// 
pub mod extension;
///
/// Contains algorithms related to Gaussian elimination. Note that standard functionality
/// relating to solving linear systems is provided by [`LinSolveRing`] instead.
/// 
pub mod gauss;

///
/// Result of trying to solve a linear system.
/// 
/// Possible values are:
///  - [`SolveResult::FoundUniqueSolution`]: The system is guaranteed to have a unique solution.
///  - [`SolveResult::FoundSomeSolution`]: The system has at least one solution. This is also an allowed
///    value for systems with a unique solution.
///  - [`SolveResult::NoSolution`]: The system is unsolvable.
/// 
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolveResult {
    /// The system has at least one solution. This is also an allowed
    /// value for systems with a unique solution.
    FoundSomeSolution,
    /// The system is guaranteed to have a unique solution 
    FoundUniqueSolution,
    /// The system has no solution. In particular, the output matrix
    /// is in an unspecified (but safe) state.
    NoSolution
}

impl SolveResult {

    ///
    /// Returns whether some solution to the system has been found.
    /// 
    pub fn is_solved(&self) -> bool {
        match self {
            Self::FoundSomeSolution | Self::FoundUniqueSolution => true,
            Self::NoSolution => false
        }
    }
    
    ///
    /// Panics if the system does not have a solution.
    /// 
    pub fn assert_solved(&self) {
        assert!(self.is_solved());
    }
}

///
/// Class for rings over which we can solve linear systems.
/// 
pub trait LinSolveRing: DivisibilityRing {

    ///
    /// Tries to find a matrix `X` such that `lhs * X = rhs`.
    /// 
    /// If a solution exists, it will be written to `out`. Otherwise, `out` will have an
    /// unspecified (but valid) value after the function returns. Similarly, `lhs` and `rhs`
    /// will be modified in an unspecified way, but its entries will always be valid ring elements.
    /// 
    /// Note that if a solution is found, either [`SolveResult::FoundSomeSolution`] or [`SolveResult::FoundUniqueSolution`]
    /// are returned. If there are multiple solutions, only former will be returned. However, implementations
    /// are free to also return [`SolveResult::FoundSomeSolution`] in cases where there is a unique solution.
    /// 
    fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<Self::Element>,
            V2: AsPointerToSlice<Self::Element>,
            V3: AsPointerToSlice<Self::Element>,
            A: Allocator;
}

///
/// [`RingStore`] corresponding to [`LinSolveRing`].
/// 
pub trait LinSolveRingStore: RingStore
    where Self::Type: LinSolveRing
{
    ///
    /// Solves a linear system `lhs * X = rhs`.
    /// 
    /// For details, see [`LinSolveRing::solve_right()`].
    /// 
    fn solve_right<V1, V2, V3>(&self, lhs: SubmatrixMut<V1, El<Self>>, rhs: SubmatrixMut<V2, El<Self>>, out: SubmatrixMut<V3, El<Self>>) -> SolveResult
        where V1: AsPointerToSlice<El<Self>>,
            V2: AsPointerToSlice<El<Self>>,
            V3: AsPointerToSlice<El<Self>>
    {
        self.get_ring().solve_right(lhs, rhs, out, Global)
    }

    ///
    /// Solves a linear system `lhs * X = rhs`.
    /// 
    /// For details, see [`LinSolveRing::solve_right()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn solve_right_with<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, El<Self>>, rhs: SubmatrixMut<V2, El<Self>>, out: SubmatrixMut<V3, El<Self>>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<El<Self>>,
            V2: AsPointerToSlice<El<Self>>,
            V3: AsPointerToSlice<El<Self>>,
            A: Allocator
    {
        self.get_ring().solve_right(lhs, rhs, out, allocator)
    }
}

impl<R> LinSolveRingStore for R
    where R: RingStore, R::Type: LinSolveRing
{}

impl<R: ?Sized + PrincipalIdealRing> LinSolveRing for R {

    default fn solve_right<V1, V2, V3, A>(&self, lhs: SubmatrixMut<V1, Self::Element>, rhs: SubmatrixMut<V2, Self::Element>, out: SubmatrixMut<V3, Self::Element>, allocator: A) -> SolveResult
        where V1: AsPointerToSlice<Self::Element>, V2: AsPointerToSlice<Self::Element>, V3: AsPointerToSlice<Self::Element>,
            A: Allocator
    {
        smith::solve_right_using_pre_smith(RingRef::new(self), lhs, rhs, out, allocator)
    }
}

impl<R, V, A_ring, C_ring> LinSolveRing for FreeAlgebraImplBase<R, V, A_ring, C_ring>
    where R: RingStore,
        R::Type: LinSolveRing,
        V: VectorView<El<R>> + Send + Sync,
        A_ring: Allocator + Clone + Send + Sync, 
        C_ring: ConvolutionAlgorithm<R::Type>
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
