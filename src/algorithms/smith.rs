use std::alloc::Global;

use smith::determinant_using_pre_smith;

use crate::matrix::*;
use crate::ring::*;
use crate::pid::PrincipalIdealRing;

use super::linsolve::*;

///
/// Computes the determinant of `A`.
/// 
/// The value of `A` will be changed by the algorithm in an unspecified way.
/// 
pub fn determinant<R, V>(A: SubmatrixMut<V, El<R>>, ring: R) -> El<R>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V: AsPointerToSlice<El<R>>
{
    determinant_using_pre_smith(ring, A, Global)
}

///
/// Finds a solution to the system `AX = B`, if it exists.
/// In the case that there are multiple solutions, an unspecified one is returned.
/// 
/// The values of `A` and `rhs` will be changed by the algorithm in an unspecified way.
/// 
pub fn solve_right<R, V1, V2>(A: SubmatrixMut<V1, El<R>>, rhs: SubmatrixMut<V2, El<R>>, ring: R) -> Option<OwnedMatrix<El<R>>>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>
{
    let mut solution = OwnedMatrix::zero(A.col_count(), rhs.col_count(), ring);
    let sol = ring.get_ring().solve_right(A, rhs, solution.data_mut(), Global);
    if sol.is_solved() {
        return Some(solution);
    } else {
        return None;
    }
}
