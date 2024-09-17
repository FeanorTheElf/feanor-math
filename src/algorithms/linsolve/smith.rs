use std::alloc::Allocator;
use std::cmp::min;

use crate::algorithms::linsolve::SolveResult;
use crate::divisibility::*;
use crate::matrix::*;
use crate::matrix::transform::{TransformCols, TransformRows, TransformTarget};
use crate::ring::*;
use crate::pid::PrincipalIdealRing;
use crate::algorithms::matmul::{STANDARD_MATMUL, MatmulAlgorithm};

///
/// Transforms `A` into `A'` via transformations `L, R` such that
/// `L A R = A'` and `A'` is diagonal.
/// 
/// # (Non-)Uniqueness of the solution
/// 
/// Note that this is not the complete Smith normal form, 
/// as that requires the entries on the diagonal to divide
/// each other. However, computing this pre-smith form is much
/// faster, and can still be used for solving equations (the main
/// use case). However, it is not unique.
/// 
/// # Warning on infinite rings (in particular Z)
/// 
/// For infinite principal ideal rings, this function is correct,
/// but in some situations, the performance can be terrible. The
/// reason is that no care is taken which of the many possible results
/// is returned - and in fact, this algorithm can sometimes choose
/// one that has exponential size in the input. Hence, in these
/// cases it is recommended to use another algorithm, e.g. based on
/// LLL to perform intermediate lattice reductions (not yet implemented
/// in feanor_math).
/// 
#[stability::unstable(feature = "enable")]
pub fn pre_smith<R, TL, TR, V>(ring: R, L: &mut TL, R: &mut TR, mut A: SubmatrixMut<V, El<R>>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        TL: TransformTarget<R::Type>,
        TR: TransformTarget<R::Type>,
        V: AsPointerToSlice<El<R>>
{
    // otherwise we might not terminate...
    assert!(ring.is_noetherian());
    assert!(ring.is_commutative());

    for k in 0..min(A.row_count(), A.col_count()) {
        let mut changed = true;
        while changed {
            changed = false;
            
            // eliminate the column
            for i in (k + 1)..A.row_count() {
                if ring.is_zero(A.at(i, k)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(i, k), A.at(k, k)) {
                    TransformRows(A.reborrow(), ring.get_ring()).subtract(ring.get_ring(), k, i, &quo);
                    L.subtract(ring.get_ring(), k, i, &quo);
                } else {
                    let (transform, _) = ring.get_ring().create_elimination_matrix(A.at(k, k), A.at(i, k));
                    TransformRows(A.reborrow(), ring.get_ring()).transform(ring.get_ring(), k, i, &transform);
                    L.transform(ring.get_ring(), k, i, &transform);
                }
            }
            
            // now eliminate the row
            for j in (k + 1)..A.col_count() {
                if ring.is_zero(A.at(k, j)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(k, j), A.at(k, k)) {
                    changed = true;
                    TransformCols(A.reborrow(), ring.get_ring()).subtract(ring.get_ring(), k, j, &quo);
                    R.subtract(ring.get_ring(), k, j, &quo);
                } else {
                    changed = true;
                    let (transform, _) = ring.get_ring().create_elimination_matrix(A.at(k, k), A.at(k, j));
                    TransformCols(A.reborrow(), ring.get_ring()).transform(ring.get_ring(), k, j, &transform);
                    R.transform(ring.get_ring(), k, j, &transform);
                }
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn solve_right_using_pre_smith<R, V1, V2, V3, A>(ring: R, mut lhs: SubmatrixMut<V1, El<R>>, mut rhs: SubmatrixMut<V2, El<R>>, out: SubmatrixMut<V3, El<R>>, allocator: A) -> SolveResult
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>, V3: AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(lhs.row_count(), rhs.row_count());
    assert_eq!(lhs.col_count(), out.row_count());
    assert_eq!(rhs.col_count(), out.col_count());

    let mut R: OwnedMatrix<El<R>, &A> = OwnedMatrix::identity_in(lhs.col_count(), lhs.col_count(), ring, &allocator);
    pre_smith(ring, &mut TransformRows(rhs.reborrow(), ring.get_ring()), &mut TransformCols(R.data_mut(), ring.get_ring()), lhs.reborrow());

    let mut result = OwnedMatrix::zero_in(lhs.col_count(), rhs.col_count(), ring, &allocator);
    for i in out.row_count()..rhs.row_count() {
        for j in 0..rhs.col_count() {
            if !ring.is_zero(rhs.at(i, j)) {
                return SolveResult::NoSolution;
            }
        }
    }
    for i in 0..min(result.row_count(), rhs.row_count()) {
        for j in 0..rhs.col_count() {
            *result.at_mut(i, j) = ring.clone_el(rhs.at(i, j));
        }
    }

    let zero = ring.zero();
    for i in 0..min(lhs.row_count(), lhs.col_count()) {
        let pivot = if i < lhs.col_count() { lhs.at(i, i) } else { &zero };
        for j in 0..rhs.col_count() {
            if let Some(quo) = ring.checked_left_div(rhs.at(i, j), pivot) {
                *result.at_mut(i, j) = quo;
            } else {
                return SolveResult::NoSolution;
            }
        }
    }
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(R.data()), TransposableSubmatrix::from(result.data()), TransposableSubmatrixMut::from(out), ring);
    return SolveResult::FoundSomeSolution;
}

#[stability::unstable(feature = "enable")]
pub fn determinant_using_pre_smith<R, V, A>(ring: R, mut matrix: SubmatrixMut<V, El<R>>, _allocator: A) -> El<R>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V:AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(matrix.row_count(), matrix.col_count());
    let mut unit_part_rows = ring.one();
    let mut unit_part_cols = ring.one();
    pre_smith(ring, &mut DetUnit { current_unit: &mut unit_part_rows }, &mut DetUnit { current_unit: &mut unit_part_cols }, matrix.reborrow());
    return ring.prod((0..matrix.row_count()).map(|i| ring.clone_el(matrix.at(i, i))).chain([unit_part_rows, unit_part_cols].into_iter()));
    }

struct DetUnit<'a, R: ?Sized + RingBase> {
    current_unit: &'a mut R::Element
}

impl<'a, R> TransformTarget<R> for DetUnit<'a, R>
    where R: ?Sized + RingBase
{
    fn subtract(&mut self, _ring: &R, _src: usize, _dst: usize, _factor: &<R as RingBase>::Element) {
        // determinant does not change
    }

    fn swap(&mut self, ring: &R, _i: usize, _j: usize) {
        ring.negate_inplace(&mut self.current_unit)
    }

    fn transform(&mut self, ring: &R, _i: usize, _j: usize, transform: &[<R as RingBase>::Element; 4]) {
        ring.mul_assign(&mut self.current_unit, ring.sub(ring.mul_ref(&transform[0], &transform[3]), ring.mul_ref(&transform[1], &transform[2])));
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::algorithms::linsolve::LinSolveRing;

#[cfg(test)]
fn multiply<'a, R: RingStore, V: AsPointerToSlice<El<R>>, I: IntoIterator<Item = Submatrix<'a, V, El<R>>>>(matrices: I, ring: R) -> OwnedMatrix<El<R>>
    where R::Type: 'a,
        V: 'a
{
    let mut it = matrices.into_iter();
    let fst = it.next().unwrap();
    let snd = it.next().unwrap();
    let mut new_result = OwnedMatrix::zero(fst.row_count(), snd.col_count(), &ring);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(fst), TransposableSubmatrix::from(snd), TransposableSubmatrixMut::from(new_result.data_mut()), &ring);
    let mut result = new_result;

    for m in it {
        let mut new_result = OwnedMatrix::zero(result.row_count(), m.col_count(), &ring);
        STANDARD_MATMUL.matmul(TransposableSubmatrix::from(result.data()), TransposableSubmatrix::from(m), TransposableSubmatrixMut::from(new_result.data_mut()), &ring);
        result = new_result;
    }
    return result;
}

#[test]
fn test_smith_integers() {
    let ring = StaticRing::<i64>::RING;
    let mut A = OwnedMatrix::new(
        vec![ 1, 2, 3, 4, 
                    2, 3, 4, 5,
                    3, 4, 5, 6 ], 
        4
    );
    let original_A = A.clone_matrix(&ring);
    let mut L: OwnedMatrix<i64> = OwnedMatrix::identity(3, 3, StaticRing::<i64>::RING);
    let mut R: OwnedMatrix<i64> = OwnedMatrix::identity(4, 4, StaticRing::<i64>::RING);
    pre_smith(ring, &mut TransformRows(L.data_mut(), ring.get_ring()), &mut TransformCols(R.data_mut(), ring.get_ring()), A.data_mut());
    
    assert_matrix_eq!(&ring, &[
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0, 0, 0]], &A);

    assert_matrix_eq!(&ring, &multiply([L.data(), original_A.data(), R.data()], ring), &A);
}

#[test]
fn test_smith_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let mut A = OwnedMatrix::new(
        vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ],
        4
    );
    let original_A = A.clone_matrix(&ring);
    let mut L: OwnedMatrix<u64> = OwnedMatrix::identity(5, 5, ring);
    let mut R: OwnedMatrix<u64> = OwnedMatrix::identity(4, 4, ring);
    pre_smith(ring, &mut TransformRows(L.data_mut(), ring.get_ring()), &mut TransformCols(R.data_mut(), ring.get_ring()), A.data_mut());

    assert_matrix_eq!(&ring, &[
        [8, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0, 15],
        [0, 0, 0, 0]], &A);
        
    assert_matrix_eq!(&ring, &multiply([L.data(), original_A.data(), R.data()], ring), &A);
}

#[test]
fn test_solve_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let A = OwnedMatrix::new(
        vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ],
        4
    );
    let B = OwnedMatrix::new(
        vec![11, 43, 10, 22,
                   18,  9, 27, 27,
                    8, 34,  7, 22,
                   41, 13, 40, 37,
                    3,  9,  3,  0],
        4
    );
    let mut solution: OwnedMatrix<_> = OwnedMatrix::zero(4, 4, ring);
    ring.get_ring().solve_right(A.clone_matrix(ring).data_mut(), B.clone_matrix(ring).data_mut(), solution.data_mut(), Global).assert_solved();

    assert_matrix_eq!(&ring, &multiply([A.data(), solution.data()], ring), &B);
}

#[test]
fn test_solve_int() {
    let ring = StaticRing::<i64>::RING;
    let A = OwnedMatrix::new(
        vec![3, 6, 2, 0, 4, 7,
                   5, 5, 4, 5, 5, 5],
        6
    );
    let B: OwnedMatrix<i64> = OwnedMatrix::identity(2, 2, ring);
    let mut solution: OwnedMatrix<i64> = OwnedMatrix::zero(6, 2, ring);
    ring.get_ring().solve_right(A.clone_matrix(ring).data_mut(), B.clone_matrix(ring).data_mut(), solution.data_mut(), Global).assert_solved();

    assert_matrix_eq!(&ring, &multiply([A.data(), solution.data()], &ring), &B);
}

#[test]
fn test_large() {
    let ring = zn_static::Zn::<16>::RING;
    let data_A = [
        [0, 0, 0, 0, 0, 0, 0, 0,11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ];
    let mut A: OwnedMatrix<u64> = OwnedMatrix::zero(6, 11, &ring);
    for i in 0..6 {
        for j in 0..11 {
            *A.at_mut(i, j) = data_A[i][j];
        }
    }
    let mut solution: OwnedMatrix<_> = OwnedMatrix::zero(11, 11, ring);
    assert!(ring.get_ring().solve_right(A.clone_matrix(&ring).data_mut(), A.data_mut(), solution.data_mut(), Global).is_solved());
}

#[test]
fn test_determinant() {
    let ring = StaticRing::<i64>::RING;
    let A = OwnedMatrix::new(
        vec![1, 0, 3, 
                   2, 1, 0, 
                   9, 8, 7],
        3
    );
    assert_el_eq!(ring, (7 + 48 - 27), determinant_using_pre_smith(ring, A.clone_matrix(&ring).data_mut(), Global));
}
