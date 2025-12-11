use std::alloc::Allocator;

use tracing::instrument;

use crate::matrix::*;
use crate::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::ring::*;
use crate::seq::*;

use super::{LinSolveRing, SolveResult};

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn solve_right_over_extension<R, V1, V2, V3, A>(ring: R, lhs: SubmatrixMut<V1, El<R>>, rhs: SubmatrixMut<V2, El<R>>, mut out: SubmatrixMut<V3, El<R>>, allocator: A) -> SolveResult
    where R: RingStore,
        R::Type: FreeAlgebra,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing,
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(lhs.row_count(), rhs.row_count());
    assert_eq!(lhs.col_count(), out.row_count());
    assert_eq!(rhs.col_count(), out.col_count());

    let mut expanded_lhs = OwnedMatrix::zero_in(lhs.row_count() * ring.rank(), lhs.col_count() * ring.rank(), ring.base_ring(), &allocator);
    let mut current;
    let g = ring.canonical_gen();
    for i in 0..lhs.row_count() {
        for j in 0..lhs.col_count() {
            current = ring.clone_el(lhs.at(i, j));
            for l in 0..ring.rank() {
                let current_wrt_basis = ring.wrt_canonical_basis(&current);
                for k in 0..ring.rank() {
                    *expanded_lhs.at_mut(i * ring.rank() + k, j * ring.rank() + l) = current_wrt_basis.at(k);
                }
                drop(current_wrt_basis);
                ring.mul_assign_ref(&mut current, &g);
            }
        }
    }

    let mut expanded_rhs = OwnedMatrix::zero_in(rhs.row_count() * ring.rank(), rhs.col_count(), ring.base_ring(), &allocator);
    for i in 0..rhs.row_count() {
        for j in 0..rhs.col_count() {
            let value_wrt_basis = ring.wrt_canonical_basis(rhs.at(i, j));
            for k in 0..ring.rank() {
                *expanded_rhs.at_mut(i * ring.rank() + k, j) = value_wrt_basis.at(k);
            }
        }
    }

    let mut solution = OwnedMatrix::zero_in(lhs.col_count() * ring.rank(), rhs.col_count(), ring.base_ring(), &allocator);
    let sol = ring.base_ring().get_ring().solve_right(expanded_lhs.data_mut(), expanded_rhs.data_mut(), solution.data_mut(), &allocator);

    if !sol.is_solved() {
        return sol;
    }

    for i in 0..lhs.col_count() {
        for j in 0..rhs.col_count() {
            let res_value = ring.from_canonical_basis((0..ring.rank()).map(|k| ring.base_ring().clone_el(solution.at(i * ring.rank() + k, j))));
            *out.at_mut(i, j) = res_value;
        }
    }

    return sol;
}

#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::algorithms::matmul::{MatmulAlgorithm, STANDARD_MATMUL};
#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_solve() {
    LogAlgorithmSubscriber::init_test();
    let base_ring = zn_static::Zn::<15>::RING;
    // Z_15[X]/(X^3 + X^2 + 1);  X^3 + X^2 + 1 = (X + 2)(X + 2X + 2) mod 3, but it is irreducible mod 5
    let ring = FreeAlgebraImpl::new(base_ring, 3, [14, 0, 14]);
    let el = |coeffs: [u64; 3]| ring.from_canonical_basis(coeffs);

    let data_A = [
        DerefArray::from([ el([1, 0, 0]), el([0, 0, 0]) ]),
        DerefArray::from([ el([2, 1, 0]), el([0, 0, 0]) ]),
        DerefArray::from([ el([0, 0, 0]), el([0, 1, 0]) ]),
    ];
    let data_B = [
        DerefArray::from([ el([10, 10, 5]) ]),
        DerefArray::from([ el([0, 0, 0]) ]),
        DerefArray::from([ el([1, 0, 0]) ]),
    ];
    let mut A = OwnedMatrix::from_fn_in(3, 2, |i, j| ring.clone_el(&data_A[i][j]), Global);
    let mut B = OwnedMatrix::from_fn_in(3, 1, |i, j| ring.clone_el(&data_B[i][j]), Global);
    let mut sol: OwnedMatrix<_> = OwnedMatrix::zero(2, 1, &ring);

    solve_right_over_extension(&ring, A.data_mut(), B.data_mut(), sol.data_mut(), Global).assert_solved();

    let A = OwnedMatrix::from_fn_in(3, 2, |i, j| ring.clone_el(&data_A[i][j]), Global);
    let B = OwnedMatrix::from_fn_in(3, 1, |i, j| ring.clone_el(&data_B[i][j]), Global);
    let mut prod: OwnedMatrix<_> = OwnedMatrix::zero(3, 1, &ring);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(A.data()), TransposableSubmatrix::from(sol.data()), TransposableSubmatrixMut::from(prod.data_mut()), &ring);

    assert_matrix_eq!(&ring, &B, &prod);

    let data_B = [
        DerefArray::from([ el([8, 8, 3]) ]),
        DerefArray::from([ el([0, 0, 0]) ]),
        DerefArray::from([ el([1, 0, 0]) ]),
    ];
    let mut A = OwnedMatrix::from_fn_in(3, 2, |i, j| ring.clone_el(&data_A[i][j]), Global);
    let mut B = OwnedMatrix::from_fn_in(3, 1, |i, j| ring.clone_el(&data_B[i][j]), Global);
    let mut sol: OwnedMatrix<_> = OwnedMatrix::zero(2, 1, &ring);
    assert!(!solve_right_over_extension(&ring, A.data_mut(), B.data_mut(), sol.data_mut(), Global).is_solved());
}

#[test]
fn test_invert() {
    LogAlgorithmSubscriber::init_test();
    let base_ring = zn_static::Zn::<15>::RING;
    // Z_15[X]/(X^3 + X^2 + 1);  X^3 + X^2 + 1 = (X + 2)(X + 2X + 2) mod 3, but it is irreducible mod 5
    let ring = FreeAlgebraImpl::new(base_ring, 3, [14, 0, 14]);

    let matrix = OwnedMatrix::from_fn(2, 2, |i, j| if i == 0 || j == 0 {
        ring.one()
    } else {
        ring.sub(ring.canonical_gen(), ring.one())
    });
    let mut inverse = OwnedMatrix::zero(2, 2, &ring);
    solve_right_over_extension(&ring, matrix.clone_matrix(&ring).data_mut(), OwnedMatrix::identity(2, 2, &ring).data_mut(), inverse.data_mut(), Global).assert_solved();

    let mut result = OwnedMatrix::zero(2, 2, &ring);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(matrix.data()), TransposableSubmatrix::from(inverse.data()), TransposableSubmatrixMut::from(result.data_mut()), &ring);

    assert_matrix_eq!(&ring, OwnedMatrix::identity(2, 2, &ring), result);
}