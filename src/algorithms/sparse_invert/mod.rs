use matrix::SparseMatrix;
use row_echelon::InternalRow;

use crate::matrix::{OwnedMatrix, AsFirstElement, AsPointerToSlice, Submatrix, SubmatrixMut};
use crate::pid::PrincipalIdealRing;
use crate::ring::*;

pub mod matrix;
mod row_echelon;

///
/// Computes the row echelon form of the given matrix `A`, using elementary row
/// operations. Be careful if the ring is not a field, however!
/// 
/// # What about rings that are not fields?
/// 
/// In the case that the underlying ring is not a field (or might even have
/// zero-divisors), this is considered to be any matrix `B` that follows the
/// "row-echelon" pattern (i.e. the first nonzero entry in each row has a larger
/// column index than the first non-zero entry in the previous row), and there is
/// a matrix `T` with unit determinant such that `A = T B`. Not that this does
/// not have to be unique, and in fact, weird situations are possible.
/// 
/// This is demonstrated by the following example.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::smith::solve_right;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::matrix::*;
/// let ring = Zn::new(18);
/// let modulo = ring.int_hom();
/// let mut A_transposed: OwnedMatrix<ZnEl> = OwnedMatrix::zero(5, 4, ring);
/// let mut B_transposed: OwnedMatrix<ZnEl> = OwnedMatrix::zero(5, 4, ring);
/// 
/// // both matrices are in row-echelon form and the same except for the last column
/// let data_A = [
///     [3, 8, 0, 1, 0],
///     [0, 3, 0,16, 0],
///     [0, 0,11, 0, 1],
///     [0, 0, 0, 8, 1]
/// ];
/// let data_B = [
///     [3, 8, 0, 1, 0],
///     [0, 3, 0,16,14],
///     [0, 0,11, 0, 1],
///     [0, 0, 0, 8, 5]
/// ];
/// for (i, j) in (0..4).flat_map(|i| (0..5).map(move |j| (i, j))) {
///     *A_transposed.at_mut(j, i) = modulo.map(data_A[i][j]);
///     *B_transposed.at_mut(j, i) = modulo.map(data_B[i][j]);
/// }
/// // this shows that in fact, A and B are equivalent up to row operations!
/// // In other words, there are S, T such that STA = SB = A, which implies that
/// // det(S) det(T) = 1, so both S and T have unit determinant
/// assert!(solve_right(A_transposed.clone_matrix(&ring).data_mut(), B_transposed.clone_matrix(&ring).data_mut(), &ring).is_some());
/// assert!(solve_right(B_transposed.clone_matrix(&ring).data_mut(), A_transposed.clone_matrix(&ring).data_mut(), &ring).is_some());
/// ```
/// 
#[inline(never)]
#[stability::unstable(feature = "enable")]
pub fn gb_sparse_row_echelon<R, const LOG: bool>(ring: R, matrix: SparseMatrix<R::Type>, block_size: usize) -> Vec<Vec<(usize, El<R>)>>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync
{
    let original = if row_echelon::EXTENSIVE_RUNTIME_ASSERTS {
        Some(matrix.clone_matrix(&ring))
    } else {
        None
    };

    let row_count = matrix.row_count();
    let col_count = matrix.col_count();
    let n = block_size;
    let global_col_count = (col_count - 1) / n + 1;
    let mut matrix = matrix.into_internal_matrix(n, ring.get_ring());
    let mut matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut matrix[..], row_count + n, global_col_count);
    row_echelon::blocked_row_echelon::<_, _, LOG>(ring, matrix.reborrow(), n);

    let matrix_ref = &matrix.as_const();
    let result = (0..row_count).map(move |i| (0..global_col_count).flat_map(move |j| 
        matrix_ref.at(i, j).data.iter().rev().skip(1).rev().map(move |(k, c)| (k + j * n, ring.clone_el(c)))
    ).collect::<Vec<_>>()).collect::<Vec<_>>();

    if row_echelon::EXTENSIVE_RUNTIME_ASSERTS {
        let mut check = SparseMatrix::new(&ring);
        check.add_cols(original.as_ref().unwrap().col_count());
        for (i, row) in result.iter().enumerate() {
            check.add_row(i, row.iter().map(|(j, c)| (*j, ring.clone_el(c))));
        }
        assert_is_correct_row_echelon(ring, &original.unwrap(), &check);
    }

    return result;
}

fn assert_left_equivalent<R, V1, V2>(ring: R, original: Submatrix<V1, InternalRow<El<R>>>, new: Submatrix<V2, InternalRow<El<R>>>, n: usize)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V1: AsPointerToSlice<InternalRow<El<R>>>,
        V2: AsPointerToSlice<InternalRow<El<R>>>
{
    use crate::algorithms::smith;

    assert_eq!(original.row_count(), new.row_count());
    assert_eq!(original.col_count(), new.col_count());

    let zero = ring.zero();

    let mut original_transposed: OwnedMatrix<_> = OwnedMatrix::zero(original.col_count() * n, original.row_count(), &ring);
    let mut actual_transposed: OwnedMatrix<_> = OwnedMatrix::zero(original.col_count() * n, original.row_count(), &ring);
    for i in 0..original.row_count() {
        for j in 0..original.col_count() {
            for k in 0..n {
                *original_transposed.at_mut(j * n + k, i) = ring.clone_el(original.at(i, j).at(k).unwrap_or(&zero));
                *actual_transposed.at_mut(j * n + k, i) = ring.clone_el(new.at(i, j).at(k).unwrap_or(&zero));
            }
        }
    }
    if !smith::solve_right::<&R, _, _>(original_transposed.clone_matrix(&ring).data_mut(), actual_transposed.clone_matrix(&ring).data_mut(), &ring).is_some() {
        // println!("{:?}", original.row_iter().map(|row| row.iter().enumerate().flat_map(move |(j, local_row)| local_row.data.iter().rev().skip(1).rev().map(move |(k, c)| (j * n + k, format!("{}", ring.format(c))))).collect::<Vec<_>>()).collect::<Vec<_>>());
        panic!();
    }
    if !smith::solve_right::<&R, _, _>(actual_transposed.clone_matrix(&ring).data_mut(), original_transposed.clone_matrix(&ring).data_mut(), &ring).is_some() {
        // println!("{:?}", original.row_iter().map(|row| row.iter().enumerate().flat_map(move |(j, local_row)| local_row.data.iter().rev().skip(1).rev().map(move |(k, c)| (j * n + k, format!("{}", ring.format(c))))).collect::<Vec<_>>()).collect::<Vec<_>>());
        panic!();
    }
}

fn assert_is_correct_row_echelon<R>(ring: R, original: &SparseMatrix<R::Type>, row_echelon_form: &SparseMatrix<R::Type>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    use crate::algorithms::smith;

    let n = original.row_count();
    let m = original.col_count();
    assert_eq!(n, row_echelon_form.row_count());
    assert_eq!(m, row_echelon_form.col_count());

    assert!(row_echelon_form.is_echelon());

    let mut original_transposed: OwnedMatrix<_> = OwnedMatrix::zero(m, n, &ring);
    let mut actual_transposed: OwnedMatrix<_> = OwnedMatrix::zero(m, n, &ring);
    for i in 0..n {
        for j in 0..m {
            *original_transposed.at_mut(j, i) = ring.clone_el(original.at(i, j));
            *actual_transposed.at_mut(j, i) = ring.clone_el(row_echelon_form.at(i, j));
        }
    }
    assert!(smith::solve_right::<&R, _, _>(original_transposed.clone_matrix(&ring).data_mut(), actual_transposed.clone_matrix(&ring).data_mut(), &ring).is_some());
    assert!(smith::solve_right::<&R, _, _>(actual_transposed.clone_matrix(&ring).data_mut(), original_transposed.clone_matrix(&ring).data_mut(), &ring).is_some());
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;

#[test]
fn test_gb_sparse_row_echelon_3x5() {
    let R = Zn::<7>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([0, 0, 3, 0, 4]));
    matrix.add_row(1, sparsify([0, 2, 1, 0, 4]));
    matrix.add_row(2, sparsify([6, 0, 1, 0, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrix::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_4x6() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 6]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(6);
    matrix.add_row(0, sparsify([2, 3, 0, 1, 0, 1]));
    matrix.add_row(1, sparsify([3, 13, 0, 1, 0, 1]));
    matrix.add_row(2, sparsify([0, 0, 11, 1, 0, 1]));
    matrix.add_row(3, sparsify([0, 0, 0, 0, 4, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrix::new(&R);
        actual.add_cols(6);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_recompute_pivot() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 3]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));
    
    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(3);
    matrix.add_row(0, sparsify([0, 1, 1]));
    matrix.add_row(1, sparsify([0, 0, 0]));
    matrix.add_row(2, sparsify([0, 1, 0]));
    matrix.add_row(3, sparsify([1, 0, 0]));
    
    let mut actual = SparseMatrix::new(&R);
    actual.add_cols(3);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(actual.row_count(), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
}

#[test]
fn test_gb_sparse_row_echelon_swap_in_twice() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 3]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));
    
    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(3);
    matrix.add_row(0, sparsify([0, 0, 1]));
    matrix.add_row(1, sparsify([0, 0, 0]));
    matrix.add_row(2, sparsify([1, 1, 1]));
    matrix.add_row(3, sparsify([1, 2, 3]));
    matrix.add_row(3, sparsify([1, 3, 3]));
    
    let mut actual = SparseMatrix::new(&R);
    actual.add_cols(3);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(actual.row_count(), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
}

#[test]
fn test_gb_sparse_row_echelon_swap_in_twice_nontrivial_transform() {
    let R = Zn::<8>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));
    
    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([2, 0, 0, 1, 1]));
    matrix.add_row(1, sparsify([0, 0, 0, 2, 0]));
    matrix.add_row(2, sparsify([0, 0, 0, 3, 0]));
    matrix.add_row(3, sparsify([2, 0, 0, 4, 1]));
    matrix.add_row(4, sparsify([1, 2, 2, 5, 0]));
    matrix.add_row(5, sparsify([1, 4, 3, 6, 0]));
    
    let mut actual = SparseMatrix::new(&R);
    actual.add_cols(5);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 3) {
        actual.add_row(actual.row_count(), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
}

#[test]
fn test_gb_sparse_row_echelon_large() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 10]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(10);
    matrix.add_row(0, sparsify([ 0,  9,  2,  5, 12, 16,  8,  4, 15,  8]));
    matrix.add_row(1, sparsify([ 1,  2,  0,  7, 13, 13, 16,  5,  1,  3]));
    matrix.add_row(2, sparsify([11,  2, 10, 16, 10,  4, 16,  3,  9, 12]));
    matrix.add_row(3, sparsify([ 5,  7, 12,  5,  1, 11, 14, 15, 12,  7]));
    matrix.add_row(4, sparsify([15,  9, 15, 10,  8,  0, 16,  7, 12, 12]));
    matrix.add_row(5, sparsify([12,  2,  7,  7, 11, 16,  1,  7, 15,  5]));
    matrix.add_row(6, sparsify([12,  9, 13, 16,  6,  0,  3,  8, 16,  2]));
    matrix.add_row(7, sparsify([ 1, 14, 16, 14,  4,  7,  1,  2,  0,  0]));
    matrix.add_row(8, sparsify([ 9, 10,  5,  7,  4,  1,  6,  8, 15, 11]));
    matrix.add_row(9, sparsify([ 6,  0,  6, 12, 15, 11, 15,  0,  2,  6]));

    for block_size in 1..15 {
        let mut actual = SparseMatrix::new(&R);
        actual.add_cols(10);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_local_ring() {
    let R = Zn::<8>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([4, 4, 6, 5, 4]));
    matrix.add_row(1, sparsify([5, 0, 6, 2, 1]));
    matrix.add_row(2, sparsify([2, 0, 7, 1, 5]));
    matrix.add_row(3, sparsify([3, 4, 3, 2, 5]));

    for block_size in 1..10 {
        let mut actual = SparseMatrix::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_no_field() {
    let R = Zn::<18>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([9, 3, 0, 1, 1]));
    matrix.add_row(1, sparsify([6, 13, 0, 0, 1]));
    matrix.add_row(2, sparsify([0, 0, 11, 0, 1]));
    matrix.add_row(3, sparsify([0, 12, 0, 0, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrix::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_pivot_row_transform_update() {
    let R = Fp::<65537>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([1, 1, 0, 1, 0]));
    matrix.add_row(1, sparsify([0, 0, 0, 1, 1]));
    matrix.add_row(2, sparsify([0, 0, 0, 0, 0]));
    matrix.add_row(3, sparsify([0, 0, 0, 0, 0]));
    matrix.add_row(4, sparsify([1, 2, 0, 0, 0]));

    let mut actual = SparseMatrix::new(&R);
    actual.add_cols(5);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 4) {
        actual.add_row(actual.row_count(), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
}

#[test]
fn test_bad_swapping_order() {
    let R = Zn::<3>::RING;
    let sparsify = |row: [u64; 10]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrix<_> = SparseMatrix::new(&R);
    matrix.add_cols(10);
    matrix.add_row(0, sparsify([0, 0, 1, 0, 0, 0, 1, 0, 0, 0]));
    matrix.add_row(1, sparsify([0, 0, 0, 1, 0, 0, 0, 1, 0, 0]));
    matrix.add_row(2, sparsify([0, 1, 0, 0, 1, 0, 0, 0, 1, 0]));
    matrix.add_row(3, sparsify([1, 0, 0, 0, 0, 1, 0, 0, 0, 1]));

    let mut actual = SparseMatrix::new(&R);
    actual.add_cols(10);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(actual.row_count(), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
    
}
