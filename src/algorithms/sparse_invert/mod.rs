use std::io::Write;
use std::mem::swap;
use std::cmp::Ordering;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use crate::algorithms::sparse_invert::matrix::{linear_combine_rows, preimage_echelon_form_matrix};
use crate::matrix::Matrix;
use crate::parallel::potential_parallel_for_each;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::{assert_matrix_eq, ring::*};
use crate::divisibility::*;
use crate::vector::{vec_fn, VectorView, VectorViewMut};

#[cfg(feature = "parallel")]
use crate::rayon::iter::IndexedParallelIterator;


pub mod builder;
mod matrix;

const EXTENSIVE_RUNTIME_ASSERTS: bool = true;

#[inline(always)]
fn mul_assign<'a, R, I>(ring: R, lhs: &InternalMatrixRef<El<R>>, rhs: I, mut out: Vec<InternalRow<El<R>>>) -> Vec<InternalRow<El<R>>>
    where R: RingStore + Copy,
        I: ExactSizeIterator<Item = &'a InternalRow<El<R>>> + Clone,
        El<R>: 'a
{
    lhs.check(&ring);
    // lhs is n x n
    // rhs is n x m
    let n = lhs.row_count();
    assert_eq!(rhs.len(), n);
    while out.len() < n {
        out.push(InternalRow::placeholder());
    }
    out.truncate(n);
    for i in 0..n {
        out[i] = std::mem::replace(&mut out[i], InternalRow::placeholder()).make_zero(&ring);
    }
    let mut tmp = InternalRow::placeholder();
    for i in 0..n {
        out[i] = linear_combine_rows(ring, lhs.local_row(i, 0), rhs.clone(), std::mem::replace(&mut out[i], InternalRow::placeholder()), &mut tmp);
    }
    return out;
}

#[inline(always)]
fn transform_2d<R>(ring: R, transform: &[[El<R>; 2]; 2], rows: [&mut InternalRow<El<R>>; 2], tmp: &mut [InternalRow<El<R>>; 2])
    where R: RingStore + Copy
{
    let [lhs, rhs] = rows;
    let lhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[0][0], &transform[0][1], std::mem::replace(&mut tmp[0], InternalRow::placeholder()));
    let rhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[1][0], &transform[1][1], std::mem::replace(&mut tmp[1], InternalRow::placeholder()));

    tmp[0] = std::mem::replace(lhs, lhs_new);
    tmp[1] = std::mem::replace(rhs, rhs_new);
}

#[inline(never)]
fn update_rows_with_transform<R>(ring: R, mut matrix: InternalMatrixRef<El<R>>, transform: &InternalMatrixRef<El<R>>) 
    where R: RingStore + Copy + Sync,
        El<R>: Send + Sync
{
    potential_parallel_for_each(
        matrix.concurrent_col_iter_mut(), 
        || Vec::new(), 
        |tmp, _, rows| 
    {
        let mut new = mul_assign(
            ring, 
            transform, 
            rows.iter(), 
            std::mem::replace(tmp, Vec::new())
        );
        for (target, new) in rows.iter_mut().zip(new.iter_mut()) {
            swap(target, new);
        }
        *tmp = new;
    });
}

fn global_eliminate_row<V, R>(ring: R, elim_coefficients: &InternalRow<El<R>>, pivot_rows: &InternalMatrixRef<El<R>>, mut row: V, tmp: [&mut InternalRow<El<R>>; 2])
    where R: RingStore + Copy,
        V: VectorViewMut<InternalRow<El<R>>>
{
    assert_eq!(pivot_rows.global_col_count(), row.len());
    let [tmp, new_row] = tmp;
    for col in 0..pivot_rows.global_col_count() {
        *new_row = linear_combine_rows(ring, &elim_coefficients, pivot_rows.row_iter().map(|r| vec_fn::VectorFn::at(&r, col)), std::mem::replace(new_row, InternalRow::placeholder()), tmp);
        *tmp = add_row_local::<_, true>(ring, row.at(col), &new_row, &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
        swap(tmp, row.at_mut(col));
    }
}

///
/// Makes a best-effort elimination of the given block column, by subtracting `x A` for some `x` and the pivot
/// matrix `A` from each row. The `x` is also added to `transform` to allow applying it later to the further columns.
/// 
/// If some rows were not completely eliminated (because a pivot is a non-unit), then any number (at least one) of indices of 
/// rows with remaining entries are returned. These rows have to be swapped into the local block and the whole operation 
/// must be performed again.
/// 
#[inline(never)]
fn eliminate_column<R>(ring: R, pivot_matrix: &InternalMatrixRef<El<R>>, mut column: InternalMatrixRef<El<R>>, mut transform: InternalMatrixRef<El<R>>) -> impl ExactSizeIterator<Item = usize>
    where R: DivisibilityRingStore + Copy + Sync,
        El<R>: Send + Sync,
        R::Type: DivisibilityRing
{
    assert_eq!(transform.global_col_count() * transform.n(), pivot_matrix.row_count());
    assert_eq!(pivot_matrix.global_col_count(), 1);
    assert_eq!(column.global_col_count(), 1);
    assert_eq!(pivot_matrix.n(), column.n());

    pivot_matrix.check(&ring);
    column.check(&ring);
    transform.check(&ring);

    let unreduced_row_index = (0..column.n()).map(|_| AtomicUsize::new(usize::MAX)).collect::<Vec<_>>();

    potential_parallel_for_each(column.concurrent_row_iter_mut().zip(transform.concurrent_row_iter_mut()), 
        || (InternalRow::placeholder(), InternalRow::placeholder(), InternalRow::placeholder()), 
        |(coefficients, new_row, tmp), row_index, (mut matrix_row, mut transform_row)|
    {
        *coefficients = preimage_echelon_form_matrix(ring, pivot_matrix, &matrix_row[0], std::mem::replace(coefficients, InternalRow::placeholder()));
        if !coefficients.is_empty() {
            global_eliminate_row(ring, coefficients, pivot_matrix, &mut matrix_row, [new_row, tmp]);
            let new_transform_row = add_row_local::<_, true>(ring, coefficients, &transform_row[0], &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
            *tmp = std::mem::replace(&mut transform_row[0], new_transform_row);
        }
        let unreduced_j = matrix_row[0].leading_entry().0;
        if unreduced_j < usize::MAX && unreduced_row_index[unreduced_j].load(std::sync::atomic::Ordering::SeqCst) == usize::MAX {
            unreduced_row_index[unreduced_j].store(row_index, std::sync::atomic::Ordering::SeqCst);
        }
    });
    column.check(&ring);
    transform.check(&ring);
    return unreduced_row_index.into_iter().map(|index| index.load(std::sync::atomic::Ordering::SeqCst)).filter(|index| *index != usize::MAX).collect::<Vec<_>>().into_iter();
}

#[inline(never)]
fn subtract_pivot_rows<R>(ring: R, pivot_rows: &InternalMatrixRef<El<R>>, mut body: InternalMatrixRef<El<R>>, elim_factors: &InternalMatrixRef<El<R>>)
    where R: RingStore + Copy + Sync,
        El<R>: Send + Sync
{
    assert_eq!(pivot_rows.global_col_count(), body.global_col_count());
    assert_eq!(elim_factors.global_col_count() * elim_factors.n(), pivot_rows.row_count());
    assert_eq!(elim_factors.row_count(), body.row_count());

    potential_parallel_for_each(body.concurrent_row_iter_mut(), || (InternalRow::placeholder(), InternalRow::placeholder()), |tmp, i, row| {
        global_eliminate_row(ring, elim_factors.local_row(i, 0), pivot_rows, row, [&mut tmp.0, &mut tmp.1])
    });
}

///
/// In the block of matrix given by `global_pivot`, performs unimodular row operation
/// to ensure that the element in the pivot position divides all elements below it. 
/// 
#[inline(never)]
fn local_make_pivot_ideal_gen<R>(ring: R, mut matrix: InternalMatrixRef<El<R>>, mut transform: InternalMatrixRef<El<R>>, local_pivot: (usize, usize), tmp: &mut [InternalRow<El<R>>; 2], original: &InternalMatrix<El<R>>) 
    where R: PrincipalIdealRingStore + Copy,
        R::Type: PrincipalIdealRing
{
    matrix.check(&ring);
    transform.check(&ring);
    let mut current = ring.clone_el(matrix.local_row(local_pivot.0, 0).at(local_pivot.1).unwrap_or(&ring.zero()));
    for i in (local_pivot.0 + 1)..matrix.row_count() {
        if ring.is_unit(&current) {
            break;
        }
        if let Some(entry) = matrix.local_row(i, 0).leading_entry_at(local_pivot.1) {
            if ring.checked_div(entry, &current).is_none() {
                let (s, t, d) = ring.ideal_gen(&current, entry);
                let local_transform = [[s, t], [ring.checked_div(entry, &d).unwrap(), ring.checked_div(&current, &d).unwrap()]];
                let (fst_row, snd_row) = matrix.two_local_rows(local_pivot.0, i, 0);
                transform_2d(ring, &local_transform, [fst_row, snd_row], tmp);
                let (fst_row, snd_row) = transform.two_local_rows(local_pivot.0, i, 0);
                transform_2d(ring, &local_transform, [fst_row, snd_row], tmp);
                current = d;
                
                let mut original2 = matrix.clone_to_owned(&ring);
                assert_left_equivalent(&ring, &original2.block(0..256, 0..1), &matrix);
            }
        }
    }
    matrix.check(&ring);
    transform.check(&ring);
}

fn local_eliminate_row<R>(ring: R, mut matrix: InternalMatrixRef<El<R>>, mut transform: InternalMatrixRef<El<R>>, local_pivot: (usize, usize), tmp: &mut [InternalRow<El<R>>; 2])
    where R: RingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check(&ring);
    transform.check(&ring);
    let pivot_entry = matrix.local_row(local_pivot.0, 0).at(local_pivot.1).unwrap();
    let pivot_entry = ring.clone_el(pivot_entry);

    for elim_i in 0..matrix.row_count() {
        if elim_i == local_pivot.0 {
            continue;
        }
        if let Some(factor) = matrix.local_row(elim_i, 0).at(local_pivot.1) {
            debug_assert!(elim_i < local_pivot.0 || matrix.local_row(elim_i, 0).leading_entry_at(local_pivot.1).is_some());
            let lhs_factor = ring.one();
            if let Some(quo) = ring.checked_div(factor, &pivot_entry) {
                let rhs_factor = ring.negate(quo);

                let new = add_row_local::<_, true>(ring, &matrix.local_row(elim_i, 0), &matrix.local_row(local_pivot.0, 0), &lhs_factor, &rhs_factor, std::mem::replace(&mut tmp[0], InternalRow::placeholder()));
                tmp[0] = std::mem::replace(matrix.local_row_mut(elim_i, 0), new);

                let new = add_row_local::<_, true>(ring, &transform.local_row(elim_i, 0), &transform.local_row(local_pivot.0, 0), &lhs_factor, &rhs_factor, std::mem::replace(&mut tmp[0], InternalRow::placeholder()));
                tmp[0] = std::mem::replace(transform.local_row_mut(elim_i, 0), new);
            } else {
                assert!(elim_i < local_pivot.0);
            }
        }
    }
    matrix.check(&ring);
    transform.check(&ring);
}

#[inline(never)]
fn local_row_echelon_optimistic<R>(ring: R, mut matrix: InternalMatrixRef<El<R>>, mut transform: InternalMatrixRef<El<R>>) -> usize
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing
{
    let original = matrix.clone_to_owned(&ring);
    matrix.check(&ring);
    transform.check(&ring);
    let col_block = matrix.n();
    let mut i = 0;
    let mut tmp = [InternalRow::placeholder(), InternalRow::placeholder()];
    for j in 0..col_block {
        local_make_pivot_ideal_gen(ring, matrix.reborrow(), transform.reborrow(), (i, j), &mut tmp, &original);
        if matrix.local_row(i, 0).leading_entry_at(j).is_some() {
            local_eliminate_row(ring, matrix.reborrow(), transform.reborrow(), (i, j), &mut tmp);
            i += 1;
            if i >= matrix.row_count() {
                return i;
            }
        }
    }
    matrix.check(&ring);
    transform.check(&ring);
    return i;
}

#[inline(never)]
fn blocked_row_echelon<R, const LOG: bool>(ring: R, matrix: &mut InternalMatrix<El<R>>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync
{
    if LOG {
        print!("[{}x{}]", matrix.row_count(), matrix.global_col_count() * matrix.n());
        std::io::stdout().flush().unwrap();
    }
    let start = Instant::now();

    let mut pivot_row = 0;
    let mut pivot_col = 0;
    let col_block = matrix.n();
    let col_block_count = matrix.global_col_count();

    // we have to pad matrix with n zero rows...
    for _ in 0..matrix.n() {
        matrix.add_row(&ring);
    }

    let row_count = matrix.row_count();
    let global_col_count = matrix.global_col_count();

    let mut row_block = matrix.n();
    let mut transform = InternalMatrix::empty(ring.zero());
    transform.clear(&ring, matrix.row_count(), 1);
    transform.set_n(row_block);
    transform.block(pivot_row..(pivot_row + row_block), 0..1).make_identity(&ring);

    while pivot_row + row_block < matrix.row_count() && pivot_col < col_block_count {

        let (mut pivot, column) = matrix.split_rows(pivot_row..(pivot_row + row_block), (pivot_row + row_block)..row_count, pivot_col..(pivot_col + 1));
        let (transform_pivot, transform_column) = transform.split_rows(pivot_row..(pivot_row + row_block), (pivot_row + row_block)..row_count, 0..1);
        let nonzero_row_count = local_row_echelon_optimistic(ring, pivot.reborrow(), transform_pivot);
        let mut reduction_result = eliminate_column(ring, &pivot, column, transform_column);

        if reduction_result.len() > 0 {
            let new_row_block = row_block + reduction_result.len();
            transform.set_n(new_row_block);
            for i in 0..reduction_result.len() {
                let swap_row_index = reduction_result.next().unwrap();
                matrix.swap_rows(pivot_row + row_block + i, swap_row_index + pivot_row + row_block);
                transform.swap_rows(pivot_row + row_block + i, swap_row_index + pivot_row + row_block);
                transform.local_row_mut(pivot_row + row_block + i, 0).append_one(&ring, row_block + i);
            }
            row_block = new_row_block;
            assert!(reduction_result.next().is_none());
        } else {
            assert!(nonzero_row_count <= row_block);
            update_rows_with_transform(ring, matrix.block(pivot_row..(pivot_row + row_block), (pivot_col + 1)..global_col_count), &transform.block(pivot_row..(pivot_row + row_block), 0..1));
            let (pivot_rows, rest) = matrix.split_rows(pivot_row..(pivot_row + row_block), (pivot_row + row_block)..row_count, (pivot_col + 1)..global_col_count);
            subtract_pivot_rows(ring, &pivot_rows, rest, &transform.block((pivot_row + row_block)..row_count, 0..1));
            pivot_row += nonzero_row_count;
            pivot_col += 1;
            row_block = col_block;
            transform.clear(&ring, matrix.row_count(), 1);
            transform.set_n(row_block);
            transform.block(pivot_row..(pivot_row + row_block), 0..1).make_identity(&ring);
            if LOG {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }
    }

    // remove the padding
    for _ in 0..matrix.n() {
        matrix.pop_row();
    }
    if LOG {
        let end = std::time::Instant::now();
        print!("[{}ms]", (end - start).as_millis());
        std::io::stdout().flush().unwrap();
    }
}

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
/// # use feanor_math::algorithms::smith::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::matrix::*;
/// let ring = Zn::new(18);
/// let modulo = ring.int_hom();
/// let mut A_transposed = DenseMatrix::zero(5, 4, ring);
/// let mut B_transposed = DenseMatrix::zero(5, 4, ring);
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
/// assert!(solve_right(&mut A_transposed.clone_matrix(&ring), B_transposed.clone_matrix(&ring), &ring).is_some());
/// assert!(solve_right(&mut B_transposed.clone_matrix(&ring), A_transposed.clone_matrix(&ring), &ring).is_some());
/// ```
/// 
#[inline(never)]
pub fn gb_sparse_row_echelon<R, const LOG: bool>(ring: R, matrix: SparseMatrixBuilder<R::Type>, block_size: usize) -> Vec<Vec<(usize, El<R>)>>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync
{
    let original = if EXTENSIVE_RUNTIME_ASSERTS {
        Some(matrix.clone_matrix(&ring))
    } else {
        None
    };

    let n = block_size;
    let mut matrix = matrix.into_internal_matrix(n, ring.get_ring());
    matrix.check(&ring);
    blocked_row_echelon::<_, LOG>(ring, &mut matrix);
    matrix.check(&ring);
    let result = matrix.destruct(&ring);

    if EXTENSIVE_RUNTIME_ASSERTS {
        let mut check = SparseMatrixBuilder::new(&ring);
        check.add_cols(original.as_ref().unwrap().col_count());
        for (i, row) in result.iter().enumerate() {
            check.add_row(i, row.iter().map(|(j, c)| (*j, ring.clone_el(c))));
        }
        assert_is_correct_row_echelon(ring, &original.unwrap(), &check);
    }

    return result;
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;

use self::builder::SparseMatrixBuilder;
use self::matrix::{InternalMatrix, InternalRow, add_row_local, InternalMatrixRef};

fn assert_left_equivalent_ex<R>(ring: R, original: &InternalMatrix<El<R>>, new: &InternalMatrixRef<El<R>>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    use crate::algorithms::smith;

    let n = original.row_count();
    let m = <_ as Matrix<R::Type>>::col_count(original);
    assert_eq!(n, new.row_count());
    assert_eq!(m, <_ as Matrix<R::Type>>::col_count(new));

    let mut original_transposed = smith::DenseMatrix::zero(m, n, &ring);
    let mut actual_transposed = smith::DenseMatrix::zero(m, n, &ring);
    for i in 0..n {
        for j in 0..m {
            *original_transposed.at_mut(j, i) = ring.clone_el(<_ as Matrix<R::Type>>::at(original, i, j));
            *actual_transposed.at_mut(j, i) = ring.clone_el(<_ as Matrix<R::Type>>::at(new, i, j));
        }
    }
    assert!(smith::solve_right::<&R>(&mut original_transposed.clone_matrix(&ring), actual_transposed.clone_matrix(&ring), &ring).is_some());
    assert!(smith::solve_right::<&R>(&mut actual_transposed.clone_matrix(&ring), original_transposed.clone_matrix(&ring), &ring).is_some());
}

fn assert_left_equivalent<R>(ring: R, original: &InternalMatrixRef<El<R>>, new: &InternalMatrixRef<El<R>>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    use crate::algorithms::smith;

    let n = original.row_count();
    let m = <_ as Matrix<R::Type>>::col_count(original);
    assert_eq!(n, new.row_count());
    assert_eq!(m, <_ as Matrix<R::Type>>::col_count(new));

    let mut original_transposed = smith::DenseMatrix::zero(m, n, &ring);
    let mut actual_transposed = smith::DenseMatrix::zero(m, n, &ring);
    for i in 0..n {
        for j in 0..m {
            *original_transposed.at_mut(j, i) = ring.clone_el(<_ as Matrix<R::Type>>::at(original, i, j));
            *actual_transposed.at_mut(j, i) = ring.clone_el(<_ as Matrix<R::Type>>::at(new, i, j));
        }
    }
    assert!(smith::solve_right::<&R>(&mut original_transposed.clone_matrix(&ring), actual_transposed.clone_matrix(&ring), &ring).is_some());
    assert!(smith::solve_right::<&R>(&mut actual_transposed.clone_matrix(&ring), original_transposed.clone_matrix(&ring), &ring).is_some());
}


fn assert_is_correct_row_echelon<R>(ring: R, original: &SparseMatrixBuilder<R::Type>, row_echelon_form: &SparseMatrixBuilder<R::Type>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    use crate::algorithms::smith;

    let n = original.row_count();
    let m = original.col_count();
    assert_eq!(n, row_echelon_form.row_count());
    assert_eq!(m, row_echelon_form.col_count());

    for i in 1..n {
        assert!(row_echelon_form.rows[i].len() == 0 || row_echelon_form.rows[i][0].0 > row_echelon_form.rows[i - 1][0].0);
    }

    let mut original_transposed = smith::DenseMatrix::zero(m, n, &ring);
    let mut actual_transposed = smith::DenseMatrix::zero(m, n, &ring);
    for i in 0..n {
        for j in 0..m {
            *original_transposed.at_mut(j, i) = ring.clone_el(original.at(i, j));
            *actual_transposed.at_mut(j, i) = ring.clone_el(row_echelon_form.at(i, j));
        }
    }
    assert!(smith::solve_right::<&R>(&mut original_transposed.clone_matrix(&ring), actual_transposed.clone_matrix(&ring), &ring).is_some());
    assert!(smith::solve_right::<&R>(&mut actual_transposed.clone_matrix(&ring), original_transposed.clone_matrix(&ring), &ring).is_some());
}

#[test]
fn test_gb_sparse_row_echelon_3x5() {
    let R = Zn::<7>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([0, 0, 3, 0, 4]));
    matrix.add_row(1, sparsify([0, 2, 1, 0, 4]));
    matrix.add_row(2, sparsify([6, 0, 1, 0, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
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

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(6);
    matrix.add_row(0, sparsify([2, 3, 0, 1, 0, 1]));
    matrix.add_row(1, sparsify([3, 13, 0, 1, 0, 1]));
    matrix.add_row(2, sparsify([0, 0, 11, 1, 0, 1]));
    matrix.add_row(3, sparsify([0, 0, 0, 0, 4, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
        actual.add_cols(6);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_large() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 10]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
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
        let mut actual = SparseMatrixBuilder::new(&R);
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

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([4, 4, 6, 5, 4]));
    matrix.add_row(1, sparsify([5, 0, 6, 2, 1]));
    matrix.add_row(2, sparsify([2, 0, 7, 1, 5]));
    matrix.add_row(3, sparsify([3, 4, 3, 2, 5]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
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

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([9, 3, 0, 1, 1]));
    matrix.add_row(1, sparsify([6, 13, 0, 0, 1]));
    matrix.add_row(2, sparsify([0, 0, 11, 0, 1]));
    matrix.add_row(3, sparsify([0, 12, 0, 0, 1]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}