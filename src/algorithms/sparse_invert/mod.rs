use std::{cmp::Ordering, io::Write, time::Instant};

use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::matrix::dense::DenseMatrix;
use crate::matrix::submatrix::*;
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::vector::{SwappableVectorViewMut, VectorView, VectorViewMut};

use self::builder::SparseMatrixBuilder;

pub mod builder;

pub const EXTENSIVE_RUNTIME_ASSERTS: bool = false;
pub const EXTENSIVE_LOG: bool = false;

pub struct InternalRow<T> {
    data: Vec<(usize, T)>
}

impl<T> InternalRow<T> {

    pub fn at<'a>(&'a self, local_j: usize) -> Option<&'a T> {
        self.data.binary_search_by_key(&local_j, |(j, _)| *j).ok().map(|idx| &self.data[idx].1)
    }

    pub fn clone_row<R>(&self, ring: &R) -> Self
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        InternalRow {
            data: self.data.iter().map(|(j, c)| (*j, ring.clone_el(c))).collect()
        }
    }

    ///
    /// Creates a new, defined (i.e. no UB) but invalid value for an InternalRow.
    /// This can be used to temporarily fill a variable, or initialize (e.g. via
    /// [`make_zero()`]). This function will never allocate memory.
    /// 
    pub const fn placeholder() -> InternalRow<T> {
        InternalRow { data: Vec::new() }
    }

    pub fn make_zero<R>(&mut self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.clear();
        self.data.push((usize::MAX, ring.zero()));
    }

    pub fn check<R>(&self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        if EXTENSIVE_RUNTIME_ASSERTS {
            assert!(self.data.is_sorted_by_key(|(idx, _)| *idx));
            assert!(self.data.iter().rev().skip(1).all(|(j, _)| *j < usize::MAX));
            assert!(self.data.iter().rev().skip(1).all(|(_, x)| !ring.is_zero(x)));
            assert!((1..self.data.len()).all(|k| self.data[k - 1].0 != self.data[k].0));
            assert!(self.data.last().unwrap().0 == usize::MAX);
            assert!(self.data.len() == 1 || self.data[self.data.len() - 2].0 < usize::MAX);
        }
    }
    
    pub fn leading_entry<'a>(&'a self) -> (usize, &'a T) {
        let (j, c) = &self.data[0];
        return (*j, c);
    }

    pub fn leading_entry_at<'a>(&'a self, col: usize) -> Option<&'a T> {
        debug_assert!(self.leading_entry().0 >= col);
        if self.leading_entry().0 == col {
            Some(self.leading_entry().1)
        } else {
            None
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.len() == 1
    }

    pub fn append_one<R>(&mut self, ring: &R, j: usize)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.insert(self.data.len() - 1, (j, ring.one()));
        self.check(ring);
    }
}

pub fn add_row_local<R, const LHS_FACTOR_ONE: bool>(ring: R, lhs: &InternalRow<El<R>>, rhs: &InternalRow<El<R>>, lhs_factor: &El<R>, rhs_factor: &El<R>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore
{
    lhs.check(&ring);
    rhs.check(&ring);
    let mut lhs_idx = 0;
    let mut rhs_idx = 0;
    out.data.clear();
    while lhs_idx + 1 < lhs.data.len() || rhs_idx + 1 < rhs.data.len() {
        let lhs_j = lhs.data[lhs_idx].0;
        let rhs_j = rhs.data[rhs_idx].0;

        match lhs_j.cmp(&rhs_j) {
            Ordering::Less => {
                debug_assert!(!ring.is_zero(&lhs.data[lhs_idx].1));
                let lhs_val = if LHS_FACTOR_ONE {
                    ring.clone_el(&lhs.data[lhs_idx].1)
                } else {
                    ring.mul_ref(&lhs.data[lhs_idx].1, lhs_factor)
                };
                if LHS_FACTOR_ONE || !ring.is_zero(&lhs_val) {
                    debug_assert!(!ring.is_zero(&lhs_val));
                    out.data.push((lhs_j, lhs_val))
                };
                lhs_idx += 1;
            },
            Ordering::Greater => {
                let rhs_val = ring.mul_ref(&rhs.data[rhs_idx].1, rhs_factor);
                if !ring.is_zero(&rhs_val) {
                    out.data.push((rhs_j, rhs_val));
                }
                rhs_idx += 1;
            },
            Ordering::Equal => {
                let lhs_val = if LHS_FACTOR_ONE { ring.clone_el(&lhs.data[lhs_idx].1) } else { ring.mul_ref(&lhs.data[lhs_idx].1, lhs_factor) };
                let value = ring.add(lhs_val, ring.mul_ref(&rhs.data[rhs_idx].1, rhs_factor));
                if !ring.is_zero(&value) {
                    out.data.push((lhs_j, value));
                }
                lhs_idx += 1;
                rhs_idx += 1;
            }
        }
    }
    debug_assert!(lhs_idx + 1 == lhs.data.len() && rhs_idx + 1 == rhs.data.len());
    out.data.push((usize::MAX, ring.zero()));
    out.check(&ring);
    return out;
}

pub fn linear_combine_rows<'a, R, V>(ring: R, coeffs: &InternalRow<El<R>>, rows: Column<V, InternalRow<El<R>>>, mut out: InternalRow<El<R>>, tmp: &mut InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore + Copy,
        V: AsPointerToSlice<InternalRow<El<R>>>,
        El<R>: 'a
{
    coeffs.check(&ring);
    if coeffs.is_empty() {
        out.make_zero(&ring);
        return out;
    }

    out.data.clear();
    let last_idx = coeffs.leading_entry().0;
    let scale = coeffs.leading_entry().1;
    out.data.extend(
        rows.at(last_idx).data.iter().map(|(j, c) | (*j, ring.mul_ref(c, scale)))
            .filter(|(_, c)| !ring.is_zero(c))
            .chain([(usize::MAX, ring.zero())].into_iter())
    );

    tmp.make_zero(&ring);

    let lhs_factor = ring.one();
    for (idx, c) in coeffs.data[1..(coeffs.data.len() - 1)].iter() {
        *tmp = add_row_local::<_, true>(ring, &out, rows.at(*idx), &lhs_factor, c, std::mem::replace(tmp, InternalRow::placeholder()));
        std::mem::swap(&mut out, tmp);
    }
    out.check(&ring);
    return out;
}
    
fn sub_assign_mul<R, V>(ring: R, elim_coefficients: &InternalRow<El<R>>, pivot_rows: Column<V, InternalRow<El<R>>>, row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>, tmp: &mut InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore + Copy,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    elim_coefficients.check(&ring);
    row.check(&ring);
    out = linear_combine_rows(ring, &elim_coefficients, pivot_rows, out, tmp);
    *tmp = add_row_local::<_, true>(ring, row, &out, &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
    std::mem::swap(tmp, &mut out);
    out.check(&ring);
    return out;
}

///
/// Computes a vector `result` such that `result * echelon_form_matrix = current_row`, if it exists.
/// Note that if the vector does not exist, this function will not throw an error but return a
/// vector that can be thought of as a "best effort solution". This is important in combination
/// with the optimistic elimination strategy.
/// 
pub fn preimage_echelon_form_matrix<R, V>(ring: R, echelon_form_matrix: Column<V, InternalRow<El<R>>>, current_row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    current_row.check(&ring);
    let zero = ring.zero();
    out.make_zero(&ring);
    out.data.pop();
    for (i, row) in echelon_form_matrix.iter().enumerate() {
        let (j, pivot) = row.leading_entry();
        let mut current = ring.clone_el(current_row.at(j).unwrap_or(&zero));
        for (k, c) in out.data.iter().rev().skip(1).rev() {
            ring.sub_assign(&mut current, ring.mul_ref(c, echelon_form_matrix.at(*k).at(j).unwrap_or(&zero)));
        }
        if !ring.is_zero(&current) {
            if let Some(quo) = ring.checked_div(&current, pivot) {
                out.data.push((i, ring.negate(quo)));
            }
        }
    }
    out.data.push((usize::MAX, ring.one()));
    out.check(&ring);
    return out;
}

fn mul_assign<R, V1, V2>(ring: R, lhs: Column<V1, InternalRow<El<R>>>, mut rhs: ColumnMut<V2, InternalRow<El<R>>>, tmp: &mut Vec<InternalRow<El<R>>>)
    where R: RingStore + Copy,
        V1: AsPointerToSlice<InternalRow<El<R>>>,
        V2: AsPointerToSlice<InternalRow<El<R>>>
{
    let n = lhs.len();
    assert_eq!(rhs.len(), n);
    
    while tmp.len() < n {
        tmp.push(InternalRow::placeholder());
    }
    for i in 0..n {
        tmp[i].make_zero(&ring);
    }
    let mut tmp_row = InternalRow::placeholder();
    for i in 0..n {
        tmp[i] = linear_combine_rows(ring, lhs.at(i), rhs.as_const(), std::mem::replace(&mut tmp[i], InternalRow::placeholder()), &mut tmp_row);
    }
    for i in 0..n {
        std::mem::swap(rhs.at_mut(i), &mut tmp[i]);
    }
}

fn swap_rows<R, V>(_ring: R, matrix: SubmatrixMut<V, InternalRow<El<R>>>, fst_row_i: usize, snd_row_i: usize, _n: usize)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    for mut col in matrix.col_iter() {
        col.swap(fst_row_i, snd_row_i);
    }
}

fn make_identity<R, V>(ring: R, mut matrix: SubmatrixMut<V, InternalRow<El<R>>>, n: usize)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    assert_eq!(matrix.row_count(), matrix.col_count() * n);
    for i in 0..matrix.row_count() {
        for j in 0..matrix.col_count() {
            matrix.at(i, j).make_zero(&ring);
            if i / n == j {
                matrix.at(i, j).data.insert(0, (i % n, ring.one()));
            }
        }
    }
}

fn transform_2d<R>(ring: R, transform: &[[El<R>; 2]; 2], rows: [&mut InternalRow<El<R>>; 2], tmp: &mut [InternalRow<El<R>>; 2])
    where R: RingStore + Copy
{
    let [lhs, rhs] = rows;
    let lhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[0][0], &transform[0][1], std::mem::replace(&mut tmp[0], InternalRow::placeholder()));
    let rhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[1][0], &transform[1][1], std::mem::replace(&mut tmp[1], InternalRow::placeholder()));

    tmp[0] = std::mem::replace(lhs, lhs_new);
    tmp[1] = std::mem::replace(rhs, rhs_new);
}

mod local {
    use crate::divisibility::*;
    use crate::matrix::submatrix::ColumnMut;
    use crate::pid::PrincipalIdealRingStore;
    use crate::vector::{VectorView, VectorViewMut};

    use super::*;
            
    ///
    /// In the block of matrix given by `global_pivot`, performs unimodular row operation
    /// to ensure that the element in the pivot position divides all elements below it. 
    /// 
    fn make_pivot_ideal_gen<R, V1, V2>(ring: R, mut matrix: ColumnMut<V1, InternalRow<El<R>>>, mut transform: ColumnMut<V2, InternalRow<El<R>>>, (pivot_i, pivot_j): (usize, usize), tmp: &mut [InternalRow<El<R>>; 2]) 
        where R: PrincipalIdealRingStore + Copy,
            R::Type: PrincipalIdealRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            V2: AsPointerToSlice<InternalRow<El<R>>>
    {
        let mut current = ring.clone_el(matrix.at(pivot_i).at(pivot_j).unwrap_or(&ring.zero()));
        for i in (pivot_i + 1)..matrix.len() {
            if ring.is_unit(&current) {
                break;
            }
            if let Some(entry) = matrix.at(i).leading_entry_at(pivot_j) {
                if ring.checked_div(entry, &current).is_none() {
                    let (s, t, d) = ring.ideal_gen(&current, entry);
                    let local_transform = [[s, t], [ring.checked_div(entry, &d).unwrap(), ring.checked_div(&current, &d).unwrap()]];
                    let (fst_row, snd_row) = matrix.two_entries(pivot_i, i);
                    transform_2d(ring, &local_transform, [fst_row, snd_row], tmp);
                    let (fst_row, snd_row) = transform.two_entries(pivot_i, i);
                    transform_2d(ring, &local_transform, [fst_row, snd_row], tmp);
                    current = d;
                }
            }
        }
    }

    fn eliminate_row<R, V1, V2>(ring: R, mut matrix: ColumnMut<V1, InternalRow<El<R>>>, mut transform: ColumnMut<V2, InternalRow<El<R>>>, (pivot_i, pivot_j): (usize, usize), tmp: &mut InternalRow<El<R>>)
        where R: RingStore + Copy,
            R::Type: DivisibilityRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            V2: AsPointerToSlice<InternalRow<El<R>>>
    {
        let pivot_entry = matrix.at(pivot_i).at(pivot_j).unwrap();
        let pivot_entry = ring.clone_el(pivot_entry);

        for elim_i in 0..matrix.len() {
            if elim_i == pivot_i {
                continue;
            }
            if let Some(factor) = matrix.at(elim_i).at(pivot_j) {
                debug_assert!(elim_i < pivot_i || matrix.at(elim_i).leading_entry_at(pivot_j).is_some());
                let lhs_factor = ring.one();
                if let Some(quo) = ring.checked_div(factor, &pivot_entry) {
                    let rhs_factor = ring.negate(quo);

                    let new = add_row_local::<_, true>(ring, &matrix.at(elim_i), &matrix.at(pivot_i), &lhs_factor, &rhs_factor, std::mem::replace(tmp, InternalRow::placeholder()));
                    *tmp = std::mem::replace(matrix.at_mut(elim_i), new);

                    let new = add_row_local::<_, true>(ring, &transform.at(elim_i), &transform.at(pivot_i), &lhs_factor, &rhs_factor, std::mem::replace(tmp, InternalRow::placeholder()));
                    *tmp = std::mem::replace(transform.at_mut(elim_i), new);
                } else {
                    assert!(elim_i < pivot_i);
                }
            }
        }
    }

    #[inline(never)]
    pub fn row_echelon_optimistic<R, V1, V2>(ring: R, mut matrix: SubmatrixMut<V1, InternalRow<El<R>>>, mut transform: SubmatrixMut<V2, InternalRow<El<R>>>, n: usize) -> usize
        where R: RingStore + Copy,
            R::Type: PrincipalIdealRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            V2: AsPointerToSlice<InternalRow<El<R>>>
    {
        assert_eq!(matrix.row_count(), transform.row_count());
        assert_eq!(matrix.col_count(), 1);
        assert_eq!(transform.col_count(), 1);

        let col_block = n;
        let row_block = matrix.row_count();
        let mut i = 0;
        let mut tmp = [InternalRow::placeholder(), InternalRow::placeholder()];

        let mut matrix = matrix.col_at(0);
        let mut transform = transform.col_at(0);

        for j in 0..col_block {
            make_pivot_ideal_gen(ring, matrix.reborrow(), transform.reborrow(), (i, j), &mut tmp);
            if matrix.at(i).leading_entry_at(j).is_some() {
                eliminate_row(ring, matrix.reborrow(), transform.reborrow(), (i, j), &mut tmp[0]);
                i += 1;
                if i >= row_block {
                    break;
                }
            }
        }

        return i;
    }
}

mod global {
    use std::sync::atomic::AtomicUsize;

    use crate::parallel::potential_parallel_for_each;
    use crate::vector::VectorViewMut;

    #[cfg(feature = "parallel")]
    use rayon::iter::IndexedParallelIterator;

    use super::*;
    
    pub fn check_for_independent_rows<R, V1, V2>(ring: R, pivot_matrix: Submatrix<V1, InternalRow<El<R>>>, column: Submatrix<V1, InternalRow<El<R>>>, transform: SubmatrixMut<V2, InternalRow<El<R>>>, n: usize) -> Vec<usize>
        where R: DivisibilityRingStore + Copy + Sync,
            El<R>: Send + Sync,
            R::Type: DivisibilityRing,
            V1: Sync + AsPointerToSlice<InternalRow<El<R>>>,
            V2: Sync + AsPointerToSlice<InternalRow<El<R>>>
    {
        assert_eq!(pivot_matrix.col_count(), 1);
        assert_eq!(column.col_count(), 1);
        assert_eq!(transform.col_count(), 1);
        assert_eq!(column.row_count(), transform.row_count());

        let pivot_matrix = pivot_matrix.col_at(0);

        let unreduced_row_index = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect::<Vec<_>>();

        potential_parallel_for_each(column.concurrent_row_iter().zip(transform.concurrent_row_iter()), 
            || (InternalRow::placeholder(), InternalRow::placeholder()), 
            |(new_row, tmp), row_index, (matrix_row, transform_row)|
        {
            *transform_row.at_mut(0) = preimage_echelon_form_matrix(ring, pivot_matrix, matrix_row.at(0), std::mem::replace(transform_row.at_mut(0), InternalRow::placeholder()));
            
            let nonzero_j: usize = if !transform_row.at(0).is_empty() {
                *new_row = sub_assign_mul(ring, transform_row.at(0), pivot_matrix, matrix_row.at(0), std::mem::replace(new_row, InternalRow::placeholder()), tmp);
                new_row.leading_entry().0
            } else {
                matrix_row.at(0).leading_entry().0
            };

            if nonzero_j < usize::MAX && unreduced_row_index[nonzero_j].load(std::sync::atomic::Ordering::SeqCst) == usize::MAX {
                unreduced_row_index[nonzero_j].store(row_index, std::sync::atomic::Ordering::SeqCst);
            }
        });
        
        return unreduced_row_index.into_iter().map(|index| index.load(std::sync::atomic::Ordering::SeqCst)).filter(|index| *index != usize::MAX).collect::<Vec<_>>();
    }

    #[inline(never)]
    pub fn apply_pivot_row_transform<R, V1, V2>(ring: R, matrix: SubmatrixMut<V1, InternalRow<El<R>>>, transform: Submatrix<V2, InternalRow<El<R>>>) 
        where R: RingStore + Copy + Sync,
            El<R>: Send + Sync,
            V1: Sync + AsPointerToSlice<InternalRow<El<R>>>,
            V2: Sync + AsPointerToSlice<InternalRow<El<R>>>
    {
        potential_parallel_for_each(
            matrix.concurrent_col_iter(), 
            || Vec::new(), 
            |tmp, _, rows| 
        {
            mul_assign(ring, transform.col_at(0), rows, tmp);
        });
    }

    fn eliminate_single_row<V, R>(ring: R, elim_coefficients: &InternalRow<El<R>>, pivot_rows: Submatrix<V, InternalRow<El<R>>>, row: &mut [InternalRow<El<R>>], tmp: [&mut InternalRow<El<R>>; 2])
        where R: RingStore + Copy,
            V: AsPointerToSlice<InternalRow<El<R>>>
    {
        assert_eq!(pivot_rows.col_count(), row.len());
        let [tmp, new_row] = tmp;

        for col in 0..pivot_rows.col_count() {
            *new_row = linear_combine_rows(ring, &elim_coefficients, pivot_rows.col_at(col), std::mem::replace(new_row, InternalRow::placeholder()), tmp);
            *tmp = add_row_local::<_, true>(ring, row.at(col), &new_row, &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
            std::mem::swap(tmp, row.at_mut(col));
        }
    }

    #[inline(never)]
    pub fn eliminate_pivot_column<R, V1, V2>(ring: R, pivot_matrix: Submatrix<V1, InternalRow<El<R>>>, column: SubmatrixMut<V1, InternalRow<El<R>>>, transform: Submatrix<V2, InternalRow<El<R>>>)
        where R: DivisibilityRingStore + Copy + Sync,
            El<R>: Send + Sync,
            R::Type: DivisibilityRing,
            V1: Sync + AsPointerToSlice<InternalRow<El<R>>>,
            V2: Sync + AsPointerToSlice<InternalRow<El<R>>>
    {
        assert_eq!(transform.col_count(), 1);
        assert_eq!(pivot_matrix.col_count(), column.col_count());
        assert_eq!(transform.row_count(), column.row_count());
    
        if pivot_matrix.col_count() == 0 {
            return;
        }
        
        potential_parallel_for_each(column.concurrent_row_iter(), 
            || (InternalRow::placeholder(), InternalRow::placeholder()), 
            |(new_row, tmp), row_index, matrix_row|
        {
            if !transform.row_at(row_index).is_empty() {
                eliminate_single_row(ring, &transform.row_at(row_index)[0], pivot_matrix, matrix_row, [new_row, tmp]);
            }
        });
    }
    
}

#[inline(never)]
fn blocked_row_echelon<R, V, const LOG: bool>(ring: R, mut matrix: SubmatrixMut<V, InternalRow<El<R>>>, n: usize)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync,
        V: Sync + AsPointerToSlice<InternalRow<El<R>>>
{
    if LOG {
        print!("[{}x{}]", matrix.row_count(), matrix.col_count() * n);
        std::io::stdout().flush().unwrap();
    }
    let start = Instant::now();

    let col_block = n;
    let col_block_count = matrix.col_count();
    let row_count = matrix.row_count();
    let global_col_count = matrix.col_count();
    let mut original = if EXTENSIVE_RUNTIME_ASSERTS {
        Some((0..row_count).map(|i| (0..global_col_count).map(|j| matrix.at(i, j).clone_row(&ring)).collect::<Vec<_>>()).collect::<Vec<_>>())
    } else {
        None
    };

    let mut i = 0;
    let mut j = 0;
    let mut row_block = n;

    let mut transform = (0..row_count).map(|_| {
        let mut result = InternalRow::placeholder();
        result.make_zero(&ring);
        result
    }).collect::<Vec<_>>();
    let mut transform = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut transform[..], row_count, 1);
    make_identity(ring, transform.reborrow().restrict_rows(i..(i + row_block)), row_block);

    while i + row_block < matrix.row_count() && j < col_block_count {

        if EXTENSIVE_LOG {
            println!();
            println!("{}", <_ as crate::matrix::Matrix<_>>::format(&as_sparse_matrix_builder(ring, matrix.as_const(), n), &ring));
            println!();
        }

        let (mut pivot_transform, mut pivot_column_transform) = transform.reborrow().restrict_cols(0..1).split_rows(i..(i + row_block), (i + row_block)..row_count);
        let (mut pivot_matrix, pivot_column) = matrix.reborrow().restrict_cols(j..(j + 1)).split_rows(i..(i + row_block), (i + row_block)..row_count);
        let nonzero_row_count = local::row_echelon_optimistic(ring, pivot_matrix.reborrow(), pivot_transform.reborrow(), n);
        let mut swap_in_rows = global::check_for_independent_rows(ring, pivot_matrix.as_const(), pivot_column.as_const(), pivot_column_transform.reborrow(), n);

        if EXTENSIVE_LOG {
            println!();
            println!("{}", <_ as crate::matrix::Matrix<_>>::format(&as_sparse_matrix_builder(ring, matrix.as_const(), n), &ring));
            println!();
        }

        if swap_in_rows.len() > 0 {
            if EXTENSIVE_LOG {
                println!("Swap in {} rows", swap_in_rows.len());
            }

            // this is necessary, otherwise we might swap back and forth the same rows
            swap_in_rows.sort_unstable();

            let new_row_block = row_block + swap_in_rows.len();
            for target_i in 0..swap_in_rows.len() {
                let swap_row_index = swap_in_rows[target_i];
                swap_rows(ring, matrix.reborrow(), i + row_block + target_i, i + row_block + swap_row_index, n);
                swap_rows(ring, transform.reborrow(), i + row_block + target_i, i + row_block + swap_row_index, n);

                transform.at(i + row_block + target_i, 0).make_zero(&ring);
                transform.at(i + row_block + target_i, 0).data.insert(0, (row_block + target_i, ring.one()));
                transform.at(i + row_block + target_i, 0).check(&ring);
            }
            row_block = new_row_block;
        } else {
            if EXTENSIVE_LOG {
                println!("Eliminate column");
            }
            assert!(nonzero_row_count <= row_block);
            
            let (mut pivot_row, other_rows) = matrix.reborrow().restrict_cols(j..global_col_count).split_rows(i..(i + row_block), (i + row_block)..row_count);
            global::apply_pivot_row_transform(ring, pivot_row.reborrow().restrict_cols(1..(global_col_count - j)), pivot_transform.as_const());
            global::eliminate_pivot_column(ring, pivot_row.as_const(), other_rows, pivot_column_transform.as_const());

            i += nonzero_row_count;
            j += 1;
            row_block = col_block;

            make_identity(ring, transform.reborrow().restrict_rows(i..(i + row_block)), row_block);

            if EXTENSIVE_LOG {
                println!();
                println!("{}", <_ as crate::matrix::Matrix<_>>::format(&as_sparse_matrix_builder(ring, matrix.as_const(), n), &ring));
                println!();
            }

            if EXTENSIVE_RUNTIME_ASSERTS {
                assert_left_equivalent(&ring, SubmatrixMut::<Vec<_>, _>::new(original.as_mut().unwrap()).as_const(), matrix.as_const(), n);
            }

            if LOG {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }
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
/// # use feanor_math::algorithms::smith::solve_right;
/// # use feanor_math::matrix::dense::*;
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

    let row_count = <_ as crate::matrix::Matrix<_>>::row_count(&matrix);
    let col_count = <_ as crate::matrix::Matrix<_>>::col_count(&matrix);
    let n = block_size;
    let global_col_count = (col_count - 1) / n + 1;
    let mut matrix = matrix.into_internal_matrix(n, ring.get_ring());
    let mut matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut matrix[..], row_count + n, global_col_count);
    blocked_row_echelon::<_, _, LOG>(ring, matrix.reborrow(), n);

    let matrix_ref = &matrix.as_const();
    let result = (0..row_count).map(move |i| (0..global_col_count).flat_map(move |j| 
        matrix_ref.at(i, j).data.iter().rev().skip(1).rev().map(move |(k, c)| (k + j * n, ring.clone_el(c)))
    ).collect::<Vec<_>>()).collect::<Vec<_>>();

    if EXTENSIVE_RUNTIME_ASSERTS {
        let mut check = SparseMatrixBuilder::new(&ring);
        check.add_cols(<_ as crate::matrix::Matrix<_>>::col_count(original.as_ref().unwrap()));
        for (i, row) in result.iter().enumerate() {
            check.add_row(i, row.iter().map(|(j, c)| (*j, ring.clone_el(c))));
        }
        assert_is_correct_row_echelon(ring, &original.unwrap(), &check);
    }

    return result;
}

fn as_sparse_matrix_builder<R, V>(ring: R, matrix: Submatrix<V, InternalRow<El<R>>>, n: usize) -> SparseMatrixBuilder<R::Type>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    let mut result = SparseMatrixBuilder::new(&ring);
    result.add_cols(matrix.col_count() * n);
    for (i, row) in matrix.row_iter().enumerate() {
        result.add_row(i, row.iter().enumerate().flat_map(|(j, int_row)| int_row.data.iter().rev().skip(1).rev().map(move |(k, c)| (j * n + k, ring.clone_el(c)))));
    }
    return result;
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;

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

    let mut original_transposed = DenseMatrix::zero(original.col_count() * n, original.row_count(), &ring);
    let mut actual_transposed = DenseMatrix::zero(original.col_count() * n, original.row_count(), &ring);
    for i in 0..original.row_count() {
        for j in 0..original.col_count() {
            for k in 0..n {
                *original_transposed.at_mut(j * n + k, i) = ring.clone_el(original.at(i, j).at(k).unwrap_or(&zero));
                *actual_transposed.at_mut(j * n + k, i) = ring.clone_el(new.at(i, j).at(k).unwrap_or(&zero));
            }
        }
    }
    if !smith::solve_right::<&R>(&mut original_transposed.clone_matrix(&ring), actual_transposed.clone_matrix(&ring), &ring).is_some() {
        // println!("{:?}", original.row_iter().map(|row| row.iter().enumerate().flat_map(move |(j, local_row)| local_row.data.iter().map(move |(k, c)| (j * n + k, format!("{}", ring.format(c))))).collect::<Vec<_>>()).collect::<Vec<_>>());
        panic!();
    }
    if !smith::solve_right::<&R>(&mut actual_transposed.clone_matrix(&ring), original_transposed.clone_matrix(&ring), &ring).is_some() {
        // println!("{:?}", original.row_iter().map(|row| row.iter().enumerate().flat_map(move |(j, local_row)| local_row.data.iter().map(move |(k, c)| (j * n + k, format!("{}", ring.format(c))))).collect::<Vec<_>>()).collect::<Vec<_>>());
        panic!();
    }
}

fn assert_is_correct_row_echelon<R>(ring: R, original: &SparseMatrixBuilder<R::Type>, row_echelon_form: &SparseMatrixBuilder<R::Type>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    use crate::algorithms::smith;

    let n = <_ as crate::matrix::Matrix<_>>::row_count(original);
    let m = <_ as crate::matrix::Matrix<_>>::col_count(original);
    assert_eq!(n, <_ as crate::matrix::Matrix<_>>::row_count(row_echelon_form));
    assert_eq!(m, <_ as crate::matrix::Matrix<_>>::col_count(row_echelon_form));

    for i in 1..n {
        assert!(row_echelon_form.rows[i].len() == 0 || row_echelon_form.rows[i][0].0 > row_echelon_form.rows[i - 1][0].0);
    }

    let mut original_transposed = DenseMatrix::zero(m, n, &ring);
    let mut actual_transposed = DenseMatrix::zero(m, n, &ring);
    for i in 0..n {
        for j in 0..m {
            *original_transposed.at_mut(j, i) = ring.clone_el(<_ as crate::matrix::Matrix<_>>::entry_at(original, i, j));
            *actual_transposed.at_mut(j, i) = ring.clone_el(<_ as crate::matrix::Matrix<_>>::entry_at(row_echelon_form, i, j));
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
            actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
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
            actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_gb_sparse_row_echelon_recompute_pivot() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 3]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));
    
    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(3);
    matrix.add_row(0, sparsify([0, 1, 1]));
    matrix.add_row(1, sparsify([0, 0, 0]));
    matrix.add_row(2, sparsify([0, 1, 0]));
    matrix.add_row(3, sparsify([1, 0, 0]));
    
    let mut actual = SparseMatrixBuilder::new(&R);
    actual.add_cols(3);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
}

#[test]
fn test_gb_sparse_row_echelon_swap_in_twice() {
    let R = Zn::<17>::RING;
    let sparsify = |row: [u64; 3]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));
    
    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(3);
    matrix.add_row(0, sparsify([0, 0, 1]));
    matrix.add_row(1, sparsify([0, 0, 0]));
    matrix.add_row(2, sparsify([1, 1, 1]));
    matrix.add_row(3, sparsify([1, 2, 3]));
    matrix.add_row(3, sparsify([1, 3, 3]));
    
    let mut actual = SparseMatrixBuilder::new(&R);
    actual.add_cols(3);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
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
            actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
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
            actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
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
            actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
        }
        assert_is_correct_row_echelon(R, &matrix, &actual);
    }
}

#[test]
fn test_bad_swapping_order() {
    let R = Zn::<3>::RING;
    let sparsify = |row: [u64; 10]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(10);
    matrix.add_row(0, sparsify([0, 0, 1, 0, 0, 0, 1, 0, 0, 0]));
    matrix.add_row(1, sparsify([0, 0, 0, 1, 0, 0, 0, 1, 0, 0]));
    matrix.add_row(2, sparsify([0, 1, 0, 0, 1, 0, 0, 0, 1, 0]));
    matrix.add_row(3, sparsify([1, 0, 0, 0, 0, 1, 0, 0, 0, 1]));

    let mut actual = SparseMatrixBuilder::new(&R);
    actual.add_cols(10);
    for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), 2) {
        actual.add_row(<_ as crate::matrix::Matrix<_>>::row_count(&actual), row.into_iter());
    }
    assert_is_correct_row_echelon(R, &matrix, &actual);
    
}
