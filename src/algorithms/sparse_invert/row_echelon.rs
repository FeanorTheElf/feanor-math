use std::alloc::Global;
use std::marker::PhantomData;
use std::{cmp::Ordering, io::Write, time::Instant};

use transform::TransformTarget;

use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::matrix::*;
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::seq::{SwappableVectorViewMut, VectorView, VectorViewMut};

use super::matrix::SparseMatrix;

pub const EXTENSIVE_RUNTIME_ASSERTS: bool = false;
pub const EXTENSIVE_LOG: bool = false;

pub struct InternalRow<T> {
    pub(super) data: Vec<(usize, T)>
}

impl<T> InternalRow<T> {

    fn index_of(&self, local_j: usize) -> Result<usize, usize> {
        self.data.binary_search_by_key(&local_j, |(j, _)| *j)
    }

    fn update(&mut self, local_j: usize, idx: Result<usize, usize>, new_val: Option<T>) {
        debug_assert!(idx.is_err() || self.data[idx.unwrap()].0 == local_j);
        debug_assert!(idx.is_ok() || idx.unwrap_err() == self.data.len() || self.data[idx.unwrap_err()].0 > local_j);
        match (idx, new_val) {
            (Ok(idx), Some(new_val)) => { self.data[idx] = (local_j, new_val); },
            (Ok(idx), None) => { self.data.remove(idx); },
            (Err(idx), Some(new_val)) => { self.data.insert(idx, (local_j, new_val)); },
            (Err(_), None) => {}
        }
    }

    fn update_two(&mut self, local_j1: usize, idx1: Result<usize, usize>, new_val1: Option<T>, local_j2: usize, idx2: Result<usize, usize>, new_val2: Option<T>) {
        debug_assert!(idx1.is_err() || self.data[idx1.unwrap()].0 == local_j1);
        debug_assert!(idx1.is_ok() || idx1.unwrap_err() == self.data.len() || self.data[idx1.unwrap_err()].0 > local_j1);
        debug_assert!(idx2.is_err() || self.data[idx2.unwrap()].0 == local_j2);
        debug_assert!(idx2.is_ok() || idx2.unwrap_err() == self.data.len() || self.data[idx2.unwrap_err()].0 > local_j2);
        if local_j1 > local_j2 {
            self.update_two(local_j2, idx2, new_val2, local_j1, idx1, new_val1);
        } else {
            self.update(local_j2, idx2, new_val2);
            self.update(local_j1, idx1, new_val1);
        }
    }

    pub fn at<'a>(&'a self, local_j: usize) -> Option<&'a T> {
        self.index_of(local_j).ok().map(|idx| &self.data[idx].1)
    }

    pub fn clone_row<R>(&self, ring: R) -> Self
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
    /// [`InternalRow::make_zero()`]). This function will never allocate memory.
    /// 
    pub fn placeholder() -> InternalRow<T> {
        InternalRow { data: Vec::new() }
    }

    pub fn make_zero<R>(&mut self, ring: R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.clear();
        self.data.push((usize::MAX, ring.zero()));
    }

    pub fn check<R>(&self, ring: R)
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
}

#[inline(never)]
pub fn add_row_local<R, const LHS_FACTOR_ONE: bool, const RHS_FACTOR_ONE: bool>(ring: R, lhs: &InternalRow<El<R>>, rhs: &InternalRow<El<R>>, lhs_factor: &El<R>, rhs_factor: &El<R>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore
{
    assert!(!LHS_FACTOR_ONE || ring.is_one(lhs_factor));
    assert!(!RHS_FACTOR_ONE || ring.is_one(rhs_factor));
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
                let rhs_val = if RHS_FACTOR_ONE {
                    ring.clone_el(&rhs.data[rhs_idx].1)
                } else {
                    ring.mul_ref(&rhs.data[rhs_idx].1, rhs_factor)
                };
                if RHS_FACTOR_ONE || !ring.is_zero(&rhs_val) {
                    out.data.push((rhs_j, rhs_val));
                }
                rhs_idx += 1;
            },
            Ordering::Equal => {
                let lhs_val = if LHS_FACTOR_ONE { ring.clone_el(&lhs.data[lhs_idx].1) } else { ring.mul_ref(&lhs.data[lhs_idx].1, lhs_factor) };
                let rhs_val = if RHS_FACTOR_ONE { ring.clone_el(&rhs.data[rhs_idx].1) } else { ring.mul_ref(&rhs.data[rhs_idx].1, rhs_factor) };
                let value = ring.add(lhs_val, rhs_val);
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
        *tmp = add_row_local::<_, true, false>(ring, &out, rows.at(*idx), &lhs_factor, c, std::mem::replace(tmp, InternalRow::placeholder()));
        std::mem::swap(&mut out, tmp);
    }
    out.check(&ring);
    return out;
}
    
fn add_assign_mul<R, V>(ring: R, elim_coefficients: &InternalRow<El<R>>, pivot_rows: Column<V, InternalRow<El<R>>>, row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>, tmp: &mut InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore + Copy,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    elim_coefficients.check(&ring);
    row.check(&ring);
    out = linear_combine_rows(ring, &elim_coefficients, pivot_rows, out, tmp);
    *tmp = add_row_local::<_, true, true>(ring, row, &out, &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
    std::mem::swap(tmp, &mut out);
    out.check(&ring);
    return out;
}

///
/// Computes a vector `result` such that `result * echelon_form_matrix + current_row = 0`, if it exists.
/// Note that if the vector does not exist, this function will not throw an error but return a
/// vector that can be thought of as a "best effort solution". This is important in combination
/// with the optimistic elimination strategy.
/// 
#[inline(never)]
pub fn preimage_echelon_form_matrix<R, V>(ring: R, echelon_form_matrix: Column<V, InternalRow<El<R>>>, current_row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    current_row.check(&ring);
    let zero = ring.zero();
    out.make_zero(&ring);
    out.data.pop();


    // currently extracted to simplify profiling
    #[inline(always)]
    fn perform_inner<R, V>(ring: R, i: usize, row: &InternalRow<El<R>>, current_row: &InternalRow<El<R>>, echelon_form_matrix: &Column<V, InternalRow<El<R>>>, zero: &El<R>, out: &mut InternalRow<El<R>>)
        where R: DivisibilityRingStore + Copy,
            R::Type: DivisibilityRing,
            V: AsPointerToSlice<InternalRow<El<R>>>
    {
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

    for (i, row) in echelon_form_matrix.as_iter().enumerate() {
        perform_inner(ring, i, row, current_row, &echelon_form_matrix, &zero, &mut out);
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
    assert!(matrix.row_count() >= matrix.col_count() * n);
    for i in 0..(matrix.col_count() * n) {
        for j in 0..matrix.col_count() {
            matrix.at_mut(i, j).make_zero(&ring);
            if i / n == j {
                matrix.at_mut(i, j).data.insert(0, (i % n, ring.one()));
            }
        }
    }
    for i in (matrix.col_count() * n)..matrix.row_count() {
        for j in 0..matrix.col_count() {
            matrix.at_mut(i, j).make_zero(&ring);
        }
    }
}

fn transform_2x2<R>(ring: R, transform: &[El<R>; 4], rows: [&mut InternalRow<El<R>>; 2], tmp: &mut [InternalRow<El<R>>; 2])
    where R: RingStore + Copy
{
    let [lhs, rhs] = rows;
    let lhs_new = add_row_local::<R, false, false>(ring, lhs, rhs, &transform[0], &transform[1], std::mem::replace(&mut tmp[0], InternalRow::placeholder()));
    let rhs_new = add_row_local::<R, false, false>(ring, lhs, rhs, &transform[2], &transform[3], std::mem::replace(&mut tmp[1], InternalRow::placeholder()));

    tmp[0] = std::mem::replace(lhs, lhs_new);
    tmp[1] = std::mem::replace(rhs, rhs_new);
}

mod local {
    use transform::TransformTarget;

    use crate::divisibility::*;
    use crate::matrix::ColumnMut;
    use crate::pid::PrincipalIdealRingStore;
    use crate::seq::{VectorView, VectorViewMut};

    use super::*;
            
    ///
    /// In the block of matrix given by `global_pivot`, performs unimodular row operation
    /// to ensure that the element in the pivot position divides all elements below it. 
    /// 
    fn make_pivot_extended_ideal_gen<R, V1, T>(ring: R, mut matrix: ColumnMut<V1, InternalRow<El<R>>>, (pivot_i, pivot_j): (usize, usize), tmp: &mut [InternalRow<El<R>>; 2], row_ops: &mut T) 
        where R: PrincipalIdealRingStore + Copy,
            R::Type: PrincipalIdealRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            T: TransformTarget<R::Type>
    {
        let mut current = ring.clone_el(matrix.at(pivot_i).at(pivot_j).unwrap_or(&ring.zero()));
        for i in (pivot_i + 1)..matrix.len() {
            if ring.is_unit(&current) {
                break;
            }
            if let Some(entry) = matrix.at(i).leading_entry_at(pivot_j) {
                if ring.checked_div(entry, &current).is_none() {
                    let (local_transform, gcd) = ring.get_ring().create_elimination_matrix(&current, entry);
                    let local_transform_det = ring.sub(ring.mul_ref(&local_transform[0], &local_transform[3]), ring.mul_ref(&local_transform[1], &local_transform[2]));
                    if EXTENSIVE_RUNTIME_ASSERTS {
                        assert!(ring.is_unit(&local_transform_det)); 
                    }
                    let (fst_row, snd_row) = matrix.two_entries(pivot_i, i);
                    transform_2x2(ring, &local_transform, [fst_row, snd_row], tmp);
                    row_ops.transform(ring, pivot_i, i, &local_transform);
                    current = gcd;
                }
            }
        }
    }

    fn eliminate_row<R, V1, T>(ring: R, mut matrix: ColumnMut<V1, InternalRow<El<R>>>, (pivot_i, pivot_j): (usize, usize), tmp: &mut InternalRow<El<R>>, row_ops: &mut T)
        where R: RingStore + Copy,
            R::Type: DivisibilityRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            T: TransformTarget<R::Type>
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

                    let new = add_row_local::<_, true, false>(ring, &matrix.at(elim_i), &matrix.at(pivot_i), &lhs_factor, &rhs_factor, std::mem::replace(tmp, InternalRow::placeholder()));
                    *tmp = std::mem::replace(matrix.at_mut(elim_i), new);

                    row_ops.subtract(ring, pivot_i, elim_i, &ring.negate(rhs_factor));
                } else {
                    assert!(elim_i < pivot_i);
                }
            }
        }
    }

    #[inline(never)]
    pub fn row_echelon_optimistic<R, V1, T>(ring: R, mut matrix: SubmatrixMut<V1, InternalRow<El<R>>>, n: usize, mut row_ops: T) -> usize
        where R: RingStore + Copy,
            R::Type: PrincipalIdealRing,
            V1: AsPointerToSlice<InternalRow<El<R>>>,
            T: TransformTarget<R::Type>
    {
        assert_eq!(matrix.col_count(), 1);

        let col_block = n;
        let row_block = matrix.row_count();
        let mut i = 0;
        let mut tmp = [InternalRow::placeholder(), InternalRow::placeholder()];

        let mut matrix = matrix.col_mut_at(0);

        for j in 0..col_block {
            make_pivot_extended_ideal_gen(ring, matrix.reborrow(), (i, j), &mut tmp, &mut row_ops);
            if matrix.at(i).leading_entry_at(j).is_some() {
                eliminate_row(ring, matrix.reborrow(), (i, j), &mut tmp[0], &mut row_ops);
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
    use crate::seq::VectorViewMut;

    #[cfg(feature = "parallel")]
    use rayon::iter::IndexedParallelIterator;

    use super::*;
    
    pub fn partial_eliminate_rows<R, V1, V2, const DISCARD_OLD_TRANSFORM: bool>(ring: R, pivot_matrix: Submatrix<V1, InternalRow<El<R>>>, column: SubmatrixMut<V1, InternalRow<El<R>>>, transform: SubmatrixMut<V2, InternalRow<El<R>>>, n: usize) -> Vec<usize>
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
            || (InternalRow::placeholder(), InternalRow::placeholder(), InternalRow::placeholder()), 
            |(new_row, add_transform_row, tmp), row_index, (matrix_row, transform_row)|
        {
            // early exit, improves performance but not necessary for functionality
            if matrix_row.is_empty() {
                return;
            }

            let mut additional_transform = preimage_echelon_form_matrix(ring, pivot_matrix, matrix_row.at(0), std::mem::replace(add_transform_row, InternalRow::placeholder()));
            
            if !additional_transform.is_empty() {
                *new_row = add_assign_mul(ring, &additional_transform, pivot_matrix, matrix_row.at(0), std::mem::replace(new_row, InternalRow::placeholder()), tmp);
                std::mem::swap(matrix_row.at_mut(0), new_row);
            }
            let nonzero_j: usize = matrix_row.at(0).leading_entry().0;
            matrix_row.at(0).check(&ring);

            if DISCARD_OLD_TRANSFORM {
                std::mem::swap(&mut transform_row[0], &mut additional_transform);
                transform_row[0].check(&ring);
            } else {
                *new_row = add_row_local::<_, true, true>(ring, &additional_transform, transform_row.at(0), &ring.one(), &ring.one(), std::mem::replace(new_row, InternalRow::placeholder()));
                std::mem::swap(transform_row.at_mut(0), new_row);
                transform_row[0].check(&ring);
            }
            std::mem::swap(add_transform_row, &mut additional_transform);

            // it is actually irrelevant which index ends up in `unreduced_row_index`, so it is fine to use `load()` and `store()` instead of `compare_exchange()`
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

    #[inline(never)]
    pub fn update_pivot_column_transform_matrix<R, V1, V2>(ring: R, pivot_column_transform: SubmatrixMut<V1, InternalRow<El<R>>>, transform_transform: Submatrix<V2, InternalRow<El<R>>>) 
        where R: RingStore + Copy + Sync,
            El<R>: Send + Sync,
            V1: Sync + AsPointerToSlice<InternalRow<El<R>>>,
            V2: Sync + AsPointerToSlice<InternalRow<El<R>>>
    {
        debug_assert_eq!(transform_transform.col_count(), 1);
        potential_parallel_for_each(
            pivot_column_transform.concurrent_row_iter(), 
            || (InternalRow::placeholder(), InternalRow::placeholder()), 
            |(tmp0, tmp1), _, row|
        {
            debug_assert_eq!(row.len(), 1);
            *tmp0 = linear_combine_rows(ring, &row[0], transform_transform.col_at(0), std::mem::replace(tmp0, InternalRow::placeholder()), tmp1);
            std::mem::swap(&mut row[0], tmp0);
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
            *tmp = add_row_local::<_, true, true>(ring, row.at(col), &new_row, &ring.one(), &ring.one(), std::mem::replace(tmp, InternalRow::placeholder()));
            std::mem::swap(tmp, row.at_mut(col));
        }
    }

    #[inline(never)]
    pub fn apply_elimination_using_transform<R, V1, V2>(ring: R, pivot_matrix: Submatrix<V1, InternalRow<El<R>>>, column: SubmatrixMut<V1, InternalRow<El<R>>>, transform: Submatrix<V2, InternalRow<El<R>>>)
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
    
    ///
    /// Swaps the rows of given indices (relative to the space underneath the pivot matrix) with the rows directly
    /// following the pivot matrix, such that they will be included in the pivot matrix once we increase `row_block`.
    /// 
    /// Furthermore, this updates the `transform` matrix in a corresponding way, such that its uppper part (the `pivot_transform`)
    /// still contains the correct transformation w.r.t. the original pivot matrix. Its lower part (the `pivot_column_transform` or
    /// "elimination transform matrix") is left unchanged, but is used to update the upper part correctly.
    /// 
    #[inline(never)]
    pub fn swap_in_rows<R, V1, V2>(ring: R, mut matrix: SubmatrixMut<V1, InternalRow<El<R>>>, mut transform: SubmatrixMut<V2, InternalRow<El<R>>>, i: usize, row_block: usize, col_block: usize, swap_in_row_idxs: &mut [usize])
        where R: RingStore + Copy + Sync,
            El<R>: Send + Sync,
            R::Type: PrincipalIdealRing,
            V1: Sync + AsPointerToSlice<InternalRow<El<R>>>,
            V2: Sync + AsPointerToSlice<InternalRow<El<R>>>
    {
        // this is necessary, otherwise we might swap back and forth the same rows
        swap_in_row_idxs.sort_unstable();

        if EXTENSIVE_RUNTIME_ASSERTS {
            for pivot_transform_i in 0..row_block {
                assert!(transform.at(i + pivot_transform_i, 0).data.iter().all(|(j, _)| *j == usize::MAX || *j < row_block));
            }
        }

        let new_row_block = row_block + swap_in_row_idxs.len();
        for target_i in 0..swap_in_row_idxs.len() {
            let swap_row_index = swap_in_row_idxs[target_i];
            swap_rows(ring, matrix.reborrow(), i + row_block + target_i, i + row_block + swap_row_index, col_block);
            swap_rows(ring, transform.reborrow(), i + row_block + target_i, i + row_block + swap_row_index, col_block);
        }

        let mut tmp = (0..row_block).map(|target_i| transform.at(i + target_i, 0).clone_row(&ring)).collect::<Vec<_>>();
        mul_assign(ring, transform.as_const().restrict_rows((i + row_block)..(i + new_row_block)).col_at(0), SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut tmp, row_block, 1).col_mut_at(0), &mut Vec::new());
        for (target_i, mut transform_row) in (0..swap_in_row_idxs.len()).zip(tmp.into_iter()) {
            if EXTENSIVE_RUNTIME_ASSERTS {
                assert!(transform_row.data.iter().all(|(j, _)| *j == usize::MAX || *j < row_block));
            }

            transform_row.data.insert(transform_row.data.len() - 1, (row_block + target_i, ring.one()));
            transform_row.check(&ring);
            *transform.at_mut(i + row_block + target_i, 0) = transform_row;
        }
    }
}

struct TransformRows<'a, R: RingBase + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>> {
    ring: PhantomData<R>,
    matrix: ColumnMut<'a, V, InternalRow<R::Element>>,
    tmp: [InternalRow<R::Element>; 2]
}

impl<'a, R: RingBase + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>> TransformRows<'a, R, V> {

    fn new(matrix: ColumnMut<'a, V, InternalRow<R::Element>>) -> Self {
        Self {
            ring: PhantomData,
            matrix: matrix,
            tmp: [InternalRow::placeholder(), InternalRow::placeholder()]
        }
    }
}

impl<'a, R: RingBase + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>> TransformTarget<R> for TransformRows<'a, R, V> {

    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let (fst, snd) = self.matrix.two_entries(i, j);
        transform_2x2(ring, transform, [fst, snd], &mut self.tmp)
    }
}

struct TransformCols<'a, R: RingBase + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>, T: TransformTarget<R>> {
    ring: PhantomData<R>,
    matrix: ColumnMut<'a, V, InternalRow<R::Element>>,
    pass_on: T
}

impl<'a, R: RingBase + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>, T: TransformTarget<R>> TransformCols<'a, R, V, T> {

    fn new(matrix: ColumnMut<'a, V, InternalRow<R::Element>>, pass_on: T) -> Self {
        Self {
            ring: PhantomData,
            matrix: matrix,
            pass_on: pass_on
        }
    }
}

impl<'a, R: DivisibilityRing + ?Sized, V: AsPointerToSlice<InternalRow<R::Element>>,  T: TransformTarget<R>> TransformTarget<R> for TransformCols<'a, R, V, T> {

    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, k: usize, l: usize, transform: &[<R as RingBase>::Element; 4]) {
        self.pass_on.transform(ring, k, l, transform);

        let transform_det = ring.sub(ring.mul_ref(&transform[0], &transform[3]), ring.mul_ref(&transform[1], &transform[2]));
        let transform_det_inv = ring.invert(&transform_det).unwrap();
        let inv_transform = [
            ring.mul_ref(&transform[3], &transform_det_inv), 
            ring.negate(ring.mul_ref(&transform[1], &transform_det_inv)),
            ring.negate(ring.mul_ref(&transform[2], &transform_det_inv)), 
            ring.mul_ref(&transform[0], &transform_det_inv)
        ];
        let zero = ring.zero();
        for i in 0..self.matrix.len() {
            let row = self.matrix.at_mut(i);
            let row_k_idx = row.index_of(k);
            let row_l_idx = row.index_of(l);
            let row_k = row_k_idx.ok().map(|idx| &row.data[idx].1);
            let row_l = row_l_idx.ok().map(|idx| &row.data[idx].1);
            let new_row_k = ring.add(ring.mul_ref(row_k.unwrap_or(&zero), &inv_transform[0]), ring.mul_ref(row_l.unwrap_or(&zero), &inv_transform[2]));
            let new_row_l = ring.add(ring.mul_ref(row_k.unwrap_or(&zero), &inv_transform[1]), ring.mul_ref(row_l.unwrap_or(&zero), &inv_transform[3]));
            let new_row_k = if ring.is_zero(&new_row_k) { None } else { Some(new_row_k) };
            let new_row_l = if ring.is_zero(&new_row_l) { None } else { Some(new_row_l) };
            row.update_two(k, row_k_idx, new_row_k, l, row_l_idx, new_row_l);
            row.check(ring);
        }
    }
}

#[inline(never)]
pub(super) fn blocked_row_echelon<R, V, const LOG: bool>(ring: R, mut matrix: SubmatrixMut<V, InternalRow<El<R>>>, n: usize)
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
    let original = if EXTENSIVE_RUNTIME_ASSERTS {
        Some(OwnedMatrix::from_fn_in(row_count, global_col_count, |i, j| matrix.at(i, j).clone_row(&ring), Global))
    } else {
        None
    };
    if let Some(original) = &original {
        for i in 0..original.row_count() {
            for j in 0..original.col_count() {
                original.at(i, j).check(&ring);
            }
        }
    }

    let mut i = 0;
    let mut j = 0;
    let mut row_block = col_block;

    let zero_row = || {
        let mut result = InternalRow::placeholder();
        result.make_zero(&ring);
        result
    };

    let mut transform = (0..row_count).map(|_| zero_row()).collect::<Vec<_>>();
    // Note that the internal rows of transform are not of length n, but of length row_block
    let mut transform = SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut transform[..], row_count, 1);
    make_identity(ring, transform.reborrow().restrict_rows(i..(i + row_block)), row_block);

    let mut transform_transform = Vec::new();

    while i + row_block < matrix.row_count() && j < col_block_count {

        if EXTENSIVE_LOG {
            println!("A");
            println!("{}", to_sparse_matrix_builder(ring, matrix.as_const(), n).format(ring.get_ring()));
            println!();
        }

        // General Strategy
        // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        // We consider the top-left kxl submatrix, and bring it in row echelon form. At the beginning, k = l but that can change later.
        // Afterwards, we use the echelonized submatrix to eliminate everything below it, unless we find a row that cannot be eliminated.
        // If the latter is the case, we swap it into the top-left submatrix (we also call it pivot matrix), thus causing k to increase by
        // one. Afterwards, we again echelonize the pivot matrix, and repeat.
        // After this is completely done, we then apply the transformations used to echelonize the pivot matrix to the part right of it,
        // and then apply the transforms used to eliminate the part below the pivot matrix to the rest of the matrix. In particular, this
        // requires to 1) remember the transformations done and 2) when re-echelonizing the pivot matrix after swapping in a row, we must
        // also update the current elimination transform matrix correspondingly. 

        // pivot_transform:         row operations performed on the pivot matrix, will later be applied to the part right of the pivot matrix
        // pivot_column_transform:  combination of the pivot matrix rows that are used to eliminate the current column
        let (mut pivot_transform, mut pivot_column_transform) = transform.reborrow().restrict_cols(0..1).split_rows(i..(i + row_block), (i + row_block)..row_count);
        let (mut pivot_matrix, mut pivot_column) = matrix.reborrow().restrict_cols(j..(j + 1)).split_rows(i..(i + row_block), (i + row_block)..row_count);

        let (nonzero_row_count, mut swap_in_rows) = if row_block != col_block {
            // re-echelonization, so we already have a nontrivial pivot_column_transform; update it

            // transform_transform: the transform made during re-echelonization, that has to be applied to the elimination transform matrix pivot_column_transform
            transform_transform.resize_with(row_block, InternalRow::placeholder);
            let mut transform_transform = SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut transform_transform[..], row_block, 1);
            make_identity(ring, transform_transform.reborrow(), row_block);

            // echelonize the pivot matrix
            let nonzero_row_count = local::row_echelon_optimistic(ring, pivot_matrix.reborrow(), n, TransformCols::new(transform_transform.col_mut_at(0), TransformRows::new(pivot_transform.col_mut_at(0))));
            // update the elimination trnasform matrix
            global::update_pivot_column_transform_matrix(ring, pivot_column_transform.reborrow(), transform_transform.as_const());
            // try to complete elimination (wasn't complete previously, since we swapped in elements)
            let swap_in_rows = global::partial_eliminate_rows::<_, _, _, false>(ring, pivot_matrix.as_const(), pivot_column.reborrow(), pivot_column_transform.reborrow(), n);

            if EXTENSIVE_LOG {
                println!("TT");
                println!("{}", to_sparse_matrix_builder(ring, transform_transform.as_const(), row_block).format(ring.get_ring()));
                println!();

                println!("T");
                println!("{}", to_sparse_matrix_builder(ring, pivot_transform.as_const(), row_block).format(ring.get_ring()));
                println!();
            }

            (nonzero_row_count, swap_in_rows)
        } else {
            // pivot_column_transform is zero, so no need to update it
            let nonzero_row_count = local::row_echelon_optimistic(ring, pivot_matrix.reborrow(), n, TransformRows::new(pivot_transform.col_mut_at(0)));
            let swap_in_rows = global::partial_eliminate_rows::<_, _, _, true>(ring, pivot_matrix.as_const(), pivot_column.reborrow(), pivot_column_transform.reborrow(), n);
            
            if EXTENSIVE_LOG {
                println!("T");
                println!("{}", to_sparse_matrix_builder(ring, pivot_transform.as_const(), row_block).format(ring.get_ring()));
                println!();
            }

            (nonzero_row_count, swap_in_rows)
        };

        if EXTENSIVE_LOG {
            println!("E");
            println!("{}", to_sparse_matrix_builder(ring, pivot_column_transform.as_const(), row_block).format(ring.get_ring()));
            println!();
            println!("A");
            println!("{}", to_sparse_matrix_builder(ring, matrix.as_const(), n).format(ring.get_ring()));
            println!();
        }

        if swap_in_rows.len() > 0 {
            if EXTENSIVE_LOG {
                println!("Swap in {:?} row(s)", swap_in_rows);
            }

            global::swap_in_rows(ring, matrix.reborrow(), transform.reborrow(), i, row_block, col_block, &mut swap_in_rows);
            row_block += swap_in_rows.len();

            if EXTENSIVE_LOG {
                println!("T");
                println!("{}", to_sparse_matrix_builder(ring, transform.as_const().restrict_rows(i..(i + row_block)), row_block).format(ring.get_ring()));
                println!();
            }
        } else {
            if EXTENSIVE_LOG {
                println!("Eliminate column");
            }
            assert!(nonzero_row_count <= row_block);
            
            let (mut pivot_row, other_rows) = matrix.reborrow().restrict_cols((j + 1)..global_col_count).split_rows(i..(i + row_block), (i + row_block)..row_count);
            global::apply_pivot_row_transform(ring, pivot_row.reborrow().restrict_cols(0..(global_col_count - j - 1)), pivot_transform.as_const());
            global::apply_elimination_using_transform(ring, pivot_row.as_const(), other_rows, pivot_column_transform.as_const());

            i += nonzero_row_count;
            j += 1;
            row_block = col_block;

            make_identity(ring, transform.reborrow().restrict_rows(i..row_count), row_block);

            if EXTENSIVE_LOG {
                println!("A");
                println!("{}", to_sparse_matrix_builder(ring, matrix.as_const(), n).format(ring.get_ring()));
                println!();
            }

            if EXTENSIVE_RUNTIME_ASSERTS {
                super::assert_left_equivalent(&ring, original.as_ref().unwrap().data(), matrix.as_const(), n);
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


fn to_sparse_matrix_builder<R, V>(ring: R, matrix: Submatrix<V, InternalRow<El<R>>>, n: usize) -> SparseMatrix<R::Type>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V: AsPointerToSlice<InternalRow<El<R>>>
{
    let mut result = SparseMatrix::new(&ring);
    result.add_cols(matrix.col_count() * n);
    for (i, row) in matrix.row_iter().enumerate() {
        result.add_row(i, row.iter().enumerate().flat_map(|(j, int_row)| int_row.data.iter().rev().skip(1).rev().map(move |(k, c)| (j * n + k, ring.clone_el(c)))));
    }
    return result;
}
