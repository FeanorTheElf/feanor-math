use std::io::Write;
use std::mem::swap;
use std::cmp::{min, Ordering};
use std::time::Instant;

use crate::matrix::Matrix;
use crate::parallel::{potential_parallel_for_each, column_iterator};
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::ring::*;
use crate::divisibility::DivisibilityRingStore;
use crate::vector::{VectorView, VectorViewMut};

const EXTENSIVE_RUNTIME_ASSERTS: bool = false;

pub struct SparseMatrixBuilder<R>
    where R: ?Sized + RingBase
{
    zero: R::Element,
    rows: Vec<Vec<(usize, R::Element)>>,
    col_permutation: Vec<usize>,
    col_count: usize
}

impl<R> SparseMatrixBuilder<R>
    where R: ?Sized + RingBase
{
    pub fn new<S>(ring: &S) -> Self
        where S: RingStore<Type = R>
    {
        SparseMatrixBuilder {
            rows: Vec::new(),
            col_count: 0,
            col_permutation: Vec::new(),
            zero: ring.zero()
        }
    }

    pub fn clone_matrix<S>(&self, ring: S) -> Self
        where S: RingStore<Type = R>
    {
        SparseMatrixBuilder {
            zero: ring.clone_el(&self.zero), 
            rows: self.rows.iter().map(|row| row.iter().map(|(i, x)| (*i, ring.clone_el(x))).collect()).collect(), 
            col_permutation: self.col_permutation.clone(), 
            col_count: self.col_count
        }
    }

    pub fn add_col(&mut self, j: usize) {
        self.col_permutation.insert(j, self.col_count);
        self.col_count += 1;
    }

    pub fn add_cols(&mut self, number: usize) {
        for _ in 0..number {
            self.add_col(self.col_count());
        }
    }

    pub fn add_zero_row(&mut self, i: usize) {
        self.rows.insert(i, Vec::new())
    }

    pub fn add_row<I>(&mut self, i: usize, values: I)
        where I: Iterator<Item = (usize, R::Element)>
    {
        let mut row = values
            .map(|(j, x)| (self.col_permutation[j], x))
            .collect::<Vec<_>>();
        row.sort_by_key(|(j, _)| *j);
        assert!((1..row.len()).all(|k| row[k].0 != row[k - 1].0));
        self.rows.insert(i, row);
    }

    pub fn set(&mut self, i: usize, j: usize, el: R::Element) -> Option<R::Element> {
        let row = &mut self.rows[i];
        let result = match row.binary_search_by_key(&self.col_permutation[j], |(c, _)| *c) {
            Ok(idx) => Some(std::mem::replace(&mut row.at_mut(idx).1, el)),
            Err(idx) => {
                row.insert(idx, (self.col_permutation[j], el));
                None
            }
        };
        debug_assert!((1..row.len()).all(|k| row[k].0 != row[k - 1].0));
        return result;
    }

    fn into_internal_matrix(self, n: usize, ring: &R) -> InternalMatrix<R::Element> {
        let mut inverted_permutation = (0..self.col_permutation.len()).collect::<Vec<_>>();
        for (i, j) in self.col_permutation.iter().enumerate() {
            inverted_permutation[*j] = i;
        }
        for i in 0..self.col_permutation.len() {
            debug_assert!(inverted_permutation[self.col_permutation[i]] == i);
            debug_assert!(self.col_permutation[inverted_permutation[i]] == i);
        }
        let global_cols = (self.col_count - 1) / n + 1;
        InternalMatrix {
            global_col_count: global_cols,
            n: n,
            zero: self.zero,
            rows: self.rows.into_iter().map(|row| {
                let mut cols = (0..global_cols).map(|_| Vec::new()).collect::<Vec<_>>();
                for (j, c) in row.into_iter() {
                    if !ring.is_zero(&c) {
                        let col = inverted_permutation[j];
                        cols[col / n].push((col % n, c));
                    }
                }
                for i in 0..global_cols {
                    cols[i].sort_by_key(|(j, _)| *j);
                    cols[i].push((usize::MAX, ring.zero()));
                }
                return cols;
            }).collect()
        }
    }
}

impl<R> Matrix<R> for SparseMatrixBuilder<R> 
    where R: ?Sized + RingBase
{    
    fn col_count(&self) -> usize {
        self.col_count
    }

    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn at(&self, i: usize, j: usize) -> &R::Element {
        match self.rows.at(i).binary_search_by_key(&self.col_permutation[j], |(c, _)| *c) {
            Ok(idx) => &self.rows.at(i).at(idx).1,
            Err(_) => &self.zero
        }
    }
}

struct InternalMatrix<T> {
    rows: Vec<Vec<Vec<(usize, T)>>>,
    global_col_count: usize,
    n: usize,
    zero: T
}

impl<T> InternalMatrix<T> {

    fn entry_at<'a>(&'a self, i: usize, j_global: usize, j_local: usize) -> Option<&'a T> {
        at(j_local, &self.rows[i][j_global])
    }

    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn check<R>(&self, ring: R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.rows[i].len() {
                    assert!(self.rows[i][j].is_sorted_by_key(|(idx, _)| *idx));
                    assert!(self.rows[i][j].iter().rev().skip(1).all(|(_, x)| !ring.is_zero(x)));
                    assert!((1..self.rows[i][j].len()).all(|k| self.rows[i][j][k - 1].0 != self.rows[i][j][k].0));
                    assert!(self.rows[i][j].last().unwrap().0 == usize::MAX);
                    assert!(self.rows[i][j].len() == 1 || self.rows[i][j][self.rows[i][j].len() - 2].0 < usize::MAX);
                }
            }
        }
    }
}

impl<R> Matrix<R> for InternalMatrix<R::Element>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn col_count(&self) -> usize {
        self.global_col_count * self.n
    }

    fn at(&self, i: usize, j: usize) -> &R::Element {
        self.entry_at(i, j / self.n, j % self.n).unwrap_or(&self.zero)
    }
}

fn empty<T>(n: usize, global_col_count: usize, zero: T) -> InternalMatrix<T> {
    InternalMatrix { n: n, global_col_count: global_col_count, rows: Vec::new(), zero: zero }
}

fn at<'a, T>(i: usize, data: &'a [(usize, T)]) -> Option<&'a T> {
    data.binary_search_by_key(&i, |(j, _)| *j).ok().map(|idx| &data[idx].1)
}

fn get_two_mut<'a, T>(data: &'a mut [T], fst: usize, snd: usize) -> (&'a mut T, &'a mut T) {
    debug_assert!(snd > fst);
    let (fst_part, snd_part) = data.split_at_mut(snd);
    return (&mut fst_part[fst], &mut snd_part[0]);
}

fn identity<R>(ring: R, n: usize, mut use_mem: InternalMatrix<El<R>>) -> InternalMatrix<El<R>>
    where R: RingStore
{
    while use_mem.rows.len() < n {
        use_mem.rows.push(Vec::new());
    }
    use_mem.rows.truncate(n);
    for i in 0..n {
        use_mem.rows[i].resize_with(1, || Vec::new());
        use_mem.rows[i][0].clear();
        use_mem.rows[i][0].extend([(i, ring.one()), (usize::MAX, ring.zero())].into_iter());
    }
    use_mem.check(ring);
    return use_mem;
}

#[inline(always)]
fn add_row_local<R, const LHS_FACTOR_ONE: bool>(ring: R, lhs: &[(usize, El<R>)], rhs: &[(usize, El<R>)], lhs_factor: &El<R>, rhs_factor: &El<R>, mut out: Vec<(usize, El<R>)>) -> Vec<(usize, El<R>)>
    where R: RingStore
{
    let mut lhs_idx = 0;
    let mut rhs_idx = 0;
    debug_assert!(lhs.last().unwrap().0 == usize::MAX);
    debug_assert!(rhs.last().unwrap().0 == usize::MAX);
    out.clear();
    while lhs_idx + 1 < lhs.len() || rhs_idx + 1 < rhs.len() {
        let lhs_j = lhs[lhs_idx].0;
        let rhs_j = rhs[rhs_idx].0;
        
        match lhs_j.cmp(&rhs_j) {
            Ordering::Less => {
                let lhs_val = if LHS_FACTOR_ONE {
                    ring.clone_el(&lhs[lhs_idx].1)
                } else {
                    ring.mul_ref(&lhs[lhs_idx].1, lhs_factor)
                };
                if LHS_FACTOR_ONE || !ring.is_zero(&lhs_val) {
                    out.push((lhs_j, lhs_val))
                };
                lhs_idx += 1;
            },
            Ordering::Greater => {
                let rhs_val = ring.mul_ref(&rhs[rhs_idx].1, rhs_factor);
                if !ring.is_zero(&rhs_val) {
                    out.push((rhs_j, rhs_val));
                }
                rhs_idx += 1;
            },
            Ordering::Equal => {
                let lhs_val = if LHS_FACTOR_ONE { ring.clone_el(&lhs[lhs_idx].1) } else { ring.mul_ref(&lhs[lhs_idx].1, lhs_factor) };
                let value = ring.add(lhs_val, ring.mul_ref(&rhs[rhs_idx].1, rhs_factor));
                if !ring.is_zero(&value) {
                    out.push((lhs_j, value));
                }
                lhs_idx += 1;
                rhs_idx += 1;
            }
        }
    }
    assert!(lhs_idx + 1 == lhs.len() && rhs_idx + 1 == rhs.len());
    if EXTENSIVE_RUNTIME_ASSERTS {
        assert!(out.iter().all(|(_, x)| !ring.is_zero(x)));
    }
    out.push((usize::MAX, ring.zero()));
    return out;
}

#[inline(never)]
fn linear_combine_rows<'a, R, I>(ring: R, coeffs: &[(usize, El<R>)], mut rows: I, mut out: Vec<(usize, El<R>)>, tmp: &mut Vec<(usize, El<R>)>) -> Vec<(usize, El<R>)>
    where R: RingStore + Copy,
        I: Iterator<Item = &'a [(usize, El<R>)]>,
        El<R>: 'a
{
    out.clear();
    if coeffs.len() == 1 {
        out.push((usize::MAX, ring.zero()));
        return out;
    }
    let mut last = coeffs[0].0;
    rows.advance_by(last).unwrap();
    out.extend(rows.next().unwrap().iter().map(|(j, c)| (*j, ring.mul_ref(c, &coeffs[0].1))));
    tmp.clear();
    let lhs_factor = ring.one();
    for (idx, c) in coeffs[1..(coeffs.len() - 1)].iter() {
        rows.advance_by(*idx - last - 1).unwrap();
        last = *idx;
        *tmp = add_row_local::<_, true>(ring, &out, rows.next().unwrap(), &lhs_factor, c, std::mem::replace(tmp, Vec::new()));
        swap(&mut out, tmp);
    }
    return out;
}

#[inline(always)]
fn mul_assign<'a, R, I>(ring: R, lhs: &[Vec<(usize, El<R>)>], rhs: I, mut out: Vec<Vec<(usize, El<R>)>>) -> Vec<Vec<(usize, El<R>)>>
    where R: RingStore + Copy,
        I: Iterator<Item = &'a [(usize, El<R>)]> + Clone,
        El<R>: 'a
{
    let n = lhs.len();
    while out.len() < n {
        out.push(Vec::new());
    }
    out.truncate(n);
    for i in 0..n {
        out[i].clear();
    }
    let mut tmp = Vec::new();
    for i in 0..n {
        out[i] = linear_combine_rows(ring, &lhs[i], rhs.clone(), std::mem::replace(&mut out[i], Vec::new()), &mut tmp);
    }
    return out;
}

#[inline(always)]
fn transform_2d<R>(ring: R, transform: &[[El<R>; 2]; 2], rows: [&mut Vec<(usize, El<R>)>; 2], tmp: &mut [Vec<(usize, El<R>)>; 2])
    where R: RingStore + Copy
{
    let [lhs, rhs] = rows;
    let lhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[0][0], &transform[0][1], std::mem::replace(&mut tmp[0], Vec::new()));
    let rhs_new = add_row_local::<R, false>(ring, lhs, rhs, &transform[1][0], &transform[1][1], std::mem::replace(&mut tmp[1], Vec::new()));
    tmp[0] = std::mem::replace(lhs, lhs_new);
    tmp[1] = std::mem::replace(rhs, rhs_new);
}

fn leading_entry<'a, T>(matrix: &'a InternalMatrix<T>, row: usize, global_col: usize) -> (usize, &'a T) {
    let (j, c) = &matrix.rows[row][global_col][0];
    return (*j, c);
}

///
/// In the block of matrtix given by `global_pivot`, performs unimodular row operation
/// to ensure that the element in the pivot position divides all elements below it. 
/// 
#[inline(never)]
fn local_make_pivot_ideal_gen<R>(ring: R, matrix: &mut InternalMatrix<El<R>>, transform: &mut InternalMatrix<El<R>>, local_pivot: (usize, usize), global_pivot: (usize, usize), tmp: &mut [Vec<(usize, El<R>)>; 2]) 
    where R: PrincipalIdealRingStore + Copy,
        R::Type: PrincipalIdealRing
{
    let n = matrix.n;
    matrix.check(ring);
    transform.check(ring);
    let current = leading_entry(&matrix, local_pivot.0 + global_pivot.0, global_pivot.1);
    let mut current = if current.0 != local_pivot.1 {
        ring.zero()
    } else {
        ring.clone_el(current.1)
    };
    for i in (local_pivot.0 + 1)..n {
        if ring.is_unit(&current) {
            break;
        }
        let entry = leading_entry(matrix, i + global_pivot.0, global_pivot.1);
        debug_assert!(entry.0 >= local_pivot.1);
        if entry.0 == local_pivot.1 && ring.checked_div(entry.1, &current).is_none() {
            let (s, t, d) = ring.ideal_gen(&current, entry.1);
            let local_transform = [[s, t], [ring.checked_div(entry.1, &d).unwrap(), ring.checked_div(&current, &d).unwrap()]];
            let (fst_row, snd_row) = get_two_mut(&mut matrix.rows, local_pivot.0 + global_pivot.0, i + global_pivot.0);
            transform_2d(ring, &local_transform, [&mut fst_row[global_pivot.1], &mut snd_row[global_pivot.1]], tmp);
            let (fst_row, snd_row) = get_two_mut(&mut transform.rows, local_pivot.0, i);
            transform_2d(ring, &local_transform, [&mut fst_row[0], &mut snd_row[0]], tmp);
            current = d;
        }
    }
    matrix.check(ring);
    transform.check(ring);
}

#[inline(never)]
fn search_pivot_in_block<T>(matrix: &InternalMatrix<T>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> Option<usize> {
    let n = matrix.n;
    for i in local_pivot_i..n {
        if leading_entry(matrix, global_pivot_i + i, global_pivot_j).0 == local_pivot_j {
            return Some(i);
        }
    }
    return None;
}

#[inline(never)]
fn swap_pivot_outside_block<T>(matrix: &mut InternalMatrix<T>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> bool {
    let n = matrix.n;
    // there is no solution within the block, start reducing and looking
    for i in (global_pivot_i + n)..matrix.row_count() {

        if EXTENSIVE_RUNTIME_ASSERTS {
            for j in 0..local_pivot_j {
                assert!(matrix.entry_at(i, global_pivot_j, j).is_none());
            }
        }

        if leading_entry(matrix, i, global_pivot_j).0 == local_pivot_j {
            matrix.rows.swap(i, global_pivot_i + local_pivot_i);
            return true;
        }
    }
    return false;
}

#[inline(never)]
fn update_rows_with_transform<R>(ring: R, matrix: &mut InternalMatrix<El<R>>, rows_start: usize, pivot_col: usize, transform: &[Vec<(usize, El<R>)>]) 
    where R: RingStore + Copy + Sync,
        El<R>: Send + Sync
{
    matrix.check(ring);
    potential_parallel_for_each(
        column_iterator(&mut matrix.rows[rows_start..], (pivot_col + 1)..(matrix.global_col_count)), 
        || Vec::new(), 
        |tmp, rows| 
    {
        let mut rows = rows;
        let mut new = mul_assign(
            ring, 
            transform, 
            rows.iter().map(|x| &x[..]), 
            std::mem::replace(tmp, Vec::new())
        );
        for (target, new) in rows.iter_mut().zip(new.iter_mut()) {
            swap(target, new);
        }
        *tmp = new;
    });
    matrix.check(ring);
}

#[inline(never)]
fn eliminate_interior_rows<R>(_ring: R, _matrix: &mut InternalMatrix<El<R>>, _rows_start: usize, _rows_end: usize, _pivot_rows_start: usize, _pivot_rows_end: usize, _global_col: usize)
    where R: RingStore + Copy,
        El<R>: Send + Sync,
        R: Sync
{
    // it turns out that we are faster by not eliminating the upper corner
}

fn eliminate_exterior_rows<R>(ring: R, matrix: &mut InternalMatrix<El<R>>, rows_start: usize, rows_end: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy,
        El<R>: Send + Sync,
        R: Sync
{
    matrix.check(ring);
    if pivot_rows_end <= pivot_rows_start {
        return;
    }
    if rows_end <= rows_start {
        return;
    }
    assert!(rows_start >= pivot_rows_end || pivot_rows_start >= rows_end);

    let global_col_count = matrix.global_col_count;

    let (pivot_rows, work_rows) = if rows_start >= pivot_rows_end {
        let (pivot_rows, work_rows) = (&mut matrix.rows[pivot_rows_start..rows_end]).split_at_mut(rows_start - pivot_rows_start);
        (&mut pivot_rows[..(pivot_rows_end - pivot_rows_start)], work_rows)
    } else {
        let (work_rows, pivot_rows) = (&mut matrix.rows[rows_start..pivot_rows_end]).split_at_mut(pivot_rows_start - rows_start);
        (pivot_rows, &mut work_rows[..(rows_end - rows_start)])
    };

    potential_parallel_for_each(work_rows, || (Vec::new(), Vec::new(), Vec::new()), |(coefficients, new_row, tmp), row| {
        coefficients.clear();
        for pivot_i in 0..pivot_rows.len() {
            let (j, _) = pivot_rows[pivot_i][global_col][0];
            if let Some(factor) = at(j, &row[global_col]) {
                coefficients.push((pivot_i, ring.negate(ring.clone_el(factor))));
            }
        }
        coefficients.push((usize::MAX, ring.zero()));
        if coefficients.len() > 1 {
            for col in global_col..global_col_count {
                *new_row = linear_combine_rows(ring, &coefficients, pivot_rows.iter().map(|r| &r[col][..]), std::mem::replace(new_row, Vec::new()), tmp);
                *tmp = add_row_local::<_, true>(ring, &row[col], &new_row, &ring.one(), &ring.one(), std::mem::replace(tmp, Vec::new()));
                swap(tmp, &mut row[col]);
            }
        }
    });
    matrix.check(ring);
}

#[inline(never)]
fn local_row_echelon<R>(ring: R, matrix: &mut InternalMatrix<El<R>>, transform: &mut InternalMatrix<El<R>>, global_pivot_i: usize, global_pivot_j: usize, start_pivot: (usize, usize)) -> (usize, Result<(), usize>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing
{
    matrix.check(ring);
    let n = matrix.n;
    let mut i = start_pivot.0;
    let mut tmp = [Vec::new(), Vec::new()];
    for j in start_pivot.1..n {
        local_make_pivot_ideal_gen(ring, matrix, transform, (i, j), (global_pivot_i, global_pivot_j), &mut tmp);
        let pivot_entry = leading_entry(&matrix, i + global_pivot_i, global_pivot_j);
        if pivot_entry.0 == j && ring.is_unit(pivot_entry.1) {

            // check that the left part remains zero and the pivot is nonzero
            if EXTENSIVE_RUNTIME_ASSERTS {
                assert!(matrix.entry_at(global_pivot_i + i, global_pivot_j, j).is_some());
                for col in 0..j {
                    for row in min(i + 1, col + 1)..n {
                        if !(matrix.entry_at(global_pivot_i + row, global_pivot_j, col).is_none()) {
                            println!();
                            println!("{}", matrix.format(&ring));
                            println!();
                            assert!(false);
                        }
                    }
                }
            }

            debug_assert!(leading_entry(&matrix, i + global_pivot_i, global_pivot_j).0 == j);
            let pivot_inv = ring.checked_div(&ring.one(), &leading_entry(&matrix, i + global_pivot_i, global_pivot_j).1).unwrap();

            for (_, c) in &mut matrix.rows[global_pivot_i + i][global_pivot_j] {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            for (_, c) in &mut transform.rows[i][0] {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            for elim_i in 0..n {
                if elim_i == i {
                    continue;
                }
                if let Some(factor) = matrix.entry_at(global_pivot_i + elim_i, global_pivot_j, j) {
                    debug_assert!(elim_i < i || leading_entry(&matrix, elim_i + global_pivot_i, global_pivot_j).0 == j);
                    let lhs_factor = ring.one();
                    let rhs_factor = ring.negate(ring.clone_el(factor));

                    let new = add_row_local::<_, true>(ring, &matrix.rows[global_pivot_i + elim_i][global_pivot_j], &matrix.rows[global_pivot_i + i][global_pivot_j], &lhs_factor, &rhs_factor, std::mem::replace(&mut tmp[0], Vec::new()));
                    tmp[0] = std::mem::replace(&mut matrix.rows[global_pivot_i + elim_i][global_pivot_j], new);

                    let new = add_row_local::<_, true>(ring, &transform.rows[elim_i][0], &transform.rows[i][0], &lhs_factor, &rhs_factor, std::mem::replace(&mut tmp[0], Vec::new()));
                    tmp[0] = std::mem::replace(&mut transform.rows[elim_i][0], new);
                }
            }
            i += 1;

        } else {
            matrix.check(ring);
            return (i, Err(j));
        }

    }
    matrix.check(ring);
    return (i, Ok(()));
}

#[inline(never)]
fn blocked_row_echelon<R, const LOG: bool>(ring: R, matrix: &mut InternalMatrix<El<R>>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync
{
    if LOG {
        print!("[{}x{}]", matrix.row_count(), matrix.global_col_count * matrix.n);
        std::io::stdout().flush().unwrap();
    }
    let start = Instant::now();

    let mut pivot_row = 0;
    let mut pivot_col = 0;
    let n = matrix.n;
    let col_block_count = matrix.global_col_count;

    // we have to pad matrix with n zero rows...
    for _ in 0..n {
        matrix.rows.push((0..col_block_count).map(|_| vec![(usize::MAX, ring.zero())]).collect());
    }
    
    let mut local_pivot_i = 0;
    let mut local_pivot_j = 0;
    while pivot_row + n < matrix.row_count() && pivot_col < col_block_count {
        let mut transform = identity(ring, n, empty(n, 1, ring.zero()));

        // now we have the nxn block in row echelon form, with the last (n - produced rows) being zero
        let (new_local_i, current_result) = 
            local_row_echelon(ring, matrix, &mut transform, pivot_row, pivot_col, (local_pivot_i, local_pivot_j));

        update_rows_with_transform(ring, matrix, pivot_row, pivot_col, &transform.rows.into_iter().map(|r| r.into_iter().next().unwrap()).collect::<Vec<_>>());
        
        eliminate_exterior_rows(ring, matrix, pivot_row + n, matrix.row_count() - n, pivot_row + local_pivot_i, pivot_row + new_local_i, pivot_col);

        match current_result {
            Ok(()) => {

                eliminate_interior_rows(ring, matrix, 0, pivot_row, pivot_row, pivot_row + new_local_i, pivot_col);

                pivot_col += 1;
                pivot_row += new_local_i;
                local_pivot_i = 0;
                local_pivot_j = 0;

                if LOG {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
            },
            Err(local_j) => {

                if swap_pivot_outside_block(matrix, new_local_i, local_j, pivot_row, pivot_col) {
                    local_pivot_i = new_local_i;
                    local_pivot_j = local_j;
                } else {
                    local_pivot_i = new_local_i;
                    local_pivot_j = local_j + 1;
                }
            }
        }
    }

    // remove the padding
    for _ in 0..n {
        matrix.rows.pop();
    }
    if LOG {
        let end = std::time::Instant::now();
        print!("[{}ms]", (end - start).as_millis());
        std::io::stdout().flush().unwrap();
    }
}

#[inline(never)]
pub fn gb_sparse_row_echelon<R, const LOG: bool>(ring: R, matrix: SparseMatrixBuilder<R::Type>, block_size: usize) -> Vec<Vec<(usize, El<R>)>>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        El<R>: Send + Sync,
        R: Sync
{
    let n = block_size;
    let global_cols = (matrix.col_count() - 1) / n + 1;
    let mut matrix = matrix.into_internal_matrix(n, ring.get_ring());
    matrix.check(ring);
    blocked_row_echelon::<_, LOG>(ring, &mut matrix);

    if EXTENSIVE_RUNTIME_ASSERTS {
        let mut last = -1;
        for i in 1..matrix.row_count() {
            let mut j = 0;
            while j < global_cols && leading_entry(&matrix, i, j).0 == usize::MAX {
                j += 1;
            }
            if j < global_cols {
                let new = leading_entry(&matrix, i, j).0 + j * n;
                assert!((new as i64) > last as i64);
                last = new as i64;
            } else {
                last = i64::MAX;
            }
        }
    }

    return matrix.rows.into_iter().map(|row| 
        row.into_iter().enumerate().flat_map(|(i, r)| r.into_iter().rev().skip(1).rev().map(move |(j, c)| (j + i * n, c)).inspect(|(_, c)| assert!(!ring.is_zero(c)))).collect()
    ).collect();
}

#[cfg(test)]
use crate::rings::zn::zn_static::*;
#[cfg(test)]
use crate::assert_matrix_eq;

#[cfg(test)]
fn assert_is_correct_row_echelon<R>(ring: R, expected: SparseMatrixBuilder<R::Type>, actual: &SparseMatrixBuilder<R::Type>)
    where R: RingStore
{
    let n = actual.row_count();
    let m = actual.col_count();
    assert_eq!(n, expected.row_count());
    assert_eq!(m, expected.col_count());
    let mut expected: InternalMatrix<<<R as RingStore>::Type as RingBase>::Element> = expected.into_internal_matrix(m, ring.get_ring());
    let mut last_pivot_i = None;
    for j in 0..m {
        let pivot_i = (0..n).rev().filter(|i| !ring.is_zero(actual.at(*i, j))).next();
        assert_eq!((0..n).rev().filter(|i| !ring.is_zero(<_ as Matrix<R::Type>>::at(&expected, *i, j))).next(), pivot_i);
        if let Some(pivot_i) = pivot_i {
            if last_pivot_i.is_some() && pivot_i <= last_pivot_i.unwrap() {
                continue;
            }
            assert!(ring.is_one(<_ as Matrix<R::Type>>::at(&expected, pivot_i, j)));
            for i in 0..pivot_i {
                assert!(ring.is_zero(<_ as Matrix<R::Type>>::at(&expected, i, j)));
                expected.rows[i][0] = add_row_local::<_, true>(&ring, &expected.rows[i][0], &expected.rows[pivot_i][0], &ring.one(), <_ as Matrix<R::Type>>::at(actual, i, j), Vec::new());
            }
            for (_, x) in &mut expected.rows[pivot_i][0] {
                ring.mul_assign_ref(x, <_ as Matrix<R::Type>>::at(actual, pivot_i, j))
            }
            last_pivot_i = Some(pivot_i);
        }
    }
    assert_matrix_eq!(&ring, &expected, actual);
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

    let mut expected = SparseMatrixBuilder::new(&R);
    expected.add_cols(5);
    expected.add_row(0, sparsify([1, 0, 0, 0, 5]));
    expected.add_row(1, sparsify([0, 1, 0, 0, 6]));
    expected.add_row(2, sparsify([0, 0, 1, 0, 6]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, expected.clone_matrix(R), &actual);
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

    let mut expected = SparseMatrixBuilder::new(&R);
    expected.add_cols(6);
    expected.add_row(0, sparsify([1, 10, 0, 0, 0, 0]));
    expected.add_row(1, sparsify([0, 0, 1, 0, 0, 0]));
    expected.add_row(2, sparsify([0, 0, 0, 1, 0, 1]));
    expected.add_row(3, sparsify([0, 0, 0, 0, 1, 13]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
        actual.add_cols(6);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, expected.clone_matrix(R), &actual);
    }
}

#[test]
#[ignore]
fn test_gb_sparse_row_echelon_no_field() {
    let R = Zn::<18>::RING;
    let sparsify = |row: [u64; 5]| row.into_iter().enumerate().filter(|(_, x)| !R.is_zero(&x));

    let mut matrix: SparseMatrixBuilder<_> = SparseMatrixBuilder::new(&R);
    matrix.add_cols(5);
    matrix.add_row(0, sparsify([9, 3, 0, 1, 1]));
    matrix.add_row(1, sparsify([6, 13, 0, 0, 1]));
    matrix.add_row(2, sparsify([0, 0, 11, 0, 1]));
    matrix.add_row(3, sparsify([0, 12, 0, 0, 1]));

    let mut expected = SparseMatrixBuilder::new(&R);
    expected.add_cols(5);
    matrix.add_row(0, sparsify([3, 8, 0, 1, 0]));
    matrix.add_row(1, sparsify([0, 3, 0, 0, 16]));
    matrix.add_row(2, sparsify([0, 0, 1, 0, 5]));
    matrix.add_row(3, sparsify([0, 0, 0, 2, 7]));

    for block_size in 1..10 {
        let mut actual = SparseMatrixBuilder::new(&R);
        actual.add_cols(5);
        for row in gb_sparse_row_echelon::<_, false>(&R, matrix.clone_matrix(&R), block_size) {
            actual.add_row(actual.row_count(), row.into_iter());
        }
        assert_is_correct_row_echelon(R, expected.clone_matrix(R), &actual);
    }
}