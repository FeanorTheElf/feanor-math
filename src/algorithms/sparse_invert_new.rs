use std::io::Write;
use std::mem::swap;
use std::cmp::{min, Ordering};

use crate::ring::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::vector::VectorView;

const EXTENSIVE_RUNTIME_ASSERTS: bool = false;

struct Matrix<T> {
    n: usize,
    rows: Vec<Vec<Vec<(usize, T)>>>
}

#[derive(Copy, Clone)]
struct Block {
    row_start: usize,
    global_col: usize
}

struct MatrixBlock<'a, T> {
    matrix: &'a mut Matrix<T>,
    block: Block
}

impl<T> Matrix<T> {

    fn at<'a>(&'a self, i: usize, j: usize) -> Option<&'a T> {
        let global_col = j / self.n;
        let local_col = j - global_col * self.n;
        self.rows[i][global_col].binary_search_by_key(&local_col, |(j, _)| *j).ok().map(|index| &self.rows[i][global_col][index].1)
    }

    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn col_count(&self) -> usize {
        self.rows[0].len() * self.n
    }

    fn block<'a>(&'a mut self, rows_start: usize, global_col: usize) -> MatrixBlock<'a, T> {
        MatrixBlock { matrix: self, block: Block { row_start: rows_start, global_col: global_col } }
    }

    fn check(&self) {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.rows[i].len() {
                    assert!(self.rows[i][j].is_sorted_by_key(|(idx, _)| *idx));
                    assert!(self.rows[i][j].last().unwrap().0 == usize::MAX);
                    assert!(self.rows[i][j].len() == 1 || self.rows[i][j][self.rows[i][j].len() - 2].0 < usize::MAX);
                }
            }
        }
    }
}

impl<'a, T> MatrixBlock<'a, T> {

    fn at<'b>(&'b self, i: usize, j: usize) -> Option<&'a T>
        where 'b: 'a
    {
        self.matrix.at(self.block.row_start + i, self.block.global_col * self.matrix.n + j)
    }

    fn col_count(&self) -> usize {
        self.matrix.n
    }

    fn row_mut<'b>(&'b mut self, i: usize) -> &'b mut Vec<(usize, T)> {
        &mut self.matrix.rows[i + self.block.row_start][self.block.global_col]
    }

    fn row<'b>(&'b self, i: usize) -> &'b Vec<(usize, T)> {
        &self.matrix.rows[i + self.block.row_start][self.block.global_col]
    }
}

fn print<R>(ring: R, matrix: &Matrix<El<R>>)
    where R: RingStore
{
    let zero = ring.zero();
    for i in 0..matrix.row_count() {
        for j in 0..matrix.col_count() {
            let string = format!("{}", ring.format(matrix.at(i, j).unwrap_or(&zero)));
            print!("{}{}, ", " ".repeat(2 - string.len()), string);
        }
        println!();
    }
}

fn empty<T>(n: usize) -> Matrix<T> {
    Matrix { n: n, rows: Vec::new() }
}

fn identity<R>(ring: R, n: usize, mut use_mem: Matrix<El<R>>) -> Matrix<El<R>>
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
    use_mem.check();
    return use_mem;
}

#[inline(always)]
fn add_row_local<R, const LHS_FACTOR_ONE: bool>(ring: R, lhs: &Vec<(usize, El<R>)>, rhs: &Vec<(usize, El<R>)>, lhs_factor: &El<R>, rhs_factor: &El<R>, mut out: Vec<(usize, El<R>)>) -> Vec<(usize, El<R>)>
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
                let lhs_val = if LHS_FACTOR_ONE { ring.clone_el(&lhs[lhs_idx].1) } else { ring.mul_ref(&lhs[lhs_idx].1, lhs_factor) };
                out.push((lhs_j, lhs_val));
                lhs_idx += 1;
            },
            Ordering::Greater => {
                out.push((rhs_j, ring.mul_ref(&rhs[rhs_idx].1, rhs_factor)));
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
    out.push((usize::MAX, ring.zero()));
    return out;
}

#[inline(never)]
fn linear_combine_rows<R>(ring: R, coeffs: &[(usize, El<R>)], rows: &MatrixBlock<El<R>>, mut out: Vec<(usize, El<R>)>, tmp: &mut Vec<(usize, El<R>)>) -> Vec<(usize, El<R>)>
    where R: RingStore + Copy
{
    out.clear();
    if coeffs.len() == 1 {
        out.push((usize::MAX, ring.zero()));
        return out;
    }
    out.extend(rows.row(coeffs[0].0).iter().map(|(j, c)| (*j, ring.mul_ref(c, &coeffs[0].1))));
    tmp.clear();
    let lhs_factor = ring.one();
    for (idx, c) in coeffs[1..(coeffs.len() - 1)].iter() {
        *tmp = add_row_local::<_, true>(ring, &out, &rows.row(*idx), &lhs_factor, c, std::mem::replace(tmp, Vec::new()));
        swap(&mut out, tmp);
    }
    return out;
}

#[inline(always)]
fn mul_assign<R>(ring: R, lhs: &MatrixBlock<El<R>>, rhs: &MatrixBlock<El<R>>, mut out: Matrix<El<R>>) -> Matrix<El<R>>
    where R: RingStore + Copy
{
    let n = lhs.col_count();
    while out.rows.len() < n {
        out.rows.push(Vec::new());
    }
    out.rows.truncate(n);
    for i in 0..n {
        out.rows[i].resize_with(1, || Vec::new());
        out.rows[i][0].clear();
    }
    let mut tmp = Vec::new();
    for i in 0..n {
        out.rows[i][0] = linear_combine_rows(ring, lhs.row(i), rhs, std::mem::replace(&mut out.rows[i][0], Vec::new()), &mut tmp);
    }
    out.check();
    return out;
}

fn set_block<T>(matrix: &mut MatrixBlock<T>, mut value: Matrix<T>) -> Matrix<T> {
    for (i, r) in (0..matrix.matrix.n).zip(value.rows.drain(..)) {
        *matrix.row_mut(i) = r.into_iter().next().unwrap();
    }
    return value;
}

fn leading_entry<'a, T>(matrix: &'a Matrix<T>, row: usize, global_col: usize) -> (usize, &'a T) {
    let (j, c) = &matrix.rows[row][global_col][0];
    return (*j, c);
}

fn search_pivot_in_block<T>(matrix: &mut Matrix<T>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> Option<usize> {
    matrix.check();
    let n = matrix.n;
    for i in local_pivot_i..n {
        if leading_entry(matrix, global_pivot_i + i, global_pivot_j).0 == local_pivot_j {
            return Some(i);
        }
    }
    return None;
}

fn search_pivot_outside_block<T>(matrix: &mut Matrix<T>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> bool {
    let n = matrix.n;
    matrix.check();
    // there is no solution within the block, start reducing and looking
    for i in (global_pivot_i + n)..matrix.row_count() {

        if EXTENSIVE_RUNTIME_ASSERTS {
            for j in 0..local_pivot_j {
                assert!(matrix.at(i, global_pivot_j * n + j).is_none());
            }
        }

        if leading_entry(matrix, i, global_pivot_j).0 == local_pivot_j {
            matrix.rows.swap(i, global_pivot_i + local_pivot_i);
            return true;
        }
    }
    matrix.check();
    return false;
}

#[inline(never)]
/* GPU */ fn update_rows_with_transform<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, pivot_col: usize, transform: &MatrixBlock<El<R>>) 
where R: RingStore + Copy
{
    let n = matrix.n;
    let mut tmp = empty(n);
    for col in (pivot_col + 1)..(matrix.col_count() / n) {
        let new = mul_assign(ring, transform, &mut matrix.block(rows_start, col), tmp);
        tmp = set_block(&mut matrix.block(rows_start, col), new);
    }
}

/* GPU */ fn eliminate_exterior_rows<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, rows_end: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy
{
    eliminate_interior_rows(ring, matrix, rows_start, rows_end, pivot_rows_start, pivot_rows_end, global_col)
}

#[inline(never)]
/* GPU */ fn eliminate_interior_rows<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, rows_end: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy
{
    if pivot_rows_end <= pivot_rows_start {
        return;
    }
    let n = matrix.n;
    let mut coefficients = Vec::new();
    let mut new_row = Vec::new();
    let mut tmp = Vec::new();

    for i in rows_start..rows_end {
        coefficients.clear();
        for pivot_i in pivot_rows_start..pivot_rows_end {
            let (j, _) = leading_entry(matrix, pivot_i, global_col);
            if let Some(factor) = matrix.at(i, global_col * n + j) {
                coefficients.push((pivot_i, ring.negate(ring.clone_el(factor))));
            }
        }
        coefficients.push((i, ring.one()));
        coefficients.push((usize::MAX, ring.zero()));
        if coefficients.len() > 2 {
            for col in global_col..(matrix.col_count() / n) {
                new_row = linear_combine_rows(ring, &coefficients, &matrix.block(0, col), new_row, &mut tmp);
                swap(&mut new_row, &mut matrix.rows[i][col]);
            }
        }
    }
}

fn get_two_mut<'a, T>(slice: &'a mut [T], i1: usize, i2: usize) -> (&'a mut T, &'a mut T) {
    assert!(i1 < i2);
    let (s1, s2) = (&mut slice[i1..=i2]).split_at_mut(1);
    return (&mut s1[0], &mut s2[s2.len() - 1]);
}

#[inline(never)]
fn local_row_echelon<R>(ring: R, matrix: &mut Matrix<El<R>>, transform_block: &mut MatrixBlock<El<R>>, global_pivot_i: usize, global_pivot_j: usize, start_pivot: (usize, usize)) -> (usize, Result<(), usize>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check();
    let n = matrix.n;
    let mut block = matrix.block(global_pivot_i, global_pivot_j);
    let mut i = start_pivot.0;
    let mut tmp = Vec::new();
    for j in start_pivot.1..n {
        if let Some(new_pivot) = search_pivot_in_block(&mut block.matrix, i, j, global_pivot_i, global_pivot_j) {

            if new_pivot != i {
                let (r1, r2) = get_two_mut(&mut block.matrix.rows[..], i + global_pivot_i, new_pivot + global_pivot_i);
                swap(&mut r1[global_pivot_j], &mut r2[global_pivot_j]);
                transform_block.matrix.rows.swap(i, new_pivot);
            }

            block.matrix.check();

            // check that the left part remains zero and the pivot is nonzero
            if EXTENSIVE_RUNTIME_ASSERTS {
                assert!(block.at(i, j).is_some());
                for col in 0..j {
                    for row in min(i + 1, col + 1)..n {
                        if !(block.at(row, col).is_none()) {
                            println!();
                            print(ring, &block.matrix);
                            println!();
                            assert!(false);
                        }
                    }
                }
            }

            debug_assert!(leading_entry(&block.matrix, i + global_pivot_i, global_pivot_j).0 == j);
            let pivot_inv = ring.checked_div(&ring.one(), &leading_entry(&block.matrix, i + global_pivot_i, global_pivot_j).1).unwrap();

            for (_, c) in block.row_mut(i) {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            for (_, c) in transform_block.row_mut(i) {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            block.matrix.check();
            for elim_i in 0..n {
                if elim_i == i {
                    continue;
                }
                if let Some(factor) = block.at(elim_i, j) {
                    debug_assert!(elim_i < i || leading_entry(&block.matrix, elim_i + global_pivot_i, global_pivot_j).0 == j);
                    let lhs_factor = ring.one();
                    let rhs_factor = ring.negate(ring.clone_el(factor));

                    let new = add_row_local::<_, true>(ring, block.row(elim_i), block.row(i), &lhs_factor, &rhs_factor, tmp);
                    tmp = std::mem::replace(block.row_mut(elim_i), new);
                    debug_assert!(block.at(elim_i, j).is_none());

                    let new = add_row_local::<_, true>(ring, transform_block.row(elim_i), transform_block.row(i), &lhs_factor, &rhs_factor, tmp);
                    tmp = std::mem::replace(transform_block.row_mut(elim_i), new);

                    transform_block.matrix.check();
                    block.matrix.check();
                }
            }
            i += 1;

        } else {
            return (i, Err(j));
        }

    }
    return (i, Ok(()));
}

#[inline(never)]
fn blocked_row_echelon<R, const LOG: bool>(ring: R, matrix: &mut Matrix<El<R>>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    let start = std::time::Instant::now();
    if LOG {
        print!("[{}x{}]", matrix.row_count(), matrix.col_count());
        std::io::stdout().flush().unwrap();
    }
    let mut pivot_row = 0;
    let mut pivot_col = 0;
    let n = matrix.n;
    let col_block_count = matrix.col_count() / n;

    // we have to pad matrix with n zero rows...
    for _ in 0..n {
        matrix.rows.push((0..col_block_count).map(|_| vec![(usize::MAX, ring.zero())]).collect());
    }
    
    let mut local_pivot_i = 0;
    let mut local_pivot_j = 0;
    let mut transform = empty(n);
    while pivot_row + n < matrix.row_count() && pivot_col < col_block_count {
        transform = identity(ring, n, transform);
        let mut transform_block = transform.block(0, 0);

        // now we have the nxn block in row echelon form, with the last (n - produced rows) being zero
        let (new_local_i, current_result) = local_row_echelon(ring, matrix, &mut transform_block, pivot_row, pivot_col, (local_pivot_i, local_pivot_j));

        update_rows_with_transform(ring, matrix, pivot_row, pivot_col, &transform_block);
        
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

                if search_pivot_outside_block(matrix, new_local_i, local_j, pivot_row, pivot_col) {
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
pub fn gb_sparse_row_echelon<F, const LOG: bool>(ring: F, rows: Vec<Vec<(usize, El<F>)>>, col_count: usize) -> Vec<Vec<(usize, El<F>)>>
    where F: DivisibilityRingStore + Copy,
        F::Type: DivisibilityRing
{
    let n = 1024;
    let global_cols = (col_count - 1) / n + 1;
    let mut matrix = Matrix {
        n: n,
        rows: rows.into_iter().rev().map(|row| {
            let mut cols = (0..global_cols).map(|_| Vec::new()).collect::<Vec<_>>();
            for (j, c) in row.into_iter() {
                cols[j / n].push((j % n, c));
            }
            for i in 0..global_cols {
                cols[i].sort_by_key(|(j, _)| *j);
                cols[i].push((usize::MAX, ring.zero()));
            }
            return cols;
        }).collect()
    };
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
        row.into_iter().enumerate().flat_map(|(i, r)| r.into_iter().rev().skip(1).rev().map(move |(j, c)| (j + i * n, c))).collect()
    ).collect();
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test() {
    let ring = Zn::<7>::RING;
    let mut counter1 = 0;
    let mut counter2 = 0;
    let mut matrix = Matrix {
        rows: (0..10).map(|_| (0..3).map(|_| (0..4).map(|k| { counter1 += 1; (k, (counter1 % 6) + 1) }).filter(|_| { counter2 = (counter2 + 1) % 7; counter2 < 2 }).chain(Some((usize::MAX, 0)).into_iter()).collect()).collect()).collect(),
        n: 4
    };

    blocked_row_echelon::<_, false>(ring, &mut matrix);

    print(ring, &matrix);
}