use std::io::Write;
use std::ops::Range;
use std::cmp::{min, Ordering};

use crate::ring::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::vector::VectorView;

use crate::rings::zn::zn_static::Zn;

const EXTENSIVE_RUNTIME_ASSERTS: bool = false;

struct Matrix<T> {
    n: usize,
    rows: Vec<Vec<Vec<(usize, T)>>>
}

#[derive(Copy, Clone)]
struct Block {
    row_start: usize,
    row_end: usize,
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

    fn block<'a>(&'a mut self, rows: Range<usize>, global_col: usize) -> MatrixBlock<'a, T> {
        MatrixBlock { matrix: self, block: Block { row_start: rows.start, row_end: rows.end, global_col: global_col } }
    }

    fn check(&self) {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.rows[i].len() {
                    assert!(self.rows[i][j].last().unwrap().0 == usize::MAX);
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

    fn row_count(&self) -> usize {
        self.block.row_end - self.block.row_start
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

fn identity<R>(ring: R, n: usize) -> Matrix<El<R>>
    where R: RingStore
{
    Matrix { n: n, rows: (0..n).map(|i| vec![vec![(i, ring.one()), (usize::MAX, ring.zero())]]).collect() }
}

fn sub_row_global<R>(ring: R, matrix: &mut Matrix<El<R>>, i_dst: usize, i_src: usize, global_col_start: usize, factor: &El<R>)
    where R: RingStore + Copy
{
    let n = matrix.n;
    for col in global_col_start..(matrix.col_count() / n) {
        matrix.rows[i_dst][col] = sub_row_local(ring, &matrix.rows[i_dst][col], &matrix.rows[i_src][col], factor, Vec::new());
    }
}

fn sub_row_local<R>(ring: R, dst: &Vec<(usize, El<R>)>, src: &Vec<(usize, El<R>)>, factor: &El<R>, mut out: Vec<(usize, El<R>)>) -> Vec<(usize, El<R>)>
    where R: RingStore
{
    let mut dst_idx = 0;
    let mut src_idx = 0;
    assert!(dst.last().unwrap().0 == usize::MAX);
    assert!(src.last().unwrap().0 == usize::MAX);
    while dst_idx + 1 < dst.len() || src_idx + 1 < src.len() {
        let dst_j = dst[dst_idx].0;
        let src_j = src[src_idx].0;
        
        match dst_j.cmp(&src_j) {
            Ordering::Less => {
                out.push((dst_j, ring.clone_el(&dst[dst_idx].1)));
                dst_idx += 1;
            },
            Ordering::Greater => {
                out.push((src_j, ring.negate(ring.mul_ref(&src[src_idx].1, factor))));
                src_idx += 1;
            },
            Ordering::Equal => {
                let subtract = ring.mul_ref(&src[src_idx].1, factor);
                let value = ring.sub_ref_fst(&dst[dst_idx].1, subtract);
                if !ring.is_zero(&value) {
                    out.push((dst_j, value));
                }
                dst_idx += 1;
                src_idx += 1;
            }
        }
    }
    out.push((usize::MAX, ring.zero()));
    return out;
}

fn set_block<T>(matrix: &mut MatrixBlock<T>, mut value: Matrix<T>) -> Matrix<T> {
    for (i, r) in (0..matrix.matrix.n).zip(value.rows.drain(..)) {
        *matrix.row_mut(i) = r.into_iter().next().unwrap();
    }
    return value;
}

fn mul_assign<R>(ring: R, lhs: &MatrixBlock<El<R>>, rhs: &MatrixBlock<El<R>>, mut out: Matrix<El<R>>) -> Matrix<El<R>>
    where R: RingStore + Copy
{
    assert!(out.rows.len() == 0);
    let n = lhs.col_count();
    for _ in 0..n {
        out.rows.push(vec![Vec::new()]);
    }
    for i in 0..n {
        for j in 0..n {
            let value = ring.sum(lhs.row(i)[..(lhs.row(i).len() - 1)].iter()
                .filter_map(|(k, a)| rhs.at(*k, j).map(|b| (a, b)))
                .map(|(a, b)| ring.mul_ref(a, b)));
            if !ring.is_zero(&value) {
                out.rows[i][0].push((j, value));
            }
        }
        out.rows[i][0].push((usize::MAX, ring.zero()));
    }
    out.check();
    return out;
}

fn check_no_zeros<R>(ring: R, matrix: &Matrix<El<R>>)
    where R: RingStore
{
    for i in 0..matrix.row_count() {
        for j in 0..(matrix.col_count() / matrix.n) {
            debug_assert!(matrix.rows[i][j].iter().all(|(j, c)| *j == usize::MAX || !ring.is_zero(c)));
        }
    }
}

fn leading_entry<'a, T>(matrix: &'a Matrix<T>, row: usize, global_col: usize) -> (usize, &'a T) {
    let (j, c) = &matrix.rows[row][global_col][0];
    return (*j, c);
}

fn partial_eliminate_row<R>(ring: R, matrix: &mut Matrix<El<R>>, row: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy
{
    let n = matrix.n;
    for i in pivot_rows_start..pivot_rows_end {
        if leading_entry(matrix, i, global_col).0 != usize::MAX {
            if let Some(factor) = matrix.at(row, global_col * n + leading_entry(matrix, i, global_col).0) {
                assert!(leading_entry(matrix, row, global_col).0 == leading_entry(matrix, i, global_col).0);
                let factor = ring.clone_el(factor);
                sub_row_global(ring, matrix, row, i, global_col, &factor);
            }
        }
    }
}

fn complete_eliminate_rowblock<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, pivot_rows_start: usize, global_col: usize)
    where R: RingStore + Copy
{
    let n = matrix.n;
    for row in rows_start..(rows_start + n) {
        partial_eliminate_row(ring, matrix, row, pivot_rows_start, pivot_rows_start + n, global_col);
    }

    if EXTENSIVE_RUNTIME_ASSERTS {
        for row in rows_start..(rows_start + n) {
            assert!(leading_entry(matrix, row, global_col).0 == usize::MAX);
        }
    }
}

fn search_pivot_in_block<R>(ring: R, matrix: &mut Matrix<El<R>>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> bool
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check();
    let n = matrix.n;
    for i in local_pivot_i..n {
        if matrix.at(global_pivot_i + i, global_pivot_j * n + local_pivot_j).is_some() {
            assert!(leading_entry(matrix, global_pivot_i + i, global_pivot_j).0 == local_pivot_j);
            // we find a solution within the block
            matrix.rows.swap(global_pivot_i + i, global_pivot_i + local_pivot_i);
            return true;
        }
    }
    return false;
}

///
/// Assumes that the current global pivot block has been row-echelonified up to the local pivot, but the rest
/// of the block does not have another row with nonzero element in local_pivot_j
/// 
/// Hence, we have to search the lower rows of the matrix, but note that the corresponding columns have not yet
/// been eliminated, which has to be done first before we can look for nonzero entries. 
/// 
fn search_eliminate_pivot<R>(ring: R, matrix: &mut Matrix<El<R>>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> bool
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check();
    let n = matrix.n;
    for i in local_pivot_i..n {
        if matrix.at(global_pivot_i + i, global_pivot_j * n + local_pivot_j).is_some() {
            assert!(leading_entry(matrix, global_pivot_i + i, global_pivot_j).0 == local_pivot_j);
            // we find a solution within the block
            matrix.rows.swap(global_pivot_i + i, global_pivot_i + local_pivot_i);
            return true;
        }
    }
    matrix.check();
    // there is no solution within the block, start reducing and looking
    for i in (global_pivot_i + n)..matrix.row_count() {
        partial_eliminate_row(ring, matrix, i, global_pivot_i, global_pivot_i + local_pivot_i, global_pivot_j);

        if EXTENSIVE_RUNTIME_ASSERTS {
            for j in 0..local_pivot_j {
                assert!(matrix.at(i, global_pivot_j * n + j).is_none());
            }
        }

        matrix.check();
        if matrix.at(i, global_pivot_j * n + local_pivot_j).is_some() {
            matrix.rows.swap(i, global_pivot_i + local_pivot_i);
            return true;
        }
    }
    matrix.check();
    return false;
}

///
/// Note that this considers the whole matrix `block.matrix` to find a pivoting row;
/// Apart from that, it only operates within `block``
/// 
fn local_row_echelon<R>(ring: R, matrix: &mut Matrix<El<R>>, transform_block: &mut MatrixBlock<El<R>>, global_pivot_i: usize, global_pivot_j: usize, start_pivot: (usize, usize)) -> Result<usize, (usize, usize)>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check();
    let n = matrix.n;
    let mut block = matrix.block(global_pivot_i..(global_pivot_i + n), global_pivot_j);
    let mut i = start_pivot.0;  
    for j in start_pivot.1..n {
        if search_pivot_in_block(ring, &mut block.matrix, i, j, global_pivot_i, global_pivot_j) {

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

            let pivot_inv = ring.checked_div(&ring.one(), block.at(i, j).unwrap()).unwrap();
            assert!(leading_entry(&block.matrix, i + global_pivot_i, global_pivot_j).0 == j);

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
                    assert!(elim_i < i || leading_entry(&block.matrix, elim_i + global_pivot_i, global_pivot_j).0 == j);
                    let factor = ring.clone_el(factor);
                    *block.row_mut(elim_i) = sub_row_local(ring, block.row(elim_i), block.row(i), &factor, Vec::new());
                    *transform_block.row_mut(elim_i) = sub_row_local(ring, transform_block.row(elim_i), transform_block.row(i), &factor, Vec::new());
                    block.matrix.check();
                }
            }
            block.matrix.check();
            i += 1;

        } else {
            return Err((i, j));
        }

    }
    return Ok(i);
}

fn blocked_row_echelon<R, const LOG: bool>(ring: R, matrix: &mut Matrix<El<R>>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
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
    
    let mut local_pivot = (0, 0);
    while pivot_row + n < matrix.row_count() && pivot_col < col_block_count {
        let mut transform_transposed = identity(ring, n);
        let mut transform_transposed_block = transform_transposed.block(0..n, 0);

        // now we have the nxn block in row echelon form, with the last (n - produced rows) being zero
        let current = local_row_echelon(ring, matrix, &mut transform_transposed_block, pivot_row, pivot_col, local_pivot);

        // we have to apply the transformation to the other blocks in the same rows
        let mut tmp = Matrix { rows: Vec::new(), n: n };
        for col in (pivot_col + 1)..col_block_count {
            let new = mul_assign(ring, &mut transform_transposed_block, &mut matrix.block(pivot_row..(pivot_row + n), col), tmp);
            tmp = set_block(&mut matrix.block(pivot_row..(pivot_row + n), col), new);
            check_no_zeros(ring, matrix);
        }
        
        match current {
            Ok(produced_rows) => {
                // and then we can eliminate the column, using matrix blocks
                for row in ((pivot_row + n)..(matrix.row_count() - n)).step_by(n) {
                    if (row..(row + n)).any(|i| matrix.rows[i][pivot_col].len() > 0) {
                        complete_eliminate_rowblock(ring, matrix, row, pivot_row, pivot_col);
                    }
                }

                pivot_col += 1;
                pivot_row += produced_rows;
                local_pivot = (0, 0);

                if LOG {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
            },
            Err((local_i, local_j)) => {
                if search_eliminate_pivot(ring, matrix, local_i, local_j, pivot_row, pivot_col) {
                    local_pivot = (local_i, local_j);
                } else {
                    local_pivot = (local_i, local_j + 1);
                }
            }
        }
    }

    // remove the padding
    for _ in 0..n {
        matrix.rows.pop();
    }
}

pub fn gb_sparse_row_echelon<F, const LOG: bool>(ring: F, rows: Vec<Vec<(usize, El<F>)>>, col_count: usize) -> Vec<Vec<(usize, El<F>)>>
    where F: DivisibilityRingStore + Copy,
        F::Type: DivisibilityRing
{
    let n = 1;
    let global_cols = (col_count - 1) / n + 1;
    let mut matrix = Matrix {
        n: n,
        rows: rows.into_iter().map(|row| {
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
    return matrix.rows.into_iter().map(|row| 
        row.into_iter().enumerate().flat_map(|(i, r)| r.into_iter().rev().skip(1).rev().map(move |(j, c)| (j + i * n, c))).collect()
    ).collect();
}

#[test]
fn test() {
    let ring = Zn::<7>::RING;
    let mut counter1 = 0;
    let mut counter2 = 0;
    let mut matrix = Matrix {
        rows: (0..10).map(|i| (0..3).map(|j| (0..4).map(|k| { counter1 += 1; (k, (counter1 % 6) + 1) }).filter(|_| { counter2 = (counter2 + 1) % 7; counter2 < 2 }).chain(Some((usize::MAX, 0)).into_iter()).collect()).collect()).collect(),
        n: 4
    };

    blocked_row_echelon::<_, false>(ring, &mut matrix);

}