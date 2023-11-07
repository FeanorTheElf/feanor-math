use std::{ops::Range, cmp::Ordering};

use crate::{ring::*, divisibility::{DivisibilityRingStore, DivisibilityRing}, vector::VectorView};

use super::zn::zn_static::Zn;

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
            print!("{}, ", ring.format(matrix.at(i, j).unwrap_or(&zero)));
        }
        println!();
    }
}

fn identity<R>(ring: R, n: usize) -> Matrix<El<R>>
    where R: RingStore
{
    Matrix { n: n, rows: (0..n).map(|i| vec![vec![(i, ring.one())]]).collect() }
}

fn sub_row_local<R>(ring: R, matrix: &mut MatrixBlock<El<R>>, i_dst: usize, i_src: usize, factor: &El<R>)
    where R: RingStore
{
    let mut out = Vec::new();
    let mut dst_idx = 0;
    let mut src_idx = 0;
    matrix.row_mut(i_dst).push((usize::MAX, ring.zero()));
    matrix.row_mut(i_src).push((usize::MAX, ring.zero()));
    while dst_idx + 1 < matrix.row_mut(i_dst).len() || src_idx + 1 < matrix.row_mut(i_src).len() {
        let dst_j = matrix.row_mut(i_dst)[dst_idx].0;
        let src_j = matrix.row_mut(i_src)[src_idx].0;
        
        match dst_j.cmp(&src_j) {
            Ordering::Less => {
                out.push((dst_j, ring.clone_el(&matrix.row_mut(i_dst)[dst_idx].1)));
                dst_idx += 1;
            },
            Ordering::Greater => {
                out.push((src_j, ring.negate(ring.mul_ref(&matrix.row_mut(i_src)[src_idx].1, factor))));
                src_idx += 1;
            },
            Ordering::Equal => {
                let subtract = ring.mul_ref(&matrix.row_mut(i_src)[src_idx].1, factor);
                let value = ring.sub_ref_fst(&matrix.row_mut(i_dst)[dst_idx].1, subtract);
                if !ring.is_zero(&value) {
                    out.push((dst_j, value));
                }
                dst_idx += 1;
                src_idx += 1;
            }
        }
    }
    *matrix.row_mut(i_dst) = out;
    // remove the blocker entry (usize::MAX, _)
    matrix.row_mut(i_src).pop();
    check_no_zeros(ring, &matrix.matrix);
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
            let value = ring.sum(lhs.row(i).iter()
                .filter_map(|(k, a)| rhs.at(*k, j).map(|b| (a, b)))
                .map(|(a, b)| ring.mul_ref(a, b)));
            if !ring.is_zero(&value) {
                out.rows[i][0].push((j, value));
            }
        }
    }
    check_no_zeros(ring, &out);
    return out;
}

fn check_no_zeros<R>(ring: R, matrix: &Matrix<El<R>>)
    where R: RingStore
{
    for i in 0..matrix.row_count() {
        for j in 0..(matrix.col_count() / matrix.n) {
            debug_assert!(matrix.rows[i][j].iter().all(|(_, c)| !ring.is_zero(c)));
        }
    }
}

fn get_two_mut<'a, T>(slice: &'a mut [T], i1: usize, i2: usize) -> (&'a mut T, &'a mut T) {
    assert!(i1 < i2);
    let (fst, snd) = slice.split_at_mut(i1 + 1);
    return (&mut fst[i1], &mut snd[i2 - i1 - 1]);
}

fn sub_assign_product<R>(ring: R, main: &mut Matrix<El<R>>, out_block: Block, lhs_block: Block, rhs_block: Block)
    where R: RingStore
{
    assert!(out_block.global_col != lhs_block.global_col || out_block.row_end <= lhs_block.row_start || out_block.row_start >= lhs_block.row_end);
    assert!(out_block.global_col != rhs_block.global_col || out_block.row_end <= rhs_block.row_start || out_block.row_start >= rhs_block.row_end);
    assert!(rhs_block.global_col != lhs_block.global_col || rhs_block.row_end <= lhs_block.row_start || rhs_block.row_start >= lhs_block.row_end);
    
    let n = main.n;

    let mut new = Vec::new();
    for i in 0..n {
        new.clear();
        // this prevents an out-of-bounds without explicit guards
        main.rows[out_block.row_start + i][out_block.global_col].push((usize::MAX, ring.zero()));

        let mut out_idx = 0;
        for j in 0..n {
            let value = ring.sum(main.rows[lhs_block.row_start + i][lhs_block.global_col].iter()
                .filter_map(|(k, c)| 
                    main.rows[*k + rhs_block.row_start][rhs_block.global_col].binary_search_by_key(&j, |(j, _)| *j).ok().map(|b| (c, &main.rows[*k + rhs_block.row_start][rhs_block.global_col][b].1))
                )
                .map(|(b, c)| ring.mul_ref(b, c))
            );
            let mut out_block = main.block(out_block.row_start..out_block.row_end, out_block.global_col);
            if !ring.is_zero(&value) {
                if out_block.row_mut(i)[out_idx].0 == j {
                    let new_value = ring.sub_ref_snd(value, &out_block.row_mut(i)[out_idx].1);
                    if !ring.is_zero(&new_value) {
                        new.push((j, new_value));
                    }
                    out_idx += 1;
                } else {
                    new.push((j, value));
                }
            } else if out_block.row_mut(i)[out_idx].0 == j {
                new.push((j, ring.clone_el(&out_block.row_mut(i)[out_idx].1)));
                out_idx += 1;
            }
        }
        std::mem::swap(&mut main.rows[out_block.row_start + i][out_block.global_col], &mut new);
    }
    check_no_zeros(ring, main);
}

fn blocked_row_echelon<R>(ring: R, matrix: &mut Matrix<El<R>>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    let mut pivot_row = 0;
    let mut pivot_col = 0;
    let n = matrix.n;
    let col_block_count = matrix.col_count() / n;

    // we have to pad matrix with n zero rows...
    for _ in 0..n {
        matrix.rows.push((0..col_block_count).map(|_| Vec::new()).collect());
    }
    
    while pivot_row + n < matrix.row_count() && pivot_col < col_block_count {
        let mut transform_transposed = identity(ring, n);
        let mut transform_transposed_block = transform_transposed.block(0..n, 0);

        // we prepare the current block

        let mut i = 0;  
        for j in 0..n {
            let mut new_pivot_row = None;
            for potential_pivot_row in (pivot_row + i)..matrix.row_count() {
                if matrix.at(potential_pivot_row, pivot_col * n + j).is_some() {
                    new_pivot_row = Some(potential_pivot_row);
                    break;
                }
            }
            if let Some(row) = new_pivot_row {
                matrix.rows.swap(row, pivot_row + i);
            } else {
                continue;
            }

            // now the pivot should be nonzero, so eliminate within the block

            let mut block = matrix.block(pivot_row..(pivot_row + n), pivot_col);
            let pivot_inv = ring.checked_div(&ring.one(), block.at(i, j).unwrap()).unwrap();

            for (_, c) in block.row_mut(i) {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            for (_, c) in transform_transposed_block.row_mut(i) {
                ring.mul_assign_ref(c, &pivot_inv);
            }
            for elim_i in 0..n {
                if elim_i == i {
                    continue;
                }
                if let Some(factor) = block.at(elim_i, j) {
                    let factor = ring.clone_el(factor);
                    sub_row_local(ring, &mut block, elim_i, i, &factor);
                    sub_row_local(ring, &mut transform_transposed_block, elim_i, i, &factor);
                }
            }
            i += 1;
        }
        let produced_rows = i;

        let mut tmp = Matrix { rows: Vec::new(), n: n };
        for col in (pivot_col + 1)..col_block_count {
            let new = mul_assign(ring, &mut transform_transposed_block, &mut matrix.block(pivot_row..(pivot_row + n), col), tmp);
            tmp = set_block(&mut matrix.block(pivot_row..(pivot_row + n), col), new);
            check_no_zeros(ring, matrix);
        }

        for row in ((pivot_row + n)..(matrix.row_count() - n)).step_by(n) {
            if (row..(row + n)).any(|i| matrix.rows[i][pivot_col].len() > 0) {
                let mut out_block = Block { global_col: 0, row_start: row, row_end: row + n };
                let lhs_block = Block { global_col: pivot_col, row_start: row, row_end: row + n };
                let mut rhs_block = Block { global_col: 0, row_start: pivot_row, row_end: pivot_row + n };
                for col in (pivot_col + 1)..col_block_count {
                    out_block.global_col = col;
                    rhs_block.global_col = col;
                    sub_assign_product(ring, matrix, out_block, lhs_block, rhs_block);
                }
                for i in 0..n {
                    matrix.rows[row + i][pivot_col].clear();
                }
            }
        }

        pivot_col += 1;
        pivot_row += produced_rows;
    }

    // remove the padding
    for _ in 0..n {
        matrix.rows.pop();
    }
}

#[test]
fn test() {
    let ring = Zn::<7>::RING;
    let mut counter1 = 0;
    let mut counter2 = 0;
    let mut matrix = Matrix {
        rows: (0..10).map(|i| (0..3).map(|j| (0..4).map(|k| { counter1 += 1; (k, (counter1 % 6) + 1) }).filter(|_| { counter2 = (counter2 + 1) % 7; counter2 < 2 }).collect()).collect()).collect(),
        n: 4
    };
    print(ring, &matrix);

    blocked_row_echelon(ring, &mut matrix);

    println!();
    print(ring, &matrix);
}