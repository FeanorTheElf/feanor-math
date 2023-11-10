use std::io::Write;
use std::mem::swap;
use std::cmp::{min, Ordering};
use std::sync::atomic::AtomicUsize;

use crate::parallel::{potential_parallel_for_each, column_iterator};
use crate::ring::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::vector::VectorView;

const EXTENSIVE_RUNTIME_ASSERTS: bool = false;

struct Matrix<T> {
    rows: Vec<Vec<Vec<(usize, T)>>>,
    global_col_count: usize,
    n: usize
}

impl<T> Matrix<T> {

    fn entry_at<'a>(&'a self, i: usize, j_global: usize, j_local: usize) -> Option<&'a T> {
        at(j_local, &self.rows[i][j_global])
    }

    fn row_count(&self) -> usize {
        self.rows.len()
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

fn print<R>(ring: R, matrix: &Matrix<El<R>>)
    where R: RingStore
{
    let zero = ring.zero();
    for i in 0..matrix.row_count() {
        for j_global in 0..matrix.global_col_count {
            for j_local in 0..matrix.n {
                let string = format!("{}", ring.format(matrix.entry_at(i, j_global, j_local).unwrap_or(&zero)));
                print!("{}{}, ", " ".repeat(2 - string.len()), string);
            }
        }
        println!();
    }
}

fn empty<T>(n: usize, global_col_count: usize) -> Matrix<T> {
    Matrix { n: n, global_col_count: global_col_count, rows: Vec::new() }
}

fn at<'a, T>(i: usize, data: &'a [(usize, T)]) -> Option<&'a T> {
    data.binary_search_by_key(&i, |(j, _)| *j).ok().map(|idx| &data[idx].1)
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

fn leading_entry<'a, T>(matrix: &'a Matrix<T>, row: usize, global_col: usize) -> (usize, &'a T) {
    let (j, c) = &matrix.rows[row][global_col][0];
    return (*j, c);
}

#[inline(never)]
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

#[inline(never)]
fn search_pivot_outside_block<T>(matrix: &mut Matrix<T>, local_pivot_i: usize, local_pivot_j: usize, global_pivot_i: usize, global_pivot_j: usize) -> bool {
    let n = matrix.n;
    matrix.check();
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
    matrix.check();
    return false;
}

static SHORT_REDUCTION_ROUND: AtomicUsize = AtomicUsize::new(0);
static LONG_REDUCTION_ROUND: AtomicUsize = AtomicUsize::new(0);
static TRANSFORM_ROUND: AtomicUsize = AtomicUsize::new(0);
static TRANSFORM_TIME: AtomicUsize = AtomicUsize::new(0);
static SHORT_REDUCTION_TIME: AtomicUsize = AtomicUsize::new(0);
static LONG_REDUCTION_TIME: AtomicUsize = AtomicUsize::new(0);

#[inline(never)]
fn update_rows_with_transform<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, pivot_col: usize, transform: &[Vec<(usize, El<R>)>]) 
    where R: RingStore + Copy + Sync,
        El<R>: Send + Sync
{
    let start = std::time::Instant::now();
    potential_parallel_for_each(
        column_iterator(&mut matrix.rows[rows_start..], (pivot_col + 1)..(matrix.global_col_count)), 
        || Vec::new(), 
        |tmp, rows| 
    {
        let rows = rows.collect::<Vec<_>>();
        let mut new = mul_assign(
            ring, 
            transform, 
            rows.iter().map(|x| &x[..]), 
            std::mem::replace(tmp, Vec::new())
        );
        for (target, new) in rows.into_iter().zip(new.iter_mut()) {
            swap(target, new);
        }
        *tmp = new;
    });
    let end = std::time::Instant::now();
    TRANSFORM_ROUND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TRANSFORM_TIME.fetch_add((end - start).as_millis() as usize, std::sync::atomic::Ordering::Relaxed);
}

fn eliminate_exterior_rows<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, rows_end: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy,
        El<R>: Send + Sync,
        R: Sync
{
    eliminate_interior_rows(ring, matrix, rows_start, rows_end, pivot_rows_start, pivot_rows_end, global_col)
}

#[inline(never)]
fn eliminate_interior_rows<R>(ring: R, matrix: &mut Matrix<El<R>>, rows_start: usize, rows_end: usize, pivot_rows_start: usize, pivot_rows_end: usize, global_col: usize)
    where R: RingStore + Copy,
        El<R>: Send + Sync,
        R: Sync
{
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

    let start = std::time::Instant::now();
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
    let end = std::time::Instant::now();
    if (end - start).as_millis() < 10 {
        SHORT_REDUCTION_ROUND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SHORT_REDUCTION_TIME.fetch_add((end - start).as_millis() as usize, std::sync::atomic::Ordering::Relaxed);
    } else {
        LONG_REDUCTION_ROUND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LONG_REDUCTION_TIME.fetch_add((end - start).as_millis() as usize, std::sync::atomic::Ordering::Relaxed);
    }
}

fn get_two_mut<'a, T>(slice: &'a mut [T], i1: usize, i2: usize) -> (&'a mut T, &'a mut T) {
    assert!(i1 < i2);
    let (s1, s2) = (&mut slice[i1..=i2]).split_at_mut(1);
    return (&mut s1[0], &mut s2[s2.len() - 1]);
}

#[inline(never)]
fn local_row_echelon<R>(ring: R, matrix: &mut Matrix<El<R>>, transform: &mut Matrix<El<R>>, global_pivot_i: usize, global_pivot_j: usize, start_pivot: (usize, usize)) -> (usize, Result<(), usize>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    matrix.check();
    let n = matrix.n;
    let mut i = start_pivot.0;
    let mut tmp = Vec::new();
    for j in start_pivot.1..n {
        if let Some(new_pivot) = search_pivot_in_block(matrix, i, j, global_pivot_i, global_pivot_j) {

            if new_pivot != i {
                let (r1, r2) = get_two_mut(&mut matrix.rows[..], i + global_pivot_i, new_pivot + global_pivot_i);
                swap(&mut r1[global_pivot_j], &mut r2[global_pivot_j]);
                transform.rows.swap(i, new_pivot);
            }

            // check that the left part remains zero and the pivot is nonzero
            if EXTENSIVE_RUNTIME_ASSERTS {
                assert!(matrix.entry_at(global_pivot_i + i, global_pivot_j, j).is_some());
                for col in 0..j {
                    for row in min(i + 1, col + 1)..n {
                        if !(matrix.entry_at(global_pivot_i + row, global_pivot_j, col).is_none()) {
                            println!();
                            print(ring, &matrix);
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

                    let new = add_row_local::<_, true>(ring, &matrix.rows[global_pivot_i + elim_i][global_pivot_j], &matrix.rows[global_pivot_i + i][global_pivot_j], &lhs_factor, &rhs_factor, tmp);
                    tmp = std::mem::replace(&mut matrix.rows[global_pivot_i + elim_i][global_pivot_j], new);

                    let new = add_row_local::<_, true>(ring, &transform.rows[elim_i][0], &transform.rows[i][0], &lhs_factor, &rhs_factor, tmp);
                    tmp = std::mem::replace(&mut transform.rows[elim_i][0], new);
                }
            }
            i += 1;

        } else {
            matrix.check();
            return (i, Err(j));
        }

    }
    matrix.check();
    return (i, Ok(()));
}

#[inline(never)]
fn blocked_row_echelon<R, const LOG: bool>(ring: R, matrix: &mut Matrix<El<R>>)
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        El<R>: Send + Sync,
        R: Sync
{
    SHORT_REDUCTION_ROUND.store(0, std::sync::atomic::Ordering::SeqCst);
    LONG_REDUCTION_ROUND.store(0, std::sync::atomic::Ordering::SeqCst);
    SHORT_REDUCTION_TIME.store(0, std::sync::atomic::Ordering::SeqCst);
    LONG_REDUCTION_TIME.store(0, std::sync::atomic::Ordering::SeqCst);
    let start = std::time::Instant::now();
    if LOG {
        print!("[{}x{}]", matrix.row_count(), matrix.global_col_count * matrix.n);
        std::io::stdout().flush().unwrap();
    }
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
        let mut transform = identity(ring, n, empty(n, 1));

        // now we have the nxn block in row echelon form, with the last (n - produced rows) being zero
        let (new_local_i, current_result) = local_row_echelon(ring, matrix, &mut transform, pivot_row, pivot_col, (local_pivot_i, local_pivot_j));

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
    println!();
    println!("Statistics");
    println!("  short rounds: {}", SHORT_REDUCTION_ROUND.load(std::sync::atomic::Ordering::SeqCst));
    println!("  long rounds: {}", LONG_REDUCTION_ROUND.load(std::sync::atomic::Ordering::SeqCst));
    println!("  short time: {} ms", SHORT_REDUCTION_TIME.load(std::sync::atomic::Ordering::SeqCst));
    println!("  long time: {} ms", LONG_REDUCTION_TIME.load(std::sync::atomic::Ordering::SeqCst));
    println!("  transform rounds: {}", TRANSFORM_ROUND.load(std::sync::atomic::Ordering::SeqCst));
    println!("  transform time: {} ms", TRANSFORM_TIME.load(std::sync::atomic::Ordering::SeqCst));
}

#[inline(never)]
pub fn gb_sparse_row_echelon<R, const LOG: bool>(ring: R, rows: Vec<Vec<(usize, El<R>)>>, col_count: usize) -> Vec<Vec<(usize, El<R>)>>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing,
        El<R>: Send + Sync,
        R: Sync
{
    let n = 256;
    let global_cols = (col_count - 1) / n + 1;
    let mut matrix = Matrix {
        n: n,
        global_col_count: global_cols,
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
        n: 4,
        global_col_count: 3
    };

    blocked_row_echelon::<_, false>(ring, &mut matrix);

    print(ring, &matrix);
}