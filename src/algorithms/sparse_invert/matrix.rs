
use std::ops::Range;

use crate::parallel::Column;

use super::*;

pub struct InternalMatrix<T> {
    rows: Vec<Vec<InternalRow<T>>>,
    global_col_count: usize,
    n: usize,
    zero: T
}

pub struct InternalMatrixRef<'a, T> {
    rows: &'a mut [Vec<InternalRow<T>>],
    global_cols: Range<usize>
}

pub struct InternalRow<T> {
    data: Vec<(usize, T)>
}

impl<T> InternalRow<T> {

    pub fn at<'a>(&'a self, local_j: usize) -> Option<&'a T> {
        self.data.binary_search_by_key(&local_j, |(j, _)| *j).ok().map(|idx| &self.data[idx].1)
    }

    pub const fn placeholder() -> InternalRow<T> {
        InternalRow { data: Vec::new() }
    }

    pub fn make_empty<R>(mut self, ring: &R) -> InternalRow<T> 
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.clear();
        self.data.push((usize::MAX, ring.zero()));
        return self;
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

    pub fn make_multiple<R>(mut self, ring: &R, factor: &El<R>, other: &InternalRow<T>) -> InternalRow<T>
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.clear();
        self.data.extend(other.data.iter()
            .map(|(j, c)| (*j, ring.mul_ref(c, factor)))
            .filter(|(_, c)| !ring.is_zero(c))
            .chain([(usize::MAX, ring.zero())].into_iter()));
        return self;
    }
}

impl<T> InternalMatrix<T> {

    pub fn from_builder<R: ?Sized>(builder: SparseMatrixBuilder<R>, n: usize, ring: &R) -> InternalMatrix<R::Element>
        where R: RingBase<Element = T>
    {
        let mut inverted_permutation = (0..builder.col_permutation.len()).collect::<Vec<_>>();
        for (i, j) in builder.col_permutation.iter().enumerate() {
            inverted_permutation[*j] = i;
        }
        for i in 0..builder.col_permutation.len() {
            debug_assert!(inverted_permutation[builder.col_permutation[i]] == i);
            debug_assert!(builder.col_permutation[inverted_permutation[i]] == i);
        }
        let global_cols = (builder.col_count - 1) / n + 1;
        InternalMatrix {
            global_col_count: global_cols,
            n: n,
            zero: builder.zero,
            rows: builder.rows.into_iter().map(|row| {
                let mut cols = (0..global_cols).map(|_| InternalRow::placeholder()).collect::<Vec<_>>();
                for (j, c) in row.into_iter() {
                    if !ring.is_zero(&c) {
                        let col = inverted_permutation[j];
                        cols[col / n].data.push((col % n, c));
                    }
                }
                for i in 0..global_cols {
                    cols[i].data.sort_by_key(|(j, _)| *j);
                    cols[i].data.push((usize::MAX, ring.zero()));
                }
                cols
            }).collect()
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn local_row<'a>(&'a self, i: usize, global_j: usize) -> &'a InternalRow<T> {
        &self.rows[i][global_j]
    }

    pub fn local_row_mut<'a>(&'a mut self, i: usize, global_j: usize) -> &'a mut InternalRow<T> {
        &mut self.rows[i][global_j]
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn swap_rows(&mut self, fst: usize, snd: usize) {
        self.rows.swap(fst, snd)
    }

    pub fn check<R>(&self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.rows[i].len() {
                    self.local_row(i, j).check(ring);
                }
            }
        }
    }

    pub const fn empty(n: usize, global_col_count: usize, zero: T) -> InternalMatrix<T> {
        InternalMatrix { n: n, global_col_count: global_col_count, rows: Vec::new(), zero: zero }
    }
    
    pub fn make_identity<R>(mut self, ring: R, n: usize) -> InternalMatrix<El<R>>
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        while self.rows.len() < n {
            self.rows.push(Vec::new());
        }
        self.rows.truncate(n);
        for i in 0..n {
            self.rows[i].resize_with(1, || InternalRow::placeholder());
            self.rows[i][0].data.clear();
            self.rows[i][0].data.extend([(i, ring.one()), (usize::MAX, ring.zero())].into_iter());
        }
        self.check(&ring);
        return self;
    }

    pub fn global_col_count(&self) -> usize {
        self.global_col_count
    }

    pub fn one_block<'a>(&'a mut self, rows: Range<usize>, global_cols: Range<usize>) -> InternalMatrixRef<'a, T> {
        InternalMatrixRef { rows: &mut self.rows[rows], global_cols: global_cols }
    }

    pub fn two_blocks<'a>(&'a mut self, fst: Range<usize>, snd: Range<usize>, global_cols: Range<usize>) -> (InternalMatrixRef<'a, T>, InternalMatrixRef<'a, T>) {
        let (fst_rows, snd_rows) = if fst.start >= snd.end {
            let (snd_rows, fst_rows) = (&mut self.rows[snd.start..fst.end]).split_at_mut(fst.start - snd.start);
            (fst_rows, &mut snd_rows[..(snd.end - snd.start)])
        } else {
            assert!(fst.end <= snd.start);
            let (fst_rows, snd_rows) = (&mut self.rows[fst.start..snd.end]).split_at_mut(snd.end - fst.start);
            (&mut fst_rows[..(fst.end - fst.start)], snd_rows)
        };
        (
            InternalMatrixRef { rows: fst_rows, global_cols: global_cols.clone() },
            InternalMatrixRef { rows: snd_rows, global_cols: global_cols.clone() }
        )
    }

    pub fn add_row<R>(&mut self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.rows.push((0..self.global_col_count).map(|_| InternalRow { data: vec![(usize::MAX, ring.zero())] }).collect())
    }

    pub fn pop_row(&mut self) {
        self.rows.pop();
    }

    pub fn destruct<R>(self, ring: &R) -> Vec<Vec<(usize, T)>> 
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        let n = self.n();
        self.rows.into_iter().map(|row| 
            row.into_iter().enumerate().flat_map(|(i, r)| r.data.into_iter().rev().skip(1).rev().map(move |(j, c)| (j + i * n, c)).inspect(|(_, c)| assert!(!ring.is_zero(c)))).collect()
        ).collect()
    }

    pub fn get_two_rows<'a>(&'a mut self, fst: usize, snd: usize, global_col: usize) -> (&'a mut InternalRow<T>, &'a mut InternalRow<T>) {
        debug_assert!(snd > fst);
        let (fst_part, snd_part) = self.rows.split_at_mut(snd);
        return (&mut fst_part[fst][global_col], &mut snd_part[0][global_col]);
    }
}

impl<'b, T> InternalMatrixRef<'b, T> {

    pub fn local_row<'a>(&'a self, i: usize, global_j: usize) -> &'a InternalRow<T> {
        &self.rows[i][global_j]
    }

    #[cfg(not(feature = "parallel"))]
    pub fn col_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = Column<'a, InternalRow<T>>>
        where T: Send
    {
        column_iterator(&mut *self.rows, self.global_cols.clone())
    }

    #[cfg(feature = "parallel")]
    pub fn col_iter_mut<'a>(&'a mut self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = Column<'a, InternalRow<T>>>
        where T: Send
    {
        column_iterator(&mut *self.rows, self.global_cols.clone())
    }

    pub fn row_iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a [InternalRow<T>]> {
        let cols = self.global_cols.clone();
        (&*self.rows).into_iter().map(move |row: &'a Vec<InternalRow<T>>| &row[cols.clone()])
    }

    #[cfg(not(feature = "parallel"))]
    pub fn row_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = &'a mut [InternalRow<T>]>
        where T: Send
    {
        let cols = self.global_cols.clone();
        (&mut *self.rows).into_iter().map(move |row: &'a mut Vec<InternalRow<T>>| &mut row[cols.clone()])
    }

    #[cfg(feature = "parallel")]
    pub fn row_iter_mut<'a>(&'a mut self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut [InternalRow<T>]>
        where T: Send
    {
        let cols = self.global_cols.clone();
        <_ as rayon::iter::IntoParallelIterator>::into_par_iter(&mut *self.rows).map(move |row| &mut row[cols.clone()])
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
        self.local_row(i, j / self.n).at(j % self.n).unwrap_or(&self.zero)
    }
}

#[inline(never)]
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
    assert!(lhs_idx + 1 == lhs.data.len() && rhs_idx + 1 == rhs.data.len());
    out.data.push((usize::MAX, ring.zero()));
    out.check(&ring);
    return out;
}

#[inline(never)]
pub fn linear_combine_rows<'a, R, I>(ring: R, coeffs: &InternalRow<El<R>>, mut rows: I, mut out: InternalRow<El<R>>, tmp: &mut InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore + Copy,
        I: Iterator<Item = &'a InternalRow<El<R>>>,
        El<R>: 'a
{
    coeffs.check(&ring);
    out = out.make_empty(&ring);
    if coeffs.is_empty() {
        return out;
    }
    let mut last_idx = coeffs.leading_entry().0;
    rows.advance_by(last_idx).unwrap();
    out = out.make_multiple(&ring, coeffs.leading_entry().1, rows.next().unwrap());
    *tmp = std::mem::replace(tmp, InternalRow::placeholder()).make_empty(&ring);
    let lhs_factor = ring.one();
    for (idx, c) in coeffs.data[1..(coeffs.data.len() - 1)].iter() {
        rows.advance_by(*idx - last_idx - 1).unwrap();
        last_idx = *idx;
        *tmp = add_row_local::<_, true>(ring, &out, rows.next().unwrap(), &lhs_factor, c, std::mem::replace(tmp, InternalRow::placeholder()));
        swap(&mut out, tmp);
    }
    out.check(&ring);
    return out;
}
    
///
/// Computes a vector `result` such that `result * echelon_form_matrix = current_row`, if it exists.
/// Note that if the vector does not exist, this function will not throw an error but return a
/// vector that can be thought of as a "best effort solution". This is important in combination
/// with the optimistic elimination strategy.
/// 
#[inline(never)]
pub fn preimage_echelon_form_matrix<R>(ring: R, echelon_form_matrix: &InternalMatrixRef<El<R>>, current_row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    current_row.check(&ring);
    let zero = ring.zero();
    out = out.make_empty(&ring);
    out.data.pop();
    for (i, row) in echelon_form_matrix.row_iter().enumerate() {
        let (j, pivot) = &row[0].leading_entry();
        let mut current = ring.clone_el(current_row.at(*j).unwrap_or(&zero));
        for (k, c) in out.data.iter().rev().skip(1).rev() {
            ring.sub_assign(&mut current, ring.mul_ref(c, echelon_form_matrix.local_row(*k, 0).at(*j).unwrap_or(&zero)));
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