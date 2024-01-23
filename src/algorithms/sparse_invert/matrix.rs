
use std::ops::Range;

use crate::matrix::subslice::*;
use crate::vector::vec_fn::{self, IntoVectorFn};

use self::builder::SparseMatrixBuilder;

use super::*;

pub struct InternalMatrix<T> {
    cols: Vec<Vec<InternalRow<T>>>,
    row_count: usize,
    n: usize,
    zero: T
}

pub struct InternalMatrixRef<'a, T> {
    data: SubmatrixMut<'a, Vec<InternalRow<T>>, InternalRow<T>>,
    n: usize,
    zero: &'a T
}

pub struct InternalRow<T> {
    pub data: Vec<(usize, T)>
}

impl<T> InternalRow<T> {

    pub fn at<'a>(&'a self, local_j: usize) -> Option<&'a T> {
        self.data.binary_search_by_key(&local_j, |(j, _)| *j).ok().map(|idx| &self.data[idx].1)
    }

    ///
    /// Creates a new, defined (i.e. no UB) but invalid value for an InternalRow.
    /// This can be used to temporarily fill a variable, or initialize (e.g. via
    /// [`make_zero()`]). This function will never allocate memory.
    /// 
    pub const fn placeholder() -> InternalRow<T> {
        InternalRow { data: Vec::new() }
    }

    pub fn make_zero<R>(mut self, ring: &R) -> InternalRow<T> 
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

    pub fn append_one<R>(&mut self, ring: &R, j: usize)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.data.insert(self.data.len() - 1, (j, ring.one()));
        self.check(ring);
    }
}

impl<T> InternalMatrix<T> {

    pub fn from_builder<R: ?Sized>(builder: SparseMatrixBuilder<R>, n: usize, ring: &R) -> InternalMatrix<R::Element>
        where R: RingBase<Element = T>
    {
        let row_count = builder.row_count();
        let mut inverted_permutation = (0..builder.col_permutation.len()).collect::<Vec<_>>();
        for (i, j) in builder.col_permutation.iter().enumerate() {
            inverted_permutation[*j] = i;
        }
        for i in 0..builder.col_permutation.len() {
            debug_assert!(inverted_permutation[builder.col_permutation[i]] == i);
            debug_assert!(builder.col_permutation[inverted_permutation[i]] == i);
        }
        let global_cols = (builder.col_count - 1) / n + 1;
        let mut cols = (0..global_cols).map(|_| (0..row_count).map(|_| InternalRow::placeholder()).collect::<Vec<_>>()).collect::<Vec<_>>();
        for (i, row) in builder.rows.into_iter().enumerate() {
            for (j, c) in row.into_iter() {
                if !ring.is_zero(&c) {
                    let col = inverted_permutation[j];
                    cols[col / n][i].data.push((col % n, c));
                }
            }
            for j in 0..global_cols {
                cols[j][i].data.sort_by_key(|(j, _)| *j);
                cols[j][i].data.push((usize::MAX, ring.zero()));
            }
        }
        InternalMatrix {
            row_count: row_count,
            n: n,
            zero: builder.zero,
            cols: cols
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn local_row<'a>(&'a self, i: usize, global_j: usize) -> &'a InternalRow<T> {
        &self.cols[global_j][i]
    }

    pub fn local_row_mut<'a>(&'a mut self, i: usize, global_j: usize) -> &'a mut InternalRow<T> {
        &mut self.cols[global_j][i]
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn swap_rows(&mut self, fst: usize, snd: usize) {
        for col in &mut self.cols {
            col.swap(fst, snd)
        }
    }

    pub fn check<R>(&self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.global_col_count() {
                    self.local_row(i, j).check(ring);
                }
            }
        }
    }

    pub const fn empty(zero: T) -> InternalMatrix<T> {
        InternalMatrix { n: 0, row_count: 0, cols: Vec::new(), zero: zero }
    }

    pub fn set_n(&mut self, new_n: usize) {
        self.n = new_n;
    }

    ///
    /// This will clear the matrix
    /// 
    pub fn clear<R>(&mut self, ring: &R, row_count: usize, global_col_count: usize)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        self.cols.clear();
        self.cols.extend((0..global_col_count).map(|_| (0..row_count).map(|_| InternalRow::placeholder()).collect()));
        self.row_count = row_count;
        self.block(0..self.row_count(), 0..self.global_col_count()).make_zero(ring);
    }
    
    pub fn global_col_count(&self) -> usize {
        self.cols.len()
    }

    pub fn block<'a>(&'a mut self, rows: Range<usize>, global_cols: Range<usize>) -> InternalMatrixRef<'a, T> {
        InternalMatrixRef {
            data: SubmatrixMut::new(&mut self.cols).submatrix(global_cols, rows),
            n: self.n,
            zero: &self.zero
        }
    }

    pub fn split_rows<'a>(&'a mut self, fst: Range<usize>, snd: Range<usize>, global_cols: Range<usize>) -> (InternalMatrixRef<'a, T>, InternalMatrixRef<'a, T>) {
        let row_count = self.row_count();
        let (fst, snd) = SubmatrixMut::new(&mut self.cols).submatrix(global_cols, 0..row_count).split_cols(fst, snd);
        return (
            InternalMatrixRef { data: fst, n: self.n, zero: &self.zero },
            InternalMatrixRef { data: snd, n: self.n, zero: &self.zero }
        )
    }

    pub fn split_cols<'a>(&'a mut self, rows: Range<usize>, fst: Range<usize>, snd: Range<usize>) -> (InternalMatrixRef<'a, T>, InternalMatrixRef<'a, T>) {
        let global_col_count = self.global_col_count();
        let (fst, snd) = SubmatrixMut::new(&mut self.cols).submatrix(0..global_col_count, rows).split_cols(fst, snd);
        return (
            InternalMatrixRef { data: fst, n: self.n,zero: &self.zero },
            InternalMatrixRef { data: snd, n: self.n,zero: &self.zero }
        )
    }

    pub fn add_row<R>(&mut self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        for col in &mut self.cols {
            col.push(InternalRow { data: vec![ (usize::MAX, ring.zero()) ] });
        }
        self.row_count += 1;
    }

    pub fn pop_row(&mut self) {
        for col in &mut self.cols {
            col.pop();
        }
        self.row_count -= 1;
    }

    pub fn two_local_rows<'a>(&'a mut self, fst: usize, snd: usize, global_col: usize) -> (&'a mut InternalRow<T>, &'a mut InternalRow<T>) {
        assert!(fst < snd);
        let col = &mut self.cols[global_col];
        let (fst_slice, snd_slice) = (&mut col[fst..=snd]).split_at_mut(snd - fst);
        return (&mut fst_slice[0], &mut snd_slice[0]);
    }

    pub fn destruct<R>(self, _ring: &R) -> Vec<Vec<(usize, T)>> 
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        let mut result = (0..self.row_count()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (j, col) in self.cols.into_iter().enumerate() {
            for (i, local_row) in col.into_iter().enumerate() {
                for (local_j, x) in local_row.data.into_iter().rev().skip(1).rev() {
                    result[i].push((local_j + j * self.n, x));
                }
            }
        }
        return result
    }
}

impl<'b, T> InternalMatrixRef<'b, T> {

    pub fn clone_to_owned<R>(&self, ring: &R) -> InternalMatrix<T>
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        InternalMatrix {
            n: self.n,
            row_count: self.row_count(),
            zero: ring.zero(),
            cols: (0..self.global_col_count()).map(|j| (0..self.row_count()).map(|i| InternalRow { data: self.local_row(i, j).data.iter().map(|(local_j, c)| (*local_j, ring.clone_el(c))).collect() }).collect()).collect()
        }
    }

    pub fn reborrow<'c>(&'c mut self) -> InternalMatrixRef<'c, T> {
        InternalMatrixRef {
            data: self.data.reborrow(),
            n: self.n,
            zero: &self.zero
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn row_count(&self) -> usize {
        self.data.col_count()
    }

    pub fn global_col_count(&self) -> usize {
        self.data.row_count()
    }

    pub fn check<R>(&self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        if EXTENSIVE_RUNTIME_ASSERTS {
            for i in 0..self.row_count() {
                for j in 0..self.global_col_count() {
                    self.local_row(i, j).check(ring);
                }
            }
        }
    }

    pub fn local_row<'a>(&'a self, i: usize, global_j: usize) -> &'a InternalRow<T> {
        &self.data[global_j][i]
    }

    pub fn local_row_mut<'a>(&'a mut self, i: usize, global_j: usize) -> &'a mut InternalRow<T> {
        &mut self.data[global_j][i]
    }

    pub fn row_iter<'a>(&'a self) -> impl 'a + Iterator<Item = impl 'a + vec_fn::VectorFn<&'a InternalRow<T>>> {
        (0..self.data.col_count()).map(move |i| vec_fn::VectorFn::map((0..self.data.row_count()).into_fn(), move |j| &self.data[j][i]))
    }

    pub fn two_local_rows<'a>(&'a mut self, fst: usize, snd: usize, global_col: usize) -> (&'a mut InternalRow<T>, &'a mut InternalRow<T>) {
        assert!(fst < snd);
        let col = &mut self.data[global_col];
        let (fst_slice, snd_slice) = (&mut col[fst..=snd]).split_at_mut(snd - fst);
        return (&mut fst_slice[0], &mut snd_slice[0]);
    }

    #[allow(unused)]
    pub fn row_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = ColumnMut<'a, Vec<InternalRow<T>>, InternalRow<T>>> {
        self.data.col_iter_mut()
    }

    pub fn make_zero<R>(&mut self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        for i in 0..self.row_count() {
            for j in 0..self.global_col_count() {
                *self.local_row_mut(i, j) = std::mem::replace(self.local_row_mut(i, j), InternalRow::placeholder()).make_zero(ring);
            }
        }
    }

    pub fn make_identity<R>(&mut self, ring: &R)
        where R: RingStore,
            R::Type: RingBase<Element = T>
    {
        assert_eq!(self.row_count(), self.n() * self.global_col_count());
        self.make_zero(ring);
        let n = self.n();
        for i in 0..self.row_count() {
            self.local_row_mut(i, i / n).data.insert(0, (i % n, ring.one()));
        }
    }
}

impl<'b, T> InternalMatrixRef<'b, T> 
    where T: Send + Sync
{
    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_row_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = ColumnMut<'a, Vec<InternalRow<T>>, InternalRow<T>>> {
        self.data.concurrent_col_iter_mut()
    }

    #[cfg(feature = "parallel")]
    pub fn concurrent_row_iter_mut<'a>(&'a mut self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColumnMut<'a, Vec<InternalRow<T>>, InternalRow<T>>>
        where T: Send
    {
        self.data.concurrent_col_iter_mut()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn concurrent_col_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = &'a mut [InternalRow<T>]>
        where T: Send
    {
        self.data.concurrent_row_iter_mut()
    }

    #[cfg(feature = "parallel")]
    pub fn concurrent_col_iter_mut<'a>(&'a mut self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut [InternalRow<T>]>
        where T: Send
    {
        self.data.concurrent_row_iter_mut()
    }
}

impl<R> Matrix<R> for InternalMatrix<R::Element>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize {
        self.row_count
    }

    fn col_count(&self) -> usize {
        self.cols.len() * self.n
    }

    fn at(&self, i: usize, j: usize) -> &R::Element {
        self.local_row(i, j / self.n).at(j % self.n).unwrap_or(&self.zero)
    }
}

impl<'a, R> Matrix<R> for InternalMatrixRef<'a, R::Element>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize {
        self.data.col_count()
    }

    fn col_count(&self) -> usize {
        self.data.row_count() * self.n
    }

    fn at(&self, i: usize, j: usize) -> &R::Element {
        self.local_row(i, j / self.n).at(j % self.n).unwrap_or(&self.zero)
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
    assert!(lhs_idx + 1 == lhs.data.len() && rhs_idx + 1 == rhs.data.len());
    out.data.push((usize::MAX, ring.zero()));
    out.check(&ring);
    return out;
}

pub fn linear_combine_rows<'a, R, I>(ring: R, coeffs: &InternalRow<El<R>>, mut rows: I, mut out: InternalRow<El<R>>, tmp: &mut InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: RingStore + Copy,
        I: Iterator<Item = &'a InternalRow<El<R>>>,
        El<R>: 'a
{
    coeffs.check(&ring);
    out = out.make_zero(&ring);
    if coeffs.is_empty() {
        return out;
    }
    let mut last_idx = coeffs.leading_entry().0;
    rows.advance_by(last_idx).unwrap();
    out = out.make_multiple(&ring, coeffs.leading_entry().1, rows.next().unwrap());
    *tmp = std::mem::replace(tmp, InternalRow::placeholder()).make_zero(&ring);
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
pub fn preimage_echelon_form_matrix<R>(ring: R, echelon_form_matrix: &InternalMatrixRef<El<R>>, current_row: &InternalRow<El<R>>, mut out: InternalRow<El<R>>) -> InternalRow<El<R>>
    where R: DivisibilityRingStore + Copy,
        R::Type: DivisibilityRing
{
    current_row.check(&ring);
    let zero = ring.zero();
    out = out.make_zero(&ring);
    out.data.pop();
    for (i, row) in echelon_form_matrix.row_iter().enumerate() {
        let (j, pivot) = vec_fn::VectorFn::at(&row, 0).leading_entry();
        let mut current = ring.clone_el(current_row.at(j).unwrap_or(&zero));
        for (k, c) in out.data.iter().rev().skip(1).rev() {
            ring.sub_assign(&mut current, ring.mul_ref(c, echelon_form_matrix.local_row(*k, 0).at(j).unwrap_or(&zero)));
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