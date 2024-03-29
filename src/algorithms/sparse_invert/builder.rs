use crate::matrix::Matrix;

use super::*;

pub struct SparseMatrixBuilder<R>
    where R: ?Sized + RingBase
{
    pub(super) zero: R::Element,
    pub(super) rows: Vec<Vec<(usize, R::Element)>>,
    pub(super) col_permutation: Vec<usize>,
    pub(super) col_count: usize
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

    pub(super) fn into_internal_matrix(self, n: usize, ring: &R) -> Vec<InternalRow<R::Element>> {
        let row_count = self.row_count();
        let mut inverted_permutation = (0..self.col_permutation.len()).collect::<Vec<_>>();
        for (i, j) in self.col_permutation.iter().enumerate() {
            inverted_permutation[*j] = i;
        }
        for i in 0..self.col_permutation.len() {
            debug_assert!(inverted_permutation[self.col_permutation[i]] == i);
            debug_assert!(self.col_permutation[inverted_permutation[i]] == i);
        }
        let global_cols = (self.col_count - 1) / n + 1;
        let mut result = (0..(global_cols * (row_count + n))).map(|_| InternalRow::placeholder()).collect::<Vec<_>>();
        for (i, row) in self.rows.into_iter().enumerate() {
            for (j, c) in row.into_iter() {
                if !ring.is_zero(&c) {
                    let col = inverted_permutation[j];
                    result[i * global_cols + col / n].data.push((col % n, c));
                }
            }
            for j in 0..global_cols {
                result[i * global_cols + j].data.sort_by_key(|(j, _)| *j);
                result[i * global_cols + j].data.push((usize::MAX, ring.zero()));
            }
        }
        for i in row_count..(row_count + n) {
            for j in 0..global_cols {
                result[i * global_cols + j].make_zero(&RingRef::new(ring));
            }
        }
        return result;
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

    fn entry_at(&self, i: usize, j: usize) -> &R::Element {
        match self.rows.at(i).binary_search_by_key(&self.col_permutation[j], |(c, _)| *c) {
            Ok(idx) => &self.rows.at(i).at(idx).1,
            Err(_) => &self.zero
        }
    }
}
