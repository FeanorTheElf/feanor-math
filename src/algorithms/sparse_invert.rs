use crate::ring::*;
use crate::field::*;
use crate::vector::*;

pub struct SparseMatrix<F: FieldStore>
    where F::Type: Field
{
    field: F,
    zero: El<F>,

    ///
    /// Stores the nontrivial entries of each row, in global coordinates.
    /// This always contains the whole matrix, even if we are currently
    /// considering only a submatrix.
    /// 
    rows: Vec<Vec<(usize, El<F>)>>,
    ///
    /// The order of the columns. In other words, the first column is the one
    /// with global index `col_permutation[0]` and so on.
    /// This always refers to the whole matrix, even if we are currently
    /// considering only a submatrix.
    /// 
    col_permutation: Vec<usize>,
    ///
    /// Column count of the whole matrix
    /// 
    col_count: usize,
    ///
    /// Inverse permutation of `col_permutation`
    /// 
    col_permutation_inv: Vec<usize>,
}

pub struct MatrixRowIter<'a, T> {
    current: std::slice::Iter<'a, (usize, T)>,
    permutation_inv: &'a [usize]
}

impl<'a, T> Iterator for MatrixRowIter<'a, T> {

    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        // since we only allow taking submatrices where the left area is zero,
        // there is no need of filtering elements here
        let (index, el) = self.current.next()?;
        return Some((self.permutation_inv[*index], el));
    }
}

pub struct MatrixRow<'a, T> {
    data: &'a [(usize, T)],
    len: usize,
    permutation: &'a [usize],
    permutation_inv: &'a [usize],
    zero: &'a T
}

impl<'a, T> VectorView<T> for MatrixRow<'a, T> {

    fn len(&self) -> usize {
        self.len
    }
    
    fn at(&self, i: usize) -> &T {
        self.data.binary_search_by_key(&self.permutation[i], |(index, _)| *index).map(|index| &self.data[index].1).unwrap_or(&self.zero)
    }
}

impl<'a, T> VectorViewSparse<T> for MatrixRow<'a, T> {

    type Iter<'b> = MatrixRowIter<'b, T>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        MatrixRowIter {
            current: self.data.iter(),
            permutation_inv: self.permutation_inv
        }
    }
}

pub struct MatrixColIter<'a, T> {
    curent: std::slice::Iter<'a, usize>,
    col_index: usize,
    row_data: &'a [Vec<(usize, T)>],
    len: usize
}

impl<'a, T> Iterator for MatrixColIter<'a, T> {

    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut row_index = *self.curent.next()?;
        while row_index >= self.len {
            row_index = *self.curent.next()?;
        }
        let entry = self.row_data[row_index].binary_search_by_key(&self.col_index, |(j, _)| *j).map(|index| &self.row_data[row_index][index].1).unwrap();
        return Some((row_index, entry));
    }
}

pub struct MatrixCol<'a, T> {
    row_data: &'a [Vec<(usize, T)>],
    col: &'a [usize],
    col_index: usize,
    zero: &'a T,
    len: usize
}

impl<'a, T> VectorView<T> for MatrixCol<'a, T> {

    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, i: usize) -> &T {
        self.row_data[i].binary_search_by_key(&self.col_index, |(j, _)| *j).map(|index| &self.row_data[i][index].1).unwrap_or(self.zero)
    }
}

impl<'a, T> VectorViewSparse<T> for MatrixCol<'a, T> {

    type Iter<'b> = MatrixColIter<'b, T>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        MatrixColIter {
            col_index: self.col_index,
            curent: self.col.iter(),
            row_data: self.row_data,
            len: self.len
        }
    }
}

impl<F: FieldStore> Clone for SparseMatrix<F>
    where F::Type: Field,
        F: Clone
{
    fn clone(&self) -> Self {
        Self {
            col_count: self.col_count,
            col_permutation: self.col_permutation.clone(),
            col_permutation_inv: self.col_permutation_inv.clone(),
            field: self.field.clone(),
            zero: self.field.zero(),
            rows: self.rows.iter().map(|data| data.iter().map(|(j, c)| (*j, self.field.clone_el(c))).collect()).collect()
        }
    }
}

impl<F: FieldStore> SparseMatrix<F>
    where F::Type: Field
{
    pub fn new<I>(field: F, row_count: usize, col_count: usize, entries: I) -> Self
        where I: Iterator<Item = (usize, usize, El<F>)>
    {
        let mut rows = Vec::new();
        rows.resize_with(row_count, Vec::new);
        
        let col_permutation = (0..col_count).collect();
        let col_permutation_inv = (0..col_count).collect();

        let mut result = SparseMatrix { 
            col_permutation, 
            col_permutation_inv,
            col_count,
            zero: field.zero(),
            field, 
            rows, 
        };

        for (i, j, e) in entries {
            assert!(!result.field.is_zero(&e));
            result.rows[i].push((j, e));
        }

        for i in 0..row_count {
            result.rows[i].sort_unstable_by_key(|(j, _)| *j);
            for index in 1..result.rows[i].len() {
                assert!(result.rows[i][index - 1].0 != result.rows[i][index].0);
            }
        }

        return result;
    }

    pub fn base_field(&self) -> &F {
        &self.field
    }

    pub fn add_column(&mut self, j: usize) {
        assert!(j <= self.col_count);
        self.col_permutation.insert(j, self.col_count);
        self.col_permutation_inv.push(0);
        for j2 in j..=self.col_count {
            self.col_permutation_inv[self.col_permutation[j2]] = j2;
        }
        self.col_count += 1;
    }

    pub fn reverse_cols(&mut self) {
        self.col_permutation.reverse();
        for j in 0..self.col_count {
            self.col_permutation_inv[j] = self.col_count - 1 - self.col_permutation_inv[j];
        }
    }

    pub fn add_row(&mut self, i: usize) {
        self.rows.insert(i, Vec::new());
    }

    pub fn set(&mut self, i: usize, j: usize, value: El<F>) {
        if !self.field.is_zero(&value) {
            let row = &mut self.rows[i];
            match row.binary_search_by_key(&self.col_permutation[j], |(index, _)| *index) {
                Ok(index) => row[index] = (self.col_permutation[j], value),
                Err(index) => row.insert(index, (self.col_permutation[j], value))
            };
        } else {
            let row = &mut self.rows[i];
            if let Ok(index) = row.binary_search_by_key(&self.col_permutation[j], |(index, _)| *index) {
                row.remove(index);
            }
        }
    }

    pub fn swap_cols(&mut self, j1: usize, j2: usize) {
        assert!(j1 < self.col_count);
        assert!(j2 < self.col_count);
        if j1 == j2 {
            return;
        }
        self.col_permutation.swap(j1, j2);
        self.col_permutation_inv.swap(self.col_permutation[j1], self.col_permutation[j2]);
    }

    pub fn swap_rows(&mut self, i1: usize, i2: usize) {
        assert!(i1 < self.row_count());
        assert!(i2 < self.row_count());
        if i1 == i2 {
            return;
        }
        self.rows.swap(i1, i2);
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn col_count(&self) -> usize {
        self.col_count
    }

    pub fn at(&self, i: usize, j: usize) -> &El<F> {
        assert!(i < self.row_count());
        assert!(j < self.col_count());
        let hard_column = self.col_permutation[j];
        let result = self.rows[i].binary_search_by_key(&hard_column, |(index, _)| *index).map(|index| &self.rows[i][index].1).unwrap_or(&self.zero);
        return result;
    }

    pub fn get_row<'a>(&'a self, i: usize) -> MatrixRow<'a, El<F>> {
        MatrixRow { data: &self.rows[i], len: self.col_count, permutation: &self.col_permutation, permutation_inv: &self.col_permutation_inv, zero: &self.zero }
    }

    pub fn column_permutation<'a>(&'a self) -> impl 'a + Iterator<Item = usize> {
        self.col_permutation.iter().rev().map(|j| self.col_count - 1 - *j)
    }

    pub fn column_permutation_inv<'a>(&'a self) -> impl 'a + Iterator<Item = usize> {
        self.col_permutation_inv.iter().rev().map(|j| self.col_count - 1 - *j)
    }

    fn sub_row<F1>(&mut self, dst_i: usize, src_i: usize, factor: &El<F>, recycle_vec: &mut Option<Vec<(usize, El<F>)>>, mut entry_added_cancelled: F1)
        where F1: FnMut(/* local row */ usize, /* local col */ usize, /* added? */ bool, /* global row */ usize, /* global col */ usize)
    {
        let mut new_row = recycle_vec.take().unwrap_or(Vec::new());
        new_row.clear();
        let mut dst_index = 0;
        let mut src_index = 0;
        let dst = &self.rows[dst_i];
        let src = &self.rows[src_i];
        while dst_index != dst.len() || src_index != src.len() {
            let dst_j = dst.get(dst_index).map(|e| e.0).unwrap_or(usize::MAX);
            let src_j = src.get(src_index).map(|e| e.0).unwrap_or(usize::MAX);

            if dst_j == src_j {
                let new_value = self.field.sub_ref_fst(&dst[dst_index].1, self.field.mul_ref(&src[src_index].1, factor));
                if self.field.is_zero(&new_value) {
                    entry_added_cancelled(dst_i, self.col_permutation_inv[dst_j], false, dst_i, dst_j);
                } else {
                    // no cancellation - this entry remains nonzero
                    new_row.push((dst_j, new_value));
                }
                dst_index += 1;
                src_index += 1;
            } else if dst_j < src_j {
                // we just keep this entry, thus it remains nonzero
                new_row.push((dst_j, self.field.clone_el(&dst[dst_index].1)));
                dst_index += 1;
            } else {
                // we get a new entry, thus we have to update position of nonzero entries
                entry_added_cancelled(dst_i, self.col_permutation_inv[src_j], true, dst_i, src_j);
                new_row.push((src_j, self.field.negate(self.field.mul_ref(&src[src_index].1, factor))));
                src_index += 1;
            }
        }
        *recycle_vec = Some(std::mem::replace(&mut self.rows[dst_i], new_row));
    }

    pub fn drop_zero_rows(&mut self) {
        self.rows.retain(|r| r.len() > 0);
    }
}

pub fn gb_rowrev_sparse_row_echelon<F>(matrix: &mut SparseMatrix<F>) 
    where F: FieldStore + Clone,
        F::Type: Field
{
    let field = matrix.base_field().clone();
    let A = matrix;
    let mut pivot_row = A.row_count() - 1;
    for pivot_col in 0..A.col_count() {
        
        for j in 0..pivot_col {
            for i in 0..=pivot_row {
                debug_assert!(field.is_zero(A.at(i, j)));
            }
        }

        if let Some(pivot_i) = (0..=pivot_row).filter(|i| !field.is_zero(A.at(*i, pivot_col))).min_by_key(|i| A.get_row(*i).nontrivial_entries().count()) {
            
            A.swap_rows(pivot_row, pivot_i);

            let pivot_inv = field.invert(A.at(pivot_row, pivot_col)).unwrap();
        
            for i in 0..A.row_count() {
                if i != pivot_row && !field.is_zero(A.at(i, pivot_col)) {
                    A.sub_row(i, pivot_row, &field.mul_ref(A.at(i, pivot_col), &pivot_inv), &mut None, |_, _, _, _, _| {});
                }
            }

            for i in 0..A.row_count() {
                debug_assert!(i == pivot_row || field.is_zero(A.at(i, pivot_col)));
            }

            if pivot_row == 0 {
                return;
            } else {
                pivot_row -= 1;
            }
        }
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_sub_row() {
    let field = Zn::<17>::RING;
    let mut base = SparseMatrix::new(field, 4, 6, [
        (0, 0, 1), (0, 3, 2), (0, 5, 4), (1, 1, 2), (1, 2, 3), (1, 5, 1), (2, 0, 3), (2, 2, 1), (3, 0, 5), (3, 4, 6)
    ].into_iter());
    // 1     2   4
    //   2 3     1
    // 3   1
    // 5       6

    assert_eq!(3, base.get_row(0).nontrivial_entries().count());
    assert_eq!(3, base.get_row(1).nontrivial_entries().count());
    assert_eq!(2, base.get_row(2).nontrivial_entries().count());
    assert_eq!(2, base.get_row(3).nontrivial_entries().count());

    base.sub_row(2, 0, &3, &mut None, |_, _, _, _, _| {});
    // 1     2   4
    //   2 3     1
    //     1 11  5
    // 5       6

    assert_eq!(3, base.get_row(0).nontrivial_entries().count());
    assert_eq!(3, base.get_row(1).nontrivial_entries().count());
    assert_eq!(3, base.get_row(2).nontrivial_entries().count());
    assert_eq!(2, base.get_row(3).nontrivial_entries().count());
    assert_el_eq!(&field, &1, base.at(2, 2));
    assert_el_eq!(&field, &11, base.at(2, 3));
    assert_el_eq!(&field, &5, base.at(2, 5));
}

#[test]
fn test_move_zero_rows_down() {
    let field = Zn::<17>::RING;
    let mut base = SparseMatrix::new(field, 4, 4, [
        (0, 0, 1), (1, 3, 2), (2, 1, 1), (3, 3, 1)
    ].into_iter());
    // 1 0 0 0
    //       2
    //   1 0 0
    //       1

    gb_rowrev_sparse_row_echelon(&mut base);

    assert_el_eq!(&field, &1, base.at(3, 0));
    assert_el_eq!(&field, &1, base.at(2, 1));
    assert_el_eq!(&field, &1, base.at(1, 3));
    assert_eq!(0, base.get_row(0).nontrivial_entries().count());
}