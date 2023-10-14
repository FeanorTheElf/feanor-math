use crate::ring::*;
use crate::field::*;
use crate::vector::*;

pub struct SparseBaseMatrix<F: FieldStore>
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
    /// Row count of the whole matrix
    /// 
    row_count: usize,
    ///
    /// Column count of the whole matrix
    /// 
    col_count: usize,
    ///
    /// Inverse permutation of `col_permutation`
    /// 
    col_permutation_inv: Vec<usize>,
}

pub struct SparseWorkMatrix<'a, F: FieldStore>
    where F::Type: Field
{
    base: &'a mut SparseBaseMatrix<F>,
    ///
    /// For each row of the current matrix, stores the number of
    /// nonzero entries in that row (indexed by global indices).
    /// This does not refer to the whole matrix, even though the
    /// difference is more conceptual, as we currently only allow
    /// shrinking the current matrix by zero columns.
    /// 
    row_nonzero_entry_counts: Vec<usize>,
    ///
    /// For each column of the current matrix, stores the number
    /// of nonzero entries in that column (indexed by global indices).
    /// This does not refer to the whole matrix, in particular this is
    /// usually different from `cols[j].len()`.
    ///  
    col_nonzero_entry_counts: Vec<usize>,
    ///
    /// Row count of the current matrix
    /// 
    row_count: usize,
    ///
    /// Column count of the current matrix
    /// 
    col_count: usize,
    recycle_vec: Option<Vec<(usize, El<F>)>>
}

pub struct WorkMatrixRowIter<'a, T> {
    current: std::slice::Iter<'a, (usize, T)>,
    permutation_inv: &'a [usize],
    len: usize
}

impl<'a, T> Iterator for WorkMatrixRowIter<'a, T> {

    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        // since we only allow taking submatrices where the left area is zero,
        // there is no need of filtering elements here
        let (index, el) = self.current.next()?;
        return Some((self.len - 1 - self.permutation_inv[*index], el));
    }
}

pub struct WorkMatrixRow<'a, T> {
    data: &'a [(usize, T)],
    len: usize,
    permutation: &'a [usize],
    permutation_inv: &'a [usize],
    zero: &'a T
}

impl<'a, T> VectorView<T> for WorkMatrixRow<'a, T> {

    fn len(&self) -> usize {
        self.len
    }
    
    fn at(&self, i: usize) -> &T {
        let global_index = self.permutation[self.len - 1 - i];
        self.data.binary_search_by_key(&global_index, |(index, _)| *index).map(|index| &self.data[index].1).unwrap_or(&self.zero)
    }
}

impl<'a, T> VectorViewSparse<T> for WorkMatrixRow<'a, T> {

    type Iter<'b> = WorkMatrixRowIter<'b, T>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        WorkMatrixRowIter {
            current: self.data.iter(),
            len: self.len,
            permutation_inv: self.permutation_inv
        }
    }
}

pub struct WorkMatrixColIter<'a, T> {
    curent: std::slice::Iter<'a, usize>,
    col_index: usize,
    row_data: &'a [Vec<(usize, T)>],
    len: usize
}

impl<'a, T> Iterator for WorkMatrixColIter<'a, T> {

    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut row_index = *self.curent.next()?;
        while row_index >= self.len {
            row_index = *self.curent.next()?;
        }
        let entry = self.row_data[row_index].binary_search_by_key(&self.col_index, |(j, _)| *j).map(|index| &self.row_data[row_index][index].1).unwrap();
        return Some((self.len - 1 - row_index, entry));
    }
}

pub struct WorkMatrixCol<'a, T> {
    row_data: &'a [Vec<(usize, T)>],
    col: &'a [usize],
    col_index: usize,
    zero: &'a T,
    len: usize
}

impl<'a, T> VectorView<T> for WorkMatrixCol<'a, T> {

    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, i: usize) -> &T {
        self.row_data[self.len - 1 - i].binary_search_by_key(&self.col_index, |(j, _)| *j).map(|index| &self.row_data[self.len - 1 - i][index].1).unwrap_or(self.zero)
    }
}

impl<'a, T> VectorViewSparse<T> for WorkMatrixCol<'a, T> {

    type Iter<'b> = WorkMatrixColIter<'b, T>
        where Self: 'b, T: 'b;

    fn nontrivial_entries<'b>(&'b self) -> Self::Iter<'b> {
        WorkMatrixColIter {
            col_index: self.col_index,
            curent: self.col.iter(),
            row_data: self.row_data,
            len: self.len
        }
    }
}

impl<F: FieldStore> Clone for SparseBaseMatrix<F>
    where F::Type: Field,
        F: Clone
{
    fn clone(&self) -> Self {
        Self {
            col_count: self.col_count,
            row_count: self.row_count,
            col_permutation: self.col_permutation.clone(),
            col_permutation_inv: self.col_permutation_inv.clone(),
            field: self.field.clone(),
            zero: self.field.zero(),
            rows: self.rows.iter().map(|data| data.iter().map(|(j, c)| (*j, self.field.clone_el(c))).collect()).collect()
        }
    }
}

impl<F: FieldStore> SparseBaseMatrix<F>
    where F::Type: Field
{
    pub fn new<I>(field: F, row_count: usize, col_count: usize, entries: I) -> Self
        where I: Iterator<Item = (usize, usize, El<F>)>
    {
        let mut rows = Vec::new();
        rows.resize_with(row_count, Vec::new);
        
        let col_permutation = (0..col_count).collect();
        let col_permutation_inv = (0..col_count).collect();

        let mut result = SparseBaseMatrix { 
            col_permutation, 
            col_permutation_inv,
            row_count,
            col_count,
            zero: field.zero(),
            field, 
            rows, 
        };

        for (i, j, e) in entries {
            assert!(!result.field.is_zero(&e));
            let global_i = row_count - i - 1;
            let global_j = col_count - j - 1;
            result.rows[global_i].push((global_j, e));
        }

        for i in 0..row_count {
            result.rows[i].sort_unstable_by_key(|(j, _)| *j);
            for index in 1..result.rows[i].len() {
                assert!(result.rows[i][index - 1].0 != result.rows[i][index].0);
            }
        }

        return result;
    }

    fn global_row_index(&self, i: usize) -> usize {
        self.row_count - i - 1
    }

    fn global_col_index(&self, j: usize) -> usize {
        self.col_count - j - 1
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn col_count(&self) -> usize {
        self.col_count
    }

    pub fn at(&self, i: usize, j: usize) -> &El<F> {
        assert!(i < self.row_count);
        assert!(j < self.col_count);
        let hard_column = self.col_permutation[self.global_col_index(j)];
        let result = self.rows[self.global_row_index(i)].binary_search_by_key(&hard_column, |(index, _)| *index).map(|index| &self.rows[self.global_row_index(i)][index].1).unwrap_or(&self.zero);
        
        #[cfg(feature = "expensive_checks")] {
            assert_eq!(result as *const _, self.get_row(i).at(j) as *const _);
        }
        return result;
    }

    pub fn get_row<'a>(&'a self, i: usize) -> WorkMatrixRow<'a, El<F>> {
        WorkMatrixRow { data: &self.rows[self.global_row_index(i)], len: self.col_count, permutation: &self.col_permutation, permutation_inv: &self.col_permutation_inv, zero: &self.zero }
    }

    pub fn column_permutation<'a>(&'a self) -> impl 'a + Iterator<Item = usize> {
        self.col_permutation.iter().rev().map(|j| self.col_count - 1 - *j)
    }

    pub fn column_permutation_inv<'a>(&'a self) -> impl 'a + Iterator<Item = usize> {
        self.col_permutation_inv.iter().rev().map(|j| self.col_count - 1 - *j)
    }
}


impl<'a, F: FieldStore> SparseWorkMatrix<'a, F>
    where F::Type: Field
{
    pub fn new(base: &'a mut SparseBaseMatrix<F>) -> Self {
        let mut result = SparseWorkMatrix {
            row_nonzero_entry_counts: (0..base.row_count).map(|i| base.rows[i].len()).collect(),
            col_nonzero_entry_counts: (0..base.col_count).map(|_| 0).collect(),
            row_count: base.row_count,
            col_count: base.col_count,
            base: base,
            recycle_vec: Some(Vec::new())
        };
        for i in 0..result.base.row_count {
            for (j, _) in &result.base.rows[i] {
                result.col_nonzero_entry_counts[*j] += 1;
            }
        }
        result.check_invariants();
        return result;
    }

    #[cfg(feature = "expensive_checks")]
    fn check_invariants(&self) {

        for j in 0..self.col_count {
            let nonzero_entry_count = (0..self.row_count).filter(|i| self.base.rows[*i].iter().any(|(j2, _)| *j2 == j)).count();
            assert_eq!(nonzero_entry_count, self.col_nonzero_entry_counts[j]);
        }

        for i in 0..self.row_count {
            assert_eq!(self.base.rows[i].iter().filter(|(j, _)| (&self.base.col_permutation[0..self.col_count]).contains(j)).count(), self.row_nonzero_entry_counts[i]);
        }

        for j in 0..self.col_count {
            assert_eq!(j, self.base.col_permutation_inv[self.base.col_permutation[j]]);
            assert_eq!(j, self.base.col_permutation[self.base.col_permutation_inv[j]]);
        }
    }
    
    #[cfg(not(feature = "expensive_checks"))]
    fn check_invariants(&self) {}

    fn nonzero_entry_added(row_nonzero_entry_counts: &mut Vec<usize>, col_nonzero_entry_counts: &mut Vec<usize>, global_col: usize, global_row: usize) {
        row_nonzero_entry_counts[global_row] += 1;
        col_nonzero_entry_counts[global_col] += 1;
    }

    fn nonzero_entry_cancelled(row_nonzero_entry_counts: &mut Vec<usize>, col_nonzero_entry_counts: &mut Vec<usize>, global_col: usize, global_row: usize) {
        row_nonzero_entry_counts[global_row] -= 1;
        col_nonzero_entry_counts[global_col] -= 1;
    }

    fn global_row_index(&self, i: usize) -> usize {
        self.row_count - i - 1
    }

    fn global_col_index(&self, j: usize) -> usize {
        self.col_count - j - 1
    }

    pub fn at(&self, i: usize, j: usize) -> &El<F> {
        self.base.at(i + self.base.row_count - self.row_count, j + self.base.row_count - self.row_count)
    }

    pub fn get_row<'b>(&'b self, i: usize) -> WorkMatrixRow<'b, El<F>> {
        let mut result = self.base.get_row(i + self.base.row_count - self.row_count);
        result.len = self.col_count;
        return result;
    }

    pub fn base_field(&self) -> &F {
        &self.base.field
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn col_count(&self) -> usize {
        self.col_count
    }

    pub fn swap_cols(&mut self, j1: usize, j2: usize) {
        assert!(j1 < self.col_count);
        assert!(j2 < self.col_count);
        if j1 == j2 {
            return;
        }
        self.check_invariants();
        let global1 = self.global_col_index(j1);
        let global2 = self.global_col_index(j2);
        self.base.col_permutation.swap(global1, global2);
        self.base.col_permutation_inv.swap(self.base.col_permutation[global1], self.base.col_permutation[global2]);
        self.check_invariants();
    }

    pub fn swap_rows(&mut self, i1: usize, i2: usize) {
        assert!(i1 < self.row_count);
        assert!(i2 < self.row_count);
        if i1 == i2 {
            return;
        }
        self.check_invariants();
        let global1 = self.global_row_index(i1);
        let global2 = self.global_row_index(i2);
        self.base.rows.swap(global1, global2);
        self.row_nonzero_entry_counts.swap(global1, global2);
        self.check_invariants();
    }

    pub fn nonzero_entries_in_row(&self, i: usize) -> usize {
        assert!(i < self.row_count);
        self.row_nonzero_entry_counts[self.global_row_index(i)]
    }

    pub fn nonzero_entries_in_col(&self, j: usize) -> usize {
        assert!(j < self.col_count);
        self.col_nonzero_entry_counts[self.base.col_permutation[self.global_col_index(j)]]
    }

    pub fn sub_row(&mut self, dst_i: usize, src_i: usize, factor: &El<F>) {
        self.check_invariants();
        let mut new_row = self.recycle_vec.take().unwrap();
        new_row.clear();
        let mut dst_index = 0;
        let mut src_index = 0;
        let dst_i_global = self.global_row_index(dst_i);
        let src_i_global = self.global_row_index(src_i);
        let dst = &self.base.rows[dst_i_global];
        let src = &self.base.rows[src_i_global];
        while dst_index != dst.len() || src_index != src.len() {
            let dst_j = dst.get(dst_index).map(|e| e.0).unwrap_or(usize::MAX);
            let src_j = src.get(src_index).map(|e| e.0).unwrap_or(usize::MAX);

            if dst_j == src_j {
                let new_value = self.base.field.sub_ref_fst(&dst[dst_index].1, self.base.field.mul_ref(&src[src_index].1, factor));
                if self.base.field.is_zero(&new_value) {
                    // cancellation occurs - we have to adjust every value that depends on the position of nonzero entries
                    Self::nonzero_entry_cancelled(&mut self.row_nonzero_entry_counts, &mut self.col_nonzero_entry_counts, src_j, dst_i_global);
                } else {
                    // no cancellation - this entry remains nonzero
                    new_row.push((dst_j, new_value));
                }
                dst_index += 1;
                src_index += 1;
            } else if dst_j < src_j {
                // we just keep this entry, thus it remains nonzero
                new_row.push((dst_j, self.base.field.clone_el(&dst[dst_index].1)));
                dst_index += 1;
            } else {
                // we get a new entry, thus we have to update position of nonzero entries
                Self::nonzero_entry_added(&mut self.row_nonzero_entry_counts, &mut self.col_nonzero_entry_counts, src_j, dst_i_global);
                new_row.push((src_j, self.base.field.negate(self.base.field.mul_ref(&src[src_index].1, factor))));
                src_index += 1;
            }
        }
        self.recycle_vec = Some(std::mem::replace(&mut self.base.rows[dst_i_global], new_row));
        self.check_invariants();
    }

    ///
    /// This requires that the area left of the lower right submatrix is completely zero!
    /// 
    pub fn into_lower_right_submatrix(mut self) -> Self {
        self.check_invariants();
        self.row_count -= 1;
        self.col_count -= 1;
        for (i, _) in &self.base.rows[self.row_count] {
            self.col_nonzero_entry_counts[*i] -= 1;
        }
        self.check_invariants();
        return self;
    }
}

pub fn sparse_row_echelon<F>(mut A: SparseWorkMatrix<F>) 
    where F: FieldStore + Clone,
        F::Type: Field
{
    let field = A.base_field().clone();
    if let Some(pivot_i) = (0..A.row_count()).filter(|i| A.nonzero_entries_in_row(*i) > 0).min_by_key(|i| A.nonzero_entries_in_row(*i)) {
        
        A.swap_rows(0, pivot_i);

        let pivot_row = A.get_row(0);
        let (pivot_j, _) = pivot_row.nontrivial_entries().min_by_key(|(j, _)| A.nonzero_entries_in_col(*j)).unwrap();
        A.swap_cols(0, pivot_j);
    
        let pivot_inv = field.invert(A.at(0, 0)).unwrap();
    
        for i in 1..A.row_count() {
            if !field.is_zero(A.at(i, 0)) {
                A.sub_row(i, 0, &field.mul_ref(A.at(i, 0), &pivot_inv));
            }
        }
    
        sparse_row_echelon(A.into_lower_right_submatrix());
    } else {
        // we are left with the zero matrix
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_sub_row() {
    let field = Zn::<17>::RING;
    let mut base = SparseBaseMatrix::new(field, 8, 8, [
        (0, 0, 5), (1, 1, 3), (2, 2, 1), (3, 3, 16), (4, 4, 12), (5, 5, 3), (6, 6, 1), (7, 7, 6), 
        (0, 3, 8), (5, 2, 1), (4, 0, 5)
    ].into_iter());
    let mut a = SparseWorkMatrix::new(&mut base);

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(1, a.nonzero_entries_in_row(1));
    assert_eq!(2, a.nonzero_entries_in_row(4));
    assert_eq!(2, a.nonzero_entries_in_row(5));

    assert_eq!(2, a.nonzero_entries_in_col(0));
    assert_eq!(1, a.nonzero_entries_in_col(1));
    assert_eq!(2, a.nonzero_entries_in_col(2));

    a.sub_row(4, 0, &1);
    assert_eq!(2, a.nonzero_entries_in_row(4));
    
    assert_eq!(0, *a.at(4, 0));
    assert_eq!(0, *a.at(4, 1));
    assert_eq!(0, *a.at(4, 2));
    assert_eq!(9, *a.at(4, 3));
    assert_eq!(12, *a.at(4, 4));
    assert_eq!(0, *a.at(4, 5));
    assert_eq!(0, *a.at(4, 6));
    assert_eq!(0, *a.at(4, 7));

    a.sub_row(5, 2, &1);
    assert_eq!(1, a.nonzero_entries_in_row(5));

    // after this cancellation, there are only 3 off-diagonal entries - two in row 4 and one in row 0 

    assert_eq!(0, *a.at(5, 0));
    assert_eq!(0, *a.at(5, 1));
    assert_eq!(0, *a.at(5, 2));
    assert_eq!(0, *a.at(5, 3));
    assert_eq!(0, *a.at(5, 4));
    assert_eq!(3, *a.at(5, 5));
    assert_eq!(0, *a.at(5, 6));
    assert_eq!(0, *a.at(5, 7));

    assert_eq!(1, a.nonzero_entries_in_col(0));
    assert_eq!(1, a.nonzero_entries_in_col(1));
    assert_eq!(1, a.nonzero_entries_in_col(2));
    assert_eq!(3, a.nonzero_entries_in_col(3));
    assert_eq!(1, a.nonzero_entries_in_col(4));
    assert_eq!(1, a.nonzero_entries_in_col(5));
    assert_eq!(1, a.nonzero_entries_in_col(6));
    assert_eq!(1, a.nonzero_entries_in_col(7));

    let a = a.into_lower_right_submatrix();

    assert_eq!(1, a.nonzero_entries_in_row(0));
    assert_eq!(1, a.nonzero_entries_in_row(1));
    assert_eq!(1, a.nonzero_entries_in_row(2));
    assert_eq!(2, a.nonzero_entries_in_row(3));
    assert_eq!(1, a.nonzero_entries_in_row(4));
    assert_eq!(1, a.nonzero_entries_in_row(5));
    assert_eq!(1, a.nonzero_entries_in_row(6));

    assert_eq!(1, a.nonzero_entries_in_col(0));
    assert_eq!(1, a.nonzero_entries_in_col(1));
    assert_eq!(2, a.nonzero_entries_in_col(2));
    assert_eq!(1, a.nonzero_entries_in_col(3));
    assert_eq!(1, a.nonzero_entries_in_col(4));
    assert_eq!(1, a.nonzero_entries_in_col(5));
    assert_eq!(1, a.nonzero_entries_in_col(6));
}

#[test]
fn test_swap_rows() {
    let field = Zn::<17>::RING;
    // 1     7
    // 9 2
    //     3 8
    //   6   4
    let mut base = SparseBaseMatrix::new(field, 4, 4, [
        (0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4),
        (1, 0, 9), (2, 3, 8), (0, 3, 7), (3, 1, 6)
    ].into_iter());
    let mut a = SparseWorkMatrix::new(&mut base);

    a.swap_cols(0, 2);
    a.swap_rows(0, 2);
    // 3     8
    //   2 9
    //     1 7
    //   6   4

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(2, a.nonzero_entries_in_row(1));
    assert_eq!(2, a.nonzero_entries_in_row(2));
    assert_eq!(2, a.nonzero_entries_in_row(3));

    assert_eq!(1, a.nonzero_entries_in_col(0));
    assert_eq!(2, a.nonzero_entries_in_col(1));
    assert_eq!(2, a.nonzero_entries_in_col(2));
    assert_eq!(3, a.nonzero_entries_in_col(3));

    let mut a = a.into_lower_right_submatrix();

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(2, a.nonzero_entries_in_row(1));
    assert_eq!(2, a.nonzero_entries_in_row(2));

    assert_eq!(2, a.nonzero_entries_in_col(0));
    assert_eq!(2, a.nonzero_entries_in_col(1));
    assert_eq!(2, a.nonzero_entries_in_col(2));

    a.sub_row(2, 0, &3);
    // 2 9
    //   1 7
    //   7 4

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(2, a.nonzero_entries_in_row(1));
    assert_eq!(2, a.nonzero_entries_in_row(2));

    assert_eq!(1, a.nonzero_entries_in_col(0));
    assert_eq!(3, a.nonzero_entries_in_col(1));
    assert_eq!(2, a.nonzero_entries_in_col(2));

    let mut a = a.into_lower_right_submatrix();

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(2, a.nonzero_entries_in_row(1));

    assert_eq!(2, a.nonzero_entries_in_col(0));
    assert_eq!(2, a.nonzero_entries_in_col(1));

    a.swap_cols(0, 1);
    a.swap_rows(0, 1);
    // 4 7
    // 7 1

    a.sub_row(1, 0, &6);
    // 4 7
    //   10

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(1, a.nonzero_entries_in_row(1));

    assert_eq!(1, a.nonzero_entries_in_col(0));
    assert_eq!(2, a.nonzero_entries_in_col(1));

    assert_eq!(10, *a.at(1, 1));
}

#[test]
fn test_nonsquare() {
    let field = Zn::<17>::RING;

    let mut base = SparseBaseMatrix::new(field, 5, 3, [
        (0, 0, 6), (0, 2, 5), (1, 1, 2), (1, 2, 3), (2, 0, 2), (3, 0, 4), (3, 2, 1), (4, 1, 6)
    ].into_iter());
    let mut a = SparseWorkMatrix::new(&mut base);
    // 6   5
    //   2 3
    // 2    
    // 4   1
    //   6

    a.swap_rows(0, 3);
    a.swap_cols(0, 2);
    // 1   4
    // 3 2  
    //     2
    // 5   6
    //   6

    assert_eq!(6, *a.at(3, 2));

    a.sub_row(1, 0, &3);
    a.sub_row(3, 0, &5);
    // 1   4
    //   2 5
    //     2
    //     3
    //   6  

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(2, a.nonzero_entries_in_row(1));
    assert_eq!(1, a.nonzero_entries_in_row(2));
    assert_eq!(1, a.nonzero_entries_in_row(3));
    assert_eq!(1, a.nonzero_entries_in_row(4));

    assert_eq!(vec![(1, &2), (2, &5)], a.get_row(1).nontrivial_entries().collect::<Vec<_>>());

    let a = a.into_lower_right_submatrix();

    assert_eq!(1, a.nonzero_entries_in_row(1));
}

#[test]
fn test_sparse_row_echelon() {
    let field = Zn::<17>::RING;
    let mut base = SparseBaseMatrix::new(field, 6, 5, [
        (0, 1, 8), (0, 2, 2), (0, 4, 1), (1, 1, 2), (1, 2, 1), (1, 4, 1), (2, 0, 3), (2, 3, 1), (3, 0, 2), (3, 3, 1), (3, 4, 16), (4, 0, 4), (4, 1, 10), (4, 2, 3), (4, 3, 2), (5, 0, 3), (5, 1, 15), (5, 2, 16), (5, 3, 1), (5, 4, 16)
    ].into_iter());
    //   8 2   1
    //   2 1   1
    // 3     1
    // 2     1 16
    // 4 10 3 2 
    // 3 15 16 1 16

    sparse_row_echelon(SparseWorkMatrix::new(&mut base));

    let zero_vec = [1, 4, 10, 14, 16];
    let mut permuted_zero_vec = [0; 5];
    for (j1, j2) in base.column_permutation().enumerate() {
        permuted_zero_vec[j1] = zero_vec[j2];
    }
    for i in 0..6 {
        assert_el_eq!(&field, &0, &field.sum((0..5).map(|j| field.mul_ref(base.at(i, j), &permuted_zero_vec[j]))));
    }
}

#[test]
#[ignore]
fn test_perf_sparse_row_echelon() {
    let field = Zn::<17>::RING;
    let row_count = 20000;
    let col_count = 10000;
    let row_entries = 10;
    let mut rand = oorandom::Rand32::from_state((1, 1));
    let base = SparseBaseMatrix::new(field, row_count, col_count, (0..row_count).flat_map(|i| {
        let mut entries = (0..row_entries).map(|_| rand.rand_u32() as usize % col_count).collect::<Vec<_>>();
        entries.sort();
        entries.dedup();
        return entries.into_iter().map(move |j| (i, j, field.from_int((rand.rand_u32() % 16) as i32 + 1)));
    }));

    let start = std::time::Instant::now();
    {
        let mut current_base = base.clone();
        sparse_row_echelon(SparseWorkMatrix::new(&mut current_base));
        std::hint::black_box(current_base);
    }
    let end = std::time::Instant::now();
    println!("Time {} ms", (end - start).as_millis());
}