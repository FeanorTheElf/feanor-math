use crate::ring::*;
use crate::field::*;

struct WorkMatrix<F: FieldStore>
    where F::Type: Field
{
    // base data
    field: F,
    rows: Vec<Vec<(usize, El<F>)>>,
    zero: El<F>,
    col_permutation: Vec<usize>,
    n: usize,
    base_n: usize,
    col_nonzero_entry_counts: Vec<usize>
}

impl<F: FieldStore> WorkMatrix<F>
    where F::Type: Field
{
    fn new<I>(field: F, n: usize, entries: I) -> Self
        where I: Iterator<Item = (usize, usize, El<F>)>
    {
        let mut rows = Vec::new();
        rows.resize_with(n, Vec::new);
        
        let mut col_nonzero_entry_counts = Vec::new();
        col_nonzero_entry_counts.resize(n, 0);
        let col_permutation = (0..n).collect();

        let mut result = WorkMatrix { 
            col_nonzero_entry_counts, 
            col_permutation, 
            n, 
            base_n: n,
            zero: field.zero(),
            field, 
            rows
        };

        for (i, j, e) in entries {
            assert!(!result.field.is_zero(&e));
            let global_i = result.global_index(i);
            let global_j = result.global_index(j);
            result.rows[global_i].push((global_j, e));
            result.col_nonzero_entry_counts[global_j] += 1;
        }

        for i in 0..n {
            result.rows[i].sort_by_key(|(j, _)| *j);
        }

        return result;
    }

    fn check_invariants(&self) {
        for j in 0..self.n {
            let nonzero_entry_count = (0..self.n).filter(|i| self.rows[*i].iter().any(|(j2, _)| *j2 == j)).count();
            assert_eq!(nonzero_entry_count, self.col_nonzero_entry_counts[j]);
        }
    }

    fn replace_entry_in_cols(cols: &mut Vec<Vec<usize>>, global_col: usize, old: usize, new: usize) {
        let old_index = cols[global_col].binary_search_by_key(&old, |index| *index).unwrap();
        if let Err(new_index) = cols[global_col].binary_search_by_key(&new, |index| *index) {
            if old_index == new_index {
                cols[global_col][new_index] = new;
            } else if old_index < new_index {
                for i in old_index..(new_index - 1) {
                    cols[global_col][i] = cols[global_col][i + 1];
                }
                cols[global_col][new_index - 1] = new;
            } else {
                for i in new_index..old_index {
                    cols[global_col][i + 1] = cols[global_col][i];
                }
                cols[global_col][new_index] = new;
            }
        } else {
            // do nothing
        }
    }

    fn nonzero_entry_added(col_nonzero_entry_counts: &mut Vec<usize>, global_col: usize, global_row: usize) {
        col_nonzero_entry_counts[global_col] += 1;
    }

    fn nonzero_entry_cancelled(col_nonzero_entry_counts: &mut Vec<usize>, global_col: usize, global_row: usize) {
        col_nonzero_entry_counts[global_col] -= 1;
    }

    fn global_index(&self, i: usize) -> usize {
        self.n - i - 1
    }

    fn at(&self, i: usize, j: usize) -> &El<F> {
        assert!(i < self.n);
        assert!(j < self.n);
        let hard_column = self.col_permutation[self.global_index(j)];
        self.rows[self.global_index(i)].binary_search_by_key(&hard_column, |(index, _)| *index).map(|index| &self.rows[self.global_index(i)][index].1).unwrap_or(&self.zero)
    }

    fn swap_cols(&mut self, j1: usize, j2: usize) {
        assert!(j1 < self.n);
        assert!(j2 < self.n);
        if j1 == j2 {
            return;
        }
        self.check_invariants();
        let global1 = self.global_index(j1);
        let global2 = self.global_index(j2);
        self.col_permutation.swap(global1, global2);
        self.check_invariants();
    }

    fn swap_rows(&mut self, i1: usize, i2: usize) {
        assert!(i1 < self.n);
        assert!(i2 < self.n);
        if i1 == i2 {
            return;
        }
        self.check_invariants();
        let global1 = self.global_index(i1);
        let global2 = self.global_index(i2);
        self.rows.swap(global1, global2);
        self.check_invariants();
    }

    fn nonzero_entries_in_row(&self, i: usize) -> usize {
        assert!(i < self.n);
        self.rows[self.global_index(i)].len()
    }

    fn nonzero_entries_in_col(&self, j: usize) -> usize {
        assert!(j < self.n);
        self.col_nonzero_entry_counts[self.col_permutation[self.global_index(j)]]
    }

    fn sub_row(&mut self, dst_i: usize, src_i: usize, factor: &El<F>) {
        self.check_invariants();
        let mut new_row = Vec::new();
        let mut dst_index = 0;
        let mut src_index = 0;
        let dst_i_global = self.global_index(dst_i);
        let src_i_global = self.global_index(src_i);
        let dst = &self.rows[dst_i_global];
        let src = &self.rows[src_i_global];
        while dst_index != dst.len() || src_index != src.len() {
            let dst_j = dst.get(dst_index).map(|e| e.0).unwrap_or(usize::MAX);
            let src_j = src.get(src_index).map(|e| e.0).unwrap_or(usize::MAX);

            if dst_j == src_j {
                let new_value = self.field.sub_ref_fst(&dst[dst_index].1, self.field.mul_ref(&src[src_index].1, factor));
                if self.field.is_zero(&new_value) {
                    // cancellation occurs - we have to adjust every value that depends on the position of nonzero entries
                    Self::nonzero_entry_cancelled(&mut self.col_nonzero_entry_counts, src_j, dst_i_global);
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
                Self::nonzero_entry_added(&mut self.col_nonzero_entry_counts, src_j, dst_i_global);
                new_row.push((src_j, self.field.negate(self.field.mul_ref(&src[src_index].1, factor))));
                src_index += 1;
            }
        }
        self.rows[dst_i_global] = new_row;
        self.check_invariants();
    }

    ///
    /// This requires that the area left of the lower right submatrix is completely zero!
    /// 
    fn into_lower_right_submatrix(mut self) -> Self {
        self.check_invariants();
        self.n -= 1;
        for (i, _) in &self.rows[self.n] {
            self.col_nonzero_entry_counts[*i] -= 1;
        }
        self.check_invariants();
        return self;
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_sub_row() {
    let field = Zn::<17>::RING;
    let mut a = WorkMatrix::new(field, 8, [
        (0, 0, 5), (1, 1, 3), (2, 2, 1), (3, 3, 16), (4, 4, 12), (5, 5, 3), (6, 6, 1), (7, 7, 6), 
        (0, 3, 8), (5, 2, 1), (4, 0, 5)
    ].into_iter());

    assert_eq!(2, a.nonzero_entries_in_row(0));
    assert_eq!(1, a.nonzero_entries_in_row(1));
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
    let mut a = WorkMatrix::new(field, 4, [
        (0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4),
        (1, 0, 9), (2, 3, 8), (0, 3, 7), (3, 1, 6)
    ].into_iter());

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