use std::fmt::Display;

use crate::ring::*;

mod submatrix;
mod transpose;
mod owned;

pub use submatrix::*;
#[allow(unused_imports)]
pub use transpose::*;
pub use owned::*;

///
/// Contains the trait [`transform::TransformTarget`], for "consumers" of elementary
/// matrix operations.
/// 
pub mod transform;

#[stability::unstable(feature = "enable")]
pub fn format_matrix<'a, M, R>(row_count: usize, col_count: usize, matrix: M, ring: R) -> impl 'a + Display
    where R: 'a + RingStore, 
        El<R>: 'a,
        M: 'a + Fn(usize, usize) -> &'a El<R>
{
    struct DisplayWrapper<'a, R: 'a + RingStore, M: Fn(usize, usize) -> &'a El<R>> {
        matrix: M,
        ring: R,
        row_count: usize,
        col_count: usize
    }

    impl<'a, R: 'a + RingStore, M: Fn(usize, usize) -> &'a El<R>> Display for DisplayWrapper<'a, R, M> {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let strings = (0..self.row_count).flat_map(|i| (0..self.col_count).map(move |j| (i, j)))
                .map(|(i, j)| format!("{}", self.ring.format((self.matrix)(i, j))))
                .collect::<Vec<_>>();
            let max_len = strings.iter().map(|s| s.chars().count()).chain([2].into_iter()).max().unwrap();
            let mut strings = strings.into_iter();
            for i in 0..self.row_count {
                write!(f, "|")?;
                if self.col_count > 0 {
                    write!(f, "{:>width$}", strings.next().unwrap(), width = max_len)?;
                }
                for _ in 1..self.col_count {
                    write!(f, ",{:>width$}", strings.next().unwrap(), width = max_len)?;
                }
                if i + 1 != self.row_count {
                    writeln!(f, "|")?;
                } else {
                    write!(f, "|")?;
                }
            }
            return Ok(());
        }
    }

    DisplayWrapper { matrix, ring, col_count, row_count }
}

///
/// Defines a very general way of comparing anything that can be interpreted as a
/// matrix in [`matrix_compare::MatrixCompare`]. Used solely in [`crate::assert_matrix_eq`],
/// with tests being the primary use case.
/// 
pub mod matrix_compare {
    use super::*;

    ///
    /// Used by [`crate::assert_matrix_eq`] to compare objects that are matrices in a
    /// very general sense.
    /// 
    pub trait MatrixCompare<T> {
        fn row_count(&self) -> usize;
        fn col_count(&self) -> usize;
        fn at(&self, i: usize, j: usize) -> &T;
    }

    impl<T, const ROWS: usize, const COLS: usize> MatrixCompare<T> for [[T; COLS]; ROWS] {

        fn col_count(&self) -> usize { COLS }
        fn row_count(&self) -> usize { ROWS }
        fn at(&self, i: usize, j: usize) -> &T { &self[i][j] }
    }

    impl<T, const ROWS: usize, const COLS: usize> MatrixCompare<T> for [DerefArray<T, COLS>; ROWS] {

        fn col_count(&self) -> usize { COLS }
        fn row_count(&self) -> usize { ROWS }
        fn at(&self, i: usize, j: usize) -> &T { &self[i][j] }
    }

    impl<'a, V: AsPointerToSlice<T>, T> MatrixCompare<T> for Submatrix<'a, V, T> {

        fn col_count(&self) -> usize { Submatrix::col_count(self) }
        fn row_count(&self) -> usize { Submatrix::row_count(self) }
        fn at(&self, i: usize, j: usize) -> &T { Submatrix::at(self, i, j) }
    }

    impl<'a, V: AsPointerToSlice<T>, T> MatrixCompare<T> for SubmatrixMut<'a, V, T> {

        fn col_count(&self) -> usize { SubmatrixMut::col_count(self) }
        fn row_count(&self) -> usize { SubmatrixMut::row_count(self) }
        fn at(&self, i: usize, j: usize) -> &T { self.as_const().into_at(i, j) }
    }

    impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> MatrixCompare<T> for TransposableSubmatrix<'a, V, T, TRANSPOSED> {

        fn col_count(&self) -> usize { TransposableSubmatrix::col_count(self) }
        fn row_count(&self) -> usize { TransposableSubmatrix::row_count(self) }
        fn at(&self, i: usize, j: usize) -> &T { TransposableSubmatrix::at(self, i, j) }
    }

    impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> MatrixCompare<T> for TransposableSubmatrixMut<'a, V, T, TRANSPOSED> {

        fn col_count(&self) -> usize { TransposableSubmatrixMut::col_count(self) }
        fn row_count(&self) -> usize { TransposableSubmatrixMut::row_count(self) }
        fn at(&self, i: usize, j: usize) -> &T { self.as_const().into_at(i, j) }
    }

    impl<T> MatrixCompare<T> for OwnedMatrix<T> {

        fn col_count(&self) -> usize { OwnedMatrix::col_count(self) }
        fn row_count(&self) -> usize { OwnedMatrix::row_count(self) }
        fn at(&self, i: usize, j: usize) -> &T { OwnedMatrix::at(self, i, j) }
    }

    impl<'a, T, M: MatrixCompare<T>> MatrixCompare<T> for &'a M {
        
        fn col_count(&self) -> usize { (**self).col_count() }
        fn row_count(&self) -> usize { (**self).row_count() }
        fn at(&self, i: usize, j: usize) -> &T { (**self).at(i, j) }
    }

    ///
    /// Checks whether two matrices are equal.
    /// 
    /// The prime use case is [`crate::assert_matrix_eq!`], which should be applicable
    /// to all types that are, in some sense, interpretable as matrices.
    /// 
    pub fn is_matrix_eq<R: ?Sized + RingBase, M1: MatrixCompare<R::Element>, M2: MatrixCompare<R::Element>>(ring: &R, lhs: &M1, rhs: &M2) -> bool {
        if lhs.row_count() != rhs.row_count() || lhs.col_count() != rhs.col_count() {
            return false;
        }
        for i in 0..lhs.row_count() {
            for j in 0..lhs.col_count() {
                if !ring.eq_el(lhs.at(i, j), rhs.at(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
}

///
/// Variant of `assert_eq!` for matrices elements, i.e. assert that two ring matrices are equal.
/// Frequently used in tests.
/// 
/// This takes any arguments that implement [`matrix_compare::MatrixCompare`], a very minimal trait
/// for matrices that is implemented for all types that are, in some sense, interpretable as matrices.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::matrix::*;
/// # use feanor_math::assert_matrix_eq;
/// let lhs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
/// let rhs = OwnedMatrix::zero(3, 3, StaticRing::<i64>::RING);
/// assert_matrix_eq!(StaticRing::<i64>::RING, lhs, rhs);
///  ```
/// 
#[macro_export]
macro_rules! assert_matrix_eq {
    ($ring:expr, $lhs:expr, $rhs:expr) => {
        match (&$ring, &$lhs, &$rhs) {
            (ring_val, lhs_val, rhs_val) => {
                assert!(
                    $crate::matrix::matrix_compare::is_matrix_eq(ring_val.get_ring(), lhs_val, rhs_val), 
                    "Assertion failed: Expected\n{}\nto be\n{}", 
                    $crate::matrix::format_matrix(<_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::row_count(lhs_val), <_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::col_count(lhs_val), |i, j| <_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::at(lhs_val, i, j), ring_val),
                    $crate::matrix::format_matrix(<_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::row_count(rhs_val), <_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::col_count(rhs_val), |i, j| <_ as $crate::matrix::matrix_compare::MatrixCompare<_>>::at(rhs_val, i, j), ring_val)
                );
            }
        }
    }
}
