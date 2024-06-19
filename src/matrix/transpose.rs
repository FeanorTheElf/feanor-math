use super::submatrix::*;

use std::ops::Range;

#[stability::unstable(feature = "enable")]
pub struct TransposableSubmatrix<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> {
    data: Submatrix<'a, V, T>
}

impl<'a, V: AsPointerToSlice<T>, T> From<Submatrix<'a, V, T>> for TransposableSubmatrix<'a, V, T, false> {

    fn from(value: Submatrix<'a, V, T>) -> Self {
        Self { data: value }
    }
}

impl<'a, V: AsPointerToSlice<T>, T> TransposableSubmatrix<'a, V, T, false> {

    #[stability::unstable(feature = "enable")]
    pub fn transpose(self) -> TransposableSubmatrix<'a, V, T, true> {
        TransposableSubmatrix { data: self.data }
    }
}

impl<'a, V: AsPointerToSlice<T>, T> TransposableSubmatrix<'a, V, T, true> {

    #[stability::unstable(feature = "enable")]
    pub fn transpose(self) -> TransposableSubmatrix<'a, V, T, false> {
        TransposableSubmatrix { data: self.data }
    }
}

impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> Copy for TransposableSubmatrix<'a, V, T, TRANSPOSED> {}

impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> Clone for TransposableSubmatrix<'a, V, T, TRANSPOSED> {

    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> TransposableSubmatrix<'a, V, T, TRANSPOSED> {
    
    #[stability::unstable(feature = "enable")]
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.submatrix(cols, rows) }
        } else {
            Self { data: self.data.submatrix(rows, cols) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.restrict_cols(rows) }
        } else {
            Self { data: self.data.restrict_rows(rows) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn into_at(self, i: usize, j: usize) -> &'a T {
        if TRANSPOSED {
            self.data.into_at(j, i)
        } else {
            self.data.into_at(i, j)
        }
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn at<'b>(&'b self, i: usize, j: usize) -> &'b T {
        if TRANSPOSED {
            self.data.at(j, i)
        } else {
            self.data.at(i, j)
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.restrict_rows(cols) }
        } else {
            Self { data: self.data.restrict_cols(cols) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn col_count(&self) -> usize {
        if TRANSPOSED {
            self.data.row_count()
        } else {
            self.data.col_count()
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn row_count(&self) -> usize {
        if TRANSPOSED {
            self.data.col_count()
        } else {
            self.data.row_count()
        }
    }
}

#[stability::unstable(feature = "enable")]
pub struct TransposableSubmatrixMut<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> {
    data: SubmatrixMut<'a, V, T>
}

impl<'a, V: AsPointerToSlice<T>, T> TransposableSubmatrixMut<'a, V, T, false> {

    #[stability::unstable(feature = "enable")]
    pub fn transpose(self) -> TransposableSubmatrixMut<'a, V, T, true> {
        TransposableSubmatrixMut { data: self.data }
    }
}

impl<'a, V: AsPointerToSlice<T>, T> TransposableSubmatrixMut<'a, V, T, true> {

    #[stability::unstable(feature = "enable")]
    pub fn transpose(self) -> TransposableSubmatrixMut<'a, V, T, false> {
        TransposableSubmatrixMut { data: self.data }
    }
}

impl<'a, V: AsPointerToSlice<T>, T> From<SubmatrixMut<'a, V, T>> for TransposableSubmatrixMut<'a, V, T, false> {

    fn from(value: SubmatrixMut<'a, V, T>) -> Self {
        Self { data: value }
    }
}

impl<'a, V: AsPointerToSlice<T>, T, const TRANSPOSED: bool> TransposableSubmatrixMut<'a, V, T, TRANSPOSED> {

    #[stability::unstable(feature = "enable")]
    pub fn submatrix(self, rows: Range<usize>, cols: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.submatrix(cols, rows) }
        } else {
            Self { data: self.data.submatrix(rows, cols) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_rows(self, rows: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.restrict_cols(rows) }
        } else {
            Self { data: self.data.restrict_rows(rows) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn at<'b>(&'b self, i: usize, j: usize) -> &'b T {
        if TRANSPOSED {
            self.data.at(j, i)
        } else {
            self.data.at(i, j)
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn restrict_cols(self, cols: Range<usize>) -> Self {
        if TRANSPOSED {
            Self { data: self.data.restrict_rows(cols) }
        } else {
            Self { data: self.data.restrict_cols(cols) }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn col_count(&self) -> usize {
        if TRANSPOSED {
            self.data.row_count()
        } else {
            self.data.col_count()
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn row_count(&self) -> usize {
        if TRANSPOSED {
            self.data.col_count()
        } else {
            self.data.row_count()
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn split_rows(self, fst_rows: Range<usize>, snd_rows: Range<usize>) -> (Self, Self) {
        if TRANSPOSED {
            let (fst, snd) = self.data.split_cols(fst_rows, snd_rows);
            return (Self { data: fst }, Self { data: snd });
        } else {
            let (fst, snd) = self.data.split_rows(fst_rows, snd_rows);
            return (Self { data: fst }, Self { data: snd });
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn split_cols(self, fst_cols: Range<usize>, snd_cols: Range<usize>) -> (Self, Self) {
        if TRANSPOSED {
            let (fst, snd) = self.data.split_rows(fst_cols, snd_cols);
            return (Self { data: fst }, Self { data: snd });
        } else {
            let (fst, snd) = self.data.split_cols(fst_cols, snd_cols);
            return (Self { data: fst }, Self { data: snd });
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn at_mut<'b>(&'b mut self, i: usize, j: usize) -> &'b mut T {
        if TRANSPOSED {
            self.data.at_mut(j, i)
        } else {
            self.data.at_mut(i, j)
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn reborrow<'b>(&'b mut self) -> TransposableSubmatrixMut<'b, V, T, TRANSPOSED> {
        TransposableSubmatrixMut { data: self.data.reborrow() }
    }

    #[stability::unstable(feature = "enable")]
    pub fn as_const<'b>(&'b self) -> TransposableSubmatrix<'b, V, T, TRANSPOSED> {
        TransposableSubmatrix { data: self.data.as_const() }
    }
}