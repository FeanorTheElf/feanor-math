use std::alloc::{Allocator, Global};
use std::fmt::{Debug, Formatter, Result};
use std::mem::MaybeUninit;

use self::submatrix::{AsFirstElement, Submatrix, SubmatrixMut};
use super::*;

/// A matrix that owns its elements.
///
/// To pass it to algorithms, use the `.data()` and `.data_mut()` functions.
///
/// # Example
/// ```rust
/// #![feature(allocator_api)]
/// # use std::alloc::*;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::matrix::*;
/// # use feanor_math::algorithms::linsolve::*;
/// let mut A = OwnedMatrix::identity(2, 2, StaticRing::<i32>::RING);
/// let mut B = OwnedMatrix::identity(2, 2, StaticRing::<i32>::RING);
/// let mut C = OwnedMatrix::identity(2, 2, StaticRing::<i32>::RING);
/// StaticRing::<i32>::RING
///     .get_ring()
///     .solve_right(A.data_mut(), B.data_mut(), C.data_mut(), Global)
///     .assert_solved();
/// ```
pub struct OwnedMatrix<T, A: Allocator = Global> {
    data: Vec<T, A>,
    col_count: usize,
    row_count: usize,
}

impl<T> OwnedMatrix<T> {
    /// Creates the `row_count x col_count` [`OwnedMatrix`] whose `(i, j)`-th entry
    /// is the output of the given function on `(i, j)`.
    pub fn from_fn<F>(row_count: usize, col_count: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_fn_in(row_count, col_count, f, Global)
    }

    /// Creates the `row_count x col_count` zero matrix over the given ring.
    pub fn zero<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
    where
        R::Type: RingBase<Element = T>,
    {
        Self::zero_in(row_count, col_count, ring, Global)
    }

    /// Creates the `row_count x col_count` identity matrix over the given ring.
    pub fn identity<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
    where
        R::Type: RingBase<Element = T>,
    {
        Self::identity_in(row_count, col_count, ring, Global)
    }
}

impl<T, A: Allocator> OwnedMatrix<T, A> {
    /// Creates the `row_count x col_count` [`OwnedMatrix`] matrix, whose entries are
    /// taken from the given vector, interpreted as a row-major matrix. The number of
    /// rows is `row_count = data.len() / col_count`.
    ///
    /// If `col_count` is zero, this will panic. If that can happen, consider
    /// using [`OwnedMatrix::new_with_shape()`].
    pub fn new(data: Vec<T, A>, col_count: usize) -> Self {
        let row_count = data.len() / col_count;
        Self::new_with_shape(data, row_count, col_count)
    }

    /// Creates the `row_count x col_count` [`OwnedMatrix`] matrix, whose entries are
    /// taken from the given vector, interpreted as a row-major matrix.
    ///
    /// # Example
    /// ```
    /// # use feanor_math::matrix::*;
    /// let matrix = OwnedMatrix::new_with_shape(vec![1, 2, 3, 4, 5, 6], 3, 2);
    /// assert_eq!(3, *matrix.at(1, 0));
    /// assert_eq!(6, *matrix.at(2, 1));
    /// ```
    pub fn new_with_shape(data: Vec<T, A>, row_count: usize, col_count: usize) -> Self {
        assert_eq!(row_count * col_count, data.len());
        Self {
            data,
            col_count,
            row_count,
        }
    }

    /// Creates the `row_count x col_count` [`OwnedMatrix`] whose `(i, j)`-th entry
    /// is the output of the given function on `(i, j)`.
    #[stability::unstable(feature = "enable")]
    pub fn from_fn_in<F>(row_count: usize, col_count: usize, mut f: F, allocator: A) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity_in(row_count * col_count, allocator);
        for i in 0..row_count {
            for j in 0..col_count {
                data.push(f(i, j));
            }
        }
        return Self::new_with_shape(data, row_count, col_count);
    }

    /// Returns a [`Submatrix`] view on the data of this matrix.
    pub fn data<'a>(&'a self) -> Submatrix<'a, AsFirstElement<T>, T> {
        Submatrix::<AsFirstElement<_>, _>::from_1d(&self.data, self.row_count(), self.col_count())
    }

    /// Returns a [`SubmatrixMut`] view on the data of this matrix.
    pub fn data_mut<'a>(&'a mut self) -> SubmatrixMut<'a, AsFirstElement<T>, T> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut self.data, row_count, col_count)
    }

    /// Returns a reference to the `(i, j)`-th entry of this matrix.
    pub fn at(&self, i: usize, j: usize) -> &T { &self.data[i * self.col_count + j] }

    /// Returns a mutable reference to the `(i, j)`-th entry of this matrix.
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut T { &mut self.data[i * self.col_count + j] }

    /// Returns the number of rows of this matrix.
    pub fn row_count(&self) -> usize { self.row_count }

    ////
    /// Returns the number of columns of this matrix.
    pub fn col_count(&self) -> usize { self.col_count }

    /// Creates the `row_count x col_count` zero matrix over the given ring.
    #[stability::unstable(feature = "enable")]
    pub fn zero_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
    where
        R::Type: RingBase<Element = T>,
    {
        let mut result = Vec::with_capacity_in(row_count * col_count, allocator);
        for _ in 0..row_count {
            for _ in 0..col_count {
                result.push(ring.zero());
            }
        }
        return Self::new_with_shape(result, row_count, col_count);
    }

    /// Creates the `row_count x col_count` identity matrix over the given ring.
    #[stability::unstable(feature = "enable")]
    pub fn identity_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
    where
        R::Type: RingBase<Element = T>,
    {
        let mut result = Vec::with_capacity_in(row_count * col_count, allocator);
        for i in 0..row_count {
            for j in 0..col_count {
                if i != j {
                    result.push(ring.zero());
                } else {
                    result.push(ring.one());
                }
            }
        }
        return Self::new_with_shape(result, row_count, col_count);
    }

    #[stability::unstable(feature = "enable")]
    pub fn clone_matrix<R: RingStore>(&self, ring: R) -> Self
    where
        R::Type: RingBase<Element = T>,
        A: Clone,
    {
        let mut result = Vec::with_capacity_in(self.row_count() * self.col_count(), self.data.allocator().clone());
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                result.push(ring.clone_el(self.at(i, j)));
            }
        }
        return Self::new_with_shape(result, self.row_count(), self.col_count());
    }

    #[stability::unstable(feature = "enable")]
    pub fn into_data(self) -> Vec<T, A> { self.data }

    #[stability::unstable(feature = "enable")]
    pub fn set_row_count<F>(&mut self, new_count: usize, new_entries: F)
    where
        F: FnMut() -> T,
    {
        self.data.resize_with(new_count * self.col_count(), new_entries);
    }
}

impl<T> OwnedMatrix<MaybeUninit<T>> {
    #[stability::unstable(feature = "enable")]
    pub fn uninit(row_count: usize, col_count: usize) -> Self { Self::uninit_in(row_count, col_count, Global) }
}

impl<T, A: Allocator> OwnedMatrix<MaybeUninit<T>, A> {
    #[stability::unstable(feature = "enable")]
    pub fn uninit_in(row_count: usize, col_count: usize, allocator: A) -> Self {
        let mut data = Vec::with_capacity_in(row_count * col_count, allocator);
        data.resize_with(row_count * col_count, || MaybeUninit::uninit());
        return Self {
            data,
            row_count,
            col_count,
        };
    }

    /// This is safe as long as every element in the matrix is initialized
    #[stability::unstable(feature = "enable")]
    pub unsafe fn assume_init(self) -> OwnedMatrix<T, A> {
        let (ptr, len, cap, alloc) = self.data.into_raw_parts_with_alloc();
        // SAFETY: MaybeUninit<T> has the same size and alignment as T, so the
        // allocation is valid for T; the caller guarantees all `len` elements
        // are initialized.
        let new_vec = unsafe { Vec::from_raw_parts_in(ptr.cast::<T>(), len, cap, alloc) };

        OwnedMatrix {
            data: new_vec,
            col_count: self.col_count,
            row_count: self.row_count,
        }
    }

    /// Initializes the elements of the matrix using the given closure.
    ///
    /// # Panics
    ///
    /// This function panics if the given closure did not initialize all of the matrix, i.e.
    /// returns a smaller submatrix.
    #[stability::unstable(feature = "enable")]
    pub fn init<F>(mut self, initialize: F) -> OwnedMatrix<T, A>
    where
        F: for<'a> FnOnce(
            SubmatrixMut<'a, AsFirstElement<MaybeUninit<T>>, MaybeUninit<T>>,
        ) -> SubmatrixMut<'a, AsFirstElement<MaybeUninit<T>>, T>,
    {
        let raw_parts = self.data().into_raw().into_raw_parts();
        let row_count = self.data().row_count();
        let col_count = self.data().col_count();

        let witness: SubmatrixMut<_, T> = initialize(self.data_mut());
        let witness_raw_parts = witness.into_raw().into_raw_parts();

        assert_eq!(
            raw_parts.1, witness_raw_parts.1,
            "initialize returned wrong number of rows"
        );
        assert_eq!(raw_parts.2, witness_raw_parts.2, "initialize returned wrong row stride");
        assert_eq!(
            raw_parts.3, witness_raw_parts.3,
            "initialize returned wrong column offset"
        );
        assert_eq!(
            raw_parts.4, witness_raw_parts.4,
            "initialize returned wrong number of columns"
        );
        if row_count != 0 && col_count != 0 {
            assert_eq!(
                raw_parts.0,
                witness_raw_parts.0.cast(),
                "initialize returned a pointer into a different buffer",
            );
        }

        let (ptr, length, capacity, alloc) = self.data.into_raw_parts_with_alloc();

        // SAFETY:
        // * ptr/capacity/alloc come straight from a live Vec, so they're mutually consistent.
        // * MaybeUninit<T> and T share size and alignment, so the ptr cast preserves the allocation's
        //   layout requirements for T.
        // * The elements referenced by the matrix are init T: the checks proved the caller's valid
        //   SubmatrixMut<T> covers exactly [base, base + len).
        let result_data = unsafe { Vec::from_raw_parts_in(ptr as *mut T, length, capacity, alloc) };
        return OwnedMatrix::new_with_shape(result_data, row_count, col_count);
    }
}

impl<T: Debug, A: Allocator> Debug for OwnedMatrix<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result { self.data().fmt(f) }
}

#[cfg(test)]
use crate::primitive_int::*;

#[test]
fn test_zero_col_matrix() {
    let A: OwnedMatrix<i64> = OwnedMatrix::new_with_shape(Vec::new(), 10, 0);
    assert_eq!(0, A.col_count());
    assert_eq!(10, A.row_count());

    let B: OwnedMatrix<i64> = OwnedMatrix::zero(11, 0, StaticRing::<i64>::RING);
    assert_eq!(0, B.col_count());
    assert_eq!(11, B.row_count());
}
