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
/// # use feanor_math::prelude::*;
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
        R::Ring: RingBase<Element = T>,
    {
        Self::zero_in(row_count, col_count, ring, Global)
    }

    /// Creates the `row_count x col_count` identity matrix over the given ring.
    pub fn identity<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
    where
        R::Ring: RingBase<Element = T>,
    {
        Self::identity_in(row_count, col_count, ring, Global)
    }
}

impl<T, A: Allocator> OwnedMatrix<T, A> {
    /// Creates the `row_count x col_count` [`OwnedMatrix`] matrix, whose entries are
    /// taken from the given vector, interpreted as a row-major matrix.
    pub fn new(data: Vec<T, A>, row_count: usize, col_count: usize) -> Self {
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
        return Self::new(data, row_count, col_count);
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

    /// Returns the number of columns of this matrix.
    pub fn col_count(&self) -> usize { self.col_count }

    pub fn map<F, U>(self, f: F) -> OwnedMatrix<U, A>
    where
        F: FnMut(T) -> U,
        A: Clone,
    {
        let (row_count, col_count) = (self.row_count(), self.col_count());
        let mut result = Vec::with_capacity_in(self.data.len(), self.data.allocator().clone());
        result.extend(self.data.into_iter().map(f));
        OwnedMatrix::new(result, row_count, col_count)
    }

    /// Creates the `row_count x col_count` zero matrix over the given ring.
    #[stability::unstable(feature = "enable")]
    pub fn zero_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
    where
        R::Ring: RingBase<Element = T>,
    {
        let mut result = Vec::with_capacity_in(row_count * col_count, allocator);
        for _ in 0..row_count {
            for _ in 0..col_count {
                result.push(ring.zero());
            }
        }
        return Self::new(result, row_count, col_count);
    }

    /// Creates the `row_count x col_count` identity matrix over the given ring.
    #[stability::unstable(feature = "enable")]
    pub fn identity_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
    where
        R::Ring: RingBase<Element = T>,
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
        return Self::new(result, row_count, col_count);
    }

    #[stability::unstable(feature = "enable")]
    pub fn set_row_count<F>(&mut self, new_count: usize, new_entries: F)
    where
        F: FnMut() -> T,
    {
        self.data.resize_with(new_count * self.col_count(), new_entries);
    }
}

impl<T: Debug, A: Allocator> Debug for OwnedMatrix<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result { self.data().fmt(f) }
}

impl<T: Clone, A: Allocator + Clone> Clone for OwnedMatrix<T, A> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            col_count: self.col_count,
            row_count: self.row_count,
        }
    }
}

impl<T, A: Allocator> OwnedMatrix<MaybeUninit<T>, A> {
    
    /// Initializes the elements of the matrix using the given closure.
    /// 
    /// # Panics
    /// 
    /// This function panics if the given closure did not initialize all of the matrix, i.e.
    /// returns a smaller submatrix.
    #[stability::unstable(feature = "enable")]
    pub fn init<F>(mut self, initialize: F) -> OwnedMatrix<T, A>
        where F: for<'a> FnOnce(SubmatrixMut<'a, AsFirstElement<MaybeUninit<T>>, MaybeUninit<T>>) -> SubmatrixMut<'a, AsFirstElement<MaybeUninit<T>>, T>
    {
        let raw_parts = self.data().into_raw().into_raw_parts();
        let row_count = self.data().row_count();
        let col_count = self.data().col_count();

        let witness: SubmatrixMut<_, T> = initialize(self.data_mut());
        let witness_raw_parts = witness.into_raw().into_raw_parts();

        assert_eq!(raw_parts.1, witness_raw_parts.1, "initialize returned wrong number of rows");
        assert_eq!(raw_parts.2, witness_raw_parts.2, "initialize returned wrong row stride");
        assert_eq!(raw_parts.3, witness_raw_parts.3, "initialize returned wrong column offset");
        assert_eq!(raw_parts.4, witness_raw_parts.4, "initialize returned wrong number of columns");
        if row_count != 0 && col_count != 0 {
            assert_eq!(
                raw_parts.0, witness_raw_parts.0.cast(),
                "initialize returned a pointer into a different buffer",
            );
        }

        let (ptr, length, capacity, alloc) = self.data.into_raw_parts_with_alloc();

        // SAFETY:
        // * ptr/capacity/alloc come straight from a live Vec, so they're mutually consistent.
        // * MaybeUninit<T> and T share size and alignment, so the ptr cast preserves the
        //   allocation's layout requirements for T.
        // * The elements referenced by the matrix are init T: the checks proved the caller's
        //   valid SubmatrixMut<T> covers exactly [base, base + len).
        let result_data = unsafe { Vec::from_raw_parts_in(ptr as *mut T, length, capacity, alloc) };
        return OwnedMatrix::new(result_data, row_count, col_count);
    }
} 

#[test]
fn test_zero_col_matrix() {
    feanor_tracing::DelayedLogger::init_test();
    let A: OwnedMatrix<i64> = OwnedMatrix::new(Vec::new(), 10, 0);
    assert_eq!(0, A.col_count());
    assert_eq!(10, A.row_count());

    let B: OwnedMatrix<i64> = OwnedMatrix::zero(11, 0, ZZi64);
    assert_eq!(0, B.col_count());
    assert_eq!(11, B.row_count());
}
