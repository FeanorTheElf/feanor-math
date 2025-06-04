use std::alloc::{Allocator, Global};

use self::submatrix::{AsFirstElement, Submatrix, SubmatrixMut};

use super::*;

///
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
/// StaticRing::<i32>::RING.get_ring().solve_right(A.data_mut(), B.data_mut(), C.data_mut(), Global).assert_solved();
/// ```
/// 
pub struct OwnedMatrix<T, A: Allocator = Global> {
    data: Vec<T, A>,
    col_count: usize
}

impl<T> OwnedMatrix<T> {

    ///
    /// Creates the `row_count x col_count` [`OwnedMatrix`] whose `(i, j)`-th entry
    /// is the output of the given function on `(i, j)`.
    /// 
    pub fn from_fn<F>(row_count: usize, col_count: usize, f: F) -> Self
        where F: FnMut(usize, usize) -> T
    {
        Self::from_fn_in(row_count, col_count, f, Global)
    }
    
    ///
    /// Creates the `row_count x col_count` zero matrix over the given ring.
    /// 
    pub fn zero<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
        where R::Type: RingBase<Element = T>
    {
        Self::zero_in(row_count, col_count, ring, Global)
    }

    ///
    /// Creates the `row_count x col_count` identity matrix over the given ring.
    /// 
    pub fn identity<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
        where R::Type: RingBase<Element = T>
    {
        Self::identity_in(row_count, col_count, ring, Global)
    }
}

impl<T, A: Allocator> OwnedMatrix<T, A> {

    ///
    /// Creates the `row_count x col_count` [`OwnedMatrix`] matrix, whose entries are
    /// taken from the given vector, interpreted as a row-major matrix. The number of
    /// rows is `row_count = data.len() / col_count`.
    /// 
    pub fn new(data: Vec<T, A>, col_count: usize) -> Self {
        assert!(data.len() % col_count == 0);
        Self { data, col_count }
    }

    ///
    /// Creates the `row_count x col_count` [`OwnedMatrix`] whose `(i, j)`-th entry
    /// is the output of the given function on `(i, j)`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn from_fn_in<F>(row_count: usize, col_count: usize, mut f: F, allocator: A) -> Self
        where F: FnMut(usize, usize) -> T
    {
        let mut data = Vec::with_capacity_in(row_count * col_count, allocator);
        for i in 0..row_count {
            for j in 0..col_count {
                data.push(f(i, j));
            }
        }
        return Self::new(data, col_count);
    }

    ///
    /// Returns a [`Submatrix`] view on the data of this matrix.
    /// 
    pub fn data<'a>(&'a self) -> Submatrix<'a, AsFirstElement<T>, T> {
        Submatrix::<AsFirstElement<_>, _>::from_1d(&self.data, self.row_count(), self.col_count())
    }

    ///
    /// Returns a [`SubmatrixMut`] view on the data of this matrix.
    /// 
    pub fn data_mut<'a>(&'a mut self) -> SubmatrixMut<'a, AsFirstElement<T>, T> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        SubmatrixMut::<AsFirstElement<_>, _>::from_1d(&mut self.data, row_count, col_count)
    }

    ///
    /// Returns a reference to the `(i, j)`-th entry of this matrix.
    /// 
    pub fn at(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.col_count + j]
    }

    ///
    /// Returns a mutable reference to the `(i, j)`-th entry of this matrix.
    /// 
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.col_count + j]
    }

    ///
    /// Returns the number of rows of this matrix.
    /// 
    pub fn row_count(&self) -> usize {
        self.data.len() / self.col_count()
    }
    
    ////
    /// Returns the number of columns of this matrix.
    /// 
    pub fn col_count(&self) -> usize {
        self.col_count
    }

    ///
    /// Creates the `row_count x col_count` zero matrix over the given ring.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn zero_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
        where R::Type: RingBase<Element = T>
    {
        let mut result = Vec::with_capacity_in(row_count * col_count, allocator);
        for _ in 0..row_count {
            for _ in 0..col_count {
                result.push(ring.zero());
            }
        }
        return Self::new(result, col_count);
    }

    ///
    /// Creates the `row_count x col_count` identity matrix over the given ring.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn identity_in<R: RingStore>(row_count: usize, col_count: usize, ring: R, allocator: A) -> Self
        where R::Type: RingBase<Element = T>
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
        return Self::new(result, col_count);
    }

    #[stability::unstable(feature = "enable")]
    pub fn clone_matrix<R: RingStore>(&self, ring: R) -> Self
        where R::Type: RingBase<Element = T>,
            A: Clone
    {
        let mut result = Vec::with_capacity_in(self.row_count() * self.col_count(), self.data.allocator().clone());
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                result.push(ring.clone_el(self.at(i, j)));
            }
        }
        return Self::new(result, self.col_count());
    }

    #[stability::unstable(feature = "enable")]
    pub fn set_row_count<F>(&mut self, new_count: usize, new_entries: F)
        where F: FnMut() -> T
    {
        self.data.resize_with(new_count * self.col_count(), new_entries);
    }
}
