use std::fmt::Display;

use submatrix::DerefArray;

use crate::{homomorphism::Homomorphism, ring::*};

use self::submatrix::{AsPointerToSlice, SubmatrixMut};

///
/// Functionality for providing mutable and immutable views on submatrices
/// of `Vec<Vec<T>>` or similar owned 2d-data structures. 
/// 
/// In particular, this enables to have simultaneous mutable references to disjoint 
/// parts of a matrix, which is very useful in algorithms, but can only be
/// achieved using unsafe code. This module provides a safe wrapper around that.
/// 
pub mod submatrix;
pub mod dense;

///
/// A very minimalistic approach to implement matrices.
/// 
/// I have not yet decided on how exactly this library should "do" matrices.
/// One problem is that efficient algorithms on matrices often need their own
/// layout of data - e.g. sparse vs dense, column-vs row-major storage, maybe
/// support for submatrices is needed, ...
/// 
/// Hence, for now, we just have this trait to support writing and equality
/// checking, which should at least vastly simplify testing matrix implementations
/// and algorithms.
/// 
/// Also, note that this is not a library for numeric computations, we will never
/// get linear algebra operations that efficiently work on floating point numbers.
/// Use e.g. `ndarray` for that.
/// 
pub trait Matrix<R>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn entry_at(&self, i: usize, j: usize) -> &R::Element;

    fn format<'a, S>(&'a self, ring: &'a S) -> MatrixDisplayWrapper<'a, R, Self>
        where S: RingStore<Type = R>
    {
        MatrixDisplayWrapper {
            matrix: self,
            ring: ring.get_ring()
        }
    }

    fn matrix_eq<M, S>(&self, other: &M, ring: S) -> bool
        where M: Matrix<R>, S: RingStore<Type = R>
    {
        assert_eq!(self.row_count(), other.row_count());
        assert_eq!(self.col_count(), other.col_count());
        (0..self.row_count()).all(|i| (0..self.col_count()).all(|j| ring.eq_el(self.entry_at(i, j), other.entry_at(i, j))))
    }
}

#[macro_export]
macro_rules! assert_matrix_eq {
    ($ring:expr, $lhs:expr, $rhs:expr) => {
        match ($ring, $lhs, $rhs) {
            (ring_val, lhs_val, rhs_val) => {
                assert!(<_ as $crate::matrix::Matrix<_>>::matrix_eq(lhs_val, rhs_val, ring_val), "Assertion failed: Expected\n{}\nto be\n{}", <_ as $crate::matrix::Matrix<_>>::format(lhs_val, ring_val), <_ as $crate::matrix::Matrix<_>>::format(rhs_val, ring_val));
            }
        }
    }
}

///
/// A wrapper for a reference to [`Matrix`] that implements [`std::fmt::Display`] to write the matrix.
/// 
pub struct MatrixDisplayWrapper<'a, R, M>
    where R: ?Sized + RingBase, M: ?Sized + Matrix<R>
{
    matrix: &'a M,
    ring: &'a R
}

impl<'a, R, M> Display for MatrixDisplayWrapper<'a, R, M>
    where R: ?Sized + RingBase, M: ?Sized + Matrix<R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ring = RingRef::new(self.ring);
        let strings = (0..self.matrix.row_count()).flat_map(|i| (0..self.matrix.col_count()).map(move |j| (i, j)))
            .map(|(i, j)| format!("{}", ring.format(self.matrix.entry_at(i, j))))
            .collect::<Vec<_>>();
        let max_len = strings.iter().map(|s| s.chars().count()).chain([2].into_iter()).max().unwrap();
        let mut strings = strings.into_iter();
        for i in 0..self.matrix.row_count() {
            write!(f, "|")?;
            if self.matrix.col_count() > 0 {
                write!(f, "{:>width$}", strings.next().unwrap(), width = max_len)?;
            }
            for _ in 1..self.matrix.col_count() {
                write!(f, ",{:>width$}", strings.next().unwrap(), width = max_len)?;
            }
            if i + 1 != self.matrix.row_count() {
                writeln!(f, "|")?;
            } else {
                write!(f, "|")?;
            }
        }
        return Ok(());
    }
}

///
/// A trait for a "target" that can "consume" elementary operations on matrices.
/// 
/// This is mainly used during algorithms that work on matrices, since in many cases
/// they transform matrices using elementary row or column operations, and have to
/// accumulate data depending on these operations.
/// 
pub trait TransformTarget<R>
    where R: ?Sized + RingBase
{
    ///
    /// The transformation given by the matrix `A` with `A[k, l]` being
    ///  - `1` if `k = l` and `k != i, j`
    ///  - `transform[0]` if `(k, l) = (i, i)`
    ///  - `transform[1]` if `(k, l) = (i, j)`
    ///  - `transform[2]` if `(k, l) = (j, i)`
    ///  - `transform[3]` if `(k, l) = (j, j)`
    ///  - `0` otherwise
    /// 
    /// In other words, the matrix looks like
    /// ```text
    /// | 1  ...  0                       |
    /// | ⋮        ⋮                       |
    /// | 0  ...  1                       |
    /// |    A             B              | <- i-th row
    /// |            1  ...  0            |
    /// |            ⋮        ⋮            |
    /// |            0  ...  1            |
    /// |    C             D              | <- j-th row
    /// |                       1  ...  0 |
    /// |                       ⋮        ⋮ |
    /// |                       0  ...  1 |
    ///      ^ i-th col    ^ j-th col
    /// ```
    /// where `transform = [A, B, C, D]`.
    /// 
    fn transform(&mut self, ring: &R, i: usize, j: usize, transform: &[R::Element; 4]);

    ///
    /// The transformation corresponding to subtracting `factor` times the `src`-th row
    /// from the `dst`-th row.
    /// 
    fn subtract(&mut self, ring: &R, src: usize, dst: usize, factor: &R::Element) {
        self.transform(ring, src, dst, &[ring.one(), ring.zero(), ring.negate(ring.clone_el(factor)), ring.one()])
    }

    fn swap(&mut self, ring: &R, i: usize, j: usize) {
        self.transform(ring, i, j, &[ring.zero(), ring.one(), ring.one(), ring.zero()])
    }
}

impl<const N: usize, const M: usize, R> Matrix<R> for [[R::Element; N]; M]
    where R: ?Sized + RingBase
{
    fn col_count(&self) -> usize {
        N
    }

    fn row_count(&self) -> usize {
        M
    }

    fn entry_at(&self, i: usize, j: usize) -> &<R as RingBase>::Element {
        &self[i][j]
    }
}

impl<const N: usize, const M: usize, R> Matrix<R> for [DerefArray<R::Element, N>; M]
    where R: ?Sized + RingBase
{
    fn col_count(&self) -> usize {
        N
    }

    fn row_count(&self) -> usize {
        M
    }

    fn entry_at(&self, i: usize, j: usize) -> &<R as RingBase>::Element {
        &self[i][j]
    }
}

pub fn matmul<M1, M2, R1, R2, S, V, H1, H2>(lhs: &M1, rhs: &M2, mut out: SubmatrixMut<V, S::Element>, hom1: &H1, hom2: &H2)
    where V: AsPointerToSlice<S::Element>,
        R1: ?Sized + RingBase,
        R2: ?Sized + RingBase,
        S: ?Sized + RingBase,
        M1: Matrix<R1>,
        M2: Matrix<R2>,
        H1: Homomorphism<R1, S>,
        H2: Homomorphism<R2, S>
{
    assert!(hom1.codomain().get_ring() == hom2.codomain().get_ring());
    assert_eq!(lhs.col_count(), rhs.row_count());
    assert_eq!(lhs.row_count(), out.row_count());
    assert_eq!(rhs.col_count(), out.col_count());
    let target_ring = hom1.codomain();
    for i in 0..out.row_count() {
        for j in 0..out.col_count() {
            *<SubmatrixMut<V, S::Element>>::at(&mut out, i, j) = target_ring.sum((0..lhs.col_count()).map(|k| target_ring.mul(hom1.map_ref(lhs.entry_at(i, k)), hom2.map_ref(rhs.entry_at(k, j)))));
        }
    }
}

pub fn matmul_fst_transposed<M1, M2, R1, R2, S, V, H1, H2>(lhs_T: &M1, rhs: &M2, mut out: SubmatrixMut<V, S::Element>, hom1: &H1, hom2: &H2)
    where V: AsPointerToSlice<S::Element>,
        R1: ?Sized + RingBase,
        R2: ?Sized + RingBase,
        S: ?Sized + RingBase,
        M1: Matrix<R1>,
        M2: Matrix<R2>,
        H1: Homomorphism<R1, S>,
        H2: Homomorphism<R2, S>
{
    assert!(hom1.codomain().get_ring() == hom2.codomain().get_ring());
    assert_eq!(lhs_T.row_count(), rhs.row_count());
    assert_eq!(lhs_T.col_count(), out.row_count());
    assert_eq!(rhs.col_count(), out.col_count());
    let target_ring = hom1.codomain();
    for i in 0..out.row_count() {
        for j in 0..out.col_count() {
            *<SubmatrixMut<V, S::Element>>::at(&mut out, i, j) = target_ring.sum((0..lhs_T.row_count()).map(|k| target_ring.mul(hom1.map_ref(lhs_T.entry_at(k, i)), hom2.map_ref(rhs.entry_at(k, j)))));
        }
    }
}