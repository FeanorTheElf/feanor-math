use std::fmt::Display;

pub mod subslice;

use crate::ring::*;

pub trait Matrix<R>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn at(&self, i: usize, j: usize) -> &R::Element;

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
        (0..self.row_count()).all(|i| (0..self.col_count()).all(|j| ring.eq_el(self.at(i, j), other.at(i, j))))
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
            .map(|(i, j)| format!("{}", ring.format(self.matrix.at(i, j))))
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

pub trait TransformTarget<R>
    where R: ?Sized + RingBase
{
    fn transform(&mut self, ring: &R, i: usize, j: usize, transform: &[R::Element; 4]);

    fn subtract(&mut self, ring: &R, src: usize, dst: usize, factor: &R::Element) {
        self.transform(ring, src, dst, &[ring.one(), ring.zero(), ring.negate(ring.clone_el(factor)), ring.one()])
    }

    fn swap_rows(&mut self, ring: &R, i: usize, j: usize) {
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

    fn at(&self, i: usize, j: usize) -> &<R as RingBase>::Element {
        &self[i][j]
    }
}