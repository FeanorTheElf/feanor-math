use std::fmt::Display;

use crate::ring::*;

pub trait Matrix<R>
    where R: RingStore
{
    fn row_count(&self) -> usize;
    fn col_count(&self) -> usize;
    fn at(&self, i: usize, j: usize) -> &El<R>;

    fn format<'a>(&'a self, ring: &'a R) -> MatrixDisplayWrapper<'a, R, Self> {
        MatrixDisplayWrapper {
            matrix: self,
            ring: ring
        }
    }
}

pub struct MatrixDisplayWrapper<'a, R, M: ?Sized>
    where R: RingStore, M: Matrix<R>
{
    matrix: &'a M,
    ring: &'a R
}

impl<'a, R, M: ?Sized> Display for MatrixDisplayWrapper<'a, R, M>
    where R: RingStore, M: Matrix<R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strings = (0..self.matrix.row_count()).flat_map(|i| (0..self.matrix.col_count()).map(move |j| (i, j)))
            .map(|(i, j)| format!("{}", self.ring.format(self.matrix.at(i, j))))
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