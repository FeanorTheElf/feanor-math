use crate::ring::*;

use self::submatrix::{AsFirstElement, Submatrix, SubmatrixMut};

use super::*;

pub struct DenseMatrix<R>
    where R: ?Sized + RingBase
{
    data: Box<[R::Element]>,
    col_count: usize
}

impl<R> DenseMatrix<R>
    where R: ?Sized + RingBase
{
    pub fn new(data: Box<[R::Element]>, col_count: usize) -> Self {
        assert!(data.len() % col_count == 0);
        Self { data, col_count }
    }

    pub fn data<'a>(&'a self) -> Submatrix<'a, AsFirstElement<R::Element>, R::Element> {
        Submatrix::<AsFirstElement<_>, _>::new(&self.data, self.row_count(), self.col_count())
    }

    pub fn data_mut<'a>(&'a mut self) -> SubmatrixMut<'a, AsFirstElement<R::Element>, R::Element> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        SubmatrixMut::<AsFirstElement<_>, _>::new(&mut self.data, row_count, col_count)
    }
}

impl<R> DenseMatrix<R>
    where R: ?Sized + RingBase
{
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut R::Element {
        &mut self.data[i * self.col_count + j]
    }

    pub fn identity<S>(n: usize, ring: S) -> Self
        where S: RingStore<Type = R>
    {
        let mut result = DenseMatrix {
            data: (0..(n * n)).map(|_| ring.zero()).collect::<Vec<_>>().into_boxed_slice(),
            col_count: n
        };
        for i in 0..n {
            result.data[i * n + i] = ring.one();
        }
        return result;
    }

    pub fn zero<S>(n: usize, m: usize, ring: S) -> Self
        where S: RingStore<Type = R>
    {
        let result = DenseMatrix {
            data: (0..(n * m)).map(|_| ring.zero()).collect::<Vec<_>>().into_boxed_slice(),
            col_count: m
        };
        return result;
    }

    pub fn mul<S>(&self, other: &DenseMatrix<R>, ring: S) -> Self
        where S: RingStore<Type = R> + Copy
    {
        assert_eq!(self.col_count(), other.row_count());
        DenseMatrix {
            col_count: other.col_count,
            data: (0..self.row_count()).flat_map(|i| (0..other.col_count()).map(move |j|
                ring.sum((0..self.col_count()).map(|k| ring.mul_ref(self.at(i, k), other.at(k, j))))
            )).collect::<Vec<_>>().into_boxed_slice()
        }
    }

    pub fn clone_matrix<S>(&self, ring: S) -> Self
        where S: RingStore<Type = R>
    {
        DenseMatrix {
            col_count: self.col_count,
            data: self.data.iter().map(|x| ring.clone_el(x)).collect::<Vec<_>>().into_boxed_slice()
        }
    }

    pub fn set_row_count<S>(&mut self, new_count: usize, ring: S)
        where S: RingStore<Type = R>
    {
        let mut new_data = std::mem::replace(&mut self.data, Box::new([])).into_vec();
        new_data.resize_with(new_count * self.col_count(), || ring.zero());
        self.data = new_data.into_boxed_slice();
    }
}

impl<R> Matrix<R> for DenseMatrix<R>
    where R: ?Sized + RingBase
{
    fn row_count(&self) -> usize {
        self.data.len() / self.col_count
    }

    fn col_count(&self) -> usize {
        self.col_count
    }

    fn at(&self, i: usize, j: usize) -> &R::Element {
        &self.data[i * self.col_count + j]
    }
}

pub struct TransformRows<'a, R>(pub &'a mut DenseMatrix<R>)
    where R: ?Sized + RingBase;

pub struct TransformCols<'a, R>(pub &'a mut DenseMatrix<R>)
    where R: ?Sized + RingBase;

impl<'a, R> TransformTarget<R> for TransformRows<'a, R>
    where R: ?Sized + RingBase
{
    fn transform(&mut self, ring: &R, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let A = &mut *self.0;
        for l in 0..A.col_count() {
            let (new_i, new_j) = (
                ring.add(ring.mul_ref(A.at(i, l), &transform[0]), ring.mul_ref(A.at(j, l), &transform[1])),
                ring.add(ring.mul_ref(A.at(i, l), &transform[2]), ring.mul_ref(A.at(j, l), &transform[3]))
            );
            *A.at_mut(i, l) = new_i;
            *A.at_mut(j, l) = new_j;
        }
    }

    fn subtract(&mut self, ring: &R, src: usize, dst: usize, factor: &<R as RingBase>::Element) {
        let A = &mut *self.0;
        for j in 0..A.col_count() {
            let to_sub = ring.mul_ref(factor, A.at(src, j));
            ring.sub_assign(A.at_mut(dst, j), to_sub);
        }
    }
}

impl<'a, R> TransformTarget<R> for TransformCols<'a, R>
    where R: ?Sized + RingBase
{
    fn transform(&mut self, ring: &R, i: usize, j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let A = &mut *self.0;
        for l in 0..A.row_count() {
            let (new_i, new_j) = (
                ring.add(ring.mul_ref(A.at(l, i), &transform[0]), ring.mul_ref(A.at(l, j), &transform[1])),
                ring.add(ring.mul_ref(A.at(l, i), &transform[2]), ring.mul_ref(A.at(l, j), &transform[3]))
            );
            *A.at_mut(l, i) = new_i;
            *A.at_mut(l, j) = new_j;
        }
    }

    fn subtract(&mut self, ring: &R, src: usize, dst: usize, factor: &<R as RingBase>::Element) {
        let A = &mut *self.0;
        for i in 0..A.row_count() {
            let to_sub = ring.mul_ref(factor, A.at(i, src));
            ring.sub_assign(A.at_mut(i, dst), to_sub);
        }
    }
}