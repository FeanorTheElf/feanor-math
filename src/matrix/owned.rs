use std::alloc::{Allocator, Global};

use self::submatrix::{AsFirstElement, Submatrix, SubmatrixMut};

use super::*;

pub struct OwnedMatrix<T, A: Allocator = Global> {
    data: Vec<T, A>,
    col_count: usize
}

impl<T, A: Allocator> OwnedMatrix<T, A> {

    pub fn new(data: Vec<T, A>, col_count: usize) -> Self {
        assert!(data.len() % col_count == 0);
        Self { data, col_count }
    }

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

    pub fn data<'a>(&'a self) -> Submatrix<'a, AsFirstElement<T>, T> {
        Submatrix::<AsFirstElement<_>, _>::new(&self.data, self.row_count(), self.col_count())
    }

    pub fn data_mut<'a>(&'a mut self) -> SubmatrixMut<'a, AsFirstElement<T>, T> {
        let row_count = self.row_count();
        let col_count = self.col_count();
        SubmatrixMut::<AsFirstElement<_>, _>::new(&mut self.data, row_count, col_count)
    }

    pub fn at(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.col_count + j]
    }

    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.col_count + j]
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.col_count()
    }
    pub fn col_count(&self) -> usize {
        self.col_count
    }

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

impl<T, A: Allocator + Default> OwnedMatrix<T, A> {

    #[stability::unstable(feature = "enable")]
    pub fn zero<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
        where R::Type: RingBase<Element = T>
    {
        Self::zero_in(row_count, col_count, ring, A::default())
    }

    #[stability::unstable(feature = "enable")]
    pub fn identity<R: RingStore>(row_count: usize, col_count: usize, ring: R) -> Self
        where R::Type: RingBase<Element = T>
    {
        Self::identity_in(row_count, col_count, ring, A::default())
    }
}