use std::cmp::min;
use std::fmt::Debug;

use crate::divisibility::DivisibilityRingStore;
use crate::ring::*;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};

fn sub_row<R>(ring: &R, A: &mut Matrix<El<R>>, k: usize, i: usize, factor: &El<R>)
    where R: RingStore
{
    for j in 0..A.col_count() {
        let to_sub = ring.mul_ref(factor, A.at(k, j));
        ring.sub_assign(A.at_mut(i, j), to_sub);
    }
}

fn sub_col<R>(ring: &R, A: &mut Matrix<El<R>>, k: usize, j: usize, factor: &El<R>)
    where R: RingStore
{
    for i in 0..A.row_count() {
        let to_sub = ring.mul_ref(factor, A.at(i, k));
        ring.sub_assign(A.at_mut(i, j), to_sub);
    }
}

fn transform_rows<R>(ring: &R, A: &mut Matrix<El<R>>, k: usize, i: usize, transform: &[El<R>; 4])
    where R: RingStore
{
    for j in 0..A.col_count() {
        let (new_k, new_i) = (
            ring.add(ring.mul_ref(A.at(k, j), &transform[0]), ring.mul_ref(A.at(i, j), &transform[1])),
            ring.add(ring.mul_ref(A.at(k, j), &transform[2]), ring.mul_ref(A.at(i, j), &transform[3]))
        );
        *A.at_mut(k, j) = new_k;
        *A.at_mut(i, j) = new_i;
    }
}

fn transform_cols<R>(ring: &R, A: &mut Matrix<El<R>>, k: usize, j: usize, transform: &[El<R>; 4])
    where R: RingStore
{
    for i in 0..A.row_count() {
        let (new_k, new_j) = (
            ring.add(ring.mul_ref(A.at(i, k), &transform[0]), ring.mul_ref(A.at(i, j), &transform[1])),
            ring.add(ring.mul_ref(A.at(i, k), &transform[2]), ring.mul_ref(A.at(i, j), &transform[3]))
        );
        *A.at_mut(i, k) = new_k;
        *A.at_mut(i, j) = new_j;
    }
}

///
/// Transforms `(L, R, A)` into `(L', R', A')` such that
/// `L' A R' = L A' R` and `A'` is in Smith normal form.
/// 
pub fn smith_like<R>(ring: R, L: &mut Matrix<El<R>>, R: &mut Matrix<El<R>>, A: &mut Matrix<El<R>>)
    where R: RingStore,
        R::Type: PrincipalIdealRing
{
    // otherwise we might not terminate...
    assert!(ring.is_noetherian());
    assert!(ring.is_commutative());

    for k in 0..min(A.row_count(), A.col_count()) {
        // eliminate the column
        for i in (k + 1)..A.row_count() {
            if let Some(quo) = ring.checked_div(A.at(i, k), A.at(k, k)) {
                sub_row(&ring, A, k, i, &quo);
                sub_row(&ring, L, k, i, &quo);
            } else {
                let (s, t, d) = ring.ideal_gen(A.at(k, k), A.at(i, k));
                println!("{}, {}, {}, {}, {}", ring.format(A.at(k, k)), ring.format(A.at(i, k)), ring.format(&s), ring.format(&t), ring.format(&d));
                let transform = [s, t, ring.negate(ring.checked_div(A.at(i, k), &d).unwrap()), ring.checked_div(A.at(k, k), &d).unwrap()];
                transform_rows(&ring, A, k, i, &transform);
                transform_rows(&ring, L, k, i, &transform);
            }
        }
        
        // now eliminate the row
        for j in (k + 1)..A.col_count() {
            if let Some(quo) = ring.checked_div(A.at(k, j), A.at(k, k)) {
                sub_col(&ring, A, k, j, &quo);
                sub_col(&ring, R, k, j, &quo);
            } else {
                let (s, t, d) = ring.ideal_gen(A.at(k, k), A.at(k, j));
                let transform = [s, t, ring.negate(ring.checked_div(A.at(k, j), &d).unwrap()), ring.checked_div(A.at(k, k), &d).unwrap()];
                transform_cols(&ring, A, k, j, &transform);
                transform_cols(&ring, R, k, j, &transform);
            }
        }
    }
}

pub struct Matrix<T> {
    data: Box<[T]>,
    col_count: usize
}

impl<T> Matrix<T> {

    fn col_count(&self) -> usize {
        self.col_count
    }

    fn row_count(&self) -> usize {
        self.data.len() / self.col_count
    }

    fn at(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.col_count + j]
    }

    fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.col_count + j]
    }

    #[cfg(test)]
    fn identity(n: usize, zero: T, one: T) -> Self
        where T: Clone
    {
        let mut result = Matrix {
            data: (0..(n * n)).map(|_| zero.clone()).collect::<Vec<_>>().into_boxed_slice(),
            col_count: n
        };
        for i in 0..n {
            result.data[i * n + i] = one.clone();
        }
        return result;
    }
}

impl<T> PartialEq for Matrix<T>
    where T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                if self.at(i, j) != other.at(i, j) {
                    return false;
                }
            }
        }
        return true;
    }
}

impl<T> Debug for Matrix<T>
    where T: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                write!(f, "{:?}, ", self.at(i, j))?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
use crate::rings::zn::zn_static;

#[test]
fn test_smith_integers() {
    let ring = StaticRing::<i64>::RING;
    let mut A = Matrix {
        data: vec![ 1, 2, 3, 4, 
                    2, 3, 4, 5,
                    3, 4, 5, 6 ].into_boxed_slice(),
        col_count: 4
    };
    let mut L = Matrix::identity(3, 0, 1);
    let mut R = Matrix::identity(4, 0, 1);
    smith_like(ring, &mut L, &mut R, &mut A);
    
    let expected = Matrix {
        data: vec![ 1,  0, 0, 0,
                    0, -1, 0, 0,
                    0,  0, 0, 0 ].into_boxed_slice(),
        col_count: 4
    };
    assert_eq!(&expected, &A);
}

#[test]
fn test_smith_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let mut A = Matrix {
        data: vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ].into_boxed_slice(),
        col_count: 4
    };
    let mut L = Matrix::identity(5, ring.zero(), ring.one());
    let mut R = Matrix::identity(4, ring.zero(), ring.one());
    smith_like(ring, &mut L, &mut R, &mut A);

    let expected = Matrix {
        data: vec![ 3, 0, 0, 0,
                    0, 9, 0, 0,
                    0, 0, 5, 0, 
                    0, 0, 0, 15,
                    0, 0, 0, 0 ].into_boxed_slice(),
        col_count: 4
    };
    assert_eq!(&expected, &A);
}