use std::cmp::min;

use crate::divisibility::DivisibilityRingStore;
use crate::matrix::{TransformTarget, Matrix};
use crate::ring::*;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};

///
/// Transforms `A` into `A'` via transformations `L, R` such that
/// `L A R = A'` and `A'` is diagonal.
/// 
/// # (Non-)Uniqueness of the solution
/// 
/// Note that this is not the complete Smith normal form, 
/// as that requires the entries on the diagonal to divide
/// each other. However, computing this pre-smith form is much
/// faster, and can still be used for solving equations (the main
/// use case). However, it is not unique.
/// 
/// # Warning on infinite rings (in particular Z)
/// 
/// For infinite principal ideal rings, this function is correct,
/// but in some situations, the performance can be terrible. The
/// reason is that no care is taken which of the many possible results
/// is returned - and in fact, this algorithm can sometimes choose
/// one that has exponential size in the input. Hence, in these
/// cases it is recommended to use another algorithm, e.g. based on
/// LLL to perform intermediate lattice reductions (not yet implemented
/// in feanor_math).
/// 
pub fn pre_smith<R, TL, TR>(ring: R, L: &mut TL, R: &mut TR, A: &mut DenseMatrix<R::Type>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        TL: TransformTarget<R::Type>,
        TR: TransformTarget<R::Type>
{
    // otherwise we might not terminate...
    assert!(ring.is_noetherian());
    assert!(ring.is_commutative());

    for k in 0..min(A.row_count(), A.col_count()) {
        let mut changed = true;
        while changed {
            changed = false;
            
            // eliminate the column
            for i in (k + 1)..A.row_count() {
                if ring.is_zero(A.at(i, k)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(i, k), A.at(k, k)) {
                    TransformRows(A).subtract(ring.get_ring(), k, i, &quo);
                    L.subtract(ring.get_ring(), k, i, &quo);
                } else {
                    let (s, t, d) = ring.ideal_gen(A.at(k, k), A.at(i, k));
                    let transform = [s, t, ring.negate(ring.checked_div(A.at(i, k), &d).unwrap()), ring.checked_div(A.at(k, k), &d).unwrap()];
                    TransformRows(A).transform(ring.get_ring(), k, i, &transform);
                    L.transform(ring.get_ring(), k, i, &transform);
                }
            }
            
            // now eliminate the row
            for j in (k + 1)..A.col_count() {
                if ring.is_zero(A.at(k, j)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(k, j), A.at(k, k)) {
                    changed = true;
                    TransformCols(A).subtract(ring.get_ring(), k, j, &quo);
                    R.subtract(ring.get_ring(), k, j, &quo);
                } else {
                    changed = true;
                    let (s, t, d) = ring.ideal_gen(A.at(k, k), A.at(k, j));
                    let transform = [s, t, ring.negate(ring.checked_div(A.at(k, j), &d).unwrap()), ring.checked_div(A.at(k, k), &d).unwrap()];
                    TransformCols(A).transform(ring.get_ring(), k, j, &transform);
                    R.transform(ring.get_ring(), k, j, &transform);
                }
            }
        }
    }
}

pub fn determinant<R>(A: &mut DenseMatrix<R::Type>, ring: R) -> El<R>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing
{
    assert_eq!(A.row_count(), A.col_count());
    let mut unit_part_rows = ring.one();
    let mut unit_part_cols = ring.one();
    pre_smith(ring, &mut DetUnit { current_unit: &mut unit_part_rows }, &mut DetUnit { current_unit: &mut unit_part_cols }, A);
    return ring.prod((0..A.row_count()).map(|i| ring.clone_el(A.at(i, i))).chain([unit_part_rows, unit_part_cols].into_iter()));
}

///
/// Finds a solution to the system `AX = B`, if it exists.
/// In the case that there are multiple solutions, an unspecified
/// one is returned.
/// 
pub fn solve_right<R>(A: &mut DenseMatrix<R::Type>, mut rhs: DenseMatrix<R::Type>, ring: R) -> Option<DenseMatrix<R::Type>>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing
{
    assert_eq!(A.row_count(), rhs.row_count());
    let mut R = DenseMatrix::identity(A.col_count(), ring);
    pre_smith(ring, &mut TransformRows(&mut rhs), &mut TransformCols(&mut R), A);

    // resize rhs
    for i in A.row_count()..rhs.row_count() {
        for j in 0..rhs.col_count() {
            if !ring.is_zero(rhs.at(i, j)) {
                return None;
            }
        }
    }

    let mut data = rhs.data.into_vec();
    data.resize_with(A.col_count() * rhs.col_count, || ring.zero());
    rhs.data = data.into_boxed_slice();


    let zero = ring.zero();
    for i in 0..min(A.row_count(), A.col_count()) {
        let pivot = if i < A.col_count() { A.at(i, i) } else { &zero };
        for j in 0..rhs.col_count() {
            *rhs.at_mut(i, j) = ring.checked_div(rhs.at(i, j), pivot)?;
        }
    }
    return Some(R.mul(&rhs, ring));
}

pub struct DenseMatrix<R>
    where R: ?Sized + RingBase
{
    data: Box<[R::Element]>,
    col_count: usize
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

pub struct DetUnit<'a, R: ?Sized + RingBase> {
    current_unit: &'a mut R::Element
}

impl<'a, R> TransformTarget<R> for DetUnit<'a, R>
    where R: ?Sized + RingBase
{
    fn subtract(&mut self, _ring: &R, _src: usize, _dst: usize, _factor: &<R as RingBase>::Element) {
        // determinant does not change
    }

    fn swap_rows(&mut self, ring: &R, _i: usize, _j: usize) {
        ring.negate_inplace(&mut self.current_unit)
    }

    fn transform(&mut self, ring: &R, _i: usize, _j: usize, transform: &[<R as RingBase>::Element; 4]) {
        ring.mul_assign(&mut self.current_unit, ring.sub(ring.mul_ref(&transform[0], &transform[3]), ring.mul_ref(&transform[1], &transform[2])));
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::assert_matrix_eq;

#[test]
fn test_smith_integers() {
    let ring = StaticRing::<i64>::RING;
    let mut A = DenseMatrix {
        data: vec![ 1, 2, 3, 4, 
                    2, 3, 4, 5,
                    3, 4, 5, 6 ].into_boxed_slice(),
        col_count: 4
    };
    let original_A = A.clone_matrix(&ring);
    let mut L = DenseMatrix::identity(3, StaticRing::<i64>::RING);
    let mut R = DenseMatrix::identity(4, StaticRing::<i64>::RING);
    pre_smith(ring, &mut TransformRows(&mut L), &mut TransformCols(&mut R), &mut A);
    
    assert_matrix_eq!(&ring, &[
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0, 0, 0]], &A);

    assert_matrix_eq!(&ring, &L.mul(&original_A, &ring).mul(&R, &ring), &A);
}

#[test]
fn test_smith_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let mut A = DenseMatrix {
        data: vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ].into_boxed_slice(),
        col_count: 4
    };
    let original_A = A.clone_matrix(&ring);
    let mut L = DenseMatrix::identity(5, ring);
    let mut R = DenseMatrix::identity(4, ring);
    pre_smith(ring, &mut TransformRows(&mut L), &mut TransformCols(&mut R), &mut A);

    assert_matrix_eq!(&ring, &[
        [8, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0, 15],
        [0, 0, 0, 0]], &A);
        
    assert_matrix_eq!(&ring, &L.mul(&original_A, &ring).mul(&R, &ring), &A);
}

#[test]
fn test_solve_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let A = DenseMatrix {
        data: vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ].into_boxed_slice(),
        col_count: 4
    };
    let B = DenseMatrix {
        data: vec![11, 43, 10, 22,
                   18,  9, 27, 27,
                    8, 34,  7, 22,
                   41, 13, 40, 37,
                    3,  9,  3,  0].into_boxed_slice(),
        col_count: 4
    };
    let solution = solve_right(&mut A.clone_matrix(ring), B.clone_matrix(ring), ring).unwrap();

    assert_matrix_eq!(&ring, &A.mul(&solution, &ring), &B);
}

#[test]
fn test_solve_int() {
    let ring = StaticRing::<i64>::RING;
    let A = DenseMatrix {
        data: vec![3, 6, 2, 0, 4, 7,
                   5, 5, 4, 5, 5, 5].into_boxed_slice(),
        col_count: 6
    };
    let B = DenseMatrix::identity(2, ring);
    let solution = solve_right(&mut A.clone_matrix(ring), B.clone_matrix(ring), ring).unwrap();

    assert_matrix_eq!(&ring, &A.mul(&solution, &ring), &B);
}

#[test]
fn test_large() {
    let ring = zn_static::Zn::<16>::RING;
    let data_A = [
        [0, 0, 0, 0, 0, 0, 0, 0,11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ];
    let mut A = DenseMatrix::zero(6, 11, &ring);
    for i in 0..6 {
        for j in 0..11 {
            *A.at_mut(i, j) = data_A[i][j];
        }
    }
    assert!(solve_right(&mut A.clone_matrix(&ring), A, &ring).is_some());
}

#[test]
fn test_determinant() {
    let ring = StaticRing::<i64>::RING;
    let A = DenseMatrix {
        data: vec![1, 0, 3, 
                   2, 1, 0, 
                   9, 8, 7].into_boxed_slice(),
        col_count: 3
    };
    assert_el_eq!(&ring, &(7 + 48 - 27), &determinant(&mut A.clone_matrix(&ring), &ring));
}