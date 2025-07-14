
pub mod float;

pub mod exact;

#[cfg(test)]
use crate::integer::IntegerRing;
#[cfg(test)]
use crate::matrix::*;
#[cfg(test)]
use crate::ring::*;
#[cfg(test)]
use crate::homomorphism::*;
#[cfg(test)]
use crate::rings::rational::RationalFieldBase;
#[cfg(test)]
use crate::seq::*;

#[cfg(test)]
fn norm_squared<I, V>(ring: I, col: &Column<V, El<I>>) -> El<I>
    where V: AsPointerToSlice<El<I>>,
        I: RingStore
{
    <_ as RingStore>::sum(&ring, (0..col.len()).map(|i| ring.mul_ref(col.at(i), col.at(i))))
}

#[cfg(test)]
fn assert_lattice_isomorphic<R, S, V1, V2>(small_ring: R, large_ring: S, lhs: Submatrix<V1, El<R>>, rhs: &Submatrix<V2, El<R>>)
    where V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>,
        R: RingStore,
        S: RingStore,
        S::Type: IntegerRing + CanHomFrom<R::Type>
{
    use std::alloc::Global;
    use crate::algorithms::linsolve::smith;

    let n = lhs.row_count();
    assert_eq!(n, rhs.row_count());
    let m = lhs.col_count();
    assert_eq!(m, rhs.col_count());
    let hom = large_ring.can_hom(&small_ring).unwrap();
    let mut A: OwnedMatrix<_> = OwnedMatrix::zero(n, m, &large_ring);
    let mut B: OwnedMatrix<_> = OwnedMatrix::zero(n, m, &large_ring);
    for i in 0..n {
        for j in 0..m {
            *A.at_mut(i, j) = hom.map_ref(lhs.at(i, j));
            *B.at_mut(i, j) = hom.map_ref(rhs.at(i, j));
        }
    }
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(n, m, &large_ring);
    assert!(smith::solve_right_using_pre_smith(&large_ring, A.clone_matrix(&large_ring).data_mut(), B.clone_matrix(&large_ring).data_mut(), U.data_mut(), Global).is_solved());
    assert!(smith::solve_right_using_pre_smith(&large_ring, B.clone_matrix(&large_ring).data_mut(), A.clone_matrix(&large_ring).data_mut(), U.data_mut(), Global).is_solved());
}

#[cfg(test)]
fn assert_rational_lattice_isomorphic<R, S, I, V1, V2>(small_ring: R, large_ring: S, lhs: Submatrix<V1, El<R>>, rhs: &Submatrix<V2, El<R>>)
    where V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>,
        R: RingStore,
        S: RingStore<Type = RationalFieldBase<I>>,
        S::Type: CanHomFrom<R::Type>,
        I: RingStore,
        I::Type: IntegerRing
{
    unimplemented!()
}
