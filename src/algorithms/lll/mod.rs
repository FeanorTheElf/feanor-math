
///
/// Contains [`float::lll_quadratic_form()`] and [`float::lll()`], an implementation
/// of the Lenstra-Lenstra-Lovasz lattice basis reduction algorithm, using floating point numbers.
/// 
pub mod float;

///
/// Contains [`exact::lll()`], an implementation of the Lenstra-Lenstra-Lovasz lattice basis
/// reduction algorithm, using arbitrary-precision arithmetic.
/// 
pub mod exact;

#[cfg(test)]
use crate::integer::IntegerRing;
#[cfg(test)]
use crate::matrix::*;
#[cfg(test)]
use crate::primitive_int::StaticRing;
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
fn assert_lattice_isomorphic<R, S, V1, V2>(small_ring: R, large_ring: S, lhs: Submatrix<V1, El<R>>, rhs: Submatrix<V2, El<R>>)
    where V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>,
        R: RingStore,
        S: RingStore,
        S::Type: IntegerRing + CanHomFrom<R::Type>
{
    use std::alloc::Global;
    use crate::algorithms::linsolve::smith::solve_right_using_pre_smith;

    let n = lhs.row_count();
    assert_eq!(n, rhs.row_count());
    let hom = large_ring.can_hom(&small_ring).unwrap();
    let mut A: OwnedMatrix<_> = OwnedMatrix::zero(n, lhs.col_count(), &large_ring);
    let mut B: OwnedMatrix<_> = OwnedMatrix::zero(n, rhs.col_count(), &large_ring);
    for i in 0..n {
        for j in 0..lhs.col_count() {
            *A.at_mut(i, j) = hom.map_ref(lhs.at(i, j));
        }
        for j in 0..rhs.col_count() {
            *B.at_mut(i, j) = hom.map_ref(rhs.at(i, j));
        }
    }
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(lhs.col_count(), rhs.col_count(), &large_ring);
    assert!(solve_right_using_pre_smith(&large_ring, A.clone_matrix(&large_ring).data_mut(), B.clone_matrix(&large_ring).data_mut(), U.data_mut(), Global).is_solved());
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(rhs.col_count(), lhs.col_count(), &large_ring);
    assert!(solve_right_using_pre_smith(&large_ring, B.clone_matrix(&large_ring).data_mut(), A.clone_matrix(&large_ring).data_mut(), U.data_mut(), Global).is_solved());
}

#[cfg(test)]
fn assert_rational_lattice_isomorphic<R, S, I, V1, V2>(small_ring: R, large_ring: S, lhs: Submatrix<V1, El<R>>, rhs: Submatrix<V2, El<R>>)
    where V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>,
        R: RingStore,
        S: RingStore<Type = RationalFieldBase<I>>,
        S::Type: CanHomFrom<R::Type>,
        I: RingStore,
        I::Type: IntegerRing
{
    use std::alloc::Global;

    use crate::divisibility::DivisibilityRingStore;
    use crate::algorithms::linsolve::smith::solve_right_using_pre_smith;
    use crate::pid::PrincipalIdealRingStore;

    let n = lhs.row_count();
    assert_eq!(n, rhs.row_count());
    let hom = large_ring.can_hom(&small_ring).unwrap();
    let den_lcm = lhs.row_iter().chain(rhs.row_iter()).flat_map(|r| r.iter()).fold(large_ring.base_ring().one(), |a, b| 
        large_ring.base_ring().ideal_intersect(&a, &large_ring.get_ring().den(&hom.map_ref(b)))
    );
    let mut A: OwnedMatrix<_> = OwnedMatrix::zero(n, lhs.col_count(), large_ring.base_ring());
    let mut B: OwnedMatrix<_> = OwnedMatrix::zero(n, rhs.col_count(), large_ring.base_ring());
    for i in 0..n {
        for j in 0..lhs.col_count() {
            let value = large_ring.inclusion().mul_ref_snd_map(hom.map_ref(lhs.at(i, j)), &den_lcm);
            *A.at_mut(i, j) = large_ring.base_ring().checked_div(large_ring.get_ring().num(&value), large_ring.get_ring().den(&value)).unwrap();
        }
        for j in 0..rhs.col_count() {
            let value = large_ring.inclusion().mul_ref_snd_map(hom.map_ref(rhs.at(i, j)), &den_lcm);
            *B.at_mut(i, j) = large_ring.base_ring().checked_div(large_ring.get_ring().num(&value), large_ring.get_ring().den(&value)).unwrap();
        }
    }
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(lhs.col_count(), rhs.col_count(), large_ring.base_ring());
    assert!(solve_right_using_pre_smith(large_ring.base_ring(), A.clone_matrix(large_ring.base_ring()).data_mut(), B.clone_matrix(large_ring.base_ring()).data_mut(), U.data_mut(), Global).is_solved());
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(rhs.col_count(), lhs.col_count(), large_ring.base_ring());
    assert!(solve_right_using_pre_smith(large_ring.base_ring(), B.clone_matrix(large_ring.base_ring()).data_mut(), A.clone_matrix(large_ring.base_ring()).data_mut(), U.data_mut(), Global).is_solved());
}

#[test]
fn test_assert_lattice_isomorphic() {
    let ZZ = StaticRing::<i64>::RING;

    let lhs = [
        DerefArray::from([1, 2, 0]),
        DerefArray::from([1, 2, 4]),
    ];
    let rhs = [
        DerefArray::from([1, -2]),
        DerefArray::from([1, 2])
    ];
    assert_lattice_isomorphic(ZZ, ZZ, Submatrix::from_2d(&lhs), Submatrix::from_2d(&rhs));
}

#[test]
#[should_panic]
fn test_assert_lattice_not_isomorphic() {
    let ZZ = StaticRing::<i64>::RING;

    let lhs = [
        DerefArray::from([1, 2, 0]),
        DerefArray::from([1, 3, 4]),
    ];
    let rhs = [
        DerefArray::from([1, -2]),
        DerefArray::from([1, 2])
    ];
    assert_lattice_isomorphic(ZZ, ZZ, Submatrix::from_2d(&lhs), Submatrix::from_2d(&rhs));
}