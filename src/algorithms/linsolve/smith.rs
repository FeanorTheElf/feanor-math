use std::alloc::Allocator;
use std::cmp::min;

use crate::algorithms::linsolve::SolveResult;
use crate::divisibility::*;
use crate::matrix::*;
use crate::matrix::transform::{TransformCols, TransformRows, TransformTarget};
use crate::ring::*;
use crate::pid::PrincipalIdealRing;

use transform::TransformList;

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
#[stability::unstable(feature = "enable")]
pub fn pre_smith<R, TL, TR, V>(ring: R, L: &mut TL, R: &mut TR, mut A: SubmatrixMut<V, El<R>>)
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        TL: TransformTarget<R::Type>,
        TR: TransformTarget<R::Type>,
        V: AsPointerToSlice<El<R>>
{
    // otherwise we might not terminate...
    assert!(ring.is_noetherian());
    assert!(ring.is_commutative());

    for k in 0..min(A.row_count(), A.col_count()) {
        let mut changed_row = true;
        while changed_row {
            changed_row = false;
            
            // eliminate the column
            for i in (k + 1)..A.row_count() {
                if ring.is_zero(A.at(i, k)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(i, k), A.at(k, k)) {
                    TransformRows(A.reborrow(), ring.get_ring()).subtract(ring, k, i, &quo);
                    L.subtract(ring, k, i, &quo);
                } else {
                    let (transform, _) = ring.get_ring().create_elimination_matrix(A.at(k, k), A.at(i, k));
                    TransformRows(A.reborrow(), ring.get_ring()).transform(ring, k, i, &transform);
                    L.transform(ring, k, i, &transform);
                }
            }
            
            // now eliminate the row
            for j in (k + 1)..A.col_count() {
                if ring.is_zero(A.at(k, j)) {
                    continue;
                } else if let Some(quo) = ring.checked_div(A.at(k, j), A.at(k, k)) {
                    changed_row = true;
                    TransformCols(A.reborrow(), ring.get_ring()).subtract(ring, k, j, &quo);
                    R.subtract(ring, k, j, &quo);
                } else {
                    changed_row = true;
                    let (transform, _) = ring.get_ring().create_elimination_matrix(A.at(k, k), A.at(k, j));
                    TransformCols(A.reborrow(), ring.get_ring()).transform(ring, k, j, &transform);
                    R.transform(ring, k, j, &transform);
                }
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn solve_right_using_pre_smith<R, V1, V2, V3, A>(ring: R, mut lhs: SubmatrixMut<V1, El<R>>, mut rhs: SubmatrixMut<V2, El<R>>, mut out: SubmatrixMut<V3, El<R>>, _allocator: A) -> SolveResult
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V1: AsPointerToSlice<El<R>>, V2: AsPointerToSlice<El<R>>, V3: AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(lhs.row_count(), rhs.row_count());
    assert_eq!(lhs.col_count(), out.row_count());
    assert_eq!(rhs.col_count(), out.col_count());

    let mut R = TransformList::new(lhs.col_count());
    pre_smith(ring, &mut TransformRows(rhs.reborrow(), ring.get_ring()), &mut R, lhs.reborrow());

    for i in out.row_count()..rhs.row_count() {
        for j in 0..rhs.col_count() {
            if !ring.is_zero(rhs.at(i, j)) {
                return SolveResult::NoSolution;
            }
        }
    }
    // the value of out[lhs.row_count().., ..] is irrelevant, since lhs
    // is zero in these places anyway. Thus we just leave it unchanged

    for i in 0..min(lhs.row_count(), lhs.col_count()) {
        let pivot = lhs.at(i, i);
        for j in 0..rhs.col_count() {
            if let Some(quo) = ring.checked_left_div(rhs.at(i, j), pivot) {
                *out.at_mut(i, j) = quo;
            } else {
                return SolveResult::NoSolution;
            }
        }
    }

    R.replay_transposed(ring, TransformRows(out, ring.get_ring()));
    return SolveResult::FoundSomeSolution;
}

#[stability::unstable(feature = "enable")]
pub fn determinant_using_pre_smith<R, V, A>(ring: R, mut matrix: SubmatrixMut<V, El<R>>, _allocator: A) -> El<R>
    where R: RingStore + Copy,
        R::Type: PrincipalIdealRing,
        V:AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(matrix.row_count(), matrix.col_count());
    let mut unit_part_rows = ring.one();
    let mut unit_part_cols = ring.one();
    pre_smith(ring, &mut DetUnit { current_unit: &mut unit_part_rows }, &mut DetUnit { current_unit: &mut unit_part_cols }, matrix.reborrow());
    return ring.checked_div(
        &ring.prod((0..matrix.row_count()).map(|i| ring.clone_el(matrix.at(i, i)))),
        &ring.prod([unit_part_rows, unit_part_cols])
    ).unwrap();
}

struct DetUnit<'a, R: ?Sized + RingBase> {
    current_unit: &'a mut R::Element
}

impl<'a, R> TransformTarget<R> for DetUnit<'a, R>
    where R: ?Sized + RingBase
{
    fn subtract<S: Copy + RingStore<Type = R>>(&mut self, _ring: S, _src: usize, _dst: usize, _factor: &<R as RingBase>::Element) {
        // determinant does not change
    }

    fn swap<S: Copy + RingStore<Type = R>>(&mut self, ring: S, _i: usize, _j: usize) {
        ring.negate_inplace(&mut self.current_unit)
    }

    fn transform<S: Copy + RingStore<Type = R>>(&mut self, ring: S, _i: usize, _j: usize, transform: &[<R as RingBase>::Element; 4]) {
        let unit = ring.sub(ring.mul_ref(&transform[0], &transform[3]), ring.mul_ref(&transform[1], &transform[2]));
        ring.mul_assign(&mut self.current_unit, unit);
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use crate::rings::extension::FreeAlgebraStore;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn64B;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;
#[cfg(test)]
use crate::seq::VectorView;
#[cfg(test)]
use crate::algorithms::matmul::ComputeInnerProduct;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::algorithms::linsolve::LinSolveRing;
#[cfg(test)]
use std::ptr::Alignment;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::time::Instant;
#[cfg(test)]
use crate::algorithms::convolution::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::algorithms::linsolve::extension::solve_right_over_extension;
#[cfg(test)]
use crate::algorithms::matmul::*;

#[cfg(test)]
fn multiply<'a, R: RingStore, V: AsPointerToSlice<El<R>>, I: IntoIterator<Item = Submatrix<'a, V, El<R>>>>(matrices: I, ring: R) -> OwnedMatrix<El<R>>
    where R::Type: 'a,
        V: 'a
{
    let mut it = matrices.into_iter();
    let fst = it.next().unwrap();
    let snd = it.next().unwrap();
    let mut new_result = OwnedMatrix::zero(fst.row_count(), snd.col_count(), &ring);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(fst), TransposableSubmatrix::from(snd), TransposableSubmatrixMut::from(new_result.data_mut()), &ring);
    let mut result = new_result;

    for m in it {
        let mut new_result = OwnedMatrix::zero(result.row_count(), m.col_count(), &ring);
        STANDARD_MATMUL.matmul(TransposableSubmatrix::from(result.data()), TransposableSubmatrix::from(m), TransposableSubmatrixMut::from(new_result.data_mut()), &ring);
        result = new_result;
    }
    return result;
}

#[test]
fn test_smith_integers() {
    let ring = StaticRing::<i64>::RING;
    let mut A = OwnedMatrix::new(
        vec![ 1, 2, 3, 4, 
                    2, 3, 4, 5,
                    3, 4, 5, 6 ], 
        4
    );
    let original_A = A.clone_matrix(&ring);
    let mut L: OwnedMatrix<i64> = OwnedMatrix::identity(3, 3, StaticRing::<i64>::RING);
    let mut R: OwnedMatrix<i64> = OwnedMatrix::identity(4, 4, StaticRing::<i64>::RING);
    pre_smith(ring, &mut TransformRows(L.data_mut(), ring.get_ring()), &mut TransformCols(R.data_mut(), ring.get_ring()), A.data_mut());
    
    assert_matrix_eq!(&ring, &[
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0, 0, 0]], &A);

    assert_matrix_eq!(&ring, &multiply([L.data(), original_A.data(), R.data()], ring), &A);
}

#[test]
fn test_smith_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let mut A = OwnedMatrix::new(
        vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ],
        4
    );
    let original_A = A.clone_matrix(&ring);
    let mut L: OwnedMatrix<u64> = OwnedMatrix::identity(5, 5, ring);
    let mut R: OwnedMatrix<u64> = OwnedMatrix::identity(4, 4, ring);
    pre_smith(ring, &mut TransformRows(L.data_mut(), ring.get_ring()), &mut TransformCols(R.data_mut(), ring.get_ring()), A.data_mut());

    assert_matrix_eq!(&ring, &[
        [8, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0, 15],
        [0, 0, 0, 0]], &A);
        
    assert_matrix_eq!(&ring, &multiply([L.data(), original_A.data(), R.data()], ring), &A);
}

#[test]
fn test_solve_zn() {
    let ring = zn_static::Zn::<45>::RING;
    let A = OwnedMatrix::new(
        vec![ 8, 3, 5, 8,
                    0, 9, 0, 9,
                    5, 9, 5, 14,
                    8, 3, 5, 23,
                    3,39, 0, 39 ],
        4
    );
    let B = OwnedMatrix::new(
        vec![11, 43, 10, 22,
                   18,  9, 27, 27,
                    8, 34,  7, 22,
                   41, 13, 40, 37,
                    3,  9,  3,  0],
        4
    );
    let mut solution: OwnedMatrix<_> = OwnedMatrix::zero(4, 4, ring);
    ring.get_ring().solve_right(A.clone_matrix(ring).data_mut(), B.clone_matrix(ring).data_mut(), solution.data_mut(), Global).assert_solved();

    assert_matrix_eq!(&ring, &multiply([A.data(), solution.data()], ring), &B);
}

#[test]
fn test_solve_int() {
    let ring = StaticRing::<i64>::RING;
    let A = OwnedMatrix::new(
        vec![3, 6, 2, 0, 4, 7,
                   5, 5, 4, 5, 5, 5],
        6
    );
    let B: OwnedMatrix<i64> = OwnedMatrix::identity(2, 2, ring);
    let mut solution: OwnedMatrix<i64> = OwnedMatrix::zero(6, 2, ring);
    ring.get_ring().solve_right(A.clone_matrix(ring).data_mut(), B.clone_matrix(ring).data_mut(), solution.data_mut(), Global).assert_solved();

    assert_matrix_eq!(&ring, &multiply([A.data(), solution.data()], &ring), &B);
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
    let mut A: OwnedMatrix<u64> = OwnedMatrix::zero(6, 11, &ring);
    for i in 0..6 {
        for j in 0..11 {
            *A.at_mut(i, j) = data_A[i][j];
        }
    }
    let mut solution: OwnedMatrix<_> = OwnedMatrix::zero(11, 11, ring);
    assert!(ring.get_ring().solve_right(A.clone_matrix(&ring).data_mut(), A.data_mut(), solution.data_mut(), Global).is_solved());
}

#[test]
fn test_determinant() {
    let ring = StaticRing::<i64>::RING;
    let A = OwnedMatrix::new(
        vec![1, 0, 3, 
                   2, 1, 0, 
                   9, 8, 7],
        3
    );
    assert_el_eq!(ring, (7 + 48 - 27), determinant_using_pre_smith(ring, A.clone_matrix(&ring).data_mut(), Global));
    
    // we need a ring that has units of order > 2 to test whether an inversion is necessary for
    // the accumulated determinant units
    #[derive(PartialEq, Clone, Copy, Debug)]
    struct TestRing;
    use crate::delegate::DelegateRing;
    impl DelegateRing for TestRing {
        type Base = zn_static::ZnSBase<45, false>;
        type Element = u64;

        fn get_delegate(&self) -> &Self::Base { zn_static::Zn::RING.get_ring() }
        fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
        fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
        fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
        fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
    }
    impl PrincipalIdealRing for TestRing {

        fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
            self.get_delegate().extended_ideal_gen(lhs, rhs)
        }

        fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
            self.get_delegate().checked_div_min(lhs, rhs)
        }

        fn create_elimination_matrix(&self, a: &Self::Element, b: &Self::Element) -> ([Self::Element; 4], Self::Element) {
            assert_eq!(9, *a);
            assert_eq!(15, *b);
            assert_eq!(3, self.add(self.mul(42, *a), self.mul(2, *b)));
            assert_eq!(0, self.add(self.mul(5, *a), self.mul(3, *b)));
            return ([42, 2, 10, 6], 3);
        }
    }

    let ring = RingValue::from(TestRing);
    let A = OwnedMatrix::new(
        vec![ 9, 0, 
                   15, 3],
        2
    );
    assert_el_eq!(ring, 27, determinant_using_pre_smith(ring, A.clone_matrix(&ring).data_mut(), Global));
}

#[test]
#[ignore]
fn time_solve_right_using_pre_smith_galois_field() {
    let n = 100;
    let base_field = Zn64B::new(257).as_field().ok().unwrap();
    let allocator = feanor_mempool::AllocArc(Arc::new(feanor_mempool::dynsize::DynLayoutMempool::new_global(Alignment::of::<u64>())));
    let field = GaloisField::new_with_convolution(base_field, 21, allocator, STANDARD_CONVOLUTION);
    let matrix = OwnedMatrix::from_fn(n, n, |i, j| field.pow(field.int_hom().mul_map(field.canonical_gen(), i as i32 + 1), j));
    
    let mut inv = OwnedMatrix::zero(n, n, &field);
    let mut copy = matrix.clone_matrix(&field);
    let start = Instant::now();
    solve_right_using_pre_smith(&field, copy.data_mut(), OwnedMatrix::identity(n, n, &field).data_mut(), inv.data_mut(), Global).assert_solved();
    let end = Instant::now();
    assert_el_eq!(&field, field.one(), <_ as ComputeInnerProduct>::inner_product_ref(field.get_ring(), inv.data().col_at(4).as_iter().zip(matrix.data().row_at(4).as_iter())));
    
    println!("total: {} us", (end - start).as_micros());
}

#[test]
#[ignore]
fn time_solve_right_using_extension() {
    let n = 126;
    let base_field = Zn64B::new(257).as_field().ok().unwrap();
    let allocator = feanor_mempool::AllocArc(Arc::new(feanor_mempool::dynsize::DynLayoutMempool::new_global(Alignment::of::<u64>())));
    let field = GaloisField::new_with_convolution(base_field, 21, allocator, STANDARD_CONVOLUTION);
    let matrix = OwnedMatrix::from_fn(n, n, |i, j| field.pow(field.int_hom().mul_map(field.canonical_gen(), i as i32 + 1), j));
    
    let mut inv = OwnedMatrix::zero(n, n, &field);
    let mut copy = matrix.clone_matrix(&field);
    let start = Instant::now();
    solve_right_over_extension(&field, copy.data_mut(), OwnedMatrix::identity(n, n, &field).data_mut(), inv.data_mut(), Global).assert_solved();
    let end = Instant::now();
    assert_el_eq!(&field, field.one(), <_ as ComputeInnerProduct>::inner_product_ref(field.get_ring(), inv.data().col_at(4).as_iter().zip(matrix.data().row_at(4).as_iter())));

    println!("total: {} us", (end - start).as_micros());
}

#[bench]
fn bench_solve_right_using_pre_smith_galois_field(bencher: &mut Bencher) {
    let base_field = Zn64B::new(257).as_field().ok().unwrap();
    let allocator = feanor_mempool::AllocArc(Arc::new(feanor_mempool::dynsize::DynLayoutMempool::new_global(Alignment::of::<u64>())));
    let field = GaloisField::create(FreeAlgebraImpl::new_with_convolution(base_field, 5, [base_field.int_hom().map(3), base_field.int_hom().map(-4)], "x", allocator, STANDARD_CONVOLUTION).as_field().ok().unwrap());
    let matrix = OwnedMatrix::from_fn(10, 10, |i, j| field.pow(field.int_hom().mul_map(field.canonical_gen(), i as i32 + 1), j));
    bencher.iter(|| {
        let mut inv = OwnedMatrix::zero(10, 10, &field);
        let mut copy = matrix.clone_matrix(&field);
        solve_right_using_pre_smith(&field, copy.data_mut(), OwnedMatrix::identity(10, 10, &field).data_mut(), inv.data_mut(), Global).assert_solved();
        assert_el_eq!(&field, field.one(), <_ as ComputeInnerProduct>::inner_product_ref(field.get_ring(), inv.data().col_at(4).as_iter().zip(matrix.data().row_at(4).as_iter())));
    });
}