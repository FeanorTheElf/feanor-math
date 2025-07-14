use crate::algorithms::matmul::*;
use crate::field::*;
use crate::integer::*;
use crate::homomorphism::*;
use crate::matrix::*;
use crate::matrix::transform::TransformTarget;
use crate::seq::*;
use crate::rings::fraction::FractionFieldStore;
use crate::rings::rational::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::ring::*;
use crate::rings::float_real::Real64Base;

use std::alloc::Allocator;
use std::cmp::max;
use std::marker::PhantomData;

mod float;

///
/// Trait for (possibly approximations to) the real numbers that are used to keep
/// an estimate of the size and orthogonality of vectors during executions of LLL.
/// 
/// Errors caused by approximation might reduce the quality of the LLL-reduced basis,
/// but won't give any other correctness errors.
/// 
#[stability::unstable(feature = "enable")]
pub trait LLLRealField<I>: OrderedRing + Field
    where I: ?Sized + IntegerRing
{
    fn from_integer(&self, x: &I::Element, ZZ: &I) -> Self::Element;
    fn round_to_integer(&self, x: &Self::Element, ZZ: &I) -> Option<I::Element>;
}

impl<I, J> LLLRealField<J> for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        J: ?Sized + IntegerRing
{
    fn from_integer(&self, x: &J::Element, ZZ: &J) -> Self::Element {
        RingRef::new(self).inclusion().map(int_cast(ZZ.clone_el(x), self.base_ring(), RingRef::new(ZZ)))
    }

    fn round_to_integer(&self, x: &Self::Element, ZZ: &J) -> Option<J::Element> {
        Some(int_cast(self.base_ring().rounded_div(self.base_ring().clone_el(self.num(x)), self.den(x)), RingRef::new(ZZ), self.base_ring()))
    }
}

impl<J> LLLRealField<J> for Real64Base
    where J: ?Sized + IntegerRing
{
    fn from_integer(&self, x: &J::Element, ZZ: &J) -> Self::Element {
        ZZ.to_float_approx(x)
    }

    fn round_to_integer(&self, x: &Self::Element, ZZ: &J) -> Option<J::Element> {
        if x.is_nan() || x.abs() > i64::MAX as f64 {
            return None;
        }
        Some(int_cast(x.round() as i64, RingRef::new(ZZ), StaticRing::<i64>::RING))
    }
}

///
/// Size-reduces `target` w.r.t. the GSO matrix, and also sends the performed
/// operations to `col_ops`.
/// 
fn size_reduce_exact<I, V, T>(
    ring: &RationalField<I>, 
    int_ring: I, 
    mut target: SubmatrixMut<V, El<RationalField<I>>>, 
    target_j: usize, 
    matrix: Submatrix<V, El<RationalField<I>>>, 
    col_ops: &mut T
)
    where I: IntegerRingStore + Copy,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<RationalField<I>>>,
        T: TransformTarget<I::Type>
{
    assert!(!ring.get_ring().is_approximate());
    for j in (0..matrix.col_count()).rev() {
        let factor = ring.get_ring().round_to_integer(target.as_const().at(j, 0), int_ring.get_ring()).unwrap();
        col_ops.subtract(int_ring, j, target_j, &factor);
        let factor = ring.get_ring().from_integer(&factor, int_ring.get_ring());
        ring.sub_assign_ref(target.at_mut(j, 0), &factor);
        for k in 0..j {
            ring.sub_assign(target.at_mut(k, 0), ring.mul_ref(matrix.at(k, j), &factor));
        }
    }
}

///
/// Updates the "gso"-matrix resulting from swapping cols i and i + 1.
/// 
/// This uses explicit formulas, in particular if `b_i` and `b_(i + 1)`
/// represent the vectors, `b_i*` and `b_(i + 1)*` represent their GS-orthogonalizations
/// and the corresponding `b'` values are the ones after swapping, then we find
/// then this gives
/// ```text
///   b'_i = b_(i + 1)
///   b'_(i + 1) = b_i
/// 
///   b'_i* = b_(i + 1)* + mu b_i*
///   b'_(i + 1) = (1 - gamma^2 mu^2) b_i* - mu * gamma^2 b_(i + 1)*
///     where gamma^2 = |b_i*|^2 / |b'_i*|^2
///   mu' = gamma^2 mu
/// ```
/// 
fn swap_gso_cols<R, V>(ring: R, mut gso: SubmatrixMut<V, El<R>>, i: usize, j: usize)
    where R: RingStore,
        R::Type: OrderedRing + Field,
        V: AsPointerToSlice<El<R>>
{
    // numerically very unstable
    assert!(!ring.get_ring().is_approximate());
    assert!(j == i + 1);

    let col_count = gso.col_count();

    // swap the columns
    let (mut col_i, mut col_i1) = gso.reborrow().restrict_cols(i..(i + 2)).split_cols(0..1, 1..2);
    for k in 0..i {
        std::mem::swap(col_i.at_mut(k, 0), col_i1.at_mut(k, 0));
    }

    // re-orthogonalize the triangle `i..(i + 2) x i..(i + 2)`

    // | b_i* |^2
    let bi_star_norm_sqr = ring.clone_el(gso.at(i, i));
    // | b_(i + 1)* |^2
    let bi1_star_norm_sqr = ring.clone_el(gso.at(i + 1, i + 1));
    // mu_(i + 1)i = <b_(i + 1), bi*> / <bi*, bi*>
    let mu = ring.clone_el(gso.at(i, i + 1));
    let mu_sqr = ring.pow(ring.clone_el(&mu), 2);

    let new_bi_star_norm_sqr = ring.add_ref_fst(&bi1_star_norm_sqr, ring.mul_ref(&mu_sqr, &bi_star_norm_sqr));
    // `gamma := |b_(i + 1)*|^2 / |bnew_i*|^2`
    let gamma = if ring.is_leq(&new_bi_star_norm_sqr, &bi1_star_norm_sqr) {
        // in this case, we must have `|bnew_i*|^2 = |b_(i + 1)*|^2`
        ring.one()
    } else {
        ring.div(&bi1_star_norm_sqr, &new_bi_star_norm_sqr)
    };
    let new_bi1_star_norm_sqr = ring.mul_ref(&gamma, &bi_star_norm_sqr);

    // we now update the `mu_ki` resp. `mu_k(i + 1)` by a linear transform;
    // Concretely it is given by the matrix multiplication
    // mu_(j, i)' = mu_(i + 1, i)' mu_(j, i) + gamma mu_(j, i + 1)
    // mu_(j, i + 1)' = mu_(j, i) - mu_(i, i + 1) mu_(j, i + 1)

    // we use `mu_(i + 1, i)' = mu_(i + 1, i) * |b_i*|^2 / |bnew_i*|^2`
    let new_mu = if ring.is_zero(&new_bi_star_norm_sqr) {
        ring.zero()
    } else {
        // this is why we cannot work with floats here, since this might cause a huge error
        ring.div(&ring.mul_ref(&mu, &bi_star_norm_sqr), &new_bi_star_norm_sqr)
    };
    let (row_i, row_i1) = gso.reborrow().restrict_rows(i..(i + 2)).split_rows(0..1, 1..2);
    let row_i = row_i.into_row_mut_at(0);
    let row_i1 = row_i1.into_row_mut_at(0);
    for k in (i + 2)..col_count {
        let new_mu_j_i = ring.add(
            ring.mul_ref(&new_mu, row_i.at(k)),
            ring.mul_ref(&gamma, row_i1.at(k))
        );
        let new_mu_j_i1 = ring.sub_ref_fst(row_i.at(k), ring.mul_ref(&mu, row_i1.at(k)));
        *row_i.at_mut(k) = new_mu_j_i;
        *row_i1.at_mut(k) = new_mu_j_i1;
    }

    *gso.at_mut(i, i + 1) = new_mu;
    *gso.at_mut(i, i) = new_bi_star_norm_sqr;
    *gso.at_mut(i + 1, i + 1) = new_bi1_star_norm_sqr;
}

///
/// Computes the LDL-decomposition of the given matrix, i.e. writes it as
/// a product `L * D * L^T`, where `D` is diagonal and `L` is lower triangle.
/// 
/// If the matrix is not invertible, or (in the floating point case) some eigenvalues
/// are very small, this function cannot proceed, and will return the index of the 
/// column in which on to small values have been detected as `Err()`. The top left
/// square until this index will contain a valid LDL-decomposition of the corresponding
/// square of the input.
/// 
/// `D` is returned on the diagonal of the matrix, and `L^T` is returned in
/// the upper triangle of the matrix.
/// 
fn ldl<R, V>(ring: R, mut matrix: SubmatrixMut<V, El<R>>) -> Result<(), usize>
    where R: RingStore,
        R::Type: Field, 
        V: AsPointerToSlice<El<R>>
{
    // only the upper triangle part of matrix is used
    assert_eq!(matrix.row_count(), matrix.col_count());
    let n = matrix.row_count();
    for i in 0..n {
        let pivot = ring.clone_el(matrix.at(i, i));
        if ring.is_zero(&pivot) {
            return Err(i);
        }
        let pivot_inv = ring.div(&ring.one(), matrix.at(i, i));
        for j in (i + 1)..n {
            ring.mul_assign_ref(matrix.at_mut(i, j), &pivot_inv);
        }
        for k in (i + 1)..n {
            for l in k..n {
                let subtract = ring.mul_ref_snd(ring.mul_ref(matrix.as_const().at(i, k), matrix.as_const().at(i, l)), &pivot);
                ring.sub_assign(matrix.at_mut(k, l), subtract);
            }
        }
    }
    return Ok(());
}

///
/// LLL-reduces the given matrix, i.e. transforms `B` into `B'` such that
/// `B' = BU` with `U in GL(Z)` and the columns of `B'` are "short".
/// 
/// The exact restrictions imposed on `B'` are that its columns `b1, ..., bn`
/// are "LLL-reduced". This means
///  - (size-reduced) `|<bi,bj*>| < <bj*,bj*> / 2` whenever `i > j`
///  - (Lovasz-condition) `delta |b(k - 1)|^2 - <bk, b(k - 1)*>^2 / |b(k - 1)|^2 <= |bk*|^2`
/// Here the `bi*` refer to the Gram-Schmidt orthogonalization of the `bi`.
/// 
/// # Internal computations with floating point numbers
/// 
/// The LLL algorithm has to handle fractional values internally, which can get huge
/// denominators. Since only their size is relevant, it is usually much better to work
/// with floating point numbers instead. This is done as a precomputation in this function,
/// and as the only version in [`lll_float()`].
/// 
/// Despite precomputing the floating-point LLL in this function, we still perform the
/// computation with rationals here, hence the runtime is likely to be vastly longer than
/// the one of [`lll_float()`]. Since it is very unlikely that you need the exact guarantees
/// of [`lll_exact()`], but just want a "short" matrix, prefer using [`lll_float()`] instead.
/// 
#[stability::unstable(feature = "enable")]
pub fn lll_exact<I, V1, V2, A>(
    ring: I, 
    quadratic_form: Submatrix<V1, El<RationalField<I>>>, 
    matrix: SubmatrixMut<V2, El<I>>, 
    delta: &El<RationalField<I>>, 
    allocator: A
)
    where I: IntegerRingStore + Copy,
        I::Type: IntegerRing,
        V1: AsPointerToSlice<El<RationalField<I>>>,
        V2: AsPointerToSlice<El<I>>,
        A: Allocator
{
    let n = matrix.row_count();
    assert_eq!(n, matrix.col_count());
    assert_eq!(n, quadratic_form.col_count());
    assert_eq!(n, quadratic_form.col_count());
    let rationals = RationalField::new(ring);
    assert!(rationals.is_lt(delta, &rationals.one()));
    assert!(rationals.is_gt(delta, &rationals.from_fraction(ring.one(), ring.int_hom().map(4))));

    let n = matrix.col_count();
    let mut gso = OwnedMatrix::zero_in(n, n, &rationals, &allocator);
    let hom = rationals.inclusion();
    let matrix_QQ: OwnedMatrix<_, &A> = OwnedMatrix::from_fn_in(matrix.row_count(), matrix.col_count(), |i, j| hom.map_ref(matrix.at(i, j)), &allocator);
    STANDARD_MATMUL.matmul(TransposableSubmatrix::from(matrix_QQ.data()).transpose(), TransposableSubmatrix::from(matrix_QQ.data()), TransposableSubmatrixMut::from(gso.data_mut()), &rationals);

    ldl(&rationals, gso.data_mut()).expect("quadratic form not positive definite");
    
    let mut col_ops = TransformLatticeBasis { basis: matrix, int_ring: PhantomData, hom: ring.identity() };
    let mut i = 0;
    while i + 1 < gso.col_count() {

        let (target, matrix) = gso.data_mut().split_cols((i + 1)..(i + 2), 0..(i + 1));
        size_reduce_exact(&rationals, ring, target, i + 1, matrix.as_const(), &mut col_ops);

        if rationals.is_gt(
            &rationals.mul_ref_snd(
                rationals.sub_ref_fst(delta, rationals.mul_ref(gso.data().at(i, i + 1), gso.data().at(i, i + 1))),
                gso.data().at(i, i)
            ),
            gso.data().at(i + 1, i + 1)
        ) {
            col_ops.swap(ring, i, i + 1);
            swap_gso_cols(&rationals, gso.data_mut(), i, i + 1);
            i = max(i, 1) - 1;
        } else {
            i += 1;
        }
    }
}

struct TransformLatticeBasis<'a, R, I, V, H>
    where R: ?Sized + RingBase,
        I: ?Sized + IntegerRing,
        H: Homomorphism<I, R>,
        V: AsPointerToSlice<R::Element>
{
    basis: SubmatrixMut<'a, V, R::Element>,
    int_ring: PhantomData<I>,
    hom: H
}

impl<'a, R, I, V, H> TransformTarget<I> for TransformLatticeBasis<'a, R, I, V, H>
    where R: ?Sized + RingBase,
        I: ?Sized + IntegerRing,
        H: Homomorphism<I, R>,
        V: AsPointerToSlice<R::Element>
{
    fn transform<S: Copy + RingStore<Type = I>>(&mut self, ring: S, i: usize, j: usize, transform: &[I::Element; 4]) {
        assert!(ring.get_ring() == self.hom.domain().get_ring());
        assert!(i != j);
        let ring = self.hom.codomain();
        for k in 0..self.basis.row_count() {
            let a = ring.clone_el(self.basis.at(k, i));
            let b = ring.clone_el(self.basis.at(k, j));
            *self.basis.at_mut(k, i) = ring.add(self.hom.mul_ref_map(&a, &transform[0]), self.hom.mul_ref_map(&b, &transform[1]));
            *self.basis.at_mut(k, i) = ring.add(self.hom.mul_ref_snd_map(a, &transform[2]), self.hom.mul_ref_snd_map(b, &transform[3]));
        }
    }

    fn subtract<S: Copy + RingStore<Type = I>>(&mut self, ring: S, src: usize, dst: usize, factor: &I::Element) {
        assert!(ring.get_ring() == self.hom.domain().get_ring());
        assert!(src != dst);
        let ring = self.hom.codomain();
        for k in 0..self.basis.row_count() {
            let subtract = self.hom.mul_ref_map(self.basis.at(k, src), factor);
            ring.sub_assign(self.basis.at_mut(k, dst), subtract);
        }
    }

    fn swap<S: Copy + RingStore<Type = I>>(&mut self, ring: S, i: usize, j: usize) {
        assert!(ring.get_ring() == self.hom.domain().get_ring());
        if i == j {
            return;
        }
        let col_count = self.basis.col_count();
        let (mut col_i, mut col_j) = self.basis.reborrow().split_cols(i..(i + 1), j..(j + 1));
        for k in 0..col_count {
            std::mem::swap(col_i.at_mut(k, 0), col_j.at_mut(k, 0));
        }
    }
}

#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::assert_matrix_eq;

#[cfg(test)]
const QQ: RationalField<StaticRing<i64>> = RationalField::new(StaticRing::<i64>::RING);

#[cfg(test)]
macro_rules! in_QQ {
    ($hom:expr; $num:literal) => {
        ($hom).map($num)
    };
    ($hom:expr; $num:literal, $den:literal) => {
        ($hom).codomain().div(&($hom).map($num), &($hom).map($den))
    };
    ($([$($num:literal $(/ $den:literal)?),*]),*) => {
        {
            let ZZ_to_QQ = QQ.inclusion();
            [
                $([$(
                    in_QQ!(ZZ_to_QQ; $num $(, $den)?)
                ),*]),*
            ]
        }
    };
    ($(DerefArray::from([$($num:literal $(/ $den:literal)?),*])),*) => {
        {
            let ZZ_to_QQ = QQ.inclusion();
            [
                $(DerefArray::from([$(
                    in_QQ!(ZZ_to_QQ; $num $(, $den)?)
                ),*])),*
            ]
        }
    };
}

#[test]
fn test_ldl() {
    let mut data = in_QQ![
        DerefArray::from([1, 2, 1]),
        DerefArray::from([2, 5, 0]),
        DerefArray::from([1, 0, 7])
    ];
    let mut matrix = SubmatrixMut::<DerefArray<_, 3>, _>::from_2d(&mut data);
    let mut expected = in_QQ![
        [1, 2, 1],
        [0, 1, -2],
        [0, 0, 2]
    ];
    ldl(QQ, matrix.reborrow()).unwrap();

    // only the upper triangle is filled
    expected[1][0] = *matrix.at(1, 0);
    expected[2][0] = *matrix.at(2, 0);
    expected[2][1] = *matrix.at(2, 1);

    assert_matrix_eq!(&QQ, &expected, &matrix);
}

#[test]
fn test_swap_gso_cols() {
    let mut matrix = in_QQ![
        DerefArray::from([2, 1/2, 2/5]),
        DerefArray::from([0, 3/2, 1/4]),
        DerefArray::from([0,   0,   1])
    ];
    let expected = in_QQ![
        [2, 1/2, 31/80],
        [0, 3/2, 11/40],
        [0,   0,     1]
    ];
    let matrix_view = SubmatrixMut::<DerefArray<_, 3>, _>::from_2d(&mut matrix);

    swap_gso_cols(&QQ, matrix_view, 0, 1);

    assert_matrix_eq!(&QQ, &expected, &matrix);
}

#[cfg(test)]
fn norm_squared<I, V>(ZZ: I, col: &Column<V, El<I>>) -> El<I>
    where V: AsPointerToSlice<El<I>>,
        I: RingStore, I::Type: IntegerRing
{
    <_ as RingStore>::sum(&ZZ, (0..col.len()).map(|i| ZZ.mul_ref(col.at(i), col.at(i))))
}

#[cfg(test)]
fn assert_lattice_isomorphic<I, V1, V2>(ZZ: I, lhs: Submatrix<V1, El<I>>, rhs: &Submatrix<V2, El<I>>)
    where V1: AsPointerToSlice<El<I>>, V2: AsPointerToSlice<El<I>>,
        I: RingStore, I::Type: IntegerRing
{
    use crate::algorithms::linsolve::smith;

    let n = lhs.row_count();
    assert_eq!(n, rhs.row_count());
    let m = lhs.col_count();
    assert_eq!(m, rhs.col_count());
    let ZZbig = BigIntRing::RING;
    let mut A: OwnedMatrix<_> = OwnedMatrix::zero(n, m, ZZbig);
    let mut B: OwnedMatrix<_> = OwnedMatrix::zero(n, m, ZZbig);
    let int_to_ZZbig: CanHom<&I, &BigIntRing> = ZZbig.can_hom(&ZZ).unwrap();
    for i in 0..n {
        for j in 0..m {
            *A.at_mut(i, j) = int_to_ZZbig.map_ref(lhs.at(i, j));
            *B.at_mut(i, j) = int_to_ZZbig.map_ref(rhs.at(i, j));
        }
    }
    let mut U: OwnedMatrix<_> = OwnedMatrix::zero(n, m, ZZbig);
    assert!(smith::solve_right_using_pre_smith(&ZZbig, A.clone_matrix(&ZZbig).data_mut(), B.clone_matrix(&ZZbig).data_mut(), U.data_mut(), Global).is_solved());
    assert!(smith::solve_right_using_pre_smith(&ZZbig, B.clone_matrix(&ZZbig).data_mut(), A.clone_matrix(&ZZbig).data_mut(), U.data_mut(), Global).is_solved());
}

#[test]
fn test_lll_exact_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let original = [
        DerefArray::from([5,   9]),
        DerefArray::from([11, 20])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_exact(ZZ, OwnedMatrix::identity(2, 2, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(9, 10), Global);

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));

    let original = [
        DerefArray::from([10, 8]),
        DerefArray::from([27, 22])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_exact(ZZ, OwnedMatrix::identity(2, 2, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(9, 10), Global);

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(4, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
}

#[test]
fn test_lll_exact_3d() {
    let ZZ = StaticRing::<i64>::RING;
    // in this case, the shortest vector is shorter than half the second successive minimum,
    // so LLL will find it (for delta = 0.9 > 0.75)
    let original = [
        DerefArray::from([72, 0, 0]),
        DerefArray::from([0,  9, 0]),
        DerefArray::from([8432, 7344, 16864])
    ];
    let _expected = [
        [144, 72, 72],
        [0, 279, -72],
        [0,   0, 272]
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 3>, _>::from_2d(&mut reduced);
    lll_exact(ZZ, OwnedMatrix::identity(3, 3, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(999, 1000), Global);

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(144 * 144, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(72 * 72 + 279 * 279, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_eq!(72 * 72 * 2 + 272 * 272, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
}

#[bench]
fn bench_lll_exact_10d(bencher: &mut Bencher) {
    let ZZ = BigIntRing::RING;

    let _expected = [
        [  2,   0,   0,  -2,  -6,  -2,  -3,   1,  -1,  -1],
        [  0,   0,   1,  -2,  -1,   2,  -7,  -8,   8,   1],
        [ -1,   1,   0,   4,  -1,   1,  -1,  -5,   1, -11],
        [  3,   1,  -2,   0,   2,   1,  -2,   1,   5, -11],
        [ -1,   5,   3,  -1,  -1,  -2,  -3,   1,  -3,   5],
        [  1,  -1,   3,   1,   1,   2,  -1,   0,  -6,   2],
        [  1,   1,   0,   3,   0,  -2,   1,  -1,   4,   6],
        [  1,   1,   2,  -1,   0,   2,   7,   1,   2,   2],
        [  1,   0,  -4,   2,   2,   4,  -1,   3,  -3,   8],
        [ -1,  -2,   1,   1,   0,   3,   0,   7,   5,  -2]
    ];
    bencher.iter(|| {
        let original: Vec<Vec<_>> = [
            [       1,        0,        0,        0,        0,        0,        0,        0,        0,        0],
            [       0,        1,        0,        0,        0,        0,        0,        0,        0,        0],
            [       0,        0,        1,        0,        0,        0,        0,        0,        0,        0],
            [       0,        0,        0,        1,        0,        0,        0,        0,        0,        0],
            [       0,        0,        0,        0,        1,        0,        0,        0,        0,        0],
            [       0,        0,        0,        0,        0,        1,        0,        0,        0,        0],
            [       0,        0,        0,        0,        0,        0,        1,        0,        0,        0],
            [       2,        2,        2,        2,        0,        0,        1,        4,        0,        0],
            [       4,        3,        3,        3,        1,        2,        1,        0,        5,        0],
            [ 3433883, 14315221, 24549008,  6570781, 32725387, 33674813, 27390657, 15726308, 43003827, 43364304]
        ].into_iter().map(|row| row.into_iter().map(|x| int_cast(x, ZZ, StaticRing::<i64>::RING)).collect()).collect();
        let mut reduced = original.clone();
        let mut reduced_matrix = SubmatrixMut::<Vec<_>, _>::from_2d(&mut reduced);
        let delta = RationalField::new(ZZ).coerce(&QQ, QQ.from_fraction(9, 10));
        lll_exact(ZZ, OwnedMatrix::identity(10, 10, RationalField::new(ZZ)).data(), reduced_matrix.reborrow(), &delta, Global);

        assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
        assert!(16 * 16 > int_cast(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)), StaticRing::<i64>::RING, ZZ));
    });
}