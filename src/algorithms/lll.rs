use crate::field::*;
use crate::integer::*;
use crate::homomorphism::*;
use crate::matrix::matmul_fst_transposed;
use crate::matrix::submatrix::*;
use crate::matrix::Matrix;
use crate::primitive_int::*;
use crate::matrix::TransformTarget;
use crate::rings::float_real::Real64;
use crate::rings::float_real::Real64Base;
use crate::rings::rational::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::ring::*;

use std::cmp::max;

pub trait LLLRealField<I>: OrderedRing + Field
    where I: ?Sized + IntegerRing
{
    fn from_integer(&self, x: I::Element, ZZ: &I) -> Self::Element;
    fn round_to_integer(&self, x: &Self::Element, ZZ: &I) -> I::Element;
}

impl<I, J> LLLRealField<J> for RationalFieldBase<I>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        J: ?Sized + IntegerRing
{
    fn from_integer(&self, x: J::Element, ZZ: &J) -> Self::Element {
        RingRef::new(self).inclusion().map(int_cast(x, self.base_ring(), RingRef::new(ZZ)))
    }

    fn round_to_integer(&self, x: &Self::Element, ZZ: &J) -> J::Element {
        int_cast(self.base_ring().rounded_div(self.base_ring().clone_el(&x.0), &x.1), RingRef::new(ZZ), self.base_ring())
    }
}

impl<J> LLLRealField<J> for Real64Base
    where J: ?Sized + IntegerRing
{
    fn from_integer(&self, x: J::Element, ZZ: &J) -> Self::Element {
        ZZ.to_float_approx(&x)
    }

    fn round_to_integer(&self, x: &Self::Element, ZZ: &J) -> J::Element {
        int_cast(x.round() as i64, RingRef::new(ZZ), StaticRing::<i64>::RING)
    }
}

fn size_reduce<R, I, V, T>(ring: R, int_ring: I, mut target: SubmatrixMut<V, El<R>>, target_j: usize, matrix: Submatrix<V, El<R>>, col_ops: &mut T)
    where R: RingStore,
        R::Type: LLLRealField<I::Type>,
        I: IntegerRingStore,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<R>>,
        T: TransformTarget<I::Type>
{
    for j in (0..matrix.col_count()).rev() {
        let factor = ring.get_ring().round_to_integer(target.as_const().at(j, 0), int_ring.get_ring());
        col_ops.subtract(int_ring.get_ring(), j, target_j, &factor);
        let factor = ring.get_ring().from_integer(factor, int_ring.get_ring());
        ring.sub_assign_ref(target.at(j, 0), &factor);
        for k in 0..j {
            ring.sub_assign(target.at(k, 0), ring.mul_ref(matrix.at(k, j), &factor));
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
/// b'_i = b_(i + 1)
/// b'_(i + 1) = b_i
/// 
/// b'_i* = b_(i + 1)* + mu b_i*
/// b'_(i + 1) = (1 - gamma^2 mu^2) b_i* - mu * gamma^2 b_(i + 1)*
///     where gamma^2 = |b_i*|^2 / |b'_i*|^2
/// mu' = gamma^2 mu
/// ```
/// 
fn swap_gso_cols<R, V>(ring: R, mut gso: SubmatrixMut<V, El<R>>, i: usize, j: usize)
    where R: RingStore,
        R::Type: OrderedRing + Field,
        V: AsPointerToSlice<El<R>>
{
    assert!(j == i + 1);

    let col_count = gso.col_count();

    // swap the columns
    let (mut col_i, mut col_i1) = gso.reborrow().restrict_cols(i..(i + 2)).split_cols(0..1, 1..2);
    for k in 0..i {
        std::mem::swap(col_i.at(k, 0), col_i1.at(k, 0));
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
    // `|b_i*|^2 / |bnew_i*|^2`
    let gamma_sqr = ring.div(&bi_star_norm_sqr, &new_bi_star_norm_sqr);
    let new_bi1_star_norm_sqr = ring.mul_ref(&gamma_sqr, &bi1_star_norm_sqr);
    let new_mu = ring.mul_ref(&gamma_sqr, &mu);

    // we now update the `mu_ki` resp. `mu_k(i + 1)` by a linear transform
    let lin_transform_muki = [ring.mul_ref(&gamma_sqr, &mu), ring.sub(ring.one(), ring.mul_ref(&gamma_sqr, &mu_sqr))];
    let (mut row_i, mut row_i1) = gso.reborrow().restrict_rows(i..(i + 2)).split_rows(0..1, 1..2);
    for k in (i + 2)..col_count {
        let mu_ki = ring.clone_el(row_i.at(0, k));
        std::mem::swap(row_i.at(0, k), row_i1.at(0, k));
        ring.sub_assign(row_i1.at(0, k), ring.mul_ref(&mu, row_i.at(0, k)));
        ring.mul_assign_ref(row_i.at(0, k), &lin_transform_muki[1]);
        ring.add_assign(row_i.at(0, k), ring.mul_ref_fst(&lin_transform_muki[0], mu_ki));
    }

    *gso.at(i, i) = new_bi_star_norm_sqr;
    *gso.at(i, i + 1) = new_mu;
    *gso.at(i + 1, i + 1) = new_bi1_star_norm_sqr;
}

///
/// gso contains on the diagonal the squared lengths of the GS-orthogonalized basis vectors `|bi*|^2`,
/// and above it the GS-coefficients `mu_ij = <bi, bj*> / <bj*, bj*>`.
/// 
fn lll_raw<R, I, V, T>(ring: R, int_ring: I, mut gso: SubmatrixMut<V, El<R>>, mut col_ops: T, delta: &El<R>)
    where R: RingStore + Copy,
        R::Type: LLLRealField<I::Type>,
        I: IntegerRingStore + Copy,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<R>>,
        T: TransformTarget<I::Type>
{
    let mut i = 0;
    while i + 1 < gso.col_count() {
        let (target, matrix) = gso.reborrow().split_cols((i + 1)..(i + 2), 0..(i + 1));
        size_reduce(ring, int_ring, target, i + 1, matrix.as_const(), &mut col_ops);
        if ring.is_gt(
            &ring.mul_ref_snd(
                ring.sub_ref_fst(delta, ring.mul_ref(gso.as_const().at(i, i + 1), gso.as_const().at(i, i + 1))),
                gso.as_const().at(i, i)
            ),
            gso.as_const().at(i + 1, i + 1)
        ) {
            col_ops.swap(int_ring.get_ring(), i, i + 1);
            swap_gso_cols(ring, gso.reborrow(), i, i + 1);
            i = max(i, 1) - 1;
        } else {
            i += 1;
        }
    }
}

///
/// Computes the LDL-decomposition of the given matrix, i.e. writes it as
/// a product `L * D * L^T`, where `D` is diagonal and `L` is lower triangle.
/// 
/// Currently this requires that the input matrix is invertible (or in the 
/// floating point case that no eigenvalues are very small in absolute value).
/// 
/// `D` is returned on the diagonal of the matrix, and `L^T` is returned in
/// the upper triangle of the matrix.
/// 
fn ldl<R, V>(ring: R, mut matrix: SubmatrixMut<V, El<R>>)
    where R: RingStore,
        R::Type: Field, 
        V: AsPointerToSlice<El<R>>
{
    // only the upper triangle part of matrix is used
    assert_eq!(matrix.row_count(), matrix.col_count());
    let n = matrix.row_count();
    for i in 0..n {
        let pivot = ring.clone_el(matrix.at(i, i));
        let pivot_inv = ring.div(&ring.one(), matrix.at(i, i));
        for j in (i + 1)..n {
            ring.mul_assign_ref(matrix.at(i, j), &pivot_inv);
        }
        for k in (i + 1)..n {
            for l in k..n {
                let subtract = ring.mul_ref_snd(ring.mul_ref(matrix.as_const().at(i, k), matrix.as_const().at(i, l)), &pivot);
                ring.sub_assign(matrix.at(k, l), subtract);
            }
        }
    }
}

pub fn lll<I, V>(ring: I, matrix: SubmatrixMut<V, El<I>>, delta: f64)
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<I>>
{
    assert!(delta < 1.);
    assert!(delta > 0.25);
    let lll_reals = Real64::RING;
    // let delta_int = ring.from_float_approx(delta * 2f64.powi(20)).unwrap();
    // let delta = lll_reals.div(&lll_reals.get_ring().from_integer(delta_int, ring.get_ring()), &lll_reals.get_ring().from_integer(ring.power_of_two(20), ring.get_ring()));

    let n = matrix.col_count();
    let mut gso_data = (0..n).flat_map(|_i| (0..n).map(|_j| lll_reals.zero())).collect::<Vec<_>>();
    let mut gso = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut gso_data, n, n);
    matmul_fst_transposed(&matrix, &matrix, gso.reborrow(), &lll_reals.can_hom(&ring).unwrap());
    ldl(&lll_reals, gso.reborrow());
    println!("{}", gso.format(&lll_reals));
    lll_raw::<_, _, _, TransformLatticeBasis<I, _>>(&lll_reals, &ring, gso, TransformLatticeBasis { basis: matrix }, &delta);
}

struct TransformLatticeBasis<'a, I, V>
    where I: IntegerRingStore,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<I>>
{
    basis: SubmatrixMut<'a, V, El<I>>
}

impl<'a, I, V> TransformTarget<I::Type> for TransformLatticeBasis<'a, I, V>
    where I: IntegerRingStore,
            I::Type: IntegerRing,
            V: AsPointerToSlice<El<I>>
{
    fn transform(&mut self, ring: &I::Type, i: usize, j: usize, transform: &[El<I>; 4]) {
        assert!(i != j);
        for k in 0..self.basis.row_count() {
            let a = ring.clone_el(self.basis.at(k, i));
            let b = ring.clone_el(self.basis.at(k, j));
            *self.basis.at(k, i) = ring.add(ring.mul_ref(&a, &transform[0]), ring.mul_ref(&b, &transform[1]));
            *self.basis.at(k, i) = ring.add(ring.mul_ref_snd(a, &transform[2]), ring.mul_ref_snd(b, &transform[3]));
        }
    }

    fn subtract(&mut self, ring: &I::Type, src: usize, dst: usize, factor: &<I::Type as RingBase>::Element) {
        assert!(src != dst);
        for k in 0..self.basis.row_count() {
            let subtract = ring.mul_ref(self.basis.at(k, src), factor);
            ring.sub_assign(self.basis.at(k, dst), subtract);
        }
        println!("Subtract");
        println!("{}", self.basis.format(&RingRef::new(ring)));
        println!();
    }

    fn swap(&mut self, _ring: &I::Type, i: usize, j: usize) {
        if i == j {
            return;
        }
        let col_count = self.basis.col_count();
        let (mut col_i, mut col_j) = self.basis.reborrow().split_cols(i..(i + 1), j..(j + 1));
        for k in 0..col_count {
            std::mem::swap(col_i.at(k, 0), col_j.at(k, 0));
        }
        println!("Swap");
        println!("{}", self.basis.format(&RingRef::new(_ring)));
        println!();
    }
}

#[cfg(test)]
use crate::vector::*;
#[cfg(test)]
use crate::algorithms;
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
}

#[test]
fn test_ldl() {
    let mut data = in_QQ![
        [1, 2, 1],
        [2, 5, 0],
        [1, 0, 7]
    ];
    let mut matrix = SubmatrixMut::<[_; 3], _>::new(&mut data);
    let mut expected = in_QQ![
        [1, 2, 1],
        [0, 1, -2],
        [0, 0, 2]
    ];
    ldl(QQ, matrix.reborrow());

    // only the upper triangle is filled
    expected[1][0] = *matrix.at(1, 0);
    expected[2][0] = *matrix.at(2, 0);
    expected[2][1] = *matrix.at(2, 1);

    assert_matrix_eq!(&QQ, &expected, &matrix);
}

#[test]
fn test_swap_gso_cols() {
    let mut matrix = in_QQ![
        [2, 1/2, 2/5],
        [0, 3/2, 1/4],
        [0,   0,   1]
    ];
    let expected = in_QQ![
        [2, 1/2, 31/80],
        [0, 3/2, 11/40],
        [0,   0,     1]
    ];
    let matrix_view = SubmatrixMut::<[_; 3], _>::new(&mut matrix);

    swap_gso_cols(&QQ, matrix_view, 0, 1);

    assert_matrix_eq!(&QQ, &expected, &matrix);
}

#[cfg(test)]
fn norm_squared<V>(col: &Column<V, i64>) -> i64
    where V: AsPointerToSlice<i64>
{
    StaticRing::<i64>::RING.sum((0..col.len()).map(|i| col.at(i) * col.at(i)))
}

#[cfg(test)]
fn assert_lattice_isomorphic<V, const N: usize, const M: usize>(lhs: &[[i64; M]; N], rhs: &Submatrix<V, i64>)
    where V: AsPointerToSlice<i64>
{
    use crate::matrix::dense::DenseMatrix;

    assert_eq!(rhs.row_count(), N);
    assert_eq!(rhs.col_count(), M);
    let ZZbig = BigIntRing::RING;
    let mut A = DenseMatrix::zero(N, M, ZZbig);
    let mut B = DenseMatrix::zero(N, M, ZZbig);
    let int_to_ZZbig = ZZbig.can_hom(&StaticRing::<i64>::RING).unwrap();
    for i in 0..N {
        for j in 0..M {
            *A.at_mut(i, j) = int_to_ZZbig.map(lhs[i][j]);
            *B.at_mut(i, j) = int_to_ZZbig.map(*rhs.at(i, j));
        }
    }
    assert!(algorithms::smith::solve_right(&mut A.clone_matrix(&ZZbig), B.clone_matrix(&ZZbig), &ZZbig).is_some());
    assert!(algorithms::smith::solve_right(&mut B.clone_matrix(&ZZbig), A.clone_matrix(&ZZbig), &ZZbig).is_some());
}

#[test]
fn test_lll_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let original = [
        [5,   9],
        [11, 20]
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<[_; 2], _>::new(&mut reduced);
    lll(&ZZ, reduced_matrix.reborrow(), 0.9);

    assert_lattice_isomorphic(&original, &reduced_matrix.as_const());
    assert_eq!(1, norm_squared(&reduced_matrix.as_const().col_at(0)));
    assert_eq!(1, norm_squared(&reduced_matrix.as_const().col_at(1)));

    let original = [
        [10, 8],
        [27, 22]
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<[_; 2], _>::new(&mut reduced);
    lll(&ZZ, reduced_matrix.reborrow(), 0.9);

    assert_lattice_isomorphic(&original, &reduced_matrix.as_const());
    assert_eq!(4, norm_squared(&reduced_matrix.as_const().col_at(0)));
    assert_eq!(5, norm_squared(&reduced_matrix.as_const().col_at(1)));
}

#[test]
fn test_lll_3d() {
    let ZZ = StaticRing::<i64>::RING;
    // in this case, the shortest vector is shorter than half the second successive minimum,
    // so LLL will find it (for delta = 0.9 > 0.75)
    let original = [
        [72, 0, 0],
        [0,  9, 0],
        [8432, 7344, 16864]
    ];
    let _expected = [
        [144, 72, 72],
        [0, 279, -72],
        [0,   0, 272]
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<[_; 3], _>::new(&mut reduced);
    lll(&ZZ, reduced_matrix.reborrow(), 0.999);

    assert_lattice_isomorphic(&original, &reduced_matrix.as_const());
    assert_eq!(144 * 144, norm_squared(&reduced_matrix.as_const().col_at(0)));
    assert_eq!(72 * 72 + 279 * 279, norm_squared(&reduced_matrix.as_const().col_at(1)));
    assert_eq!(72 * 72 * 2 + 272 * 272, norm_squared(&reduced_matrix.as_const().col_at(2)));
}

#[test]
fn test_lll_10d() {
    let ZZ = StaticRing::<i64>::RING;

    let original = [
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
    ];
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
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<[_; 10], _>::new(&mut reduced);
    lll(&ZZ, reduced_matrix.reborrow(), 0.9);

    assert_lattice_isomorphic(&original, &reduced_matrix.as_const());
    assert!(16 * 16 > norm_squared(&reduced_matrix.as_const().col_at(0)));
}