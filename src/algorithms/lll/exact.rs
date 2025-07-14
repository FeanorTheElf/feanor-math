use crate::algorithms::lll::float::lll_float_quadratic_form;
use crate::algorithms::matmul::*;
use crate::field::*;
use crate::integer::*;
use crate::homomorphism::*;
use crate::matrix::transform::TransformCols;
use crate::matrix::*;
use crate::matrix::transform::TransformTarget;
use crate::rings::approx_real::float::Real64;
use crate::seq::*;
use crate::rings::fraction::FractionFieldStore;
use crate::rings::rational::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::ring::*;

use std::alloc::Allocator;
use std::cmp::max;

///
/// Size-reduces `target` w.r.t. the GSO matrix, and also sends the performed
/// operations to `col_ops`.
/// 
fn size_reduce<R, I, V, T>(
    ring: R,
    mut target: SubmatrixMut<V, El<R>>, 
    target_j: usize, 
    matrix: Submatrix<V, El<R>>, 
    col_ops: &mut T
)
    where R: RingStore<Type = RationalFieldBase<I>> + Copy,
        I: RingStore,
        I::Type: IntegerRing,
        V: AsPointerToSlice<El<RationalField<I>>>,
        T: TransformTarget<R::Type>
{
    assert!(!ring.get_ring().is_approximate());
    for j in (0..matrix.col_count()).rev() {
        let target_const = target.as_const();
        let mu = target_const.at(j, 0);
        let factor = ring.base_ring().rounded_div(ring.base_ring().clone_el(ring.get_ring().num(mu)), ring.get_ring().den(mu));
        let factor = ring.inclusion().map(factor);
        col_ops.subtract(ring, j, target_j, &factor);
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
pub fn lll_exact<R, I, V1, V2, A>(
    ring: R, 
    quadratic_form: Submatrix<V1, El<R>>, 
    mut matrix: SubmatrixMut<V2, El<R>>, 
    delta: &El<R>, 
    allocator: A
)
    where R: RingStore<Type = RationalFieldBase<I>> + Copy,
        I: RingStore,
        I::Type: IntegerRing,
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        A: Allocator
{
    let n = matrix.row_count();
    assert_eq!(n, matrix.col_count());
    assert_eq!(n, quadratic_form.col_count());
    assert_eq!(n, quadratic_form.col_count());
    assert!(ring.is_lt(delta, &ring.one()));
    assert!(ring.is_gt(delta, &ring.from_fraction(ring.base_ring().one(), ring.base_ring().int_hom().map(4))));

    let n = matrix.col_count();
    let mut gram_matrix = OwnedMatrix::zero_in(n, n, &ring, &allocator);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(matrix.as_const()).transpose(), 
        TransposableSubmatrix::from(matrix.as_const()), 
        TransposableSubmatrixMut::from(gram_matrix.data_mut()), 
        &ring
    );

    let RR = Real64::RING;
    let QQ_to_RR = RR.can_hom(&ring).unwrap();
    _ = lll_float_quadratic_form(
        gram_matrix.data_mut(),
        QQ_to_RR,
        &0.999,
        &0.51,
        &mut TransformCols(matrix.reborrow(), ring.get_ring())
    );
    println!("{}", format_matrix(n, n, |i, j| matrix.at(i, j), ring));

    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(matrix.as_const()).transpose(), 
        TransposableSubmatrix::from(matrix.as_const()), 
        TransposableSubmatrixMut::from(gram_matrix.data_mut()), 
        &ring
    );
    ldl(ring, gram_matrix.data_mut()).expect("quadratic form not positive definite");
    let mut gso = gram_matrix;
    
    let mut col_ops = TransformCols(matrix, ring.get_ring());
    let mut i = 0;
    while i + 1 < gso.col_count() {

        let (target, matrix) = gso.data_mut().split_cols((i + 1)..(i + 2), 0..(i + 1));
        size_reduce(ring, target, i + 1, matrix.as_const(), &mut col_ops);

        if ring.is_gt(
            &ring.mul_ref_snd(
                ring.sub_ref_fst(delta, ring.mul_ref(gso.data().at(i, i + 1), gso.data().at(i, i + 1))),
                gso.data().at(i, i)
            ),
            gso.data().at(i + 1, i + 1)
        ) {
            col_ops.swap(ring, i, i + 1);
            swap_gso_cols(&ring, gso.data_mut(), i, i + 1);
            i = max(i, 1) - 1;
        } else {
            i += 1;
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
use crate::algorithms::lll::{assert_rational_lattice_isomorphic, norm_squared};
#[cfg(test)]
use crate::primitive_int::StaticRing;

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
            let ZZ_to_QQ = RationalField::new(StaticRing::<i64>::RING).into_inclusion();
            [
                $([$(
                    in_QQ!(ZZ_to_QQ; $num $(, $den)?)
                ),*]),*
            ]
        }
    };
    ($(DerefArray::from([$($num:literal $(/ $den:literal)?),*])),*) => {
        {
            let ZZ_to_QQ = RationalField::new(StaticRing::<i64>::RING).into_inclusion();
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
    let ZZ = StaticRing::<i64>::RING;
    let QQ = RationalField::new(ZZ);
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
    let ZZ = StaticRing::<i64>::RING;
    let QQ = RationalField::new(ZZ);
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

#[test]
fn test_lll_exact_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let QQ = RationalField::new(ZZ);
    let original = in_QQ![
        DerefArray::from([5,   9]),
        DerefArray::from([11, 20])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_exact(QQ, OwnedMatrix::identity(2, 2, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(9, 10), Global);

    assert_rational_lattice_isomorphic(QQ, RationalField::new(BigIntRing::RING), Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_el_eq!(QQ, QQ.int_hom().map(1), norm_squared(QQ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(QQ, QQ.int_hom().map(1), norm_squared(QQ, &reduced_matrix.as_const().col_at(1)));

    let original = in_QQ![
        DerefArray::from([10, 8]),
        DerefArray::from([27, 22])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_exact(QQ, OwnedMatrix::identity(2, 2, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(9, 10), Global);

    assert_rational_lattice_isomorphic(QQ, RationalField::new(BigIntRing::RING), Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_el_eq!(QQ, QQ.int_hom().map(4), norm_squared(QQ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(QQ, QQ.int_hom().map(5), norm_squared(QQ, &reduced_matrix.as_const().col_at(1)));
}

#[test]
fn test_lll_exact_3d() {
    let ZZ = StaticRing::<i64>::RING;
    let QQ = RationalField::new(ZZ);
    // in this case, the shortest vector is shorter than half the second successive minimum,
    // so LLL will find it (for delta = 0.9 > 0.75)
    let original = in_QQ![
        DerefArray::from([72, 0, 0]),
        DerefArray::from([0,  9, 0]),
        DerefArray::from([8432, 7344, 16864])
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 3>, _>::from_2d(&mut reduced);
    lll_exact(QQ, OwnedMatrix::identity(3, 3, QQ).data(), reduced_matrix.reborrow(), &QQ.from_fraction(999, 1000), Global);

    assert_rational_lattice_isomorphic(QQ, RationalField::new(BigIntRing::RING), Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_el_eq!(QQ, QQ.int_hom().map(144 * 144), norm_squared(QQ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(QQ, QQ.int_hom().map(72 * 72 + 279 * 279), norm_squared(QQ, &reduced_matrix.as_const().col_at(1)));
    assert_el_eq!(QQ, QQ.int_hom().map(72 * 72 * 2 + 272 * 272), norm_squared(QQ, &reduced_matrix.as_const().col_at(2)));
}

#[bench]
fn bench_lll_exact_10d(bencher: &mut Bencher) {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);

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
        ].into_iter().map(|row| row.into_iter().map(|x| QQ.inclusion().map(int_cast(x, ZZ, StaticRing::<i64>::RING))).collect()).collect();
        let mut reduced = original.clone();
        let mut reduced_matrix = SubmatrixMut::<Vec<_>, _>::from_2d(&mut reduced);
        let delta = QQ.from_fraction(int_cast(9, ZZ, StaticRing::<i64>::RING), int_cast(10, ZZ, StaticRing::<i64>::RING));
        lll_exact(&QQ, OwnedMatrix::identity(10, 10, &QQ).data(), reduced_matrix.reborrow(), &delta, Global);

        assert_rational_lattice_isomorphic(&QQ, &QQ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
        assert!(QQ.is_geq(&QQ.int_hom().map(16 * 16), &norm_squared(&QQ, &reduced_matrix.as_const().col_at(0))));
    });
}