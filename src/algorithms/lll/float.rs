use std::cmp::max;

use crate::algorithms::lll::assert_lattice_isomorphic;
use crate::algorithms::lll::norm_squared;
use crate::algorithms::matmul::*;
use crate::field::*;
use crate::integer::*;
use crate::homomorphism::*;
use crate::matrix::*;
use crate::matrix::transform::TransformTarget;
use crate::primitive_int::StaticRing;
use crate::seq::*;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::ring::*;

struct NotEnoughPrecision;

struct GSOMatrix<'a, I, R, V1, V2, V3>
    where I: ?Sized + RingBase,
        R: ?Sized + Field + OrderedRing,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    /// The quadratic form to be reduced
    quadratic_form: SubmatrixMut<'a, V1, I::Element>,
    /// Cholesky-decomposition of the upper left part of `quadratic_form`;
    /// stored in the upper triangle
    cholesky: SubmatrixMut<'a, V2, R::Element>,
    /// Bound on the error of the Cholesky-decomposition, relative to the pivot;
    /// in other words, `E[i, j]` is an upper bound on `|C[i, j] - C*[i, j]| / C[i, i]`
    /// for `j >= i`; only the upper triangle is used
    error_bound: SubmatrixMut<'a, V3, R::Element>
} 

///
/// Size-reduces the `i`-th basis vector, implicitly defined by
/// the quadratic form `A`. 
/// 
/// This will fill in the entries `C[..i, i]` in the partial
/// Cholesky decomposition `C` and the corresponding entries in
/// the error bound matrix.
/// 
fn size_reduce<I, R, H, V1, V2, V3>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: H,
    precision_bound: &R::Element
) -> Result<(), NotEnoughPrecision>
    where I: ?Sized + RingBase,
        R: ?Sized + Field + OrderedRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    return Ok(());
}

enum LovaszCondition {
    Satisfied, Swap
}

///
/// Checks whether the Lovasz-condition is satisfied for the `(i - 1)`-th
/// and `i`-th basis vectors, implicitly defined by the quadratic form `A`.
/// 
/// This will fill in the entry `C[i, i]` in the partial Cholesky decomposition
/// and the corresponding entry in the error bound matrix.
/// 
/// Note that here we handle numerical issues differently than in [`size_reduce()`].
/// In particular, even if the precision of the Cholesky decomposition is not
/// enough to determine for sure whether the Lovasz condition is satisfied or
/// not, we just work with the best guess. The reason is that, in the LLL book,
/// Damien Stehlé says, in practice, this almost never causes problems.
/// 
fn check_lovasz_condition<I, R, H, V1, V2, V3>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: H,
    delta: &R::Element
) -> LovaszCondition
    where I: ?Sized + RingBase,
        R: ?Sized + Field + OrderedRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    return unimplemented!();
}

///
/// Swaps the `(i - 1)`-th and `i`-th rows, and then the `(i - 1)`-th and `i`-th columns.
/// 
fn swap_basis_vectors<I, V1>(
    A: SubmatrixMut<V1, I::Element>,
    i: usize,
    ring: &I
)
    where I: ?Sized + RingBase,
        V1: AsPointerToSlice<I::Element>,
{
    unimplemented!()
}

///
/// LLL-reduces the given positive semidefinite quadratic form `A`.
/// 
/// Note that the algorithm may return [`NotEnoughPrecision`], if it cannot
/// prove that the result is `(delta, eta)`-LLL-reduced. However, it will usually
/// be quite reduced already, and may even be `(delta, eta)`-LLL-reduced.
/// 
/// # Algorithm and numerical stability
/// 
/// The used algorithm is a custom variant of the L^2 algorithm by
/// Nguyen and Stehlé. More concretely, it internally computes with
/// floating point numbers, with the following notable features:
///  - The GSO coefficients are always recomputed after a column swap.
///    This is necessary, since continuously adjusting them fundamentally
///    has very bad numerical properties. For this purpose, a variant
///    of the Cholesky decomposition that keeps track of the current error
///    is used.
///  - It may happen that the error after computing the Cholesky decomposition
///    is too large to compute a `(delta, eta)`-LLL reduced basis. In that
///    case, [`NotEnoughPrecision`] is returned. If you are using arbitrary
///    precision floating point numbers, it can make sense to increase the precision
///    and try again.
///
fn lll_float<I, R, H, V1>(
    mut A: SubmatrixMut<V1, I::Element>,
    h: H,
    delta: &R::Element,
    eta: &R::Element
) -> Result<(), NotEnoughPrecision>
    where I: ?Sized + RingBase,
        R: ?Sized + Field + OrderedRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>
{
    assert!(!h.domain().get_ring().is_approximate());
    let RR = h.codomain();
    let n = A.row_count();
    assert_eq!(n, A.col_count());
    let half = RR.div(&RR.one(), &RR.int_hom().map(2));
    assert!(RR.is_gt(eta, &half));
    assert!(RR.is_lt(delta, &RR.one()));
    assert!(RR.is_gt(delta, &RR.mul_ref(eta, eta)));
    let strict_delta = RR.mul_ref_snd(RR.add_ref_fst(delta, RR.one()), &half);
    let precision_bound = RR.sub_ref_fst(eta, half);

    let mut C = OwnedMatrix::zero(n, n, RR);
    let mut E = OwnedMatrix::zero(n, n, RR);
    let mut gso = GSOMatrix {
        cholesky: C.data_mut(),
        error_bound: E.data_mut(),
        quadratic_form: A
    };

    let mut i = 1;
    let mut remaining_swaps = 100000;
    while i < n {
        size_reduce(&mut gso, i, &h, &precision_bound)?;
        match check_lovasz_condition(&mut gso, i, &h, &strict_delta) {
            LovaszCondition::Swap if remaining_swaps == 0 => {
                return Err(NotEnoughPrecision);
            }
            LovaszCondition::Swap => {
                remaining_swaps -= 1;
                swap_basis_vectors(A.reborrow(), i - 1, h.domain().get_ring());
                i = max(i, 2) - 1;
            },
            LovaszCondition::Satisfied => {
                i += 1;
            }
        }
    }

    // check if it is indeed (delta, eta)-LLL-reduced, considering errors

    return Ok(());
}

#[test]
fn test_lll_float_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let original = [
        DerefArray::from([5,   9]),
        DerefArray::from([11, 20])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_float(&ZZ, OwnedMatrix::<_>::identity(2, 2, Real64::RING).data(), reduced_matrix.reborrow(), 0.9, Global).unwrap();

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));

    let original = [
        DerefArray::from([10, 8]),
        DerefArray::from([27, 22])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 2>, _>::from_2d(&mut reduced);
    lll_float(&ZZ, OwnedMatrix::<_>::identity(2, 2, Real64::RING).data(), reduced_matrix.reborrow(), 0.9, Global).unwrap();

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(4, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
}

#[test]
fn test_lll_float_3d() {
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
    lll_float(&ZZ, OwnedMatrix::<_>::identity(3, 3, Real64::RING).data(), reduced_matrix.reborrow(), 0.999, Global).unwrap();

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert_eq!(144 * 144, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(72 * 72 + 279 * 279, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_eq!(72 * 72 * 2 + 272 * 272, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
}

#[test]
fn test_lll_precision() {
    let ZZ = StaticRing::<i128>::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([65208, 1, 0, 0, 0]),
        DerefArray::from([0, 65208, 1, 0, 0]),
        DerefArray::from([0, 0, 65208, 1, 0]),
        DerefArray::from([0, 0, 0, 65208, 999769]),
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 5>, _>::from_2d(&mut reduced);
    lll_float(&ZZ, OwnedMatrix::<_>::identity(5, 5, Real64::RING).data(), reduced_matrix.reborrow(), 0.999, Global).unwrap();

    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 200);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 400);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 500);
    
    let ZZ = StaticRing::<i64>::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([-3085729, 1, 0, 0, 0]),
        DerefArray::from([0, -3085729, 1, 0, 0]),
        DerefArray::from([0, 0, -3085729, 1, 0]),
        DerefArray::from([0, 0, 0, -3085729, 23068673]),
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 5>, _>::from_2d(&mut reduced);
    lll_float(&ZZ, OwnedMatrix::<_>::identity(5, 5, Real64::RING).data(), reduced_matrix.reborrow(), 0.999, Global).unwrap();
    
    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 500);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 900);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 1200);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 1300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 2000);

    
    let ZZ = StaticRing::<i128>::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([207432708, 1, 0, 0, 0]),
        DerefArray::from([0, 207432708, 1, 0, 0]),
        DerefArray::from([0, 0, 207432708, 1, 0]),
        DerefArray::from([0, 0, 0, 207432708, 447741953]),
    ];
    
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 5>, _>::from_2d(&mut reduced);
    lll_float(&ZZ, OwnedMatrix::<_>::identity(5, 5, Real64::RING).data(), reduced_matrix.reborrow(), 0.9, Global).unwrap();
    
    assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 1800);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 1800);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 4600);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 4600);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 5600);
}

#[bench]
fn bench_lll_float_10d(bencher: &mut Bencher) {
    let ZZ = StaticRing::<i64>::RING;

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
        let original = [
            DerefArray::from([       1,        0,        0,        0,        0,        0,        0,        0,        0,        0]),
            DerefArray::from([       0,        1,        0,        0,        0,        0,        0,        0,        0,        0]),
            DerefArray::from([       0,        0,        1,        0,        0,        0,        0,        0,        0,        0]),
            DerefArray::from([       0,        0,        0,        1,        0,        0,        0,        0,        0,        0]),
            DerefArray::from([       0,        0,        0,        0,        1,        0,        0,        0,        0,        0]),
            DerefArray::from([       0,        0,        0,        0,        0,        1,        0,        0,        0,        0]),
            DerefArray::from([       0,        0,        0,        0,        0,        0,        1,        0,        0,        0]),
            DerefArray::from([       2,        2,        2,        2,        0,        0,        1,        4,        0,        0]),
            DerefArray::from([       4,        3,        3,        3,        1,        2,        1,        0,        5,        0]),
            DerefArray::from([ 3433883, 14315221, 24549008,  6570781, 32725387, 33674813, 27390657, 15726308, 43003827, 43364304])
        ];
        let mut reduced = original;
        let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 10>, _>::from_2d(&mut reduced);
        lll_float(&ZZ, OwnedMatrix::<_>::identity(10, 10, Real64::RING).data(), reduced_matrix.reborrow(), 0.9, Global).unwrap();

        assert_lattice_isomorphic(ZZ, Submatrix::from_2d(&original), &reduced_matrix.as_const());
        assert!(16 * 16 > norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    });
}
