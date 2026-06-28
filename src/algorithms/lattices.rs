use tracing::instrument;

use crate::algorithms::int_bisect::bisect_floor;
use crate::algorithms::linsolve::smith::pre_smith;
use crate::algorithms::lll::exact::lll;
use crate::matrix::*;
use crate::matrix::transform::{InvertTransform, TransformCols, TransformList, TransformRows};
use crate::prelude::*;
use crate::ring_impls::fraction::FractionFieldStore;
use crate::ring_impls::rational::RationalField;
use crate::seq::VectorView;
use crate::algorithms::linsolve::LinSolveRingStore;

/// Computes a basis of the lattice `A Z^m`. The basis is returned as the columns of the result
/// matrix.
#[instrument(skip_all, level = "trace")]
pub fn lattice_eq<I, V1, V2>(ZZ: I, A: Submatrix<V1, El<I>>, B: Submatrix<V2, El<I>>) -> bool
where
    I: RingStore,
    I::Ring: IntegerRing,
    V1: AsPointerToSlice<El<I>>,
    V2: AsPointerToSlice<El<I>>,
{
    let mut tmp_A = OwnedMatrix::from_fn(A.row_count(), A.col_count(), |i, j| A.at(i, j).clone());
    let mut tmp_B = OwnedMatrix::from_fn(B.row_count(), B.col_count(), |i, j| B.at(i, j).clone());
    let mut tmp_sol = (0..(A.col_count() * B.col_count())).map(|_| ZZ.zero()).collect::<Vec<_>>();
    if !ZZ.solve_right(tmp_A.data_mut(), tmp_B.data_mut(), SubmatrixMut::from_1d(&mut tmp_sol, A.col_count(), B.col_count())).is_solved() {
        return false;
    }
    for i in 0..A.row_count() {
        for j in 0..A.col_count() {
            *tmp_A.at_mut(i, j) = A.at(i, j).clone();
        }
    }
    for i in 0..B.row_count() {
        for j in 0..B.col_count() {
            *tmp_B.at_mut(i, j) = B.at(i, j).clone();
        }
    }
    if !ZZ.solve_right(tmp_B.data_mut(), tmp_A.data_mut(), SubmatrixMut::from_1d(&mut tmp_sol, B.col_count(), A.col_count())).is_solved() {
        return false;
    }
    return true;
}

/// Computes a basis of the lattice `A Z^m`. The basis is returned as the columns of the result
/// matrix.
#[instrument(skip_all, level = "trace")]
pub fn lattice_basis_from_generating_set<I>(ZZ: I, mut A: OwnedMatrix<El<I>>) -> OwnedMatrix<El<I>>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    let QQ = RationalField::new(&ZZ);
    let delta = QQ.from_fraction(int_cast(9, &ZZ, ZZi64), int_cast(10, &ZZ, ZZi64));
    lll(A.data_mut(), QQ.inclusion(), &delta, false, &mut ());
    let zero_cols = (0..A.col_count())
        .take_while(|i| A.data().col_at(*i).as_iter().all(|x| ZZ.is_zero(x)))
        .count();
    assert!(A.col_count() - zero_cols <= A.row_count());
    return OwnedMatrix::from_fn(A.row_count(), A.col_count() - zero_cols, |i, j| {
        A.at(i, j + zero_cols).clone()
    });
}

/// Computes the "partial" and the full p-saturation of the lattice `A Z^m`, for a prime p.
///
/// Concretely, this function computes bases of the lattices L_1, L_p, L_(p^2), ..., where `L_(p^i)
/// = { x | p^i x in L }` is the preimage of L under the multiplication-by-`p^i`-map. The returned
/// iterator terminates as soon as the sequence becomes stationary, and the last element is thus the
/// standard p-saturation of L.
///
/// # Implementation
///
/// This function computes the [`pre_smith`]-form of the matrix first, and then derives every
/// L_(p^i). Note that if only the saturation is desired, you can just take the last element of the
/// iterator, and this function will avoid computing the previous partial p-saturations.
#[instrument(skip_all, level = "trace")]
pub fn lattice_p_saturation_tower<I>(
    ZZ: I,
    p: El<I>,
    mut A: OwnedMatrix<El<I>>,
) -> impl use<I> + ExactSizeIterator + DoubleEndedIterator<Item = OwnedMatrix<El<I>>>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    let mut L = TransformList::new(A.row_count());
    pre_smith(&ZZ, &mut InvertTransform::new(&mut L), &mut (), A.data_mut());
    let p_val = |n| {
        ZZ.abs_log2_floor(n).map(|bound| {
            bisect_floor(ZZi64, 0, (bound + 1).try_into().unwrap(), |e| {
                if ZZ.divides(n, &ZZ.pow(p.clone(), *e as usize)) {
                    -1
                } else {
                    1
                }
            }) as usize
        })
    };
    let max_p_val = (0..usize::min(A.row_count(), A.col_count()))
        .map(|i| p_val(A.at(i, i)))
        .max()
        .unwrap_or(None)
        .unwrap_or(0);
    (0..(max_p_val + 1)).map(move |k| {
        let p_k = ZZ.pow(p.clone(), k);
        let mut result = OwnedMatrix::from_fn(A.row_count(), A.col_count(), |i, j| {
            if i == j {
                ZZ.checked_div(A.at(i, j), &ZZ.gcd(A.at(i, j), &p_k)).unwrap()
            } else {
                ZZ.zero()
            }
        });
        L.replay_reversed(&ZZ, TransformRows::new(result.data_mut()));
        result
    })
}

/// Computes the intersection of the two lattices `A Z^n` and `B Z^m`. The basis is returned as the
/// columns of the result matrix.
#[instrument(skip_all, level = "trace")]
pub fn lattice_intersect<I>(ZZ: I, A: OwnedMatrix<El<I>>, B: OwnedMatrix<El<I>>) -> OwnedMatrix<El<I>>
where
    I: RingStore,
    I::Ring: IntegerRing,
{
    assert_eq!(A.row_count(), B.row_count());
    let k = A.row_count();
    let n = A.col_count();
    let m = B.col_count();
    let h = ZZbig.can_hom::<I>(&ZZ).unwrap();

    let mut V = OwnedMatrix::from_fn(k, n + m, |i, j| {
        if j < n {
            h.map_ref(A.at(i, j))
        } else {
            ZZbig.negate(h.map_ref(B.at(i, j - n)))
        }
    });

    let mut U = OwnedMatrix::from_fn(n + m, n + m, |i, j| if i == j { ZZbig.one() } else { ZZbig.zero() });

    let mut tracker = TransformCols::new(U.data_mut());
    let QQ = RationalField::new(&ZZbig);
    let delta = QQ.from_fraction(int_cast(9, ZZbig, ZZi64), int_cast(10, ZZbig, ZZi64));
    lll(V.data_mut(), QQ.inclusion(), &delta, false, &mut tracker);

    // Kernel columns: indices j where the j-th reduced column of V is zero.
    let kernel_cols: Vec<usize> = (0..V.col_count())
        .filter(|&j| (0..k).all(|i| ZZbig.is_zero(V.at(i, j))))
        .collect();

    let result = lattice_basis_from_generating_set(
        ZZbig,
        OwnedMatrix::from_fn(k, kernel_cols.len(), |i, j| {
            ZZbig.sum((0..n).map(|s| h.mul_ref_map(U.at(s, kernel_cols[j]), A.at(i, s))))
        }),
    );

    OwnedMatrix::from_fn(k, result.col_count(), |i, j| {
        int_cast(result.at(i, j).clone(), &ZZ, ZZbig)
    })
}

#[test]
fn test_lattice_intersect() {
    feanor_tracing::DelayedLogger::init_test();
    let A = OwnedMatrix::new(vec![1, 2, 3, 4, 5, 6], 3, 2);
    let B = OwnedMatrix::new(vec![3, 7, 11], 3, 1);
    let expected = B.clone();
    let intersection = lattice_intersect(ZZi64, A.clone(), B.clone());
    assert_eq!(1, intersection.col_count());
    assert!(lattice_eq(ZZi64, expected.data(), intersection.data()));

    let B = OwnedMatrix::new(vec![3, 7, 12], 3, 1);
    let intersection = lattice_intersect(ZZi64, A.clone(), B.clone());
    let expected = OwnedMatrix::new(Vec::new(), 3, 0);
    assert!(lattice_eq(ZZi64, expected.data(), intersection.data()));
}

#[test]
fn test_lattice_p_saturation_tower() {
    feanor_tracing::DelayedLogger::init_test();
    let A = OwnedMatrix::new(vec![54, 27, -27, 0, 3, -2, 27, 21, -18], 3, 3);
    let expected = [
        A.clone(),
        OwnedMatrix::new(vec![18, 9, -27, 0, 1, -2, 9, 7, -18], 3, 3),
        OwnedMatrix::new(vec![6, 9, -27, 0, 1, -2, 3, 7, -18], 3, 3),
        OwnedMatrix::new(vec![2, 9, -27, 0, 1, -2, 1, 7, -18], 3, 3)
    ];
    let actual = lattice_p_saturation_tower(ZZi64, 3, A).collect::<Vec<_>>();
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        assert!(lattice_eq(ZZi64, expected.data(), actual.data()));
    }
}
