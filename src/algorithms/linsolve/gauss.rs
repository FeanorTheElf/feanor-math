use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::local::{PrincipalLocalRing, PrincipalLocalRingStore};
use crate::matrix::{AsPointerToSlice, SubmatrixMut, TransposableSubmatrixMut};
use crate::ring::*;

///
/// Computes the largest square submatrix of `A` that has nonzero determinant. In particular,
/// if the ring is a field, this computes the rank of `A`.
/// 
/// The result are two sorted lists `rows_idxs` and `col_idxs` both of same length `k` such 
/// that the submatrix is of size `k x k` and has entries `A[row_idxs[i], col_idxs[j]]` for `i, j < k`.
/// 
/// This function changes the content of `A` in an unspecified way.
/// 
/// While this functionality seems to be well-defined for arbitrary rings, the current algorithm
/// (based on gaussian elimination) only works for (possibly non-integral) valuation rings, since it
/// relies on divisibility inducing a total order.
/// 
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::rings::local::*;
/// # use feanor_math::algorithms;
/// # use feanor_math::matrix::SubmatrixMut;
/// let ZZ = StaticRing::<i64>::RING;
/// let QQ = RationalField::new(ZZ);
/// // due to a suboptimal interface of `RationalField`, it currently doesn't implement `PrincipalLocalRing`;
/// // Thus we must use a wrapper
/// let QQ = AsLocalPIR::from_field(QQ);
/// let hom = QQ.can_hom(&ZZ).unwrap();
/// let (row_idxs, col_idxs) = algorithms::linsolve::gauss::largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut [vec![hom.map(0), hom.map(1)], vec![hom.map(0), hom.map(0)]]), QQ);
/// let rank = row_idxs.len();
/// assert_eq!(rank, col_idxs.len());
/// assert_eq!(vec![0], row_idxs);
/// assert_eq!(vec![1], col_idxs);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn largest_nonzero_minor<R, V>(A: SubmatrixMut<V, El<R>>, ring: R) -> (Vec<usize>, Vec<usize>)
    where R: RingStore + Copy,
        R::Type: PrincipalLocalRing,
        V: AsPointerToSlice<El<R>>
{
    assert!(ring.is_noetherian());
    assert!(ring.is_commutative());
    let n = A.row_count();
    let m = A.col_count();

    fn largest_nonzero_minor_impl<R, V, const T: bool>(mut A: TransposableSubmatrixMut<V, El<R>, T>, ring: R) -> (Vec<usize>, Vec<usize>)
        where R: RingStore + Copy,
            R::Type: PrincipalLocalRing,
            V: AsPointerToSlice<El<R>>
    {
        let n = A.row_count();
        let m = A.col_count();
        assert!(n >= m);
        let mut row_perm: Vec<usize> = (0..n).collect();
        let mut col_perm: Vec<usize> = (0..m).collect();

        fn swap_rows<V, E, const TRANSPOSED: bool>(mut A: TransposableSubmatrixMut<V, E, TRANSPOSED>, i: usize, j: usize)
            where V: AsPointerToSlice<E>
        {
            if i == j {
                return;
            }
            let m = A.col_count();
            let (mut fst, mut snd) = A.reborrow().split_rows(i..(i + 1), j..(j + 1));
            for l in 0..m {
                std::mem::swap(fst.at_mut(0, l), snd.at_mut(0, l));
            }
        }

        fn swap_cols<V, E, const TRANSPOSED: bool>(mut A: TransposableSubmatrixMut<V, E, TRANSPOSED>, i: usize, j: usize)
            where V: AsPointerToSlice<E>
        {
            if i == j {
                return;
            }
            let n = A.row_count();
            let (mut fst, mut snd) = A.reborrow().split_cols(i..(i + 1), j..(j + 1));
            for k in 0..n {
                std::mem::swap(fst.at_mut(k, 0), snd.at_mut(k, 0));
            }
        }

        fn elim_row<V, R, const TRANSPOSED: bool>(mut A: TransposableSubmatrixMut<V, El<R>, TRANSPOSED>, pivot: usize, src: usize, dst: usize, ring: R)
            where R: RingStore,
                R::Type: DivisibilityRing,
                V: AsPointerToSlice<El<R>>
        {
            let m = A.col_count();
            let (src, mut dst) = A.reborrow().split_rows(src..(src + 1), dst..(dst + 1));
            let factor = ring.checked_div(dst.at(0, pivot), src.at(0, pivot)).unwrap();
            for l in 0..m {
                ring.sub_assign(dst.at_mut(0, l), ring.mul_ref(&factor, src.at(0, l)));
            }
        }

        let mut i = 0;
        while i < m {
            let (pivot_i, pivot_j) = (i..n).flat_map(|k| (i..m).map(move |l| (k, l))).min_by_key(|(k, l)| ring.valuation(A.at(*k, *l)).unwrap_or(usize::MAX)).unwrap();
            if ring.valuation(A.at(pivot_i, pivot_j)).is_none() {
                break;
            }
            row_perm.swap(i, pivot_i);
            col_perm.swap(i, pivot_j);
            swap_rows(A.reborrow(), i, pivot_i);
            swap_cols(A.reborrow(), i, pivot_j);
            for k in (i + 1)..n {
                elim_row(A.reborrow(), i, i, k, &ring);
            }
            i += 1;
        }

        let mut k = 0;
        let mut current = ring.clone_el(A.at(0, 0));
        while (k + 1) < m && !ring.is_zero(&current) {
            k += 1;
            ring.mul_assign_ref(&mut current, A.at(k, k));
        }
        if !ring.is_zero(&current) {
            k += 1;
        }
        let mut row_result = row_perm;
        _ = row_result.drain(k..);
        let mut col_result = col_perm;
        _ = col_result.drain(k..);
        row_result.sort_unstable();
        col_result.sort_unstable();
        return (row_result, col_result);
    }

    if n >= m {
        return largest_nonzero_minor_impl(TransposableSubmatrixMut::from(A), ring);
    } else {
        let (col_res, row_res) = largest_nonzero_minor_impl(TransposableSubmatrixMut::from(A).transpose(), ring);
        return (row_res, col_res);
    }
}

#[cfg(test)]
use crate::rings::local::AsLocalPIR;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn64B;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;
#[cfg(test)]
use crate::homomorphism::Homomorphism;

#[test]
fn test_largest_nonzero_minor_field() {
    let field = Fp::<17>::RING;

    let mut matrix = [vec![1, 0], vec![1, 0]];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), field);
    assert_eq!(1, rows.len());
    assert_eq!(vec![0], cols);

    let mut matrix = [vec![0, 0], vec![0, 1]];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), field);
    assert_eq!(vec![1], rows);
    assert_eq!(vec![1], cols);

    let mut matrix = [vec![1, 2, 3], vec![1, 2, 3], vec![2, 3, 4]];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), field);
    assert!(rows == vec![0, 2] || rows == vec![1, 2]);
    assert_eq!(2, cols.len());

    let mut matrix = [
        vec![15,  3,  9, 15,  9,],
        vec![10,  6,  7,  3,  9,],
        vec![ 2, 14, 14,  8,  6,],
        vec![12, 16,  8,  6, 16,],
        vec![15,  4, 14,  1, 11,]
    ];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), field);
    assert_eq!(3, rows.len());
    assert_eq!(3, cols.len());
}

#[test]
fn test_largest_nonzero_minor_localpir() {
    let ring = AsLocalPIR::from_zn(Zn64B::new(8)).unwrap();
    let i = |x: i32| ring.int_hom().map(x);

    let mut matrix = [vec![i(0), i(0)], vec![i(0), i(1)]];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), ring);
    assert_eq!(vec![1], rows);
    assert_eq!(vec![1], cols);

    let mut matrix = [vec![i(4), i(0), i(0)], vec![i(0), i(0), i(2)], vec![i(0), i(1), i(0)]];
    let (rows, cols) = largest_nonzero_minor(SubmatrixMut::<Vec<_>, _>::from_2d(&mut matrix), ring);
    assert!((&vec![0, 2], &vec![0, 1]) == (&rows, &cols) || (&vec![1, 2], &vec![1, 2]) == (&rows, &cols));
}