
use strassen::*;

use crate::matrix::*;
use crate::ring::*;

use std::alloc::Allocator;
use std::alloc::Global;
use std::ops::Deref;

///
/// Contains [`strassen::strassen()`], an implementation of Strassen's algorithm
/// for matrix multiplication.
/// 
pub mod strassen;

///
/// Trait to allow rings to provide specialized implementations for inner products, i.e.
/// the sums `sum_i a[i] * b[i]`.
/// 
pub trait ComputeInnerProduct: RingBase {

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref<'a, I: IntoIterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref_fst<'a, I: IntoIterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product<I: IntoIterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element;
}

impl<R: ?Sized + RingBase> ComputeInnerProduct for R {

    default fn inner_product_ref_fst<'a, I: IntoIterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        let mut result = self.zero();
        for (l, r) in els {
            result = self.fma(l, &r, result);
        }
        return result;
    }

    default fn inner_product_ref<'a, I: IntoIterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        let mut result = self.zero();
        for (l, r) in els {
            result = self.fma(l, r, result);
        }
        return result;
    }

    default fn inner_product<I: IntoIterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        let mut result = self.zero();
        for (l, r) in els {
            result = self.fma(&l, &r, result);
        }
        return result;
    }
}

///
/// Trait for objects that can compute a matrix multiplications over some ring.
/// 
pub trait MatmulAlgorithm<R: ?Sized + RingBase> {

    ///
    /// Computes the matrix product of `lhs` and `rhs`, and adds the result to `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] += sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn add_matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: S)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy;
         
    ///
    /// Computes the matrix product of `lhs` and `rhs`, and stores the result in `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] = sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    ///    
    fn matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: S)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        for i in 0..dst.row_count() {
            for j in 0..dst.col_count() {
                *dst.at_mut(i, j) = ring.zero();
            }
        }
        self.add_matmul(lhs, rhs, dst, ring);
    }
}

impl<R, T> MatmulAlgorithm<R> for T
    where R: ?Sized + RingBase,
        T: Deref,
        T::Target: MatmulAlgorithm<R>
{
    fn add_matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: S)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        (**self).add_matmul(lhs, rhs, dst, ring)
    }

    fn matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: S)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        (**self).matmul(lhs, rhs, dst, ring)
    }
}

///
/// Trait to allow rings to customize the parameters with which [`StrassenAlgorithm`] will
/// compute matrix multiplications.
/// 
pub trait StrassenHint: RingBase {

    ///
    /// Define a threshold from which on [`StrassenAlgorithm`] will use the Strassen algorithm.
    /// 
    /// Concretely, when this returns `k`, [`StrassenAlgorithm`] will reduce the 
    /// matrix multipliction down to `2^k x 2^k` matrices using Strassen's algorithm,
    /// and then use naive matmul for the rest.
    /// 
    /// The value is `0`, but if the considered rings have fast multiplication (compared to addition), 
    /// then setting this higher may result in a performance gain.
    /// 
    fn strassen_threshold(&self) -> usize;
}

impl<R: RingBase + ?Sized> StrassenHint for R {

    default fn strassen_threshold(&self) -> usize {
        0
    }
}

pub const STANDARD_MATMUL: StrassenAlgorithm = StrassenAlgorithm::new(Global);

#[derive(Clone, Copy)]
pub struct StrassenAlgorithm<A: Allocator = Global> {
    allocator: A
}

impl<A: Allocator> StrassenAlgorithm<A> {

    pub const fn new(allocator: A) -> Self {
        Self { allocator }
    }
}

impl<R: ?Sized + RingBase, A: Allocator> MatmulAlgorithm<R> for StrassenAlgorithm<A> {

    fn add_matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: S
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        strassen::<_, _, _, _, _, T1, T2, T3>(true, <_ as StrassenHint>::strassen_threshold(ring.get_ring()), lhs, rhs, dst, ring, &self.allocator)
    }

    fn matmul<S, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: S
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>,
            S: RingStore<Type = R> + Copy
    {
        strassen::<_, _, _, _, _, T1, T2, T3>(false, <_ as StrassenHint>::strassen_threshold(ring.get_ring()), lhs, rhs, dst, ring, &self.allocator)
    }
}

///
/// Computes `dst = lhs * rhs` if `ADD_ASSIGN = false` and `dst += lhs * rhs` if `ADD_ASSIGN = true`,
/// using the standard cubic formula for matrix multiplication. 
/// 
/// This implementation is very simple and not very optimized. Usually it is used as a fallback
/// for more sophisticated implementations. 
/// 
#[stability::unstable(feature = "enable")]
pub fn naive_matmul<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
    lhs: TransposableSubmatrix<V1, El<R>, T1>, 
    rhs: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    assert_eq!(lhs.row_count(), dst.row_count());
    assert_eq!(rhs.col_count(), dst.col_count());
    assert_eq!(lhs.col_count(), rhs.row_count());
    for i in 0..lhs.row_count() {
        for j in 0..rhs.col_count() {
            let inner_prod = <_ as ComputeInnerProduct>::inner_product_ref(ring.get_ring(), (0..lhs.col_count()).map(|k| (lhs.at(i, k), rhs.at(k, j))));
            if ADD_ASSIGN {
                ring.add_assign(dst.at_mut(i, j), inner_prod);
            } else {
                *dst.at_mut(i, j) = inner_prod;
            }
        }
    }
}

#[cfg(test)]
use test;
#[cfg(test)]
use crate::primitive_int::*;

#[cfg(test)]
const BENCH_SIZE: usize = 128;
#[cfg(test)]
type BenchInt = i64;

#[bench]
fn bench_naive_matmul(bencher: &mut test::Bencher) {
    let lhs = OwnedMatrix::from_fn_in(BENCH_SIZE, BENCH_SIZE, |i, j| std::hint::black_box(i as BenchInt + j as BenchInt), Global);
    let rhs = OwnedMatrix::from_fn_in(BENCH_SIZE, BENCH_SIZE, |i, j| std::hint::black_box(i as BenchInt + j as BenchInt), Global);
    let mut result: OwnedMatrix<BenchInt> = OwnedMatrix::zero(BENCH_SIZE, BENCH_SIZE, StaticRing::<BenchInt>::RING);
    bencher.iter(|| {
        strassen::<_, _, _, _, _, false, false, false>(
            false, 
            100, 
            std::hint::black_box(TransposableSubmatrix::from(lhs.data())), 
            std::hint::black_box(TransposableSubmatrix::from(rhs.data())), 
            std::hint::black_box(TransposableSubmatrixMut::from(result.data_mut())), 
            StaticRing::<BenchInt>::RING, 
            &Global
        );
        assert_eq!((BENCH_SIZE * (BENCH_SIZE + 1) * (BENCH_SIZE * 2 + 1) / 6 - BENCH_SIZE * BENCH_SIZE) as BenchInt, *result.at(0, 0));
    });
}

#[bench]
fn bench_strassen_matmul(bencher: &mut test::Bencher) {
    let threshold_log_2 = 4;
    let lhs = OwnedMatrix::from_fn_in(BENCH_SIZE, BENCH_SIZE, |i, j| std::hint::black_box(i as BenchInt + j as BenchInt), Global);
    let rhs = OwnedMatrix::from_fn_in(BENCH_SIZE, BENCH_SIZE, |i, j| std::hint::black_box(i as BenchInt + j as BenchInt), Global);
    let mut result: OwnedMatrix<BenchInt> = OwnedMatrix::zero(BENCH_SIZE, BENCH_SIZE, StaticRing::<BenchInt>::RING);
    bencher.iter(|| {
        strassen::<_, _, _, _, _, false, false, false>(
            false, 
            threshold_log_2, 
            std::hint::black_box(TransposableSubmatrix::from(lhs.data())), 
            std::hint::black_box(TransposableSubmatrix::from(rhs.data())), 
            std::hint::black_box(TransposableSubmatrixMut::from(result.data_mut())), 
            StaticRing::<BenchInt>::RING, 
            &Global
        );
        assert_eq!((BENCH_SIZE * (BENCH_SIZE + 1) * (BENCH_SIZE * 2 + 1) / 6 - BENCH_SIZE * BENCH_SIZE) as BenchInt, *result.at(0, 0));
    });
}