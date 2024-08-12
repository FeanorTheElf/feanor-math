
use strassen::*;

use crate::matrix::*;
use crate::ring::*;

use std::alloc::Allocator;
use std::alloc::Global;

pub mod strassen;

///
/// Trait to allow rings to provide specialized implementations for inner products, i.e.
/// the sums `sum_i a[i] * b[i]`.
/// 
#[stability::unstable(feature = "enable")]
pub trait ComputeInnerProduct: RingBase {

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element;
}

impl<R: ?Sized + RingBase> ComputeInnerProduct for R {

    default fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product(els.map(|(l, r)| (self.clone_el(l), r)))
    }

    default fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product_ref_fst(els.map(|(l, r)| (l, self.clone_el(r))))
    }

    default fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        self.sum(els.map(|(l, r)| self.mul(l, r)))
    }
}

///
/// Trait for objects that can compute a matrix multiplications over a fixed ring.
/// 
#[stability::unstable(feature = "enable")]
pub trait MatmulAlgorithm<R: ?Sized + RingBase> {

    ///
    /// Computes the matrix product of `lhs` and `rhs`, and adds the result to `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] += sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    /// 
    fn add_matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>;
         
    ///
    /// Computes the matrix product of `lhs` and `rhs`, and stores the result in `dst`.
    /// 
    /// This requires that `lhs` is a `nxk` matrix, `rhs` is a `kxm` matrix and `dst` is a `nxm` matrix.
    /// In this case, the function concretely computes `dst[i, j] = sum_l lhs[i, l] * rhs[l, j]` where
    /// `l` runs from `0` to `k - 1`.
    ///    
    fn matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(&self, lhs: TransposableSubmatrix<V1, R::Element, T1>, rhs: TransposableSubmatrix<V2, R::Element, T2>, mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, ring: &R)
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        for i in 0..dst.row_count() {
            for j in 0..dst.col_count() {
                *dst.at_mut(i, j) = ring.zero();
            }
        }
        self.add_matmul(lhs, rhs, dst, ring);
    }
}

///
/// Trait to allow rings to customize the parameters with which [`StrassenAlgorithm`] will
/// compute matrix multiplications.
/// 
#[stability::unstable(feature = "enable")]
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

#[stability::unstable(feature = "enable")]
pub const STANDARD_MATMUL: StrassenAlgorithm = StrassenAlgorithm::new(Global);

#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
pub struct StrassenAlgorithm<A: Allocator = Global> {
    allocator: A
}

impl<A: Allocator> StrassenAlgorithm<A> {

    #[stability::unstable(feature = "enable")]
    pub const fn new(allocator: A) -> Self {
        Self { allocator }
    }
}

impl<R: ?Sized + RingBase, A: Allocator> MatmulAlgorithm<R> for StrassenAlgorithm<A> {

    fn add_matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: &R
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        strassen::<_, _, _, _, _, T1, T2, T3>(true, <_ as StrassenHint>::strassen_threshold(ring), lhs, rhs, dst, RingRef::new(ring), &self.allocator)
    }

    fn matmul<V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
        &self,
        lhs: TransposableSubmatrix<V1, R::Element, T1>,
        rhs: TransposableSubmatrix<V2, R::Element, T2>,
        dst: TransposableSubmatrixMut<V3, R::Element, T3>,
        ring: &R
    )
        where V1: AsPointerToSlice<R::Element>,
            V2: AsPointerToSlice<R::Element>,
            V3: AsPointerToSlice<R::Element>
    {
        strassen::<_, _, _, _, _, T1, T2, T3>(false, <_ as StrassenHint>::strassen_threshold(ring), lhs, rhs, dst, RingRef::new(ring), &self.allocator)
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
            TransposableSubmatrix::from(lhs.data()), 
            TransposableSubmatrix::from(rhs.data()), 
            TransposableSubmatrixMut::from(result.data_mut()), 
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
            TransposableSubmatrix::from(lhs.data()), 
            TransposableSubmatrix::from(rhs.data()), 
            TransposableSubmatrixMut::from(result.data_mut()), 
            StaticRing::<BenchInt>::RING, 
            &Global
        );
        assert_eq!((BENCH_SIZE * (BENCH_SIZE + 1) * (BENCH_SIZE * 2 + 1) / 6 - BENCH_SIZE * BENCH_SIZE) as BenchInt, *result.at(0, 0));
    });
}