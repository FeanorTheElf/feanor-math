use std::alloc::{Allocator, Global};

use crate::ring::*;
use crate::seq::subvector::SubvectorView;
use crate::seq::VectorView;

use karatsuba::*;

///
/// Contains an optimized implementation of Karatsuba's for computing convolutions
/// 
pub mod karatsuba;

///
/// Contains multiple implementations of computing convolutions using complex FFTs.
/// 
pub mod fft;

///
/// Trait for objects that can compute a convolution over some ring.
/// 
/// # Example
/// ```
/// # use std::cmp::{min, max};
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::seq::*;
/// # use feanor_math::algorithms::convolution::*;
/// struct NaiveConvolution;
/// // we support all rings!
/// impl<R: ?Sized + RingBase> ConvolutionAlgorithm<R> for NaiveConvolution {
///     fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
///         for i in 0..(lhs.len() + rhs.len() - 1) {
///             for j in max(0, i as isize - rhs.len() as isize + 1)..min(lhs.len() as isize, i as isize + 1) {
///                 ring.add_assign(&mut dst[i], ring.mul_ref(lhs.at(j as usize), rhs.at(i - j as usize)));
///             }
///         }
///     }
///     fn supports_ring<S: RingStore<Type = R>>(&self, _: S) -> bool
///         where S: Copy
///     { true }
/// }
/// let lhs = [1, 2, 3, 4, 5];
/// let rhs = [2, 3, 4, 5, 6];
/// let mut expected = [0; 10];
/// let mut actual = [0; 10];
/// STANDARD_CONVOLUTION.compute_convolution(lhs, rhs, &mut expected, StaticRing::<i64>::RING);
/// NaiveConvolution.compute_convolution(lhs, rhs, &mut actual, StaticRing::<i64>::RING);
/// assert_eq!(expected, actual);
/// ```
/// 
pub trait ConvolutionAlgorithm<R: ?Sized + RingBase> {

    ///
    /// Elementwise adds the convolution of `lhs` and `rhs` to `dst`.
    /// 
    /// In other words, computes `dst[i] += sum_j lhs[j] * rhs[i - j]` for all `i`, where
    /// `j` runs through all positive integers for which `lhs[j]` and `rhs[i - j]` are defined,
    /// i.e. not out-of-bounds.
    /// 
    /// In particular, it is necessary that `dst.len() >= lhs.len() + rhs.len() - 1`. However,
    /// to allow for more efficient implementations, it is instead required that 
    /// `dst.len() >= lhs.len() + rhs.len()`.
    /// 
    /// # Panic
    /// 
    /// Panics if `dst` is shorter than `lhs.len() + rhs.len() - 1`. May panic if `dst` is shorter
    /// than `lhs.len() + rhs.len()`.
    /// 
    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S);

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, ring: S) -> bool;
}

#[stability::unstable(feature = "enable")]
pub trait PreparedConvolutionAlgorithm<R: ?Sized + RingBase>: ConvolutionAlgorithm<R> {

    type PreparedConvolutionOperand;

    fn prepare_convolution_operand<S: RingStore<Type = R> + Copy, V: VectorView<R::Element>>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand;

    fn compute_convolution_lhs_prepared<S: RingStore<Type = R> + Copy, V: VectorView<R::Element>>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S);
    fn compute_convolution_prepared<S: RingStore<Type = R> + Copy>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S);

    fn compute_convolution_rhs_prepared<S: RingStore<Type = R> + Copy, V: VectorView<R::Element>>(&self, lhs: V, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S) {
        assert!(ring.is_commutative());
        self.compute_convolution_lhs_prepared(rhs, lhs, dst, ring);
    }
}

impl<'a, R: ?Sized + RingBase, C: ConvolutionAlgorithm<R>> ConvolutionAlgorithm<R> for &'a C {

    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        (**self).compute_convolution(lhs, rhs, dst, ring)
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, ring: S) -> bool {
        (**self).supports_ring(ring)
    }
}

///
/// Implementation of convolutions that uses Karatsuba's algorithm
/// with a threshold defined by [`KaratsubaHint`].
/// 
#[derive(Clone, Copy)]
pub struct KaratsubaAlgorithm<A: Allocator = Global> {
    allocator: A
}

///
/// Good default algorithm for computing convolutions, using Karatsuba's algorithm
/// with a threshold defined by [`KaratsubaHint`].
/// 
pub const STANDARD_CONVOLUTION: KaratsubaAlgorithm = KaratsubaAlgorithm::new(Global);

impl<A: Allocator> KaratsubaAlgorithm<A> {
    
    #[stability::unstable(feature = "enable")]
    pub const fn new(allocator: A) -> Self {
        Self { allocator }
    }
}

impl<R: ?Sized + RingBase, A: Allocator> ConvolutionAlgorithm<R> for KaratsubaAlgorithm<A> {

    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<<R as RingBase>::Element>, V2: VectorView<<R as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut[<R as RingBase>::Element], ring: S) {
        karatsuba(ring.get_ring().karatsuba_threshold(), dst, SubvectorView::new(&lhs), SubvectorView::new(&rhs), &ring, &self.allocator)
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

///
/// Trait to allow rings to customize the parameters with which [`KaratsubaAlgorithm`] will
/// compute convolutions over the ring.
/// 
#[stability::unstable(feature = "enable")]
pub trait KaratsubaHint: RingBase {

    ///
    /// Define a threshold from which on [`KaratsubaAlgorithm`] will use the Karatsuba algorithm.
    /// 
    /// Concretely, when this returns `k`, [`KaratsubaAlgorithm`] will reduce the 
    /// convolution down to ones on slices of size `2^k`, and compute their convolution naively. The default
    /// value is `0`, but if the considered rings have fast multiplication (compared to addition), then setting
    /// it higher may result in a performance gain.
    /// 
    fn karatsuba_threshold(&self) -> usize;
}

impl<R: RingBase + ?Sized> KaratsubaHint for R {

    default fn karatsuba_threshold(&self) -> usize {
        0
    }
}

#[cfg(test)]
use test;
#[cfg(test)]
use crate::primitive_int::*;

#[bench]
fn bench_naive_mul(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(10, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &Global);
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}

#[bench]
fn bench_karatsuba_mul(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(4, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &Global);
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}
