use std::alloc::Global;

use crate::ring::*;
use crate::seq::subvector::SubvectorView;
use crate::seq::VectorView;
use karatsuba::*;

pub mod karatsuba;

///
/// Trait for specializing the algorithm used to compute convolutions for a ring.
/// This is default-implemented for any ring, using karatsuba's algorithm.
/// 
/// For details, see the function [`ConvolutionAlgorithm::compute_convolution()`].
/// 
#[stability::unstable(feature = "enable")]
pub trait ComputeConvolutionRing: RingBase {

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
    /// # Panics
    /// 
    /// If `dst.len() < lhs.len() + rhs.len()`, or the given ring is not supported by the algorithm.
    /// 
    fn compute_convolution<V1: VectorView<Self::Element>, V2: VectorView<Self::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [Self::Element]);
}

impl<R: ?Sized + RingBase> ComputeConvolutionRing for R {

    default fn compute_convolution<V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element]) {
        karatsuba(self.karatsuba_threshold(), dst, SubvectorView::new(&lhs), SubvectorView::new(&rhs), RingRef::new(self), &Global)
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