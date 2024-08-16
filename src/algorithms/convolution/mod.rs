use std::alloc::{Allocator, Global};

use crate::ring::*;
use crate::seq::subvector::SubvectorView;
use crate::seq::VectorView;

use karatsuba::*;

pub mod karatsuba;

pub mod fftconv;

///
/// Trait for objects that can compute a convolution over a fixed ring.
/// 
#[stability::unstable(feature = "enable")]
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
    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S);
}

impl<'a, R: ?Sized + RingBase, C: ConvolutionAlgorithm<R>> ConvolutionAlgorithm<R> for &'a C {

    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        (**self).compute_convolution(lhs, rhs, dst, ring)
    }
}

#[stability::unstable(feature = "enable")]
#[derive(Clone, Copy)]
pub struct KaratsubaAlgorithm<A: Allocator = Global> {
    allocator: A
}

#[stability::unstable(feature = "enable")]
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