use std::alloc::{Allocator, Global};
use std::ops::Deref;

use crate::ring::*;
use crate::seq::subvector::SubvectorView;
use crate::seq::VectorView;

use karatsuba::*;

///
/// Contains an optimized implementation of Karatsuba's for computing convolutions
/// 
pub mod karatsuba;

///
/// Contains an implementation of computing convolutions using complex FFTs.
/// 
pub mod fft;

pub mod ntt;

pub mod rns;

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

    ///
    /// Returns whether this convolution algorithm supports computations of 
    /// the given ring.
    /// 
    /// Note that most algorithms will support all rings of type `R`. However in some cases,
    /// e.g. for finite fields, required data might only be precomputed for some moduli,
    /// and thus only these will be supported.
    /// 
    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, ring: S) -> bool;
}

///
/// Trait for convolution algorithms that can "prepare" one (or both) operands in advance
/// by computing additional data, and then use this data to perform the actual convolution
/// more efficiently.
/// 
#[stability::unstable(feature = "enable")]
pub trait PreparedConvolutionAlgorithm<R: ?Sized + RingBase>: ConvolutionAlgorithm<R> {

    type PreparedConvolutionOperand;

    fn prepare_convolution_operand<S, V>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>;

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>;

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy;

    fn compute_convolution_rhs_prepared<S, V>(&self, lhs: V, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        assert!(ring.is_commutative());
        self.compute_convolution_lhs_prepared(rhs, lhs, dst, ring);
    }

    fn compute_convolution_inner_product_lhs_prepared<'a, S, I, V>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: Iterator<Item = (&'a Self::PreparedConvolutionOperand, V)>,
            V: VectorView<R::Element>,
            Self: 'a,
            R: 'a,
            Self::PreparedConvolutionOperand: 'a
    {
        for (lhs, rhs) in values {
            self.compute_convolution_lhs_prepared(lhs, rhs, dst, ring)
        }
    }

    fn compute_convolution_inner_product_prepared<'a, S, I>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            Self::PreparedConvolutionOperand: 'a,
            Self: 'a,
            R: 'a,
    {
        for (lhs, rhs) in values {
            self.compute_convolution_prepared(lhs, rhs, dst, ring)
        }
    }
}

impl<'a, R, C> ConvolutionAlgorithm<R> for C
    where R: ?Sized + RingBase,
        C: Deref,
        C::Target: ConvolutionAlgorithm<R>
{
    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        (**self).compute_convolution(lhs, rhs, dst, ring)
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, ring: S) -> bool {
        (**self).supports_ring(ring)
    }
}

impl<'a, R, C> PreparedConvolutionAlgorithm<R> for C
    where R: ?Sized + RingBase,
        C: Deref,
        C::Target: PreparedConvolutionAlgorithm<R>
{
    type PreparedConvolutionOperand = <C::Target as PreparedConvolutionAlgorithm<R>>::PreparedConvolutionOperand;

    fn prepare_convolution_operand<S, V>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        (**self).prepare_convolution_operand(val, ring)
    }

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        (**self).compute_convolution_lhs_prepared(lhs, rhs, dst, ring)
    }

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy
    {
        (**self).compute_convolution_prepared(lhs, rhs, dst, ring)
    }

    fn compute_convolution_rhs_prepared<S, V>(&self, lhs: V, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        (**self).compute_convolution_rhs_prepared(lhs, rhs, dst, ring)
    }

    fn compute_convolution_inner_product_lhs_prepared<'b, S, I, V>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: Iterator<Item = (&'b Self::PreparedConvolutionOperand, V)>,
            V: VectorView<R::Element>,
            Self: 'b,
            R: 'b,
            Self::PreparedConvolutionOperand: 'b
    {
        (**self).compute_convolution_inner_product_lhs_prepared(values, dst, ring)
    }

    fn compute_convolution_inner_product_prepared<'b, S, I>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: Iterator<Item = (&'b Self::PreparedConvolutionOperand, &'b Self::PreparedConvolutionOperand)>,
            Self: 'b,
            R: 'b,
            Self::PreparedConvolutionOperand: 'b
    {
        (**self).compute_convolution_inner_product_prepared(values, dst, ring)
    }
}

///
/// Implementation of convolutions that uses Karatsuba's algorithm
/// with a threshold defined by [`KaratsubaHint`].
/// 
#[derive(Clone, Copy, Debug)]
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


#[allow(missing_docs)]
#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use std::cmp::min;
    use crate::homomorphism::*;
    use crate::ring::*;
    use super::*;

    pub fn test_convolution<C, R>(convolution: C, ring: R, scale: El<R>)
        where C: ConvolutionAlgorithm<R::Type>,
            R: RingStore
    {
        for lhs_len in [2, 3, 4, 15] {
            for rhs_len in [2, 3, 4, 15, 31, 32, 33] {
                let lhs = (0..lhs_len).map(|i| ring.mul_ref_snd(ring.int_hom().map(i), &scale)).collect::<Vec<_>>();
                let rhs = (0..rhs_len).map(|i| ring.mul_ref_snd(ring.int_hom().map(i), &scale)).collect::<Vec<_>>();
                let expected = (0..(lhs_len + rhs_len)).map(|i| if i < lhs_len + rhs_len {
                    min(i, lhs_len - 1) * (min(i, lhs_len - 1) + 1) * (3 * i - 2 * min(i, lhs_len - 1) - 1) / 6 - 
                    (i - 1 - min(i, rhs_len - 1)) * (i - min(i, rhs_len - 1)) * (i + 2 * min(i, rhs_len - 1) + 1) / 6
                } else {
                    0
                }).map(|x| ring.mul(ring.int_hom().map(x), ring.pow(ring.clone_el(&scale), 2))).collect::<Vec<_>>();
    
                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution(&lhs, &rhs, &mut actual, &ring);
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let expected = (0..(lhs_len + rhs_len)).map(|i| if i < lhs_len + rhs_len {
                    i * i +
                    min(i, lhs_len - 1) * (min(i, lhs_len - 1) + 1) * (3 * i - 2 * min(i, lhs_len - 1) - 1) / 6 - 
                    (i - 1 - min(i, rhs_len - 1)) * (i - min(i, rhs_len - 1)) * (i + 2 * min(i, rhs_len - 1) + 1) / 6
                } else {
                    0
                }).map(|x| ring.mul(ring.int_hom().map(x), ring.pow(ring.clone_el(&scale), 2))).collect::<Vec<_>>();
    
                let mut actual = Vec::new();
                actual.extend((0..(lhs_len + rhs_len)).map(|i| ring.mul(ring.int_hom().map(i * i), ring.pow(ring.clone_el(&scale), 2))));
                convolution.compute_convolution(&lhs, &rhs, &mut actual, &ring);
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn test_prepared_convolution<C, R>(convolution: C, ring: R, scale: El<R>)
        where C: PreparedConvolutionAlgorithm<R::Type>,
            R: RingStore
    {
        for lhs_len in [2, 3, 4, 14, 15] {
            for rhs_len in [2, 3, 4, 15, 31, 32, 33] {
                let lhs = (0..lhs_len).map(|i| ring.mul_ref_snd(ring.int_hom().map(i), &scale)).collect::<Vec<_>>();
                let rhs = (0..rhs_len).map(|i| ring.mul_ref_snd(ring.int_hom().map(i), &scale)).collect::<Vec<_>>();
                let expected = (0..(lhs_len + rhs_len)).map(|i| if i < lhs_len + rhs_len {
                    min(i, lhs_len - 1) * (min(i, lhs_len - 1) + 1) * (3 * i - 2 * min(i, lhs_len - 1) - 1) / 6 - 
                    (i - 1 - min(i, rhs_len - 1)) * (i - min(i, rhs_len - 1)) * (i + 2 * min(i, rhs_len - 1) + 1) / 6
                } else {
                    0
                }).map(|x| ring.mul(ring.int_hom().map(x), ring.pow(ring.clone_el(&scale), 2))).collect::<Vec<_>>();
    
                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution_prepared(
                    &convolution.prepare_convolution_operand(&lhs, &ring),
                    &convolution.prepare_convolution_operand(&rhs, &ring),
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution_lhs_prepared(
                    &convolution.prepare_convolution_operand(&lhs, &ring),
                    &rhs,
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (convolution.prepare_convolution_operand(&lhs, &ring), convolution.prepare_convolution_operand(&rhs, &ring)),
                    (convolution.prepare_convolution_operand(&[ring.one()], &ring), convolution.prepare_convolution_operand(&[ring.one()], &ring))
                ];
                convolution.compute_convolution_inner_product_prepared(
                    data.iter().map(|(l, r)| (l, r)),
                    &mut actual, 
                    &ring
                );
                assert_el_eq!(&ring, ring.add_ref_fst(&expected[0], ring.one()), &actual[0]);
                for i in 1..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (convolution.prepare_convolution_operand(&lhs, &ring), rhs),
                    (convolution.prepare_convolution_operand(&[ring.one()], &ring), vec![ring.one()])
                ];
                convolution.compute_convolution_inner_product_lhs_prepared(
                    data.iter().map(|(l, r)| (l, r)),
                    &mut actual, 
                    &ring
                );
                assert_el_eq!(&ring, ring.add_ref_fst(&expected[0], ring.one()), &actual[0]);
                for i in 1..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }
            }
        }
    }
}