use std::alloc::{Allocator, Global};
use std::ops::Deref;

use crate::ring::*;
use crate::seq::subvector::SubvectorView;
use crate::seq::*;

use karatsuba::*;

///
/// Contains an optimized implementation of Karatsuba's for computing convolutions
/// 
pub mod karatsuba;

///
/// Contains an implementation of computing convolutions using complex floating-point FFTs.
/// 
pub mod fft;

///
/// Contains an implementation of computing convolutions using NTTs, i.e. FFTs over
/// a finite field that has suitable roots of unity.
/// 
pub mod ntt;

///
/// Contains an implementation of computing convolutions by considering them modulo
/// various primes that are either smaller or allow for suitable roots of unity.
/// 
pub mod rns;

///
/// Trait for objects that can compute a convolution over some ring.
/// 
/// # Example
/// ```rust
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
    /// Additional data associated to a list of ring elements, which can be used to
    /// compute a convolution where this list is one of the operands faster.
    ///
    /// For more details, see [`ConvolutionAlgorithm::prepare_convolution_operand()`].
    /// Note that a `PreparedConvolutionOperand` can only be used for convolutions
    /// with the same list of values it was created for.
    /// 
    #[stability::unstable(feature = "enable")]
    type PreparedConvolutionOperand = ();

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

    ///
    /// Takes an input list of values and computes an opaque [`ConvolutionAlgorithm::PreparedConvolutionOperand`],
    /// which can be used to compute future convolutions with this list of values faster.
    /// 
    /// Although the [`ConvolutionAlgorithm::PreparedConvolutionOperand`] does not have any explicit reference
    /// to the list of values it was created for, passing it to [`ConvolutionAlgorithm::compute_convolution_prepared()`]
    /// with another list of values will give erroneous results.
    /// 
    /// # Length-dependence when preparing a convolution
    /// 
    /// For some algorithms, different data is required to speed up the convolution with an operand, depending on the 
    /// length of the other operand. For example, for FFT-based convolutions, the prepared data would consist of the
    /// Fourier transform of the list of values, zero-padded to a length that can store the complete result of the
    /// (future) convolution.
    /// 
    /// To handle this, implementations can make use of the `length_hint`, which - if given - should be an upper bound
    /// to the length of the output of any future convolution that uses the given operand. Alternatively, implementations
    /// are encouraged to not compute any data during [`ConvolutionAlgorithm::prepare_convolution_operand()`],
    /// but initialize an object with interior mutability, and use it to cache data computed during
    /// [`ConvolutionAlgorithm::compute_convolution_prepared()`].
    /// 
    /// TODO: At next breaking release, remove the default implementation
    /// 
    /// # Example
    /// 
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::algorithms::convolution::*;
    /// # use feanor_math::algorithms::convolution::ntt::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::rings::finite::*;
    /// let ring = Zn::new(65537);
    /// let convolution = NTTConvolution::new(ring);
    /// let lhs = ring.elements().take(10).collect::<Vec<_>>();
    /// let rhs = ring.elements().take(10).collect::<Vec<_>>();
    /// // "standard" use
    /// let mut expected = (0..19).map(|_| ring.zero()).collect::<Vec<_>>();
    /// convolution.compute_convolution(&lhs, &rhs, &mut expected, ring);
    /// 
    /// // "prepared" variant
    /// let lhs_prep = convolution.prepare_convolution_operand(&lhs, None, ring);
    /// let rhs_prep = convolution.prepare_convolution_operand(&rhs, None, ring);
    /// let mut actual = (0..19).map(|_| ring.zero()).collect::<Vec<_>>();
    /// // this will now be faster than `convolution.compute_convolution()`
    /// convolution.compute_convolution_prepared(&lhs, Some(&lhs_prep), &rhs, Some(&rhs_prep), &mut actual, ring);
    /// println!("{:?}, {:?}", actual.iter().map(|x| ring.format(x)).collect::<Vec<_>>(), expected.iter().map(|x| ring.format(x)).collect::<Vec<_>>());
    /// assert!(expected.iter().zip(actual.iter()).all(|(l, r)| ring.eq_el(l, r)));
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    fn prepare_convolution_operand<S, V>(&self, _val: V, _length_hint: Option<usize>, _ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        struct ProduceUnitType;
        trait ProduceValue<T> {
            fn produce() -> T;
        }
        impl<T> ProduceValue<T> for ProduceUnitType {
            default fn produce() -> T {
                panic!("if you specialize ConvolutionAlgorithm::PreparedConvolutionOperand, you must also specialize ConvolutionAlgorithm::prepare_convolution_operand()")
            }
        }
        impl ProduceValue<()> for ProduceUnitType {
            fn produce() -> () {}
        }
        return <ProduceUnitType as ProduceValue<Self::PreparedConvolutionOperand>>::produce();
    }
    
    ///
    /// Elementwise adds the convolution of `lhs` and `rhs` to `dst`. If provided, the given
    /// prepared convolution operands are used for a faster computation.
    /// 
    /// When called with `None` as both the prepared convolution operands, this is exactly
    /// equivalent to [`ConvolutionAlgorithm::compute_convolution()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, _lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, _rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>
    {
        self.compute_convolution(lhs, rhs, dst, ring)
    }

    ///
    /// Computes a convolution for each tuple in the given sequence, and sums the result of each convolution
    /// to `dst`.
    /// 
    /// In other words, this computes `dst[k] += sum_l sum_(i + j = k) values[l][i] * values[l][k]`.
    /// It can be faster than calling [`ConvolutionAlgorithm::prepare_convolution_operand()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn compute_convolution_sum<'a, S, I, V1, V2>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: ExactSizeIterator<Item = (V1, Option<&'a Self::PreparedConvolutionOperand>, V2, Option<&'a Self::PreparedConvolutionOperand>)>,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            Self: 'a,
            R: 'a
    {
        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            self.compute_convolution_prepared(lhs, lhs_prep, rhs, rhs_prep, dst, ring)
        }
    }
}

impl<'a, R, C> ConvolutionAlgorithm<R> for C
    where R: ?Sized + RingBase,
        C: Deref,
        C::Target: ConvolutionAlgorithm<R>
{
    type PreparedConvolutionOperand = <C::Target as ConvolutionAlgorithm<R>>::PreparedConvolutionOperand;

    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        (**self).compute_convolution(lhs, rhs, dst, ring)
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, ring: S) -> bool {
        (**self).supports_ring(ring)
    }

    fn prepare_convolution_operand<S, V>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        (**self).prepare_convolution_operand(val, len_hint, ring)
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>
    {
        (**self).compute_convolution_prepared(lhs, lhs_prep, rhs, rhs_prep, dst, ring);
    }

    fn compute_convolution_sum<'b, S, I, V1, V2>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: ExactSizeIterator<Item = (V1, Option<&'b Self::PreparedConvolutionOperand>, V2, Option<&'b Self::PreparedConvolutionOperand>)>,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            Self: 'b,
            R: 'b
    {
        (**self).compute_convolution_sum(values, dst, ring);
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
        test_prepared_convolution(convolution, ring, scale);
    }

    fn test_prepared_convolution<C, R>(convolution: C, ring: R, scale: El<R>)
        where C: ConvolutionAlgorithm<R::Type>,
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
                    &lhs,
                    Some(&convolution.prepare_convolution_operand(&lhs, None, &ring)),
                    &rhs,
                    Some(&convolution.prepare_convolution_operand(&rhs, None, &ring)),
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution_prepared(
                    &lhs,
                    Some(&convolution.prepare_convolution_operand(&lhs, None, &ring)),
                    &rhs,
                    None,
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }
                
                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution_prepared(
                    &lhs,
                    None,
                    &rhs,
                    Some(&convolution.prepare_convolution_operand(&rhs, None, &ring)),
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (&lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, &ring)), &rhs[..], Some(convolution.prepare_convolution_operand(&rhs, None, &ring))),
                    (&rhs[..], None, &lhs[..], None)
                ];
                convolution.compute_convolution_sum(
                    data.as_fn().map_fn(|(l, l_prep, r, r_prep): &(_, _, _, _)| (l, l_prep.as_ref(), r, r_prep.as_ref())).iter(),
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &ring.add_ref(&expected[i as usize], &expected[i as usize]), &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (&lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, &ring)), &rhs[..], None),
                    (&rhs[..], None, &lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, &ring)))
                ];
                convolution.compute_convolution_sum(
                    data.as_fn().map_fn(|(l, l_prep, r, r_prep)| (l, l_prep.as_ref(), r, r_prep.as_ref())).iter(),
                    &mut actual, 
                    &ring
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &ring.add_ref(&expected[i as usize], &expected[i as usize]), &actual[i as usize]);
                }
            }
        }
    }
}