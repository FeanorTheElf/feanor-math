use std::alloc::{Allocator, Global};
use std::any::Any;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use crate::ring::*;

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
/// The functions all take the ring by reference to [`RingBase`], since
/// that makes [`ConvolutionAlgorithm`] dyn-compatible.
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
/// 
///     fn prepare_convolution_operand(&self, val: &[R::Element], length_hint: Option<usize>, ring: &R) -> () { () }
/// 
///     fn compute_convolution(&self, lhs: &[R::Element], _: Option<&()>, rhs: &[R::Element], _: Option<&()>, dst: &mut [R::Element], ring: &R) {
///         for i in 0..(lhs.len() + rhs.len() - 1) {
///             for j in max(0, i as isize - rhs.len() as isize + 1)..min(lhs.len() as isize, i as isize + 1) {
///                 ring.add_assign(&mut dst[i], ring.mul_ref(lhs.at(j as usize), rhs.at(i - j as usize)));
///             }
///         }
///     }
/// 
///     fn supports_ring(&self, ring: &R) -> bool { true }
/// }
/// let lhs = [1, 2, 3, 4, 5];
/// let rhs = [2, 3, 4, 5, 6];
/// let mut expected = [0; 10];
/// let mut actual = [0; 10];
/// STANDARD_CONVOLUTION.compute_convolution(&lhs[..], None, &rhs[..], None, &mut expected, StaticRing::<i64>::RING.get_ring());
/// NaiveConvolution.compute_convolution(&lhs[..], None, &rhs[..], None, &mut actual, StaticRing::<i64>::RING.get_ring());
/// assert_eq!(expected, actual);
/// ```
/// 
pub trait ConvolutionAlgorithm<R: ?Sized + RingBase>: Send + Sync {

    ///
    /// Additional data associated to a list of ring elements, which can be used to
    /// compute a convolution where this list is one of the operands faster.
    ///
    /// For more details, see [`ConvolutionAlgorithm::prepare_convolution_operand()`].
    /// Note that a `PreparedConvolutionOperand` can only be used for convolutions
    /// with the same list of values it was created for.
    /// 
    type PreparedConvolutionOperand = ();

    ///
    /// Returns whether this convolution algorithm supports computations of 
    /// the given ring.
    /// 
    /// Note that most algorithms will support all rings of type `R`. However in some cases,
    /// e.g. for finite fields, required data might only be precomputed for some moduli,
    /// and thus only these will be supported.
    /// 
    fn supports_ring(&self, ring: &R) -> bool;

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
    /// [`ConvolutionAlgorithm::compute_convolution()`].
    ///  
    /// # Example
    /// 
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::algorithms::convolution::*;
    /// # use feanor_math::algorithms::convolution::ntt::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64b::*;
    /// # use feanor_math::rings::finite::*;
    /// let ring = Zn64B::new(65537);
    /// let convolution = NTTConvolution::new(ring);
    /// let lhs = ring.elements().take(10).collect::<Vec<_>>();
    /// let rhs = ring.elements().take(10).collect::<Vec<_>>();
    /// // "standard" use
    /// let mut expected = (0..19).map(|_| ring.zero()).collect::<Vec<_>>();
    /// convolution.compute_convolution(&lhs, None, &rhs, None, &mut expected, ring.get_ring());
    /// 
    /// // "prepared" variant
    /// let lhs_prep = convolution.prepare_convolution_operand(&lhs, None, ring.get_ring());
    /// let rhs_prep = convolution.prepare_convolution_operand(&rhs, None, ring.get_ring());
    /// let mut actual = (0..19).map(|_| ring.zero()).collect::<Vec<_>>();
    /// // this will now be faster than `convolution.compute_convolution()`
    /// convolution.compute_convolution(&lhs, Some(&lhs_prep), &rhs, Some(&rhs_prep), &mut actual, ring.get_ring());
    /// println!("{:?}, {:?}", actual.iter().map(|x| ring.formatted_el(x)).collect::<Vec<_>>(), expected.iter().map(|x| ring.formatted_el(x)).collect::<Vec<_>>());
    /// assert!(expected.iter().zip(actual.iter()).all(|(l, r)| ring.eq_el(l, r)));
    /// ```
    /// 
    /// Maybe also consider taking the ring by `&RingBase`, since this would then allow
    /// for dynamic dispatch.
    /// 
    fn prepare_convolution_operand(&self, val: &[R::Element], length_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand;
    
    ///
    /// Elementwise adds the convolution of `lhs` and `rhs` to `dst`. If provided, the given
    /// prepared convolution operands are used for a faster computation.
    /// 
    /// This will produce nonsensical results, if the prepared convolution operands don't
    /// match the passed data, i.e. if `lhs_prep` has not been created by calling
    /// [`ConvolutionAlgorithm::prepare_convolution_operand()`] on `lhs`, and similarly
    /// for `rhs`.
    /// 
    /// # Panic
    /// 
    /// Panics if `dst` is shorter than `lhs.len() + rhs.len() - 1`. May panic if `dst` is shorter
    /// than `lhs.len() + rhs.len()`.
    /// 
    fn compute_convolution(&self, lhs: &[R::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R);

    ///
    /// Computes a convolution for each tuple in the given sequence, and sums the result of
    /// each convolution to `dst`.
    /// 
    /// In other words, this computes `dst[k] += sum_l sum_(i + j = k) values[l][i] * values[l][k]`.
    /// It can be faster than calling [`ConvolutionAlgorithm::compute_convolution()`] multiple times.
    /// 
    fn compute_convolution_sum(&self, values: &[(&[R::Element], Option<&Self::PreparedConvolutionOperand>, &[R::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R::Element], ring: &R) {
        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            self.compute_convolution(lhs, *lhs_prep, rhs, *rhs_prep, dst, ring)
        }
    }
}

impl<R, C> ConvolutionAlgorithm<R> for C
    where R: ?Sized + RingBase,
        C: Deref + Send + Sync,
        C::Target: ConvolutionAlgorithm<R>
{
    type PreparedConvolutionOperand = <C::Target as ConvolutionAlgorithm<R>>::PreparedConvolutionOperand;

    fn supports_ring(&self, ring: &R) -> bool { (**self).supports_ring(ring) }
    fn prepare_convolution_operand(&self, val: &[R::Element], length_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand { (**self).prepare_convolution_operand(val, length_hint, ring) }
    fn compute_convolution(&self, lhs: &[R::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R) { (**self).compute_convolution(lhs, lhs_prep, rhs, rhs_prep, dst, ring); }
    fn compute_convolution_sum(&self, values: &[(&[R::Element], Option<&Self::PreparedConvolutionOperand>, &[R::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R::Element], ring: &R) { (**self).compute_convolution_sum(values, dst, ring); }
}

///
/// Implementation of convolutions that uses Karatsuba's algorithm
/// with a threshold defined by [`KaratsubaHint`].
/// 
#[derive(Clone, Copy, Debug)]
pub struct KaratsubaAlgorithm<A: Allocator + Send + Sync = Global> {
    allocator: A,
    threshold_log2: usize
}

impl<A: Allocator + Send + Sync> KaratsubaAlgorithm<A> {
    
    #[stability::unstable(feature = "enable")]
    pub const fn new(threshold_log2: usize, allocator: A) -> Self {
        Self { threshold_log2, allocator }
    }
}

impl<R: ?Sized + RingBase, A: Allocator + Send + Sync> ConvolutionAlgorithm<R> for KaratsubaAlgorithm<A> {
    
    fn compute_convolution(&self, lhs: &[R::Element], _lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], _rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R) {
        karatsuba(self.threshold_log2, dst, lhs, rhs, RingRef::new(ring), &self.allocator);
    }
    
    fn prepare_convolution_operand(&self, _val: &[<R as RingBase>::Element], _length_hint: Option<usize>, _ring: &R) -> Self::PreparedConvolutionOperand {
        ()
    }

    fn supports_ring(&self, _ring: &R) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NaiveConvolution;

impl<R: ?Sized + RingBase> ConvolutionAlgorithm<R> for NaiveConvolution {

    type PreparedConvolutionOperand = ();

    fn compute_convolution(&self, lhs: &[R::Element], _: Option<&()>, rhs: &[R::Element], _: Option<&()>, dst: &mut [R::Element], ring: &R) {
        naive_assign_mul::<_, _, _, _, true>(dst, lhs, rhs, RingRef::new(ring));
    }

    fn prepare_convolution_operand(&self, _: &[R::Element], _: Option<usize>, _: &R) -> Self::PreparedConvolutionOperand {
        ()
    }

    fn supports_ring(&self, _: &R) -> bool {
        true
    }
}

pub type DynConvolution<'a, R> = Arc<dyn 'a + ConvolutionAlgorithm<R, PreparedConvolutionOperand = Box<dyn Any>>>;

pub struct TypeErasableConvolution<R: ?Sized + RingBase, C: ConvolutionAlgorithm<R>>
    where C::PreparedConvolutionOperand: 'static
{
    ring: PhantomData<R>,
    convolution: C
}

impl<R: ?Sized + RingBase, C: ConvolutionAlgorithm<R>> TypeErasableConvolution<R, C>
    where C::PreparedConvolutionOperand: 'static
{
    pub fn new(convolution: C) -> Self {
        Self { ring: PhantomData, convolution: convolution }
    }
}

impl<R: ?Sized + RingBase, C: ConvolutionAlgorithm<R>> ConvolutionAlgorithm<R> for TypeErasableConvolution<R, C>
    where C::PreparedConvolutionOperand: 'static
{
    type PreparedConvolutionOperand = Box<dyn Any>;

    fn compute_convolution(&self, lhs: &[<R as RingBase>::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[<R as RingBase>::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [<R as RingBase>::Element], ring: &R) {
        self.convolution.compute_convolution(lhs, lhs_prep.map(|op| op.downcast_ref().unwrap()), rhs, rhs_prep.map(|op| op.downcast_ref().unwrap()), dst, ring);
    }

    fn compute_convolution_sum(&self, values: &[(&[<R as RingBase>::Element], Option<&Self::PreparedConvolutionOperand>, &[<R as RingBase>::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [<R as RingBase>::Element], ring: &R) {
        self.convolution.compute_convolution_sum(&values.iter().map(|(l, l_prep, r, r_prep)| 
            (*l, l_prep.map(|op| op.downcast_ref().unwrap()), *r, r_prep.map(|op| op.downcast_ref().unwrap()))
        ).collect::<Vec<_>>(), dst, ring);
    }

    fn prepare_convolution_operand(&self, val: &[<R as RingBase>::Element], length_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand {
        Box::new(self.convolution.prepare_convolution_operand(val, length_hint, ring))
    }

    fn supports_ring(&self, ring: &R) -> bool {
        self.convolution.supports_ring(ring)
    }
}

pub trait DefaultConvolutionRing: RingBase {

    fn create_default_convolution<'conv>(&self, max_len: Option<usize>) -> DynConvolution<'conv, Self>
        where Self: 'conv;
}

impl<R: ?Sized + RingBase> DefaultConvolutionRing for R {

    default fn create_default_convolution<'conv>(&self, _max_len_hint: Option<usize>) -> DynConvolution<'conv, Self>
        where Self: 'conv
    {
        Arc::new(TypeErasableConvolution::new(KaratsubaAlgorithm::new(0, Global)))
    }
}

#[cfg(test)]
use test;
#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[bench]
fn bench_naive_mul(bencher: &mut test::Bencher) {
    LogAlgorithmSubscriber::init_test();
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
    LogAlgorithmSubscriber::init_test();
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
                convolution.compute_convolution(&lhs, None, &rhs, None, &mut actual, ring.get_ring());
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
                convolution.compute_convolution(&lhs, None, &rhs, None, &mut actual, ring.get_ring());
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
                convolution.compute_convolution(
                    &lhs,
                    Some(&convolution.prepare_convolution_operand(&lhs, None, ring.get_ring())),
                    &rhs,
                    Some(&convolution.prepare_convolution_operand(&rhs, None, ring.get_ring())),
                    &mut actual, 
                    ring.get_ring()
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution(
                    &lhs,
                    Some(&convolution.prepare_convolution_operand(&lhs, None, ring.get_ring())),
                    &rhs,
                    None,
                    &mut actual, 
                    ring.get_ring()
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }
                
                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                convolution.compute_convolution(
                    &lhs,
                    None,
                    &rhs,
                    Some(&convolution.prepare_convolution_operand(&rhs, None, ring.get_ring())),
                    &mut actual, 
                    ring.get_ring()
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &expected[i as usize], &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (&lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, ring.get_ring())), &rhs[..], Some(convolution.prepare_convolution_operand(&rhs, None, ring.get_ring()))),
                    (&rhs[..], None, &lhs[..], None)
                ];
                convolution.compute_convolution_sum(
                    &data.iter().map(|(l, l_prep, r, r_prep): &(_, _, _, _)| (&l[..], l_prep.as_ref(), &r[..], r_prep.as_ref())).collect::<Vec<_>>(),
                    &mut actual, 
                    ring.get_ring()
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &ring.add_ref(&expected[i as usize], &expected[i as usize]), &actual[i as usize]);
                }

                let mut actual = Vec::new();
                actual.resize_with((lhs_len + rhs_len) as usize, || ring.zero());
                let data = [
                    (&lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, ring.get_ring())), &rhs[..], None),
                    (&rhs[..], None, &lhs[..], Some(convolution.prepare_convolution_operand(&lhs, None, ring.get_ring())))
                ];
                convolution.compute_convolution_sum(
                    &data.iter().map(|(l, l_prep, r, r_prep)| (&l[..], l_prep.as_ref(), &r[..], r_prep.as_ref())).collect::<Vec<_>>(),
                    &mut actual, 
                    ring.get_ring()
                );
                for i in 0..(lhs_len + rhs_len) {
                    assert_el_eq!(&ring, &ring.add_ref(&expected[i as usize], &expected[i as usize]), &actual[i as usize]);
                }
            }
        }
    }
}