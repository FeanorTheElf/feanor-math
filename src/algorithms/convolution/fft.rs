use std::alloc::{Allocator, Global};
use std::borrow::Cow;
use std::cmp::max;
use std::marker::PhantomData;

use crate::algorithms::fft::complex_fft::FFTErrorEstimate;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::algorithms::fft::FFTAlgorithm;
use crate::lazy::LazyVec;
use crate::primitive_int::StaticRingBase;
use crate::integer::*;
use crate::ring::*;
use crate::seq::*;
use crate::primitive_int::*;
use crate::homomorphism::*;
use crate::rings::float_complex::*;
use crate::rings::zn::*;

use super::ConvolutionAlgorithm;

const CC: Complex64 = Complex64::RING;

#[stability::unstable(feature = "enable")]
pub struct FFTConvolution<A = Global> {
    allocator: A,
    fft_tables: LazyVec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>
}

#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, A = Global>
    where R: ?Sized + RingBase,
        A: Allocator + Clone
{
    ring: PhantomData<Box<R>>,
    fft_data: LazyVec<Vec<El<Complex64>, A>>,
    log2_data_size: usize
}

impl<A> FFTConvolution<A>
    where A: Allocator + Clone
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(allocator: A) -> Self {
        Self {
            allocator: allocator,
            fft_tables: LazyVec::new()
        }
    }

    fn get_fft_table(&self, log2_len: usize) -> &CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>> {
        return self.fft_tables.get_or_init(log2_len, || CooleyTuckeyFFT::for_complex(CC, log2_len));
    }

    fn get_fft_data<'a, R, V, ToInt>(
        &self,
        data: V,
        data_prep: Option<&'a PreparedConvolutionOperand<R, A>>,
        _ring: &R,
        log2_len: usize,
        mut to_int: ToInt,
        log2_el_size: Option<usize>
    ) -> Cow<'a, Vec<El<Complex64>, A>>
        where R: ?Sized + RingBase,
            V: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> i64
    {
        let log2_data_size = if let Some(log2_data_size) = log2_el_size {
            if let Some(data_prep) = data_prep {
                assert_eq!(log2_data_size, data_prep.log2_data_size);
            }
            log2_data_size 
        } else {
            data.as_iter().map(|x| StaticRing::<i64>::RING.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
        };
        assert!(data.len() <= (1 << log2_len));
        assert!(self.has_sufficient_precision(log2_len, log2_data_size));

        let mut compute_result = || {
            let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
            result.extend(data.as_iter().map(|x| Complex64::RING.from_f64(to_int(x) as f64)));
            result.resize(1 << log2_len, Complex64::RING.zero());
            self.get_fft_table(log2_len).unordered_fft(&mut result, Complex64::RING);
            return result;
        };

        return if let Some(data_prep) = data_prep {
            Cow::Borrowed(data_prep.fft_data.get_or_init(log2_len, compute_result))
        } else {
            Cow::Owned(compute_result())
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn has_sufficient_precision(&self, log2_len: usize, log2_input_size: usize) -> bool {
        self.max_sum_len(log2_len, log2_input_size) > 0
    }
    
    fn max_sum_len(&self, log2_len: usize, log2_input_size: usize) -> usize {
        let fft_table = self.get_fft_table(log2_len);
        let input_size = 2f64.powi(log2_input_size.try_into().unwrap());
        (0.5 / fft_table.expected_absolute_error(input_size * input_size, input_size * input_size * f64::EPSILON + fft_table.expected_absolute_error(input_size, 0.))).floor() as usize
    }

    fn prepare_convolution_impl<R, V, ToInt>(
        &self,
        data: V,
        ring: &R,
        length_hint: Option<usize>,
        mut to_int: ToInt,
        ring_log2_el_size: Option<usize>
    ) -> PreparedConvolutionOperand<R, A>
        where R: ?Sized + RingBase,
            V: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> i64
    {
        let log2_data_size = if let Some(log2_data_size) = ring_log2_el_size {
            log2_data_size 
        } else {
            data.as_iter().map(|x| StaticRing::<i64>::RING.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
        };
        let result = PreparedConvolutionOperand {
            fft_data: LazyVec::new(),
            ring: PhantomData,
            log2_data_size: log2_data_size
        };
        // if a length-hint is given, initialize the corresponding length entry;
        // this might avoid confusing performance characteristics when the user does
        // not expect lazy behavior
        if let Some(len) = length_hint {
            let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();
            _ = self.get_fft_data(data, Some(&result), ring, log2_len, to_int, ring_log2_el_size);
        }
        return result;
    }

    fn compute_convolution_impl<R, V1, V2, ToInt, FromInt>(
        &self,
        lhs: V1,
        lhs_prep: Option<&PreparedConvolutionOperand<R, A>>,
        rhs: V2,
        rhs_prep: Option<&PreparedConvolutionOperand<R, A>>,
        dst: &mut [R::Element],
        ring: &R,
        mut to_int: ToInt,
        mut from_int: FromInt,
        ring_log2_el_size: Option<usize>
    )
        where R: ?Sized + RingBase,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> i64,
            FromInt: FnMut(i64) -> R::Element
    {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let len = lhs.len() + rhs.len() - 1;
        assert!(dst.len() >= len);
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut lhs_fft = self.get_fft_data(lhs, lhs_prep, ring, log2_len, &mut to_int, ring_log2_el_size);
        let mut rhs_fft = self.get_fft_data(rhs, rhs_prep, ring, log2_len, &mut to_int, ring_log2_el_size);
        if rhs_fft.is_owned() {
            std::mem::swap(&mut lhs_fft, &mut rhs_fft);
        }
        let lhs_fft: &mut Vec<El<Complex64>, A> = lhs_fft.to_mut();

        for i in 0..(1 << log2_len) {
            CC.mul_assign(&mut lhs_fft[i], rhs_fft[i]);
        }

        self.get_fft_table(log2_len).unordered_inv_fft(&mut *lhs_fft, CC);

        for i in 0..len {
            let result = CC.closest_gaussian_int(lhs_fft[i]);
            debug_assert_eq!(0, result.1);
            ring.add_assign(&mut dst[i], from_int(result.0));
        }
    }

    fn compute_convolution_sum_impl<'a, R, I, V1, V2, ToInt, FromInt>(
        &self,
        data: I,
        dst: &mut [R::Element],
        ring: &R,
        mut to_int: ToInt,
        mut from_int: FromInt,
        ring_log2_el_size: Option<usize>
    )
        where R: ?Sized + RingBase,
            I: Iterator<Item = (V1, Option<&'a PreparedConvolutionOperand<R, A>>, V2, Option<&'a PreparedConvolutionOperand<R, A>>)>,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> i64,
            FromInt: FnMut(i64) -> R::Element,
            Self: 'a,
            R: 'a
    {
        let len = dst.len();
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();
        let mut buffer = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        buffer.resize(1 << log2_len, CC.zero());

        let mut count_since_last_reduction = 0;
        let mut current_max_sum_len = usize::MAX;
        let mut current_log2_data_size = if let Some(log2_data_size) = ring_log2_el_size {
            log2_data_size
        } else {
            0
        };
        for (lhs, lhs_prep, rhs, rhs_prep) in data {
            if lhs.len() == 0 || rhs.len() == 0 {
                continue;
            }
            assert!(lhs.len() + rhs.len() - 1 <= dst.len());

            if ring_log2_el_size.is_none() {
                current_log2_data_size = max(
                    current_log2_data_size,
                    lhs.as_iter().chain(rhs.as_iter()).map(|x| StaticRing::<i64>::RING.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
                );
                current_max_sum_len = self.max_sum_len(log2_len, current_log2_data_size);
            }
            assert!(current_max_sum_len > 0);
            
            if count_since_last_reduction > current_max_sum_len {
                count_since_last_reduction = 0;
                self.get_fft_table(log2_len).unordered_inv_fft(&mut *buffer, CC);
                for i in 0..len {
                    let result = CC.closest_gaussian_int(buffer[i]);
                    debug_assert_eq!(0, result.1);
                    ring.add_assign(&mut dst[i], from_int(result.0));
                }
                for i in 0..(1 << log2_len) {
                    buffer[i] = CC.zero();
                }
            }
            
            let mut lhs_fft = self.get_fft_data(lhs, lhs_prep, ring, log2_len, &mut to_int, ring_log2_el_size);
            let mut rhs_fft = self.get_fft_data(rhs, rhs_prep, ring, log2_len, &mut to_int, ring_log2_el_size);
            if rhs_fft.is_owned() {
                std::mem::swap(&mut lhs_fft, &mut rhs_fft);
            }
            let lhs_fft: &mut Vec<El<Complex64>, A> = lhs_fft.to_mut();
            for i in 0..(1 << log2_len) {
                CC.mul_assign(&mut lhs_fft[i], rhs_fft[i]);
                CC.add_assign(&mut buffer[i], lhs_fft[i]);
            }
            count_since_last_reduction += 1;
        }
        self.get_fft_table(log2_len).unordered_inv_fft(&mut *buffer, CC);
        for i in 0..len {
            let result = CC.closest_gaussian_int(buffer[i]);
            debug_assert_eq!(0, result.1);
            ring.add_assign(&mut dst[i], from_int(result.0));
        }
    }
}

fn to_int_int<I>(ring: I) -> impl use<I> + Fn(&El<I>) -> i64
    where I: RingStore, I::Type: IntegerRing
{
    move |x| int_cast(ring.clone_el(x), StaticRing::<i64>::RING, &ring)
}

fn from_int_int<I>(ring: I) -> impl use<I> + Fn(i64) -> El<I>
    where I: RingStore, I::Type: IntegerRing
{
    move |x| int_cast(x, &ring, StaticRing::<i64>::RING)
}

fn to_int_zn<R>(ring: R) -> impl use<R> + Fn(&El<R>) -> i64
    where R: RingStore, R::Type: ZnRing
{
    move |x| int_cast(ring.smallest_lift(ring.clone_el(x)), StaticRing::<i64>::RING, ring.integer_ring())
}

fn from_int_zn<R>(ring: R) -> impl use<R> + Fn(i64) -> El<R>
    where R: RingStore, R::Type: ZnRing
{
    let hom = ring.can_hom(ring.integer_ring()).unwrap().into_raw_hom();
    move |x| ring.get_ring().map_in(ring.integer_ring().get_ring(), int_cast(x, ring.integer_ring(), StaticRing::<i64>::RING), &hom)
}

impl<A> Clone for FFTConvolution<A>
    where A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
            fft_tables: self.fft_tables.clone()
        }
    }
}

impl<A> From<FFTConvolutionZn<A>> for FFTConvolution<A>
    where A: Allocator
{
    fn from(value: FFTConvolutionZn<A>) -> Self {
        value.base
    }
}

impl<'a, A> From<&'a FFTConvolutionZn<A>> for &'a FFTConvolution<A>
    where A: Allocator
{
    fn from(value: &'a FFTConvolutionZn<A>) -> Self {
        &value.base
    }
}

impl<A> From<FFTConvolution<A>> for FFTConvolutionZn<A>
    where A: Allocator
{
    fn from(value: FFTConvolution<A>) -> Self {
        FFTConvolutionZn { base: value }
    }
}

impl<'a, A> From<&'a FFTConvolution<A>> for &'a FFTConvolutionZn<A>
    where A: Allocator
{
    fn from(value: &'a FFTConvolution<A>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct FFTConvolutionZn<A = Global> {
    base: FFTConvolution<A>
}

impl<A> Clone for FFTConvolutionZn<A>
    where A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<R, A> ConvolutionAlgorithm<R> for FFTConvolutionZn<A>
    where R: ?Sized + ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        self.base.compute_convolution_impl(
            lhs,
            None,
            rhs,
            None,
            dst,
            ring.get_ring(),
            to_int_zn(&ring),
            from_int_zn(&ring),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }

    fn prepare_convolution_operand<S, V>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        self.base.prepare_convolution_impl(
            val,
            ring.get_ring(),
            len_hint,
            to_int_zn(&ring),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V1: VectorView<R::Element>, V2: VectorView<R::Element>
    {
        self.base.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst,
            ring.get_ring(),
            to_int_zn(&ring),
            from_int_zn(&ring),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }

    fn compute_convolution_sum<'a, S, I, V1, V2>(&self, values: I, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            I: Iterator<Item = (V1, Option<&'a Self::PreparedConvolutionOperand>, V2, Option<&'a Self::PreparedConvolutionOperand>)>,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            Self: 'a,
            R: 'a,
            Self::PreparedConvolutionOperand: 'a
    {
        self.base.compute_convolution_sum_impl(
            values,
            dst,
            ring.get_ring(),
            to_int_zn(&ring),
            from_int_zn(&ring),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }
}

impl<I, A> ConvolutionAlgorithm<I> for FFTConvolution<A>
    where I: ?Sized + IntegerRing,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<I, A>;

    fn compute_convolution<S: RingStore<Type = I>, V1: VectorView<I::Element>, V2: VectorView<I::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [I::Element], ring: S) {
        self.compute_convolution_impl(
            lhs,
            None,
            rhs,
            None,
            dst,
            ring.get_ring(),
            to_int_int(&ring),
            from_int_int(&ring),
            None
        )
    }

    fn supports_ring<S: RingStore<Type = I> + Copy>(&self, _ring: S) -> bool {
        true
    }

    fn prepare_convolution_operand<S, V>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = I> + Copy, V: VectorView<I::Element>
    {
        self.prepare_convolution_impl(
            val,
            ring.get_ring(),
            len_hint,
            to_int_int(&ring),
            None
        )
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [I::Element], ring: S)
        where S: RingStore<Type = I> + Copy, V1: VectorView<I::Element>, V2: VectorView<I::Element>
    {
        self.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst,
            ring.get_ring(),
            to_int_int(&ring),
            from_int_int(&ring),
            None
        )
    }

    fn compute_convolution_sum<'a, S, J, V1, V2>(&self, values: J, dst: &mut [I::Element], ring: S) 
        where S: RingStore<Type = I> + Copy, 
            J: Iterator<Item = (V1, Option<&'a Self::PreparedConvolutionOperand>, V2, Option<&'a Self::PreparedConvolutionOperand>)>,
            V1: VectorView<I::Element>,
            V2: VectorView<I::Element>,
            Self: 'a,
            I: 'a,
            Self::PreparedConvolutionOperand: 'a
    {
        self.compute_convolution_sum_impl(
            values,
            dst,
            ring.get_ring(),
            to_int_int(&ring),
            from_int_int(&ring),
            None
        )
    }
}

#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;

#[test]
fn test_convolution_zn() {
    let convolution: FFTConvolutionZn = FFTConvolution::new_with(Global).into();
    let ring = Zn::new(17 * 257);

    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}

#[test]
fn test_convolution_int() {
    let convolution: FFTConvolution = FFTConvolution::new_with(Global);
    let ring = StaticRing::<i64>::RING;

    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}

#[test]
#[should_panic(expected = "precision")]
fn test_fft_convolution_not_enough_precision() {
    let convolution_algorithm: FFTConvolutionZn = FFTConvolution::new_with(Global).into();

    let ring = Zn::new(1099511627791);
    let lhs = ring.elements().take(1024).collect::<Vec<_>>();
    let rhs = ring.elements().take(1024).collect::<Vec<_>>();
    let mut actual = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();

    convolution_algorithm.compute_convolution(&lhs, &rhs, &mut actual, &ring);
}
