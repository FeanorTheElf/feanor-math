use std::alloc::{Allocator, Global};
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

use super::{ConvolutionAlgorithm, PreparedConvolutionAlgorithm, PreparedConvolutionOperation};

const ZZ: StaticRing<i64> = StaticRing::RING;
const CC: Complex64 = Complex64::RING;

struct RoundGaussianInt;

impl FnOnce<(El<Complex64>,)> for RoundGaussianInt {
    type Output = i64;
    extern "rust-call" fn call_once(self, args: (El<Complex64>,)) -> Self::Output {
        self.call(args)
    }
}

impl FnMut<(El<Complex64>,)> for RoundGaussianInt {
    extern "rust-call" fn call_mut(&mut self, args: (El<Complex64>,)) -> Self::Output {
        self.call(args)
    }
}

impl Fn<(El<Complex64>,)> for RoundGaussianInt {
    extern "rust-call" fn call(&self, args: (El<Complex64>,)) -> Self::Output {
        let x = CC.closest_gaussian_int(args.0);
        debug_assert!(x.1 == 0);
        return x.0;
    }
}

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
    original_data: Vec<f64, A>,
    fft_data: Vec<El<Complex64>, A>
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

    #[stability::unstable(feature = "enable")]
    pub fn has_sufficient_precision(&self, log2_len: usize, log2_input_size: usize) -> bool {
        let fft_table = self.get_fft_table(log2_len);
        let input_size = 2f64.powi(log2_input_size.try_into().unwrap());
        fft_table.expected_absolute_error(input_size * input_size, input_size * input_size * f64::EPSILON + fft_table.expected_absolute_error(input_size, 0.)) < 0.5
    }

    fn compute_convolution_unprepared_impl<V1, V2>(&self, lhs: V1, rhs: V2, log2_data_size: Option<usize>) -> std::iter::Take<std::iter::Map<std::vec::IntoIter<El<Complex64>, A>, RoundGaussianInt>>
        where V1: VectorFn<f64>,
            V2: VectorFn<f64>
    {
        if lhs.len() == 0 || rhs.len() == 0 {
            return Vec::new_in(self.allocator.clone()).into_iter().map(RoundGaussianInt).take(0);
        }
        let out_len = lhs.len() + rhs.len() - 1;
        let out_len_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        let lhs_prep = self.prepare_convolution_impl(lhs, Some(1 << out_len_log2), log2_data_size);
        let rhs_prep = self.prepare_convolution_impl(rhs, Some(1 << out_len_log2), log2_data_size);
        return self.compute_convolution_impl(lhs_prep, &rhs_prep, out_len);
    }

    fn compute_convolution_prepared_impl(&self, lhs_base: &[f64], lhs: &[El<Complex64>], rhs_base: &[f64], rhs: &[El<Complex64>], log2_data_size: Option<usize>) -> std::iter::Take<std::iter::Map<std::vec::IntoIter<El<Complex64>, A>, RoundGaussianInt>> {
        if lhs_base.len() == 0 || rhs_base.len() == 0 {
            return Vec::new_in(self.allocator.clone()).into_iter().map(RoundGaussianInt).take(0);
        }
        let out_len = lhs_base.len() + rhs_base.len() - 1;
        let out_len_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        match (lhs.len() == 1 << out_len_log2, rhs.len() == 1 << out_len_log2) {
            (true, true) => {
                let mut lhs_copy = Vec::with_capacity_in(lhs.len(), self.allocator.clone());
                lhs_copy.extend(lhs.iter().copied());
                self.compute_convolution_impl(lhs_copy, rhs, out_len)
            },
            (true, false) => self.compute_convolution_lhs_prepared_impl(lhs_base, lhs, rhs_base.copy_els(), log2_data_size),
            (false, true) => self.compute_convolution_lhs_prepared_impl(rhs_base, rhs, lhs_base.copy_els(), log2_data_size),
            (false, false) => self.compute_convolution_unprepared_impl(lhs_base.copy_els(), rhs_base.copy_els(), log2_data_size)
        }
    }

    fn compute_convolution_lhs_prepared_impl<V>(&self, lhs_base: &[f64], lhs: &[El<Complex64>], rhs: V, log2_data_size: Option<usize>) -> std::iter::Take<std::iter::Map<std::vec::IntoIter<El<Complex64>, A>, RoundGaussianInt>>
        where V: VectorFn<f64>
    {
        if lhs_base.len() == 0 || rhs.len() == 0 {
            return Vec::new_in(self.allocator.clone()).into_iter().map(RoundGaussianInt).take(0);
        }
        let out_len = lhs_base.len() + rhs.len() - 1;
        let out_len_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        if lhs.len() == 1 << out_len_log2 {
            let rhs_prep = self.prepare_convolution_impl(rhs, Some(1 << out_len_log2), log2_data_size);
            return self.compute_convolution_impl(rhs_prep, lhs, out_len);
        } else {
            return self.compute_convolution_unprepared_impl(lhs_base.copy_els(), rhs, log2_data_size);
        }
    }

    fn compute_convolution_impl(&self, mut lhs: Vec<El<Complex64>, A>, rhs: &[El<Complex64>], target_len: usize) -> std::iter::Take<std::iter::Map<std::vec::IntoIter<El<Complex64>, A>, RoundGaussianInt>> {
        let log2_n = ZZ.abs_log2_ceil(&lhs.len().try_into().unwrap()).unwrap();
        assert_eq!(lhs.len(), 1 << log2_n);
        assert_eq!(rhs.len(), 1 << log2_n);

        for i in 0..(1 << log2_n) {
            CC.mul_assign(&mut lhs[i], rhs[i]);
        }
        self.get_fft_table(log2_n).unordered_inv_fft(&mut lhs[..], CC);
        return lhs.into_iter().map(RoundGaussianInt).take(target_len);
    }

    fn prepare_convolution_impl<V>(&self, data: V, length_hint: Option<usize>, log2_data_size: Option<usize>) -> Vec<El<Complex64>, A>
        where V: VectorFn<f64>
    {
        let n_out = if let Some(len_hint) = length_hint {
            assert!(len_hint >= data.len());
            len_hint
        } else {
            data.len() * 2
        };
        let log2_n_out = StaticRing::<i64>::RING.abs_log2_ceil(&n_out.try_into().unwrap()).unwrap();
        assert!(data.len() <= 1 << log2_n_out);

        let log2_data_size = if let Some(log2_data_size) = log2_data_size {
            log2_data_size 
        } else {
            data.iter().map(|x| x.abs()).max_by(f64::total_cmp).unwrap().log2().ceil() as usize
        };
        assert!(self.has_sufficient_precision(log2_n_out, log2_data_size));

        let mut fft_data = Vec::with_capacity_in(1 << log2_n_out, self.allocator.clone());
        fft_data.extend(data.iter().map(|x| CC.from_f64(x)));
        fft_data.resize(1 << log2_n_out, CC.zero());
        let fft = self.get_fft_table(log2_n_out);
        fft.unordered_fft(&mut fft_data[..], CC);
        return fft_data;
    }
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
    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        let log2_data_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        let to_f64 = |x| int_cast(ring.smallest_lift(x), ZZ, ring.integer_ring()) as f64;
        let hom = ring.can_hom(&ZZ).unwrap();
        for (i, x) in self.base.compute_convolution_unprepared_impl(lhs.clone_ring_els(&ring).map_fn(to_f64), rhs.clone_ring_els(&ring).map_fn(to_f64), Some(log2_data_size)).enumerate() {
            ring.add_assign(&mut dst[i], hom.map(x));
        }
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }

    fn specialize_prepared_convolution<F>(function: F) -> F::Output
        where F: PreparedConvolutionOperation<Self, R>
    {
        function.execute()
    }
}

impl<I, A> ConvolutionAlgorithm<I> for FFTConvolution<A>
    where I: ?Sized + IntegerRing,
        A: Allocator + Clone
{
    fn compute_convolution<S: RingStore<Type = I>, V1: VectorView<I::Element>, V2: VectorView<I::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [I::Element], ring: S) {
        let to_f64 = |x| int_cast(x, ZZ, &ring) as f64;
        for (i, x) in self.compute_convolution_unprepared_impl(lhs.clone_ring_els(&ring).map_fn(to_f64), rhs.clone_ring_els(&ring).map_fn(to_f64), None).enumerate() {
            ring.add_assign(&mut dst[i], int_cast(x, &ring, ZZ));
        }
    }

    fn supports_ring<S: RingStore<Type = I> + Copy>(&self, _ring: S) -> bool {
        true
    }

    fn specialize_prepared_convolution<F>(function: F) -> F::Output
        where F: PreparedConvolutionOperation<Self, I>
    {
        function.execute()
    }
}

impl<I, A> PreparedConvolutionAlgorithm<I> for FFTConvolution<A>
    where I: ?Sized + IntegerRing,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<I, A>;

    fn prepare_convolution_operand<S, V>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = I> + Copy, V: VectorView<I::Element>
    {
        let mut original_data = Vec::new_in(self.allocator.clone());
        original_data.extend(val.clone_ring_els(&ring).iter().map(|x| int_cast(x, ZZ, &ring) as f64));
        let fft_data = self.prepare_convolution_impl(original_data.copy_els(), len_hint, None);
        return PreparedConvolutionOperand {
            fft_data: fft_data,
            original_data: original_data,
            ring: PhantomData
        };
    }

    fn prepared_operand_len(&self, prepared_operand: &Self::PreparedConvolutionOperand) -> usize {
        prepared_operand.original_data.len()
    }

    fn prepared_operand_data<S, V>(&self, prepared_operand: &Self::PreparedConvolutionOperand, mut output: V, ring: S)
        where S: RingStore<Type = I> + Copy, V: VectorViewMut<I::Element>
    {
        assert!(output.len() >= self.prepared_operand_len(prepared_operand));
        let hom = ring.can_hom(&ZZ).unwrap();
        for i in 0..self.prepared_operand_len(prepared_operand) {
            *output.at_mut(i) = hom.map(prepared_operand.original_data[i] as i64);
        }
        for i in self.prepared_operand_len(prepared_operand)..output.len() {
            *output.at_mut(i) = ring.zero();
        }
    }

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [I::Element], ring: S)
        where S: RingStore<Type = I> + Copy, V: VectorView<I::Element>
    {
        let to_f64 = |x| int_cast(x, ZZ, &ring) as f64;
        for (i, x) in self.compute_convolution_lhs_prepared_impl(&lhs.original_data, &lhs.fft_data[..], rhs.clone_ring_els(&ring).map_fn(to_f64), None).enumerate() {
            ring.add_assign(&mut dst[i], int_cast(x, ring, ZZ));
        }
    }

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [I::Element], ring: S)
        where S: RingStore<Type = I> + Copy
    {
        for (i, x) in self.compute_convolution_prepared_impl(&lhs.original_data, &lhs.fft_data[..], &rhs.original_data, &rhs.fft_data[..], None).enumerate() {
            ring.add_assign(&mut dst[i], int_cast(x, ring, ZZ));
        }
    }
}

impl<R, A> PreparedConvolutionAlgorithm<R> for FFTConvolutionZn<A>
    where R: ?Sized + ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn prepare_convolution_operand<S, V>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        let log2_data_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        let mut original_data = Vec::new_in(self.base.allocator.clone());
        original_data.extend(val.clone_ring_els(&ring).iter().map(|x| int_cast(ring.smallest_lift(x), ZZ, ring.integer_ring()) as f64));
        let fft_data = self.base.prepare_convolution_impl(original_data.copy_els(), len_hint, Some(log2_data_size));
        return PreparedConvolutionOperand {
            fft_data: fft_data,
            original_data: original_data,
            ring: PhantomData
        };
    }

    fn prepared_operand_len(&self, prepared_operand: &Self::PreparedConvolutionOperand) -> usize {
        prepared_operand.original_data.len()
    }

    fn prepared_operand_data<S, V>(&self, prepared_operand: &Self::PreparedConvolutionOperand, mut output: V, ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorViewMut<R::Element>
    {
        assert!(output.len() >= self.prepared_operand_len(prepared_operand));
        let hom = ring.can_hom(&ZZ).unwrap();
        for i in 0..self.prepared_operand_len(prepared_operand) {
            *output.at_mut(i) = hom.map(prepared_operand.original_data[i] as i64);
        }
        for i in self.prepared_operand_len(prepared_operand)..output.len() {
            *output.at_mut(i) = ring.zero();
        }
    }

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        let log2_data_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        let to_f64 = |x| int_cast(ring.smallest_lift(x), ZZ, ring.integer_ring()) as f64;
        let hom = ring.can_hom(&ZZ).unwrap();
        for (i, x) in self.base.compute_convolution_lhs_prepared_impl(&lhs.original_data, &lhs.fft_data[..], rhs.clone_ring_els(&ring).map_fn(to_f64), Some(log2_data_size)).enumerate() {
            ring.add_assign(&mut dst[i], hom.map(x));
        }
    }

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy
    {
        let log2_data_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        let hom = ring.can_hom(&ZZ).unwrap();
        for (i, x) in self.base.compute_convolution_prepared_impl(&lhs.original_data, &lhs.fft_data[..], &rhs.original_data, &rhs.fft_data, Some(log2_data_size)).enumerate() {
            ring.add_assign(&mut dst[i], hom.map(x));
        }
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
    super::generic_tests::test_prepared_convolution(&convolution, &ring, ring.one());
}

#[test]
fn test_convolution_int() {
    let convolution: FFTConvolution = FFTConvolution::new_with(Global);
    let ring = StaticRing::<i64>::RING;

    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
    super::generic_tests::test_prepared_convolution(&convolution, &ring, ring.one());
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
