use core::f64;
use std::alloc::Allocator;
use std::alloc::Global;

use crate::algorithms::fft::complex_fft::FFTErrorEstimate;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::algorithms::fft::FFTAlgorithm;
use crate::lazy::LazyVec;
use crate::primitive_int::StaticRingBase;
use crate::ordered::OrderedRingStore;
use crate::integer::*;
use crate::ring::*;
use crate::seq::*;
use crate::primitive_int::*;
use crate::homomorphism::*;
use crate::rings::float_complex::*;
use crate::rings::zn::*;

use super::ConvolutionAlgorithm;

#[stability::unstable(feature = "enable")]
pub struct FFTConvolution<A = Global> {
    allocator: A,
    fft_tables: LazyVec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>
}

impl<A> FFTConvolution<A>
    where A: Allocator
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(allocator: A) -> Self {
        Self {
            allocator: allocator,
            fft_tables: LazyVec::new()
        }
    }

    fn get_fft_table(&self, log2_len: usize) -> &CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>> {
        let CC = Complex64::RING;
        return self.fft_tables.get_or_init(log2_len, || CooleyTuckeyFFT::for_complex(CC, log2_len));
    }

    #[stability::unstable(feature = "enable")]
    pub fn can_compute(&self, log2_len: usize, log2_input_size: usize) -> bool {
        let fft_table = self.get_fft_table(log2_len);
        let input_size = 2f64.powi(log2_input_size as i32);
        fft_table.expected_absolute_error(input_size * input_size, input_size * input_size * f64::EPSILON + fft_table.expected_absolute_error(input_size, 0.)) < 0.5
    }

    fn convolution_unchecked<'a, R: 'a + RingStore, V1: VectorFn<El<R>>, V2: VectorFn<El<R>>>(&'a self, lhs: V1, rhs: V2, ring: R) -> impl 'a + Iterator<Item = El<R>>
        where R::Type: IntegerRing
    {
        let log2_size = StaticRing::<i64>::RING.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        let CC = Complex64::RING;
        let fft_table = self.get_fft_table(log2_size);
        let hom = CC.can_hom(&ring).unwrap();
        let mut lhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        lhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in lhs.iter().enumerate() {
            lhs_data[i] = hom.map(c);
        }

        let mut rhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        rhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in rhs.iter().enumerate() {
            rhs_data[i] = hom.map(c);
        }

        fft_table.unordered_fft(&mut lhs_data[..], CC);
        fft_table.unordered_fft(&mut rhs_data[..], CC);
        for i in 0..(1 << log2_size) {
            CC.mul_assign_ref(&mut lhs_data[i], &rhs_data[i]);
        }
        fft_table.unordered_inv_fft(&mut lhs_data[..], CC);

        let hom = ring.into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        (0..(lhs.len() + rhs.len())).map(move |i| {
            let x = CC.closest_gaussian_int(lhs_data[i]);
            debug_assert!(x.1 == 0);
            return hom.map(x.0);
        })
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
        A: Allocator
{
    fn compute_convolution<S: RingStore<Type = R>, V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: S) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let log2_size = StaticRing::<i64>::RING.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        let input_log2_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        assert!(
            self.base.can_compute(log2_size, input_log2_size), 
            "f64 does not have enough precision for computing a convolution of result length {} modulo {}",
            lhs.len() + rhs.len() - 1, 
            ring.integer_ring().format(ring.modulus())
        );
        let hom = ring.can_hom(ring.integer_ring()).unwrap();
        for (i, c) in self.base.convolution_unchecked(
            lhs.as_fn().map_fn(|x| ring.smallest_lift(ring.clone_el(x))), 
            rhs.as_fn().map_fn(|x| ring.smallest_lift(ring.clone_el(x))), 
            ring.integer_ring()
        ).enumerate() {
            ring.add_assign(&mut dst[i], hom.map(c));
        }
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

impl<I, A> ConvolutionAlgorithm<I> for FFTConvolution<A>
    where I: ?Sized + IntegerRing,
        A: Allocator
{
    fn compute_convolution<S: RingStore<Type = I>, V1: VectorView<I::Element>, V2: VectorView<I::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [I::Element], ring: S) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let log2_size = StaticRing::<i64>::RING.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        let largest_element = lhs.as_iter().chain(rhs.as_iter()).max_by(|l, r| ring.abs_cmp(l, r));
        let input_log2_size = largest_element.and_then(|n| ring.abs_log2_ceil(n)).unwrap_or(0);
        assert!(
            self.can_compute(log2_size, input_log2_size), 
            "f64 does not have enough precision for computing a convolution of result length {} with inputs of size up to {}",
            lhs.len() + rhs.len() - 1, 
            ring.format(largest_element.unwrap())
        );
        for (i, c) in self.convolution_unchecked(
            lhs.into_clone_ring_els(&ring), 
            rhs.into_clone_ring_els(&ring), 
            &ring
        ).enumerate() {
            ring.add_assign(&mut dst[i], c);
        }
    }

    fn supports_ring<S: RingStore<Type = I> + Copy>(&self, _ring: S) -> bool {
        true
    }
}


#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use super::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;

#[test]
fn test_fft_convolution() {
    let convolution_algorithm: FFTConvolutionZn = FFTConvolution::new_with(Global).into();

    let ring = Zn::new(17 * 257);
    let lhs = ring.elements().collect::<Vec<_>>();
    let rhs = ring.elements().collect::<Vec<_>>();
    let mut actual = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();
    let mut expected = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();

    convolution_algorithm.compute_convolution(&lhs[..16], &rhs[..7], &mut actual, &ring);
    STANDARD_CONVOLUTION.compute_convolution(&lhs[..16], &rhs[..7], &mut expected, &ring);

    for i in 0..actual.len() {
        assert_el_eq!(ring, &expected[i], &actual[i]);
    }

    convolution_algorithm.compute_convolution(&lhs, &rhs, &mut actual, &ring);
    STANDARD_CONVOLUTION.compute_convolution(&lhs, &rhs, &mut expected, &ring);

    for i in 0..actual.len() {
        assert_el_eq!(ring, &expected[i], &actual[i]);
    }
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
