use core::f64;
use std::alloc::Allocator;
use std::alloc::Global;
use std::cell::Ref;
use std::cell::RefCell;

use crate::algorithms::fft::complex_fft::FFTErrorEstimate;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::algorithms::fft::FFTAlgorithm;
use crate::primitive_int::StaticRingBase;
use crate::integer::*;
use crate::ring::*;
use crate::seq::*;
use crate::primitive_int::*;
use crate::homomorphism::*;
use crate::rings::float_complex::*;
use crate::rings::zn::*;

use super::ConvolutionAlgorithm;

#[stability::unstable(feature = "enable")]
pub struct FFTBasedConvolutionZn<A = Global> {
    allocator: A,
    fft_tables: RefCell<Vec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>
}

impl<A> FFTBasedConvolutionZn<A>
    where A: Allocator
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(allocator: A) -> Self {
        Self {
            allocator: allocator,
            fft_tables: RefCell::new(Vec::new())
        }
    }

    fn get_fft_table(&self, log2_size: usize) -> Ref<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>> {
        let CC = Complex64::RING;
        let mut fft_tables = self.fft_tables.borrow_mut();
        while fft_tables.len() < log2_size {
            let log2_n = fft_tables.len() + 1;
            let fft_table = CooleyTuckeyFFT::for_complex(CC, log2_n);
            fft_tables.push(fft_table);
        }
        drop(fft_tables);
        return Ref::map(self.fft_tables.borrow(), |fft_tables: &Vec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>| &fft_tables[log2_size - 1]);
    }

    #[stability::unstable(feature = "enable")]
    pub fn can_compute<R>(&self, log2_size: usize, ring: &R) -> bool
        where R: ?Sized + ZnRing + CanHomFrom<StaticRingBase<i64>>
    {
        let fft_table = self.get_fft_table(log2_size);
        let modulus = ring.integer_ring().to_float_approx(ring.modulus());
        fft_table.expected_absolute_error(modulus * modulus, modulus * modulus * f64::EPSILON + fft_table.expected_absolute_error(modulus, 0.)) < 0.5
    }
}

impl<R, A> ConvolutionAlgorithm<R> for FFTBasedConvolutionZn<A>
    where R: ?Sized + ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    fn compute_convolution<V1: VectorView<R::Element>, V2: VectorView<R::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R::Element], ring: &R) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let log2_size = StaticRing::<i64>::RING.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        debug_assert!(log2_size >= 1);

        assert!(
            self.can_compute(log2_size, ring), 
            "f64 does not have enough precision for computing a convolution modulo {}", 
            ring.integer_ring().format(ring.modulus())
        );

        let CC = Complex64::RING;
        let fft_table = self.get_fft_table(log2_size);
        let hom = CC.can_hom(ring.integer_ring()).unwrap();
        let mut lhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        lhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in lhs.as_iter().enumerate() {
            lhs_data[i] = hom.map(ring.smallest_lift(ring.clone_el(c)));
        }

        let mut rhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        rhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in rhs.as_iter().enumerate() {
            rhs_data[i] = hom.map(ring.smallest_lift(ring.clone_el(c)));
        }

        fft_table.unordered_fft(&mut lhs_data[..], CC.get_ring());
        fft_table.unordered_fft(&mut rhs_data[..], CC.get_ring());
        for i in 0..(1 << log2_size) {
            CC.mul_assign_ref(&mut lhs_data[i], &rhs_data[i]);
        }
        fft_table.unordered_inv_fft(&mut lhs_data[..], CC.get_ring());

        let hom = RingRef::new(ring).into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
        for i in 0..(lhs.len() + rhs.len()) {
            let x = CC.closest_gaussian_int(lhs_data[i]);
            debug_assert!(x.1 == 0);
            ring.add_assign(&mut dst[i], hom.map(x.0));
        }
    }
}

#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use super::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;

#[test]
fn test_convolution() {
    let convolution_algorithm = FFTBasedConvolutionZn::new_with(Global);

    let ring = Zn::new(17 * 257);
    let lhs = ring.elements().collect::<Vec<_>>();
    let rhs = ring.elements().collect::<Vec<_>>();
    let mut actual = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();
    let mut expected = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();

    convolution_algorithm.compute_convolution(&lhs[..16], &rhs[..7], &mut actual, ring.get_ring());
    STANDARD_CONVOLUTION.compute_convolution(&lhs[..16], &rhs[..7], &mut expected, ring.get_ring());

    for i in 0..actual.len() {
        assert_el_eq!(ring, &expected[i], &actual[i]);
    }

    convolution_algorithm.compute_convolution(&lhs, &rhs, &mut actual, ring.get_ring());
    STANDARD_CONVOLUTION.compute_convolution(&lhs, &rhs, &mut expected, ring.get_ring());

    for i in 0..actual.len() {
        assert_el_eq!(ring, &expected[i], &actual[i]);
    }
}