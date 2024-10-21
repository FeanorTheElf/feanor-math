use core::f64;
use std::alloc::Allocator;
use std::alloc::Global;
use std::cell::Ref;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::Bound;

use crate::algorithms::fft::complex_fft::FFTErrorEstimate;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::algorithms::fft::FFTAlgorithm;
use crate::algorithms::int_bisect;
use crate::algorithms::miller_rabin::prev_prime;
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
pub struct FFTBasedConvolution<A = Global> {
    allocator: A,
    fft_tables: RefCell<Vec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>
}

impl<A> FFTBasedConvolution<A>
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
    pub fn can_compute(&self, log2_size: usize, log2_modulus: usize) -> bool {
        let fft_table = self.get_fft_table(log2_size);
        let modulus = 2f64.powi(log2_modulus as i32);
        fft_table.expected_absolute_error(modulus * modulus, modulus * modulus * f64::EPSILON + fft_table.expected_absolute_error(modulus, 0.)) < 0.5
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

impl<A> Clone for FFTBasedConvolution<A>
    where A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
            fft_tables: self.fft_tables.clone()
        }
    }
}

impl<A> From<FFTBasedConvolutionZn<A>> for FFTBasedConvolution<A>
    where A: Allocator
{
    fn from(value: FFTBasedConvolutionZn<A>) -> Self {
        value.base
    }
}

impl<'a, A> From<&'a FFTBasedConvolutionZn<A>> for &'a FFTBasedConvolution<A>
    where A: Allocator
{
    fn from(value: &'a FFTBasedConvolutionZn<A>) -> Self {
        &value.base
    }
}

impl<A> From<FFTBasedConvolution<A>> for FFTBasedConvolutionZn<A>
    where A: Allocator
{
    fn from(value: FFTBasedConvolution<A>) -> Self {
        FFTBasedConvolutionZn { base: value }
    }
}

impl<'a, A> From<&'a FFTBasedConvolution<A>> for &'a FFTBasedConvolutionZn<A>
    where A: Allocator
{
    fn from(value: &'a FFTBasedConvolution<A>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct FFTBasedConvolutionZn<A = Global> {
    base: FFTBasedConvolution<A>
}

impl<A> Clone for FFTBasedConvolutionZn<A>
    where A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<R, A> ConvolutionAlgorithm<R> for FFTBasedConvolutionZn<A>
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

impl<I, A> ConvolutionAlgorithm<I> for FFTBasedConvolution<A>
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
            lhs.into_ring_el_fn(&ring), 
            rhs.into_ring_el_fn(&ring), 
            &ring
        ).enumerate() {
            ring.add_assign(&mut dst[i], c);
        }
    }

    fn supports_ring<S: RingStore<Type = I> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

#[stability::unstable(feature = "enable")]
pub struct FFTRNSBasedConvolution<R = zn_64::ZnBase, I = BigIntRing, A = Global>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    integer_ring: I,
    max_log2_len: usize,
    convolution: FFTBasedConvolution<A>,
    rns_rings: RefCell<BTreeMap<usize, zn_rns::Zn<RingValue<R>, I, A>>>
}

impl<R, I, A> FFTRNSBasedConvolution<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(max_log2_len: usize, integer_ring: I, allocator: A) -> Self {
        Self {
            max_log2_len: max_log2_len,
            integer_ring: integer_ring,
            convolution: FFTBasedConvolution::new_with(allocator),
            rns_rings: RefCell::new(BTreeMap::new())
        }
    }

    fn create_rns_ring(&self, min_log2_rns_ring_modulus: usize) -> (usize, zn_rns::Zn<RingValue<R>, I, A>) {
        let max_log2_n = R::create(|int_ring| Err(int_ring.representable_bits().unwrap_or(64) - 1)).err().unwrap();
        let sample_primes_of_size = int_bisect::find_root_floor(StaticRing::<i64>::RING, 1, |log2_n| {
            if *log2_n as usize > max_log2_n {
                return 1;
            } else {
                if self.convolution.can_compute(self.max_log2_len, *log2_n as usize) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });
        let mut rns_base = Vec::new();
        let mut current = prev_prime(&self.integer_ring, self.integer_ring.power_of_two(sample_primes_of_size as usize)).unwrap();
        let initial_modulus_bits = self.integer_ring.abs_log2_floor(&current).unwrap();
        let mut current_modulus_bits = initial_modulus_bits;
        rns_base.push(self.integer_ring.clone_el(&current));
        
        while current_modulus_bits < min_log2_rns_ring_modulus {
            current = prev_prime(&self.integer_ring, current).unwrap();
            let new_modulus_bits = self.integer_ring.abs_log2_floor(&current).unwrap();
            // if the bitlength of the moduli has decreased, then we already used half the available primes, and all further primes
            // will add less moduli for the same cost; it is usually not sensible to continue from here
            assert!(new_modulus_bits == initial_modulus_bits, "not enough precision to reasonably compute convolution of length {}", 1 << self.max_log2_len);
            current_modulus_bits += new_modulus_bits;
            rns_base.push(self.integer_ring.clone_el(&current));
        }

        return (
            current_modulus_bits, 
            zn_rns::Zn::new_with(rns_base.into_iter().map(|n| 
                RingValue::from(R::create(|new_int_ring| Ok(int_cast(n, RingRef::new(new_int_ring), &self.integer_ring))).unwrap_or_else(|x| x))
            ).collect(), self.integer_ring.clone(), self.convolution.allocator.clone())
        );
    }

    fn get_rns_ring<'a>(&'a self, input_size_log2: usize) -> Ref<'a, zn_rns::Zn<RingValue<R>, I, A>> {
        assert!(input_size_log2 > 0);
        let min_log2_rns_ring_modulus = 2 * input_size_log2 + self.max_log2_len + 1;
        let mut rns_rings = self.rns_rings.borrow_mut();
        if rns_rings.lower_bound(Bound::Included(&min_log2_rns_ring_modulus)).next().is_none() {
            let (supported_int_log2, rns_ring) = self.create_rns_ring(min_log2_rns_ring_modulus);
            rns_rings.insert(supported_int_log2, rns_ring);
        }
        drop(rns_rings);
        return Ref::map(self.rns_rings.borrow(), |rns_rings| rns_rings.lower_bound(Bound::Included(&min_log2_rns_ring_modulus)).next().unwrap().1);
    }
    
    fn convolution_unchecked<'a, V1: VectorFn<El<I>>, V2: VectorFn<El<I>>>(&'a self, input_size_log2: usize, lhs: V1, rhs: V2) -> impl 'a + Iterator<Item = El<I>> {
        assert!(lhs.len() + rhs.len() <= (1 << self.max_log2_len));
        let rns_ring = self.get_rns_ring(input_size_log2);
        let total_len = lhs.len() + rhs.len();
        let mut result: Vec<R::Element, _> = Vec::with_capacity_in(total_len * rns_ring.len(), &self.convolution.allocator);
        for Zp in rns_ring.as_iter() {
            let hom1 = Zp.can_hom(&self.integer_ring).unwrap();
            let hom2 = Zp.can_hom(Zp.integer_ring()).unwrap();
            result.extend(
                self.convolution.convolution_unchecked(
                    (&lhs).map_fn(|x| Zp.smallest_lift(hom1.map(x))),
                    (&rhs).map_fn(|x| Zp.smallest_lift(hom1.map(x))),
                    Zp.integer_ring()
                ).map(|x| hom2.map(x))
            );
        }
        return (0..total_len).map(move |i| rns_ring.smallest_lift(rns_ring.from_congruence((0..rns_ring.len()).map(|j| rns_ring.at(j).clone_el(&result[i + j * total_len])))));
    }
}

impl<R, I, A> Clone for FFTRNSBasedConvolution<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing + Clone,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self {
            convolution: self.convolution.clone(),
            integer_ring: self.integer_ring.clone(),
            max_log2_len: self.max_log2_len,
            rns_rings: self.rns_rings.clone()
        }
    }
}

impl<I2, R, I, A> ConvolutionAlgorithm<I2> for FFTRNSBasedConvolution<R, I, A>
    where I2: ?Sized + IntegerRing,
        I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn compute_convolution<S: RingStore<Type = I2>, V1: VectorView<I2::Element>, V2: VectorView<I2::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [I2::Element], ring: S) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let largest_element = lhs.as_iter().chain(rhs.as_iter()).max_by(|l, r| ring.abs_cmp(l, r));
        let input_size_log2 = largest_element.and_then(|n| ring.abs_log2_ceil(n)).unwrap_or(0);
        
        for (i, c) in self.convolution_unchecked(
            input_size_log2,
            lhs.into_ring_el_fn(&ring).map_fn(|x| int_cast(x, &self.integer_ring, &ring)), 
            rhs.into_ring_el_fn(&ring).map_fn(|x| int_cast(x, &self.integer_ring, &ring))
        ).enumerate() {
            ring.add_assign(&mut dst[i], int_cast(c, &ring, &self.integer_ring));
        }
    }

    fn supports_ring<S: RingStore<Type = I2> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct FFTRNSBasedConvolutionZn<R = zn_64::ZnBase, I = BigIntRing, A = Global>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    base: FFTRNSBasedConvolution<R, I, A>
}

impl<R, I, A> From<FFTRNSBasedConvolutionZn<R, I, A>> for FFTRNSBasedConvolution<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn from(value: FFTRNSBasedConvolutionZn<R, I, A>) -> Self {
        value.base
    }
}

impl<'a, R, I, A> From<&'a FFTRNSBasedConvolutionZn<R, I, A>> for &'a FFTRNSBasedConvolution<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn from(value: &'a FFTRNSBasedConvolutionZn<R, I, A>) -> Self {
        &value.base
    }
}

impl<R, I, A> From<FFTRNSBasedConvolution<R, I, A>> for FFTRNSBasedConvolutionZn<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn from(value: FFTRNSBasedConvolution<R, I, A>) -> Self {
        FFTRNSBasedConvolutionZn { base: value }
    }
}

impl<'a, R, I, A> From<&'a FFTRNSBasedConvolution<R, I, A>> for &'a FFTRNSBasedConvolutionZn<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn from(value: &'a FFTRNSBasedConvolution<R, I, A>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<R, I, A> Clone for FFTRNSBasedConvolutionZn<R, I, A>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing + Clone,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<R2, R, I, A> ConvolutionAlgorithm<R2> for FFTRNSBasedConvolutionZn<R, I, A>
    where R2: ?Sized + ZnRing + CanHomFrom<I::Type>,
        I: RingStore + Clone,
        I::Type: IntegerRing,
        R: ZnRing + CanHomFrom<I::Type> + CanHomFrom<StaticRingBase<i64>> + FromModulusCreateableZnRing,
        zn_64::ZnBase: CanHomFrom<I::Type>,
        A: Allocator + Clone
{
    fn compute_convolution<S: RingStore<Type = R2>, V1: VectorView<R2::Element>, V2: VectorView<R2::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R2::Element], ring: S) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let input_log2_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap();
        
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        for (i, c) in self.base.convolution_unchecked(
            input_log2_size,
            lhs.as_fn().map_fn(|x| int_cast(ring.smallest_lift(ring.clone_el(x)), &self.base.integer_ring, ring.integer_ring())), 
            rhs.as_fn().map_fn(|x| int_cast(ring.smallest_lift(ring.clone_el(x)), &self.base.integer_ring, ring.integer_ring()))
        ).enumerate() {
            ring.add_assign(&mut dst[i], hom.map(c));
        }
    }

    fn supports_ring<S: RingStore<Type = R2> + Copy>(&self, _ring: S) -> bool {
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
    let convolution_algorithm: FFTBasedConvolutionZn = FFTBasedConvolution::new_with(Global).into();

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
    let convolution_algorithm: FFTBasedConvolutionZn = FFTBasedConvolution::new_with(Global).into();

    let ring = Zn::new(1099511627791);
    let lhs = ring.elements().take(1024).collect::<Vec<_>>();
    let rhs = ring.elements().take(1024).collect::<Vec<_>>();
    let mut actual = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();

    convolution_algorithm.compute_convolution(&lhs, &rhs, &mut actual, &ring);
}

#[test]
fn test_fft_rns_convolution() {
    let convolution_algorithm: FFTRNSBasedConvolutionZn<_, BigIntRing, _> = FFTRNSBasedConvolution::<zn_64::ZnBase, BigIntRing, _>::new_with(16, BigIntRing::RING, Global).into();

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

    let ring = Zn::new(1099511627791);
    let lhs = ring.elements().take(1024).collect::<Vec<_>>();
    let rhs = ring.elements().take(1024).collect::<Vec<_>>();
    let mut actual = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();
    let mut expected = (0..(lhs.len() + rhs.len())).map(|_| ring.zero()).collect::<Vec<_>>();

    convolution_algorithm.compute_convolution(&lhs, &rhs, &mut actual, &ring);
    STANDARD_CONVOLUTION.compute_convolution(&lhs, &rhs, &mut expected, &ring);

    for i in 0..actual.len() {
        assert_el_eq!(ring, &expected[i], &actual[i]);
    }
}