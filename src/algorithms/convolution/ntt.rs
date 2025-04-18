use std::alloc::{Allocator, Global};
use std::cmp::min;

use crate::{algorithms::fft::cooley_tuckey::CooleyTuckeyFFT, lazy::LazyVec};
use crate::homomorphism::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::algorithms::fft::*;
use crate::rings::zn::*;
use crate::integer::*;
use crate::seq::VectorView;

use super::{ConvolutionAlgorithm, PreparedConvolutionAlgorithm, PreparedConvolutionOperation};

///
/// Computes the convolution over a finite field that has suitable roots of unity
/// using a power-of-two length FFT (sometimes called Number-Theoretic Transform,
/// NTT in this context).
/// 
#[stability::unstable(feature = "enable")]
pub struct NTTConvolution<R, A = Global>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    ring: R,
    fft_algos: LazyVec<CooleyTuckeyFFT<R::Type, R::Type, Identity<R>>>,
    allocator: A
}

#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, A = Global>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    len: usize,
    data: Vec<El<R>, A>
}

impl<R, A> NTTConvolution<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(ring: R, allocator: A) -> Self {
        Self {
            fft_algos: LazyVec::new(),
            ring: ring,
            allocator: allocator
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn ring(&self) -> &R {
        &self.ring
    }

    fn add_assign_elementwise_product(lhs: &[El<R>], rhs: &[El<R>], dst: &mut [El<R>], ring: RingRef<R::Type>) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            ring.add_assign(&mut dst[i], ring.mul_ref(&lhs[i], &rhs[i]));
        }
    }

    fn compute_convolution_impl(&self, mut lhs: PreparedConvolutionOperand<R, A>, rhs: &PreparedConvolutionOperand<R, A>, out: &mut [El<R>]) {
        let log2_n = ZZ.abs_log2_ceil(&lhs.data.len().try_into().unwrap()).unwrap();
        assert_eq!(lhs.data.len(), 1 << log2_n);
        assert_eq!(rhs.data.len(), 1 << log2_n);
        assert!(lhs.len + rhs.len <= 1 << log2_n);
        assert!(out.len() >= lhs.len + rhs.len);
        for i in 0..(1 << log2_n) {
            self.ring.mul_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        self.get_fft(log2_n).unordered_inv_fft(&mut lhs.data[..], self.ring());
        for i in 0..min(out.len(), 1 << log2_n) {
            self.ring.add_assign_ref(&mut out[i], &lhs.data[i]);
        }
    }

    fn un_and_redo_fft(&self, input: &[El<R>], log2_n: usize) -> Vec<El<R>, A> {
        let log2_in_len = ZZ.abs_log2_ceil(&input.len().try_into().unwrap()).unwrap();
        assert_eq!(input.len(), 1 << log2_in_len);
        assert!(log2_in_len < log2_n);

        let mut tmp = Vec::with_capacity_in(input.len(), self.allocator.clone());
        tmp.extend(input.iter().map(|x| self.ring.clone_el(x)));
        self.get_fft(log2_in_len).unordered_inv_fft(&mut tmp[..], self.ring());

        tmp.resize_with(1 << log2_n, || self.ring.zero());
        self.get_fft(log2_n).unordered_fft(&mut tmp[..], self.ring());
        return tmp;
    }

    fn get_fft<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R::Type, R::Type, Identity<R>> {
        self.fft_algos.get_or_init(log2_n, || CooleyTuckeyFFT::for_zn(self.ring().clone(), log2_n).unwrap())
    }

    fn clone_prepared_operand(&self, operand: &PreparedConvolutionOperand<R, A>) -> PreparedConvolutionOperand<R, A> {
        let mut result = Vec::with_capacity_in(operand.data.len(), self.allocator.clone());
        result.extend(operand.data.iter().map(|x| self.ring.clone_el(x)));
        return PreparedConvolutionOperand {
            len: operand.len,
            data: result
        };
    }
    
    fn prepare_convolution_impl<V: VectorView<El<R>>>(&self, val: V, log2_n: usize) -> PreparedConvolutionOperand<R, A> {
        let mut result = Vec::with_capacity_in(1 << log2_n, self.allocator.clone());
        result.extend(val.as_iter().map(|x| self.ring.clone_el(x)));
        result.resize_with(1 << log2_n, || self.ring.zero());
        let fft = self.get_fft(log2_n);
        fft.unordered_fft(&mut result[..], self.ring());
        return PreparedConvolutionOperand {
            len: val.len(),
            data: result
        };
    }
}

impl<R, A> ConvolutionAlgorithm<R::Type> for NTTConvolution<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    fn supports_ring<S: RingStore<Type = R::Type> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring.get_ring()
    }

    fn compute_convolution<S: RingStore<Type = R::Type> + Copy, V1: VectorView<<R::Type as RingBase>::Element>, V2: VectorView<<R::Type as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [El<R>], ring: S) {
        assert!(self.supports_ring(&ring));
        let log2_n = ZZ.abs_log2_ceil(&(lhs.len() + rhs.len()).try_into().unwrap()).unwrap();
        let lhs_prep = self.prepare_convolution_impl(lhs, log2_n);
        let rhs_prep = self.prepare_convolution_impl(rhs, log2_n);
        self.compute_convolution_impl(lhs_prep, &rhs_prep, dst);
    }

    fn specialize_prepared_convolution<F>(function: F) -> Result<F::Output, F>
        where F: PreparedConvolutionOperation<Self, R::Type>
    {
        Ok(function.execute())
    }
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

impl<R, A> PreparedConvolutionAlgorithm<R::Type> for NTTConvolution<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn prepare_convolution_operand<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_n_in = ZZ.abs_log2_ceil(&val.len().try_into().unwrap()).unwrap();
        let log2_n_out = log2_n_in + 1;
        return self.prepare_convolution_impl(val, log2_n_out);
    }

    fn compute_convolution_lhs_prepared<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [El<R>], ring: S) {
        assert!(ring.is_commutative());
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_lhs = ZZ.abs_log2_ceil(&lhs.data.len().try_into().unwrap()).unwrap();
        assert_eq!(lhs.data.len(), 1 << log2_lhs);
        let log2_n = ZZ.abs_log2_ceil(&(lhs.len + rhs.len()).try_into().unwrap()).unwrap().max(log2_lhs);
        assert!(log2_lhs <= log2_n);
        self.compute_convolution_prepared(lhs, &self.prepare_convolution_impl(rhs, log2_n), dst, ring);
    }

    fn compute_convolution_prepared<S: RingStore<Type = R::Type> + Copy>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [El<R>], ring: S) {
        assert!(ring.is_commutative());
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_lhs = ZZ.abs_log2_ceil(&lhs.data.len().try_into().unwrap()).unwrap();
        assert_eq!(1 << log2_lhs, lhs.data.len());
        let log2_rhs = ZZ.abs_log2_ceil(&rhs.data.len().try_into().unwrap()).unwrap();
        assert_eq!(1 << log2_rhs, rhs.data.len());
        match log2_lhs.cmp(&log2_rhs) {
            std::cmp::Ordering::Equal => self.compute_convolution_impl(self.clone_prepared_operand(lhs), rhs, dst),
            std::cmp::Ordering::Greater => self.compute_convolution_impl(PreparedConvolutionOperand { data: self.un_and_redo_fft(&rhs.data, log2_lhs), len: rhs.len }, lhs, dst),
            std::cmp::Ordering::Less => self.compute_convolution_impl(PreparedConvolutionOperand { data: self.un_and_redo_fft(&lhs.data, log2_rhs), len: lhs.len }, rhs, dst)
        }
    }

    fn compute_convolution_inner_product_prepared<'a, S, I>(&self, values: I, dst: &mut [El<R>], ring: S)
        where S: RingStore<Type = R::Type> + Copy, 
            I: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            Self: 'a,
            R: 'a,
            PreparedConvolutionOperand<R, A>: 'a
    {
        assert!(ring.get_ring() == self.ring.get_ring());
        let mut values_it = values.peekable();
        if values_it.peek().is_none() {
            return;
        }
        let expected_len = values_it.peek().unwrap().0.data.len().max(values_it.peek().unwrap().1.data.len());
        let mut current_log2_len = ZZ.abs_log2_ceil(&expected_len.try_into().unwrap()).unwrap();
        assert_eq!(expected_len, 1 << current_log2_len);
        let mut tmp = Vec::with_capacity_in(1 << current_log2_len, self.allocator.clone());
        tmp.resize_with(1 << current_log2_len, || ring.zero());
        for (lhs, rhs) in values_it {
            assert!(dst.len() >= lhs.len + rhs.len);
            let lhs_log2_len = ZZ.abs_log2_ceil(&lhs.data.len().try_into().unwrap()).unwrap();
            let rhs_log2_len = ZZ.abs_log2_ceil(&rhs.data.len().try_into().unwrap()).unwrap();
            let new_log2_len = current_log2_len.max(lhs_log2_len).max(rhs_log2_len);
            
            if current_log2_len < new_log2_len {
                tmp = self.un_and_redo_fft(&tmp, new_log2_len);
                current_log2_len = new_log2_len;
            }
            match (lhs_log2_len < current_log2_len, rhs_log2_len < current_log2_len) {
                (false, false) => Self::add_assign_elementwise_product(&lhs.data, &rhs.data, &mut tmp, RingRef::new(ring.get_ring())),
                (true, false) => Self::add_assign_elementwise_product(&self.un_and_redo_fft(&lhs.data, new_log2_len), &rhs.data, &mut tmp, RingRef::new(ring.get_ring())),
                (false, true) => Self::add_assign_elementwise_product(&lhs.data, &self.un_and_redo_fft(&rhs.data, new_log2_len), &mut tmp, RingRef::new(ring.get_ring())),
                (true, true) => Self::add_assign_elementwise_product(&self.un_and_redo_fft(&lhs.data, new_log2_len), &self.un_and_redo_fft(&rhs.data, new_log2_len), &mut tmp, RingRef::new(ring.get_ring())),
            }
        }
        self.get_fft(current_log2_len).unordered_inv_fft(&mut tmp[..], self.ring());
        for i in 0..min(dst.len(), 1 << current_log2_len) {
            self.ring.add_assign_ref(&mut dst[i], &tmp[i]);
        }
    }
}

#[test]
fn test_convolution() {
    let ring = zn_64::Zn::new(65537);
    let convolution = NTTConvolution::new_with(ring, Global);
    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
    super::generic_tests::test_prepared_convolution(&convolution, &ring, ring.one());
}