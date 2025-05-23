use std::alloc::{Allocator, Global};

use crate::{algorithms::fft::cooley_tuckey::CooleyTuckeyFFT, lazy::LazyVec};
use crate::homomorphism::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::zn::*;
use crate::integer::*;
use crate::seq::{VectorView, VectorViewMut};

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
    original_len: usize,
    significant_entries: usize,
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

    fn add_assign_elementwise_product(&self, lhs: &[El<R>], rhs: &[El<R>], dst: &mut [El<R>]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            self.ring.add_assign(&mut dst[i], self.ring.mul_ref(&lhs[i], &rhs[i]));
        }
    }

    fn compute_convolution_impl(&self, mut lhs: PreparedConvolutionOperand<R, A>, rhs: &PreparedConvolutionOperand<R, A>, dst: &mut [El<R>]) {
        if lhs.original_len == 0 || rhs.original_len == 0 {
            return;
        }
        let log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&lhs.data.len().try_into().unwrap()).unwrap();
        assert_eq!(1 << log2_n, lhs.data.len());
        assert_eq!(1 << log2_n, rhs.data.len());

        let significant_entries =  lhs.original_len + rhs.original_len - 1;
        assert!(significant_entries <= lhs.significant_entries);
        assert!(significant_entries <= rhs.significant_entries);
        assert!(dst.len() >= significant_entries);

        for i in 0..significant_entries {
            self.ring.mul_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        self.get_fft(log2_n).unordered_truncated_fft_inv(&mut lhs.data, significant_entries);
        for (i, x) in lhs.data.into_iter().take(significant_entries).enumerate() {
            self.ring.add_assign(&mut dst[i], x);
        }
    }

    fn retrieve_original_data(&self, value: &PreparedConvolutionOperand<R, A>) -> Vec<El<R>, A> {
        let log2_n = ZZ.abs_log2_ceil(&value.data.len().try_into().unwrap()).unwrap();
        assert_eq!(value.data.len(), 1 << log2_n);
        let mut result = Vec::with_capacity_in(1 << log2_n, self.allocator.clone());
        result.extend(value.data.iter().map(|x| self.ring.clone_el(x)));
        self.get_fft(log2_n).unordered_truncated_fft_inv(&mut result, value.significant_entries);
        result.truncate(value.original_len);
        return result;
    }

    fn get_fft<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R::Type, R::Type, Identity<R>> {
        self.fft_algos.get_or_init(log2_n, || CooleyTuckeyFFT::for_zn(self.ring().clone(), log2_n).unwrap())
    }

    fn clone_prepared_operand(&self, operand: &PreparedConvolutionOperand<R, A>) -> PreparedConvolutionOperand<R, A> {
        let mut result = Vec::with_capacity_in(operand.data.len(), self.allocator.clone());
        result.extend(operand.data.iter().map(|x| self.ring.clone_el(x)));
        return PreparedConvolutionOperand {
            original_len: operand.original_len,
            significant_entries: operand.significant_entries,
            data: result
        };
    }
    
    fn prepare_convolution_impl<V: VectorView<El<R>>>(&self, val: V, len_hint: Option<usize>) -> PreparedConvolutionOperand<R, A> {
        let out_len = if let Some(len_hint) = len_hint {
            assert!(val.len() <= len_hint);
            len_hint
        } else {
            2 * val.len()
        };
        let log2_out_len = ZZ.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        let mut result = Vec::with_capacity_in(1 << log2_out_len, self.allocator.clone());
        result.extend(val.as_iter().map(|x| self.ring.clone_el(x)));
        result.resize_with(1 << log2_out_len, || self.ring.zero());
        self.get_fft(log2_out_len).unordered_truncated_fft(&mut result[..], out_len);
        return PreparedConvolutionOperand {
            significant_entries: out_len,
            original_len: val.len(),
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
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let out_len = lhs.len() + rhs.len() - 1;
        self.compute_convolution_impl(
            self.prepare_convolution_impl(lhs, Some(out_len)),
            &self.prepare_convolution_impl(rhs, Some(out_len)),
            dst
        );
    }

    fn specialize_prepared_convolution<F>(function: F) -> F::Output
        where F: PreparedConvolutionOperation<Self, R::Type>
    {
        function.execute()
    }
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

impl<R, A> PreparedConvolutionAlgorithm<R::Type> for NTTConvolution<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn prepare_convolution_operand<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, val: V, len_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand {
        assert!(self.supports_ring(&ring));
        return self.prepare_convolution_impl(val, len_hint);
    }

    fn prepared_operand_len(&self, prepared_operand: &Self::PreparedConvolutionOperand) -> usize {
        prepared_operand.original_len
    }

    fn prepared_operand_data<S, V>(&self, prepared_operand: &Self::PreparedConvolutionOperand, mut output: V, ring: S)
        where S: RingStore<Type = R::Type> + Copy,
            V: VectorViewMut<<R::Type as RingBase>::Element>
    {
        assert!(self.supports_ring(&ring));
        let data = self.retrieve_original_data(prepared_operand);
        let len = data.len();
        assert!(output.len() >= len);
        for (i, x) in data.into_iter().enumerate() {
            *output.at_mut(i) = x;
        }
        for i in len..output.len() {
            *output.at_mut(i) = ring.zero();
        }
    }

    fn compute_convolution_lhs_prepared<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [El<R>], ring: S) {
        assert!(self.supports_ring(&ring));
        if lhs.original_len == 0 || rhs.len() == 0 {
            return;
        }
        let out_len = lhs.original_len + rhs.len() - 1;
        let log2_out_len = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        if out_len <= lhs.significant_entries && lhs.data.len() == 1 << log2_out_len {
            self.compute_convolution_impl(
                self.prepare_convolution_impl(rhs, Some(out_len)),
                lhs,
                dst
            );
        } else {
            let lhs_data = self.retrieve_original_data(lhs);
            self.compute_convolution(lhs_data, rhs, dst, ring);
        }
    }

    fn compute_convolution_prepared<S: RingStore<Type = R::Type> + Copy>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [El<R>], ring: S) {
        assert!(self.supports_ring(&ring));
        if lhs.original_len == 0 || rhs.original_len == 0 {
            return;
        }
        let out_len = lhs.original_len + rhs.original_len - 1;
        let log2_out_len = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        match (
            out_len <= lhs.significant_entries && lhs.data.len() == 1 << log2_out_len, 
            out_len <= rhs.significant_entries && rhs.data.len() == 1 << log2_out_len
        ) {
            (true, true) => self.compute_convolution_impl(self.clone_prepared_operand(lhs), rhs, dst),
            (true, false) => self.compute_convolution_lhs_prepared(lhs, self.retrieve_original_data(rhs), dst, ring),
            (false, true) => self.compute_convolution_lhs_prepared(rhs, self.retrieve_original_data(lhs), dst, ring),
            (false, false) => self.compute_convolution(self.retrieve_original_data(lhs), self.retrieve_original_data(rhs), dst, ring)
        }
    }

    fn compute_convolution_inner_product_prepared<'a, S, I>(&self, values: I, dst: &mut [El<R>], ring: S)
        where S: RingStore<Type = R::Type> + Copy, 
            I: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            Self: 'a,
            R: 'a,
            PreparedConvolutionOperand<R, A>: 'a
    {
        assert!(self.supports_ring(&ring));
        let data = values.collect::<Vec<_>>();
        let out_len = data.iter().map(|(lhs, rhs)| (lhs.original_len + rhs.original_len).saturating_sub(1)).max();
        if out_len.is_none() || out_len == Some(0) {
            return;
        }
        let out_len = out_len.unwrap();
        let log2_out_len = StaticRing::<i64>::RING.abs_log2_ceil(&out_len.try_into().unwrap()).unwrap();
        let mut tmp = Vec::with_capacity_in(1 << log2_out_len, self.allocator.clone());
        tmp.resize_with(1 << log2_out_len, || self.ring.zero());

        for (lhs, rhs) in data.into_iter() {
            match (
                out_len <= lhs.significant_entries && (1 << log2_out_len) == lhs.data.len(), 
                out_len <= rhs.significant_entries && (1 << log2_out_len) == rhs.data.len()
            ) {
                (true, true) => self.add_assign_elementwise_product(&lhs.data, &rhs.data, &mut tmp),
                (true, false) => self.add_assign_elementwise_product(
                    &lhs.data, 
                    &self.prepare_convolution_impl(self.retrieve_original_data(rhs), Some(out_len)).data, 
                    &mut tmp
                ),
                (false, true) => self.add_assign_elementwise_product(
                    &self.prepare_convolution_impl(self.retrieve_original_data(lhs), Some(out_len)).data, 
                    &rhs.data, 
                    &mut tmp
                ),
                (false, false) => self.add_assign_elementwise_product(
                    &self.prepare_convolution_impl(self.retrieve_original_data(lhs), Some(out_len)).data, 
                    &self.prepare_convolution_impl(self.retrieve_original_data(rhs), Some(out_len)).data, 
                    &mut tmp
                )
            }
        }

        self.get_fft(log2_out_len).unordered_truncated_fft_inv(&mut tmp, out_len);
        for (i, x) in tmp.into_iter().enumerate().take(out_len) {
            ring.add_assign(&mut dst[i], x);
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