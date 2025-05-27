use std::alloc::{Allocator, Global};

use crate::boo::Boo;
use crate::{algorithms::fft::cooley_tuckey::CooleyTuckeyFFT, lazy::LazyVec};
use crate::homomorphism::*;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::zn::*;
use crate::integer::*;
use crate::seq::VectorView;

use super::ConvolutionAlgorithm;

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
    significant_entries: usize,
    ntt_data: Vec<El<R>, A>
}

impl<R> NTTConvolution<R>
    where R: RingStore + Clone,
        R::Type: ZnRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        Self::new_with(ring, Global)
    }
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

    fn get_ntt_table<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R::Type, R::Type, Identity<R>> {
        self.fft_algos.get_or_init(log2_n, || CooleyTuckeyFFT::for_zn(self.ring().clone(), log2_n).unwrap())
    }

    fn get_ntt_data<'a, V>(
        &self,
        data: V,
        data_prep: Option<&'a PreparedConvolutionOperand<R, A>>,
        significant_entries: usize,
    ) -> Boo<'a, Vec<El<R>, A>>
        where V: VectorView<El<R>>
    {
        assert!(data.len() <= significant_entries);
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let compute_result = || {
            let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
            result.extend(data.as_iter().map(|x| self.ring.clone_el(x)));
            result.resize_with(1 << log2_len, || self.ring.zero());
            self.get_ntt_table(log2_len).unordered_truncated_fft(&mut result, significant_entries);
            return result;
        };

        return if let Some(data_prep) = data_prep {
            assert!(data_prep.significant_entries >= significant_entries);
            Boo::Borrowed(&data_prep.ntt_data)
        } else {
            Boo::Owned(compute_result())
        }
    }

    fn prepare_convolution_impl<V>(
        &self,
        data: V,
        len_hint: Option<usize>
    ) -> PreparedConvolutionOperand<R, A>
        where V: VectorView<El<R>>
    {
        let significant_entries = if let Some(out_len) = len_hint {
            assert!(data.len() <= out_len);
            out_len
        } else {
            2 * data.len()
        };
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        result.extend(data.as_iter().map(|x| self.ring.clone_el(x)));
        result.resize_with(1 << log2_len, || self.ring.zero());
        self.get_ntt_table(log2_len).unordered_truncated_fft(&mut result, significant_entries);

        return PreparedConvolutionOperand {
            ntt_data: result,
            significant_entries: significant_entries
        };
    }

    fn compute_convolution_impl<V1, V2>(
        &self,
        lhs: V1,
        mut lhs_prep: Option<&PreparedConvolutionOperand<R, A>>,
        rhs: V2,
        mut rhs_prep: Option<&PreparedConvolutionOperand<R, A>>,
        dst: &mut [El<R>]
    )
        where V1: VectorView<El<R>>,
            V2: VectorView<El<R>>
    {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let len = lhs.len() + rhs.len() - 1;
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        if lhs_prep.is_some() && (lhs_prep.unwrap().significant_entries < len || lhs_prep.unwrap().ntt_data.len() != 1 << log2_len) {
            lhs_prep = None;
        }
        if rhs_prep.is_some() && (rhs_prep.unwrap().significant_entries < len || rhs_prep.unwrap().ntt_data.len() != 1 << log2_len) {
            rhs_prep = None;
        }

        let mut lhs_ntt = self.get_ntt_data(lhs, lhs_prep, len);
        let mut rhs_ntt = self.get_ntt_data(rhs, rhs_prep, len);
        if rhs_ntt.is_owned() {
            std::mem::swap(&mut lhs_ntt, &mut rhs_ntt);
        }
        let mut lhs_ntt = match lhs_ntt {
            Boo::Owned(data) => data,
            Boo::Borrowed(data) => {
                let mut copied_data = Vec::with_capacity_in(data.len(), self.allocator.clone());
                copied_data.extend(data.iter().map(|x| self.ring.clone_el(x)));
                copied_data
            }
        };

        for i in 0..len {
            self.ring.mul_assign_ref(&mut lhs_ntt[i], &rhs_ntt[i]);
        }

        self.get_ntt_table(log2_len).unordered_truncated_fft_inv(&mut lhs_ntt, len);

        for (i, x) in lhs_ntt.into_iter().enumerate().take(len) {
            self.ring.add_assign(&mut dst[i], x);
        }
    }
}

impl<R, A> ConvolutionAlgorithm<R::Type> for NTTConvolution<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn supports_ring<S: RingStore<Type = R::Type> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring.get_ring()
    }

    fn compute_convolution<S: RingStore<Type = R::Type> + Copy, V1: VectorView<<R::Type as RingBase>::Element>, V2: VectorView<<R::Type as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [El<R>], ring: S) {
        assert!(self.supports_ring(ring));
        self.compute_convolution_impl(
            lhs,
            None,
            rhs,
            None,
            dst
        )
    }

    fn prepare_convolution_operand<S, V>(&self, val: V, length_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>
    {
        assert!(self.supports_ring(ring));
        self.prepare_convolution_impl(
            val,
            length_hint
        )
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [El<R>], ring: S)
        where S: RingStore<Type = R::Type> + Copy, V1: VectorView<El<R>>, V2: VectorView<El<R>>
    {
        assert!(self.supports_ring(ring));
        self.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst
        )
    }
}

#[test]
fn test_convolution() {
    let ring = zn_64::Zn::new(65537);
    let convolution = NTTConvolution::new_with(ring, Global);
    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}