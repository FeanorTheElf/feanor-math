use std::alloc::{Allocator, Global};

use crate::cow::*;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::lazy::LazyVec;
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
pub struct NTTConvolution<R_main, R_twiddle, H, A = Global>
    where R_main: ?Sized + ZnRing,
        R_twiddle: ?Sized + ZnRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Clone
{
    hom: H,
    fft_algos: LazyVec<CooleyTuckeyFFT<R_main, R_twiddle, H>>,
    allocator: A
}

///
/// A prepared convolution operand for a [`NTTConvolution`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, A = Global>
    where R: ?Sized + ZnRing,
        A: Allocator + Clone
{
    significant_entries: usize,
    ntt_data: Vec<R::Element, A>
}

impl<R> NTTConvolution<R::Type, R::Type, Identity<R>>
    where R: RingStore + Clone,
        R::Type: ZnRing
{
    ///
    /// Creates a new [`NTTConvolution`].
    /// 
    /// Note that this convolution will be able to compute convolutions whose output is
    /// of length `<= n`, where `n` is the largest power of two such that the given ring
    /// has a primitive `n`-th root of unity.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R) -> Self {
        Self::new_with(ring.into_identity(), Global)
    }
}

impl<R_main, R_twiddle, H, A> NTTConvolution<R_main, R_twiddle, H, A>
    where R_main: ?Sized + ZnRing,
        R_twiddle: ?Sized + ZnRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Clone
{
    ///
    /// Creates a new [`NTTConvolution`].
    /// 
    /// Note that this convolution will be able to compute convolutions whose output is
    /// of length `<= n`, where `n` is the largest power of two such that the domain of
    /// the given homomorphism has a primitive `n`-th root of unity.
    /// 
    /// Internally, twiddle factors for the underlying NTT will be stored as elements of
    /// the domain of the given homomorphism, while the convolutions are performed over the
    /// codomain. This can be used for more efficient NTTs, see e.g. [`zn_64::ZnFastmul`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with(hom: H, allocator: A) -> Self {
        Self {
            fft_algos: LazyVec::new(),
            hom: hom,
            allocator: allocator
        }
    }

    ///
    /// Returns the ring over which this object can compute convolutions.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn ring(&self) -> RingRef<'_, R_main> {
        RingRef::new(self.hom.codomain().get_ring())
    }

    fn get_ntt_table<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R_main, R_twiddle, H> {
        self.fft_algos.get_or_init(log2_n, || CooleyTuckeyFFT::for_zn_with_hom(self.hom.clone(), log2_n).expect("NTTConvolution was instantiated with parameters that don't support this length"))
    }

    fn get_ntt_data<'a, V>(
        &self,
        data: V,
        data_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        significant_entries: usize,
    ) -> MyCow<'a, Vec<R_main::Element, A>>
        where V: VectorView<R_main::Element>
    {
        assert!(data.len() <= significant_entries);
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let compute_result = || {
            let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
            result.extend(data.as_iter().map(|x| self.ring().clone_el(x)));
            result.resize_with(1 << log2_len, || self.ring().zero());
            self.get_ntt_table(log2_len).unordered_truncated_fft(&mut result, significant_entries);
            return result;
        };

        return if let Some(data_prep) = data_prep {
            assert!(data_prep.significant_entries >= significant_entries);
            MyCow::Borrowed(&data_prep.ntt_data)
        } else {
            MyCow::Owned(compute_result())
        }
    }

    fn prepare_convolution_impl<V>(
        &self,
        data: V,
        len_hint: Option<usize>
    ) -> PreparedConvolutionOperand<R_main, A>
        where V: VectorView<R_main::Element>
    {
        let significant_entries = if let Some(out_len) = len_hint {
            assert!(data.len() <= out_len);
            out_len
        } else {
            2 * data.len()
        };
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        result.extend(data.as_iter().map(|x| self.ring().clone_el(x)));
        result.resize_with(1 << log2_len, || self.ring().zero());
        self.get_ntt_table(log2_len).unordered_truncated_fft(&mut result, significant_entries);

        return PreparedConvolutionOperand {
            ntt_data: result,
            significant_entries: significant_entries
        };
    }

    fn compute_convolution_impl<V1, V2>(
        &self,
        lhs: V1,
        mut lhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        rhs: V2,
        mut rhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        dst: &mut [R_main::Element]
    )
        where V1: VectorView<R_main::Element>,
            V2: VectorView<R_main::Element>
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
        let mut lhs_ntt = lhs_ntt.to_mut_with(|data| {
            let mut copied_data = Vec::with_capacity_in(data.len(), self.allocator.clone());
            copied_data.extend(data.iter().map(|x| self.ring().clone_el(x)));
            copied_data
        });

        for i in 0..len {
            self.ring().mul_assign_ref(&mut lhs_ntt[i], &rhs_ntt[i]);
        }

        self.get_ntt_table(log2_len).unordered_truncated_fft_inv(&mut lhs_ntt, len);

        for (i, x) in lhs_ntt.drain(..).enumerate().take(len) {
            self.ring().add_assign(&mut dst[i], x);
        }
    }
}

impl<R_main, R_twiddle, H, A> ConvolutionAlgorithm<R_main> for NTTConvolution<R_main, R_twiddle, H, A>
    where R_main: ?Sized + ZnRing,
        R_twiddle: ?Sized + ZnRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R_main, A>;

    fn supports_ring<S: RingStore<Type = R_main> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring().get_ring()
    }

    fn compute_convolution<S: RingStore<Type = R_main> + Copy, V1: VectorView<<R_main as RingBase>::Element>, V2: VectorView<<R_main as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [R_main::Element], ring: S) {
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
        where S: RingStore<Type = R_main> + Copy, V: VectorView<R_main::Element>
    {
        assert!(self.supports_ring(ring));
        self.prepare_convolution_impl(
            val,
            length_hint
        )
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R_main::Element], ring: S)
        where S: RingStore<Type = R_main> + Copy, V1: VectorView<R_main::Element>, V2: VectorView<R_main::Element>
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
    let convolution = NTTConvolution::new_with(ring.identity(), Global);
    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}