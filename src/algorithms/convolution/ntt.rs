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
        A: Allocator + Clone + Send + Sync
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
        A: Allocator + Clone + Send + Sync
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
        Self::new_with_hom(ring.into_identity(), Global)
    }
}

impl<R_main, R_twiddle, H, A> NTTConvolution<R_main, R_twiddle, H, A>
    where R_main: ?Sized + ZnRing,
        R_twiddle: ?Sized + ZnRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Clone + Send + Sync
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
    pub fn new_with_hom(hom: H, allocator: A) -> Self {
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

    #[instrument(skip_all, level = "trace")]
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

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_ntt<'a, V1, V2>(&self,
        lhs: V1,
        mut lhs_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        rhs: V2,
        mut rhs_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        len: usize
    ) -> MyCow<'a, Vec<R_main::Element, A>>
        where V1: VectorView<R_main::Element>,
            V2: VectorView<R_main::Element>
    {
        if lhs.len() == 0 || rhs.len() == 0 {
            return MyCow::Owned(Vec::new_in(self.allocator.clone()));
        }
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
        let lhs_ntt_data = lhs_ntt.to_mut_with(|data| {
            let mut copied_data = Vec::with_capacity_in(data.len(), self.allocator.clone());
            copied_data.extend(data.iter().map(|x| self.ring().clone_el(x)));
            copied_data
        });

        for i in 0..len {
            self.ring().mul_assign_ref(&mut lhs_ntt_data[i], &rhs_ntt[i]);
        }
        return lhs_ntt;
    }

    #[instrument(skip_all, level = "trace")]
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

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_impl(
        &self,
        lhs: &[R_main::Element],
        lhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        rhs: &[R_main::Element],
        rhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        dst: &mut [R_main::Element]
    ) {
        assert!(lhs.len() + rhs.len() - 1 <= dst.len());
        let len = lhs.len() + rhs.len() - 1;
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut lhs_ntt = self.compute_convolution_ntt(lhs, lhs_prep, rhs, rhs_prep, len);
        let lhs_ntt = lhs_ntt.to_mut_with(|_| unreachable!());
        self.get_ntt_table(log2_len).unordered_truncated_fft_inv(&mut lhs_ntt[..], len);
        for (i, x) in lhs_ntt.drain(..).enumerate().take(len) {
            self.ring().add_assign(&mut dst[i], x);
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_sum_impl(&self, values: &[(&[R_main::Element], Option<&PreparedConvolutionOperand<R_main, A>>, &[R_main::Element], Option<&PreparedConvolutionOperand<R_main, A>>)], dst: &mut [R_main::Element]) {
        let len = dst.len();
        let log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut buffer = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        buffer.resize_with(1 << log2_len, || self.ring().zero());

        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            assert!(lhs.len() + rhs.len() - 1 <= len);

            let res_ntt = self.compute_convolution_ntt(lhs, *lhs_prep, rhs, *rhs_prep, len);
            for i in 0..len {
                self.ring().add_assign_ref(&mut buffer[i], &res_ntt[i]);
            }
        }
        self.get_ntt_table(log2_len).unordered_truncated_fft_inv(&mut buffer, len);
        for (i, x) in buffer.drain(..).enumerate().take(len) {
            self.ring().add_assign(&mut dst[i], x);
        }
    }
}

impl<R_main, R_twiddle, H, A> ConvolutionAlgorithm<R_main> for NTTConvolution<R_main, R_twiddle, H, A>
    where R_main: ?Sized + ZnRing,
        R_twiddle: ?Sized + ZnRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Clone + Send + Sync
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R_main, A>;

    fn supports_ring(&self, ring: &R_main) -> bool {
        ring == self.ring().get_ring()
    }

    fn compute_convolution(&self, lhs: &[R_main::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R_main::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R_main::Element], ring: &R_main) {
        assert!(self.supports_ring(ring));
        self.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst
        )
    }

    fn prepare_convolution_operand(&self, val: &[R_main::Element], length_hint: Option<usize>, ring: &R_main) -> Self::PreparedConvolutionOperand {
        assert!(self.supports_ring(ring));
        self.prepare_convolution_impl(
            val,
            length_hint
        )
    }

    fn compute_convolution_sum(&self, values: &[(&[R_main::Element], Option<&Self::PreparedConvolutionOperand>, &[R_main::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R_main::Element], ring: &R_main) {
        assert!(self.supports_ring(ring));
        self.compute_convolution_sum_impl(
            values, 
            dst
        )
    }
}

#[cfg(test)]
use test::Bencher;
use tracing::instrument;
#[cfg(test)]
use crate::algorithms::convolution::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::rings::zn::zn_64::{Zn64B, Zn64BBase, Zn64BEl};
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_convolution() {
    LogAlgorithmSubscriber::init_test();
    let ring = zn_64::Zn64B::new(65537);
    let convolution = NTTConvolution::new(ring);
    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}

#[cfg(test)]
fn run_benchmark<F>(ring: Zn64B, bencher: &mut Bencher, mut f: F)
    where F: FnMut(&[(&[Zn64BEl], Option<&PreparedConvolutionOperand<Zn64BBase>>, &[Zn64BEl], Option<&PreparedConvolutionOperand<Zn64BBase>>)], &mut [Zn64BEl], Zn64B)
{
    let mut expected = (0..512).map(|_| ring.zero()).collect::<Vec<_>>();
    let value = (0..256).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    STANDARD_CONVOLUTION.compute_convolution(
        &value,
        None,
        &value,
        None,
        &mut expected,
        ring.get_ring()
    );

    let mut i = 1;
    let mut actual = Vec::with_capacity(511);
    let hom = ring.can_hom(&StaticRing::<i64>::RING).unwrap();
    bencher.iter(|| {
        actual.clear();
        actual.resize_with(511, || ring.zero());
        f(
            &(0..256).map(|j| (
                (0..256).map(|k| hom.map(i * j as i64 * k)).collect::<Vec<_>>(),
                None,
                (0..256).map(|k| hom.map(i * j as i64 * k)).collect::<Vec<_>>(),
                None
            )).collect::<Vec<_>>()
                .iter().map(|(lhs, lhs_prep, rhs, rhs_prep)| (&lhs[..], *lhs_prep, &rhs[..], *rhs_prep)).collect::<Vec<_>>(),
            &mut actual,
            ring
        );
        let factor = hom.map(i * i * 128 * 511 * 85);
        for (l, r) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(ring, ring.mul_ref(l, &factor), r);
        }
        i += 1;
    });
}

#[bench]
fn bench_convolution_sum(bencher: &mut Bencher) {
    LogAlgorithmSubscriber::init_test();
    let ring = zn_64::Zn64B::new(65537);
    let convolution = NTTConvolution::new(ring);

    run_benchmark(ring, bencher, |values, dst, _| convolution.compute_convolution_sum_impl(values, dst));
}

#[bench]
fn bench_convolution_sum_default(bencher: &mut Bencher) {
    LogAlgorithmSubscriber::init_test();
    let ring = zn_64::Zn64B::new(65537);
    let convolution = NTTConvolution::new(ring);

    run_benchmark(ring, bencher, |values, dst, ring| {
        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            convolution.compute_convolution(lhs, *lhs_prep, rhs, *rhs_prep, dst, ring.get_ring());
        }
    });
}