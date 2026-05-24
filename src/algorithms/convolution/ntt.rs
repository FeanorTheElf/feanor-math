use std::alloc::{Allocator, Global};

use elsa::sync::FrozenMap;
use tracing::instrument;

use super::ConvolutionAlgorithm;
use crate::algorithms::cyclotomic::{get_prim_root_of_unity_pow2_zn, is_prim_root_of_unity_pow2};
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use crate::algorithms::int_factor::factor;
use crate::cow::*;
use crate::homomorphism::*;
use crate::prelude::*;
use crate::ring_impls::zn::*;
use crate::seq::VectorView;

/// Computes the convolution over a finite field that has suitable roots of unity
/// using a power-of-two length FFT (sometimes called Number-Theoretic Transform,
/// NTT in this context).
#[stability::unstable(feature = "enable")]
pub struct NTTConvolution<R_main, R_twiddle, H, A = Global>
where
    R_main: ?Sized + RingBase,
    R_twiddle: ?Sized + RingBase + DivisibilityRing,
    H: Homomorphism<R_twiddle, R_main> + Clone,
    A: Allocator + Clone + Send + Sync,
{
    hom: H,
    base_root_of_unity: El<H::DomainStore>,
    max_log2_len: usize,
    fft_algos: FrozenMap<usize, Box<CooleyTuckeyFFT<R_main, R_twiddle, H>>>,
    allocator: A,
}

/// A prepared convolution operand for a [`NTTConvolution`].
#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, A = Global>
where
    R: ?Sized + RingBase,
    A: Allocator + Clone + Send + Sync,
{
    significant_entries: usize,
    ntt_data: Vec<R::Element, A>,
}

impl<R> NTTConvolution<R::Ring, R::Ring, Identity<R>>
where
    R: RingStore + Clone,
    R::Ring: DivisibilityRing,
{
    /// Creates a new [`NTTConvolution`].
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R, base_root_of_unity: El<R>, max_log2_len: usize) -> Self {
        Self::new_with_hom(ring.into_identity(), base_root_of_unity, max_log2_len, Global)
    }

    /// Creates a new [`NTTConvolution`].
    ///
    /// For the motivation behind separating the twiddle ring and the main ring, see
    /// [`CooleyTuckeyFFT`].
    ///
    /// # Performance
    ///
    /// This function will factor the modulus `n` of the ring, which in some cases is a very
    /// computationally demanding task.
    #[stability::unstable(feature = "enable")]
    pub fn for_zn(ring: R) -> Self
    where
        R::Ring: ZnRing,
    {
        Self::for_zn_with_hom(ring.into_identity())
    }
}

impl<R_main, R_twiddle, H> NTTConvolution<R_main, R_twiddle, H>
where
    R_main: ?Sized + RingBase,
    R_twiddle: ?Sized + RingBase + DivisibilityRing,
    H: Homomorphism<R_twiddle, R_main> + Clone,
{
    /// Creates a new [`NTTConvolution`].
    ///
    /// For the motivation behind separating the twiddle ring and the main ring, see
    /// [`CooleyTuckeyFFT`].
    ///
    /// # Performance
    ///
    /// This function will factor the modulus `n` of the ring, which in some cases is a very
    /// computationally demanding task.
    #[stability::unstable(feature = "enable")]
    pub fn for_zn_with_hom(hom: H) -> Self
    where
        R_twiddle: ZnRing,
    {
        let two = int_cast(2, ZZbig, ZZi64);
        let max_log2_len = factor(
            ZZbig,
            int_cast(hom.domain().modulus().clone(), ZZbig, hom.domain().integer_ring()),
        )
        .into_iter()
        .map(|(p, _)| {
            if ZZbig.eq_el(&p, &two) {
                0
            } else {
                ZZbig.abs_lowest_set_bit(&ZZbig.sub(p, ZZbig.one())).unwrap()
            }
        })
        .min()
        .unwrap();
        let root_of_unity = get_prim_root_of_unity_pow2_zn(hom.domain(), max_log2_len).unwrap();
        Self::new_with_hom(hom, root_of_unity, max_log2_len, Global)
    }
}

impl<R_main, R_twiddle, H, A> NTTConvolution<R_main, R_twiddle, H, A>
where
    R_main: ?Sized + RingBase,
    R_twiddle: ?Sized + RingBase + DivisibilityRing,
    H: Homomorphism<R_twiddle, R_main> + Clone,
    A: Allocator + Clone + Send + Sync,
{
    /// Creates a new [`NTTConvolution`].
    ///
    /// For the motivation behind separating the twiddle ring and the main ring, see
    /// [`CooleyTuckeyFFT`].
    #[stability::unstable(feature = "enable")]
    pub fn new_with_hom(hom: H, base_root_of_unity: El<H::DomainStore>, max_log2_len: usize, allocator: A) -> Self {
        assert!(is_prim_root_of_unity_pow2(
            hom.domain(),
            &base_root_of_unity,
            max_log2_len
        ));
        Self {
            fft_algos: FrozenMap::new(),
            hom,
            max_log2_len,
            base_root_of_unity,
            allocator,
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn max_log2_len(&self) -> usize { self.max_log2_len }

    #[stability::unstable(feature = "enable")]
    pub fn change_ring<R_main_new, H_new>(self, hom: H_new) -> NTTConvolution<R_main_new, R_twiddle, H_new, A>
    where
        R_main_new: ?Sized + RingBase,
        H_new: Clone + Homomorphism<R_twiddle, R_main_new>,
    {
        let new_map = FrozenMap::new();
        for (log2_len, fft) in self.fft_algos.into_tuple_vec().into_iter() {
            _ = new_map.insert(log2_len, Box::new(fft.change_ring(hom.clone()).0));
        }
        NTTConvolution {
            fft_algos: new_map,
            hom,
            base_root_of_unity: self.base_root_of_unity,
            max_log2_len: self.max_log2_len,
            allocator: self.allocator,
        }
    }

    /// Returns the ring over which this object can compute convolutions.
    #[stability::unstable(feature = "enable")]
    pub fn ring(&self) -> RingRef<'_, R_main> { RingRef::from(self.hom.codomain().get_ring()) }

    fn get_ntt_table<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R_main, R_twiddle, H> {
        if let Some(res) = self.fft_algos.get(&log2_n) {
            res
        } else {
            assert!(
                log2_n <= self.max_log2_len,
                "this NTTConvolution only supports convolutions up to length {}",
                1 << self.max_log2_len
            );
            let root_of_unity = self
                .hom
                .domain()
                .pow(self.base_root_of_unity.clone(), 1 << (self.max_log2_len - log2_n));
            self.fft_algos.insert(
                log2_n,
                Box::new(CooleyTuckeyFFT::new_with_hom(self.hom.clone(), root_of_unity, log2_n)),
            )
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn get_ntt_data<'a, V>(
        &self,
        data: V,
        data_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        significant_entries: usize,
    ) -> MyCow<'a, Vec<R_main::Element, A>>
    where
        V: VectorView<R_main::Element>,
    {
        assert!(data.len() <= significant_entries);
        let log2_len = ZZi64.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let compute_result = || {
            let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
            result.extend(data.as_iter().map(|x| x.clone()));
            result.resize_with(1 << log2_len, || self.ring().zero());
            self.get_ntt_table(log2_len)
                .unordered_truncated_fft(&mut result, significant_entries);
            return result;
        };

        return if let Some(data_prep) = data_prep {
            assert!(data_prep.significant_entries >= significant_entries);
            MyCow::Borrowed(&data_prep.ntt_data)
        } else {
            MyCow::Owned(compute_result())
        };
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_ntt<'a, V1, V2>(
        &self,
        lhs: V1,
        mut lhs_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        rhs: V2,
        mut rhs_prep: Option<&'a PreparedConvolutionOperand<R_main, A>>,
        len: usize,
    ) -> MyCow<'a, Vec<R_main::Element, A>>
    where
        V1: VectorView<R_main::Element>,
        V2: VectorView<R_main::Element>,
    {
        let log2_len = ZZi64.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        if lhs_prep.is_some()
            && (lhs_prep.unwrap().significant_entries < len || lhs_prep.unwrap().ntt_data.len() != 1 << log2_len)
        {
            lhs_prep = None;
        }
        if rhs_prep.is_some()
            && (rhs_prep.unwrap().significant_entries < len || rhs_prep.unwrap().ntt_data.len() != 1 << log2_len)
        {
            rhs_prep = None;
        }

        let mut lhs_ntt = self.get_ntt_data(lhs, lhs_prep, len);
        let mut rhs_ntt = self.get_ntt_data(rhs, rhs_prep, len);
        if rhs_ntt.is_owned() {
            std::mem::swap(&mut lhs_ntt, &mut rhs_ntt);
        }
        let lhs_ntt_data = lhs_ntt.to_mut_with(|data| {
            let mut copied_data = Vec::with_capacity_in(data.len(), self.allocator.clone());
            copied_data.extend(data.iter().map(|x| x.clone()));
            copied_data
        });

        for i in 0..len {
            self.ring().mul_assign_ref(&mut lhs_ntt_data[i], &rhs_ntt[i]);
        }
        return lhs_ntt;
    }

    #[instrument(skip_all, level = "trace")]
    fn prepare_convolution_impl<V>(&self, data: V, len_hint: Option<usize>) -> PreparedConvolutionOperand<R_main, A>
    where
        V: VectorView<R_main::Element>,
    {
        let significant_entries = if let Some(out_len) = len_hint {
            assert!(
                data.len() <= out_len,
                "length_hint cannot be smaller than the length of a single convolution operand"
            );
            out_len
        } else {
            2 * data.len()
        };
        if significant_entries == 0 {
            return PreparedConvolutionOperand {
                ntt_data: Vec::new_in(self.allocator.clone()),
                significant_entries,
            };
        }
        let log2_len = ZZi64.abs_log2_ceil(&significant_entries.try_into().unwrap()).unwrap();

        let mut result = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        result.extend(data.as_iter().map(|x| x.clone()));
        result.resize_with(1 << log2_len, || self.ring().zero());
        self.get_ntt_table(log2_len)
            .unordered_truncated_fft(&mut result, significant_entries);

        return PreparedConvolutionOperand {
            ntt_data: result,
            significant_entries,
        };
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_impl(
        &self,
        lhs: &[R_main::Element],
        lhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        rhs: &[R_main::Element],
        rhs_prep: Option<&PreparedConvolutionOperand<R_main, A>>,
        dst: &mut [R_main::Element],
    ) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let len = lhs.len() + rhs.len() - 1;
        assert!(len <= dst.len() + 1);

        let log2_len = ZZi64.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut lhs_ntt = self.compute_convolution_ntt(lhs, lhs_prep, rhs, rhs_prep, len);
        let lhs_ntt = lhs_ntt.to_mut_with(|_| unreachable!());
        self.get_ntt_table(log2_len)
            .unordered_truncated_fft_inv(&mut lhs_ntt[..], len);
        for (i, x) in lhs_ntt.drain(..).enumerate().take(len) {
            self.ring().add_assign(&mut dst[i], x);
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_sum_impl(
        &self,
        values: &[(
            &[R_main::Element],
            Option<&PreparedConvolutionOperand<R_main, A>>,
            &[R_main::Element],
            Option<&PreparedConvolutionOperand<R_main, A>>,
        )],
        dst: &mut [R_main::Element],
    ) {
        if values.len() == 0 {
            return;
        }
        let len = values
            .iter()
            .map(|(l, _, r, _)| {
                if l.len() == 0 || r.len() == 0 {
                    0
                } else {
                    l.len() + r.len() - 1
                }
            })
            .max()
            .unwrap();
        if len == 0 {
            return;
        }
        assert!(len <= dst.len() + 1);
        let log2_len = ZZi64.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut buffer = Vec::with_capacity_in(1 << log2_len, self.allocator.clone());
        buffer.resize_with(1 << log2_len, || self.ring().zero());

        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            if lhs.len() == 0 || rhs.len() == 0 {
                continue;
            }
            assert!(lhs.len() + rhs.len() <= len + 1);

            let res_ntt = self.compute_convolution_ntt(lhs, *lhs_prep, rhs, *rhs_prep, len);
            for i in 0..len {
                self.ring().add_assign_ref(&mut buffer[i], &res_ntt[i]);
            }
        }
        self.get_ntt_table(log2_len)
            .unordered_truncated_fft_inv(&mut buffer, len);
        for (i, x) in buffer.drain(..).enumerate().take(len) {
            self.ring().add_assign(&mut dst[i], x);
        }
    }
}

impl<R_main, R_twiddle, H, A> ConvolutionAlgorithm<R_main> for NTTConvolution<R_main, R_twiddle, H, A>
where
    R_main: ?Sized + RingBase,
    R_twiddle: ?Sized + ZnRing + DivisibilityRing,
    H: Homomorphism<R_twiddle, R_main> + Clone,
    A: Allocator + Clone + Send + Sync,
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R_main, A>;

    fn supports_ring(&self, ring: &R_main) -> bool { ring == self.ring().get_ring() }

    fn compute_convolution(
        &self,
        lhs: &[R_main::Element],
        lhs_prep: Option<&Self::PreparedConvolutionOperand>,
        rhs: &[R_main::Element],
        rhs_prep: Option<&Self::PreparedConvolutionOperand>,
        dst: &mut [R_main::Element],
        ring: &R_main,
    ) {
        assert!(self.supports_ring(ring));
        self.compute_convolution_impl(lhs, lhs_prep, rhs, rhs_prep, dst)
    }

    fn prepare_convolution_operand(
        &self,
        val: &[R_main::Element],
        length_hint: Option<usize>,
        ring: &R_main,
    ) -> Self::PreparedConvolutionOperand {
        assert!(self.supports_ring(ring));
        self.prepare_convolution_impl(val, length_hint)
    }

    fn compute_convolution_sum(
        &self,
        values: &[(
            &[R_main::Element],
            Option<&Self::PreparedConvolutionOperand>,
            &[R_main::Element],
            Option<&Self::PreparedConvolutionOperand>,
        )],
        dst: &mut [R_main::Element],
        ring: &R_main,
    ) {
        assert!(self.supports_ring(ring));
        self.compute_convolution_sum_impl(values, dst)
    }
}

#[cfg(test)]
use test::Bencher;

#[cfg(test)]
use crate::algorithms::convolution::KaratsubaAlgorithm;
#[cfg(test)]
use crate::ring_impls::zn::zn_64b::{Zn64B, Zn64BBase, Zn64BEl};

#[test]
fn test_convolution() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = zn_64b::Zn64B::new(65537);
    let convolution = NTTConvolution::for_zn(ring);
    super::generic_tests::test_convolution(&convolution, &ring, ring.one());
}

#[cfg(test)]
fn run_benchmark<F>(ring: Zn64B, bencher: &mut Bencher, mut f: F)
where
    F: FnMut(
        &[(
            &[Zn64BEl],
            Option<&PreparedConvolutionOperand<Zn64BBase>>,
            &[Zn64BEl],
            Option<&PreparedConvolutionOperand<Zn64BBase>>,
        )],
        &mut [Zn64BEl],
        Zn64B,
    ),
{
    let mut expected = (0..512).map(|_| ring.zero()).collect::<Vec<_>>();
    let value: Vec<Zn64BEl> = (0..256).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    KaratsubaAlgorithm::new_with_alloc(4, Global).compute_convolution(
        &value,
        None,
        &value,
        None,
        &mut expected,
        ring.get_ring(),
    );

    let mut i = 1;
    let mut actual = Vec::with_capacity(511);
    let hom = ring.can_hom(&ZZi64).unwrap();
    bencher.iter(|| {
        actual.clear();
        actual.resize_with(511, || ring.zero());
        f(
            &(0..256)
                .map(|j| {
                    (
                        (0..256).map(|k| hom.map(i * j as i64 * k)).collect::<Vec<_>>(),
                        None,
                        (0..256).map(|k| hom.map(i * j as i64 * k)).collect::<Vec<_>>(),
                        None,
                    )
                })
                .collect::<Vec<_>>()
                .iter()
                .map(|(lhs, lhs_prep, rhs, rhs_prep)| (&lhs[..], *lhs_prep, &rhs[..], *rhs_prep))
                .collect::<Vec<_>>(),
            &mut actual,
            ring,
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
    feanor_tracing::DelayedLogger::init_test();
    let ring = zn_64b::Zn64B::new(65537);
    let convolution = NTTConvolution::for_zn(ring);

    run_benchmark(ring, bencher, |values, dst, _| {
        convolution.compute_convolution_sum_impl(values, dst)
    });
}

#[bench]
fn bench_convolution_sum_default(bencher: &mut Bencher) {
    feanor_tracing::DelayedLogger::init_test();
    let ring = zn_64b::Zn64B::new(65537);
    let convolution = NTTConvolution::for_zn(ring);

    run_benchmark(ring, bencher, |values, dst, ring| {
        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            convolution.compute_convolution(lhs, *lhs_prep, rhs, *rhs_prep, dst, ring.get_ring());
        }
    });
}
