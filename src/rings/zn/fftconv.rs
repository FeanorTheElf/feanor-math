use core::f64;
use std::alloc::Allocator;
use std::alloc::Global;

use algorithms::convolution::ComputeConvolutionRing;
use algorithms::fft::complex_fft::FFTErrorEstimate;
use algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use append_only_vec::AppendOnlyVec;

use crate::delegate::DelegateRing;
use crate::primitive_int::StaticRingBase;
use crate::ring::*;
use crate::rings::float_complex::*;
use crate::rings::zn::*;
use crate::seq::VectorView;
use crate::algorithms::fft::FFTAlgorithm;

#[stability::unstable(feature = "enable")]
pub struct FFTConvZnBase<R: RingStore, A = Global>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    ring: R,
    allocator: A,
    fft_tables: AppendOnlyVec<CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>
}

#[stability::unstable(feature = "enable")]
pub type FFTConvZn<R, A = Global> = RingValue<FFTConvZnBase<R, A>>;

impl<R: RingStore, A> FFTConvZn<R, A>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(ring: R, allocator: A) -> Self {
        Self::from(FFTConvZnBase {
            ring: ring,
            allocator: allocator,
            fft_tables: AppendOnlyVec::new()
        })
    }
}

impl<R: RingStore, A> PartialEq for FFTConvZnBase<R, A>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring()
    }
}

impl<R: RingStore, A> DelegateRing for FFTConvZnBase<R, A>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    type Base = R::Type;
    type Element = El<R>;

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }

    fn get_delegate(&self) -> &Self::Base {
        self.ring.get_ring()
    }
}

impl<R: RingStore, A> PrincipalIdealRing for FFTConvZnBase<R, A>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        self.get_delegate().extended_ideal_gen(lhs, rhs)
    }

    fn cancel_common_factors(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        self.get_delegate().cancel_common_factors(lhs, rhs)
    }

    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.get_delegate().ideal_gen(lhs, rhs)
    }

    fn create_left_elimination_matrix(&self, a: &Self::Element, b: &Self::Element) -> ([Self::Element; 4], Self::Element) {
        self.get_delegate().create_left_elimination_matrix(a, b)
    }
}

impl<R: RingStore, S, A> CanHomFrom<S> for FFTConvZnBase<R, A>
    where R::Type: ZnRing + CanHomFrom<S> + CanHomFrom<StaticRingBase<i64>>,
        S: ?Sized + RingBase,
        A: Allocator
{
    type Homomorphism = <R::Type as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.get_delegate().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: <S as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.get_delegate().map_in(from, el, hom)
    }
}

impl<R: RingStore, A> ComputeConvolutionRing for FFTConvZnBase<R, A>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator
{
    fn compute_convolution<V1: VectorView<Self::Element>, V2: VectorView<Self::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [Self::Element]) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }
        let CC = Complex64::RING;
        let log2_size = StaticRing::<i64>::RING.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        debug_assert!(log2_size >= 1);

        while self.fft_tables.len() < log2_size {
            // TODO: there should be some operation `push_if_len()` on AppendOnlyVec, as long as it does not exist, we cannot do this here
            // properly without getting into locking
            let log2_n = self.fft_tables.len() + 1;
            let fft_table = CooleyTuckeyFFT::for_complex(CC, log2_n);
            if self.fft_tables.len() == log2_n - 1 {
                self.fft_tables.push(fft_table);
                assert!(self.fft_tables.len() == log2_n, "error due to concurrent operations on AppendOnlyVec");
            }
        }

        let fft_table = &self.fft_tables[log2_size - 1];
        let modulus = self.integer_ring().to_float_approx(<Self as ZnRing>::modulus(self));
        assert!(
            fft_table.expected_absolute_error(modulus * modulus, modulus * modulus * f64::EPSILON + fft_table.expected_absolute_error(modulus, 0.)) < 0.5, 
            "f64 does not have enough precision for computing a convolution modulo {}", 
            self.integer_ring().format(self.modulus())
        );

        let hom = CC.can_hom(self.integer_ring()).unwrap();
        let mut lhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        lhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in lhs.as_iter().enumerate() {
            lhs_data[i] = hom.map(self.smallest_lift(self.clone_el(c)));
        }

        let mut rhs_data = Vec::with_capacity_in(1 << log2_size, &self.allocator);
        rhs_data.resize(1 << log2_size, CC.zero());
        for (i, c) in rhs.as_iter().enumerate() {
            rhs_data[i] = hom.map(self.smallest_lift(self.clone_el(c)));
        }

        fft_table.unordered_fft(&mut lhs_data[..], CC.get_ring());
        fft_table.unordered_fft(&mut rhs_data[..], CC.get_ring());
        for i in 0..(1 << log2_size) {
            CC.mul_assign_ref(&mut lhs_data[i], &rhs_data[i]);
        }
        fft_table.unordered_inv_fft(&mut lhs_data[..], CC.get_ring());

        let hom = self.ring.can_hom(&StaticRing::<i64>::RING).unwrap();
        for i in 0..dst.len() {
            let x = CC.closest_gaussian_int(lhs_data[i]);
            debug_assert!(x.1 == 0);
            dst[i] = hom.map(x.0);
        }
    }
}

#[test]
fn test_ring_axioms() {
    let ring = FFTConvZn::new_with(zn_64::Zn::new(17), Global);
    crate::ring::generic_tests::test_ring_axioms(&ring, ring.elements());
}

#[test]
fn test_zn_ring_axioms() {
    let ring = FFTConvZn::new_with(zn_64::Zn::new(17), Global);
    crate::rings::zn::generic_tests::test_zn_axioms(&ring);
}

#[test]
fn test_convolution() {
    let ring = FFTConvZn::new_with(zn_64::Zn::new(17), Global);
    crate::algorithms::convolution::generic_tests::test_convolution(&ring, ring.elements());

    let ring = FFTConvZn::new_with(zn_64::Zn::new(17 * 17), Global);
    crate::algorithms::convolution::generic_tests::test_convolution(&ring, ring.elements());

    let ring = FFTConvZn::new_with(zn_64::Zn::new(17 * 257), Global);
    crate::algorithms::convolution::generic_tests::test_convolution(&ring, ring.elements());
}