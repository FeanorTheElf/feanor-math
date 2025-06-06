
use std::alloc::Allocator;

use crate::seq::subvector::SubvectorView;
use crate::ring::*;
use crate::homomorphism::*;
use crate::algorithms::fft::*;
use crate::algorithms::unity_root::is_prim_root_of_unity;
use crate::algorithms::fft::complex_fft::*;
use crate::rings::float_complex::*;
use crate::divisibility::DivisibilityRing;
use crate::algorithms::fft::radix3::CooleyTukeyRadix3FFT;
use crate::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;

/// 
/// A generic variant of the Cooley-Tukey FFT algorithm that can be used to compute the Fourier
/// transform of an array of length `n1 * n2` given Fourier transforms for length `n1` resp. `n2`.
/// 
#[stability::unstable(feature = "enable")]
pub struct GeneralCooleyTukeyFFT<R_main, R_twiddle, H, T1, T2> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>,
        T1: FFTAlgorithm<R_main>,
        T2: FFTAlgorithm<R_main>
{
    twiddle_factors: Vec<R_twiddle::Element>,
    inv_twiddle_factors: Vec<R_twiddle::Element>,
    left_table: T1,
    right_table: T2,
    root_of_unity: R_main::Element,
    root_of_unity_twiddle: R_twiddle::Element,
    hom: H
}

impl<R, T1, T2> GeneralCooleyTukeyFFT<R::Type, R::Type, Identity<R>, T1, T2> 
    where R: RingStore,
        T1: FFTAlgorithm<R::Type>,
        T2: FFTAlgorithm<R::Type>
{
    ///
    /// Creates a new [`GeneralCooleyTukeyFFT`] over the given ring of length `n`, based on FFTs
    /// of length `n1` and `n2`, where `n = n1 * n2`.
    /// 
    /// The closure `root_of_unity_pows` should, on input `i`, return `z^i` for the primitive `n`-th root of
    /// unity `z` satisfying `z^n1 == right_table.root_of_unity()` and `z^n2 - left_table.root_of_unity()`,
    /// where `n1` and `n2` are the lengths of `left_table` and `right_table`, respectively. 
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with_pows<F>(ring: R, root_of_unity_pows: F, left_table: T1, right_table: T2) -> Self
        where F: FnMut(i64) -> El<R>
    {
        Self::new_with_pows_with_hom(ring.into_identity(), root_of_unity_pows, left_table, right_table)
    }

    ///
    /// Creates a new [`GeneralCooleyTukeyFFT`] over the given ring of length `n`, based on FFTs
    /// of length `n1` and `n2`, where `n = n1 * n2`.
    /// 
    /// The given root of unity should be the primitive `n`-th root of unity satisfying
    /// `root_of_unity^n1 == right_table.root_of_unity()` and `root_of_unity^n2 - left_table.root_of_unity()`,
    /// where `n1` and `n2` are the lengths of `left_table` and `right_table`, respectively. 
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: R, root_of_unity: El<R>, left_table: T1, right_table: T2) -> Self {
        Self::new_with_hom(ring.into_identity(), root_of_unity, left_table, right_table)
    }
}

impl<R_main, R_twiddle, H, A1, A2> GeneralCooleyTukeyFFT<R_main, R_twiddle, H, CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A1>, CooleyTuckeyFFT<R_main, R_twiddle, H, A2>> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    ///
    /// Replaces the ring that this object can compute FFTs over, assuming that the current
    /// twiddle factors can be mapped into the new ring with the given homomorphism.
    /// 
    /// In particular, this function does not recompute twiddles, but uses a different
    /// homomorphism to map the current twiddles into a new ring. Hence, it is extremely
    /// cheap. 
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn change_ring<R_new: ?Sized + RingBase, H_new: Clone + Homomorphism<R_twiddle, R_new>>(self, new_hom: H_new) -> (GeneralCooleyTukeyFFT<R_new, R_twiddle, H_new, CooleyTukeyRadix3FFT<R_new, R_twiddle, H_new, A1>, CooleyTuckeyFFT<R_new, R_twiddle, H_new, A2>>, H) {
        let ring = new_hom.codomain();
        let root_of_unity = new_hom.map_ref(&self.root_of_unity_twiddle);
        assert!(ring.is_commutative());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity(&ring, &root_of_unity, self.len()));

        return (
            GeneralCooleyTukeyFFT {
                twiddle_factors: self.twiddle_factors,
                left_table: self.left_table.change_ring(new_hom.clone()).0,
                right_table: self.right_table.change_ring(new_hom.clone()).0,
                inv_twiddle_factors: self.inv_twiddle_factors,
                root_of_unity: root_of_unity,
                root_of_unity_twiddle: self.root_of_unity_twiddle,
                hom: new_hom,
            },
            self.hom
        );
    }
}

impl<R_main, R_twiddle, H, T1, T2> GeneralCooleyTukeyFFT<R_main, R_twiddle, H, T1, T2> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>,
        T1: FFTAlgorithm<R_main>,
        T2: FFTAlgorithm<R_main>
{
    ///
    /// Creates a new [`GeneralCooleyTukeyFFT`] over the given ring of length `n`, based on FFTs
    /// of length `n1` and `n2`, where `n = n1 * n2`.
    /// 
    /// The closure `root_of_unity_pows` should, on input `i`, return `z^i` for the primitive `n`-th root of
    /// unity `z` satisfying `z^n1 == right_table.root_of_unity()` and `z^n2 - left_table.root_of_unity()`,
    /// where `n1` and `n2` are the lengths of `left_table` and `right_table`, respectively. 
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with_pows_with_hom<F>(hom: H, root_of_unity_pows: F, left_table: T1, right_table: T2) -> Self
        where F: FnMut(i64) -> R_twiddle::Element
    {
        Self::create(hom, root_of_unity_pows, left_table, right_table)
    }

    ///
    /// Most general way to create a [`GeneralCooleyTukeyFFT`]. Currently equivalent to [`GeneralCooleyTukeyFFT::new_with_pows_with_hom()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create<F>(hom: H, mut root_of_unity_pows: F, left_table: T1, right_table: T2) -> Self
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let ring = hom.codomain();

        assert!(ring.is_commutative());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity(ring, &hom.map(root_of_unity_pows(1)), left_table.len() * right_table.len()));
        assert!(ring.get_ring().is_approximate() || ring.eq_el(&hom.map(root_of_unity_pows(right_table.len().try_into().unwrap())), left_table.root_of_unity(ring)));
        assert!(ring.get_ring().is_approximate() || ring.eq_el(&hom.map(root_of_unity_pows(left_table.len().try_into().unwrap())), right_table.root_of_unity(ring)));

        let root_of_unity = root_of_unity_pows(1);
        let inv_twiddle_factors = Self::create_twiddle_factors(|i| root_of_unity_pows(-i), &left_table, &right_table);
        let twiddle_factors = Self::create_twiddle_factors(root_of_unity_pows, &left_table, &right_table);

        GeneralCooleyTukeyFFT {
            twiddle_factors: twiddle_factors,
            inv_twiddle_factors: inv_twiddle_factors,
            left_table: left_table, 
            right_table: right_table,
            root_of_unity: hom.map_ref(&root_of_unity),
            root_of_unity_twiddle: root_of_unity,
            hom: hom
        }
    }

    ///
    /// Returns the length-`n1` FFT used by this object to compute length-`n` FFTs.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn left_fft_table(&self) -> &T1 {
        &self.left_table
    }
    
    ///
    /// Returns the length-`n2` FFT used by this object to compute length-`n` FFTs.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn right_fft_table(&self) -> &T2 {
        &self.right_table
    }
    
    ///
    /// Returns the homomorphism used to map twiddle factors into the main
    /// ring during the computation of FFTs.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn hom<'a>(&'a self) -> &'a H {
        &self.hom
    }

    ///
    /// Creates a new [`GeneralCooleyTukeyFFT`] over the given ring of length `n`, based on FFTs
    /// of length `n1` and `n2`, where `n = n1 * n2`.
    /// 
    /// The given root of unity should be the primitive `n`-th root of unity satisfying
    /// `root_of_unity^n1 == right_table.root_of_unity()` and `root_of_unity^n2 - left_table.root_of_unity()`,
    /// where `n1` and `n2` are the lengths of `left_table` and `right_table`, respectively. 
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with_hom(hom: H, root_of_unity: R_twiddle::Element, left_table: T1, right_table: T2) -> Self {
        let len = left_table.len() * right_table.len();
        let root_of_unity_pows = |i: i64| if i >= 0 {
            hom.domain().pow(hom.domain().clone_el(&root_of_unity), i.try_into().unwrap())
        } else {
            let len_i64: i64 = len.try_into().unwrap();
            hom.domain().pow(hom.domain().clone_el(&root_of_unity), (len_i64 + (i % len_i64)).try_into().unwrap())
        };
        let result = GeneralCooleyTukeyFFT::create(&hom, root_of_unity_pows, left_table, right_table);
        GeneralCooleyTukeyFFT {
            twiddle_factors: result.twiddle_factors,
            inv_twiddle_factors: result.inv_twiddle_factors,
            left_table: result.left_table, 
            right_table: result.right_table,
            root_of_unity: result.root_of_unity,
            root_of_unity_twiddle: result.root_of_unity_twiddle,
            hom: hom
        }
    }

    fn create_twiddle_factors<F>(mut root_of_unity_pows: F, left_table: &T1, right_table: &T2) -> Vec<R_twiddle::Element>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        (0..(left_table.len() * right_table.len())).map(|i| {
            let ri: i64 = (i % right_table.len()).try_into().unwrap();
            let li = i / right_table.len();
            return root_of_unity_pows(TryInto::<i64>::try_into(left_table.unordered_fft_permutation(li)).unwrap() * ri);
        }).collect::<Vec<_>>()
    }
    
    ///
    /// Returns the ring over which this object can compute FFTs.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom.codomain()
    }
}

impl<R_main, R_twiddle, H, T1, T2> PartialEq for GeneralCooleyTukeyFFT<R_main, R_twiddle, H, T1, T2> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>,
        T1: FFTAlgorithm<R_main> + PartialEq,
        T2: FFTAlgorithm<R_main> + PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.left_table == other.left_table &&
            self.right_table == other.right_table &&
            self.ring().eq_el(self.root_of_unity(self.ring()), other.root_of_unity(self.ring()))
    }
}

impl<R_main, R_twiddle, H, T1, T2> FFTAlgorithm<R_main> for GeneralCooleyTukeyFFT<R_main, R_twiddle, H, T1, T2> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>,
        T1: FFTAlgorithm<R_main>,
        T2: FFTAlgorithm<R_main>
{
    fn len(&self) -> usize {
        self.left_table.len() * self.right_table.len()
    }

    fn root_of_unity<S: RingStore<Type = R_main> + Copy>(&self, ring: S) -> &R_main::Element {
        assert!(self.ring().get_ring() == ring.get_ring(), "unsupported ring");
        &self.root_of_unity
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(self.ring().get_ring() == ring.get_ring(), "unsupported ring");
        if self.left_table.len() > 1 {
            for i in 0..self.right_table.len() {
                let mut v = SubvectorView::new(&mut values).restrict(i..).step_by_view(self.right_table.len());
                self.left_table.unordered_fft(&mut v, ring);
            }
            for i in 0..self.len() {
                self.hom.mul_assign_ref_map(values.at_mut(i), self.inv_twiddle_factors.at(i));
            }
        }
        for i in 0..self.left_table.len() {
            let mut v = SubvectorView::new(&mut values).restrict((i * self.right_table.len())..((i + 1) * self.right_table.len()));
            self.right_table.unordered_fft(&mut v, ring);
        }
    }

    fn unordered_inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(self.ring().get_ring() == ring.get_ring(), "unsupported ring");
        for i in 0..self.left_table.len() {
            let mut v = SubvectorView::new(&mut values).restrict((i * self.right_table.len())..((i + 1) * self.right_table.len()));
            self.right_table.unordered_inv_fft(&mut v, ring);
        }
        if self.left_table.len() > 1 {
            for i in 0..self.len() {
                self.hom.mul_assign_ref_map(values.at_mut(i), self.twiddle_factors.at(i));
                debug_assert!(self.ring().get_ring().is_approximate() || self.hom.domain().is_one(&self.hom.domain().mul_ref(self.twiddle_factors.at(i), self.inv_twiddle_factors.at(i))));
            }
            for i in 0..self.right_table.len() {
                let mut v = SubvectorView::new(&mut values).restrict(i..).step_by_view(self.right_table.len());
                self.left_table.unordered_inv_fft(&mut v, ring);
            }
        }
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        assert!(i < self.len());
        self.left_table.unordered_fft_permutation(i / self.right_table.len()) + self.left_table.len() * self.right_table.unordered_fft_permutation(i % self.right_table.len())
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        assert!(i < self.len());
        self.left_table.unordered_fft_permutation_inv(i % self.left_table.len()) * self.right_table.len() + self.right_table.unordered_fft_permutation_inv(i / self.left_table.len())
    }
}

impl<H, T1, T2> FFTErrorEstimate for GeneralCooleyTukeyFFT<Complex64Base, Complex64Base, H, T1, T2> 
    where H: Homomorphism<Complex64Base, Complex64Base>,
        T1: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        T2: FFTAlgorithm<Complex64Base> + FFTErrorEstimate
{
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        let error_after_first_fft = self.left_table.expected_absolute_error(input_bound, input_error);
        let new_input_bound = self.left_table.len() as f64 * input_bound;
        let error_after_twiddling = error_after_first_fft + new_input_bound * (root_of_unity_error() + f64::EPSILON);
        return self.right_table.expected_absolute_error(new_input_bound, error_after_twiddling);
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::{Zn, Fp};
#[cfg(test)]
use crate::algorithms::unity_root::*;
#[cfg(test)]
use crate::rings::zn::zn_64;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::algorithms::fft::bluestein::BluesteinFFT;

#[test]
fn test_fft_basic() {
    let ring = Zn::<97>::RING;
    let z = ring.int_hom().map(39);
    let fft = GeneralCooleyTukeyFFT::new(ring, ring.pow(z, 16), 
        CooleyTuckeyFFT::new(ring, ring.pow(z, 48), 1),
        BluesteinFFT::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3, Global),
    );
    let mut values = [1, 0, 0, 1, 0, 1];
    let expected = [3, 62, 63, 96, 37, 36];
    let mut permuted_expected = [0; 6];
    for i in 0..6 {
        permuted_expected[i] = expected[fft.unordered_fft_permutation(i)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, permuted_expected);
}

#[test]
fn test_fft_long() {
    let ring = Fp::<97>::RING;
    let z = ring.int_hom().map(39);
    let fft = GeneralCooleyTukeyFFT::new(ring, ring.pow(z, 4), 
        CooleyTuckeyFFT::new(ring, ring.pow(z, 12), 3),
        BluesteinFFT::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3, Global),
    );
    let mut values = [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 2, 0, 1, 2, 3, 4];
    let expected = [26, 0, 75, 47, 41, 31, 28, 62, 39, 93, 53, 27, 0, 54, 74, 61, 65, 81, 63, 38, 53, 94, 89, 91];
    let mut permuted_expected = [0; 24];
    for i in 0..24 {
        permuted_expected[i] = expected[fft.unordered_fft_permutation(i)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, permuted_expected);
}

#[test]
fn test_fft_unordered() {
    let ring = Fp::<1409>::RING;
    let z = get_prim_root_of_unity(ring, 64 * 11).unwrap();
    let fft = GeneralCooleyTukeyFFT::new(
        ring, 
        ring.pow(z, 4),
        CooleyTuckeyFFT::new(ring, ring.pow(z, 44), 4),
        BluesteinFFT::new(ring, ring.pow(z, 32), ring.pow(z, 22), 11, 5, Global),
    );
    const LEN: usize = 16 * 11;
    let mut values = [0; LEN];
    for i in 0..LEN {
        values[i] = ring.int_hom().map(i as i32);
    }
    let original = values;

    fft.unordered_fft(&mut values, ring);

    let mut ordered_fft = [0; LEN];
    for i in 0..LEN {
        ordered_fft[fft.unordered_fft_permutation(i)] = values[i];
    }

    fft.unordered_inv_fft(&mut values, ring);
    assert_eq!(values, original);

    fft.inv_fft(&mut ordered_fft, ring);
    assert_eq!(ordered_fft, original);
}


#[test]
fn test_unordered_fft_permutation_inv() {
    let ring = Fp::<1409>::RING;
    let z = get_prim_root_of_unity(ring, 64 * 11).unwrap();
    let fft = GeneralCooleyTukeyFFT::new(
        ring, 
        ring.pow(z, 4),
        CooleyTuckeyFFT::new(ring, ring.pow(z, 44), 4),
        BluesteinFFT::new(ring, ring.pow(z, 32), ring.pow(z, 22), 11, 5, Global),
    );
    for i in 0..(16 * 11) {
        assert_eq!(fft.unordered_fft_permutation_inv(fft.unordered_fft_permutation(i)), i);
        assert_eq!(fft.unordered_fft_permutation(fft.unordered_fft_permutation_inv(i)), i);
    }
    
    let fft = GeneralCooleyTukeyFFT::new(
        ring, 
        ring.pow(z, 4),
        BluesteinFFT::new(ring, ring.pow(z, 32), ring.pow(z, 22), 11, 5, Global),
        CooleyTuckeyFFT::new(ring, ring.pow(z, 44), 4),
    );
    for i in 0..(16 * 11) {
        assert_eq!(fft.unordered_fft_permutation_inv(fft.unordered_fft_permutation(i)), i);
        assert_eq!(fft.unordered_fft_permutation(fft.unordered_fft_permutation_inv(i)), i);
    }
}

#[test]
fn test_inv_fft() {
    let ring = Fp::<97>::RING;
    let z = ring.int_hom().map(39);
    let fft = GeneralCooleyTukeyFFT::new(ring, ring.pow(z, 16), 
        CooleyTuckeyFFT::new(ring, ring.pow(z, 16 * 3), 1),
        BluesteinFFT::new(ring, ring.pow(z, 16), ring.pow(z, 12), 3, 3, Global),
    );
    let mut values = [3, 62, 63, 96, 37, 36];
    let expected = [1, 0, 0, 1, 0, 1];

    fft.inv_fft(&mut values, ring);
    assert_eq!(values, expected);
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for (p, log2_n) in [(5, 3), (53, 5), (101, 8), (503, 10)] {
        let fft = GeneralCooleyTukeyFFT::new_with_pows(
            CC,
            |i| CC.root_of_unity(i, TryInto::<i64>::try_into(p).unwrap() << log2_n), 
            BluesteinFFT::for_complex(CC, p, Global), 
            CooleyTuckeyFFT::for_complex(CC, log2_n)
        );
        let mut array = (0..(p << log2_n)).map(|i| CC.root_of_unity(i.try_into().unwrap(), TryInto::<i64>::try_into(p).unwrap() << log2_n)).collect::<Vec<_>>();
        fft.fft(&mut array, CC);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}

#[cfg(test)]
const BENCH_N1: usize = 31;
#[cfg(test)]
const BENCH_N2: usize = 601;

#[bench]
fn bench_factor_fft(bencher: &mut test::Bencher) {
    let ring = zn_64::Zn::new(1602564097);
    let fastmul_ring = zn_64::ZnFastmul::new(ring).unwrap();
    let embedding = ring.can_hom(&fastmul_ring).unwrap();
    let ring_as_field = ring.as_field().ok().unwrap();
    let root_of_unity = fastmul_ring.coerce(&ring, ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity(&ring_as_field, 2 * 31 * 601).unwrap()));
    let fastmul_ring_as_field = fastmul_ring.as_field().ok().unwrap();
    let fft = GeneralCooleyTukeyFFT::new_with_hom(
        embedding.clone(), 
        fastmul_ring.pow(root_of_unity, 2),
        BluesteinFFT::new_with_hom(embedding.clone(), fastmul_ring.pow(root_of_unity, BENCH_N1), fastmul_ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(&fastmul_ring_as_field, 11).unwrap()), BENCH_N2, 11, Global),
        BluesteinFFT::new_with_hom(embedding, fastmul_ring.pow(root_of_unity, BENCH_N2), fastmul_ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(&fastmul_ring_as_field, 6).unwrap()), BENCH_N1, 6, Global),
    );
    let data = (0..(BENCH_N1 * BENCH_N2)).map(|i| ring.int_hom().map(i as i32)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(BENCH_N1 * BENCH_N2);
    bencher.iter(|| {
        copy.clear();
        copy.extend(data.iter().map(|x| ring.clone_el(x)));
        fft.unordered_fft(&mut copy[..], &ring);
        fft.unordered_inv_fft(&mut copy[..], &ring);
        assert_el_eq!(ring, copy[0], data[0]);
    });
}
