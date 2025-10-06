use std::alloc::{Allocator, Global};
use std::ops::Range;
use std::fmt::Debug;

use crate::algorithms::unity_root::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::rings::zn::*;
use crate::seq::SwappableVectorViewMut;
use crate::ring::*;
use crate::seq::VectorViewMut;
use crate::homomorphism::*;
use crate::algorithms::fft::*;
use crate::rings::float_complex::*;
use super::complex_fft::*;

///
/// An optimized implementation of the Cooley-Tukey FFT algorithm, to compute
/// the Fourier transform of an array with power-of-two length.
/// 
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// // this ring has a 256-th primitive root of unity
/// let ring = zn_64::Zn::new(257);
/// let fft_table = CooleyTuckeyFFT::for_zn(ring, 8).unwrap();
/// let mut data = [ring.one()].into_iter().chain((0..255).map(|_| ring.zero())).collect::<Vec<_>>();
/// fft_table.unordered_fft(&mut data, &ring);
/// assert_el_eq!(ring, ring.one(), data[0]);
/// assert_el_eq!(ring, ring.one(), data[1]);
/// ```
/// 
/// # Convention
/// 
/// This implementation does not follows the standard convention for the mathematical
/// DFT, by performing the standard/forward FFT with the inverse root of unity `z^-1`.
/// In other words, the forward FFT computes
/// ```text
///   (a_0, ..., a_(n - 1)) -> (sum_j a_j z^(-ij))_i
/// ```
/// as demonstrated by
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::algorithms::fft::cooley_tuckey::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::divisibility::*;
/// // this ring has a 4-th primitive root of unity
/// let ring = zn_64::Zn::new(5);
/// let root_of_unity = ring.int_hom().map(2);
/// let fft_table = CooleyTuckeyFFT::new(ring, root_of_unity, 2);
/// let mut data = [ring.one(), ring.one(), ring.zero(), ring.zero()];
/// fft_table.fft(&mut data, ring);
/// let inv_root_of_unity = ring.invert(&root_of_unity).unwrap();
/// assert_el_eq!(ring, ring.add(ring.one(), inv_root_of_unity), data[1]);
/// ```
/// 
/// # On optimizations
/// 
/// I tried my best to make this as fast as possible in general, with special focus
/// on the Number-theoretic transform case. I did not implement the following
/// optimizations, for the following reasons:
///  - Larger butterflies: This would improve data locality, but decrease twiddle
///    locality (or increase arithmetic operation count). Since I focused mainly on
///    the `Z/nZ` case, where the twiddles are larger than the ring elements (since they
///    have additional data to speed up multiplications), this is not sensible.
///  - The same reasoning applies to a SplitRadix approach, which only actually decreases
///    the total number of operations if multiplication-by-`i` is free.
/// 
pub struct CooleyTuckeyFFT<R_main, R_twiddle, H, A = Global> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    hom: H,
    root_of_unity: R_main::Element,
    log2_n: usize,
    // stores the powers of `root_of_unity^-1` in special bitreversed order
    root_of_unity_list: Vec<Vec<R_twiddle::Element>>,
    // stores the powers of `root_of_unity` in special bitreversed order
    inv_root_of_unity_list: Vec<Vec<R_twiddle::Element>>,
    allocator: A,
    two_inv: R_twiddle::Element,
    n_inv: R_twiddle::Element
}

///
/// Assumes that `index` has only the least significant `bits` bits set.
/// Then computes the value that results from reversing the least significant `bits`
/// bits.
/// 
pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits().checked_shr(usize::BITS - bits as u32).unwrap_or(0)
}

#[inline(never)]
fn butterfly_loop<T, S, F>(log2_n: usize, data: &mut [T], butterfly_range: Range<usize>, stride_range: Range<usize>, log2_step: usize, twiddles: &[S], butterfly: F)
    where F: Fn(&mut T, &mut T, &S) + Clone
{
    assert_eq!(1 << log2_n, data.len());
    assert!(log2_step < log2_n);

    // the coefficients of a group of inputs have this distance to each other
    let stride = 1 << (log2_n - log2_step - 1);
    assert!(stride_range.start <= stride_range.end);
    assert!(stride_range.end <= stride);

    // how many butterflies we compute within each group
    assert!(butterfly_range.start <= butterfly_range.end);
    assert!(butterfly_range.end <= (1 << log2_step));
    assert!(butterfly_range.end <= twiddles.len());
    
    let current_data = &mut data[(stride_range.start + butterfly_range.start * 2 * stride)..];
    let stride_range_len = stride_range.end - stride_range.start;
    
    if stride == 1 && stride_range_len == 1 {
        for (twiddle, butterfly_data) in twiddles[butterfly_range].iter().zip(current_data.as_chunks_mut::<2>().0.iter_mut()) {
            let [a, b] = butterfly_data.each_mut();
            butterfly(a, b, &twiddle);
        }
    } else if stride_range_len >= 1 {
        for (twiddle, butterfly_data) in twiddles[butterfly_range].iter().zip(current_data.chunks_mut(2 * stride)) {
            let (first, second) = butterfly_data[..(stride + stride_range_len)].split_at_mut(stride);
            let (first_chunks, first_rem) = first[..stride_range_len].as_chunks_mut::<4>();
            let (second_chunks, second_rem) = second.as_chunks_mut::<4>();
            for (a, b) in first_chunks.iter_mut().zip(second_chunks.iter_mut()) {
                butterfly(&mut a[0], &mut b[0], &twiddle);
                butterfly(&mut a[1], &mut b[1], &twiddle);
                butterfly(&mut a[2], &mut b[2], &twiddle);
                butterfly(&mut a[3], &mut b[3], &twiddle);
            }
            for (a, b) in first_rem.iter_mut().zip(second_rem.iter_mut()) {
                butterfly(a, b, &twiddle);
            }
        }
    }
}

impl<R_main, H> CooleyTuckeyFFT<R_main, Complex64Base, H, Global> 
    where R_main: ?Sized + RingBase,
        H: Homomorphism<Complex64Base, R_main>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the complex field, using the given homomorphism
    /// to connect the ring implementation for twiddles with the main ring implementation.
    /// 
    /// This function is mainly provided for parity with other rings, since in the complex case
    /// it currently does not make much sense to use a different homomorphism than the identity.
    /// Hence, it is simpler to use [`CooleyTuckeyFFT::for_complex()`].
    /// 
    pub fn for_complex_with_hom(hom: H, log2_n: usize) -> Self {
        let CC = *hom.domain().get_ring();
        Self::new_with_pows_with_hom(hom, |i| CC.root_of_unity(i, 1 << log2_n), log2_n)
    }
}

impl<R> CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<R>, Global> 
    where R: RingStore<Type = Complex64Base>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the complex field.
    /// 
    pub fn for_complex(ring: R, log2_n: usize) -> Self {
        Self::for_complex_with_hom(ring.into_identity(), log2_n)
    }
}

impl<R> CooleyTuckeyFFT<R::Type, R::Type, Identity<R>, Global> 
    where R: RingStore,
        R::Type: DivisibilityRing
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given ring, using the given root of unity. 
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_hom(ring.into_identity(), root_of_unity, log2_n)
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given ring, using the passed function to
    /// provide the necessary roots of unity.
    /// 
    /// Concretely, `root_of_unity_pow(i)` should return `z^i`, where `z` is a `2^log2_n`-th
    /// primitive root of unity.
    /// 
    pub fn new_with_pows<F>(ring: R, root_of_unity_pow: F, log2_n: usize) -> Self 
        where F: FnMut(i64) -> El<R>
    {
        Self::new_with_pows_with_hom(ring.into_identity(), root_of_unity_pow, log2_n)
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for a prime field, assuming it has a characteristic
    /// congruent to 1 modulo `2^log2_n`.
    /// 
    pub fn for_zn(ring: R, log2_n: usize) -> Option<Self>
        where R::Type: ZnRing
    {
        Self::for_zn_with_hom(ring.into_identity(), log2_n)
    }
}

impl<R_main, R_twiddle, H> CooleyTuckeyFFT<R_main, R_twiddle, H, Global> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given rings, using the given root of unity.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    /// Do not use this for approximate rings, as computing the powers of `root_of_unity`
    /// will incur avoidable precision loss.
    /// 
    pub fn new_with_hom(hom: H, root_of_unity: R_twiddle::Element, log2_n: usize) -> Self {
        let ring = hom.domain();
        let root_of_unity_pow = |i: i64| if i >= 0 {
            ring.pow(ring.clone_el(&root_of_unity), i as usize)
        } else {
            ring.invert(&ring.pow(ring.clone_el(&root_of_unity), (-i) as usize)).unwrap()
        };
        let result = CooleyTuckeyFFT::create(&hom, root_of_unity_pow, log2_n, Global);
        
        return CooleyTuckeyFFT {
            root_of_unity_list: result.root_of_unity_list,
            inv_root_of_unity_list: result.inv_root_of_unity_list,
            two_inv: result.two_inv,
            n_inv: result.n_inv,
            root_of_unity: result.root_of_unity, 
            log2_n: result.log2_n, 
            allocator: result.allocator,
            hom: hom, 
        };
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given rings, using the given function to create
    /// the necessary powers of roots of unity.
    /// 
    /// Concretely, `root_of_unity_pow(i)` should return `z^i`, where `z` is a `2^log2_n`-th
    /// primitive root of unity.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    pub fn new_with_pows_with_hom<F>(hom: H, root_of_unity_pow: F, log2_n: usize) -> Self 
        where F: FnMut(i64) -> R_twiddle::Element
    {
        Self::create(hom, root_of_unity_pow, log2_n, Global)
    }

    ///
    /// Creates an [`CooleyTuckeyFFT`] for the given prime fields, assuming they have
    /// a characteristic congruent to 1 modulo `2^log2_n`.
    /// 
    /// Instead of a ring, this function takes a homomorphism `R -> S`. Twiddle factors that are
    /// precomputed will be stored as elements of `R`, while the main FFT computations will be 
    /// performed in `S`. This allows both implicit ring conversions, and using patterns like 
    /// [`zn_64::ZnFastmul`] to precompute some data for better performance.
    /// 
    pub fn for_zn_with_hom(hom: H, log2_n: usize) -> Option<Self>
        where R_twiddle: ZnRing
    {
        let ring_as_field = hom.domain().as_field().ok().unwrap();
        let root_of_unity = ring_as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(&ring_as_field, log2_n)?);
        drop(ring_as_field);
        Some(Self::new_with_hom(hom, root_of_unity, log2_n))
    }
}

impl<R_main, R_twiddle, H, A> PartialEq for CooleyTuckeyFFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.log2_n == other.log2_n &&
            self.ring().eq_el(self.root_of_unity(self.ring()), other.root_of_unity(self.ring()))
    }
}

impl<R_main, R_twiddle, H, A> Debug for CooleyTuckeyFFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CooleyTuckeyFFT")
            .field("ring", &self.ring().get_ring())
            .field("n", &(1 << self.log2_n))
            .field("root_of_unity", &self.ring().formatted_el(&self.root_of_unity(self.ring())))
            .finish()
    }
}

impl<R_main, R_twiddle, H, A> Clone for CooleyTuckeyFFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Sync + Send + Clone
{
    fn clone(&self) -> Self {
        Self {
            two_inv: self.hom.domain().clone_el(&self.two_inv),
            n_inv: self.hom.domain().clone_el(&self.n_inv),
            hom: self.hom.clone(),
            inv_root_of_unity_list: self.inv_root_of_unity_list.iter().map(|list| list.iter().map(|x| self.hom.domain().clone_el(x)).collect()).collect(),
            root_of_unity_list: self.root_of_unity_list.iter().map(|list| list.iter().map(|x| self.hom.domain().clone_el(x)).collect()).collect(),
            root_of_unity: self.hom.codomain().clone_el(&self.root_of_unity),
            log2_n: self.log2_n,
            allocator: self.allocator.clone()
        }
    }
}

///
/// A helper trait that defines the Cooley-Tukey butterfly operation.
/// It is default-implemented for all rings, but for increase FFT performance, some rings
/// might wish to provide a specialization.
/// 
/// # Why not a subtrait of [`Homomorphism`]?
/// 
/// With the current design, indeed making this a subtrait of [`Homomorphism`] would
/// indeed be the conceptually most fitting choice. It would allow specializing on
/// the twiddle ring, the main ring and the inclusion. 
/// 
/// Unfortunately, there is a technical issue: With the current `min_specialization`, 
/// we can only specialize on concrete type. If this is a subtrait of [`Homomorphism`], this 
/// means we can only specialize on, say, `CanHom<ZnFastmul, Zn>`, which then does not give a
/// specialization for `CanHom<&ZnFastmul, Zn>` - in other words, we would specialize on
/// the [`RingStore`], and not on the [`RingBase`] as we should. Hence, we'll keep this
/// suboptimal design until full specialization works.
/// 
pub trait CooleyTuckeyButterfly<S>: RingBase
    where S: ?Sized + RingBase
{
    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + twiddle * values[i2], values[i1] - twiddle * values[i2])`.
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTuckeyButterfly::butterfly()`] or of [`CooleyTuckeyButterfly::prepare_for_fft()`].
    /// 
    /// Deprecated in favor of [`CooleyTuckeyButterfly::butterfly_new()`].
    /// 
    #[deprecated]
    fn butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: H, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);

    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + values[i2], (values[i1] - values[i2]) * twiddle)`
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTuckeyButterfly::inv_butterfly()`] or of [`CooleyTuckeyButterfly::prepare_for_inv_fft()`].
    /// 
    /// Deprecated in favor of [`CooleyTuckeyButterfly::inv_butterfly_new()`].
    /// 
    #[deprecated]
    fn inv_butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: H, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);
    
    ///
    /// Should compute `(x, y) := (x + twiddle * y, x - twiddle * y)`.
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTuckeyButterfly::butterfly_new()`] or of [`CooleyTuckeyButterfly::prepare_for_fft()`].
    /// 
    fn butterfly_new<H: Homomorphism<S, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &S::Element);
    
    ///
    /// Should compute `(x, y) := (x + y, (x - y) * twiddle)`
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTuckeyButterfly::inv_butterfly_new()`] or of [`CooleyTuckeyButterfly::prepare_for_inv_fft()`].
    /// 
    fn inv_butterfly_new<H: Homomorphism<S, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &S::Element);

    ///
    /// Possibly pre-processes elements before the FFT starts. Here you can
    /// bring ring element into a certain form, and assume during [`CooleyTuckeyButterfly::butterfly_new()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    ///
    /// Possibly pre-processes elements before the inverse FFT starts. Here you can
    /// bring ring element into a certain form, and assume during [`CooleyTuckeyButterfly::inv_butterfly_new()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    fn prepare_for_inv_fft(&self, _value: &mut Self::Element) {}
}

#[allow(deprecated)]
impl<R, S> CooleyTuckeyButterfly<S> for R
    where S: ?Sized + RingBase, R: ?Sized + RingBase
{
    #[inline(always)]
    default fn butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: H, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        hom.mul_assign_ref_map(values.at_mut(i2), twiddle);
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
    }

    #[inline(always)]
    #[allow(deprecated)]
    default fn butterfly_new<H: Homomorphism<S, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &S::Element) {
        let mut values = [hom.codomain().clone_el(x), hom.codomain().clone_el(y)];
        <Self as CooleyTuckeyButterfly<S>>::butterfly(hom.codomain().get_ring(), &hom, &mut values, twiddle, 0, 1);
        [*x, *y] = values;
    }

    #[inline(always)]
    default fn inv_butterfly<V: VectorViewMut<Self::Element>, H: Homomorphism<S, Self>>(&self, hom: H, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
        hom.mul_assign_ref_map(values.at_mut(i2), twiddle);
    }
    
    #[inline(always)]
    #[allow(deprecated)]
    default fn inv_butterfly_new<H: Homomorphism<S, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &S::Element) {
        let mut values = [hom.codomain().clone_el(x), hom.codomain().clone_el(y)];
        <Self as CooleyTuckeyButterfly<S>>::inv_butterfly(hom.codomain().get_ring(), &hom, &mut values, twiddle, 0, 1);
        [*x, *y] = values;
    }

    #[inline(always)]
    default fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    #[inline(always)]
    default fn prepare_for_inv_fft(&self, _value: &mut Self::Element) {}
}

impl<R_main, R_twiddle, H, A> CooleyTuckeyFFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    ///
    /// Most general way to create a [`CooleyTuckeyFFT`].
    /// 
    /// This is currently the same as [`CooleyTuckeyFFT::new_with_pows_with_hom()`], except
    /// that it additionally accepts an Allocator + Sync + Send, which is used to copy the input data in
    /// cases where the input data layout is not optimal for the algorithm.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create<F>(hom: H, mut root_of_unity_pow: F, log2_n: usize, allocator: A) -> Self 
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let ring = hom.domain();
        assert!(ring.is_commutative());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity_pow2(&hom.codomain(), &hom.map(root_of_unity_pow(1)), log2_n));

        let root_of_unity_list = Self::create_root_of_unity_list(|i| root_of_unity_pow(-i), log2_n);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(|i| root_of_unity_pow(i), log2_n);
        let root_of_unity = root_of_unity_pow(1);
        
        let store_twiddle_ring = root_of_unity_list.len();
        CooleyTuckeyFFT {
            root_of_unity_list: root_of_unity_list.into_iter().take(store_twiddle_ring).collect(),
            inv_root_of_unity_list: inv_root_of_unity_list.into_iter().take(store_twiddle_ring).collect(),
            two_inv: hom.domain().invert(&hom.domain().int_hom().map(2)).unwrap(),
            n_inv: hom.domain().invert(&hom.domain().int_hom().map(1 << log2_n)).unwrap(),
            root_of_unity: hom.map(root_of_unity), 
            hom, 
            log2_n, 
            allocator
        }
    }

    ///
    /// Replaces the ring that this object can compute FFTs over, assuming that the current
    /// twiddle factors can be mapped into the new ring with the given homomorphism.
    /// 
    /// In particular, this function does not recompute twiddles, but uses a different
    /// homomorphism to map the current twiddles into a new ring. Hence, it is extremely
    /// cheap. 
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn change_ring<R_new: ?Sized + RingBase, H_new: Homomorphism<R_twiddle, R_new>>(self, new_hom: H_new) -> (CooleyTuckeyFFT<R_new, R_twiddle, H_new, A>, H) {
        let ring = new_hom.codomain();
        let root_of_unity = if self.log2_n == 0 {
            new_hom.codomain().one()
        } else {
            new_hom.map_ref(&self.inv_root_of_unity_list[self.log2_n - 1][bitreverse(1, self.log2_n - 1)])
        };
        assert!(ring.is_commutative());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity, self.log2_n));

        return (
            CooleyTuckeyFFT {
                root_of_unity_list: self.root_of_unity_list,
                inv_root_of_unity_list: self.inv_root_of_unity_list,
                two_inv: self.two_inv,
                n_inv: self.n_inv,
                root_of_unity: root_of_unity, 
                hom: new_hom, 
                log2_n: self.log2_n, 
                allocator: self.allocator
            },
            self.hom
        );
    }

    fn create_root_of_unity_list<F>(mut root_of_unity_pow: F, log2_n: usize) -> Vec<Vec<R_twiddle::Element>>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let mut twiddles: Vec<Vec<R_twiddle::Element>> = (0..log2_n).map(|_| Vec::new()).collect();
        for log2_step in 0..log2_n {
            let butterfly_count = 1 << log2_step;
            for i in 0..butterfly_count {
                twiddles[log2_step].push(root_of_unity_pow(bitreverse(i, log2_n - 1) as i64));
            }
        }
        return twiddles;
    }

    ///
    /// Returns the ring over which this object can compute FFTs.
    /// 
    pub fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom.codomain()
    }

    /// 
    /// Computes the main butterfly step, either forward or backward (without division by two).
    /// 
    /// The forward butterfly is
    /// ```text
    ///   (a, b) -> (a + twiddle * b, a - twiddle * b)
    /// ```
    /// The backward butterfly is
    /// ```text
    ///   (u, v) -> (u + v, twiddle * (u - v))
    /// ```
    /// 
    /// The `#[inline(never)]` here is absolutely important for performance!
    /// No idea why...
    /// 
    #[inline(never)]
    fn butterfly_step_main<const INV: bool, const IS_PREPARED: bool>(&self, data: &mut [R_main::Element], butterfly_range: Range<usize>, stride_range: Range<usize>, log2_step: usize) {
        let twiddles = if INV {
            &self.inv_root_of_unity_list[log2_step]
        } else {
            &self.root_of_unity_list[log2_step]
        };
        // let start = std::time::Instant::now();
        let butterfly = |a: &mut _, b: &mut _, twiddle: &_| {
            if INV {
                if !IS_PREPARED {
                    <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_inv_fft(self.ring().get_ring(), a);
                    <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_inv_fft(self.ring().get_ring(), b);
                }
                <R_main as CooleyTuckeyButterfly<R_twiddle>>::inv_butterfly_new(&self.hom, a, b, twiddle);
            } else {
                if !IS_PREPARED {
                    <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_fft(self.ring().get_ring(), a);
                    <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_fft(self.ring().get_ring(), b);
                }
                <R_main as CooleyTuckeyButterfly<R_twiddle>>::butterfly_new(&self.hom, a, b, twiddle);
            }
        };
        butterfly_loop(self.log2_n, data, butterfly_range, stride_range, log2_step, twiddles, butterfly);
        // let end = std::time::Instant::now();
        // BUTTERFLY_TIMES[log2_step].fetch_add((end - start).as_micros() as usize, std::sync::atomic::Ordering::Relaxed);
    }
    
    ///
    /// The definitions are
    /// ```text
    ///   u = a/2 + twiddle * b/2,
    ///   v = a/2 - twiddle * b/2
    /// ```
    /// 
    #[inline(never)]
    fn butterfly_ub_from_ab(&self, data: &mut [R_main::Element], butterfly_range: Range<usize>, stride_range: Range<usize>, log2_step: usize) {
        butterfly_loop(self.log2_n, data, butterfly_range, stride_range, log2_step, &self.root_of_unity_list[log2_step], |a, b, twiddle| {
            *a = self.hom.mul_ref_snd_map(
                self.ring().add_ref_fst(a, self.hom.mul_ref_map(b, twiddle)),
                &self.two_inv
            );
        });
    }

    ///
    /// The definitions are
    /// ```text
    ///   u = a/2 + twiddle * b/2,
    ///   v = a/2 - twiddle * b/2
    /// ```
    /// 
    #[inline(never)]
    fn butterfly_uv_from_ub(&self, data: &mut [R_main::Element], butterfly_range: Range<usize>, stride_range: Range<usize>, log2_step: usize) {
        butterfly_loop(self.log2_n, data, butterfly_range, stride_range, log2_step, &self.root_of_unity_list[log2_step], |a, b, twiddle| {
            *b = self.ring().sub_ref_fst(a, self.hom.mul_ref_map(b, twiddle));
        });
    }

    ///
    /// The definitions are
    /// ```text
    ///   u = a/2 + twiddle * b/2,
    ///   v = a/2 - twiddle * b/2
    /// ```
    /// 
    #[inline(never)]
    fn butterfly_ab_from_ub(&self, data: &mut [R_main::Element], butterfly_range: Range<usize>, stride_range: Range<usize>, log2_step: usize) {
        butterfly_loop(self.log2_n, data, butterfly_range, stride_range, log2_step, &self.root_of_unity_list[log2_step], |a, b, twiddle| {
            *a = self.ring().add_ref(a, a);
            self.ring().sub_assign(a, self.hom.mul_ref_map(b, twiddle));
        });
    }

    ///
    /// Returns a reference to the allocator currently used for temporary allocations by this FFT.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    ///
    /// Replaces the allocator used for temporary allocations by this FFT.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn with_allocator<A_new: Allocator + Sync + Send>(self, allocator: A_new) -> CooleyTuckeyFFT<R_main, R_twiddle, H, A_new> {
        CooleyTuckeyFFT {
            root_of_unity_list: self.root_of_unity_list,
            inv_root_of_unity_list: self.inv_root_of_unity_list,
            two_inv: self.two_inv,
            n_inv: self.n_inv,
            root_of_unity: self.root_of_unity, 
            hom: self.hom, 
            log2_n: self.log2_n, 
            allocator: allocator
        }
    }

    ///
    /// Returns a reference to the homomorphism that is used to map the stored twiddle
    /// factors into main ring, over which FFTs are computed.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn hom(&self) -> &H {
        &self.hom
    }

    ///
    /// Computes the unordered, truncated FFT.
    /// 
    /// The truncated FFT is the standard DFT, applied to a list for which only the first
    /// `nonzero_entries` entries are nonzero, and the (bitreversed) result truncated to
    /// length `nonzero_entries`.
    /// 
    /// Therefore, this function is equivalent to the following pseudocode
    /// ```text
    /// data[nonzero_entries..] = 0;
    /// unordered_fft(data);
    /// data[nonzero_entries] = unspecified;
    /// ```
    /// 
    /// It can be inverted using [`CooleyTuckey::unordered_truncated_fft_inv()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn unordered_truncated_fft(&self, data: &mut [R_main::Element], nonzero_entries: usize) {
        assert_eq!(self.len(), data.len());
        assert!(nonzero_entries > self.len() / 2);
        assert!(nonzero_entries <= self.len());
        for i in nonzero_entries..self.len() {
            debug_assert!(self.ring().get_ring().is_approximate() || self.ring().is_zero(&data[i]));
        }

        for i in 0..data.len() {
            <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_fft(self.ring().get_ring(), &mut data[i]);
        }
        for log2_step in 0..self.log2_n {
            let stride = 1 << (self.log2_n - log2_step - 1);
            let butterfly_count = nonzero_entries.div_ceil(2 * stride);
            self.butterfly_step_main::<false, true>(data, 0..butterfly_count, 0..stride, log2_step);
        }
    }
    
    ///
    /// Computes the inverse of the unordered, truncated FFT.
    /// 
    /// The truncated FFT is the standard DFT, applied to a list for which only the first
    /// `nonzero_entries` entries are nonzero, and the (bitreversed) result truncated to
    /// length `nonzero_entries`. Therefore, this function computes a list of `nonzero_entries`
    /// many values, followed by zeros, whose DFT agrees with the input on the first `nonzero_entries`
    /// many elements.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn unordered_truncated_fft_inv(&self, data: &mut [R_main::Element], nonzero_entries: usize) {   
        assert_eq!(self.len(), data.len());
        assert!(nonzero_entries > self.len() / 2);
        assert!(nonzero_entries <= self.len());

        for i in 0..data.len() {
            <R_main as CooleyTuckeyButterfly<R_twiddle>>::prepare_for_inv_fft(self.ring().get_ring(), &mut data[i]);
        }
        for log2_step in (0..self.log2_n).rev() {
            let stride = 1 << (self.log2_n - log2_step - 1);
            let current_block = nonzero_entries / (2 * stride);
            self.butterfly_step_main::<true, true>(data, 0..current_block, 0..stride, log2_step);
        }
        if nonzero_entries < (1 << self.log2_n) {
            for i in nonzero_entries..(1 << self.log2_n) {
                data[i] = self.ring().zero();
            }
            for log2_step in 0..self.log2_n {
                let stride = 1 << (self.log2_n - log2_step - 1);
                let current_block = nonzero_entries / (2 * stride);
                let known_area = nonzero_entries % (2 * stride);
                if known_area >= stride {
                    self.butterfly_uv_from_ub(data, current_block..(current_block + 1), (known_area - stride)..stride, log2_step);
                } else {
                    self.butterfly_ub_from_ab(data, current_block..(current_block + 1), known_area..stride, log2_step);
                }
            }
            for log2_step in (0..self.log2_n).rev() {
                let stride = 1 << (self.log2_n - log2_step - 1);
                let current_block = nonzero_entries / (2 * stride);
                let known_area = nonzero_entries % (2 * stride);
                if known_area >= stride {
                    self.butterfly_step_main::<true, false>(data, current_block..(current_block + 1), 0..stride, log2_step);
                } else {
                    self.butterfly_ab_from_ub(data, current_block..(current_block + 1), 0..stride, log2_step);
                }
            }
        }
        for i in 0..(1 << self.log2_n) {
            self.hom.mul_assign_ref_map(&mut data[i], &self.n_inv);
        }
    }
    
    ///
    /// Permutes the given list of length `n` according to `values[bitreverse(i, log2(n))] = values[i]`.
    /// This is exactly the permutation that is implicitly applied by [`CooleyTuckeyFFT::unordered_fft()`].
    /// 
    pub fn bitreverse_permute_inplace<V, T>(&self, mut values: V) 
        where V: SwappableVectorViewMut<T>
    {
        assert!(values.len() == 1 << self.log2_n);
        for i in 0..(1 << self.log2_n) {
            if bitreverse(i, self.log2_n) < i {
                values.swap(i, bitreverse(i, self.log2_n));
            }
        }
    }
}

impl<R_main, R_twiddle, H, A> FFTAlgorithm<R_main> for CooleyTuckeyFFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    fn len(&self) -> usize {
        1 << self.log2_n
    }

    fn root_of_unity<S: Copy + RingStore<Type = R_main>>(&self, ring: S) -> &R_main::Element {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        &self.root_of_unity
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        self.unordered_fft(&mut values, ring);
        self.bitreverse_permute_inplace(&mut values);
    }

    fn inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        self.bitreverse_permute_inplace(&mut values);
        self.unordered_inv_fft(&mut values, ring);
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        if let Some(data) = values.as_slice_mut() {
            self.unordered_truncated_fft(data, 1 << self.log2_n);
        } else {
            let mut data = Vec::with_capacity_in(1 << self.log2_n, self.allocator());
            data.extend(values.clone_ring_els(ring).iter());
            self.unordered_truncated_fft(&mut data, 1 << self.log2_n);
            for (i, x) in data.into_iter().enumerate() {
                *values.at_mut(i) = x;
            }
        }
    }
    
    fn unordered_inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<<R_main as RingBase>::Element>,
            S: RingStore<Type = R_main> + Copy 
    {
        assert!(ring.get_ring() == self.ring().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        if let Some(data) = values.as_slice_mut() {
            self.unordered_truncated_fft_inv(data, 1 << self.log2_n);
        } else {
            let mut data = Vec::with_capacity_in(1 << self.log2_n, self.allocator());
            data.extend(values.clone_ring_els(ring).iter());
            self.unordered_truncated_fft_inv(&mut data, 1 << self.log2_n);
            for (i, x) in data.into_iter().enumerate() {
                *values.at_mut(i) = x;
            }
        }
    }
}

impl<H, A> FFTErrorEstimate for CooleyTuckeyFFT<Complex64Base, Complex64Base, H, A> 
    where H: Homomorphism<Complex64Base, Complex64Base>,
        A: Allocator + Sync + Send
{
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        // the butterfly performs a multiplication with a root of unity, and an addition
        let multiply_absolute_error = input_bound * root_of_unity_error() + input_bound * f64::EPSILON;
        let addition_absolute_error = input_bound * f64::EPSILON;
        let butterfly_absolute_error = multiply_absolute_error + addition_absolute_error;
        // the operator inf-norm of the FFT is its length
        return 2. * self.len() as f64 * butterfly_absolute_error + self.len() as f64 * input_error;
    }
}

#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;
#[cfg(test)]
use crate::rings::zn::zn_big;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::field::*;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_bitreverse_fft_inplace_basic() {
    let ring = Fp::<5>::RING;
    let z = ring.int_hom().map(2);
    let fft = CooleyTuckeyFFT::new(ring, ring.div(&1, &z), 2);
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace_advanced() {
    let ring = Fp::<17>::RING;
    let z = ring.int_hom().map(3);
    let fft = CooleyTuckeyFFT::new(ring, z, 4);
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.unordered_fft(&mut values, ring);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_unordered_fft_permutation() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(&ring, 4).unwrap();
    let mut values = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut expected = [0; 16];
    for i in 0..16 {
        let power_of_zeta = ring.pow(*fft.root_of_unity(&ring), 16 - fft.unordered_fft_permutation(i));
        expected[i] = ring.add(power_of_zeta, ring.pow(power_of_zeta, 4));
    }
    fft.unordered_fft(&mut values, ring);
    assert_eq!(expected, values);
}

#[test]
fn test_bitreverse_inv_fft_inplace() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(&ring, 4).unwrap();
    let values: [u64; 16] = [1, 2, 3, 2, 1, 0, 17 - 1, 17 - 2, 17 - 1, 0, 1, 2, 3, 4, 5, 6];
    let mut work = values;
    fft.unordered_fft(&mut work, ring);
    fft.unordered_inv_fft(&mut work, ring);
    assert_eq!(&work, &values);
}

#[test]
fn test_truncated_fft() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::new(ring, 2, 3);

    let data = [2, 3, 0, 1, 1, 0, 0, 0];
    let mut complete_fft = data;
    fft.unordered_fft(&mut complete_fft, ring);
    for k in 5..=8 {
        println!("{}", k);
        let mut truncated_fft = data;
        fft.unordered_truncated_fft(&mut truncated_fft, k);
        assert_eq!(&complete_fft[..k], &truncated_fft[..k]);

        fft.unordered_truncated_fft_inv(&mut truncated_fft, k);
        assert_eq!(data, truncated_fft);
    }
}

#[test]
fn test_for_zn() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));

    let ring = Fp::<97>::RING;
    let fft = CooleyTuckeyFFT::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
}

#[cfg(test)]
fn run_fft_bench_round<R, S, H>(fft: &CooleyTuckeyFFT<S, R, H>, data: &Vec<S::Element>, copy: &mut Vec<S::Element>)
    where R: ZnRing, S: ZnRing, H: Homomorphism<R, S>
{
    copy.clear();
    copy.extend(data.iter().map(|x| fft.ring().clone_el(x)));
    fft.unordered_fft(&mut copy[..], &fft.ring());
    fft.unordered_inv_fft(&mut copy[..], &fft.ring());
    assert_el_eq!(fft.ring(), copy[0], data[0]);
}

#[cfg(test)]
const BENCH_SIZE_LOG2: usize = 13;

#[bench]
fn bench_fft_zn_big(bencher: &mut test::Bencher) {
    let ring = zn_big::Zn::new(StaticRing::<i128>::RING, 1073872897);
    let fft = CooleyTuckeyFFT::for_zn(&ring, BENCH_SIZE_LOG2).unwrap();
    let data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    bencher.iter(|| {
        run_fft_bench_round(&fft, &data, &mut copy)
    });
}

#[bench]
fn bench_fft_zn_64(bencher: &mut test::Bencher) {
    let ring = zn_64::Zn::new(1073872897);
    let fft = CooleyTuckeyFFT::for_zn(&ring, BENCH_SIZE_LOG2).unwrap();
    let data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    bencher.iter(|| {
        run_fft_bench_round(&fft, &data, &mut copy)
    });
}

#[bench]
fn bench_fft_zn_64_fastmul(bencher: &mut test::Bencher) {
    let ring = zn_64::Zn::new(1073872897);
    let fastmul_ring = zn_64::ZnFastmul::new(ring).unwrap();
    let fft = CooleyTuckeyFFT::for_zn_with_hom(ring.into_can_hom(fastmul_ring).ok().unwrap(), BENCH_SIZE_LOG2).unwrap();
    let data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    bencher.iter(|| {
        run_fft_bench_round(&fft, &data, &mut copy)
    });
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for log2_n in [4, 7, 11, 15] {
        let fft = CooleyTuckeyFFT::new_with_pows(CC, |x| CC.root_of_unity(x, 1 << log2_n), log2_n);
        let mut array = (0..(1 << log2_n)).map(|i|  CC.root_of_unity(i.try_into().unwrap(), 1 << log2_n)).collect::<Vec<_>>();
        fft.fft(&mut array, CC);
        let err = fft.expected_absolute_error(1., 0.);
        assert!(CC.is_absolute_approx_eq(array[0], CC.zero(), err));
        assert!(CC.is_absolute_approx_eq(array[1], CC.from_f64(fft.len() as f64), err));
        for i in 2..fft.len() {
            assert!(CC.is_absolute_approx_eq(array[i], CC.zero(), err));
        }
    }
}

#[test]
fn test_size_1_fft() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTuckeyFFT::for_zn(&ring, 0).unwrap().change_ring(ring.identity()).0;
    let values: [u64; 1] = [3];
    let mut work = values;
    fft.unordered_fft(&mut work, ring);
    assert_eq!(&work, &values);
    fft.unordered_inv_fft(&mut work, ring);
    assert_eq!(&work, &values);
    assert_eq!(0, fft.unordered_fft_permutation(0));
    assert_eq!(0, fft.unordered_fft_permutation_inv(0));
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_cooley_tuckey_butterfly<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(ring: R, base: S, edge_case_elements: I, test_twiddle: &El<S>)
    where R::Type: CanHomFrom<S::Type>,
        S::Type: DivisibilityRing
{
    let test_inv_twiddle = base.invert(&test_twiddle).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();
    let hom = ring.can_hom(&base).unwrap();

    for a in &elements {
        for b in &elements {

            let [mut x, mut y] = [ring.clone_el(a), ring.clone_el(b)];
            <R::Type as CooleyTuckeyButterfly<S::Type>>::butterfly_new(&hom, &mut x, &mut y, &test_twiddle);
            assert_el_eq!(ring, ring.add_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &x);
            assert_el_eq!(ring, ring.sub_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &y);

            <R::Type as CooleyTuckeyButterfly<S::Type>>::inv_butterfly_new(&hom, &mut x, &mut y, &test_inv_twiddle);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(a, 2), &x);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(b, 2), &y);

            let [mut x, mut y] = [ring.clone_el(a), ring.clone_el(b)];
            <R::Type as CooleyTuckeyButterfly<S::Type>>::inv_butterfly_new(&hom, &mut x, &mut y, &test_twiddle);
            assert_el_eq!(ring, ring.add_ref(a, b), &x);
            assert_el_eq!(ring, ring.mul(ring.sub_ref(a, b), hom.map_ref(test_twiddle)), &y);

            <R::Type as CooleyTuckeyButterfly<S::Type>>::butterfly_new(&hom, &mut x, &mut y, &test_inv_twiddle);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(a, 2), &x);
            assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(b, 2), &y);
        }
    }
}

#[test]
fn test_butterfly() {
    generic_test_cooley_tuckey_butterfly(zn_static::F17, zn_static::F17, zn_static::F17.elements(), &get_prim_root_of_unity_pow2(zn_static::F17, 4).unwrap());
}