use std::alloc::{Allocator, Global};

use crate::algorithms::fft::FFTAlgorithm;
use crate::rings::float_complex::Complex64Base;
use crate::algorithms::fft::complex_fft::*;
use crate::algorithms::unity_root::{get_prim_root_of_unity, is_prim_root_of_unity};
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::homomorphism::*;
use crate::primitive_int::StaticRing;
use crate::seq::SwappableVectorViewMut;
use crate::rings::zn::*;
use crate::ring::*;
use crate::seq::VectorFn;

///
/// Implementation of the Cooley-Tukey FFT algorithm for power-of-three lengths.
/// 
#[stability::unstable(feature = "enable")]
pub struct CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A = Global>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    log3_n: usize,
    hom: H,
    twiddles: Vec<Vec<R_twiddle::Element>>,
    inv_twiddles: Vec<Vec<R_twiddle::Element>>,
    third_root_of_unity: R_twiddle::Element,
    main_root_of_unity: R_main::Element,
    allocator: A
}

const ZZ: StaticRing<i64> = StaticRing::RING;

#[inline(never)]
fn butterfly_loop<T, S, F>(log3_n: usize, data: &mut [T], step: usize, twiddles: &[S], butterfly: F)
    where F: Fn(&mut T, &mut T, &mut T, &S, &S) + Clone
{
    assert_eq!(ZZ.pow(3, log3_n) as usize, data.len());
    assert!(step < log3_n);
    assert_eq!(2 * ZZ.pow(3, step) as usize, twiddles.len());
    
    let stride = ZZ.pow(3, log3_n - step - 1) as usize;
    assert!(data.len() % (3 * stride) == 0);
    assert_eq!(twiddles.as_chunks::<2>().0.len(), data.chunks_mut(3 * stride).len());

    if stride == 1 {
        for ([twiddle1, twiddle2], butterfly_data) in twiddles.as_chunks::<2>().0.iter().zip(data.as_chunks_mut::<3>().0.iter_mut()) {
            let [a, b, c] = butterfly_data.each_mut();
            butterfly(a, b, c, twiddle1, twiddle2);
        }
    } else {
        for ([twiddle1, twiddle2], butterfly_data) in twiddles.as_chunks::<2>().0.iter().zip(data.chunks_mut(3 * stride)) {
            let (first, rest) = butterfly_data.split_at_mut(stride);
            let (second, third) = rest.split_at_mut(stride);
            for ((a, b), c) in first.iter_mut().zip(second.iter_mut()).zip(third.iter_mut()) {
                butterfly(a, b, c, twiddle1, twiddle2);
            }
        }
    }
}

fn threeadic_reverse(mut number: usize, log3_n: usize) -> usize {
    debug_assert!((number as i64) < ZZ.pow(3, log3_n));
    let mut result = 0;
    for _ in 0..log3_n {
        let (quo, rem) = (number / 3, number % 3);
        result = 3 * result + rem;
        number = quo;
    }
    assert_eq!(0, number);
    return result;
}

impl<R> CooleyTukeyRadix3FFT<R::Type, R::Type, Identity<R>>
    where R: RingStore,
        R::Type: DivisibilityRing
{
    ///
    /// Creates an [`CooleyTukeyRadix3FFT`] for a prime field, assuming it has a characteristic
    /// congruent to 1 modulo `3^lo32_n`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn for_zn(ring: R, log3_n: usize) -> Option<Self>
        where R::Type: ZnRing
    {
        let n = ZZ.pow(3, log3_n);
        let as_field = (&ring).as_field().ok().unwrap();
        let root_of_unity = as_field.get_ring().unwrap_element(get_prim_root_of_unity(&as_field, n as usize)?);
        return Some(Self::new_with_hom(ring.into_identity(), root_of_unity, log3_n));
    }
}

impl<R_main, R_twiddle, H> CooleyTukeyRadix3FFT<R_main, R_twiddle, H>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>
{
    ///
    /// Creates an [`CooleyTukeyRadix3FFT`] for the given rings, using the given root of unity.
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
    pub fn new_with_hom(hom: H, zeta: R_twiddle::Element, log3_n: usize) -> Self {
        let ring = hom.domain();
        let pow_zeta = |i: i64| if i < 0 {
            ring.invert(&ring.pow(ring.clone_el(&zeta), (-i).try_into().unwrap())).unwrap()
        } else {
            ring.pow(ring.clone_el(&zeta), i.try_into().unwrap())
        };
        let result = CooleyTukeyRadix3FFT::create(&hom, pow_zeta, log3_n, Global);
        return Self {
            allocator: result.allocator,
            inv_twiddles: result.inv_twiddles,
            log3_n: result.log3_n,
            main_root_of_unity: result.main_root_of_unity,
            third_root_of_unity: result.third_root_of_unity,
            twiddles: result.twiddles,
            hom: hom
        };
    }
}

impl<R_main, R_twiddle, H, A> CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    ///
    /// Most general way to create a [`CooleyTukeyRadix3FFT`].
    /// 
    /// The given closure should, on input `i`, return `z^i` for a primitive
    /// `3^log3_n`-th root of unity. The given Allocator + Sync + Send is used to copy the input
    /// data in cases where the input data layout is not optimal for the algorithm
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create<F>(hom: H, mut root_of_unity_pow: F, log3_n: usize, allocator: A) -> Self 
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let n = ZZ.pow(3, log3_n) as usize;
        assert!(hom.codomain().get_ring().is_approximate() || is_prim_root_of_unity(hom.codomain(), &hom.map(root_of_unity_pow(1)), n));
        
        return Self {
            main_root_of_unity: hom.map(root_of_unity_pow(1)),
            log3_n: log3_n, 
            twiddles: Self::create_twiddle_list(hom.domain(), log3_n, &mut root_of_unity_pow), 
            inv_twiddles: Self::create_inv_twiddle_list(hom.domain(), log3_n, &mut root_of_unity_pow),
            third_root_of_unity: root_of_unity_pow(2 * n as i64 / 3),
            hom: hom,
            allocator: allocator
        };
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
    pub fn change_ring<R_new: ?Sized + RingBase, H_new: Homomorphism<R_twiddle, R_new>>(self, new_hom: H_new) -> (CooleyTukeyRadix3FFT<R_new, R_twiddle, H_new, A>, H) {
        let ring = new_hom.codomain();
        let root_of_unity = if self.log3_n == 0 {
            new_hom.codomain().one()
        } else if self.log3_n == 1 {
            let root_of_unity = self.hom.domain().pow(self.hom.domain().clone_el(&self.third_root_of_unity), 2);
            debug_assert!(self.ring().eq_el(&self.hom.map_ref(&root_of_unity), self.root_of_unity(self.hom.codomain())));
            new_hom.map(root_of_unity)
        } else {
            let root_of_unity = &self.inv_twiddles[self.log3_n - 1][threeadic_reverse(1, self.log3_n - 1)];
            debug_assert!(self.ring().eq_el(&self.hom.map_ref(root_of_unity), self.root_of_unity(self.hom.codomain())));
            new_hom.map_ref(root_of_unity)
        };
        assert!(ring.is_commutative());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity(&ring, &root_of_unity, self.len()));

        return (
            CooleyTukeyRadix3FFT {
                twiddles: self.twiddles,
                inv_twiddles: self.inv_twiddles,
                main_root_of_unity: root_of_unity, 
                third_root_of_unity: self.third_root_of_unity,
                hom: new_hom, 
                log3_n: self.log3_n, 
                allocator: self.allocator
            },
            self.hom
        );
    }

    ///
    /// Returns the ring over which this object can compute FFTs.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn ring<'a>(&'a self) -> &'a <H as Homomorphism<R_twiddle, R_main>>::CodomainStore {
        self.hom.codomain()
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
    pub fn with_allocator<A_new: Allocator + Sync + Send>(self, allocator: A_new) -> CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A_new> {
        CooleyTukeyRadix3FFT {
            twiddles: self.twiddles,
            inv_twiddles: self.inv_twiddles,
            main_root_of_unity: self.main_root_of_unity, 
            third_root_of_unity: self.third_root_of_unity,
            hom: self.hom, 
            log3_n: self.log3_n, 
            allocator: allocator
        }
    }

    ///
    /// Returns a reference to the homomorphism that is used to map the stored twiddle
    /// factors into main ring, over which FFTs are computed.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn hom<'a>(&'a self) -> &'a H {
        &self.hom
    }

    fn create_twiddle_list<F>(ring: &H::DomainStore, log3_n: usize, mut pow_zeta: F) -> Vec<Vec<R_twiddle::Element>>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let n = ZZ.pow(3, log3_n);
        let third_root_of_unity = pow_zeta(-(n / 3));
        let mut result: Vec<_> = (0..log3_n).map(|_| Vec::new()).collect();
        for i in 0..log3_n {
            let current = &mut result[i];
            for j in 0..ZZ.pow(3, i) {
                let base_twiddle = pow_zeta(-(threeadic_reverse(j as usize, log3_n - 1) as i64));
                current.push(ring.clone_el(&base_twiddle));
                current.push(ring.pow(ring.mul_ref_snd(base_twiddle, &third_root_of_unity), 2));
            }
        }
        return result;
    }
    
    fn create_inv_twiddle_list<F>(ring: &H::DomainStore, log3_n: usize, mut pow_zeta: F) -> Vec<Vec<R_twiddle::Element>>
        where F: FnMut(i64) -> R_twiddle::Element
    {
        let mut result: Vec<_> = (0..log3_n).map(|_| Vec::new()).collect();
        for i in 0..log3_n {
            let current = &mut result[i];
            for j in 0..ZZ.pow(3, i) {
                let base_twiddle = pow_zeta(threeadic_reverse(j as usize, log3_n - 1) as i64);
                current.push(ring.clone_el(&base_twiddle));
                current.push(ring.pow(base_twiddle, 2));
            }
        }
        return result;
    }

    fn butterfly_step_main<const INV: bool>(&self, data: &mut [R_main::Element], step: usize) {
        let twiddles = if INV {
            &self.inv_twiddles[step]
        } else {
            &self.twiddles[step]
        };
        let third_root_of_unity = &self.third_root_of_unity;
        // let start = std::time::Instant::now();
        butterfly_loop(self.log3_n, data, step, twiddles, |x, y, z, twiddle1, twiddle2| if INV {
            <R_main as CooleyTukeyRadix3Butterfly<R_twiddle>>::inv_butterfly(&self.hom, x, y, z, &third_root_of_unity, twiddle1, twiddle2)
        } else {
            <R_main as CooleyTukeyRadix3Butterfly<R_twiddle>>::butterfly(&self.hom, x, y, z, &third_root_of_unity, twiddle1, twiddle2)
        });
        // let end = std::time::Instant::now();
        // BUTTERFLY_RADIX3_TIMES[step].fetch_add((end - start).as_micros() as usize, std::sync::atomic::Ordering::Relaxed);
    }

    fn fft_impl(&self, data: &mut [R_main::Element]) {
        for i in 0..data.len() {
            <R_main as CooleyTukeyRadix3Butterfly<R_twiddle>>::prepare_for_fft(self.hom.codomain().get_ring(), &mut data[i]);
        }
        for step in 0..self.log3_n {
            self.butterfly_step_main::<false>(data, step);
        }
    }

    fn inv_fft_impl(&self, data: &mut [R_main::Element]) {
        for i in 0..data.len() {
            <R_main as CooleyTukeyRadix3Butterfly<R_twiddle>>::prepare_for_inv_fft(self.hom.codomain().get_ring(), &mut data[i]);
        }
        for step in (0..self.log3_n).rev() {
            self.butterfly_step_main::<true>(data, step);
        }
        let n_inv = self.hom.domain().invert(&self.hom.domain().int_hom().map(self.len() as i32)).unwrap();
        for i in 0..data.len() {
            self.hom.mul_assign_ref_map(&mut data[i], &n_inv);
        }
    }
}

impl<R_main, R_twiddle, H, A> Clone for CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A> 
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main> + Clone,
        A: Allocator + Sync + Send + Clone
{
    fn clone(&self) -> Self {
        Self {
            hom: self.hom.clone(),
            inv_twiddles: self.inv_twiddles.iter().map(|list| list.iter().map(|x| self.hom.domain().clone_el(x)).collect()).collect(),
            twiddles: self.twiddles.iter().map(|list| list.iter().map(|x| self.hom.domain().clone_el(x)).collect()).collect(),
            main_root_of_unity: self.hom.codomain().clone_el(&self.main_root_of_unity),
            third_root_of_unity: self.hom.domain().clone_el(&self.third_root_of_unity),
            log3_n: self.log3_n,
            allocator: self.allocator.clone()
        }
    }
}

impl<R_main, R_twiddle, H, A> FFTAlgorithm<R_main> for CooleyTukeyRadix3FFT<R_main, R_twiddle, H, A>
    where R_main: ?Sized + RingBase,
        R_twiddle: ?Sized + RingBase + DivisibilityRing,
        H: Homomorphism<R_twiddle, R_main>,
        A: Allocator + Sync + Send
{
    fn len(&self) -> usize {
        if self.log3_n == 0 {
            return 1;
        }
        let result = self.twiddles[self.log3_n - 1].len() / 2 * 3;
        debug_assert_eq!(ZZ.pow(3, self.log3_n) as usize, result);
        return result;
    }

    fn unordered_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<R_main::Element>,
            S: RingStore<Type = R_main> + Copy
    {
        assert!(ring.get_ring() == self.hom.codomain().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        if let Some(data) = values.as_slice_mut() {
            self.fft_impl(data);
        } else {
            let mut data = Vec::with_capacity_in(self.len(), &self.allocator);
            data.extend(values.clone_ring_els(ring).iter());
            self.fft_impl(&mut data);
            for (i, x) in data.into_iter().enumerate() {
                *values.at_mut(i) = x;
            }
        }
    }

    fn unordered_inv_fft<V, S>(&self, mut values: V, ring: S)
        where V: SwappableVectorViewMut<R_main::Element>,
            S: RingStore<Type = R_main> + Copy
    {
        assert!(ring.get_ring() == self.hom.codomain().get_ring(), "unsupported ring");
        assert_eq!(self.len(), values.len());
        if let Some(data) = values.as_slice_mut() {
            self.inv_fft_impl(data);
        } else {
            let mut data = Vec::with_capacity_in(self.len(), &self.allocator);
            data.extend(values.clone_ring_els(ring).iter());
            self.inv_fft_impl(&mut data);
            for (i, x) in data.into_iter().enumerate() {
                *values.at_mut(i) = x;
            }
        }
    }

    fn root_of_unity<S>(&self, ring: S) -> &R_main::Element
        where S: RingStore<Type = R_main> + Copy
    {
        assert!(ring.get_ring() == self.hom.codomain().get_ring(), "unsupported ring");   
        &self.main_root_of_unity 
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        threeadic_reverse(i, self.log3_n)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        threeadic_reverse(i, self.log3_n)
    }
}

#[stability::unstable(feature = "enable")]
pub trait CooleyTukeyRadix3Butterfly<S: ?Sized + RingBase>: RingBase {

    ///
    /// Should compute `(a, b, c) := (a + t b + t^2 c, a + t z b + t^2 z^2 c, a + t z^2 b + t^2 z c)`.
    /// 
    /// Here `z` is a third root of unity (i.e. `z^2 + z + 1 = 0`) and `t` is the twiddle factor.
    /// The function should be given `z, t, t^2 z^2`.
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTukeyRadix3Butterfly::butterfly()`] or of [`CooleyTukeyRadix3Butterfly::prepare_for_fft()`].
    /// 
    fn butterfly<H: Homomorphism<S, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element, 
        c: &mut Self::Element, 
        z: &S::Element,
        t: &S::Element,
        t_sqr_z_sqr: &S::Element
    );
    
    ///
    /// Should compute `(a, b, c) := (a + b + c, t (a + z^2 b + z c), t^2 (a + z b + z^2 c))`.
    /// 
    /// Here `z` is a third root of unity (i.e. `z^2 + z + 1 = 0`) and `t` is the twiddle factor.
    /// The function should be given `z, t, t^2`.
    /// 
    /// It is guaranteed that the input elements are either outputs of
    /// [`CooleyTukeyRadix3Butterfly::inv_butterfly()`] or of [`CooleyTukeyRadix3Butterfly::prepare_for_inv_fft()`].
    /// 
    fn inv_butterfly<H: Homomorphism<S, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element,
        c: &mut Self::Element,
        z: &S::Element,
        t: &S::Element,
        t_sqr: &S::Element
    );

    ///
    /// Possibly pre-processes elements before the FFT starts. Here you can
    /// bring ring element into a certain form, and assume during [`CooleyTukeyRadix3Butterfly::butterfly()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    ///
    /// Possibly pre-processes elements before the inverse FFT starts. Here you can
    /// bring ring element into a certain form, and assume during [`CooleyTukeyRadix3Butterfly::inv_butterfly()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    fn prepare_for_inv_fft(&self, _value: &mut Self::Element) {}
}

impl<R: ?Sized + RingBase, S: ?Sized + RingBase> CooleyTukeyRadix3Butterfly<S> for R {

    default fn butterfly<H: Homomorphism<S, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element, 
        c: &mut Self::Element, 
        z: &S::Element,
        t: &S::Element,
        t_sqr_z_sqr: &S::Element
    ) {
        let ring = hom.codomain();
        hom.mul_assign_ref_map(b, t); // this is now `t b`
        hom.mul_assign_ref_map(c, t_sqr_z_sqr); // this is now `t^2 z^2 c`
        let b_ = hom.mul_ref_map(b, z); // this is now `t z b`
        let c_ = hom.mul_ref_map(c, z); // this is now `t^2 c z`
        let s1 = ring.add_ref(b, &c_); // this is now `t b + t^2 c`
        let s2 = ring.add_ref(&b_, c); // this is now `t z b + t^2 z^2 c`
        let s3 = ring.add_ref(&s1, &s2); // this is now `-(t z^2 b + t^2 z c)`
        *b = ring.add_ref_fst(a, s2); // this is now `a + t z b + t^2 z^2 c`
        *c = ring.sub_ref_fst(a, s3); // this is now `a + t z^2 b + t^2 z c`
        ring.add_assign(a, s1); // this is now `a + t b + t^2 c`
    }
    
    default fn inv_butterfly<H: Homomorphism<S, Self>>(
        hom: H, 
        a: &mut Self::Element, 
        b: &mut Self::Element,
        c: &mut Self::Element,
        z: &S::Element,
        t: &S::Element,
        t_sqr: &S::Element
    ) {
        let ring = hom.codomain();
        let b_ = hom.mul_ref_map(b, z); // this is now `z b`
        let s1 = ring.add_ref(b, c); // this is now `b + c`
        let s2 = ring.add_ref(&b_, &c); // this is now `z b + c`
        let s2_ = hom.mul_ref_snd_map(s2, z); // this is now `z^2 b + z c`
        let s3 = ring.add_ref(&s1, &s2_); // this is now `-(z b + z^2 c)`
        *b = ring.add_ref(a, &s2_); // this is now `a + z^2 b + z c`
        *c = ring.sub_ref(a, &s3); // this is now `a + z b + z^2 c`
        ring.add_assign(a, s1); // this is now `a + b + c`
        hom.mul_assign_ref_map(b, t); // this is now `t (a + z^2 b + z c)`
        hom.mul_assign_ref_map(c, t_sqr); // this is now `t^2 (a + z b + z^2 c`
    }

    ///
    /// Possibly pre-processes elements before the FFT starts. Here you can bring ring element 
    /// into a certain form, and assume during [`CooleyTukeyRadix3Butterfly::butterfly()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    default fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    ///
    /// Possibly pre-processes elements before the inverse FFT starts. Here you can bring ring element
    /// into a certain form, and assume during [`CooleyTukeyRadix3Butterfly::inv_butterfly()`]
    /// that the inputs are in this form.
    /// 
    #[inline(always)]
    default fn prepare_for_inv_fft(&self, _value: &mut Self::Element) {}
}

impl<H, A> FFTErrorEstimate for CooleyTukeyRadix3FFT<Complex64Base, Complex64Base, H, A> 
    where H: Homomorphism<Complex64Base, Complex64Base>,
        A: Allocator + Sync + Send
{
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        // the butterfly performs two multiplications with roots of unity, and then two additions
        let multiply_absolute_error = 2. * input_bound * root_of_unity_error() + input_bound * f64::EPSILON;
        let addition_absolute_error = 2. * input_bound * f64::EPSILON;
        let butterfly_absolute_error = multiply_absolute_error + addition_absolute_error;
        // the operator inf-norm of the FFT is its length
        return 2. * self.len() as f64 * butterfly_absolute_error + self.len() as f64 * input_error;
    }
}

#[cfg(test)]
use std::array::from_fn;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use crate::assert_el_eq;
#[cfg(test)]
use crate::rings::zn::zn_64::*;
#[cfg(test)]
use crate::rings::zn::zn_static::Fp;

#[test]
fn test_radix3_butterflies() {
    let log3_n = 3;
    let ring = Zn::new(109);
    let ring_fastmul = ZnFastmul::new(ring).unwrap();
    let int_hom = ring.int_hom();
    let i = |x| int_hom.map(x);
    let zeta = i(97);
    let zeta_inv = ring.invert(&zeta).unwrap();
    let fft = CooleyTukeyRadix3FFT::new_with_hom(ring.into_can_hom(ring_fastmul).ok().unwrap(), ring_fastmul.coerce(&ring, zeta), log3_n);

    const LEN: usize = 27;
    let data: [_; LEN] = from_fn(|j| i(j as i32));
    let expected_std_order = |step: usize, group_idx: usize, value_idx: usize| ring.sum(
        (0..ZZ.pow(3, step)).map(|k| ring.mul(
            ring.pow(zeta_inv, value_idx * (k * ZZ.pow(3, log3_n - step)) as usize),
            data[group_idx + (k * ZZ.pow(3, log3_n - step)) as usize]
        ))
    );
    let expected_threeadic_reverse = |step: usize| from_fn(|i| expected_std_order(
        step,
        i % ZZ.pow(3, log3_n - step) as usize,
        threeadic_reverse(i / ZZ.pow(3, log3_n - step) as usize, step)
    ));
    let begin = expected_threeadic_reverse(0);
    for (a, e) in data.iter().zip(begin.iter()) {
        assert_el_eq!(ring, a, e);
    }

    let mut actual = data;
    for i in 0..log3_n {
        fft.butterfly_step_main::<false>(&mut actual, i);
        let expected: [ZnEl; LEN] = expected_threeadic_reverse(i + 1);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_el_eq!(ring, a, e);
        }
    }
}

#[test]
fn test_radix3_inv_fft() {
    let log3_n = 3;
    let ring = Zn::new(109);
    let ring_fastmul = ZnFastmul::new(ring).unwrap();
    let zeta = ring.int_hom().map(97);
    let fft = CooleyTukeyRadix3FFT::new_with_hom(ring.into_can_hom(ring_fastmul).ok().unwrap(), ring_fastmul.coerce(&ring, zeta), log3_n);

    let data = (0..ZZ.pow(3, log3_n)).map(|x| ring.int_hom().map(x as i32)).collect::<Vec<_>>();
    let mut actual = data.clone();
    fft.unordered_fft(&mut actual, &ring);
    fft.unordered_inv_fft(&mut actual, &ring);

    for i in 0..data.len() {
        assert_el_eq!(&ring, &data[i], &actual[i]);
    }
}

#[test]
fn test_size_1_fft() {
    let ring = Fp::<17>::RING;
    let fft = CooleyTukeyRadix3FFT::for_zn(&ring, 0).unwrap().change_ring(ring.identity()).0;
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
pub mod generic_tests {
    use super::*;

    pub fn test_cooley_tuckey_radix3_butterfly<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(ring: R, base: S, edge_case_elements: I, test_zeta: &El<S>, test_twiddle: &El<S>)
        where R::Type: CanHomFrom<S::Type>,
            S::Type: DivisibilityRing
    {
        assert!(base.is_zero(&base.sum([base.one(), base.clone_el(&test_zeta), base.pow(base.clone_el(&test_zeta), 2)])));
        let test_inv_twiddle = base.invert(&test_twiddle).unwrap();
        let elements = edge_case_elements.collect::<Vec<_>>();
        let hom = ring.can_hom(&base).unwrap();

        for a in &elements {
            for b in &elements {
                for c in &elements {
                    
                    let [mut x, mut y, mut z] = [ring.clone_el(a), ring.clone_el(b), ring.clone_el(c)];
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_fft(ring.get_ring(), &mut x);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_fft(ring.get_ring(), &mut y);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_fft(ring.get_ring(), &mut z);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::butterfly(
                        &hom, 
                        &mut x, 
                        &mut y, 
                        &mut z, 
                        &test_zeta, 
                        &test_twiddle, 
                        &base.pow(base.mul_ref(&test_twiddle, &test_zeta), 2)
                    );
                    let mut t = hom.map_ref(&test_twiddle);
                    assert_el_eq!(ring, ring.add_ref_fst(a, ring.mul_ref_snd(ring.add_ref_fst(b, ring.mul_ref(c, &t)), &t)), &x);
                    ring.mul_assign(&mut t, hom.map_ref(&test_zeta));
                    assert_el_eq!(ring, ring.add_ref_fst(a, ring.mul_ref_snd(ring.add_ref_fst(b, ring.mul_ref(c, &t)), &t)), &y);
                    ring.mul_assign(&mut t, hom.map_ref(&test_zeta));
                    assert_el_eq!(ring, ring.add_ref_fst(a, ring.mul_ref_snd(ring.add_ref_fst(b, ring.mul_ref(c, &t)), &t)), &z);
                    
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_inv_fft(ring.get_ring(), &mut x);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_inv_fft(ring.get_ring(), &mut y);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::prepare_for_inv_fft(ring.get_ring(), &mut z);
                    <R::Type as CooleyTukeyRadix3Butterfly<S::Type>>::inv_butterfly(
                        &hom, 
                        &mut x, 
                        &mut y, 
                        &mut z, 
                        &test_zeta, 
                        &test_inv_twiddle, 
                        &base.pow(base.clone_el(&test_inv_twiddle), 2)
                    );
                    assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(a, 3), &x);
                    assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(b, 3), &y);
                    assert_el_eq!(ring, ring.int_hom().mul_ref_fst_map(c, 3), &z);
                }
            }
        }
    }
}

#[test]
fn test_butterfly() {
    let ring = Fp::<109>::RING;
    generic_tests::test_cooley_tuckey_radix3_butterfly(
        ring,
        ring,
        ring.elements().step_by(10),
        &63,
        &97,
    );
}