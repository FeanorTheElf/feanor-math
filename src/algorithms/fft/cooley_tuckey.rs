use crate::algorithms::unity_root::*;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::primitive_int::*;
use crate::mempool::{MemoryProvider, AllocatingMemoryProvider};
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::vector::SwappableVectorViewMut;
use crate::ring::*;
use crate::vector::VectorViewMut;
use crate::algorithms::fft::*;
use crate::algorithms;
use crate::rings::float_complex::*;
use super::complex_fft::*;

pub struct FFTTableCooleyTuckey<R, M: MemoryProvider<El<R>> = AllocatingMemoryProvider> 
    where R: RingStore
{
    ring: R,
    root_of_unity: El<R>,
    log2_n: usize,
    // stores the powers of root_of_unity in special bitreversed order
    root_of_unity_list: M::Object,
    // stores the powers of inv_root_of_unity in special bitreversed order
    inv_root_of_unity_list: M::Object
}

pub fn bitreverse(index: usize, bits: usize) -> usize {
    index.reverse_bits().checked_shr(usize::BITS - bits as u32).unwrap_or(0)
}

impl<R> FFTTableCooleyTuckey<R>
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    pub fn new(ring: R, root_of_unity: El<R>, log2_n: usize) -> Self {
        Self::new_with_mem(ring, root_of_unity, log2_n, &AllocatingMemoryProvider)
    }
    
    pub fn for_zn(ring: R, log2_n: usize) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        Self::for_zn_with_mem(ring, log2_n, &AllocatingMemoryProvider)
    }
}

impl<R, M: MemoryProvider<El<R>>> FFTTableCooleyTuckey<R, M>
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    pub fn new_with_mem(ring: R, root_of_unity: El<R>, log2_n: usize, memory_provider: &M) -> Self {
        let mut root_of_unity_pow = |i: i64| if i >= 0 {
            ring.pow(ring.clone_el(&root_of_unity), i as usize)
        } else {
            ring.invert(&ring.pow(ring.clone_el(&root_of_unity), (-i) as usize)).unwrap()
        };
        // cannot call new_with_mem_and_pows() because of borrowing conflict
        assert!(ring.is_commutative());
        assert!(!ring.get_ring().is_approximate());
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        let root_of_unity_list = Self::create_root_of_unity_list(&ring, &mut root_of_unity_pow, log2_n, memory_provider);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(&ring, |i| root_of_unity_pow(-i), log2_n, memory_provider);
        let root_of_unity = root_of_unity_pow(1);
        FFTTableCooleyTuckey { ring, root_of_unity, log2_n, root_of_unity_list, inv_root_of_unity_list }
    }

    pub fn new_with_mem_and_pows<F>(ring: R, mut root_of_unity_pow: F, log2_n: usize, memory_provider: &M) -> Self 
        where F: FnMut(i64) -> El<R>
    {
        assert!(ring.is_commutative());
        assert!(log2_n > 0);
        assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &root_of_unity_pow(1), log2_n));
        let root_of_unity_list = Self::create_root_of_unity_list(&ring, &mut root_of_unity_pow, log2_n, memory_provider);
        let inv_root_of_unity_list = Self::create_root_of_unity_list(&ring, |i| root_of_unity_pow(-i), log2_n, memory_provider);
        let root_of_unity = root_of_unity_pow(1);
        FFTTableCooleyTuckey { ring, root_of_unity, log2_n, root_of_unity_list, inv_root_of_unity_list }
    }

    fn create_root_of_unity_list<F>(ring: &R, mut root_of_unity_pow: F, log2_n: usize, memory_provider: &M) -> M::Object
        where F: FnMut(i64) -> El<R>
    {
        // in fact, we could choose this to have only length `(1 << log2_n) - 1`, but a power of two length is probably faster
        let mut root_of_unity_list = memory_provider.get_new_init(1 << log2_n, |_| ring.zero());
        let mut index = 0;
        for s in 0..log2_n {
            let m = 1 << s;
            let log2_group_size = log2_n - s;
            for i_bitreverse in (0..(1 << log2_group_size)).step_by(2) {
                let current_twiddle = root_of_unity_pow(m * bitreverse(i_bitreverse, log2_group_size) as i64);
                root_of_unity_list[index] = current_twiddle;
                index += 1;
            }
        }
        assert_eq!(index, (1 << log2_n) - 1);
        return root_of_unity_list;
    }

    ///
    /// Returns `inv_root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    #[inline(always)]
    fn inv_root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.inv_root_of_unity_list[(1 << self.log2_n) - (1 << (self.log2_n - exp_2)) + (bitreverse_exp / 2)];
        return result;
    }

    ///
    /// Returns `root_of_unity^(2^exp_2 * bitreverse(bitreverse_exp, log2_n - exp_2))`.
    /// 
    #[inline(always)]
    fn root_of_unity_pow(&self, exp_2: usize, bitreverse_exp: usize) -> &El<R> {
        let result = &self.root_of_unity_list[(1 << self.log2_n) - (1 << (self.log2_n - exp_2)) + (bitreverse_exp / 2)];
        return result;
    }

    pub fn len(&self) -> usize {
        1 << self.log2_n
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn for_zn_with_mem(ring: R, log2_n: usize, memory_provider: &M) -> Option<Self>
        where R: ZnRingStore,
            R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanonicalHom<StaticRingBase<i64>>
    {
        let root_of_unity = algorithms::unity_root::get_prim_root_of_unity_pow2(&ring, log2_n)?;
        Some(Self::new_with_mem(ring, root_of_unity, log2_n, memory_provider))
    }

    pub fn for_complex_with_mem(ring: R, log2_n: usize, memory_provider: &M) -> Self
        where R: RingStore<Type = Complex64>
    {
        let CC = Complex64::RING;
        Self::new_with_mem_and_pows(ring, |i| CC.root_of_unity(i, 1 << log2_n), log2_n, memory_provider)
    }

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

impl<R, M: MemoryProvider<El<R>>> PartialEq for FFTTableCooleyTuckey<R, M> 
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    fn eq(&self, other: &Self) -> bool {
        self.ring().get_ring() == other.ring().get_ring() &&
            self.log2_n == other.log2_n &&
            self.ring().eq_el(self.root_of_unity(), other.root_of_unity())
    }
}

///
/// A helper trait that defines the Cooley-Tuckey butterfly operation.
/// It is default-implemented for all rings, but for increase FFT performance, some rings
/// might wish to provide a specialization.
/// 
pub trait CooleyTuckeyButterfly<S>: RingBase + CanonicalHom<S>
    where S: ?Sized + RingBase
{
    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + twiddle * values[i2], values[i1] - twiddle * values[i2])`
    /// 
    fn butterfly<V: VectorViewMut<Self::Element>>(&self, from: &S, hom: &<Self as CanonicalHom<S>>::Homomorphism, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);

    ///
    /// Should compute `(values[i1], values[i2]) := (values[i1] + values[i2], (values[i1] - values[i2]) * twiddle)`
    /// 
    fn inv_butterfly<V: VectorViewMut<Self::Element>>(&self, from: &S, hom: &<Self as CanonicalHom<S>>::Homomorphism, values: &mut V, twiddle: &S::Element, i1: usize, i2: usize);
}

impl<R, S> CooleyTuckeyButterfly<S> for R
    where S: ?Sized + RingBase, R: ?Sized + RingBase + CanonicalHom<S>
{
    default fn butterfly<V: VectorViewMut<Self::Element>>(&self, from: &S, hom: &<Self as CanonicalHom<S>>::Homomorphism, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        self.mul_assign_map_in_ref(from, values.at_mut(i2), twiddle, hom);
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
    }

    default fn inv_butterfly<V: VectorViewMut<Self::Element>>(&self, from: &S, hom: &<Self as CanonicalHom<S>>::Homomorphism, values: &mut V, twiddle: &<S as RingBase>::Element, i1: usize, i2: usize) {
        let new_a = self.add_ref(values.at(i1), values.at(i2));
        let a = std::mem::replace(values.at_mut(i1), new_a);
        self.sub_self_assign(values.at_mut(i2), a);
        self.mul_assign_map_in_ref(from, values.at_mut(i2), twiddle, hom);
    }
}

#[cfg(not(feature = "const_generic_fft"))]
impl<R, M: MemoryProvider<El<R>>> FFTTableCooleyTuckey<R, M> 
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    fn unordered_fft_dispatch<V, S, const INV: bool>(&self, values: &mut V, ring: &S)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        assert!(values.len() == (1 << self.log2_n));
        let hom = ring.can_hom(&self.ring).unwrap();
        // check if the canonical hom `R -> S` maps `self.root_of_unity` to a primitive N-th root of unity
        debug_assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &hom.map_ref(&self.root_of_unity), self.log2_n));

        for s in 0..self.log2_n {
            let (m, group_size) = if INV {
                (1 << s, 1 << (self.log2_n - s)) 
            } else {
                (1 << (self.log2_n - s - 1), 1 << (s + 1))  
            };
            
            for k in 0..m {
                // 
                // we want to compute a bitreverse_fft_inplace for `v_k, v_(k + m), v_(k + 2m), ..., v_(k + n - m)`;
                // call this sequence a
                //
                // we already have a bitreverse fft of `v_k, v_(k + 2m), v_(k + 4m), ..., v_(k + n - 2m) `
                // and `v_(k + m), v_(k + 3m), v_(k + 5m), ..., v_(k + n - m)` in the corresponding entries;
                // call these sequences a1 and a2
                //
                // Note that a1_i is stored in `(k + 2m * bitrev(i, n/m))` and a2_i in `(k + m + 2m * bitrev(i, n/m))`;
                // We want to store a_i in `(k + m + m * bitrev(i, 2n/m))`
                //
                for i_bitreverse in (0..group_size).step_by(2) {
                    //
                    // we want to compute `(a_i, a_(i + group_size/2)) = (a1_i + z^i a2_i, a1_i - z^i a2_i)`
                    //
                    // in bitreverse order, have
                    // `i_bitreverse     = bitrev(i, group_size) = 2 bitrev(i, group_size/2)` and
                    // `i_bitreverse + 1 = bitrev(i + group_size/2, group_size) = 2 bitrev(i, group_size/2) + 1`
                    //
                    let index1 = i_bitreverse * m + k;
                    let index2 = (i_bitreverse + 1) * m + k;

                    if !INV {
                        let current_twiddle = self.inv_root_of_unity_pow(self.log2_n - s - 1, i_bitreverse);
                        ring.get_ring().butterfly(self.ring.get_ring(), hom.raw_hom(), values, current_twiddle, index1, index2);
                    } else {
                        let current_twiddle = self.root_of_unity_pow(s, i_bitreverse);
                        ring.get_ring().inv_butterfly(self.ring.get_ring(), hom.raw_hom(), values, current_twiddle, index1, index2);
                    }
                }
            }
        }
    }
}

impl<R, M: MemoryProvider<El<R>>> FFTTable for FFTTableCooleyTuckey<R, M> 
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    type Ring = R;
    
    fn len(&self) -> usize {
        1 << self.log2_n
    }

    fn ring(&self) -> &R {
        &self.ring
    }

    fn root_of_unity(&self) -> &El<R> {
        &self.root_of_unity
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        bitreverse(i, self.log2_n)
    }

    fn fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        self.unordered_fft(&mut values, ring, memory_provider);
        self.bitreverse_permute_inplace(&mut values);
    }
        
    fn inv_fft<V, S, N>(&self, mut values: V, ring: S, memory_provider: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: SwappableVectorViewMut<El<S>>,
            N: MemoryProvider<El<S>>
    {
        self.bitreverse_permute_inplace(&mut values);
        self.unordered_inv_fft(&mut values, ring, memory_provider);
    }

    fn unordered_fft<V, S, N>(&self, mut values: V, ring: S, _: &N)
        where S: RingStore, 
            S::Type: CanonicalHom<<Self::Ring as RingStore>::Type>, 
            V: VectorViewMut<El<S>>,
            N: MemoryProvider<El<S>> 
    {
        self.unordered_fft_dispatch::<V, S, false>(&mut values, &ring);
    }    
        
    fn unordered_inv_fft<V, S, N>(&self, mut values: V, ring: S, _: &N)
        where S: RingStore,
            S::Type: CanonicalHom<R::Type>,
            V: VectorViewMut<El<S>>
    {
        self.unordered_fft_dispatch::<V, S, true>(&mut values, &ring);
        let inv = ring.coerce(&self.ring, self.ring.invert(&self.ring.from_int(1 << self.log2_n)).unwrap());
        for i in 0..values.len() {
            ring.mul_assign_ref(values.at_mut(i), &inv);
        }
    }
}

impl<R: RingStore<Type = Complex64>, M: MemoryProvider<El<R>>> ErrorEstimate for FFTTableCooleyTuckey<R, M> {
    
    fn expected_absolute_error(&self, input_bound: f64, input_error: f64) -> f64 {
        // each butterfly doubles the error, and then adds up to 
        let butterfly_absolute_error = input_bound * (root_of_unity_error() + f64::EPSILON);
        // the operator inf-norm of the FFT is its length
        return 2. * self.len() as f64 * butterfly_absolute_error + self.len() as f64 * input_error;
    }
}

#[cfg(feature = "const_generic_fft")]
impl<R, M: MemoryProvider<El<R>>> FFTTableCooleyTuckey<R, M> 
    where R: DivisibilityRingStore, 
        R::Type: DivisibilityRing
{
    fn unordered_fft_const_iter<V, S, const LOG2_N: usize, const ITER: usize, const INV: bool>(&self, values: &mut V, ring: &S, hom: &CanHom<&R, &S>) 
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        assert!(ITER <= LOG2_N);
        assert!(ITER >= 1);
        assert_eq!(LOG2_N, self.log2_n);

        let (m, group_size) = if INV {
            (1 << (ITER - 1), 1 << (LOG2_N - ITER + 1))
        } else {
            (1 << (LOG2_N - ITER), 1<< ITER)   
        };
        
        for k in 0..m {
            for i_bitreverse in (0..group_size).step_by(2) {
                let index1 = i_bitreverse * m + k;
                let index2 = (i_bitreverse + 1) * m + k;

                if !INV {
                    let current_twiddle = self.inv_root_of_unity_pow(LOG2_N - ITER, i_bitreverse);
                    ring.get_ring().butterfly(self.ring.get_ring(), hom.raw_hom(), values, current_twiddle, index1, index2);
                } else {
                    let current_twiddle = self.root_of_unity_pow(ITER - 1, i_bitreverse);
                    ring.get_ring().inv_butterfly(self.ring.get_ring(), hom.raw_hom(), values, current_twiddle, index1, index2);

                }
            }
        }
    }

    #[inline(always)]
    fn unordered_fft_const_iter_opt<V, S, const LOG2_N: usize, const ITER: usize, const INV: bool>(&self, values: &mut V, ring: &S, hom: &CanHom<&R, &S>) 
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        if LOG2_N >= ITER {
            self.unordered_fft_const_iter::<V, S, LOG2_N, ITER, INV>(values, ring, hom);
        }
    }

    fn unordered_fft_const<V, S, const LOG2_N: usize, const INV: bool>(&self, values: &mut V, ring: &S)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        assert_eq!(LOG2_N, self.log2_n);
        assert!(LOG2_N <= 16);
        assert!(values.len() == (1 << LOG2_N));
        let hom = ring.can_hom(&self.ring).unwrap();
        // check if the canonical hom `R -> S` maps `self.root_of_unity` to a primitive N-th root of unity
        debug_assert!(ring.get_ring().is_approximate() || is_prim_root_of_unity_pow2(&ring, &hom.map_ref(&self.root_of_unity), self.log2_n));

        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 1, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 2, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 3, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 4, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 5, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 6, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 7, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 8, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 9, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 10, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 11, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 12, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 13, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 14, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 15, INV>(values, ring, &hom);
        self.unordered_fft_const_iter_opt::<V, S, LOG2_N, 16, INV>(values, ring, &hom);
    }

    fn unordered_fft_dispatch<V, S, const INV: bool>(&self, values: &mut V, ring: &S)
        where S: RingStore, 
            S::Type: CanonicalHom<R::Type>, 
            V: VectorViewMut<El<S>> 
    {
        match self.log2_n {
            0 => {},
            1 => self.unordered_fft_const::<V, S, 1, INV>(values, ring),
            2 => self.unordered_fft_const::<V, S, 2, INV>(values, ring),
            3 => self.unordered_fft_const::<V, S, 3, INV>(values, ring),
            4 => self.unordered_fft_const::<V, S, 4, INV>(values, ring),
            5 => self.unordered_fft_const::<V, S, 5, INV>(values, ring),
            6 => self.unordered_fft_const::<V, S, 6, INV>(values, ring),
            7 => self.unordered_fft_const::<V, S, 7, INV>(values, ring),
            8 => self.unordered_fft_const::<V, S, 8, INV>(values, ring),
            9 => self.unordered_fft_const::<V, S, 9, INV>(values, ring),
            10 => self.unordered_fft_const::<V, S, 10, INV>(values, ring),
            11 => self.unordered_fft_const::<V, S, 11, INV>(values, ring),
            12 => self.unordered_fft_const::<V, S, 12, INV>(values, ring),
            13 => self.unordered_fft_const::<V, S, 13, INV>(values, ring),
            14 => self.unordered_fft_const::<V, S, 14, INV>(values, ring),
            15 => self.unordered_fft_const::<V, S, 15, INV>(values, ring),
            16 => self.unordered_fft_const::<V, S, 16, INV>(values, ring),
            _ => panic!("The const-generic implementation of Cooley-Tuckey only supports lengths up to 2^16")
        }
    }
}

#[cfg(test)]
use crate::rings::zn::zn_static::Zn;
#[cfg(test)]
use crate::rings::zn::zn_barett;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::rings::zn::zn_42;
#[cfg(test)]
use crate::field::*;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;

#[test]
fn test_bitreverse_fft_inplace_basic() {
    let ring = Zn::<5>::RING;
    let z = ring.from_int(2);
    let fft = FFTTableCooleyTuckey::new(ring, ring.div(&1, &z), 2);
    let mut values = [1, 0, 0, 1];
    let expected = [2, 4, 0, 3];
    let mut bitreverse_expected = [0; 4];
    for i in 0..4 {
        bitreverse_expected[i] = expected[bitreverse(i, 2)];
    }

    fft.unordered_fft(&mut values, ring, &AllocatingMemoryProvider);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_fft_inplace_advanced() {
    let ring = Zn::<17>::RING;
    let z = ring.from_int(3);
    let fft = FFTTableCooleyTuckey::new(ring, z, 4);
    let mut values = [1, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 4, 3, 2, 1];
    let expected = [5, 2, 0, 11, 5, 4, 0, 6, 6, 13, 0, 1, 7, 6, 0, 1];
    let mut bitreverse_expected = [0; 16];
    for i in 0..16 {
        bitreverse_expected[i] = expected[bitreverse(i, 4)];
    }

    fft.unordered_fft(&mut values, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(values, bitreverse_expected);
}

#[test]
fn test_bitreverse_inv_fft_inplace() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 4).unwrap();
    let values: [u64; 16] = [1, 2, 3, 2, 1, 0, 17 - 1, 17 - 2, 17 - 1, 0, 1, 2, 3, 4, 5, 6];
    let mut work = values;
    fft.unordered_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    fft.unordered_inv_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(&work, &values);
}

#[test]
fn test_for_zn() {
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));

    let ring = Zn::<97>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(ring, 4).unwrap();
    assert!(ring.is_neg_one(&ring.pow(fft.root_of_unity, 8)));
}

#[cfg(test)]
fn run_fft_bench_round<R, S>(ring: S, fft: &FFTTableCooleyTuckey<R>, data: &Vec<El<S>>, copy: &mut Vec<El<S>>)
    where R: ZnRingStore, R::Type: ZnRing, S: ZnRingStore, S::Type: ZnRing + CanonicalHom<R::Type>
{
    copy.clear();
    copy.extend(data.iter().map(|x| ring.clone_el(x)));
    fft.unordered_fft(&mut copy[..], &ring, &AllocatingMemoryProvider);
    fft.unordered_inv_fft(&mut copy[..], &ring, &AllocatingMemoryProvider);
    assert_el_eq!(&ring, &copy[0], &data[0]);
}

#[bench]
fn bench_fft(bencher: &mut test::Bencher) {
    let ring = zn_barett::Zn::new(StaticRing::<i128>::RING, 1073872897);
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 15).unwrap();
    let data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << 15);
    bencher.iter(|| {
        run_fft_bench_round(&ring, &fft, &data, &mut copy)
    });
}

#[bench]
fn bench_fft_optimized(bencher: &mut test::Bencher) {
    let ring = zn_42::Zn::new(1073872897);
    let fastmul_ring = zn_42::ZnFastmul::new(ring);
    let fft = FFTTableCooleyTuckey::for_zn(&fastmul_ring, 15).unwrap();
    let data = (0..(1 << 15)).map(|i| ring.from_int(i)).collect::<Vec<_>>();
    let mut copy = Vec::with_capacity(1 << 15);
    bencher.iter(|| {
        run_fft_bench_round(&ring, &fft, &data, &mut copy)
    });
}

#[test]
fn test_approximate_fft() {
    let CC = Complex64::RING;
    for log2_n in [4, 7, 11, 15] {
        let fft = FFTTableCooleyTuckey::new_with_mem_and_pows(CC, |x| CC.root_of_unity(x, 1 << log2_n), log2_n, &AllocatingMemoryProvider);
        let mut array = AllocatingMemoryProvider.get_new_init(1 << log2_n, |i|  CC.root_of_unity(i as i64, 1 << log2_n));
        fft.fft(&mut array, CC, &AllocatingMemoryProvider);
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
    let ring = Zn::<17>::RING;
    let fft = FFTTableCooleyTuckey::for_zn(&ring, 0).unwrap();
    let values: [u64; 1] = [3];
    let mut work = values;
    fft.unordered_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(&work, &values);
    fft.unordered_inv_fft(&mut work, fft.ring(), &AllocatingMemoryProvider);
    assert_eq!(&work, &values);
    assert_eq!(0, fft.unordered_fft_permutation(0));
    assert_eq!(0, fft.unordered_fft_permutation_inv(0));
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn generic_test_cooley_tuckey_butterfly<R: RingStore, S: RingStore, I: Iterator<Item = El<R>>>(ring: R, base: S, edge_case_elements: I, test_twiddle: &El<S>)
    where R::Type: CanonicalHom<S::Type>,
        S::Type: DivisibilityRing
{
    let test_inv_twiddle = base.invert(&test_twiddle).unwrap();
    let elements = edge_case_elements.collect::<Vec<_>>();
    let hom = ring.can_hom(&base).unwrap();

    for a in &elements {
        for b in &elements {

            let mut vector = [ring.clone_el(a), ring.clone_el(b)];
            ring.get_ring().butterfly(base.get_ring(), hom.raw_hom(), &mut vector, &test_twiddle, 0, 1);
            assert_el_eq!(&ring, &ring.add_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &vector[0]);
            assert_el_eq!(&ring, &ring.sub_ref_fst(a, ring.mul_ref_fst(b, hom.map_ref(test_twiddle))), &vector[1]);

            ring.get_ring().inv_butterfly(base.get_ring(), hom.raw_hom(), &mut vector, &test_inv_twiddle, 0, 1);
            assert_el_eq!(&ring, &ring.mul_int_ref(a, 2), &vector[0]);
            assert_el_eq!(&ring, &ring.mul_int_ref(b, 2), &vector[1]);

            let mut vector = [ring.clone_el(a), ring.clone_el(b)];
            ring.get_ring().butterfly(base.get_ring(), hom.raw_hom(), &mut vector, &test_twiddle, 1, 0);
            assert_el_eq!(&ring, &ring.add_ref_fst(b, ring.mul_ref_fst(a, hom.map_ref(test_twiddle))), &vector[1]);
            assert_el_eq!(&ring, &ring.sub_ref_fst(b, ring.mul_ref_fst(a, hom.map_ref(test_twiddle))), &vector[0]);

            ring.get_ring().inv_butterfly(base.get_ring(), hom.raw_hom(), &mut vector, &test_inv_twiddle, 1, 0);
            assert_el_eq!(&ring, &ring.mul_int_ref(a, 2), &vector[0]);
            assert_el_eq!(&ring, &ring.mul_int_ref(b, 2), &vector[1]);
        }
    }
}

#[test]
fn test_butterfly() {
    generic_test_cooley_tuckey_butterfly(zn_static::Z17, zn_static::Z17, zn_static::Z17.elements(), &get_prim_root_of_unity_pow2(zn_static::Z17, 4).unwrap());
}