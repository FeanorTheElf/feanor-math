use std::alloc::{Allocator, Global};
use std::cmp::{min, max};

use crate::algorithms::miller_rabin::is_prime;
use crate::homomorphism::*;
use crate::integer::*;
use crate::lazy::LazyVec;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::zn::zn_64::{Zn, ZnBase};
use crate::rings::zn::*;
use crate::divisibility::*;
use crate::seq::*;

use super::ntt::NTTConvolution;
use super::{ConvolutionAlgorithm, PreparedConvolutionAlgorithm};

///
/// A [`ConvolutionAlgorithm`] that computes convolutions by computing them modulo a
/// suitable composite modulus `q`, whose factors are of a certain shape, usually such
/// as to allow for NTT-based convolutions.
/// 
/// Due to overlapping blanket impls, this type can only be used to compute convolutions
/// over [`IntegerRing`]s. For computing convolutions over [`ZnRing`]s, wrap it in a
/// [`RNSConvolutionZn`].
/// 
#[stability::unstable(feature = "enable")]
pub struct RNSConvolution<I = BigIntRing, C = NTTConvolution<Zn>, A = Global, CreateC = CreateNTTConvolution>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    integer_ring: I,
    rns_rings: LazyVec<zn_rns::Zn<Zn, I, A>>,
    convolutions: LazyVec<C>,
    create_convolution: CreateC,
    required_root_of_unity_log2: usize,
    allocator: A
}

///
/// Same as [`RNSConvolution`], but computes convolutions over [`ZnRing`]s.
/// 
#[stability::unstable(feature = "enable")]
#[repr(transparent)]
pub struct RNSConvolutionZn<I = BigIntRing, C = NTTConvolution<Zn>, A = Global, CreateC = CreateNTTConvolution>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    base: RNSConvolution<I, C, A, CreateC>
}

///
/// A prepared convolution operand for a [`RNSConvolution`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, C = NTTConvolution<Zn>>
    where R: ?Sized + RingBase,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    data: Vec<R::Element>,
    prepared: LazyVec<C::PreparedConvolutionOperand>,
    log2_data_size: usize
}

///
/// A prepared convolution operand for a [`RNSConvolutionZn`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperandZn<R, C = NTTConvolution<Zn>>(PreparedConvolutionOperand<R::IntegerRingBase, C>)
    where R: ?Sized + ZnRing,
        C: PreparedConvolutionAlgorithm<ZnBase>;

///
/// Function that creates an [`NTTConvolution`] when given a suitable modulus.
/// 
#[stability::unstable(feature = "enable")]
pub struct CreateNTTConvolution<A = Global>
    where A: Allocator + Clone
{
    allocator: A
}

impl<I, C, A, CreateC> From<RNSConvolutionZn<I, C, A, CreateC>> for RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn from(value: RNSConvolutionZn<I, C, A, CreateC>) -> Self {
        value.base
    }
}

impl<'a, I, C, A, CreateC> From<&'a RNSConvolutionZn<I, C, A, CreateC>> for &'a RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn from(value: &'a RNSConvolutionZn<I, C, A, CreateC>) -> Self {
        &value.base
    }
}

impl<I, C, A, CreateC> From<RNSConvolution<I, C, A, CreateC>> for RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn from(value: RNSConvolution<I, C, A, CreateC>) -> Self {
        RNSConvolutionZn { base: value }
    }
}

impl<'a, I, C, A, CreateC> From<&'a RNSConvolution<I, C, A, CreateC>> for &'a RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn from(value: &'a RNSConvolution<I, C, A, CreateC>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<A> FnOnce<(Zn,)> for CreateNTTConvolution<A>
    where A: Allocator + Clone
{
    type Output = NTTConvolution<Zn, A>;

    extern "rust-call" fn call_once(self, args: (Zn,)) -> Self::Output {
        self.call(args)
    }
}

impl<A> FnMut<(Zn,)> for CreateNTTConvolution<A>
    where A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (Zn,)) -> Self::Output {
        self.call(args)
    }
}

impl<A> Fn<(Zn,)> for CreateNTTConvolution<A>
    where A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (Zn,)) -> Self::Output {
        NTTConvolution::new_with(args.0, self.allocator.clone())
    }
}

impl RNSConvolution {

    ///
    /// Creates a new [`RNSConvolution`] that can compute convolutions of sequences with output
    /// length `<= 2^max_log2_n`. As base convolution, the [`NTTConvolution`] is used.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(max_log2_n: usize) -> Self {
        Self::new_with(max_log2_n, usize::MAX, BigIntRing::RING, Global, CreateNTTConvolution { allocator: Global })
    }
}

impl<I, C, A, CreateC> RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    ///
    /// Creates a new [`RNSConvolution`] with all the given configuration parameters.
    /// 
    /// In particular
    ///  - `required_root_of_unity_log2` and `max_prime_size_log2` control which prime factors
    ///    are used for the underlying composite modulus; Only primes `<= 2^max_prime_size_log2` and
    ///    `= 1` mod `required_root_of_unity_log2` are sampled
    ///  - `integer_ring` is the ring to store intermediate lifts in; this probably has to be [`BigIntRing`],
    ///    unless inputs are pretty small
    ///  - `allocator` is used to allocate elements modulo the internal modulus, as elements of [`zn_rns::Zn`]
    ///  - `create_convolution` is called whenever a new convolution algorithm for a new prime has to be
    ///    created; the modulus of the given [`Zn`] always satisfies the constraints defined by `max_prime_size_log2`
    ///    and `required_root_of_unity_log2`
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with(required_root_of_unity_log2: usize, mut max_prime_size_log2: usize, integer_ring: I, allocator: A, create_convolution: CreateC) -> Self {
        max_prime_size_log2 = min(max_prime_size_log2, 57);
        let result = Self {
            integer_ring: integer_ring,
            create_convolution: create_convolution,
            convolutions: LazyVec::new(),
            rns_rings: LazyVec::new(),
            required_root_of_unity_log2: required_root_of_unity_log2,
            allocator: allocator
        };
        let initial_ring = zn_rns::Zn::new_with(vec![Zn::new(Self::sample_next_prime(required_root_of_unity_log2, (1 << max_prime_size_log2) + 1).unwrap() as u64)], result.integer_ring.clone(), result.allocator.clone());
        _ = result.rns_rings.get_or_init(0, || initial_ring);
        return result;
    }

    fn sample_next_prime(required_root_of_unity_log2: usize, current: i64) -> Option<i64> {
        let mut k = StaticRing::<i64>::RING.checked_div(&(current - 1), &(1 << required_root_of_unity_log2)).unwrap();
        while k > 0 {
            k -= 1;
            let candidate = (k << required_root_of_unity_log2) + 1;
            if is_prime(StaticRing::<i64>::RING, &candidate, 10) {
                return Some(candidate);
            }
        }
        return None;
    }

    fn get_rns_ring(&self, moduli_count: usize) -> &zn_rns::Zn<Zn, I, A> {
        self.rns_rings.get_or_init_incremental(moduli_count, |_, prev| zn_rns::Zn::new_with(
            prev.as_iter().cloned().chain([Zn::new(Self::sample_next_prime(self.required_root_of_unity_log2, *prev.at(prev.len() - 1).modulus()).unwrap() as u64)]).collect(),
            self.integer_ring.clone(),
            self.allocator.clone()
        ));
        return self.rns_rings.get(moduli_count - 1).unwrap();
    }

    fn get_rns_factor(&self, i: usize) -> &Zn {
        let rns_ring = self.get_rns_ring(i + 1);
        return rns_ring.at(rns_ring.len() - 1);
    }

    fn get_convolution(&self, i: usize) -> &C {
        self.convolutions.get_or_init(i, || (self.create_convolution)(*self.get_rns_factor(i)))
    }

    fn extend_operand<R, F>(&self, operand: &PreparedConvolutionOperand<R, C>, target_width: usize, mut mod_part: F)
        where R: ?Sized + RingBase,
            C: PreparedConvolutionAlgorithm<ZnBase>,
            F: FnMut(&R::Element, usize) -> El<Zn>
    {
        let mut tmp = Vec::new();
        tmp.resize_with(operand.data.len(), || self.get_rns_factor(0).zero());
        for i in 0..target_width {
            _ = operand.prepared.get_or_init(i, || {
                for j in 0..operand.data.len() {
                    tmp[j] = mod_part(&operand.data[j], i);
                }
                self.get_convolution(i).prepare_convolution_operand(&tmp, self.get_rns_factor(i))
            });
        }
    }

    fn compute_required_width(&self, input_size_log2: usize, lhs_len: usize, rhs_len: usize, inner_prod_len: usize) -> usize {
        let log2_output_size = input_size_log2 * 2 + 
            StaticRing::<i64>::RING.abs_log2_ceil(&(min(lhs_len, rhs_len) as i64)).unwrap_or(0) +
            StaticRing::<i64>::RING.abs_log2_ceil(&(inner_prod_len as i64)).unwrap_or(0) +
            1;
        let mut width = (log2_output_size - 1) / 57 + 1;
        while log2_output_size > self.integer_ring.abs_log2_floor(self.get_rns_ring(width).modulus()).unwrap() {
            width += 1;
        }
        return width;
    }
}

impl<I, C, A, CreateC> RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn compute_convolution_impl<S, V1, V2, D>(&self, input_size_log2: usize, lhs: V1, rhs: V2, mut dst: D, ring: S)
        where S: RingStore,
            S::Type: RingBase + IntegerRing,
            D: FnMut(usize, El<I>),
            V1: VectorFn<El<S>>,
            V2: VectorFn<El<S>>
    {
        let width = self.compute_required_width(input_size_log2, lhs.len(), rhs.len(), 1);
        let len = lhs.len() + rhs.len();
        let mut res_data = Vec::with_capacity(len * width);
        for i in 0..width {
            res_data.extend((0..len).map(|_| self.get_rns_factor(i).zero()));
        }

        let mut lhs_tmp = Vec::with_capacity(lhs.len());
        lhs_tmp.resize_with(lhs.len(), || self.get_rns_factor(0).zero());
        let mut rhs_tmp = Vec::with_capacity(rhs.len());
        rhs_tmp.resize_with(rhs.len(), || self.get_rns_factor(0).zero());
        for i in 0..width {
            let hom = self.get_rns_factor(i).can_hom(&ring).unwrap();
            for j in 0..lhs.len() {
                lhs_tmp[j] = hom.map(lhs.at(j));
            }
            for j in 0..rhs.len() {
                rhs_tmp[j] = hom.map(rhs.at(j));
            }
            self.get_convolution(i).compute_convolution(&lhs_tmp, &rhs_tmp, &mut res_data[(i * len)..((i + 1) * len)], self.get_rns_factor(i));
        }

        for j in 0..(len - 1) {
            dst(j, self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * len + j]))));
        }
    }
}

impl<I, C, A, CreateC> RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: PreparedConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C
{
    fn prepare_convolution_operand_impl<S, V>(&self, input_size_log2: usize, val: V, _ring: S) -> PreparedConvolutionOperand<S::Type, C>
        where S: RingStore + Copy,
            S::Type: IntegerRing, 
            V: VectorFn<El<S>>
    {
        let mut data = Vec::with_capacity(val.len());
        data.extend(val.iter());
        return PreparedConvolutionOperand {
            data: data,
            prepared: LazyVec::new(),
            log2_data_size: input_size_log2
        };
    }

    fn compute_convolution_inner_product_lhs_prepared_impl<'a, S, V, D>(&self, rhs_input_size_log2: usize, values: &[(&'a PreparedConvolutionOperand<S::Type, C>, V)], mut dst: D, ring: S)
        where S: RingStore + Copy,
            S::Type: IntegerRing, 
            D: FnMut(usize, El<I>),
            V: VectorFn<El<S>>,
            S: 'a,
            Self: 'a,
            PreparedConvolutionOperand<S::Type, C>: 'a
    {
        let max_len = values.iter().map(|(lhs, rhs)| lhs.data.len() + rhs.len()).max().unwrap_or(0);
        let input_size_log2 = max(rhs_input_size_log2, values.iter().map(|(lhs, _)| lhs.log2_data_size).max().unwrap_or(0));
        let width = self.compute_required_width(input_size_log2, (max_len - 1) / 2 + 1, (max_len - 1) / 2 + 1, values.len());
        let mut res_data = Vec::with_capacity(max_len * width);
        for i in 0..width {
            res_data.extend((0..max_len).map(|_| self.get_rns_factor(i).zero()));
        }

        let mut rhs_tmp = Vec::with_capacity(max_len * values.len());
        rhs_tmp.resize_with(max_len * values.len(), || self.get_rns_factor(0).zero());

        let homs = (0..width).map(|i| self.get_rns_factor(i).can_hom(&ring).unwrap()).collect::<Vec<_>>();
        for j in 0..values.len() {
            self.extend_operand(values[j].0, width, |x, i| homs[i].map_ref(x));
        }

        for i in 0..width {
            for j in 0..values.len() {
                for k in 0..values[j].1.len() {
                    rhs_tmp[j * max_len + k] = homs[i].map(values[j].1.at(k));
                }
            }
            self.get_convolution(i).compute_convolution_inner_product_lhs_prepared(
                values.iter().enumerate().map(|(j, (lhs, _))| (lhs.prepared.get(i).unwrap(), &rhs_tmp[(j * max_len)..(j * max_len + values[j].1.len())])), 
                &mut res_data[(i * max_len)..((i + 1) * max_len)], 
                self.get_rns_factor(i)
            );
        }
        
        for j in 0..(max_len - 1) {
            dst(j, self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * max_len + j]))));
        }
    }

    fn compute_convolution_inner_product_prepared_impl<'a, S, D>(&self, values: &[(&'a PreparedConvolutionOperand<S::Type, C>, &'a PreparedConvolutionOperand<S::Type, C>)], mut dst: D, ring: S)
        where S: RingStore + Copy,
            S::Type: IntegerRing, 
            D: FnMut(usize, El<I>),
            Self: 'a,
            S: 'a,
            PreparedConvolutionOperand<S::Type, C>: 'a
    {
        let max_len = values.iter().map(|(lhs, rhs)| lhs.data.len() + rhs.data.len()).max().unwrap_or(0);
        let input_size_log2 = values.iter().map(|(lhs, rhs)| max(lhs.log2_data_size, rhs.log2_data_size)).max().unwrap_or(0);
        let width = self.compute_required_width(input_size_log2, (max_len - 1) / 2 + 1, (max_len - 1) / 2 + 1, values.len());
        let mut res_data = Vec::with_capacity(max_len * width);
        for i in 0..width {
            res_data.extend((0..max_len).map(|_| self.get_rns_factor(i).zero()));
        }

        let homs = (0..width).map(|i| self.get_rns_factor(i).can_hom(&ring).unwrap()).collect::<Vec<_>>();
        for j in 0..values.len() {
            self.extend_operand(values[j].0, width, |x, i| homs[i].map_ref(x));
            self.extend_operand(values[j].1, width, |x, i| homs[i].map_ref(x));
        }

        for i in 0..width {
            self.get_convolution(i).compute_convolution_inner_product_prepared(
                values.iter().map(|(lhs, rhs)| (lhs.prepared.get(i).unwrap(), rhs.prepared.get(i).unwrap())), 
                &mut res_data[(i * max_len)..((i + 1) * max_len)], 
                self.get_rns_factor(i)
            );
        }
        
        for j in 0..(max_len - 1) {
            dst(j, self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * max_len + j]))));
        }
    }

    fn compute_convolution_lhs_prepared_impl<S, V, D>(&self, rhs_input_size_log2: usize, lhs: &PreparedConvolutionOperand<S::Type, C>, rhs: V, mut dst: D, ring: S)
        where S: RingStore + Copy,
            S::Type: IntegerRing, 
            D: FnMut(usize, El<I>),
            V: VectorFn<El<S>>
    {
        let width = self.compute_required_width(max(rhs_input_size_log2, lhs.log2_data_size), lhs.data.len(), rhs.len(), 1);
        let len = lhs.data.len() + rhs.len();
        let mut res_data = Vec::with_capacity(len * width);
        for i in 0..width {
            res_data.extend((0..len).map(|_| self.get_rns_factor(i).zero()));
        }

        let mut rhs_tmp = Vec::with_capacity(rhs.len());
        rhs_tmp.resize_with(rhs.len(), || self.get_rns_factor(0).zero());

        let homs = (0..width).map(|i| self.get_rns_factor(i).can_hom(&ring).unwrap()).collect::<Vec<_>>();
        self.extend_operand(lhs, width, |x, i| homs[i].map_ref(x));

        for i in 0..width {
            for j in 0..rhs.len() {
                rhs_tmp[j] = homs[i].map(rhs.at(j));
            }
            self.get_convolution(i).compute_convolution_lhs_prepared(lhs.prepared.get(i).unwrap(), &rhs_tmp, &mut res_data[(i * len)..((i + 1) * len)], self.get_rns_factor(i));
        }
        
        for j in 0..(len - 1) {
            dst(j, self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * len + j]))));
        }
    }

    fn compute_convolution_prepared_impl<S, D>(&self, lhs: &PreparedConvolutionOperand<S::Type, C>, rhs: &PreparedConvolutionOperand<S::Type, C>, mut dst: D, ring: S)
        where S: RingStore + Copy,
            S::Type: IntegerRing, 
            D: FnMut(usize, El<I>),
    {
        let width = self.compute_required_width(max(lhs.log2_data_size, rhs.log2_data_size), lhs.data.len(), rhs.data.len(), 1);
        let len = lhs.data.len() + rhs.data.len();
        let mut res_data = Vec::with_capacity(len * width);
        for i in 0..width {
            res_data.extend((0..len).map(|_| self.get_rns_factor(i).zero()));
        }

        let homs = (0..width).map(|i| self.get_rns_factor(i).can_hom(&ring).unwrap()).collect::<Vec<_>>();
        self.extend_operand(lhs, width, |x, i| homs[i].map_ref(x));
        self.extend_operand(rhs, width, |x, i| homs[i].map_ref(x));

        for i in 0..width {
            self.get_convolution(i).compute_convolution_prepared(lhs.prepared.get(i).unwrap(), rhs.prepared.get(i).unwrap(), &mut res_data[(i * len)..((i + 1) * len)], self.get_rns_factor(i));
        }
        
        for j in 0..(len - 1) {
            dst(j, self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * len + j]))));
        }
    }
}

impl<R, I, C, A, CreateC> ConvolutionAlgorithm<R> for RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C,
        R: ?Sized + IntegerRing
{
    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<<R as RingBase>::Element>, V2: VectorView<<R as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [<R as RingBase>::Element], ring: S) {
        assert!(dst.len() >= lhs.len() + rhs.len() - 1);
        let log2_input_size = lhs.as_iter().chain(rhs.as_iter()).map(|x| ring.abs_log2_ceil(x).unwrap_or(0)).max().unwrap_or(0);
        println!("{}", log2_input_size);
        let hom = ring.can_hom(&self.integer_ring).unwrap();
        return self.compute_convolution_impl(
            log2_input_size,
            lhs.clone_ring_els(ring),
            rhs.clone_ring_els(ring),
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring
        );
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

impl<R, I, C, A, CreateC> PreparedConvolutionAlgorithm<R> for RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: PreparedConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C,
        R: ?Sized + IntegerRing
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, C>;
    
    fn prepare_convolution_operand<S, V>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        let log2_input_size = val.as_iter().map(|x| ring.abs_log2_ceil(x).unwrap_or(0)).max().unwrap_or(0);
        return self.prepare_convolution_operand_impl(log2_input_size, val.clone_ring_els(ring), ring);
    }

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        assert!(dst.len() >= lhs.data.len() + rhs.len() - 1);
        let rhs_log2_input_size = rhs.as_iter().map(|x| ring.abs_log2_ceil(x).unwrap_or(0)).max().unwrap_or(0);
        let hom = ring.can_hom(&self.integer_ring).unwrap();
        return self.compute_convolution_lhs_prepared_impl(
            rhs_log2_input_size, 
            lhs, 
            rhs.clone_ring_els(ring), 
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)), 
            ring
        );
    }

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy
    {
        assert!(dst.len() >= lhs.data.len() + rhs.data.len() - 1);
        let hom = ring.can_hom(&self.integer_ring).unwrap();
        return self.compute_convolution_prepared_impl(
            lhs, 
            rhs, 
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)), 
            ring
        );
    }

    fn compute_convolution_inner_product_lhs_prepared<'a, S, J, V>(&self, values: J, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            J: Iterator<Item = (&'a Self::PreparedConvolutionOperand, V)>,
            V: VectorView<R::Element>,
            Self: 'a,
            R: 'a,
            Self::PreparedConvolutionOperand: 'a
    {
        let values = values.map(|(lhs, rhs)| (lhs, rhs.into_clone_ring_els(ring))).collect::<Vec<_>>();
        let rhs_log2_input_size = values.iter().flat_map(|(_, rhs)| rhs.iter()).map(|x| ring.abs_log2_ceil(&x).unwrap_or(0)).max().unwrap_or(0);
        let hom = ring.can_hom(&self.integer_ring).unwrap();
        return self.compute_convolution_inner_product_lhs_prepared_impl(
            rhs_log2_input_size,
            &values,
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring
        );
    }

    fn compute_convolution_inner_product_prepared<'a, S, J>(&self, values: J, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            J: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            Self::PreparedConvolutionOperand: 'a,
            Self: 'a,
            R: 'a,
    {
        let values = values.collect::<Vec<_>>();
        let hom = ring.can_hom(&self.integer_ring).unwrap();
        return self.compute_convolution_inner_product_prepared_impl(
            &values,
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring
        );
    }
}

impl<R, I, C, A, CreateC> ConvolutionAlgorithm<R> for RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C,
        R: ?Sized + ZnRing + CanHomFrom<I::Type>
{
    fn compute_convolution<S: RingStore<Type = R> + Copy, V1: VectorView<<R as RingBase>::Element>, V2: VectorView<<R as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [<R as RingBase>::Element], ring: S) {
        assert!(dst.len() >= lhs.len() + rhs.len() - 1);
        let log2_input_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap() - 1;
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        return self.base.compute_convolution_impl(
            log2_input_size,
            lhs.clone_ring_els(ring).map_fn(|x| ring.smallest_lift(x)),
            rhs.clone_ring_els(ring).map_fn(|x| ring.smallest_lift(x)),
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring.integer_ring()
        );
    }

    fn supports_ring<S: RingStore<Type = R> + Copy>(&self, _ring: S) -> bool {
        true
    }
}

impl<R, I, C, A, CreateC> PreparedConvolutionAlgorithm<R> for RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: PreparedConvolutionAlgorithm<ZnBase>,
        A: Allocator + Clone,
        CreateC: Fn(Zn) -> C,
        R: ?Sized + ZnRing + CanHomFrom<I::Type>
{
    type PreparedConvolutionOperand = PreparedConvolutionOperandZn<R, C>;
    
    fn prepare_convolution_operand<S, V>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        let log2_input_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap() - 1;
        return PreparedConvolutionOperandZn(self.base.prepare_convolution_operand_impl(log2_input_size, val.clone_ring_els(ring).map_fn(|x| ring.smallest_lift(x)), ring.integer_ring()));
    }

    fn compute_convolution_lhs_prepared<S, V>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy, V: VectorView<R::Element>
    {
        assert!(dst.len() >= lhs.0.data.len() + rhs.len() - 1);
        let rhs_log2_input_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap() - 1;
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        return self.base.compute_convolution_lhs_prepared_impl(
            rhs_log2_input_size, 
            &lhs.0, 
            rhs.clone_ring_els(ring).map_fn(|x| ring.smallest_lift(x)), 
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring.integer_ring()
        );
    }

    fn compute_convolution_prepared<S>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [R::Element], ring: S)
        where S: RingStore<Type = R> + Copy
    {
        assert!(dst.len() >= lhs.0.data.len() + rhs.0.data.len() - 1);
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        return self.base.compute_convolution_prepared_impl(
            &lhs.0, 
            &rhs.0, 
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring.integer_ring()
        );
    }

    fn compute_convolution_inner_product_lhs_prepared<'a, S, J, V>(&self, values: J, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            J: Iterator<Item = (&'a Self::PreparedConvolutionOperand, V)>,
            V: VectorView<R::Element>,
            Self: 'a,
            R: 'a,
            Self::PreparedConvolutionOperand: 'a
    {
        let values = values.map(|(lhs, rhs)| (&lhs.0, rhs.into_clone_ring_els(ring).map_fn(|x| ring.smallest_lift(x)))).collect::<Vec<_>>();
        let rhs_log2_input_size = ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap() - 1;
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        return self.base.compute_convolution_inner_product_lhs_prepared_impl(
            rhs_log2_input_size,
            &values,
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring.integer_ring()
        );
    }

    fn compute_convolution_inner_product_prepared<'a, S, J>(&self, values: J, dst: &mut [R::Element], ring: S) 
        where S: RingStore<Type = R> + Copy, 
            J: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            Self::PreparedConvolutionOperand: 'a,
            Self: 'a,
            R: 'a,
    {
        let values = values.map(|(lhs, rhs)| (&lhs.0, &rhs.0)).collect::<Vec<_>>();
        let hom = ring.can_hom(&self.base.integer_ring).unwrap();
        return self.base.compute_convolution_inner_product_prepared_impl(
            &values,
            |i, x| ring.add_assign(&mut dst[i], hom.map(x)),
            ring.integer_ring()
        );
    }
}

#[test]
fn test_convolution_integer() {
    let ring = StaticRing::<i128>::RING;
    let convolution = RNSConvolution::new_with(7, usize::MAX, BigIntRing::RING, Global, |Fp| NTTConvolution::new_with(Fp, Global));

    super::generic_tests::test_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
    super::generic_tests::test_prepared_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
}

#[test]
fn test_convolution_zn() {
    let ring = Zn::new((1 << 57) + 1);
    let convolution = RNSConvolutionZn::from(RNSConvolution::new_with(7, usize::MAX, BigIntRing::RING, Global, |Fp| NTTConvolution::new_with(Fp, Global)));

    super::generic_tests::test_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
    super::generic_tests::test_prepared_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
}