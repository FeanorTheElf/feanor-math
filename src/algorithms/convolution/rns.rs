use std::alloc::{Allocator, Global};
use std::cmp::{min, max};
use std::marker::PhantomData;

use tracing::instrument;

use crate::algorithms::miller_rabin::is_prime;
use crate::homomorphism::*;
use crate::integer::*;
use crate::lazy::LazyVec;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::zn::zn_64::{Zn64B, Zn64BBase, ZnFastmul, ZnFastmulBase};
use crate::rings::zn::*;
use crate::divisibility::*;
use crate::seq::*;

use super::ntt::NTTConvolution;
use super::ConvolutionAlgorithm;

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
pub struct RNSConvolution<I = BigIntRing, C = NTTConvolution<Zn64BBase, ZnFastmulBase, CanHom<ZnFastmul, Zn64B>>, A = Global, CreateC = CreateNTTConvolution>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    integer_ring: I,
    rns_rings: LazyVec<zn_rns::ZnRNS<Zn64B, I, A>>,
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
pub struct RNSConvolutionZn<I = BigIntRing, C = NTTConvolution<Zn64BBase, ZnFastmulBase, CanHom<ZnFastmul, Zn64B>>, A = Global, CreateC = CreateNTTConvolution>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    base: RNSConvolution<I, C, A, CreateC>
}

///
/// A prepared convolution operand for a [`RNSConvolution`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PreparedConvolutionOperand<R, C = NTTConvolution<Zn64BBase, ZnFastmulBase, CanHom<ZnFastmul, Zn64B>>>
    where R: ?Sized + RingBase,
        C: ConvolutionAlgorithm<Zn64BBase>
{
    prepared: LazyVec<C::PreparedConvolutionOperand>,
    log2_data_size: usize,
    ring: PhantomData<R>,
    len_hint: Option<usize>
}

///
/// Function that creates an [`NTTConvolution`] when given a suitable modulus.
/// 
#[stability::unstable(feature = "enable")]
pub struct CreateNTTConvolution<A = Global>
    where A: Send + Sync + Allocator + Clone
{
    allocator: A
}

impl<I, C, A, CreateC> From<RNSConvolutionZn<I, C, A, CreateC>> for RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    fn from(value: RNSConvolutionZn<I, C, A, CreateC>) -> Self {
        value.base
    }
}

impl<'a, I, C, A, CreateC> From<&'a RNSConvolutionZn<I, C, A, CreateC>> for &'a RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    fn from(value: &'a RNSConvolutionZn<I, C, A, CreateC>) -> Self {
        &value.base
    }
}

impl<I, C, A, CreateC> From<RNSConvolution<I, C, A, CreateC>> for RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    fn from(value: RNSConvolution<I, C, A, CreateC>) -> Self {
        RNSConvolutionZn { base: value }
    }
}

impl<'a, I, C, A, CreateC> From<&'a RNSConvolution<I, C, A, CreateC>> for &'a RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
{
    fn from(value: &'a RNSConvolution<I, C, A, CreateC>) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl CreateNTTConvolution<Global> {

    ///
    /// Creates a new [`CreateNTTConvolution`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub const fn new() -> Self {
        Self { allocator: Global }
    }
}

impl<A> FnOnce<(Zn64B,)> for CreateNTTConvolution<A>
    where A: Send + Sync + Allocator + Clone
{
    type Output = NTTConvolution<Zn64BBase, ZnFastmulBase, CanHom<ZnFastmul, Zn64B>, A>;

    extern "rust-call" fn call_once(self, args: (Zn64B,)) -> Self::Output {
        self.call(args)
    }
}

impl<A> FnMut<(Zn64B,)> for CreateNTTConvolution<A>
    where A: Send + Sync + Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (Zn64B,)) -> Self::Output {
        self.call(args)
    }
}

impl<A> Fn<(Zn64B,)> for CreateNTTConvolution<A>
    where A: Send + Sync + Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (Zn64B,)) -> Self::Output {
        let ring = args.0;
        let ring_fastmul = ZnFastmul::new(ring).unwrap();
        let hom = ring.into_can_hom(ring_fastmul).ok().unwrap();
        NTTConvolution::new_with_hom(hom, self.allocator.clone())
    }
}

impl RNSConvolution {

    ///
    /// Creates a new [`RNSConvolution`] that can compute convolutions of sequences with output
    /// length `<= 2^max_log2_n`. As base convolution, the [`NTTConvolution`] is used.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(max_log2_n: usize) -> Self {
        Self::new_with_convolution(max_log2_n, usize::MAX, BigIntRing::RING, Global, CreateNTTConvolution { allocator: Global })
    }
}

impl<I, C, A, CreateC> RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C
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
    pub fn new_with_convolution(required_root_of_unity_log2: usize, mut max_prime_size_log2: usize, integer_ring: I, allocator: A, create_convolution: CreateC) -> Self {
        max_prime_size_log2 = min(max_prime_size_log2, 57);
        let result = Self {
            integer_ring: integer_ring,
            create_convolution: create_convolution,
            convolutions: LazyVec::new(),
            rns_rings: LazyVec::new(),
            required_root_of_unity_log2: required_root_of_unity_log2,
            allocator: allocator
        };
        let initial_ring = zn_rns::ZnRNS::new_with_alloc(
            vec![Zn64B::new(Self::sample_next_prime(required_root_of_unity_log2, (1 << max_prime_size_log2) + 1).unwrap() as u64)],
            result.integer_ring.clone(), 
            result.allocator.clone()
        );
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

    fn get_rns_ring(&self, moduli_count: usize) -> &zn_rns::ZnRNS<Zn64B, I, A> {
        self.rns_rings.get_or_init_incremental(moduli_count - 1, |_, prev| zn_rns::ZnRNS::new_with_alloc(
            prev.as_iter().cloned().chain([Zn64B::new(Self::sample_next_prime(self.required_root_of_unity_log2, *prev.at(prev.len() - 1).modulus()).unwrap() as u64)]).collect(),
            self.integer_ring.clone(),
            self.allocator.clone()
        ))
    }

    fn get_rns_factor(&self, i: usize) -> &Zn64B {
        let rns_ring = self.get_rns_ring(i + 1);
        return rns_ring.at(rns_ring.len() - 1);
    }

    fn get_convolution(&self, i: usize) -> &C {
        self.convolutions.get_or_init(i, || (self.create_convolution)(*self.get_rns_factor(i)))
    }

    ///
    /// "width" refers to the number of RNS factors we need
    /// 
    fn compute_required_width(&self, input_size_log2: usize, lhs_len: usize, rhs_len: usize, inner_prod_len: usize) -> usize {
        let log2_output_size = input_size_log2 * 2 + 
            StaticRing::<i64>::RING.abs_log2_ceil(&min(lhs_len, rhs_len).try_into().unwrap()).unwrap_or(0) +
            StaticRing::<i64>::RING.abs_log2_ceil(&inner_prod_len.try_into().unwrap()).unwrap_or(0) +
            1;
        let mut width = log2_output_size.div_ceil(57);
        while log2_output_size > self.integer_ring.abs_log2_floor(self.get_rns_ring(width).modulus()).unwrap() {
            width += 1;
        }
        return width;
    }

    fn get_log2_input_size<R, V1, V2, ToInt>(
        &self,
        lhs: V1,
        lhs_prep: Option<&PreparedConvolutionOperand<R, C>>,
        rhs: V2,
        rhs_prep: Option<&PreparedConvolutionOperand<R, C>>,
        _ring: &R,
        mut to_int: ToInt,
        ring_log2_el_size: Option<usize>
    ) -> usize
        where R: ?Sized + RingBase,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> El<I>,
    {
        if let Some(log2_data_size) = ring_log2_el_size {
            assert!(lhs_prep.is_none() || lhs_prep.unwrap().log2_data_size == log2_data_size);
            assert!(rhs_prep.is_none() || rhs_prep.unwrap().log2_data_size == log2_data_size);
            log2_data_size
        } else { 
            max(
                if let Some(lhs_prep) = lhs_prep {
                    lhs_prep.log2_data_size
                } else {
                    lhs.as_iter().map(|x| self.integer_ring.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
                },
                if let Some(rhs_prep) = rhs_prep {
                    rhs_prep.log2_data_size
                } else {
                    rhs.as_iter().map(|x| self.integer_ring.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
                }
            )
        }
    }
    
    fn get_prepared_operand<'a, R>(
        &self,
        data: &[El<Zn64B>],
        data_prep: &'a PreparedConvolutionOperand<R, C>,
        rns_index: usize
    ) -> &'a C::PreparedConvolutionOperand
        where R: ?Sized + RingBase
    {
        data_prep.prepared.get_or_init(rns_index, || self.get_convolution(rns_index).prepare_convolution_operand(data, data_prep.len_hint, self.get_rns_factor(rns_index).get_ring()))
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_impl<R, V1, V2, ToInt, FromInt>(
        &self,
        lhs: V1,
        lhs_prep: Option<&PreparedConvolutionOperand<R, C>>,
        rhs: V2,
        rhs_prep: Option<&PreparedConvolutionOperand<R, C>>,
        dst: &mut [R::Element],
        ring: &R,
        mut to_int: ToInt,
        mut from_int: FromInt,
        ring_log2_el_size: Option<usize>
    )
        where R: ?Sized + RingBase,
            V1: VectorView<R::Element>,
            V2: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> El<I>,
            FromInt: FnMut(El<I>) -> R::Element
    {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }

        let input_size_log2 = self.get_log2_input_size(&lhs, lhs_prep, &rhs, rhs_prep, ring, &mut to_int, ring_log2_el_size);
        let width = self.compute_required_width(input_size_log2, lhs.len(), rhs.len(), 1);
        let len = lhs.len() + rhs.len() - 1;

        let mut res_data = Vec::with_capacity_in(len * width, self.allocator.clone());
        for i in 0..width {
            res_data.extend((0..len).map(|_| self.get_rns_factor(i).zero()));
        }
        let mut lhs_tmp = Vec::with_capacity_in(lhs.len(), self.allocator.clone());
        let mut rhs_tmp = Vec::with_capacity_in(rhs.len(), self.allocator.clone());
        for i in 0..width {
            let hom = self.get_rns_factor(i).into_can_hom(&self.integer_ring).ok().unwrap();
            lhs_tmp.clear();
            lhs_tmp.extend(lhs.as_iter().map(|x| hom.map(to_int(x))));
            rhs_tmp.clear();
            rhs_tmp.extend(rhs.as_iter().map(|x| hom.map(to_int(x))));
            self.get_convolution(i).compute_convolution(
                &lhs_tmp, 
                lhs_prep.map(|lhs_prep| self.get_prepared_operand(&lhs_tmp, lhs_prep, i)),
                &rhs_tmp, 
                rhs_prep.map(|rhs_prep| self.get_prepared_operand(&rhs_tmp, rhs_prep, i)),
                &mut res_data[(i * len)..((i + 1) * len)], 
                self.get_rns_factor(i).get_ring()
            );
        }
        for j in 0..len {
            let add = self.get_rns_ring(width).smallest_lift(self.get_rns_ring(width).from_congruence((0..width).map(|i| res_data[i * len + j])));
            ring.add_assign(&mut dst[j], from_int(add));
        }
    }

    #[instrument(skip_all, level = "trace")]
    fn compute_convolution_sum_impl<R, ToInt, FromInt>(
        &self, 
        values: &[(&[R::Element], Option<&PreparedConvolutionOperand<R, C>>, &[R::Element], Option<&PreparedConvolutionOperand<R, C>>)], 
        dst: &mut [R::Element], 
        ring: &R,
        mut to_int: ToInt,
        mut from_int: FromInt,
        ring_log2_el_size: Option<usize>
    ) 
        where R: ?Sized + RingBase, 
            ToInt: FnMut(&R::Element) -> El<I>,
            FromInt: FnMut(El<I>) -> R::Element
    {
        let out_len = dst.len();
        let inner_product_length = dst.len();

        let mut current_width = 0;
        let mut current_input_size_log2 = 0;
        let mut lhs_max_len = 0;
        let mut rhs_max_len = 0;
        let mut res_data = Vec::new_in(self.allocator.clone());
        let mut lhs_tmp = Vec::new_in(self.allocator.clone());
        let mut rhs_tmp = Vec::new_in(self.allocator.clone());
        
        // the algorithm is as follows:
        //  - we keep track of the current "width" (i.e. number of RNS factors) to represent the result
        //  - we collect iterator elements, until the current width is not sufficient anymore
        //  - then we do `merge_current()`: forward all collected samples to the child convolutions,
        //    add the result to `dst`, and clear the buffers; continue with updated width

        let mut merge_current = |
            current_width: usize, 
            lhs_tmp: &mut Vec<(Vec<El<Zn64B>, _>, Option<&PreparedConvolutionOperand<R, C>>), _>, 
            rhs_tmp: &mut Vec<(Vec<El<Zn64B>, _>, Option<&PreparedConvolutionOperand<R, C>>), _>
        | {
            if current_width == 0 {
                lhs_tmp.clear();
                rhs_tmp.clear();
                return;
            }
            res_data.clear();
            for i in 0..current_width {
                res_data.extend((0..out_len).map(|_| self.get_rns_factor(i).zero()));
                self.get_convolution(i).compute_convolution_sum(
                    &lhs_tmp.iter().zip(rhs_tmp.iter()).map(|((lhs, lhs_prep), (rhs, rhs_prep))| {
                        let lhs_data = &lhs[(i * lhs.len() / current_width)..((i + 1) * lhs.len() / current_width)];
                        let rhs_data = &rhs[(i * rhs.len() / current_width)..((i + 1) * rhs.len() / current_width)];
                        (
                            lhs_data,
                            lhs_prep.map(|lhs_prep| self.get_prepared_operand(lhs_data, lhs_prep, i)),
                            rhs_data,
                            rhs_prep.map(|rhs_prep| self.get_prepared_operand(rhs_data, rhs_prep, i)),
                        )
                    }).collect::<Vec<_>>(),
                    &mut res_data[(i * out_len)..((i + 1) * out_len)],
                    self.get_rns_factor(i).get_ring()
                );
            }
            lhs_tmp.clear();
            rhs_tmp.clear();
            for j in 0..out_len {
                let add = self.get_rns_ring(current_width).smallest_lift(self.get_rns_ring(current_width).from_congruence((0..current_width).map(|i| res_data[i * out_len + j])));
                ring.add_assign(&mut dst[j], from_int(add));
            }
        };

        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            if lhs.len() == 0 || rhs.len() == 0 {
                continue;
            }
            assert!(out_len >= lhs.len() + rhs.len() - 1);
            current_input_size_log2 = max(
                current_input_size_log2,
                self.get_log2_input_size(lhs, *lhs_prep, rhs, *rhs_prep, ring, &mut to_int, ring_log2_el_size)
            );
            lhs_max_len = max(lhs_max_len, lhs.len());
            rhs_max_len = max(rhs_max_len, rhs.len());
            let required_width = self.compute_required_width(current_input_size_log2, lhs_max_len, rhs_max_len, inner_product_length);

            if required_width > current_width {
                merge_current(current_width, &mut lhs_tmp, &mut rhs_tmp);
                current_width = required_width;
            }

            lhs_tmp.push((Vec::with_capacity_in(lhs.len() * current_width, self.allocator.clone()), *lhs_prep));
            rhs_tmp.push((Vec::with_capacity_in(rhs.len() * current_width, self.allocator.clone()), *rhs_prep));
            for i in 0..current_width {
                let hom = self.get_rns_factor(i).into_can_hom(&self.integer_ring).ok().unwrap();
                lhs_tmp.last_mut().unwrap().0.extend(lhs.as_iter().map(|x| hom.map(to_int(x))));
                rhs_tmp.last_mut().unwrap().0.extend(rhs.as_iter().map(|x| hom.map(to_int(x))));
            }
        }
        merge_current(current_width, &mut lhs_tmp, &mut rhs_tmp);
    }
    
    #[instrument(skip_all, level = "trace")]
    fn prepare_convolution_impl<R, V, ToInt>(
        &self,
        data: V,
        _ring: &R,
        length_hint: Option<usize>,
        mut to_int: ToInt,
        ring_log2_el_size: Option<usize>
    ) -> PreparedConvolutionOperand<R, C>
        where R: ?Sized + RingBase,
            V: VectorView<R::Element>,
            ToInt: FnMut(&R::Element) -> El<I>
    {
        let input_size_log2 = if let Some(log2_data_size) = ring_log2_el_size {
            log2_data_size
        } else { 
            data.as_iter().map(|x| self.integer_ring.abs_log2_ceil(&to_int(x)).unwrap_or(0)).max().unwrap()
        };
        return PreparedConvolutionOperand {
            ring: PhantomData,
            len_hint: length_hint,
            prepared: LazyVec::new(),
            log2_data_size: input_size_log2
        };
    }
}

impl<R, I, C, A, CreateC> ConvolutionAlgorithm<R> for RNSConvolution<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C,
        R: ?Sized + IntegerRing
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, C>;

    fn supports_ring(&self, _ring: &R) -> bool {
        true
    }

    fn prepare_convolution_operand(&self, val: &[R::Element], len_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand {
        self.prepare_convolution_impl(
            val,
            ring,
            len_hint,
            |x| int_cast(ring.clone_el(x), &self.integer_ring, RingRef::new(ring)),
            None
        )
    }
    
    fn compute_convolution(&self, lhs: &[R::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R) {
        self.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst,
            ring,
            |x| int_cast(ring.clone_el(x), &self.integer_ring, RingRef::new(ring)),
            |x| int_cast(x, RingRef::new(ring), &self.integer_ring),
            None
        )
    }

    fn compute_convolution_sum(&self, values: &[(&[R::Element], Option<&Self::PreparedConvolutionOperand>, &[R::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R::Element], ring: &R) {
        self.compute_convolution_sum_impl(
            values,
            dst,
            ring,
            |x| int_cast(ring.clone_el(x), &self.integer_ring, RingRef::new(ring)),
            |x| int_cast(x, RingRef::new(ring), &self.integer_ring),
            None
        )
    }
}

impl<R, I, C, A, CreateC> ConvolutionAlgorithm<R> for RNSConvolutionZn<I, C, A, CreateC>
    where I: RingStore + Clone,
        I::Type: IntegerRing,
        C: ConvolutionAlgorithm<Zn64BBase>,
        A: Send + Sync + Allocator + Clone,
        CreateC: Send + Sync + Fn(Zn64B) -> C,
        R: ?Sized + ZnRing + CanHomFrom<I::Type>
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, C>;

    fn supports_ring(&self, _ring: &R) -> bool {
        true
    }

    fn prepare_convolution_operand(&self, val: &[R::Element], len_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand {
        self.base.prepare_convolution_impl(
            val,
            ring,
            len_hint,
            |x| int_cast(ring.smallest_lift(ring.clone_el(x)), &self.base.integer_ring, ring.integer_ring()),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }

    fn compute_convolution(&self, lhs: &[R::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R) {
        let hom = RingRef::new(ring).into_can_hom(&self.base.integer_ring).ok().unwrap();
        self.base.compute_convolution_impl(
            lhs,
            lhs_prep,
            rhs,
            rhs_prep,
            dst,
            ring,
            |x| int_cast(ring.smallest_lift(ring.clone_el(x)), &self.base.integer_ring, ring.integer_ring()),
            |x| hom.map(x),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }

    fn compute_convolution_sum(&self, values: &[(&[R::Element], Option<&Self::PreparedConvolutionOperand>, &[R::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R::Element], ring: &R) {
        let hom = RingRef::new(ring).into_can_hom(&self.base.integer_ring).ok().unwrap();
        self.base.compute_convolution_sum_impl(
            values,
            dst,
            ring,
            |x| int_cast(ring.smallest_lift(ring.clone_el(x)), &self.base.integer_ring, ring.integer_ring()),
            |x| hom.map(x),
            Some(ring.integer_ring().abs_log2_ceil(ring.modulus()).unwrap())
        )
    }
}

#[cfg(test)]
use super::STANDARD_CONVOLUTION;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_convolution_integer() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i128>::RING;
    let convolution = RNSConvolution::new_with_convolution(7, usize::MAX, BigIntRing::RING, Global, NTTConvolution::new);

    super::generic_tests::test_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
}

#[test]
fn test_convolution_zn() {
    LogAlgorithmSubscriber::init_test();
    let ring = Zn64B::new((1 << 57) + 1);
    let convolution = RNSConvolutionZn::from(RNSConvolution::new_with_convolution(7, usize::MAX, BigIntRing::RING, Global, NTTConvolution::new));

    super::generic_tests::test_convolution(&convolution, &ring, ring.int_hom().map(1 << 30));
}

#[test]
fn test_convolution_sum() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i128>::RING;
    let convolution = RNSConvolution::new_with_convolution(7, 20, BigIntRing::RING, Global, NTTConvolution::new);
    
    let data = (0..40usize).map(|i| (
        (0..(5 + i % 5)).map(|x| (1 << i) * (x as i128 - 2)).collect::<Vec<_>>(),
        (0..(13 - i % 7)).map(|x| (1 << i) * (x as i128 + 1)).collect::<Vec<_>>(),
    )).collect::<Vec<_>>();
    let mut expected = (0..22).map(|_| 0).collect::<Vec<_>>();
    STANDARD_CONVOLUTION.compute_convolution_sum(&data.iter().map(|(l, r)| (&l[..], None, &r[..], None)).collect::<Vec<_>>(), &mut expected, ring.get_ring());

    let mut actual = (0..21).map(|_| 0).collect::<Vec<_>>();
    convolution.compute_convolution_sum(&data.iter().map(|(l, r)| (&l[..], None, &r[..], None)).collect::<Vec<_>>(), &mut actual, ring.get_ring());
    assert_eq!(&expected[..21], actual);
    
    let data_prep = data.iter().map(|(l, r)| {
        let l_prep = convolution.prepare_convolution_operand(&l, Some(21), ring.get_ring());
        let r_prep = convolution.prepare_convolution_operand(&r, Some(21), ring.get_ring());
        (&l[..], l_prep, &r[..], r_prep)
    }).collect::<Vec<_>>();
    let mut actual = (0..21).map(|_| 0).collect::<Vec<_>>();
    convolution.compute_convolution_sum(&data_prep.iter().map(|(l, l_prep, r, r_prep)| (*l, Some(l_prep), *r, Some(r_prep))).collect::<Vec<_>>(), &mut actual, ring.get_ring());
    assert_eq!(&expected[..21], actual);
    
    let mut actual = (0..21).map(|_| 0).collect::<Vec<_>>();
    convolution.compute_convolution_sum(&data_prep.iter().enumerate().map(|(i, (l, l_prep, r, r_prep))| match i % 4 {
        0 => (*l, Some(l_prep), *r, Some(r_prep)),
        1 => (*l, None, *r, Some(r_prep)),
        2 => (*l, Some(l_prep), *r, None),
        3 => (*l, None, *r, None),
        _ => unreachable!()
    }).collect::<Vec<_>>(), &mut actual, ring.get_ring());
    assert_eq!(&expected[..21], actual);
}