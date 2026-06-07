use std::alloc::{Allocator, Global};

use crate::algorithms::convolution::ConvolutionAlgorithm;
use crate::delegate::DelegateRing;
use crate::prelude::*;

#[stability::unstable(feature = "enable")]
pub struct DelegateConvolution<C, A: Allocator + Send + Sync = Global> {
    convolution: C,
    allocator: A,
}

impl<C, A: Allocator + Send + Sync> DelegateConvolution<C, A> {
    #[stability::unstable(feature = "enable")]
    pub fn new(convolution: C, allocator: A) -> Self { Self { convolution, allocator } }

    #[stability::unstable(feature = "enable")]
    pub fn into(self) -> C { self.convolution }
}

impl<C: Clone, A: Allocator + Send + Sync + Clone> Clone for DelegateConvolution<C, A> {
    fn clone(&self) -> Self {
        Self {
            convolution: self.convolution.clone(),
            allocator: self.allocator.clone(),
        }
    }
}

impl<R, C, A> ConvolutionAlgorithm<R> for DelegateConvolution<C, A>
where
    R: ?Sized + RingBase + DelegateRing,
    C: ConvolutionAlgorithm<<R as DelegateRing>::Base>,
    A: Allocator + Send + Sync,
{
    type PreparedConvolutionOperand = C::PreparedConvolutionOperand;

    fn compute_convolution(
        &self,
        lhs: &[<R as RingBase>::Element],
        lhs_prep: Option<&Self::PreparedConvolutionOperand>,
        rhs: &[<R as RingBase>::Element],
        rhs_prep: Option<&Self::PreparedConvolutionOperand>,
        dst: &mut [<R as RingBase>::Element],
        ring: &R,
    ) {
        let mut lhs_ = Vec::with_capacity_in(lhs.len(), &self.allocator);
        lhs_.extend(lhs.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
        let mut rhs_ = Vec::with_capacity_in(lhs.len(), &self.allocator);
        rhs_.extend(rhs.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
        let mut dst_ = Vec::with_capacity_in(dst.len(), &self.allocator);
        dst_.extend(dst.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
        self.convolution
            .compute_convolution(&lhs_, lhs_prep, &rhs_, rhs_prep, &mut dst_, ring.get_delegate());
        dst.iter_mut().zip(dst_.into_iter()).for_each(|(d, x)| {
            *d = ring.element_cast(ring.rev_delegate(x));
        });
    }

    fn compute_convolution_sum(
        &self,
        values: &[(
            &[<R as RingBase>::Element],
            Option<&Self::PreparedConvolutionOperand>,
            &[<R as RingBase>::Element],
            Option<&Self::PreparedConvolutionOperand>,
        )],
        dst: &mut [<R as RingBase>::Element],
        ring: &R,
    ) {
        let mut data = Vec::with_capacity_in(values.len(), &self.allocator);
        for (lhs, _, rhs, _) in values {
            let mut lhs_ = Vec::with_capacity_in(lhs.len(), &self.allocator);
            lhs_.extend(lhs.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
            let mut rhs_ = Vec::with_capacity_in(lhs.len(), &self.allocator);
            rhs_.extend(rhs.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
            data.push((lhs_, rhs_));
        }
        let mut values_ = Vec::with_capacity_in(values.len(), &self.allocator);
        for ((_, lhs_prep, _, rhs_prep), (lhs, rhs)) in values.iter().zip(&data) {
            values_.push((&lhs[..], *lhs_prep, &rhs[..], *rhs_prep));
        }
        let mut dst_ = Vec::with_capacity_in(dst.len(), &self.allocator);
        dst_.extend(dst.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
        self.convolution
            .compute_convolution_sum(&values_[..], &mut dst_, ring.get_delegate());
        dst.iter_mut().zip(dst_.into_iter()).for_each(|(d, x)| {
            *d = ring.element_cast(ring.rev_delegate(x));
        });
    }

    fn prepare_convolution_operand(
        &self,
        val: &[<R as RingBase>::Element],
        length_hint: Option<usize>,
        ring: &R,
    ) -> Self::PreparedConvolutionOperand {
        let mut val_ = Vec::with_capacity_in(val.len(), &self.allocator);
        val_.extend(val.iter().map(|x| ring.delegate(ring.rev_element_cast(x.clone()))));
        return self
            .convolution
            .prepare_convolution_operand(&val_, length_hint, ring.get_delegate());
    }

    fn supports_ring(&self, ring: &R) -> bool { self.convolution.supports_ring(ring.get_delegate()) }
}
