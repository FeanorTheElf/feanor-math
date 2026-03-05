use crate::ring::*;
use crate::algorithms::convolution::*;

use std::marker::PhantomData;
use std::cmp::min;

pub struct LengthExtendedConvolution<C> {
    base_convolution: C,
    base_max_len: usize
}

pub struct PreparedConvolutionOperand<R, C>
    where R: ?Sized + RingBase,
        C: ConvolutionAlgorithm<R>
{
    prepared_parts: Vec<C::PreparedConvolutionOperand>,
    convolution: PhantomData<C>,
    ring: PhantomData<R>
}

impl<C> LengthExtendedConvolution<C> {

    pub fn new(base_convolution: C, base_max_len: usize) -> Self {
        assert!(base_max_len >= 2);
        Self { base_convolution, base_max_len }
    }

    fn chunk_len(&self) -> usize {
        self.base_max_len / 2
    }
}

impl<R, C> ConvolutionAlgorithm<R> for LengthExtendedConvolution<C>
    where R: ?Sized + RingBase,
        C: ConvolutionAlgorithm<R>
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, C>;

    fn supports_ring(&self, ring: &R) -> bool {
        self.base_convolution.supports_ring(ring)
    }

    fn prepare_convolution_operand(&self, val: &[R::Element], length_hint: Option<usize>, ring: &R) -> Self::PreparedConvolutionOperand {
        let base_length_hint = match length_hint {
            None => None,
            Some(len) if len < val.len() => panic!("length_hint cannot be smaller than the length of a single convolution operand"),
            Some(len) if len < self.base_max_len => Some(len),
            Some(len) => Some(len)
        };
        PreparedConvolutionOperand {
            prepared_parts: val.chunks(self.chunk_len()).map(|chunk| 
                self.base_convolution.prepare_convolution_operand(chunk, base_length_hint, ring)
            ).collect(),
            ring: PhantomData,
            convolution: PhantomData
        }
    }
    
    fn compute_convolution(&self, lhs: &[R::Element], lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: &[R::Element], rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [R::Element], ring: &R) {
        if lhs.len() == 0 || rhs.len() == 0 {
            return;
        }

        let len = dst.len();
        assert!(lhs.len() + rhs.len() <= len + 1);
        let lhs_prep_owned = if lhs_prep.is_none() {
            Some(self.prepare_convolution_operand(lhs, Some(lhs.len() + rhs.len()), ring))
        } else {
            None
        };
        let lhs_prep = lhs_prep.or(lhs_prep_owned.as_ref()).unwrap();
        assert!((lhs_prep.prepared_parts.len() - 1) * self.chunk_len() < lhs.len());
        
        let rhs_prep_owned = if rhs_prep.is_none() {
            Some(self.prepare_convolution_operand(rhs, Some(lhs.len() + rhs.len()), ring))
        } else {
            None
        };
        let rhs_prep = rhs_prep.or(rhs_prep_owned.as_ref()).unwrap();
        assert!((rhs_prep.prepared_parts.len() - 1) * self.chunk_len() < rhs.len());

        for k in 0..(lhs_prep.prepared_parts.len() + rhs_prep.prepared_parts.len() - 1) {
            let values = (k.saturating_sub(rhs_prep.prepared_parts.len() - 1)..min(k + 1, lhs_prep.prepared_parts.len())).map(|i| (
                &lhs[(i * self.chunk_len())..min((i + 1) * self.chunk_len(), lhs.len())],
                Some(&lhs_prep.prepared_parts[i]),
                &rhs[((k - i) * self.chunk_len())..min((k - i + 1) * self.chunk_len(), rhs.len())],
                Some(&rhs_prep.prepared_parts[k - i])
            )).collect::<Vec<_>>();
            self.base_convolution.compute_convolution_sum(&values, &mut dst[(k * self.chunk_len())..min((k + 2) * self.chunk_len(), len)], ring);
        }
    }

    fn compute_convolution_sum(&self, values: &[(&[R::Element], Option<&Self::PreparedConvolutionOperand>, &[R::Element], Option<&Self::PreparedConvolutionOperand>)], dst: &mut [R::Element], ring: &R) {
        let len = dst.len();
        for (lhs, _, rhs, _) in values {
            assert!(lhs.len() + rhs.len() <= len + 1);
        }
        let lhs_prep_owned = values.iter().map(|(lhs, lhs_prep, rhs, _)| if lhs_prep.is_none() {
            Some(self.prepare_convolution_operand(lhs, Some(lhs.len() + rhs.len()), ring))
        } else {
            None
        }).collect::<Vec<_>>();
        let lhs_prep = values.iter().zip(lhs_prep_owned.iter()).map(|((_, lhs_prep, _, _), lhs_owned)| lhs_prep.or(lhs_owned.as_ref()).unwrap());
        
        let rhs_prep_owned = values.iter().map(|(lhs, _, rhs, rhs_prep)| if rhs_prep.is_none() {
            Some(self.prepare_convolution_operand(rhs, Some(lhs.len() + rhs.len()), ring))
        } else {
            None
        }).collect::<Vec<_>>();
        let rhs_prep = values.iter().zip(rhs_prep_owned.iter()).map(|((_, _, _, rhs_prep), rhs_owned)| rhs_prep.or(rhs_owned.as_ref()).unwrap());

        for k in 0..=((len - 1) / self.chunk_len()) {
            let values = values.iter().map(|(lhs, _, rhs, _)| (lhs, rhs)).zip(lhs_prep.clone().zip(rhs_prep.clone())).flat_map(|((lhs, rhs), (lhs_prep, rhs_prep))| 
                (k.saturating_sub(rhs_prep.prepared_parts.len() - 1)..min(k + 1, lhs_prep.prepared_parts.len())).map(move |i| (
                    &lhs[(i * self.chunk_len())..min((i + 1) * self.chunk_len(), lhs.len())],
                    Some(&lhs_prep.prepared_parts[i]),
                    &rhs[((k - i) * self.chunk_len())..min((k - i + 1) * self.chunk_len(), rhs.len())],
                    Some(&rhs_prep.prepared_parts[k - i])
                )
            )).collect::<Vec<_>>();
            self.base_convolution.compute_convolution_sum(&values, &mut dst[(k * self.chunk_len())..min((k + 2) * self.chunk_len(), len)], ring);
        }
    }
}

#[cfg(test)]
use crate::algorithms::convolution::ntt::*;
#[cfg(test)]
use crate::rings::zn::*;

#[test]
fn test_convolution() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = zn_64b::Zn64B::new(65537);
    let base_convolution = NTTConvolution::new(ring);
    for l in [2, 3, 4, 8] {
        super::generic_tests::test_convolution(LengthExtendedConvolution::new(&base_convolution, l), &ring, ring.one());
    }
}