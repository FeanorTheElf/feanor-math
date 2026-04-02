use std::cmp::min;
use std::marker::PhantomData;

use crate::algorithms::convolution::ntt::NTTConvolution;
use crate::algorithms::convolution::*;
use crate::algorithms::cyclotomic::get_prim_root_of_unity_pow2_zn;
use crate::homomorphism::*;
use crate::integer::*;
use crate::primitive_int::*;
use crate::rings::zn::*;

pub struct LengthExtendedConvolution<C> {
    base_convolution: C,
    base_max_len: usize,
}

pub struct PreparedConvolutionOperand<R, C>
where
    R: ?Sized + RingBase,
    C: ConvolutionAlgorithm<R>,
{
    prepared_parts: Vec<C::PreparedConvolutionOperand>,
    convolution: PhantomData<C>,
    ring: PhantomData<R>,
}

impl<C> LengthExtendedConvolution<C> {
    pub fn new(base_convolution: C, base_max_len: usize) -> Self {
        assert!(base_max_len >= 2);
        Self {
            base_convolution,
            base_max_len,
        }
    }

    fn chunk_len(&self) -> usize { self.base_max_len / 2 }
}

impl<R> LengthExtendedConvolution<NTTConvolution<R::Type, R::Type, Identity<R>>>
where
    R: RingStore,
    R::Type: ZnRing,
{
    /// Constructs a [`LengthExtendedConvolution`], which will use a suitable, NTT-based convolution
    /// over the given ring.
    ///
    /// If the given ring doesn't support NTT-based convolutions of length at least
    /// `abort_if_ntt_len_le`, then `Err(())` will be returned. Note that this, in particular,
    /// includes the case that the characteristic of the base ring is even (and no power-of-two
    /// primitive roots of unity exist at all).
    pub fn for_zn(ring: R, abort_if_ntt_len_le: usize) -> Result<Self, ()> {
        if BigIntRing::RING.is_even(&ring.characteristic(BigIntRing::RING).unwrap()) {
            return Err(());
        }
        let mut log2_len = StaticRing::<i64>::RING
            .abs_log2_ceil(&abort_if_ntt_len_le.try_into().unwrap())
            .unwrap();
        if get_prim_root_of_unity_pow2_zn(&ring, log2_len).is_none() {
            return Err(());
        }
        for _ in 0..20 {
            if get_prim_root_of_unity_pow2_zn(&ring, log2_len + 1).is_some() {
                log2_len += 1;
            } else {
                break;
            }
        }
        let base_convolution = NTTConvolution::new_with_hom(ring.into_identity(), Global);
        return Ok(Self::new(base_convolution, 1 << log2_len));
    }
}

impl<R>
    LengthExtendedConvolution<
        NTTConvolution<R::Type, <<R::Type as RingExtension>::BaseRing as RingStore>::Type, Inclusion<R>>,
    >
where
    R: RingStore,
    R::Type: RingExtension,
    <<R::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
{
    /// Constructs a [`LengthExtendedConvolution`], which will use a suitable, NTT-based convolution
    /// over the base ring of the given.
    ///
    /// If the given ring doesn't support NTT-based convolutions of length at least
    /// `abort_if_ntt_len_le`, then `Err(())` will be returned. Note that this, in particular,
    /// includes the case that the characteristic of the base ring is even (and no power-of-two
    /// primitive roots of unity exist at all).
    pub fn for_zn_extension(ring: R, abort_if_ntt_len_le: usize) -> Result<Self, ()> {
        let ring_incl = ring.into_inclusion();
        let result = LengthExtendedConvolution::for_zn(ring_incl.domain().clone(), abort_if_ntt_len_le)?;
        Ok(Self::new(
            result.base_convolution.change_ring(ring_incl),
            result.base_max_len,
        ))
    }
}

impl<R, C> ConvolutionAlgorithm<R> for LengthExtendedConvolution<C>
where
    R: ?Sized + RingBase,
    C: ConvolutionAlgorithm<R>,
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, C>;

    fn supports_ring(&self, ring: &R) -> bool { self.base_convolution.supports_ring(ring) }

    fn prepare_convolution_operand(
        &self,
        val: &[R::Element],
        length_hint: Option<usize>,
        ring: &R,
    ) -> Self::PreparedConvolutionOperand {
        let base_length_hint = match length_hint {
            None => None,
            Some(len) if len < val.len() => {
                panic!("length_hint cannot be smaller than the length of a single convolution operand")
            }
            Some(len) if len < self.base_max_len => Some(len),
            Some(_) => Some(self.base_max_len),
        };
        PreparedConvolutionOperand {
            prepared_parts: val
                .chunks(self.chunk_len())
                .map(|chunk| {
                    self.base_convolution
                        .prepare_convolution_operand(chunk, base_length_hint, ring)
                })
                .collect(),
            ring: PhantomData,
            convolution: PhantomData,
        }
    }

    fn compute_convolution(
        &self,
        lhs: &[R::Element],
        lhs_prep: Option<&Self::PreparedConvolutionOperand>,
        rhs: &[R::Element],
        rhs_prep: Option<&Self::PreparedConvolutionOperand>,
        dst: &mut [R::Element],
        ring: &R,
    ) {
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
            let values = (k.saturating_sub(rhs_prep.prepared_parts.len() - 1)
                ..min(k + 1, lhs_prep.prepared_parts.len()))
                .map(|i| {
                    (
                        &lhs[(i * self.chunk_len())..min((i + 1) * self.chunk_len(), lhs.len())],
                        Some(&lhs_prep.prepared_parts[i]),
                        &rhs[((k - i) * self.chunk_len())..min((k - i + 1) * self.chunk_len(), rhs.len())],
                        Some(&rhs_prep.prepared_parts[k - i]),
                    )
                })
                .collect::<Vec<_>>();
            self.base_convolution.compute_convolution_sum(
                &values,
                &mut dst[(k * self.chunk_len())..min((k + 2) * self.chunk_len(), len)],
                ring,
            );
        }
    }

    fn compute_convolution_sum(
        &self,
        values: &[(
            &[R::Element],
            Option<&Self::PreparedConvolutionOperand>,
            &[R::Element],
            Option<&Self::PreparedConvolutionOperand>,
        )],
        dst: &mut [R::Element],
        ring: &R,
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

        let lhs_prep_owned = values
            .iter()
            .map(|(lhs, lhs_prep, ..)| {
                if lhs_prep.is_none() {
                    Some(self.prepare_convolution_operand(lhs, Some(len), ring))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let lhs_prep = values
            .iter()
            .zip(lhs_prep_owned.iter())
            .map(|((_, lhs_prep, ..), lhs_owned)| lhs_prep.or(lhs_owned.as_ref()).unwrap());

        let rhs_prep_owned = values
            .iter()
            .map(|(_, _, rhs, rhs_prep)| {
                if rhs_prep.is_none() {
                    Some(self.prepare_convolution_operand(rhs, Some(len), ring))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let rhs_prep = values
            .iter()
            .zip(rhs_prep_owned.iter())
            .map(|((_, _, _, rhs_prep), rhs_owned)| rhs_prep.or(rhs_owned.as_ref()).unwrap());

        for k in 0..=((len - 1) / self.chunk_len()) {
            let values = values
                .iter()
                .map(|(lhs, _, rhs, _)| (lhs, rhs))
                .zip(lhs_prep.clone().zip(rhs_prep.clone()))
                .filter(|((lhs, rhs), _)| lhs.len() != 0 && rhs.len() != 0)
                .flat_map(|((lhs, rhs), (lhs_prep, rhs_prep))| {
                    (k.saturating_sub(rhs_prep.prepared_parts.len() - 1)..min(k + 1, lhs_prep.prepared_parts.len()))
                        .map(move |i| {
                            (
                                &lhs[(i * self.chunk_len())..min((i + 1) * self.chunk_len(), lhs.len())],
                                Some(&lhs_prep.prepared_parts[i]),
                                &rhs[((k - i) * self.chunk_len())..min((k - i + 1) * self.chunk_len(), rhs.len())],
                                Some(&rhs_prep.prepared_parts[k - i]),
                            )
                        })
                })
                .collect::<Vec<_>>();
            self.base_convolution.compute_convolution_sum(
                &values,
                &mut dst[(k * self.chunk_len())..min((k + 2) * self.chunk_len(), len)],
                ring,
            );
        }
    }
}

#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use crate::rings::zn::zn_64b::Zn64B;
#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
fn test_convolution() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = zn_64b::Zn64B::new(65537);
    let base_convolution = NTTConvolution::new(ring);
    for l in [2, 3, 4, 8] {
        super::generic_tests::test_convolution(LengthExtendedConvolution::new(&base_convolution, l), &ring, ring.one());
    }
}

#[test]
fn test_for_zn() {
    feanor_tracing::DelayedLogger::init_test();
    let field = Zn64B::new(257);
    let convolution = LengthExtendedConvolution::for_zn(&field, 64).unwrap();
    assert_eq!(256, convolution.base_max_len);
    super::generic_tests::test_convolution(convolution, &field, field.one());

    let ring = Zn::<{ 17 * 5 }>::RING;
    let convolution = LengthExtendedConvolution::for_zn(&ring, 4).unwrap();
    assert_eq!(4, convolution.base_max_len);
    super::generic_tests::test_convolution(convolution, &ring, ring.one());

    assert!(LengthExtendedConvolution::for_zn(&ring, 8).is_err());
}

#[test]
fn test_for_zn_extension() {
    feanor_tracing::DelayedLogger::init_test();
    let galois_field = GaloisField::new(257, 2);
    let convolution = LengthExtendedConvolution::for_zn_extension(&galois_field, 64).unwrap();
    assert_eq!(256, convolution.base_max_len);
    super::generic_tests::test_convolution(convolution, &galois_field, galois_field.one());

    let base_ring = Zn::<{ 17 * 5 }>::RING;
    let ring = FreeAlgebraImpl::new(base_ring, 2, [2]);
    let convolution = LengthExtendedConvolution::for_zn_extension(&ring, 4).unwrap();
    assert_eq!(4, convolution.base_max_len);
    super::generic_tests::test_convolution(convolution, &ring, ring.one());

    assert!(LengthExtendedConvolution::for_zn_extension(&ring, 8).is_err());
}
