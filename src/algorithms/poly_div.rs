use std::alloc::*;
use std::cmp::max;

use tracing::instrument;

use crate::algorithms::convolution::*;
use crate::prelude::*;
use crate::ring_impls::extension::poly_modulus::PolyModulus;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::poly::*;

/// Computes the polynomial division of `lhs` by `rhs`, i.e. `lhs = q * rhs + r` with
/// `deg(r) < deg(rhs)`.
///
/// Note that this function does not compute the proper polynomial division if the leading
/// coefficient of `rhs` is a zero-divisor in the ring. See [`poly_div_rem_finite_reduced()`]
/// for details.
///
/// This requires a function `left_div_lc` that computes the division of an element of the
/// base ring by the leading coefficient of `rhs`. If the base ring is a field, this can
/// just be standard division. In other cases, this depends on the exact situation you are
/// in - e.g. `rhs` might be monic or in in a specific context, it might be guaranteed that the
/// division always works. If this is not the case, look also at [`poly_div_rem_domain()`], which
/// implicitly performs the polynomial division over the field of fractions.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_div_rem<P, F, E>(poly_ring: P, mut lhs: El<P>, rhs: &El<P>, mut left_div_lc: F) -> Result<(El<P>, El<P>), E>
where
    P: RingStore,
    P::Ring: PolyRing,
    F: FnMut(&El<BaseRingStore<P>>) -> Result<El<BaseRingStore<P>>, E>,
{
    assert!(poly_ring.degree(rhs).is_some());

    let rhs_deg = poly_ring.degree(rhs).unwrap();
    if poly_ring.degree(&lhs).is_none() {
        return Ok((poly_ring.zero(), lhs));
    }
    let lhs_deg = poly_ring.degree(&lhs).unwrap();
    if lhs_deg < rhs_deg {
        return Ok((poly_ring.zero(), lhs));
    }
    let result = poly_ring.try_from_terms((0..(lhs_deg + 1 - rhs_deg)).rev().map(|i| {
        let quo = left_div_lc(poly_ring.coefficient_at(&lhs, i + rhs_deg))?;
        let neg_quo = poly_ring.base_ring().negate(quo);
        if !poly_ring.base_ring().is_zero(&neg_quo) {
            poly_ring.get_ring().add_assign_from_terms(
                &mut lhs,
                poly_ring
                    .terms(rhs)
                    .map(|(c, j)| (poly_ring.base_ring().mul_ref(&neg_quo, c), i + j)),
            );
        }
        Ok((poly_ring.base_ring().negate(neg_quo), i))
    }))?;
    return Ok((result, lhs));
}

/// Computes the remainder of the polynomial division of `lhs` by `rhs`, i.e. `r` in the
/// expression `lhs = q * rhs + r` with `deg(r) < deg(rhs)`.
///
/// As opposed to [`poly_div_rem()`], this function only computes the remainder, but may
/// be slightly faster because of this.
///
/// Note that this function does not compute the proper polynomial division if the leading
/// coefficient of `rhs` is a zero-divisor in the ring. See [`poly_div_rem_finite_reduced()`]
/// for details.
///
/// This requires a function `left_div_lc` that computes the division of an element of the
/// base ring by the leading coefficient of `rhs`. If the base ring is a field, this can
/// just be standard division. In other cases, this depends on the exact situation you are
/// in - e.g. `rhs` might be monic or in in a specific context, it might be guaranteed that the
/// division always works. If this is not the case, look also at [`poly_div_rem_domain()`], which
/// implicitly performs the polynomial division over the field of fractions.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_rem<P, F, E>(poly_ring: P, mut lhs: El<P>, rhs: &El<P>, mut left_div_lc: F) -> Result<El<P>, E>
where
    P: RingStore,
    P::Ring: PolyRing,
    F: FnMut(&El<BaseRingStore<P>>) -> Result<El<BaseRingStore<P>>, E>,
{
    assert!(poly_ring.degree(rhs).is_some());

    let rhs_deg = poly_ring.degree(rhs).unwrap();
    if poly_ring.degree(&lhs).is_none() {
        return Ok(lhs);
    }
    let lhs_deg = poly_ring.degree(&lhs).unwrap();
    if lhs_deg < rhs_deg {
        return Ok(lhs);
    }
    for i in (0..(lhs_deg + 1 - rhs_deg)).rev() {
        let quo = left_div_lc(poly_ring.coefficient_at(&lhs, i + rhs_deg))?;
        let neg_quo = poly_ring.base_ring().negate(quo);
        if !poly_ring.base_ring().is_zero(&neg_quo) {
            poly_ring.get_ring().add_assign_from_terms(
                &mut lhs,
                poly_ring
                    .terms(rhs)
                    .map(|(c, j)| (poly_ring.base_ring().mul_ref(&neg_quo, c), i + j)),
            );
        }
    }
    return Ok(lhs);
}

#[stability::unstable(feature = "enable")]
pub const FAST_POLY_DIV_THRESHOLD: usize = 32;

/// Computes the polynomial division of `lhs` by `rhs`, i.e. `lhs = q * rhs + r` with
/// `deg(r) < deg(rhs)`, i.e. is functionally equivalent to [`poly_div_rem()`].
///
/// As opposed to [`poly_div_rem()`], this function uses a fast polynomial division algorithm,
/// which is faster for large inputs.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn fast_poly_div_rem<P, F, E>(poly_ring: P, f: El<P>, g: &El<P>, mut left_div_lc: F) -> Result<(El<P>, El<P>), E>
where
    P: RingStore + Copy,
    P::Ring: PolyRing,
    F: FnMut(&El<BaseRingStore<P>>) -> Result<El<BaseRingStore<P>>, E>,
{
    fn fast_poly_div_impl<P, F, E>(poly_ring: P, f: El<P>, g: &El<P>, left_div_lc: &mut F) -> Result<(El<P>, El<P>), E>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing,
        F: FnMut(&El<BaseRingStore<P>>) -> Result<El<BaseRingStore<P>>, E>,
    {
        let deg_g = poly_ring.degree(g).unwrap();
        if poly_ring.degree(&f).is_none() || poly_ring.degree(&f).unwrap() < deg_g {
            return Ok((poly_ring.zero(), f));
        }
        let deg_f = poly_ring.degree(&f).unwrap();
        if deg_g < FAST_POLY_DIV_THRESHOLD || (deg_f - deg_g) < FAST_POLY_DIV_THRESHOLD {
            return poly_div_rem(poly_ring, f, g, left_div_lc);
        }

        let (split_degree_f, split_degree_g) = if deg_f >= 3 * deg_g {
            (deg_f / 3, 0)
        } else if 2 * (deg_f / 3) < deg_g {
            (deg_g / 2, deg_g / 2)
        } else {
            (deg_f / 3, deg_g - deg_f / 3)
        };
        assert!(split_degree_f >= split_degree_g);
        assert!(split_degree_f <= deg_f);
        assert!(split_degree_g <= deg_g);

        let f_upper = poly_ring.from_terms(
            poly_ring
                .terms(&f)
                .filter(|(_, i)| *i >= split_degree_f)
                .map(|(c, i)| (c.clone(), i - split_degree_f)),
        );
        let mut f_lower = f;
        poly_ring.truncate_monomials(&mut f_lower, split_degree_f);
        let g_upper = poly_ring.from_terms(
            poly_ring
                .terms(&g)
                .filter(|(_, i)| *i >= split_degree_g)
                .map(|(c, i)| (c.clone(), i - split_degree_g)),
        );
        let mut g_lower = g.clone();
        poly_ring.truncate_monomials(&mut g_lower, split_degree_g);

        let (q_upper, r) = fast_poly_div_impl(poly_ring, f_upper.clone(), &g_upper, &mut *left_div_lc)?;
        debug_assert!(
            poly_ring.degree(&q_upper).is_none()
                || poly_ring.degree(&q_upper).unwrap() <= deg_f + split_degree_g - split_degree_f - deg_g
        );
        debug_assert!(poly_ring.degree(&r).is_none() || poly_ring.degree(&r).unwrap() < deg_g - split_degree_g);

        poly_ring.get_ring().add_assign_from_terms(
            &mut f_lower,
            poly_ring.terms(&r).map(|(c, i)| (c.clone(), i + split_degree_f)),
        );
        debug_assert!(
            poly_ring.degree(&f_lower).is_none()
                || poly_ring.degree(&f_lower).unwrap() <= deg_g + split_degree_f - split_degree_g
        );
        poly_ring.mul_assign_ref(&mut g_lower, &q_upper);
        poly_ring.get_ring().add_assign_from_terms(
            &mut f_lower,
            poly_ring.terms(&g_lower).map(|(c, i)| {
                (
                    poly_ring.base_ring().negate(c.clone()),
                    i + split_degree_f - split_degree_g,
                )
            }),
        );
        debug_assert!(
            poly_ring.degree(&f_lower).is_none()
                || poly_ring.degree(&f_lower).unwrap()
                    <= max(deg_f + split_degree_g - deg_g, deg_g + split_degree_f - split_degree_g)
        );

        let (mut q_lower, r) = fast_poly_div_impl(poly_ring, f_lower.clone(), g, &mut *left_div_lc)?;

        poly_ring.get_ring().add_assign_from_terms(
            &mut q_lower,
            poly_ring
                .terms(&q_upper)
                .map(|(c, i)| (c.clone(), i + split_degree_f - split_degree_g)),
        );
        return Ok((q_lower, r));
    }

    assert!(!poly_ring.is_zero(g));
    if poly_ring.is_zero(&f) {
        return Ok((poly_ring.zero(), f));
    }
    return fast_poly_div_impl(poly_ring, f, g, &mut left_div_lc);
}

/// A [`PolyModulus`] that uses Barrett reduction for faster reduction.
///
/// Note that as opposed to the simple [`PolyModulus`], this requires a significant
/// amount of precomputation, but has online complexity `C(n - d + 1, n) + C(n - d, d + 1)`,
/// where `C(s, t)` is the cost of computing a convolution of a length-`s` and a length-`t`
/// sequence, `d` is the degree of the polynomial to divide by, and `n` is the length of
/// the operand.
pub struct BarrettPolyModulus<R: RingStore, C: ConvolutionAlgorithm<R::Ring>, A: Allocator> {
    ring: R,
    n: usize,
    neg_Xn_div_f: Vec<El<R>>,
    neg_Xn_div_f_prep: C::PreparedConvolutionOperand,
    f: Vec<El<R>>,
    f_prep: C::PreparedConvolutionOperand,
    f_deg: usize,
    x_pow_rank: Vec<El<R>>,
    convolution: C,
    allocator: A,
}

impl<R: RingStore, C: ConvolutionAlgorithm<R::Ring> + Clone, A: Allocator> BarrettPolyModulus<R, C, A> {
    pub fn new(ring: R, operand_deg: usize, x_pow_rank: Vec<El<R>>, convolution: C, allocator: A) -> Self {
        let f_deg = x_pow_rank.len();
        let n = usize::max(1, usize::max(operand_deg + 1, f_deg));
        let poly_ring = DensePolyRing::new(&ring, "X");
        let f = poly_ring.from_terms(
            (0..f_deg)
                .map(|i| (ring.negate(x_pow_rank[i].clone()), i))
                .chain([(ring.one(), f_deg)]),
        );
        let neg_Xn = poly_ring.from_terms([(ring.neg_one(), n)]);
        let neg_Xn_div_f = poly_ring.div_rem_monic(neg_Xn, &f).0;
        let neg_Xn_div_f = (0..=poly_ring.degree(&neg_Xn_div_f).unwrap_or(0))
            .map(|i| poly_ring.coefficient_at(&neg_Xn_div_f, i).clone())
            .collect::<Vec<_>>();
        debug_assert_eq!(n + 1 - f_deg, neg_Xn_div_f.len());
        let neg_Xn_div_f_prep =
            convolution.prepare_convolution_operand(&neg_Xn_div_f, Some(2 * n - f_deg), ring.get_ring());
        let f = (0..=poly_ring.degree(&f).unwrap_or(0))
            .map(|i| poly_ring.coefficient_at(&f, i).clone())
            .collect::<Vec<_>>();
        debug_assert_eq!(f_deg + 1, f.len());
        let f_prep = convolution.prepare_convolution_operand(&f, Some(n), ring.get_ring());
        drop(poly_ring);
        Self {
            n,
            neg_Xn_div_f,
            neg_Xn_div_f_prep,
            f,
            f_prep,
            f_deg,
            x_pow_rank,
            convolution,
            ring,
            allocator,
        }
    }
}

impl<R: RingStore, C: ConvolutionAlgorithm<R::Ring> + Clone, A: Allocator + Clone> Clone
    for BarrettPolyModulus<R, C, A>
{
    fn clone(&self) -> Self {
        let neg_Xn_div_f_prep = self.convolution.prepare_convolution_operand(
            &self.neg_Xn_div_f,
            Some(2 * self.n - self.f_deg),
            self.ring.get_ring(),
        );
        let poly_prep = self
            .convolution
            .prepare_convolution_operand(&self.f, Some(self.n), self.ring.get_ring());
        Self {
            ring: self.ring.clone(),
            n: self.n,
            neg_Xn_div_f: self.neg_Xn_div_f.clone(),
            neg_Xn_div_f_prep,
            f: self.f.clone(),
            f_prep: poly_prep,
            f_deg: self.f_deg,
            x_pow_rank: self.x_pow_rank.clone(),
            convolution: self.convolution.clone(),
            allocator: self.allocator.clone(),
        }
    }
}

impl<R: RingStore, C: ConvolutionAlgorithm<R::Ring>, A: Allocator> PolyModulus<R> for BarrettPolyModulus<R, C, A> {
    fn degree(&self) -> usize { self.f_deg }

    fn ring(&self) -> &R { &self.ring }

    fn supported_operand_degree(&self) -> usize { self.n - 1 }

    fn x_pow_rank(&self) -> &[El<R>] { &self.x_pow_rank }

    #[instrument(skip_all, level = "trace")]
    fn perform_reduction(&self, operand: &mut [El<R>]) {
        assert!(operand.len() <= self.supported_operand_degree() + 1);
        let mut data = (0..(2 * self.n - self.f_deg + 2))
            .map(|_| self.ring.zero())
            .collect::<Vec<_>>();
        let quotient = &mut data[1..];
        debug_assert_eq!(
            self.neg_Xn_div_f.len() + self.supported_operand_degree() + 1,
            quotient.len()
        );
        self.convolution.compute_convolution(
            &operand,
            None,
            &self.neg_Xn_div_f,
            Some(&self.neg_Xn_div_f_prep),
            quotient,
            self.ring.get_ring(),
        );
        let (tmp, quotient) = data[..(2 * self.n - self.f_deg + 1)].split_at_mut(self.n + 1);
        debug_assert_eq!(self.f_deg + quotient.len() + 1, tmp.len());
        for i in 0..tmp.len() {
            tmp[i] = self.ring.zero();
        }
        self.convolution.compute_convolution(
            quotient,
            None,
            &self.f,
            Some(&self.f_prep),
            &mut *tmp,
            self.ring.get_ring(),
        );
        for (i, x) in data.into_iter().take(usize::min(self.f_deg, operand.len())).enumerate() {
            self.ring.add_assign(&mut operand[i], x);
        }
    }
}

#[cfg(test)]
use crate::function::no_error;
#[cfg(test)]
use crate::ring_impls::zn::zn_64b::Zn64B;

#[test]
fn test_fast_poly_div() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let ZZX = DensePolyRing::new(ZZ, "X");
    let [f, g] = ZZX.with_wrapped_indeterminate(|X| {
        [
            X.pow_ref(80) - 1,
            X.pow_ref(40) - 2 * X.pow_ref(33) + X.pow_ref(21) - X + 10,
        ]
    });
    assert_el_eq!(
        &ZZX,
        ZZX.div_rem_monic(f.clone(), &g).0,
        fast_poly_div_rem(&ZZX, f.clone(), &g, |c| Ok(c.clone()))
            .unwrap_or_else(no_error)
            .0
    );
}

#[test]
fn test_barrett_poly_reduction() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = Zn64B::new(65537);
    // f = X^3 - 2 X^2 - 1
    let x_pow_rank = [1, 0, 2].into_iter().map(|x| ring.int_hom().map(x)).collect::<Vec<_>>();
    let reducer = BarrettPolyModulus::new(ring, 9, x_pow_rank, KaratsubaAlgorithm::new(4), Global);

    let mut operand = [4, 3, 2, 1]
        .into_iter()
        .map(|x| ring.int_hom().map(x))
        .collect::<Vec<_>>();
    reducer.perform_reduction(&mut operand);
    for i in 0..3 {
        assert_el_eq!(&ring, ring.int_hom().map([5, 3, 4][i]), operand[i]);
    }

    let mut operand = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        .into_iter()
        .map(|x| ring.int_hom().map(x))
        .collect::<Vec<_>>();
    reducer.perform_reduction(&mut operand);
    for i in 0..3 {
        assert_el_eq!(&ring, ring.int_hom().map([330, 152, 711][i]), operand[i]);
    }
}
