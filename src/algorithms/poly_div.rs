use crate::ring::*;
use crate::rings::poly::*;

use tracing::instrument;

use std::cmp::max;

///
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
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_div_rem<P, F, E>(poly_ring: P, mut lhs: El<P>, rhs: &El<P>, mut left_div_lc: F) -> Result<(El<P>, El<P>), E>
    where P: RingStore,
        P::Type: PolyRing,
        F: FnMut(&El<BaseRing<P>>) -> Result<El<BaseRing<P>>, E>
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
    let result = poly_ring.try_from_terms(
        (0..(lhs_deg + 1 - rhs_deg)).rev().map(|i| {
            let quo = left_div_lc(poly_ring.coefficient_at(&lhs, i +  rhs_deg))?;
            let neg_quo = poly_ring.base_ring().negate(quo);
            if !poly_ring.base_ring().is_zero(&neg_quo) {
                poly_ring.get_ring().add_assign_from_terms(
                    &mut lhs, 
                    poly_ring.terms(rhs).map(|(c, j)| 
                        (poly_ring.base_ring().mul_ref(&neg_quo, c), i + j)
                    )
                );
            }
            Ok((poly_ring.base_ring().negate(neg_quo), i))
        })
    )?;
    return Ok((result, lhs));
}

///
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
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_rem<P, F, E>(poly_ring: P, mut lhs: El<P>, rhs: &El<P>, mut left_div_lc: F) -> Result<El<P>, E>
    where P: RingStore,
        P::Type: PolyRing,
        F: FnMut(&El<BaseRing<P>>) -> Result<El<BaseRing<P>>, E>
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
        let quo = left_div_lc(poly_ring.coefficient_at(&lhs, i +  rhs_deg))?;
        let neg_quo = poly_ring.base_ring().negate(quo);
        if !poly_ring.base_ring().is_zero(&neg_quo) {
            poly_ring.get_ring().add_assign_from_terms(
                &mut lhs, 
                poly_ring.terms(rhs).map(|(c, j)| 
                    (poly_ring.base_ring().mul_ref(&neg_quo, c), i + j)
                )
            );
        }
    }
    return Ok(lhs);
}

#[stability::unstable(feature = "enable")]
pub const FAST_POLY_DIV_THRESHOLD: usize = 32;

///
/// Computes the polynomial division of `lhs` by `rhs`, i.e. `lhs = q * rhs + r` with
/// `deg(r) < deg(rhs)`, i.e. is functionally equivalent to [`poly_div_rem()`].
/// 
/// As opposed to [`poly_div_rem()`], this function uses a fast polynomial division algorithm,
/// which is faster for large inputs.
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn fast_poly_div_rem<P, F, E>(poly_ring: P, f: El<P>, g: &El<P>, mut left_div_lc: F)-> Result<(El<P>, El<P>), E>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        F: FnMut(&El<BaseRing<P>>) -> Result<El<BaseRing<P>>, E>
{
    fn fast_poly_div_impl<P, F, E>(poly_ring: P, f: El<P>, g: &El<P>, left_div_lc: &mut F)-> Result<(El<P>, El<P>), E>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            F: FnMut(&El<BaseRing<P>>) -> Result<El<BaseRing<P>>, E>
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

        let f_upper = poly_ring.from_terms(poly_ring.terms(&f).filter(|(_, i)| *i >= split_degree_f).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_degree_f)));
        let mut f_lower = f;
        poly_ring.truncate_monomials(&mut f_lower, split_degree_f);
        let g_upper = poly_ring.from_terms(poly_ring.terms(&g).filter(|(_, i)| *i >= split_degree_g).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i - split_degree_g)));
        let mut g_lower = poly_ring.clone_el(g);
        poly_ring.truncate_monomials(&mut g_lower, split_degree_g);

        let (q_upper, r) = fast_poly_div_impl(poly_ring, poly_ring.clone_el(&f_upper), &g_upper, &mut *left_div_lc)?;
        debug_assert!(poly_ring.degree(&q_upper).is_none() || poly_ring.degree(&q_upper).unwrap() <= deg_f + split_degree_g - split_degree_f - deg_g);
        debug_assert!(poly_ring.degree(&r).is_none() || poly_ring.degree(&r).unwrap() <= deg_g - split_degree_g - 1);

        poly_ring.get_ring().add_assign_from_terms(&mut f_lower, poly_ring.terms(&r).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i + split_degree_f)));
        debug_assert!(poly_ring.degree(&f_lower).is_none() || poly_ring.degree(&f_lower).unwrap() <= deg_g + split_degree_f - split_degree_g);
        poly_ring.mul_assign_ref(&mut g_lower, &q_upper);
        poly_ring.get_ring().add_assign_from_terms(&mut f_lower, poly_ring.terms(&g_lower).map(|(c, i)| (poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(c)), i + split_degree_f - split_degree_g)));
        debug_assert!(poly_ring.degree(&f_lower).is_none() || poly_ring.degree(&f_lower).unwrap() <= max(deg_f + split_degree_g - deg_g, deg_g + split_degree_f - split_degree_g));

        let (mut q_lower, r) = fast_poly_div_impl(poly_ring, poly_ring.clone_el(&f_lower), g, &mut *left_div_lc)?;

        poly_ring.get_ring().add_assign_from_terms(&mut q_lower, poly_ring.terms(&q_upper).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i + split_degree_f - split_degree_g)));
        return Ok((q_lower, r));
    }

    assert!(!poly_ring.is_zero(g));
    if poly_ring.is_zero(&f) {
        return Ok((poly_ring.zero(), f));
    }
    return fast_poly_div_impl(poly_ring, f, g, &mut left_div_lc);
}

#[cfg(test)]
use crate::integer::*;
#[cfg(test)]
use dense_poly::DensePolyRing;
#[cfg(test)]
use crate::function::no_error;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_fast_poly_div() {
    LogAlgorithmSubscriber::init_test();
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");
    let [f, g] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(80) - 1, X.pow_ref(40) - 2 * X.pow_ref(33) + X.pow_ref(21) - X + 10]);
    assert_el_eq!(&ZZX, ZZX.div_rem_monic(ZZX.clone_el(&f), &g).0, fast_poly_div_rem(&ZZX, ZZX.clone_el(&f), &g, |c| Ok(ZZ.clone_el(c))).unwrap_or_else(no_error).0);
}