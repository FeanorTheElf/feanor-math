use tracing::instrument;

use crate::algorithms::convolution::DynConvolution;
use crate::algorithms::poly_gcd::finite::{
    fast_poly_eea, poly_power_decomposition_finite_field, poly_squarefree_part_finite_field,
};
use crate::homomorphism::Identity;
use crate::prelude::*;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::poly::*;

/// Contains an implementation of polynomial gcd and squarefree decomposition over finite fields.
pub mod finite;
pub mod gcd_lift;
/// Contains an implementation of polynomial gcd and squarefree decomposition over the integers.
pub mod integer;
/// Contains an implementation of polynomial gcd and squarefree decomposition over a number field.
pub mod numberfield;
pub mod power_decomposition_lift;

/// Trait for rings R, whose total ring of fractions `TFrac(R)` gives rise to a well-defined and
/// efficiently computable notion of the gcd of univariate polynomials over `TFrac(R)`.
/// of fractions.
///
/// Reminder: The total ring of fractions is the ring extension in which every non-zero
/// divisor is a unity, or equivalently the localization of R at all non-zero divisors
/// of R.
///
/// However, computations in `TFrac(R)` are avoided by most implementations due to
/// performance reasons, and both inputs and outputs are polynomials over `R`. Despite
/// this, the gcd is the gcd over `TFrac(R)` and not `R` (the gcd over `R` is often not
/// even defined, since `R` does not have to be UFD).
///
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::algorithms::poly_gcd::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// let ZZX = DensePolyRing::new(ZZi64, "X");
/// let [f, g, expected] =
///     ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X + 1, X.pow_ref(2) - 1, X - 1]);
/// assert_el_eq!(&ZZX, expected, <_ as PolyTFracGCDRing>::gcd(&ZZX, &f, &g));
/// ```
///
/// # Implementation notes
///
/// Efficient implementations for polynomial gcds are often quite complicated, since the standard
/// euclidean algorithm is only efficient over finite fields, where no coefficient explosion
/// happens. The general idea for other rings/fields is to reduce it to the finite case, by
/// considering the situation modulo a finite-index ideal. The requirements for this approach are
/// defined by the trait [`PolyLiftFactorsDomain`], and there is a blanket impl `R: PolyTFracGCDRing
/// where R: PolyLiftFactorsDomain`.
///
/// Note that this blanket impl used
/// [`crate::ring_properties::specialization::FiniteRingSpecializable`] to use the
/// standard algorithm whenever the corresponding ring is actually finite. In other words, despite
/// the fact that the blanket implementation for `PolyLiftFactorsDomain`s also applies to finite
/// fields, the local implementation is not actually used in these cases.
pub trait PolyTFracGCDRing {
    /// Computes the square-free part of a polynomial `f`, which is the largest-degree squarefree
    /// polynomial `d` such that `d | a f` for some non-zero-divisor `a` of this ring.
    ///
    /// This value is unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that it be monic, which makes it unique.
    ///
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::algorithms::poly_gcd::*;
    /// # use feanor_math::rings::poly::*;
    /// # use feanor_math::rings::poly::dense_poly::*;
    /// # use feanor_math::primitive_int::*;
    /// let ZZX = DensePolyRing::new(ZZi64, "X");
    /// let [f] = ZZX.with_wrapped_indeterminate(|X| [1 - X.pow_ref(2)]);
    /// assert_el_eq!(
    ///     &ZZX,
    ///     &f,
    ///     <_ as PolyTFracGCDRing>::squarefree_part(&ZZX, &ZZX.mul_ref(&f, &f))
    /// );
    /// ```
    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        poly_ring.prod(Self::power_decomposition(poly_ring, poly).into_iter().map(|(f, _)| f))
    }

    fn is_squarefree<P>(poly_ring: P, poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        poly_ring.degree(poly) == poly_ring.degree(&Self::squarefree_part(poly_ring, poly))
    }

    /// Compute square-free polynomials `f1, f2, ...` such that `a f = f1 f2^2 f3^3 ...`
    /// for some non-zero-divisor `a` of this ring. They are returned as tuples `(fi, i)`
    /// where `deg(fi) > 0`.
    ///
    /// These values are unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that all `fi` be monic, which makes them unique.
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>;

    /// Computes the greatest common divisor of two polynomials `f, g` over the fraction field,
    /// which is the largest-degree polynomial `d` such that `d | a f, a g` for some
    /// non-zero-divisor `a` of this ring.
    ///
    /// This value is unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that it be monic, which makes it unique.
    ///
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::algorithms::poly_gcd::*;
    /// # use feanor_math::rings::poly::*;
    /// # use feanor_math::rings::poly::dense_poly::*;
    /// # use feanor_math::primitive_int::*;
    /// let ZZX = DensePolyRing::new(ZZi64, "X");
    /// let [f, g, expected] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X + 1, 2 * X.pow_ref(2) - 2, X - 1]);
    /// // note that `expected` is not the gcd over `ZZ[X]` (which would be `2 X - 2`), but `X - 1`, i.e. the (monic) gcd over `QQ[X]`
    /// assert_el_eq!(&ZZX, expected, <_ as PolyTFracGCDRing>::gcd(&ZZX, &f, &g));
    ///
    /// // of course, the result does not have to be monic
    /// let [f, g, expected] = ZZX.with_wrapped_indeterminate(|X| [4 * X.pow_ref(2) - 1, 4 * X.pow_ref(2) - 4 * X + 1, - 2 * X + 1]);
    /// assert_el_eq!(&ZZX, expected, <_ as PolyTFracGCDRing>::gcd(&ZZX, &f, &g));
    /// ```
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>;
}

/// Computes the map
/// ```text
///   R[X] -> R[X],  f(X) -> a^(deg(f) - 1) f(X / a)
/// ```
/// that can be used to make polynomials over a domain monic (when setting `a = lc(f)`).
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn evaluate_aX<P>(poly_ring: P, f: &El<P>, a: &El<BaseRingStore<P>>) -> El<P>
where
    P: RingStore,
    P::Ring: PolyRing,
    <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing + Domain,
{
    if poly_ring.is_zero(f) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&f).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| {
        if i == d {
            (ring.checked_div(c, a).unwrap(), d)
        } else {
            (ring.mul_ref_fst(c, ring.pow(a.clone(), d - i - 1)), i)
        }
    }));
    return result;
}

/// Computes the inverse to [`evaluate_aX()`].
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn unevaluate_aX<P>(poly_ring: P, g: &El<P>, a: &El<BaseRingStore<P>>) -> El<P>
where
    P: RingStore,
    P::Ring: PolyRing,
    <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing + Domain,
{
    if poly_ring.is_zero(g) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&g).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(g).map(|(c, i)| {
        if i == d {
            (ring.mul_ref(c, a), d)
        } else {
            (ring.checked_div(c, &ring.pow(a.clone(), d - i - 1)).unwrap(), i)
        }
    }));
    return result;
}

/// Given a polynomial `f` over a PID, returns `(f/cont(f), cont(f))`, where `cont(f)`
/// is the content of `f`, i.e. the gcd of all coefficients of `f`.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn make_primitive<P>(poly_ring: P, f: &El<P>) -> (El<P>, El<BaseRingStore<P>>)
where
    P: RingStore,
    P::Ring: PolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing + Domain,
{
    if poly_ring.is_zero(f) {
        return (poly_ring.zero(), poly_ring.base_ring().one());
    }
    let ring = poly_ring.base_ring();
    let content = poly_ring
        .terms(f)
        .map(|(c, _)| c)
        .fold(ring.zero(), |a, b| ring.ideal_gen(&a, b));
    let result = poly_ring.from_terms(
        poly_ring
            .terms(f)
            .map(|(c, i)| (ring.checked_div(c, &content).unwrap(), i)),
    );
    return (result, content);
}

impl<R> PolyTFracGCDRing for R
where
    R: ?Sized + FiniteRing + Field,
{
    default fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        let (to, from) = make_poly_ring_euclidean(&poly_ring);
        poly_power_decomposition_finite_field(to.codomain(), &to.map_ref(poly))
            .into_iter()
            .map(|(f, e)| (from.map(f), e))
            .collect()
    }

    default fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        let (to, from) = make_poly_ring_euclidean(&poly_ring);
        from.map(
            to.codomain()
                .normalize(fast_poly_eea(to.codomain(), to.map_ref(lhs), to.map_ref(rhs)).2)
                .0,
        )
    }

    default fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        let (to, from) = make_poly_ring_euclidean(&poly_ring);
        from.map(poly_squarefree_part_finite_field(to.codomain(), &to.map_ref(poly)))
    }
}

fn make_poly_ring_euclidean<'a, P>(
    poly_ring: &'a P,
) -> (
    CoefficientHom<
        &'a P,
        DensePolyRing<&'a BaseRingStore<P>, DynConvolution<'a, BaseRingBase<P>>>,
        Identity<&'a BaseRingStore<P>>,
    >,
    CoefficientHom<
        DensePolyRing<&'a BaseRingStore<P>, DynConvolution<'a, BaseRingBase<P>>>,
        &'a P,
        Identity<&'a BaseRingStore<P>>,
    >,
)
where
    P: RingStore,
    P::Ring: PolyRing,
    BaseRingBase<P>: Field + PolyTFracGCDRing,
{
    let new_poly_ring = DensePolyRing::new(poly_ring.base_ring(), "X");
    let to = new_poly_ring
        .clone()
        .into_lifted_hom(poly_ring, poly_ring.base_ring().identity());
    let from = poly_ring.into_lifted_hom(new_poly_ring.clone(), poly_ring.base_ring().identity());
    return (to, from);
}

#[cfg(test)]
use crate::ring_impls::extension::galois_field::GaloisField;
#[cfg(test)]
use crate::ring_impls::zn::ZnRingStore;
#[cfg(test)]
use crate::ring_impls::zn::zn_64b;

#[test]
fn test_poly_gcd_galois_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = GaloisField::new(5, 3);
    let poly_ring = DensePolyRing::new(&field, "X");
    let [f, g, f_g_gcd] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            (X.pow_ref(2) + 2) * (X.pow_ref(5) + 1),
            (X.pow_ref(2) + 2) * (X + 1) * (X + 2),
            (X.pow_ref(2) + 2) * (X + 1),
        ]
    });
    assert_el_eq!(&poly_ring, &f_g_gcd, <_ as PolyTFracGCDRing>::gcd(&poly_ring, &f, &g));
}

#[test]
fn test_poly_gcd_prime_field() {
    feanor_tracing::DelayedLogger::init_test();
    let field = zn_64b::Zn64B::new(5).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(&field, "X");
    let [f, g, f_g_gcd] = poly_ring.with_wrapped_indeterminate(|X| {
        [
            (X.pow_ref(2) + 2) * (X.pow_ref(5) + 1),
            (X.pow_ref(2) + 2) * (X + 1) * (X + 2),
            (X.pow_ref(2) + 2) * (X + 1),
        ]
    });
    assert_el_eq!(&poly_ring, &f_g_gcd, <_ as PolyTFracGCDRing>::gcd(&poly_ring, &f, &g));
}
