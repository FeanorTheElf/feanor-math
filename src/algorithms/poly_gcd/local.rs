use std::fmt::Display;

use crate::divisibility::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::primitive_int::*;
use crate::ring::*;
use crate::rings::field::*;
use crate::rings::poly::*;
use crate::rings::zn::*;
use crate::rings::zn::zn_big;

use crate::algorithms::miller_rabin::next_prime;
use crate::algorithms::poly_factor::FactorPolyField;

use super::PolyGCDRing;

///
/// Trait for rings that support lifting local partial factorizations of polynomials to the ring.
/// For infinite fields, this is the most important approach to computing gcd's.
/// 
/// The general philosophy is similar to [`crate::compute_locally::EvaluatePolyLocallyRing`].
/// However, when working with factorizations, it is usually better to compute modulo a power
/// `p^e` of a maximal ideal `p` and use Hensel lifting. `EvaluatePolyLocallyRing` on the other
/// hand is designed to allow computations modulo multiple different maximal ideals.
/// 
/// More concretely, a ring `R` implementing this trait should be endowed with a "pseudo norm"
/// ```text
///   |.|: R  ->  [0, ∞)
/// ```
/// i.e. a symmetric, sub-additive, sub-multiplicative map.
/// Furthermore, for any bound `B` and any maximal ideal `p`, there should be a positive integer
/// `e` such that the restriction of the reduction map
/// ```text
///   { x in R | |x| <= B }  ->  R / p^e
/// ```
/// is injective.
/// This means that a (possibly partial) factorization in the simpler ring `R / p^e`,
/// can be uniquely mapped back into `R` - assuming all coefficients of the actual factors 
/// have pseudo-norm at most `b` and lie within the ring `R`. 
/// 
/// Because of this, it must be possible to give a bound on the `l∞`-pseudo norm of any monic factor 
/// of a monic polynomial `f`. Previously I thought it would be nice to be able to explicitly compute
/// this bound, but actually this is hard in some cases, and also in many cases, much smaller bounds
/// are sufficient in practice - hence taking the "worst-case" bound gives very poor performance. 
/// Thus the idea is to use progressively higher bounds, corresponding to progressively larger exponents `e`.
/// 
/// Unfortunately, this makes this trait mostly unsuitable to compute factorizations.
/// In particular, by choosing larger `e`, we are able to find all factors eventually, but
/// we can never know when we are done. More concretely, assuming the factorization of `h` mod `p^e`
/// gives some irreducible factors `f1, ..., fk` such that `f = f1 ... fk` lifts to a factor of `h`.
/// Then, it might be the case that `f` is irreducible over `R`, but it might also be that a larger
/// power of `p^e` will give us a more fine-grained factor, say `f' = f1 ... fl` with `l < k`.
/// Without an explicit irreducibility check (over `R`), we cannot distinguish these cases, and
/// if we don't have an upper bound on `e`, we never know if we still have to consider factorizations
/// modulo larger powers of `p`.
/// 
/// # Mathematical examples
/// 
/// There are three main classes of rings that this trait is designed for
///  - The integers `ZZ`. By Gauss' lemma, we know that any factor of an integral polynomial
///    is again integral, and we can bound its size (using the absolute value `|.|` of the real
///    numbers) in terms of size and degree of the original polynomial. 
///    This motivates the main algorithm for factoring over `Z`: Reduce modulo a prime
///    `p`, lift the factorization to `p^e` for a large enough `p`, and (more or less) read of
///    the factors over `Z`.
///  - Multivariate polynomial rings `R[X1, ..., Xm]` where `R` is a finite field (or another ring
///    where we can efficiently compute polynomial factorizations, e.g. another [`PolyGCDLocallyDomain`]).
///    For simplicity, let's focus on the finite field case, i.e. `R = k`. Then we can take a maximal
///    ideal `m` of `k[X1, ..., Xm]`, and reduce a polynomial `f in k[X1, ..., Xm][Y]` modulo `m`, compute
///    its factorization there (i.e. over `k`), lift it to `k[X1, ..., Xm]/m^e` and (more or less) read
///    of the factors over `R[X1, ..., Xm]`.
///  - Orders in algebraic number fields. This case is somewhat more complicated, since (in general) we
///    don't have a UFD anymore. Concretely, the factors of a polynomial with coefficients in an order `R`
///    don't necessarily have coefficients in `R`. Example: `X^2 - sqrt(3) X - 1` over `Z[sqrt(3), sqrt(7)]`
///    has the factor `X - (sqrt(3) + sqrt(7)) / 2`.
///    However, it turns out that if the original polynomial is monic, then its factors have coefficients in
///    the maximal order `O` of `R ⊗ QQ`. In particular, if we scale the factor by `[R : O] | disc(R)`, then
///    we do end up with coefficients in `R`. This is why we allow for the function
///    [`PolyGCDLocallyDomain::factor_scaling()`].
///    Furthermore, the "pseudo-norm" here should be the canonical norm.
/// 
/// I cannot think of any other good examples (these were the ones I had in mind when writing this trait), but 
/// who knows, maybe there are other rings that satisfy this and which we can thus do polynomial factorization in!
/// 
#[stability::unstable(feature = "enable")]
pub trait PolyGCDLocallyDomain: Domain + DivisibilityRing {

    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// see also [`crate::compute_locally::ComputeLocallyRing`]
    /// 
    type LocalRingBase<'ring>: DivisibilityRing
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;
    
    ///
    /// Again, this is required to restrict `AsFieldBase<Self::LocalRing<'ring>>: FactorPolyField`,
    /// which in turn is required since the straightforward bound `for<'a> AsFieldBase<Self::LocalRing<'a>>: FactorPolyField`
    /// is bugged, see also [`crate::compute_locally::ComputeLocallyRing`].
    /// 
    type LocalFieldBase<'ring>: CanIsoFromTo<Self::LocalRingBase<'ring>> + PolyGCDRing + FactorPolyField
        where Self: 'ring;

    type LocalField<'ring>: RingStore<Type = Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type MaximalIdeal<'ring>
        where Self: 'ring;

    ///
    /// Computes a nonzero element `a` with the property that whenever `f in R[X]` is a monic polynomial
    /// over this ring `R`, with a factor `g` over `Frac(R)` the fraction field of this ring, we
    /// have `ag in R[X]`.
    /// 
    /// If this ring is a UFD, this `a` can always be chosen as `1` by Gauss' lemma. Otherwise,
    /// this factor "measures" the "extend of failure" of Gauss' lemma.
    /// 
    fn factor_scaling(&self) -> Self::Element;

    ///
    /// Returns an exponent `e` such that we hope that the factors of `poly` can already be read of
    /// their reductions modulo `p^e`. Note that this is just a heuristic, and if it does not work,
    /// the implementation will gradually try larger `e`. Thus, even if this function returns constant
    /// 1, correctness will not be affected, but giving a good guess can improve performance
    /// 
    fn heuristic_exponent<'ring, P>(&self, _maximal_ideal: &Self::MaximalIdeal<'ring>, _poly_ring: P, _poly: &El<P>) -> usize
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Self: 'ring;

    ///
    /// Returns an ideal sampled at random from the interval of all supported maximal ideals.
    /// 
    fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
        where F: FnMut() -> u64;

    ///
    /// Returns `R / p`.
    /// 
    /// This will always be a field, since `p` is a maximal ideal.
    /// 
    fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
        where Self: 'ring;
    
    ///
    /// Returns `R / p^e`.
    /// 
    fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
        where Self: 'ring;

    ///
    /// Computes the reduction map
    /// ```text
    ///   R -> R / p^e
    /// ```
    /// 
    fn reduce_full<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes the reduction map
    /// ```text
    ///   R / p^e1 -> R / p^e2
    /// ```
    /// where `e1 >= e2`.
    /// 
    fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes any element `y` in `R / p^to_e` such that `y = x mod p^from_e`.
    /// In particular, `y` does not have to be "short" in any sense, but any lift
    /// is a valid result.
    /// 
    fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes "the" shortest element `y in `R` such that `y = x mod p^from_e`.
    /// The computed `y` does not necessarily have to be unique, but must be among the
    /// set of shortest ones, i.e.
    /// ```text
    ///   |y| = min { |z| | z = x mod p^from_e }
    /// ```
    /// 
    fn lift_full<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring;

    fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring;
}

#[stability::unstable(feature = "enable")]
pub trait IntegerPolyGCDRing: PolyGCDLocallyDomain + IntegerRing {

    fn maximal_ideal_gen<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> i64
        where Self: 'ring;
}

#[stability::unstable(feature = "enable")]
pub struct IdealDisplayWrapper<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> {
    ring: &'a R,
    ideal: &'a R::MaximalIdeal<'ring>
}

impl<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> IdealDisplayWrapper<'a, 'ring, R> {

    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'a R, ideal: &'a R::MaximalIdeal<'ring>) -> Self {
        Self {
            ring: ring, 
            ideal: ideal
        }
    }
}

impl<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> Display for IdealDisplayWrapper<'a, 'ring, R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.dbg_maximal_ideal(self.ideal, f)
    }
}

#[stability::unstable(feature = "enable")]
pub struct ReductionMap<'ring, 'data, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    p: &'data R::MaximalIdeal<'ring>,
    to: (R::LocalRing<'ring>, usize)
}

impl<'ring, 'data, R> ReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, p: &'data R::MaximalIdeal<'ring>, power: usize) -> Self {
        assert!(power >= 1);
        Self { ring: RingRef::new(ring), p: p, to: (ring.local_ring_at(p, power), power) }
    }
}

impl<'ring, 'data, R> Homomorphism<R, R::LocalRingBase<'ring>> for ReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    type CodomainStore = R::LocalRing<'ring>;
    type DomainStore = RingRef<'data, R>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to.0
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.ring
    }

    fn map(&self, x: <R as RingBase>::Element) -> <R::LocalRingBase<'ring> as RingBase>::Element {
        self.ring.get_ring().reduce_full(self.p, (&self.to.0, self.to.1), x)
    }
}

#[stability::unstable(feature = "enable")]
pub struct IntermediateReductionMap<'ring, 'data, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    p: &'data R::MaximalIdeal<'ring>,
    from: (R::LocalRing<'ring>, usize),
    to: (R::LocalRing<'ring>, usize)
}

impl<'ring, 'data, R> IntermediateReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, p: &'data R::MaximalIdeal<'ring>, from: usize, to: usize) -> Self {
        assert!(from >= to);
        assert!(to >= 1);
        Self { ring: RingRef::new(ring), p: p, from: (ring.local_ring_at(p, from), from), to: (ring.local_ring_at(p, to), to) }
    }

    #[stability::unstable(feature = "enable")]
    pub fn parent_ring<'a>(&'a self) -> RingRef<'a, R> {
        RingRef::new(self.ring.get_ring())
    }

    #[stability::unstable(feature = "enable")]
    pub fn from_e(&self) -> usize {
        self.from.1
    }

    #[stability::unstable(feature = "enable")]
    pub fn to_e(&self) -> usize {
        self.to.1
    }

    #[stability::unstable(feature = "enable")]
    pub fn maximal_ideal<'a>(&'a self) -> &'a R::MaximalIdeal<'ring> {
        &self.p
    }
}

impl<'ring, 'data, R> Homomorphism<R::LocalRingBase<'ring>, R::LocalRingBase<'ring>> for IntermediateReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    type CodomainStore = R::LocalRing<'ring>;
    type DomainStore = R::LocalRing<'ring>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to.0
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.from.0
    }

    fn map(&self, x: <R::LocalRingBase<'ring> as RingBase>::Element) -> <R::LocalRingBase<'ring> as RingBase>::Element {
        self.ring.get_ring().reduce_partial(self.p, (&self.from.0, self.from.1), (&self.to.0, self.to.1), x)
    }
}

impl<I: IntegerRing> PolyGCDLocallyDomain for I {

    type LocalRing<'ring> = zn_big::Zn<BigIntRing>
        where Self: 'ring;
    type LocalRingBase<'ring> = zn_big::ZnBase<BigIntRing>
        where Self: 'ring;
    type LocalFieldBase<'ring> = AsFieldBase<zn_64::Zn>
        where Self: 'ring;
    type LocalField<'ring> = AsField<zn_64::Zn>
        where Self: 'ring;
    type MaximalIdeal<'ring> = i64
        where Self: 'ring;

    fn factor_scaling(&self) -> Self::Element {
        self.one()
    }

    fn lift_full<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring
    {
        int_cast(from.0.smallest_lift(x), RingRef::new(self), BigIntRing::RING)
    }

    fn lift_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(from.1 <= to.1);
        let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
        return hom.map(from.0.any_lift(x));
    }

    fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        zn_64::Zn::new(*p as u64).as_field().ok().unwrap()
    }

    fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.pow(int_cast(*p, BigIntRing::RING, StaticRing::<i64>::RING), e))
    }

    fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
        where F: FnMut() -> u64
    {
        let lower_bound = StaticRing::<i64>::RING.get_ring().get_uniformly_random_bits(24, rng);
        return next_prime(StaticRing::<i64>::RING, lower_bound);
    }

    fn reduce_full<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        let self_ref = RingRef::new(self);
        let hom = to.0.can_hom(&self_ref).unwrap();
        return hom.map(x);
    }

    fn reduce_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(from.1 >= to.1);
        let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
        return hom.map(from.0.smallest_lift(x));
    }

    fn heuristic_exponent<'ring, P>(&self, p: &i64, poly_ring: P, poly: &El<P>) -> usize
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Self: 'ring
    {
        let log2_largest_exponent = poly_ring.terms(poly).map(|(c, _)| RingRef::new(self).abs_log2_ceil(c).unwrap() as f64).max_by(f64::total_cmp).unwrap();
        // this is in no way a rigorous bound, but equals the worst-case bound at least asymptotically (up to constants)
        return ((log2_largest_exponent + poly_ring.degree(poly).unwrap() as f64) / (*p as f64).log2() / /* just some factor that seemed good when playing around */ 4.).ceil() as usize + 1;
    }
    
    fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        write!(out, "{}", p)
    }
}

impl<I: IntegerRing> IntegerPolyGCDRing for I {

    fn maximal_ideal_gen<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> i64
        where Self: 'ring
    {
        *p
    }
}
