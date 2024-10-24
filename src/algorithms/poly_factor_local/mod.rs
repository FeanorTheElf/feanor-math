use crate::divisibility::*;
use crate::field::PerfectField;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::primitive_int::*;
use crate::ring::*;
use crate::rings::field::*;
use crate::rings::poly::*;
use crate::rings::zn::*;
use crate::rings::zn::zn_big;

use super::miller_rabin::next_prime;
use super::poly_factor::FactorPolyField;

pub mod hensel;
pub mod squarefree_part;
pub mod gcd;
pub mod factor;

///
/// Trait for rings that support lifting local factorizations of polynomials to the ring.
/// For infinite fields, this is the most important approach to computing gcd's or factorizations
/// of polynomials.
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
/// Furthermore, for any bound `B` and any maximal ideal `p`, the ring should be able to provide an
/// exponent `e` (via [`FactorPolyLocallyDomain::required_power()`]) such that the restriction of 
/// the reduction map
/// ```text
///   { x in R | |x| <= B }  ->  R / p^e
/// ```
/// is injective.
/// This means that a (possibly partial) factorization in the simpler ring `R / p^e`,
/// can be uniquely mapped back into `R` - assuming all coefficients of the actual factors 
/// have pseudo-norm at most `b` and lie within the ring `R`. 
/// 
/// Because of this, it must be possible to give a bound on the `l∞`-pseudo norm of any monic factor 
/// of a monic polynomial `f`, in terms of the `l2`-pseudo norm of `f` and `deg(f)`.
/// This bound should be provided by [`FactorPolyLocallyDomain::factor_bound()`].
/// Furthermore, it must be the case that the coefficients of any factor of `f` are "almost" in
/// the ring again. More concretely, there must be a factor `a`, given by 
/// [`FactorPolyLocallyDomain::factor_scaling()`], such that for each factor `g` of `f` over
/// the field of fractions, we have that `ag` has coefficients in the ring. In many cases (including
/// if the ring is a UFD, by Gauss' lemma), `a` can be chosen as `1`.
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
///    where we can efficiently compute polynomial factorizations, e.g. another [`FactorPolyLocallyDomain`]).
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
///    we do end up with coefficients in `R`. This is why I allow for the function
///    [`FactorPolyLocallyDomain::factor_scaling()`].
///    Furthermore, the "pseudo-norm" here should be the canonical norm.
/// 
/// I cannot think of any other good examples (these were the ones I had in mind when writing this trait), but 
/// who knows, maybe there are other rings that satisfy this and which we can thus do polynomial factorization in!
/// 
#[stability::unstable(feature = "enable")]
pub trait FactorPolyLocallyDomain: Domain + DivisibilityRing {

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
    type LocalFieldBase<'ring>: CanIsoFromTo<Self::LocalRingBase<'ring>> + FactorPolyField + PerfectField
        where Self: 'ring;

    type LocalField<'ring>: RingStore<Type = Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type MaximalIdeal<'ring>
        where Self: 'ring;

    fn ln_pseudo_norm(&self, el: &Self::Element) -> f64;

    ///
    /// Returns `ln(B)` for some `B > 0` such that for all monic polynomials `f` over this ring with
    /// `| f |_2 <= exp(ln_poly_l2_pseudo_norm)` and `deg(f) <= poly_deg` and any monic factor `g`
    /// of `f`, we have `| g |_∞ <= B`.
    /// 
    /// Here `| . |_2` resp. `| . |_∞` refer to the "pseudo norms" on polynomials that we get
    /// when taking pseudo norms coefficient-wise, and then taking the `l2`-resp. `l∞`-norm
    /// of the resulting vectors. 
    /// 
    fn ln_factor_coeff_bound(&self, ln_poly_l2_pseudo_norm: f64, poly_deg: usize) -> f64;

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
    /// Given `ln(B)`, returns an integer `e > 0` such that the restriction of the reduction map
    /// ```text
    ///   { x in R | |x| <= B }  ->  R / p^e
    /// ```
    /// is injective.
    /// 
    /// In other words, two elements of norm bounded by the given value should also be different
    /// modulo `p^e`.
    /// 
    fn required_power<'ring>(&self, p: &Self::MaximalIdeal<'ring>, ln_uniquely_representable_norm: f64) -> usize
        where Self: 'ring;

    ///
    /// Returns an ideal sampled at random from the interval of all supported maximal ideals.
    /// 
    fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
        where F: FnMut() -> u64;

    ///
    /// Returns a generator of the given maximal ideal
    /// 
    fn maximal_ideal_gen<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::Element
        where Self: 'ring;

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
}

#[stability::unstable(feature = "enable")]
pub struct ReductionMap<'ring, 'data, R>
    where R: 'ring + ?Sized + FactorPolyLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    p: &'data R::MaximalIdeal<'ring>,
    to: (R::LocalRing<'ring>, usize)
}

impl<'ring, 'data, R> ReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + FactorPolyLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, p: &'data R::MaximalIdeal<'ring>, power: usize) -> Self {
        assert!(power >= 1);
        Self { ring: RingRef::new(ring), p: p, to: (ring.local_ring_at(p, power), power) }
    }
}

impl<'ring, 'data, R> Homomorphism<R, R::LocalRingBase<'ring>> for ReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + FactorPolyLocallyDomain, 'ring: 'data
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
    where R: 'ring + ?Sized + FactorPolyLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    p: &'data R::MaximalIdeal<'ring>,
    from: (R::LocalRing<'ring>, usize),
    to: (R::LocalRing<'ring>, usize)
}

impl<'ring, 'data, R> IntermediateReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + FactorPolyLocallyDomain, 'ring: 'data
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
    where R: 'ring +?Sized + FactorPolyLocallyDomain, 'ring: 'data
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

impl FactorPolyLocallyDomain for BigIntRingBase {

    type LocalRing<'ring> = zn_big::Zn<BigIntRing>;
    type LocalRingBase<'ring> = zn_big::ZnBase<BigIntRing>;
    type LocalFieldBase<'ring> = AsFieldBase<zn_64::Zn>;
    type LocalField<'ring> = AsField<zn_64::Zn>;
    type MaximalIdeal<'ring> = i64;

    fn ln_pseudo_norm(&self, el: &Self::Element) -> f64 {
        if self.is_zero(el) {
            1.
        } else {
            BigIntRing::RING.abs_log2_ceil(el).unwrap() as f64 * 2f64.ln()
        }
    }

    fn ln_factor_coeff_bound(&self, ln_poly_l2_pseudo_norm: f64, poly_deg: usize) -> f64 {
        let ZZbig = BigIntRing::RING;
        let d = poly_deg as i32;
        // we use Theorem 3.5.1 from "A course in computational algebraic number theory", Cohen,
        // or equivalently Ex. 20 from Chapter 4.6.2 in Knuth's Art
        let result = ln_poly_l2_pseudo_norm + (ZZbig.abs_log2_ceil(&binomial(ZZbig.int_hom().map(d), &ZZbig.int_hom().map(d / 2), ZZbig)).unwrap() as f64 * 2f64.ln());
        return result;
    }

    fn factor_scaling(&self) -> Self::Element {
        self.one()
    }

    fn lift_full<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring
    {
        from.0.smallest_lift(x)
    }

    fn lift_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>> {
        assert!(from.1 <= to.1);
        let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
        return hom.map(from.0.any_lift(x));
    }

    fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        zn_64::Zn::new(*p as u64).as_field().ok().unwrap()
    }

    fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring> {
        zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.pow(int_cast(*p, BigIntRing::RING, StaticRing::<i64>::RING), e))
    }

    fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
        where F: FnMut() -> u64
    {
        let lower_bound = StaticRing::<i64>::RING.get_ring().get_uniformly_random_bits(32, rng);
        return next_prime(StaticRing::<i64>::RING, lower_bound);
    }

    fn reduce_full<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>> {
        let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
        return hom.map(x);
    }

    fn reduce_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>> {
        assert!(from.1 >= to.1);
        let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
        return hom.map(from.0.smallest_lift(x));
    }

    fn maximal_ideal_gen<'ring>(&self, p: &i64) -> Self::Element
        where Self: 'ring
    {
        int_cast(*p, RingRef::new(self), StaticRing::<i64>::RING)
    }

    fn required_power<'ring>(&self, p: &Self::MaximalIdeal<'ring>, ln_uniquely_representable_norm: f64) -> usize
        where Self: 'ring
    {
        assert!(ln_uniquely_representable_norm.is_finite());
        // the two is required to distinguish +/-
        ((1. + ln_uniquely_representable_norm) / (*p as f64).ln()).ceil() as usize
    }
}

///
/// Computes the map
/// ```text
///   R[X] -> R[X],  f(X) -> a^(deg(f) - 1) f(X / a)
/// ```
/// that can be used to make polynomials over a domain monic (when setting `a = lc(f)`).
/// 
fn evaluate_aX<P>(poly_ring: P, f: &El<P>, a: &El<<P::Type as RingExtension>::BaseRing>) -> El<P>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(f) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&f).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| if i == d { (ring.checked_div(c, a).unwrap(), d) } else { (ring.mul_ref_fst(c, ring.pow(ring.clone_el(a), d - i - 1)), i) }));
    return result;
}

///
/// Computes the inverse to [`evaluate_aX()`].
/// 
fn unevaluate_aX<P>(poly_ring: P, g: &El<P>, a: &El<<P::Type as RingExtension>::BaseRing>) -> El<P>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(g) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&g).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(g).map(|(c, i)| if i == d { (ring.clone_el(a), d) } else { (ring.checked_div(c, &ring.pow(ring.clone_el(a), d - i - 1)).unwrap(), i) }));
    return result;
}

///
/// Given a polynomial `f` over a PID, returns `(f/cont(f), cont(f))`, where `cont(f)`
/// is the content of `f`, i.e. the gcd of all coefficients of `f`.
/// 
#[stability::unstable(feature = "enable")]
pub fn make_primitive<P>(poly_ring: P, f: &El<P>) -> (El<P>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing + Domain
{
    if poly_ring.is_zero(f) {
        return (poly_ring.zero(), poly_ring.base_ring().one());
    }
    let ring = poly_ring.base_ring();
    let content = poly_ring.terms(f).map(|(c, _)| c).fold(ring.zero(), |a, b| ring.ideal_gen(&a, b));
    let result = poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (ring.checked_div(c, &content).unwrap(), i)));
    return (result, content);
}

///
/// A weaker version of [`make_primitive()`] that just divides out the "balance factor" of
/// all coefficients of `f`. The definition of the balance factor is completely up to the
/// underlying ring, see [`DivisibilityRing::balance_factor()`].
/// 
fn balance_poly<P>(poly_ring: P, f: El<P>) -> (El<P>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(&f) {
        return (poly_ring.zero(), poly_ring.base_ring().one());
    }
    let ring = poly_ring.base_ring();
    let factor = ring.get_ring().balance_factor(poly_ring.terms(&f).map(|(c, _)| c));
    let result = poly_ring.from_terms(poly_ring.terms(&f).map(|(c, i)| (ring.checked_div(c, &factor).unwrap(), i)));
    return (result, factor);
}

///
/// Checks whether there exists a polynomial `g` such that `g^k = f`, and if yes,
/// returns `g`.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::algorithms::poly_factor_local::*;
/// let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let [f, f_sqrt] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 2 * X + 1, X + 1]);
/// assert_el_eq!(&poly_ring, f_sqrt, poly_root(&poly_ring, &f, 2).unwrap());
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_root<P>(poly_ring: P, f: &El<P>, k: usize) -> Option<El<P>>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    assert!(poly_ring.degree(&f).unwrap() % k == 0);
    let d = poly_ring.degree(&f).unwrap() / k;
    let ring = poly_ring.base_ring();
    let k_in_ring = ring.int_hom().map(k as i32);

    let mut result_reversed = Vec::new();
    result_reversed.push(ring.one());
    for i in 1..=d {
        let g = poly_ring.pow(poly_ring.from_terms((0..i).map(|j| (ring.clone_el(&result_reversed[j]), j))), k);
        let partition_sum = poly_ring.coefficient_at(&g, i);
        let next_coeff = ring.checked_div(&ring.sub_ref(poly_ring.coefficient_at(&f, k * d - i), partition_sum), &k_in_ring)?;
        result_reversed.push(next_coeff);
    }

    let result = poly_ring.from_terms(result_reversed.into_iter().enumerate().map(|(i, c)| (c, d - i)));
    if poly_ring.eq_el(&f, &poly_ring.pow(poly_ring.clone_el(&result), k)) {
        return Some(result);
    } else {
        return None;
    }
}

#[cfg(test)]
use crate::rings::poly::dense_poly::*;
#[cfg(test)]
use factor::factor_poly_local;
#[cfg(test)]
use test::Bencher;

#[test]
fn test_poly_root() {
    let ring = BigIntRing::RING;
    let poly_ring = DensePolyRing::new(ring, "X");
    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(7) + X.pow_ref(6) + X.pow_ref(5) + X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1]);
    for k in 1..5 {
        assert_el_eq!(&poly_ring, &f, poly_root(&poly_ring, &poly_ring.pow(poly_ring.clone_el(&f), k), k).unwrap());
    }

    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(5) + 2 * X.pow_ref(4) + 3 * X.pow_ref(3) + 4 * X.pow_ref(2) + 5 * X + 6]);
    for k in 1..5 {
        assert_el_eq!(&poly_ring, &f, poly_root(&poly_ring, &poly_ring.pow(poly_ring.clone_el(&f), k), k).unwrap());
    }
}

#[bench]
fn bench_factor_rational_poly_new(bencher: &mut Bencher) {
    let ZZ = BigIntRing::RING;
    let incl = ZZ.int_hom();
    let poly_ring = DensePolyRing::new(&ZZ, "X");
    let f1 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 2), (incl.map(3), 4), (incl.map(1), 8)].into_iter());
    let f2 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4), (incl.map(1), 5), (incl.map(1), 10)].into_iter());
    let f3 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 1), (incl.map(-2), 5), (incl.map(1), 17)].into_iter());
    bencher.iter(|| {
        let actual = factor_poly_local(&poly_ring, poly_ring.prod([poly_ring.clone_el(&f1), poly_ring.clone_el(&f1), poly_ring.clone_el(&f2), poly_ring.clone_el(&f3), poly_ring.int_hom().map(9)].into_iter()));
        assert_eq!(3, actual.len());
        for (f, e) in actual.into_iter() {
            if poly_ring.eq_el(&f, &f1) {
                assert_el_eq!(poly_ring, f1, f);
                assert_eq!(2, e);
            } else if poly_ring.eq_el(&f, &f2) {
                assert_el_eq!(poly_ring, f2, f);
                assert_eq!(1, e);
           } else if poly_ring.eq_el(&f, &f3) {
               assert_el_eq!(poly_ring, f3, f);
               assert_eq!(1, e);
            } else {
                panic!("Factorization returned wrong factor {} of ({})^2 * ({}) * ({})", poly_ring.format(&f), poly_ring.format(&f1), poly_ring.format(&f2), poly_ring.format(&f3));
            }
        }
    });
}
