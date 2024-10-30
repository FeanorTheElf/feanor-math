use std::fmt::Display;

use crate::algorithms::linsolve::LinSolveRing;
use crate::divisibility::*;
use crate::homomorphism::*;
use crate::ring::*;

use crate::rings::zn::ZnRing;
use crate::specialization::FiniteRingSpecializable;

use super::Field;
use super::FiniteRing;

///
/// Trait for rings that support lifting local partial factorizations of polynomials to the ring.
/// For infinite fields, this is the most important approach to computing gcd's.
/// 
/// The general philosophy is similar to [`crate::compute_locally::EvaluatePolyLocallyRing`].
/// However, when working with factorizations, it is usually better to compute modulo a power
/// `p^e` of a maximal ideal `p` and use Hensel lifting. `EvaluatePolyLocallyRing` on the other
/// hand is designed to allow computations modulo multiple different maximal ideals.
/// 
/// I am currently not yet completely sure what the exact requirements of this trait would be,
/// but conceptually, we want that for a random maximal ideal `m`, the quotient `R / m` should be
/// finite, and there should be a power `e` such that we can derive the factorization of some polynomial
/// over `R` from the factorization over `R / m^e`. Note that we don't assume that this `e` can be
/// computed, except possibly in special cases (like the integers).
/// 
/// Note also that I want this to work even when going to algebraic extensions or even the algebraic
/// closure. This however seems to follow naturally in most cases.
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
///    we do end up with coefficients in `R`. Unfortunately, the discriminant can become really huge, which
///    is why in the literature, rational reconstruction is used, to elements from `R / p^e` to "small" fractions
///    in `Frac(R)`.
/// 
/// I cannot think of any other good examples (these were the ones I had in mind when writing this trait), but 
/// who knows, maybe there are other rings that satisfy this and which we can thus do polynomial factorization in!
/// 
#[stability::unstable(feature = "enable")]
pub trait PolyGCDLocallyDomain: Domain + DivisibilityRing + FiniteRingSpecializable {

    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// see also [`crate::compute_locally::ComputeLocallyRing`]
    /// 
    type LocalRingBase<'ring>: LinSolveRing
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;
    
    ///
    /// Again, this is required to restrict `AsFieldBase<Self::LocalRing<'ring>>: FactorPolyField`,
    /// which in turn is required since the straightforward bound `for<'a> AsFieldBase<Self::LocalRing<'a>>: FactorPolyField`
    /// is bugged, see also [`crate::compute_locally::ComputeLocallyRing`].
    /// 
    /// Note that restricting this to `PolyGCDLocallyDomain` is necessary to close the loop and provide a blanket implementation
    /// for all algebraic extensions of a `PolyGCDLocallyDomain`. The properties of `PolyGCDLocallyDomain` are actually never used
    /// for this ring, since things should always boil down to the finite case, where `PolyGCDRing` can be implemented globally.
    /// 
    type LocalFieldBase<'ring>: CanIsoFromTo<Self::LocalRingBase<'ring>> + FiniteRing + Field
        where Self: 'ring;

    type LocalField<'ring>: RingStore<Type = Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type MaximalIdeal<'ring>
        where Self: 'ring;

    ///
    /// Returns an exponent `e` such that we hope that the factors of a polynomial of given degree, 
    /// involving the given coefficient can already be read of (via [`PolyGCDLocallyDomain::reconstruct_ring_el()`]) 
    /// their reductions modulo `p^e`. Note that this is just a heuristic, and if it does not work,
    /// the implementation will gradually try larger `e`. Thus, even if this function returns constant
    /// 1, correctness will not be affected, but giving a good guess can improve performance
    /// 
    fn heuristic_exponent<'ring, 'a, I>(&self, _maximal_ideal: &Self::MaximalIdeal<'ring>, _poly_deg: usize, _coefficients: I) -> usize
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a,
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
    fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
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
    /// Computes a "small" element `x in R` such that `x mod p^e` is equal to the given value.
    /// In cases where the factors of polynomials in `R[X]` do not necessarily have coefficients
    /// in `R`, this function might have to do rational reconstruction. 
    /// 
    fn reconstruct_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring;

    fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring;
}

///
/// Subtrait of [`PolyGCDLocallyDomain`] that restricts the local rings to be [`ZnRing`],
/// which is sometimes necessary when implementing some base cases.
/// 
#[stability::unstable(feature = "enable")]
pub trait IntegerPolyGCDRing: PolyGCDLocallyDomain {

    type LocalRingAsZnBase<'ring>: CanIsoFromTo<Self::LocalRingBase<'ring>> + ZnRing
        where Self: 'ring;

    type LocalRingAsZn<'ring>: RingStore<Type = Self::LocalRingAsZnBase<'ring>>
        where Self: 'ring;

    fn local_ring_as_zn<'a, 'ring>(&self, local_field: &'a Self::LocalRing<'ring>) -> &'a Self::LocalRingAsZn<'ring>;

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
        self.ring.get_ring().reduce_ring_el(self.p, (&self.to.0, self.to.1), x)
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

#[macro_export]
macro_rules! impl_poly_gcd_locally_for_ZZ {
    (<{$($gen_args:tt)*}> IntegerPolyGCDRing for $int_ring_type:ty where $($constraints:tt)*) => {

        impl<$($gen_args)*> $crate::algorithms::poly_gcd::gcd_locally::PolyGCDLocallyDomain for $int_ring_type
            where $($constraints)*
        {
            type LocalRing<'ring> = $crate::rings::zn::zn_big::Zn<BigIntRing>
                where Self: 'ring;
            type LocalRingBase<'ring> = $crate::rings::zn::zn_big::ZnBase<BigIntRing>
                where Self: 'ring;
            type LocalFieldBase<'ring> = $crate::rings::field::AsFieldBase<$crate::rings::zn::zn_64::Zn>
                where Self: 'ring;
            type LocalField<'ring> = $crate::rings::field::AsField<$crate::rings::zn::zn_64::Zn>
                where Self: 'ring;
            type MaximalIdeal<'ring> = i64
                where Self: 'ring;
        
            fn reconstruct_ring_el<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
                where Self: 'ring
            {
                use $crate::rings::zn::*;

                int_cast(from.0.smallest_lift(x), RingRef::new(self), BigIntRing::RING)
            }
        
            fn lift_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::rings::zn::*;
                use $crate::homomorphism::*;

                assert!(from.1 <= to.1);
                let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
                return hom.map(from.0.any_lift(x));
            }
        
            fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
                where Self: 'ring
            {
                use $crate::rings::zn::*;

                $crate::rings::zn::zn_64::Zn::new(*p as u64).as_field().ok().unwrap()
            }
        
            fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
                where Self: 'ring
            {
                $crate::rings::zn::zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.pow(int_cast(*p, BigIntRing::RING, StaticRing::<i64>::RING), e))
            }
        
            fn random_maximal_ideal<'ring, F>(&'ring self, rng: F) -> Self::MaximalIdeal<'ring>
                where F: FnMut() -> u64
            {
                let lower_bound = StaticRing::<i64>::RING.get_ring().get_uniformly_random_bits(24, rng);
                return $crate::algorithms::miller_rabin::next_prime(StaticRing::<i64>::RING, lower_bound);
            }
        
            fn reduce_ring_el<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::homomorphism::*;

                let self_ref = RingRef::new(self);
                let hom = to.0.can_hom(&self_ref).unwrap();
                return hom.map(x);
            }
        
            fn reduce_partial<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::rings::zn::*;
                use $crate::homomorphism::*;

                assert!(from.1 >= to.1);
                let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
                return hom.map(from.0.smallest_lift(x));
            }
        
            fn heuristic_exponent<'ring, 'a, I>(&self, p: &i64, poly_deg: usize, coefficients: I) -> usize
                where I: Iterator<Item = &'a Self::Element>,
                    Self: 'a,
                    Self: 'ring
            {
                let log2_largest_exponent = coefficients.map(|c| RingRef::new(self).abs_log2_ceil(c).unwrap() as f64).max_by(f64::total_cmp).unwrap();
                // this is in no way a rigorous bound, but equals the worst-case bound at least asymptotically (up to constants)
                return ((log2_largest_exponent + poly_deg as f64) / (*p as f64).log2() / /* just some factor that seemed good when playing around */ 4.).ceil() as usize + 1;
            }
            
            fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
                where Self: 'ring
            {
                write!(out, "({})", p)
            }
        }

        impl<$($gen_args)*> $crate::algorithms::poly_gcd::gcd_locally::IntegerPolyGCDRing for $int_ring_type
            where $($constraints)*
        {
            type LocalRingAsZnBase<'ring> = Self::LocalRingBase<'ring>
                where Self: 'ring;

            type LocalRingAsZn<'ring> = Self::LocalRing<'ring>
                where Self: 'ring;

            fn local_ring_as_zn<'a, 'ring>(&self, local_field: &'a Self::LocalRing<'ring>) -> &'a Self::LocalRingAsZn<'ring>
                where Self: 'ring
            {
                local_field
            }

            fn maximal_ideal_gen<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> i64
                where Self: 'ring
            {
                *p
            }
        }
    };
}

///
/// We cannot provide a blanket impl of [`super::PolyGCDRing`] for finite fields, since it would
/// conflict with the one for all rings that impl [`PolyGCDLocallyDomain`]. Thus, we implement
/// [`PolyGCDLocallyDomain`] for all finite fields, and reuse the blanket impl.
/// 
/// Note that while technically, finite fields are always [`PolyGCDLocallyDomain`] - where the
/// local rings are all equal to itself - we still panic in the implementations. In particular,
/// giving a working implementation would be "correct", but this implementation should never be
/// called anyway, since we specialize on finite fields previously anyway.
/// 
#[allow(unused)]
impl<R> PolyGCDLocallyDomain for R
    where R: FiniteRing + Field + FiniteRingSpecializable + SelfIso
{
    type LocalRingBase<'ring> = Self
        where Self: 'ring;
    type LocalRing<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
                
    type LocalFieldBase<'ring> = Self
        where Self: 'ring;
    type LocalField<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
    type MaximalIdeal<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
        
    fn heuristic_exponent<'ring, 'element, IteratorType>(&self, _maximal_ideal: &Self::MaximalIdeal<'ring>, _poly_deg: usize, _coefficients: IteratorType) -> usize
        where IteratorType: Iterator<Item = &'element Self::Element>,
            Self: 'element,
            Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn random_maximal_ideal<'ring, RandomNumberFunction>(&'ring self, rng: RandomNumberFunction) -> Self::MaximalIdeal<'ring>
        where RandomNumberFunction: FnMut() -> u64
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
                
    fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reconstruct_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
    
    fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
}