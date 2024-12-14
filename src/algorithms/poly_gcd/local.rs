use std::fmt::Display;

use crate::algorithms::linsolve::LinSolveRing;
use crate::algorithms::miller_rabin::is_prime;
use crate::computation::no_error;
use crate::delegate::DelegateRing;
use crate::delegate::DelegateRingImplFiniteRing;
use crate::divisibility::*;
use crate::homomorphism::*;
use crate::local::PrincipalLocalRing;
use crate::primitive_int::StaticRing;
use crate::integer::*;
use crate::ring::*;
use crate::rings::zn::*;
use crate::seq::*;
use crate::specialization::FiniteRingSpecializable;

use super::AsField;
use super::AsFieldBase;
use super::Field;
use super::FiniteRing;

///
/// Trait for rings that support lifting partial factorizations of polynomials modulo a prime
/// to the ring. For infinite fields, this is the most important approach to computing gcd's,
/// and also factorizations (with some caveats...).
/// 
/// Note that here (and in `feanor-math` generally), the term "local" is used to refer to algorithms
/// that work modulo prime ideals (or their powers), which is different from the mathematical concept
/// of localization.
/// 
/// The general philosophy is similar to [`crate::compute_locally::EvaluatePolyLocallyRing`].
/// However, when working with factorizations, it is usually better to compute modulo the factorization
/// modulo an ideal `I`, and use Hensel lifting to derive the factorization modulo `I^e`.
/// `EvaluatePolyLocallyRing` on the other hand is designed to allow computations modulo multiple different 
/// maximal ideals.
/// 
/// I am currently not yet completely sure what the exact requirements of this trait would be,
/// but conceptually, we want that for a random ideal `I` taken from some set of ideals (e.g. maximal
/// ideals, prime ideals, ... depending on the context), the quotient `R / I` should be
/// finite, and there should be a power `e` such that we can derive the factorization of some polynomial
/// over `R` from the factorization over `R / I^e`. Note that we don't assume that this `e` can be
/// computed, except possibly in special cases (like the integers). Also, we cannot always assume
/// that `I` is a maximal ideal (unfortunately - it was a nasty surprise when I realized this after
/// running my first implementation), however I believe we can choose it such that we know its decomposition
/// into maximal ideals `a = m1 ∩ ... ∩ mr`.
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
///    of the factors over `R[X1, ..., Xm]`. Note that if the base ring is not a field, the ideals will
///    only be prime and not maximal. Also, the polynomial coefficients of the result might only live in
///    the fraction field of `R`, similar to the algebraic number field case.
///  - Orders in algebraic number fields. This case is much more complicated, since (in general) we
///    don't have a UFD anymore. In particular, this is the case where we cannot generally choose `a = m` to be
///    a maximal ideal, and also need rational reconstruction when lifting the factorization back to the number
///    field. More concretely, the factors of a polynomial with coefficients in an order `R` don't necessarily 
///    have coefficients in `R`. Example: `X^2 - sqrt(3) X - 1` over `Z[sqrt(3), sqrt(7)]`
///    has the factor `X - (sqrt(3) + sqrt(7)) / 2`.
///    However, it turns out that if the original polynomial is monic, then its factors have coefficients in
///    the maximal order `O` of `R ⊗ QQ`. In particular, if we scale the factor by `[R : O] | disc(R)`, then
///    we do end up with coefficients in `R`. Unfortunately, the discriminant can become really huge, which
///    is why in the literature, rational reconstruction is used, to elements from `R / p^e` to "small" fractions
///    in `Frac(R)`.
/// 
/// I cannot think of any other good examples (these were the ones I had in mind when writing this trait), but 
/// who knows, maybe there are other rings that satisfy this and which we can thus do polynomial factorization in.
/// 
#[stability::unstable(feature = "enable")]
pub trait PolyGCDLocallyDomain: Domain + DivisibilityRing + FiniteRingSpecializable {

    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// see also [`crate::compute_locally::EvaluatePolyLocallyRing::LocalRingBase`]
    /// 
    type LocalRingBase<'ring>: ?Sized + LinSolveRing
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;
    
    type LocalFieldBase<'ring>: ?Sized + CanIsoFromTo<Self::LocalRingBase<'ring>> + FiniteRing + Field + SelfIso
        where Self: 'ring;

    type LocalField<'ring>: RingStore<Type = Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type SuitableIdeal<'ring>
        where Self: 'ring;

    ///
    /// Returns an exponent `e` such that we hope that the factors of a polynomial of given degree, 
    /// involving the given coefficient can already be read of (via [`PolyGCDLocallyDomain::reconstruct_ring_el()`]) 
    /// their reductions modulo `I^e`. Note that this is just a heuristic, and if it does not work,
    /// the implementation will gradually try larger `e`. Thus, even if this function returns constant
    /// 1, correctness will not be affected, but giving a good guess can improve performance
    /// 
    fn heuristic_exponent<'ring, 'a, I>(&self, _ideal: &Self::SuitableIdeal<'ring>, _poly_deg: usize, _coefficients: I) -> usize
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a,
            Self: 'ring;

    ///
    /// Returns an ideal sampled at random from the interval of all supported ideals.
    /// 
    fn random_suitable_ideal<'ring, F>(&'ring self, rng: F) -> Self::SuitableIdeal<'ring>
        where F: FnMut() -> u64;

    fn maximal_ideal_factor_count<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>) -> usize
        where Self: 'ring;

    ///
    /// Returns `R / mi`, where `mi` is the `i`-th maximal ideal over `I`.
    /// 
    /// This will always be a field, since `mi` is a maximal ideal.
    /// 
    fn local_field_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, max_ideal_idx: usize) -> Self::LocalField<'ring>
        where Self: 'ring;
    
    ///
    /// Returns `R / mi^e`, where `mi` is the `i`-th maximal ideal over `I`.
    /// 
    fn local_ring_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, e: usize, max_ideal_idx: usize) -> Self::LocalRing<'ring>
        where Self: 'ring;

    ///
    /// Computes the reduction map
    /// ```text
    ///   R -> R / mi^e
    /// ```
    /// where `mi` is the `i`-th maximal ideal over `I`.
    /// 
    fn reduce_ring_el<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes the reduction map
    /// ```text
    ///   R / mi^e1 -> R / mi^e2
    /// ```
    /// where `e1 >= e2` and `mi` is the `i`-th maximal ideal over `I`.
    /// 
    fn reduce_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes any element `y` in `R / mi^to_e` such that `y = x mod mi^from_e`.
    /// In particular, `y` does not have to be "short" in any sense, but any lift
    /// is a valid result.
    /// 
    fn lift_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring;

    ///
    /// Computes a "small" element `x in R` such that `x mod mi^e` is equal to the given value,
    /// for every maximal ideal `mi` over `I`.
    /// In cases where the factors of polynomials in `R[X]` do not necessarily have coefficients
    /// in `R`, this function might have to do rational reconstruction. 
    /// 
    fn reconstruct_ring_el<'local, 'element, 'ring, V1, V2>(&self, ideal: &Self::SuitableIdeal<'ring>, from: V1, e: usize, x: V2) -> Self::Element
        where Self: 'ring, 
            V1: VectorFn<&'local Self::LocalRing<'ring>>,
            V2: VectorFn<&'element El<Self::LocalRing<'ring>>>,
            Self::LocalRing<'ring>: 'local,
            El<Self::LocalRing<'ring>>: 'element,
            'ring: 'local + 'element;

    fn dbg_ideal<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring;
}

///
/// Subtrait of [`PolyGCDLocallyDomain`] that restricts the local rings to be [`ZnRing`],
/// which is necessary when implementing some base cases.
/// 
#[stability::unstable(feature = "enable")]
pub trait IntegerPolyGCDRing: PolyGCDLocallyDomain {

    ///
    /// It would be much preferrable if we could restrict associated types from supertraits,
    /// this is just a workaround (and an ugly one at that)
    /// 
    type LocalRingAsZnBase<'ring>: ?Sized + CanIsoFromTo<Self::LocalRingBase<'ring>> + ZnRing
        where Self: 'ring;

    type LocalRingAsZn<'ring>: RingStore<Type = Self::LocalRingAsZnBase<'ring>>
        where Self: 'ring;

    fn local_ring_as_zn<'a, 'ring>(&self, local_field: &'a Self::LocalRing<'ring>) -> &'a Self::LocalRingAsZn<'ring>;

    fn principal_ideal_generator<'ring>(&self, p: &Self::SuitableIdeal<'ring>) -> i64
        where Self: 'ring
    {
        assert_eq!(1, self.maximal_ideal_factor_count(p));
        let Fp = self.local_ring_at(p, 1, 0);
        let Fp = self.local_ring_as_zn(&Fp);
        return int_cast(Fp.integer_ring().clone_el(Fp.modulus()), StaticRing::<i64>::RING, Fp.integer_ring());
    }
}

#[stability::unstable(feature = "enable")]
pub struct IdealDisplayWrapper<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> {
    ring: &'a R,
    ideal: &'a R::SuitableIdeal<'ring>
}

impl<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> IdealDisplayWrapper<'a, 'ring, R> {

    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'a R, ideal: &'a R::SuitableIdeal<'ring>) -> Self {
        Self {
            ring: ring, 
            ideal: ideal
        }
    }
}

impl<'a, 'ring, R: 'ring + ?Sized + PolyGCDLocallyDomain> Display for IdealDisplayWrapper<'a, 'ring, R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.dbg_ideal(self.ideal, f)
    }
}

///
/// The map `R -> R/m^e`, where `m` is a maximal ideal factor of the ideal `I`,
/// as specified by [`PolyGCDLocallyDomain`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PolyGCDLocallyReductionMap<'ring, 'data, 'local, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    ideal: &'data R::SuitableIdeal<'ring>,
    to: (&'local R::LocalRing<'ring>, usize),
    max_ideal_idx: usize
}

impl<'ring, 'data, 'local, R> PolyGCDLocallyReductionMap<'ring, 'data, 'local, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, ideal: &'data R::SuitableIdeal<'ring>, to: &'local R::LocalRing<'ring>, to_e: usize, max_ideal_idx: usize) -> Self {
        assert!(to.get_ring() == ring.local_ring_at(ideal, to_e, max_ideal_idx).get_ring());
        Self {
            ring: RingRef::new(ring),
            ideal: ideal,
            to: (to, to_e),
            max_ideal_idx: max_ideal_idx
        }
    }
}

impl<'ring, 'data, 'local, R> Homomorphism<R, R::LocalRingBase<'ring>> for PolyGCDLocallyReductionMap<'ring, 'data, 'local, R>
    where R: 'ring +?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    type CodomainStore = &'local R::LocalRing<'ring>;
    type DomainStore = RingRef<'data, R>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to.0
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.ring
    }

    fn map(&self, x: <R as RingBase>::Element) -> <R::LocalRingBase<'ring> as RingBase>::Element {
        self.ring.get_ring().reduce_ring_el(self.ideal, (&self.to.0, self.to.1), self.max_ideal_idx, x)
    }
}

///
/// The map `R/m^r -> R/m^e`, where `r > e` and `m` is a maximal ideal factor of the 
/// ideal `I`, as specified by [`PolyGCDLocallyDomain`].
/// 
#[stability::unstable(feature = "enable")]
pub struct PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    ideal: &'data R::SuitableIdeal<'ring>,
    from: (&'local R::LocalRing<'ring>, usize),
    to: (&'local R::LocalRing<'ring>, usize),
    max_ideal_idx: usize
}

impl<'ring, 'data, 'local, R> PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, ideal: &'data R::SuitableIdeal<'ring>, from: &'local R::LocalRing<'ring>, from_e: usize, to: &'local R::LocalRing<'ring>, to_e: usize, max_ideal_idx: usize) -> Self {
        assert!(ring.local_ring_at(ideal, from_e, max_ideal_idx).get_ring() == from.get_ring());
        assert!(ring.local_ring_at(ideal, to_e, max_ideal_idx).get_ring() == to.get_ring());
        Self {
            ring: RingRef::new(ring),
            ideal: ideal,
            from: (from, from_e),
            to: (to, to_e),
            max_ideal_idx: max_ideal_idx
        }
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
    pub fn ideal<'a>(&'a self) -> &'a R::SuitableIdeal<'ring> {
        &self.ideal
    }

    #[stability::unstable(feature = "enable")]
    pub fn max_ideal_idx(&self) -> usize {
        self.max_ideal_idx
    }
}

impl<'ring, 'data, 'local, R> Homomorphism<R::LocalRingBase<'ring>, R::LocalRingBase<'ring>> for PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    type CodomainStore = &'local R::LocalRing<'ring>;
    type DomainStore = &'local R::LocalRing<'ring>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.to.0
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.from.0
    }

    fn map(&self, x: <R::LocalRingBase<'ring> as RingBase>::Element) -> <R::LocalRingBase<'ring> as RingBase>::Element {
        self.ring.get_ring().reduce_partial(self.ideal, (&self.from.0, self.from.1), (&self.to.0, self.to.1), self.max_ideal_idx, x)
    }
}

///
/// The sequence of maps `R  ->  R/m1^e x ... x R/mr^e  ->  R/m1 x ... x R/mr`, where
/// `m1, ..., mr` are the maximal ideals containing `I`, as specified by [`PolyGCDLocallyDomain`].
/// 
/// This sequence of maps is very relevant when using the compute-mod-p-and-lift approach
/// for polynomial operations over infinite rings (e.g. `QQ`). This is also the primary use case
/// for [`ReductionContext`].
///
#[stability::unstable(feature = "enable")]
pub struct ReductionContext<'ring, 'data, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    ring: RingRef<'data, R>,
    ideal: &'data R::SuitableIdeal<'ring>,
    from_e: usize,
    from: Vec<R::LocalRing<'ring>>,
    to: Vec<R::LocalRing<'ring>>
}

impl<'ring, 'data, R> ReductionContext<'ring, 'data, R>
    where R: 'ring + ?Sized + PolyGCDLocallyDomain, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, ideal: &'data R::SuitableIdeal<'ring>, e: usize) -> Self {
        assert!(e >= 1);
        let maximal_ideal_factor_count = ring.maximal_ideal_factor_count(ideal);
        Self {
            ring: RingRef::new(ring), 
            ideal: ideal, 
            from_e: e,
            from: (0..maximal_ideal_factor_count).map(|idx| ring.local_ring_at(ideal, e, idx)).collect::<Vec<_>>(), 
            to: (0..maximal_ideal_factor_count).map(|idx| ring.local_ring_at(ideal, 1, idx)).collect::<Vec<_>>(),
        }
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn ideal(&self) -> &'data R::SuitableIdeal<'ring> {
        self.ideal
    }

    #[stability::unstable(feature = "enable")]
    pub fn main_ring_to_field_reduction<'local>(&'local self, max_ideal_idx: usize) -> PolyGCDLocallyReductionMap<'ring, 'data, 'local, R> {
        PolyGCDLocallyReductionMap {
            ideal: self.ideal,
            ring: self.ring,
            to: (self.to.at(max_ideal_idx), 1),
            max_ideal_idx: max_ideal_idx
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn main_ring_to_intermediate_ring_reduction<'local>(&'local self, max_ideal_idx: usize) -> PolyGCDLocallyReductionMap<'ring, 'data, 'local, R> {
        PolyGCDLocallyReductionMap {
            ideal: self.ideal,
            ring: self.ring,
            to: (self.from.at(max_ideal_idx), self.from_e),
            max_ideal_idx: max_ideal_idx
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn intermediate_ring_to_field_reduction<'local>(&'local self, max_ideal_idx: usize) -> PolyGCDLocallyIntermediateReductionMap<'ring, 'data, 'local, R> {
        PolyGCDLocallyIntermediateReductionMap {
            from: (&self.from[max_ideal_idx], self.from_e),
            to: (&self.to[max_ideal_idx], 1),
            max_ideal_idx: max_ideal_idx,
            ring: self.ring,
            ideal: self.ideal
        }
    }


    #[stability::unstable(feature = "enable")]
    pub fn len(&self) -> usize {
        self.from.len()
    }

    #[stability::unstable(feature = "enable")]
    pub fn reconstruct_ring_el<'local, V>(&self, els: V) -> R::Element
        where V: VectorFn<&'local El<R::LocalRing<'ring>>>,
            El<R::LocalRing<'ring>>: 'local,
            'ring: 'local
    {
        fn do_reconstruction<'ring, 'local, R, V>(ring: &R, ideal: &R::SuitableIdeal<'ring>, local_rings: &[R::LocalRing<'ring>], e: usize, els: V) -> R::Element
            where R: 'ring + ?Sized + PolyGCDLocallyDomain,
                V: VectorFn<&'local El<R::LocalRing<'ring>>>,
                El<R::LocalRing<'ring>>: 'local,
                'ring: 'local
        {
            ring.reconstruct_ring_el(ideal, local_rings.as_fn(), e, els)
        }
        do_reconstruction(self.ring.get_ring(), self.ideal, &self.from[..], self.from_e, els)
    }
}

#[stability::unstable(feature = "enable")]
pub const INTRING_HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR: f64 = 0.25;

#[macro_export]
macro_rules! impl_poly_gcd_locally_for_ZZ {
    (IntegerPolyGCDRing for $int_ring_type:ty) => {
        impl_poly_gcd_locally_for_ZZ!{ <{}> IntegerPolyGCDRing for $int_ring_type where }
    };
    (<{$($gen_args:tt)*}> IntegerPolyGCDRing for $int_ring_type:ty where $($constraints:tt)*) => {

        impl<$($gen_args)*> $crate::algorithms::poly_gcd::local::PolyGCDLocallyDomain for $int_ring_type
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
            type SuitableIdeal<'ring> = i64
                where Self: 'ring;
        
            fn reconstruct_ring_el<'local, 'element, 'ring, V1, V2>(&self, _p: &Self::SuitableIdeal<'ring>, from: V1, _e: usize, x: V2) -> Self::Element
                where Self: 'ring,
                    V1: $crate::seq::VectorFn<&'local Self::LocalRing<'ring>>,
                    V2: $crate::seq::VectorFn<&'element El<Self::LocalRing<'ring>>>
            {
                use $crate::rings::zn::*;
                #[allow(unused)]
                use $crate::seq::*;
                assert_eq!(1, from.len());
                assert_eq!(1, x.len());
                int_cast(from.at(0).smallest_lift(from.at(0).clone_el(x.at(0))), RingRef::new(self), BigIntRing::RING)
            }

            fn maximal_ideal_factor_count<'ring>(&self, _p: &Self::SuitableIdeal<'ring>) -> usize
                where Self: 'ring
            {
                1
            }
        
            fn lift_partial<'ring>(&self, _p: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::rings::zn::*;
                use $crate::homomorphism::*;

                assert_eq!(0, max_ideal_idx);
                assert!(from.1 <= to.1);
                let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
                return hom.map(from.0.any_lift(x));
            }
        
            fn local_field_at<'ring>(&self, p: &Self::SuitableIdeal<'ring>, max_ideal_idx: usize) -> Self::LocalField<'ring>
                where Self: 'ring
            {
                use $crate::rings::zn::*;

                assert_eq!(0, max_ideal_idx);
                $crate::rings::zn::zn_64::Zn::new(*p as u64).as_field().ok().unwrap()
            }
        
            fn local_ring_at<'ring>(&self, p: &Self::SuitableIdeal<'ring>, e: usize, max_ideal_idx: usize) -> Self::LocalRing<'ring>
                where Self: 'ring
            {
                assert_eq!(0, max_ideal_idx);
                $crate::rings::zn::zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.pow(int_cast(*p, BigIntRing::RING, StaticRing::<i64>::RING), e))
            }
        
            fn random_suitable_ideal<'ring, F>(&'ring self, rng: F) -> Self::SuitableIdeal<'ring>
                where F: FnMut() -> u64
            {
                let lower_bound = StaticRing::<i64>::RING.get_ring().get_uniformly_random_bits(24, rng);
                return $crate::algorithms::miller_rabin::next_prime(StaticRing::<i64>::RING, lower_bound);
            }
        
            fn reduce_ring_el<'ring>(&self, _p: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::homomorphism::*;

                assert_eq!(0, max_ideal_idx);
                let self_ref = RingRef::new(self);
                let hom = to.0.can_hom(&self_ref).unwrap();
                return hom.map(x);
            }
        
            fn reduce_partial<'ring>(&self, _p: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
                where Self: 'ring
            {
                use $crate::rings::zn::*;
                use $crate::homomorphism::*;

                assert_eq!(0, max_ideal_idx);
                assert!(from.1 >= to.1);
                let hom = to.0.can_hom(to.0.integer_ring()).unwrap();
                return hom.map(from.0.smallest_lift(x));
            }
        
            fn heuristic_exponent<'ring, 'a, I>(&self, p: &i64, poly_deg: usize, coefficients: I) -> usize
                where I: Iterator<Item = &'a Self::Element>,
                    Self: 'a,
                    Self: 'ring
            {
                let log2_max_coeff = coefficients.map(|c| RingRef::new(self).abs_log2_ceil(c).unwrap_or(0)).max().unwrap_or(0);
                // this is in no way a rigorous bound, but equals the worst-case bound at least asymptotically (up to constants)
                return ((log2_max_coeff as f64 + poly_deg as f64) / (*p as f64).log2() * $crate::algorithms::poly_gcd::local::INTRING_HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR).ceil() as usize + 1;
            }
            
            fn dbg_ideal<'ring>(&self, p: &Self::SuitableIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
                where Self: 'ring
            {
                write!(out, "({})", p)
            }
        }

        impl<$($gen_args)*> $crate::algorithms::poly_gcd::local::IntegerPolyGCDRing for $int_ring_type
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
        }
    };
}

///
/// We cannot provide a blanket impl of [`super::PolyTFracGCDRing`] for finite fields, since it would
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
    where R: ?Sized + FiniteRing + Field + SelfIso
{
    type LocalRingBase<'ring> = Self
        where Self: 'ring;
    type LocalRing<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
                
    type LocalFieldBase<'ring> = Self
        where Self: 'ring;
    type LocalField<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
    type SuitableIdeal<'ring> = RingRef<'ring, Self>
        where Self: 'ring;
        
    fn heuristic_exponent<'ring, 'element, IteratorType>(&self, _maximal_ideal: &Self::SuitableIdeal<'ring>, _poly_deg: usize, _coefficients: IteratorType) -> usize
        where IteratorType: Iterator<Item = &'element Self::Element>,
            Self: 'element,
            Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn maximal_ideal_factor_count<'ring>(&self,ideal: &Self::SuitableIdeal<'ring>) -> usize where Self:'ring {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn random_suitable_ideal<'ring, RandomNumberFunction>(&'ring self, rng: RandomNumberFunction) -> Self::SuitableIdeal<'ring>
        where RandomNumberFunction: FnMut() -> u64
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn local_field_at<'ring>(&self, p: &Self::SuitableIdeal<'ring>, max_ideal_idx: usize) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
                
    fn local_ring_at<'ring>(&self, p: &Self::SuitableIdeal<'ring>, e: usize, max_ideal_idx: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reduce_ring_el<'ring>(&self, p: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reduce_partial<'ring>(&self, p: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn lift_partial<'ring>(&self, p: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }

    fn reconstruct_ring_el<'local, 'element, 'ring, V1, V2>(&self, p: &Self::SuitableIdeal<'ring>, from: V1, e: usize, x: V2) -> Self::Element
        where Self: 'ring, 
            V1: VectorFn<&'local Self::LocalRing<'ring>>,
            V2: VectorFn<&'element El<Self::LocalRing<'ring>>>,
            'ring: 'local + 'element,
            Self::LocalRing<'ring>: 'local,
            El<Self::LocalRing<'ring>>: 'element
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
    
    fn dbg_ideal<'ring>(&self, p: &Self::SuitableIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        unreachable!("this should never be called for finite fields, since specialized functions are available in this case")
    }
}

#[stability::unstable(feature = "enable")]
pub struct IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing + LinSolveRing
{
    integers: &'a R::IntegerRing,
    prime: El<R::IntegerRing>
}

impl<'a, R> IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing + LinSolveRing,
        AsFieldBase<RingValue<R>>: CanIsoFromTo<R> + SelfIso
{
    #[stability::unstable(feature = "enable")]
    pub fn new(integers: &'a R::IntegerRing, prime: El<R::IntegerRing>) -> Self {
        assert!(is_prime(integers, &prime, 10));
        return Self { integers, prime };
    }

    #[stability::unstable(feature = "enable")]
    pub fn reduction_context<'b>(&'b self, from_e: usize, to_e: usize) -> ReductionContext<'b, 'b, Self> {
        ReductionContext {
            ring: RingRef::new(self),
            ideal: &self.prime,
            from_e: from_e,
            from: vec![self.local_ring_at(&self.prime, from_e, 0)], 
            to: vec![self.local_ring_at(&self.prime, to_e, 0)], 
        }
    }
}

impl<'a, R> PartialEq for IntegersWithLocalZnQuotient<'a, R>
    where R: ?Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing + LinSolveRing
{
    fn eq(&self, other: &Self) -> bool {
        self.integers.get_ring() == other.integers.get_ring()
    }
}

impl<'a, R> DelegateRing for IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing + LinSolveRing
{
    type Base = <R as ZnRing>::IntegerRingBase;
    type Element = El<<R as ZnRing>::IntegerRing>;

    fn get_delegate(&self) -> &Self::Base {
        self.integers.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'b>(&self, el: &'b mut Self::Element) -> &'b mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'b>(&self, el: &'b Self::Element) -> &'b <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<'a, R> Domain for IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing
{}

impl<'a, R> DelegateRingImplFiniteRing for IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing
{}

impl<'a, R> PolyGCDLocallyDomain for IntegersWithLocalZnQuotient<'a, R>
    where R: Sized + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing + LinSolveRing,
        AsFieldBase<RingValue<R>>: CanIsoFromTo<R> + SelfIso
{
    type LocalRingBase<'ring> = R
        where Self: 'ring;

    type LocalRing<'ring> = RingValue<R>
        where Self: 'ring;
    
    type LocalFieldBase<'ring> = AsFieldBase<RingValue<R>>
        where Self: 'ring;

    type LocalField<'ring> = AsField<RingValue<R>>
        where Self: 'ring;

    type SuitableIdeal<'ring> = Self::Element
        where Self: 'ring;

    fn heuristic_exponent<'ring, 'el, I>(&self, ideal: &Self::SuitableIdeal<'ring>, poly_deg: usize, coefficients: I) -> usize
        where I: Iterator<Item = &'el Self::Element>,
            Self: 'el + 'ring
    {
        let log2_max_coeff = coefficients.map(|c| self.integers.abs_log2_ceil(c).unwrap_or(0)).max().unwrap_or(0);
        // this is in no way a rigorous bound, but equals the worst-case bound at least asymptotically (up to constants)
        return ((log2_max_coeff as f64 + poly_deg as f64) / self.integers.abs_log2_floor(ideal).unwrap_or(1) as f64 * crate::algorithms::poly_gcd::local::INTRING_HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR).ceil() as usize + 1;
    }

    fn random_suitable_ideal<'ring, F>(&'ring self, _rng: F) -> Self::SuitableIdeal<'ring>
        where F: FnMut() -> u64
    {
        self.integers.clone_el(&self.prime)
    }

    fn maximal_ideal_factor_count<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>) -> usize
        where Self: 'ring
    {
        debug_assert!(self.integers.eq_el(ideal, &self.prime));
        1
    }

    fn local_field_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, max_ideal_idx: usize) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        assert_eq!(0, max_ideal_idx);
        self.local_ring_at(ideal, 1, 0).as_field().ok().unwrap()
    }
    
    fn local_ring_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, e: usize, max_ideal_idx: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        assert_eq!(0, max_ideal_idx);
        assert!(self.integers.eq_el(ideal, &self.prime));
        RingValue::from(R::create(|ZZ| Ok(RingRef::new(ZZ).pow(int_cast(self.integers.clone_el(&self.prime), RingRef::new(ZZ), &self.integers), e))).unwrap_or_else(no_error))
    }

    fn reduce_ring_el<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        debug_assert_eq!(0, max_ideal_idx);
        debug_assert!(self.integers.eq_el(ideal, &self.prime));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), to.1), to.0.modulus()));
        to.0.coerce(&self.integers, x)
    }

    fn reduce_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        debug_assert_eq!(0, max_ideal_idx);
        debug_assert!(self.integers.eq_el(ideal, &self.prime));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), to.1), to.0.modulus()));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), from.1), from.0.modulus()));
        to.0.coerce(&self.integers, from.0.smallest_positive_lift(x))
    }

    fn lift_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), max_ideal_idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        debug_assert_eq!(0, max_ideal_idx);
        debug_assert!(self.integers.eq_el(ideal, &self.prime));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), to.1), to.0.modulus()));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), from.1), from.0.modulus()));
        to.0.coerce(&self.integers, from.0.smallest_positive_lift(x))
    }

    fn reconstruct_ring_el<'local, 'element, 'ring, V1, V2>(&self, ideal: &Self::SuitableIdeal<'ring>, from: V1, e: usize, x: V2) -> Self::Element
        where Self: 'ring, 
            V1: VectorFn<&'local Self::LocalRing<'ring>>,
            V2: VectorFn<&'element El<Self::LocalRing<'ring>>>,
            Self::LocalRing<'ring>: 'local,
            El<Self::LocalRing<'ring>>: 'element,
            'ring: 'local + 'element
    {
        assert_eq!(1, x.len());
        assert_eq!(1, from.len());
        debug_assert!(self.integers.eq_el(ideal, &self.prime));
        debug_assert!(self.integers.eq_el(&self.integers.pow(self.integers.clone_el(&self.prime), e), from.at(0).modulus()));
        from.at(0).smallest_lift(from.at(0).clone_el(x.at(0)))
    }

    fn dbg_ideal<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        self.dbg(ideal, out)
    }
}