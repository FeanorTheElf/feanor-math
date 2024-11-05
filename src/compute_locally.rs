
use crate::algorithms::miller_rabin::prev_prime;
use crate::divisibility::{DivisibilityRing, Domain};
use crate::homomorphism::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::pid::PrincipalIdealRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::zn::{zn_64, zn_rns, ZnRingStore};
use crate::seq::VectorView;

///
/// Trait for rings that can be temporarily replaced by an extension when we need more points,
/// e.g. for interpolation.
/// 
/// Note that a trivial implementation is possible for every ring of characteristic 0, since
/// these already have infinitely many points whose pairwise differences are non-zero-divisors. 
/// Such an implementation can be added to new types using the macro [`impl_interpolation_base_ring_char_zero!`].
/// 
#[stability::unstable(feature = "enable")]
pub trait InterpolationBaseRing: DivisibilityRing {

    ///
    /// Restricting this here to be `DivisibilityRing + PrincipalIdealRing + Domain`
    /// is necessary, because of a compiler bug, see also
    /// [`crate::compute_locally::EvaluatePolyLocallyRing::LocalRingBase`]
    /// 
    type ExtendedRingBase<'a>: ?Sized + DivisibilityRing + PrincipalIdealRing + Domain
        where Self: 'a;

    type ExtendedRing<'a>: RingStore<Type = Self::ExtendedRingBase<'a>> + Clone
        where Self: 'a;

    fn in_base<'a, S>(&self, ext_ring: S, el: El<S>) -> Option<Self::Element>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>;

    fn in_extension<'a, S>(&self, ext_ring: S, el: Self::Element) -> El<S>
        where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>;

    ///
    /// Returns `count` points such that the difference between any two of them
    /// is a non-zero-divisor.
    /// 
    /// Any two calls must give elements in the same order.
    /// 
    fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>);
}

///
/// [`RingStore`] for [`InterpolationBaseRing`].
/// 
#[stability::unstable(feature = "enable")]
pub trait InterpolationBaseRingStore: RingStore
    where Self::Type: InterpolationBaseRing
{}

impl<R> InterpolationBaseRingStore for R
    where R: RingStore, R::Type: InterpolationBaseRing
{}

///
/// The inclusion map `R -> S` for a ring `R` and one of its extensions `S`
/// as given by [`InterpolationBaseRing`].
/// 
#[stability::unstable(feature = "enable")]
pub struct ToExtRingMap<'a, R>
    where R: ?Sized + InterpolationBaseRing
{
    ring: RingRef<'a, R>,
    ext_ring: R::ExtendedRing<'a>
}

impl<'a, R> ToExtRingMap<'a, R>
    where R: ?Sized + InterpolationBaseRing
{
    #[stability::unstable(feature = "enable")]
    pub fn for_interpolation(ring: &'a R, point_count: usize) -> (Self, Vec<El<R::ExtendedRing<'a>>>) {
        let (ext_ring, points) = ring.interpolation_points(point_count);
        return (Self { ring: RingRef::new(ring), ext_ring: ext_ring }, points);
    }

    #[stability::unstable(feature = "enable")]
    pub fn as_base_ring_el(&self, el: El<R::ExtendedRing<'a>>) -> R::Element {
        self.ring.get_ring().in_base(&self.ext_ring, el).unwrap()
    }
}

impl<'a, R> Homomorphism<R, R::ExtendedRingBase<'a>> for ToExtRingMap<'a, R>
    where R: ?Sized + InterpolationBaseRing
{
    type CodomainStore = R::ExtendedRing<'a>;
    type DomainStore = RingRef<'a, R>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.ext_ring
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.ring
    }

    fn map(&self, x: <R as RingBase>::Element) -> <R::ExtendedRingBase<'a> as RingBase>::Element {
        self.ring.get_ring().in_extension(&self.ext_ring, x)
    }
}

///
/// Trait for rings that support performing computations locally. Locally here refers
/// not to its localization, but to quotients by primes ideals.
/// 
/// More concretely, a ring `R` implementing this trait should be endowed with a
/// "pseudo norm"
/// ```text
///   |.|: R  ->  [0, ∞)
/// ```
/// i.e. a symmetric, sub-additive, sub-multiplicative map.
/// Furthermore, for any bound `b`, the ring should be able to provide prime ideals
/// `p1, ..., pk` together with the rings `Ri = R / pi`, such that the restricted
/// reduction map
/// ```text
///   { x in R | |x| <= b }  ->  R1 x ... x Rk
/// ```
/// is injective.
/// This means that a computation can be performed in the simpler ring `R1 x ... x Rk`,
/// and - assuming the result is of pseudo-norm `<= b`, mapped back to `R`.
/// 
/// The standard use case is the evaluation of a multivariate polynomial `f(X1, ..., Xm)`
/// over this ring. The trait is designed to enable the following approach:
///  - Given ring elements `a1, ..., am`, compute an upper bound `B` on `|f(a1, ..., am)|`.
///    The values `|ai|` are given by [`EvaluatePolyLocallyRing::pseudo_norm()`].
///  - Get a sufficient number of prime ideals, using [`EvaluatePolyLocallyRing::local_computation()`] 
///  - Compute `f(a1 mod pi, ..., am mod pi) mod pi` for each prime `pi` within the ring given by 
///    [`EvaluatePolyLocallyRing::local_ring_at()`]. The reductions `ai mod pj` are given by
///    [`EvaluatePolyLocallyRing::reduce()`].
///  - Recombine the results to an element of `R` by using [`EvaluatePolyLocallyRing::lift_combine()`].
/// 
#[stability::unstable(feature = "enable")]
pub trait EvaluatePolyLocallyRing: RingBase {
    
    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// to allow it to reference both the ring itself and the current `LocalComputationData`.
    /// However, when doing this, I ran into the compiler bug [https://github.com/rust-lang/rust/issues/100013].
    /// 
    /// This is also the reason why we restrict this type here to be [`PrincipalIdealRing`], because
    /// unfortunately, the a constraint `for<'a> SomeRing::LocalRingBase<'a>: PrincipalIdealRing` triggers
    /// the bug in nontrivial settings.
    /// 
    type LocalRingBase<'ring>: ?Sized + PrincipalIdealRing + Domain
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;

    type LocalComputationData<'ring>
        where Self: 'ring;

    ///
    /// Computes (an upper bound of) the natural logarithm of the pseudo norm of a ring element.
    /// 
    /// The pseudo norm should be
    ///  - symmetric, i.e. `|-x| = |x|`,
    ///  - sub-additive, i.e. `|x + y| <= |x| + |y|`
    ///  - sub-multiplicative, i.e. `|xy| <= |x| |y|`
    /// and this function should give `ln|x|`
    /// 
    fn ln_pseudo_norm(&self, el: &Self::Element) -> f64;

    ///
    /// Sets up the context for a new polynomial evaluation, whose output
    /// should have pseudo norm less than the given bound.
    /// 
    fn local_computation<'ring>(&'ring self, ln_pseudo_norm_bound: f64) -> Self::LocalComputationData<'ring>;

    ///
    /// Returns the number `k` of local rings that are required
    /// to get the correct result of the given computation.
    /// 
    fn local_ring_count<'ring>(&self, computation: &Self::LocalComputationData<'ring>) -> usize
        where Self: 'ring;

    ///
    /// Returns the `i`-th local ring belonging to the given computation.
    /// 
    fn local_ring_at<'ring>(&self, computation: &Self::LocalComputationData<'ring>, i: usize) -> Self::LocalRing<'ring>
        where Self: 'ring;

    ///
    /// Computes the map `R -> R1 x ... x Rk`, i.e. maps the given element into each of
    /// the local rings.
    /// 
    fn reduce<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
        where Self: 'ring;

    ///
    /// Computes a preimage under the map `R -> R1 x ... x Rk`, i.e. a ring element `x` that reduces
    /// to each of the given local rings under the map [`EvaluatePolyLocallyRing::reduce()`].
    /// 
    /// The result should have pseudo-norm bounded by the bound given when the computation
    /// was initialized, via [`EvaluatePolyLocallyRing::local_computation()`].
    /// 
    fn lift_combine<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring;
}

impl<I> EvaluatePolyLocallyRing for I
    where I: IntegerRing
{
    type LocalComputationData<'ring> = zn_rns::Zn<AsField<zn_64::Zn>, RingRef<'ring, Self>>
        where Self: 'ring;

    type LocalRing<'ring> = AsField<zn_64::Zn>
        where Self: 'ring;

    type LocalRingBase<'ring> = AsFieldBase<zn_64::Zn>
        where Self: 'ring;

    fn ln_pseudo_norm(&self, el: &Self::Element) -> f64 {
        RingRef::new(self).abs_log2_ceil(el).unwrap_or(0) as f64 * 2f64.ln()
    }

    fn local_computation<'ring>(&'ring self, ln_pseudo_norm_bound: f64) -> Self::LocalComputationData<'ring> {
        let mut primes = Vec::new();
        let mut ln_current = 0.;
        let mut current_value = (1 << 62) / 9;
        while ln_current < ln_pseudo_norm_bound + 1. {
            current_value = prev_prime(StaticRing::<i64>::RING, current_value).unwrap();
            if current_value < (1 << 32) {
                panic!("not enough primes");
            }
            primes.push(current_value);
            ln_current += (current_value as f64).ln();
        }
        return zn_rns::Zn::new(
            primes.into_iter().map(|p| AsField::from(AsFieldBase::promise_is_perfect_field(zn_64::Zn::new(p as u64)))).collect(),
            RingRef::new(self)
        );
    }

    fn local_ring_at<'ring>(&self, computation: &Self::LocalComputationData<'ring>, i: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        computation.at(i).clone()
    }

    fn local_ring_count<'ring>(&self, computation: &Self::LocalComputationData<'ring>) -> usize
        where Self: 'ring
    {
        computation.len()
    }

    fn reduce<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
        where Self: 'ring
    {
        computation.get_congruence(&computation.coerce(RingValue::from_ref(self), self.clone_el(el))).as_iter().map(|x| *x).collect()
    }

    fn lift_combine<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring
    {
        computation.smallest_lift(computation.from_congruence(el.iter().copied()))
    }
}

///
/// The map `R -> R/p` for a ring `R` and one of its local quotients `R/p` as
/// given by [`EvaluatePolyLocallyRing`].
/// 
#[stability::unstable(feature = "enable")]
pub struct ToLocalRingMap<'ring, 'data, R>
    where R: 'ring + ?Sized + EvaluatePolyLocallyRing, 'ring: 'data
{
    ring: RingRef<'data, R>,
    data: &'data R::LocalComputationData<'ring>,
    local_ring: R::LocalRing<'ring>,
    index: usize
}

impl<'ring, 'data, R> ToLocalRingMap<'ring, 'data, R>
    where R: 'ring +?Sized + EvaluatePolyLocallyRing, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, data: &'data R::LocalComputationData<'ring>, index: usize) -> Self {
        Self { ring: RingRef::new(ring), data: data, local_ring: ring.local_ring_at(data, index), index: index }
    }
}

impl<'ring, 'data, R> Homomorphism<R, R::LocalRingBase<'ring>> for ToLocalRingMap<'ring, 'data, R>
    where R: 'ring +?Sized + EvaluatePolyLocallyRing, 'ring: 'data
{
    type CodomainStore = R::LocalRing<'ring>;
    type DomainStore = RingRef<'data, R>;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        &self.local_ring
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        &self.ring
    }

    fn map(&self, x: <R as RingBase>::Element) -> <R::LocalRingBase<'ring> as RingBase>::Element {
        let ring_ref: &'data R = self.ring.into();
        let mut reductions: Vec<<R::LocalRingBase<'ring> as RingBase>::Element> = ring_ref.reduce(self.data, &x);
        return reductions.swap_remove(self.index);
    }
}

#[macro_export]
macro_rules! impl_interpolation_base_ring_char_zero {
    (<{$($gen_args:tt)*}> InterpolationBaseRing for $self_type:ty where $($constraints:tt)*) => {
        impl<$($gen_args)*> $crate::compute_locally::InterpolationBaseRing for $self_type where $($constraints)* {
                
            type ExtendedRing<'a> = RingRef<'a, Self>
                where Self: 'a;

            type ExtendedRingBase<'a> = Self
                where Self: 'a;

            fn in_base<'a, S>(&self, _ext_ring: S, el: El<S>) -> Option<Self::Element>
                where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
            {
                Some(el)
            }

            fn in_extension<'a, S>(&self, _ext_ring: S, el: Self::Element) -> El<S>
                where Self: 'a, S: RingStore<Type = Self::ExtendedRingBase<'a>>
            {
                el
            }

            fn interpolation_points<'a>(&'a self, count: usize) -> (Self::ExtendedRing<'a>, Vec<El<Self::ExtendedRing<'a>>>) {
                let ZZbig = $crate::integer::BigIntRing::RING;
                assert!(ZZbig.is_zero(&self.characteristic(&ZZbig).unwrap()));
                let ring = $crate::ring::RingRef::new(self);
                (ring, (0..count).map(|n| <_ as $crate::homomorphism::Homomorphism<_, _>>::map(&ring.int_hom(), n as i32)).collect())
            }
        }
    };
}