
use crate::algorithms::linsolve::LinSolveRing;
use crate::algorithms::resultant::ComputeResultantRing;
use crate::divisibility::{DivisibilityRing, Domain};
use crate::homomorphism::*;
use crate::pid::PrincipalIdealRing;
use crate::ring::*;
use crate::field::*;
use crate::rings::finite::FiniteRing;
use crate::specialization::FiniteRingSpecializable;

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
    /// Restricting this here to be `DivisibilityRing + PrincipalIdealRing + Domain + EvaluatePolyLocallyRing`
    /// is necessary, because of a compiler bug, see also
    /// [`crate::reduce_lift::poly_eval::EvalPolyLocallyRing::LocalRingBase`]
    /// 
    type ExtendedRingBase<'a>: ?Sized + PrincipalIdealRing + Domain + ComputeResultantRing + LinSolveRing
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
/// Trait for rings that support performing computations locally. 
/// 
/// Note that here (and in `feanor-math` generally), the term "local" is used to refer to algorithms
/// that work modulo prime ideals (or their powers), which is different from the mathematical concept
/// of localization.
/// 
/// More concretely, a ring `R` implementing this trait should be endowed with a "pseudo norm"
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
///    The values `|ai|` are given by [`EvalPolyLocallyRing::pseudo_norm()`].
///  - Get a sufficient number of prime ideals, using [`EvalPolyLocallyRing::local_computation()`] 
///  - Compute `f(a1 mod pi, ..., am mod pi) mod pi` for each prime `pi` within the ring given by 
///    [`EvalPolyLocallyRing::local_ring_at()`]. The reductions `ai mod pj` are given by
///    [`EvalPolyLocallyRing::reduce()`].
///  - Recombine the results to an element of `R` by using [`EvalPolyLocallyRing::lift_combine()`].
/// 
/// # Relationship with [`crate::reduce_lift::poly_factor_gcd::PolyGCDLocallyDomain`]
/// 
/// There are generally two ways of computing something via a reduce-modulo-primes-then-lift
/// approach. Either one can take many different prime ideals, or one can take a large power
/// of a single/a small amount of prime ideals.
/// 
/// This trait is for the former approach, which is especially suitable when the computation to
/// perform can be written as a polynomial evaluation. In particular, this applicable to determinants,
/// resultant, and (with some caveats) solving linear systems.
/// 
/// On the other hand, when factoring polynomials or computing their gcds, it is common to instead
/// rely on Hensel lifting to compute the result modulo a large power of a single prime, or very
/// few primes. This approach is formalized by [`crate::reduce_lift::poly_factor_gcd::PolyGCDLocallyDomain`].
/// 
#[stability::unstable(feature = "enable")]
pub trait EvalPolyLocallyRing: RingBase + FiniteRingSpecializable {
    
    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// to allow it to reference both the ring itself and the current `LocalComputationData`.
    /// However, when doing this, I ran into the compiler bug
    /// [https://github.com/rust-lang/rust/issues/100013](https://github.com/rust-lang/rust/issues/100013).
    /// 
    /// This is also the reason why we restrict this type here to be [`PrincipalIdealRing`], because
    /// unfortunately, the a constraint `for<'a> SomeRing::LocalRingBase<'a>: PrincipalIdealRing` again
    /// triggers the bug in some settings.
    /// 
    type LocalRingBase<'ring>: ?Sized + PrincipalIdealRing + Domain + ComputeResultantRing + LinSolveRing
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;

    ///
    /// A collection of prime ideals of the ring, and additionally any data required to reconstruct
    /// a small ring element from its projections onto each prime ideal.
    /// 
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
    /// to each of the given local rings under the map [`EvalPolyLocallyRing::reduce()`].
    /// 
    /// The result should have pseudo-norm bounded by the bound given when the computation
    /// was initialized, via [`EvalPolyLocallyRing::local_computation()`].
    /// 
    fn lift_combine<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring;
}

impl<R> EvalPolyLocallyRing for R
    where R: ?Sized + FiniteRing + Field
{
    type LocalComputationData<'ring> = RingRef<'ring, Self>
        where Self: 'ring;

    type LocalRing<'ring> = RingRef<'ring, Self>
        where Self: 'ring;

    type LocalRingBase<'ring> = Self
        where Self: 'ring;

    fn ln_pseudo_norm(&self, _el: &Self::Element) -> f64 {
        0.
    }

    fn local_computation<'ring>(&'ring self, _ln_pseudo_norm_bound: f64) -> Self::LocalComputationData<'ring> {
        RingRef::new(self)
    }

    fn local_ring_at<'ring>(&self, computation: &Self::LocalComputationData<'ring>, _i: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        *computation
    }

    fn local_ring_count<'ring>(&self, _computation: &Self::LocalComputationData<'ring>) -> usize
        where Self: 'ring
    {
        1
    }

    fn reduce<'ring>(&self, _computation: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
        where Self: 'ring
    {
        vec![self.clone_el(el)]
    }

    fn lift_combine<'ring>(&self, _computation: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring
    {
        assert_eq!(1, el.len());
        return self.clone_el(&el[0]);
    }
}

///
/// The map `R -> R/p` for a ring `R` and one of its local quotients `R/p` as
/// given by [`EvalPolyLocallyRing`].
/// 
#[stability::unstable(feature = "enable")]
pub struct EvaluatePolyLocallyReductionMap<'ring, 'data, R>
    where R: 'ring + ?Sized + EvalPolyLocallyRing, 'ring: 'data
{
    ring: RingRef<'data, R>,
    data: &'data R::LocalComputationData<'ring>,
    local_ring: R::LocalRing<'ring>,
    index: usize
}

impl<'ring, 'data, R> EvaluatePolyLocallyReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + EvalPolyLocallyRing, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, data: &'data R::LocalComputationData<'ring>, index: usize) -> Self {
        Self { ring: RingRef::new(ring), data: data, local_ring: ring.local_ring_at(data, index), index: index }
    }
}

impl<'ring, 'data, R> Homomorphism<R, R::LocalRingBase<'ring>> for EvaluatePolyLocallyReductionMap<'ring, 'data, R>
    where R: 'ring +?Sized + EvalPolyLocallyRing, 'ring: 'data
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

///
/// Generates an implementation of [`crate::reduce_lift::poly_eval::InterpolationBaseRing`]
/// for a ring of characteristic zero. Not that in this case, the only sensible implementation
/// is trivial, since the ring itself has enough elements for any interpolation task.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::reduce_lift::poly_eval::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::pid::*;
/// # use feanor_math::impl_interpolation_base_ring_char_zero;
/// // we wrap a `RingBase` here for simplicity, but in practice a wrapper should always
/// // store a `RingStore` instead
/// #[derive(PartialEq)]
/// struct MyRingWrapper<R: RingBase>(R);
/// impl<R: RingBase> DelegateRing for MyRingWrapper<R> {
///     type Element = R::Element;
///     type Base = R;
///     fn get_delegate(&self) -> &Self::Base { &self.0 }
///     fn delegate(&self, x: R::Element) -> R::Element { x }
///     fn rev_delegate(&self, x: R::Element) -> R::Element { x }
///     fn delegate_ref<'a>(&self, x: &'a R::Element) -> &'a R::Element { x }
///     fn delegate_mut<'a>(&self, x: &'a mut R::Element) -> &'a mut R::Element { x }
/// }
/// impl<R: RingBase> DelegateRingImplEuclideanRing for MyRingWrapper<R> {}
/// impl<R: Domain> Domain for MyRingWrapper<R> {}
/// impl_interpolation_base_ring_char_zero!{ <{ R }> InterpolationBaseRing for MyRingWrapper<R> where R: PrincipalIdealRing + Domain }
/// 
/// // now we can use `InterpolationBaseRing`-functionality
/// let ring = MyRingWrapper(StaticRing::<i64>::RING.into());
/// let (embedding, points) = ToExtRingMap::for_interpolation(&ring, 3);
/// assert_eq!(0, points[0]);
/// assert_eq!(1, points[1]);
/// assert_eq!(2, points[2]);
/// ```
/// 
#[macro_export]
macro_rules! impl_interpolation_base_ring_char_zero {
    (<{$($gen_args:tt)*}> InterpolationBaseRing for $self_type:ty where $($constraints:tt)*) => {
        impl<$($gen_args)*> $crate::reduce_lift::poly_eval::InterpolationBaseRing for $self_type where $($constraints)* {
                
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
                (ring, (0..count).map(|n| <_ as $crate::homomorphism::Homomorphism<_, _>>::map(&ring.int_hom(), n.try_into().unwrap())).collect())
            }
        }
    };
    (InterpolationBaseRing for $self_type:ty) => {
        impl_interpolation_base_ring_char_zero!{ <{}> InterpolationBaseRing for $self_type where }
    }
}

///
/// Implements [`EvalPolyLocallyRing`] for an integer ring.
/// 
/// This uses a default implementation, where the prime ideals are given by the largest prime numbers
/// such that the corresponding residue field can be implemented using [`crate::rings::zn::zn_64::Zn`]. 
/// This should be suitable in almost all scenarios.
/// 
/// The syntax is the same as for other impl-macros, see e.g. [`crate::impl_interpolation_base_ring_char_zero!`].
/// 
#[macro_export]
macro_rules! impl_eval_poly_locally_for_ZZ {
    (EvalPolyLocallyRing for $int_ring_type:ty) => {
        impl_eval_poly_locally_for_ZZ!{ <{}> EvalPolyLocallyRing for $int_ring_type where }
    };
    (<{$($gen_args:tt)*}> EvalPolyLocallyRing for $int_ring_type:ty where $($constraints:tt)*) => {

        impl<$($gen_args)*> $crate::reduce_lift::poly_eval::EvalPolyLocallyRing for $int_ring_type
            where $($constraints)*
        {
            type LocalComputationData<'ring> = $crate::rings::zn::zn_rns::Zn<$crate::rings::field::AsField<$crate::rings::zn::zn_64::Zn>, RingRef<'ring, Self>>
                where Self: 'ring;

            type LocalRing<'ring> = $crate::rings::field::AsField<$crate::rings::zn::zn_64::Zn>
                where Self: 'ring;

            type LocalRingBase<'ring> = $crate::rings::field::AsFieldBase<$crate::rings::zn::zn_64::Zn>
                where Self: 'ring;

            fn ln_pseudo_norm(&self, el: &Self::Element) -> f64 {
                RingRef::new(self).abs_log2_ceil(el).unwrap_or(0) as f64 * 2f64.ln()
            }

            fn local_computation<'ring>(&'ring self, ln_pseudo_norm_bound: f64) -> Self::LocalComputationData<'ring> {
                let mut primes = Vec::new();
                let mut ln_current = 0.;
                let mut current_value = (1 << 62) / 9;
                while ln_current < ln_pseudo_norm_bound + 1. {
                    current_value = $crate::algorithms::miller_rabin::prev_prime(StaticRing::<i64>::RING, current_value).unwrap();
                    if current_value < (1 << 32) {
                        panic!("not enough primes");
                    }
                    primes.push(current_value);
                    ln_current += (current_value as f64).ln();
                }
                return $crate::rings::zn::zn_rns::Zn::new(
                    primes.into_iter().map(|p| $crate::rings::field::AsField::from($crate::rings::field::AsFieldBase::promise_is_perfect_field($crate::rings::zn::zn_64::Zn::new(p as u64)))).collect(),
                    RingRef::new(self)
                );
            }

            fn local_ring_at<'ring>(&self, computation: &Self::LocalComputationData<'ring>, i: usize) -> Self::LocalRing<'ring>
                where Self: 'ring
            {
                <_ as $crate::seq::VectorView<_>>::at(computation, i).clone()
            }

            fn local_ring_count<'ring>(&self, computation: &Self::LocalComputationData<'ring>) -> usize
                where Self: 'ring
            {
                <_ as $crate::seq::VectorView<_>>::len(computation)
            }

            fn reduce<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
                where Self: 'ring
            {
                <_ as $crate::seq::VectorView<_>>::as_iter(&computation.get_congruence(&computation.coerce(RingValue::from_ref(self), self.clone_el(el)))).map(|x| *x).collect()
            }

            fn lift_combine<'ring>(&self, computation: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
                where Self: 'ring
            {
                <_ as $crate::rings::zn::ZnRingStore>::smallest_lift(computation, computation.from_congruence(el.iter().copied()))
            }
        }
    };
}
