
use crate::divisibility::DivisibilityRing;
use crate::homomorphism::*;
use crate::ring::*;

#[stability::unstable(feature = "enable")]
pub trait InterpolationBaseRing: DivisibilityRing {

    type ExtendedRingBase<'a>: ?Sized + DivisibilityRing
        where Self: 'a;

    type ExtendedRing<'a>: RingStore<Type = Self::ExtendedRingBase<'a>>
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

#[stability::unstable(feature = "enable")]
pub trait ComputeLocallyRing: RingBase {
    
    ///
    /// The proper way would be to define this with two lifetime parameters `'ring` and `'data`,
    /// to allow it to reference both the ring itself and the current `LocalComputationData`.
    /// However, when doing this, I ran into the compiler bug (https://github.com/rust-lang/rust/issues/100013).
    /// 
    type LocalRingBase<'ring>: ?Sized + RingBase
        where Self: 'ring;

    type LocalRing<'ring>: RingStore<Type = Self::LocalRingBase<'ring>>
        where Self: 'ring;

    type LocalComputationData<'ring>
        where Self: 'ring;

    fn pseudo_norm(&self, el: &Self::Element) -> f64;

    fn local_computation<'ring>(&'ring self, uniquely_representable_norm: f64) -> Self::LocalComputationData<'ring>;

    fn local_ring_count<'ring>(&self, data: &Self::LocalComputationData<'ring>) -> usize
        where Self: 'ring;

    fn local_ring_at<'ring>(&self, data: &Self::LocalComputationData<'ring>, i: usize) -> Self::LocalRing<'ring>
        where Self: 'ring;

    fn reduce<'ring>(&self, data: &Self::LocalComputationData<'ring>, el: &Self::Element) -> Vec<<Self::LocalRingBase<'ring> as RingBase>::Element>
        where Self: 'ring;

    fn lift<'ring>(&self, data: &Self::LocalComputationData<'ring>, el: &[<Self::LocalRingBase<'ring> as RingBase>::Element]) -> Self::Element
        where Self: 'ring;
}

#[stability::unstable(feature = "enable")]
pub struct ToLocalRingMap<'ring, 'data, R>
    where R: 'ring + ?Sized + ComputeLocallyRing, 'ring: 'data
{
    ring: RingRef<'data, R>,
    data: &'data R::LocalComputationData<'ring>,
    local_ring: R::LocalRing<'ring>,
    index: usize
}

impl<'ring, 'data, R> ToLocalRingMap<'ring, 'data, R>
    where R: 'ring +?Sized + ComputeLocallyRing, 'ring: 'data
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: &'data R, data: &'data R::LocalComputationData<'ring>, index: usize) -> Self {
        Self { ring: RingRef::new(ring), data: data, local_ring: ring.local_ring_at(data, index), index: index }
    }
}

impl<'ring, 'data, R> Homomorphism<R, R::LocalRingBase<'ring>> for ToLocalRingMap<'ring, 'data, R>
    where R: 'ring +?Sized + ComputeLocallyRing, 'ring: 'data
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