use crate::divisibility::DivisibilityRingStore;
use crate::pid::EuclideanRingStore;
use crate::pid::PrincipalIdealRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::divisibility::DivisibilityRing;
use crate::algorithms;
use crate::integer::*;
use crate::homomorphism::*;
use crate::ordered::*;
use super::field::AsFieldBase;
use super::finite::FiniteRing;
use crate::rings::finite::FiniteRingStore;
use crate::pid::*;

///
/// This module contains [`zn_big::Zn`], a general-purpose implementation of
/// Barett reduction. It is relatively slow when instantiated with small fixed-size
/// integer type. 
/// 
pub mod zn_big;
///
/// This module contains [`zn_64::Zn`], the new, heavily optimized implementation of `Z/nZ`
/// for moduli `n` of size slightly smaller than 64 bits.
/// 
pub mod zn_64;
///
/// This module contains [`zn_static::Zn`], an implementation of `Z/nZ` for a small `n`
/// that is known at compile-time.
/// 
pub mod zn_static;
///
/// This module contains [`zn_rns::Zn`], a residue number system (RNS) implementation of
/// `Z/nZ` for highly composite `n`. 
/// 
pub mod zn_rns;

///
/// Trait for all rings that represent a quotient of the integers `Z/nZ` for some integer `n`.
/// 
pub trait ZnRing: PrincipalIdealRing + FiniteRing + CanHomFrom<Self::IntegerRingBase> {

    /// 
    /// there seems to be a problem with associated type bounds, hence we cannot use `Integers: IntegerRingStore`
    /// or `Integers: RingStore<Type: IntegerRing>`
    /// 
    type IntegerRingBase: IntegerRing + ?Sized;
    type IntegerRing: RingStore<Type = Self::IntegerRingBase>;

    fn integer_ring(&self) -> &Self::IntegerRing;
    fn modulus(&self) -> &El<Self::IntegerRing>;

    ///
    /// Computes the smallest positive lift for some `x` in `Z/nZ`, i.e. the smallest positive integer `m` such that
    /// `m = x mod n`.
    /// 
    /// This will be one of `0, 1, ..., n - 1`. If an integer in `-(n - 1)/2, ..., -1, 0, 1, ..., (n - 1)/2` (for odd `n`)
    /// is needed instead, use [`ZnRing::smallest_lift()`].
    /// 
    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing>;

    ///
    /// Computes any lift for some `x` in `Z/nZ`, i.e. the some integer `m` such that `m = x mod n`.
    /// 
    /// The only requirement is that `m` is a valid element of the integer ring, in particular that
    /// it fits within the required amount of bits, if [`ZnRing::IntegerRing`] is a fixed-size integer ring.
    /// 
    fn any_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        self.smallest_positive_lift(el)
    }

    ///
    /// If the given integer is within `{ 0, ..., n - 1 }`, returns the corresponding
    /// element in `Z/nZ`. Any other input is considered a logic error.
    /// 
    /// Unless the context is absolutely performance-critical, it might be safer to use
    /// the homomorphism provided by [`CanHomFrom`] which performs proper modular reduction
    /// of the input.
    /// 
    /// This function never causes undefined behavior, but an invalid input leads to
    /// a logic error. In particular, the result in such a case does not have to be
    /// congruent to the input mod `n`, nor does it even have to be a valid element
    /// of the ring (i.e. operations involving it may not follow the ring axioms).
    /// Implementors are strongly encouraged to check the element during debug builds. 
    /// 
    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element;

    ///
    /// Computes the smallest lift for some `x` in `Z/nZ`, i.e. the smallest integer `m` such that
    /// `m = x mod n`.
    /// 
    /// This will be one of `-(n - 1)/2, ..., -1, 0, 1, ..., (n - 1)/2` (for odd `n`). If an integer 
    /// in `0, 1, ..., n - 1` is needed instead, use [`ZnRing::smallest_positive_lift()`].
    /// 
    fn smallest_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        let result = self.smallest_positive_lift(el);
        let mut mod_half = self.integer_ring().clone_el(self.modulus());
        self.integer_ring().euclidean_div_pow_2(&mut mod_half, 1);
        if self.integer_ring().is_gt(&result, &mod_half) {
            return self.integer_ring().sub_ref_snd(result, self.modulus());
        } else {
            return result;
        }
    }

    ///
    /// Returns whether this ring is a field, i.e. whether `n` is prime.
    /// 
    fn is_field(&self) -> bool {
        algorithms::miller_rabin::is_prime_base(RingRef::new(self), 10)
    }
}

///
/// Trait for implementations of [`ZnRing`] that can be created (possibly with a 
/// default configuration) from just the integer modulus.
/// 
/// I am not yet sure whether to use this trait, or opt for a factory trait (which
/// would then offer more flexibility).
/// 
#[stability::unstable(feature = "enable")]
pub trait FromModulusCreateableZnRing: Sized + ZnRing {

    fn from_modulus<F, E>(create_modulus: F) -> Result<Self, E>
        where F: FnOnce(&Self::IntegerRingBase) -> Result<El<Self::IntegerRing>, E>;
}

pub mod generic_impls {
    use std::alloc::Global;
    use std::marker::PhantomData;

    use crate::algorithms::convolution::STANDARD_CONVOLUTION;
    use crate::algorithms::int_bisect;
    use crate::ordered::*;
    use crate::primitive_int::{StaticRing, StaticRingBase};
    use crate::field::*;
    use crate::rings::zn::*;
    use crate::divisibility::DivisibilityRingStore;
    use crate::integer::{IntegerRing, IntegerRingStore};
    use crate::rings::extension::galois_field::{GaloisField, GaloisFieldOver};

    ///
    /// A generic `ZZ -> Z/nZ` homomorphism. Optimized for the case that values of `ZZ` can be very
    /// large, but allow for efficient estimation of their approximate size.
    /// 
    
        pub struct BigIntToZnHom<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing>
        where I: CanIsoFromTo<R::IntegerRingBase> + CanIsoFromTo<J>
    {
        highbit_mod: usize,
        highbit_bound: usize,
        int_ring: PhantomData<I>,
        to_large_int_ring: PhantomData<J>,
        hom: <I as CanHomFrom<R::IntegerRingBase>>::Homomorphism,
        iso: <I as CanIsoFromTo<R::IntegerRingBase>>::Isomorphism,
        iso2: <I as CanIsoFromTo<J>>::Isomorphism
    }

    ///
    /// See [`map_in_from_bigint()`].
    /// 
    /// This will only ever return `None` if one of the integer ring `has_canonical_hom/iso` returns `None`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn has_canonical_hom_from_bigint<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing>(from: &I, to: &R, to_large_int_ring: &J, bounded_reduce_bound: Option<&J::Element>) -> Option<BigIntToZnHom<I, J, R>>
        where I: CanIsoFromTo<R::IntegerRingBase> + CanIsoFromTo<J>
    {
        if let Some(bound) = bounded_reduce_bound {
            Some(BigIntToZnHom {
                highbit_mod: to.integer_ring().abs_highest_set_bit(to.modulus()).unwrap(),
                highbit_bound: to_large_int_ring.abs_highest_set_bit(bound).unwrap(),
                int_ring: PhantomData,
                to_large_int_ring: PhantomData,
                hom: from.has_canonical_hom(to.integer_ring().get_ring())?,
                iso: from.has_canonical_iso(to.integer_ring().get_ring())?,
                iso2: from.has_canonical_iso(to_large_int_ring)?
            })
        } else {
            Some(BigIntToZnHom {
                highbit_mod: to.integer_ring().abs_highest_set_bit(to.modulus()).unwrap(),
                highbit_bound: usize::MAX,
                int_ring: PhantomData,
                to_large_int_ring: PhantomData,
                hom: from.has_canonical_hom(to.integer_ring().get_ring())?,
                iso: from.has_canonical_iso(to.integer_ring().get_ring())?,
                iso2: from.has_canonical_iso(to_large_int_ring)?
            })
        }
    }

    ///
    /// A parameterized, generic variant of the reduction `Z -> Z/nZ`.
    /// It considers the following situations:
    ///  - the source ring `Z` might not be large enough to represent `n`
    ///  - the integer ring associated to the destination ring `Z/nZ` might not be large enough to represent the input
    ///  - the destination ring might use Barett reductions (or similar) for fast modular reduction if the input is bounded by some fixed bound `B`
    ///  - general modular reduction modulo `n` is only performed in the source ring if necessary
    /// 
    /// In particular, we use the following additional parameters:
    ///  - `to_large_int_ring`: an integer ring that can represent all integers for which we can perform fast modular reduction (i.e. those bounded by `B`)
    ///  - `from_positive_representative_exact`: a function that performs the restricted reduction `{0, ..., n - 1} -> Z/nZ`
    ///  - `from_positive_representative_bounded`: a function that performs the restricted reduction `{0, ..., B - 1} -> Z/nZ`
    /// 
    /// It first estimates the size of numbers by their bitlength, so don't use this for small integers (i.e. `ixx`-types), as the estimation
    /// is likely to take longer than the actual modular reduction.
    /// 
    /// Note that the input size estimates consider only the bitlength of numbers, and so there is a small margin in which a reduction method for larger
    /// numbers than necessary is used. Furthermore, if the integer rings used can represent some but not all positive numbers of a certain bitlength, 
    /// there might be rare edge cases with panics/overflows. 
    /// 
    /// In particular, if the input integer ring `Z` can represent the input `x`, but not `n` AND `x` and `n` have the same bitlength, this function might
    /// decide that we have to perform generic modular reduction (even though `x < n`), and try to map `n` into `Z`. This is never a problem if the primitive
    /// integer rings `StaticRing::<ixx>::RING` are used, or if `B >= 2n`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn map_in_from_bigint<I: ?Sized + IntegerRing, J: ?Sized + IntegerRing, R: ?Sized + ZnRing, F, G>(from: &I, to: &R, to_large_int_ring: &J, el: I::Element, hom: &BigIntToZnHom<I, J, R>, from_positive_representative_exact: F, from_positive_representative_bounded: G) -> R::Element
        where I: CanIsoFromTo<R::IntegerRingBase> + CanIsoFromTo<J>,
            F: FnOnce(El<R::IntegerRing>) -> R::Element,
            G: FnOnce(J::Element) -> R::Element
    {
        let (neg, n) = if from.is_neg(&el) {
            (true, from.negate(el))
        } else {
            (false, el)
        };
        let ZZ = to.integer_ring().get_ring();
        let highbit_el = from.abs_highest_set_bit(&n).unwrap_or(0);

        let reduced = if highbit_el < hom.highbit_mod {
            from_positive_representative_exact(from.map_out(ZZ, n, &hom.iso))
        } else if highbit_el < hom.highbit_bound {
            from_positive_representative_bounded(from.map_out(to_large_int_ring, n, &hom.iso2))
        } else {
            from_positive_representative_exact(from.map_out(ZZ, from.euclidean_rem(n, &from.map_in_ref(ZZ, to.modulus(), &hom.hom)), &hom.iso))
        };
        if neg {
            to.negate(reduced)
        } else {
            reduced
        }
    }

    ///
    /// Generates a uniformly random element of `Z/nZ` using the randomness of `rng`.
    /// Designed to be used when implementing [`crate::rings::finite::FiniteRing::random_element()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn random_element<R: ZnRing, G: FnMut() -> u64>(ring: &R, rng: G) -> R::Element {
        ring.map_in(
            ring.integer_ring().get_ring(), 
            ring.integer_ring().get_uniformly_random(ring.modulus(), rng), 
            &ring.has_canonical_hom(ring.integer_ring().get_ring()).unwrap()
        )
    }

    ///
    /// Computes the checked division in `Z/nZ`. Designed to be used when implementing
    /// [`crate::divisibility::DivisibilityRing::checked_left_div()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn checked_left_div<R: ZnRingStore>(ring: R, lhs: &El<R>, rhs: &El<R>) -> Option<El<R>>
        where R::Type: ZnRing
    {
        if ring.is_zero(lhs) {
            return Some(ring.zero());
        }
        let int_ring = ring.integer_ring();
        let lhs_lift = ring.smallest_positive_lift(ring.clone_el(lhs));
        let rhs_lift = ring.smallest_positive_lift(ring.clone_el(rhs));
        let (s, _, d) = int_ring.extended_ideal_gen(&rhs_lift, ring.modulus());
        if let Some(quotient) = int_ring.checked_div(&lhs_lift, &d) {
            Some(ring.mul(ring.coerce(int_ring, quotient), ring.coerce(int_ring, s)))
        } else {
            None
        }
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn checked_div_min<R: ZnRingStore>(ring: R, lhs: &El<R>, rhs: &El<R>) -> Option<El<R>>
        where R::Type: ZnRing
    {
        if ring.is_zero(lhs) && ring.is_zero(rhs) {
            return Some(ring.one());
        }
        assert!(ring.is_noetherian());
        let int_ring = ring.integer_ring();
        let rhs_ann = int_ring.checked_div(ring.modulus(), &int_ring.ideal_gen(ring.modulus(), &ring.smallest_positive_lift(ring.clone_el(rhs)))).unwrap();
        let some_sol = ring.smallest_positive_lift(ring.checked_div(lhs, rhs)?);
        let minimal_solution = int_ring.euclidean_rem(some_sol, &rhs_ann);
        if int_ring.is_zero(&minimal_solution) {
            return Some(ring.coerce(&int_ring, rhs_ann));
        } else {
            return Some(ring.coerce(&int_ring, minimal_solution));
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn interpolation_ring<R: ZnRingStore>(ring: R, count: usize) -> GaloisFieldOver<R>
        where R: Clone,
            R::Type: ZnRing + Field + SelfIso + CanHomFrom<StaticRingBase<i64>>
    {
        let ZZbig = BigIntRing::RING;
        let modulus = int_cast(ring.integer_ring().clone_el(ring.modulus()), ZZbig, ring.integer_ring());
        let count = int_cast(count.try_into().unwrap(), ZZbig, StaticRing::<i64>::RING);
        let degree = int_bisect::find_root_floor(StaticRing::<i64>::RING, 1, |d| if *d > 0 && ZZbig.is_gt(&ZZbig.pow(ZZbig.clone_el(&modulus), *d as usize), &count) {
            1
        } else {
            -1
        }) + 1;
        assert!(degree >= 1);
        return GaloisField::new_with_convolution(ring, degree as usize, Global, STANDARD_CONVOLUTION);
    }
}

///
/// The [`crate::ring::RingStore`] corresponding to [`ZnRing`].
/// 
pub trait ZnRingStore: FiniteRingStore
    where Self::Type: ZnRing
{    
    delegate!{ ZnRing, fn integer_ring(&self) -> &<Self::Type as ZnRing>::IntegerRing }
    delegate!{ ZnRing, fn modulus(&self) -> &El<<Self::Type as ZnRing>::IntegerRing> }
    delegate!{ ZnRing, fn smallest_positive_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::IntegerRing> }
    delegate!{ ZnRing, fn smallest_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::IntegerRing> }
    delegate!{ ZnRing, fn any_lift(&self, el: El<Self>) -> El<<Self::Type as ZnRing>::IntegerRing> }
    delegate!{ ZnRing, fn is_field(&self) -> bool }

    fn as_field(self) -> Result<RingValue<AsFieldBase<Self>>, Self> 
        where Self: Sized
    {
        if self.is_field() {
            Ok(RingValue::from(AsFieldBase::promise_is_perfect_field(self)))
        } else {
            Err(self)
        }
    }
}

impl<R: RingStore> ZnRingStore for R
    where R::Type: ZnRing
{}

///
/// Trait for algorithms that require some implementation of
/// `Z/nZ`, but do not care which. 
/// 
/// See [`choose_zn_impl()`] for details.
/// 
pub trait ZnOperation {
    
    type Output<'a>
        where Self: 'a;

    fn call<'a, R>(self, ring: R) -> Self::Output<'a>
        where Self: 'a, 
            R: 'a + RingStore, 
            R::Type: ZnRing, 
            El<R>: Send;
}

///
/// Calls the given function with some implementation of the ring
/// `Z/nZ`, chosen depending on `n` to provide best performance.
/// 
/// It is currently necessary to write all the boilerplate code that
/// comes with manually implementing [`ZnOperation`]. I experimented with
/// macros, but currently something simple seems like the best solution.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::assert_el_eq;
/// 
/// let int_value = 4;
/// // work in Z/17Z without explicitly choosing an implementation
/// struct DoStuff { int_value: i64 }
/// impl ZnOperation for DoStuff {
///     type Output<'a> = ()
///         where Self: 'a;
/// 
///     fn call<'a, R>(self, Zn: R) -> ()
///         where Self: 'a,
///             R: 'a + RingStore,
///             R::Type: ZnRing
///     {
///         let value = Zn.coerce(Zn.integer_ring(), int_cast(self.int_value, Zn.integer_ring(), &StaticRing::<i64>::RING));
///         assert_el_eq!(Zn, Zn.int_hom().map(-1), Zn.mul_ref(&value, &value));
///     } 
/// }
/// choose_zn_impl(StaticRing::<i64>::RING, 17, DoStuff { int_value });
/// ```
/// 
pub fn choose_zn_impl<'a, I, F>(ZZ: I, n: El<I>, f: F) -> F::Output<'a>
    where I: 'a + RingStore,
        I::Type: IntegerRing,
        F: ZnOperation
{
    if ZZ.abs_highest_set_bit(&n).unwrap_or(0) < 57 {
        f.call(zn_64::Zn64B::new(StaticRing::<i64>::RING.coerce(&ZZ, n) as u64))
    } else {
        f.call(zn_big::ZnGB::new(BigIntRing::RING, int_cast(n, &BigIntRing::RING, &ZZ)))
    }
}

#[test]
fn test_choose_zn_impl() {
    let int_value = 4;
    // work in Z/17Z without explicitly choosing an implementation
    struct DoStuff { int_value: i64 }
    impl ZnOperation for DoStuff {

        type Output<'a> = ()
            where Self: 'a;

        fn call<'a, R>(self, Zn: R)
            where R: 'a + RingStore, R::Type: ZnRing
        {
            let value = Zn.coerce(Zn.integer_ring(), int_cast(self.int_value, Zn.integer_ring(), &StaticRing::<i64>::RING));
            assert_el_eq!(Zn, Zn.int_hom().map(-1), Zn.mul_ref(&value, &value));
        } 
    }
    choose_zn_impl(StaticRing::<i64>::RING, 17, DoStuff { int_value });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReductionMapRequirements {
    SmallestLift,
    ExplicitReduce
}

///
/// The homomorphism `Z/nZ -> Z/mZ` that exists whenever `m | n`. In
/// addition to the map, this also provides a function [`ZnReductionMap::smallest_lift()`]
/// that computes the "smallest" preimage under the map, and a function
/// [`ZnReductionMap::mul_quotient_fraction()`], that computes the multiplication
/// with `n/m` while also changing from `Z/mZ` to `Z/nZ`. This is very
/// useful in many number theoretic applications, where one often has to switch
/// between `Z/nZ` and `Z/mZ`.
/// 
/// Furthermore, many implementations of `ZnRing` currently do not support
/// [`CanHomFrom`]-homomorphisms when the moduli are different (but divide each
/// other).
/// 
pub struct ZnReductionMap<R, S>
    where R: RingStore,
        R::Type: ZnRing,
        S: RingStore,
        S::Type: ZnRing
{
    from: R,
    to: S,
    fraction_of_quotients: El<R>,
    to_modulus: El<<R::Type as ZnRing>::IntegerRing>,
    to_from_int: <S::Type as CanHomFrom<<S::Type as ZnRing>::IntegerRingBase>>::Homomorphism,
    from_from_int: <R::Type as CanHomFrom<<R::Type as ZnRing>::IntegerRingBase>>::Homomorphism,
    map_forward_requirement: ReductionMapRequirements
}

impl<R, S> ZnReductionMap<R, S>
    where R: RingStore,
        R::Type: ZnRing,
        S: RingStore,
        S::Type: ZnRing
{
    pub fn new(from: R, to: S) -> Option<Self> {
        let from_char = from.characteristic(&BigIntRing::RING).unwrap();
        let to_char = to.characteristic(&BigIntRing::RING).unwrap();
        if let Some(frac) = BigIntRing::RING.checked_div(&from_char, &to_char) {
            let map_forward_requirement: ReductionMapRequirements = if to.integer_ring().get_ring().representable_bits().is_none() || BigIntRing::RING.is_lt(&from_char, &BigIntRing::RING.power_of_two(to.integer_ring().get_ring().representable_bits().unwrap())) {
                ReductionMapRequirements::SmallestLift
            } else {
                ReductionMapRequirements::ExplicitReduce
            };
            Some(Self {
                map_forward_requirement: map_forward_requirement,
                to_modulus: int_cast(to.integer_ring().clone_el(to.modulus()), from.integer_ring(), to.integer_ring()),
                to_from_int: to.get_ring().has_canonical_hom(to.integer_ring().get_ring()).unwrap(),
                from_from_int: from.get_ring().has_canonical_hom(from.integer_ring().get_ring()).unwrap(),
                fraction_of_quotients: from.can_hom(from.integer_ring()).unwrap().map(int_cast(frac, from.integer_ring(), BigIntRing::RING)),
                from: from,
                to: to,
            })
        } else {
            None
        }
    }

    ///
    /// Computes the additive group homomorphism `Z/mZ -> Z/nZ, x -> (n/m)x`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let Z5 = Zn::new(5);
    /// let Z25 = Zn::new(25);
    /// let f = ZnReductionMap::new(&Z25, &Z5).unwrap();
    /// assert_el_eq!(Z25, Z25.int_hom().map(15), f.mul_quotient_fraction(Z5.int_hom().map(3)));
    /// ```
    /// 
    pub fn mul_quotient_fraction(&self, x: El<S>) -> El<R> {
        self.from.mul_ref_snd(self.any_preimage(x), &self.fraction_of_quotients)
    }

    ///
    /// Computes the smallest preimage under the reduction map `Z/nZ -> Z/mZ`, where
    /// "smallest" refers to the element that has the smallest lift to `Z`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let Z5 = Zn::new(5);
    /// let Z25 = Zn::new(25);
    /// let f = ZnReductionMap::new(&Z25, &Z5).unwrap();
    /// assert_el_eq!(Z25, Z25.int_hom().map(-2), f.smallest_lift(Z5.int_hom().map(3)));
    /// ```
    /// 
    pub fn smallest_lift(&self, x: El<S>) -> El<R> {
        self.from.get_ring().map_in(self.from.integer_ring().get_ring(), int_cast(self.to.smallest_lift(x), self.from.integer_ring(), self.to.integer_ring()), &self.from_from_int)
    }

    pub fn any_preimage(&self, x: El<S>) -> El<R> {
        // the problem is that we don't know if `to.any_lift(x)` will fit into `from.integer_ring()`;
        // furthermore, profiling indicates that it won't help a lot anyway, since taking the smallest lift
        // now will usually make reduction cheaper later
        self.smallest_lift(x)
    }

    pub fn smallest_lift_ref(&self, x: &El<S>) -> El<R> {
        self.smallest_lift(self.codomain().clone_el(x))
    }
}

impl<R, S> Homomorphism<R::Type, S::Type> for ZnReductionMap<R, S>
    where R: RingStore,
        R::Type: ZnRing,
        S: RingStore,
        S::Type: ZnRing
{
    type CodomainStore = S;
    type DomainStore = R;

    fn map(&self, x: El<R>) -> El<S> {
        let value = match self.map_forward_requirement {
            ReductionMapRequirements::SmallestLift => self.from.smallest_lift(x),
            ReductionMapRequirements::ExplicitReduce => self.from.integer_ring().euclidean_rem(self.from.any_lift(x), &self.to_modulus)
        };
        self.to.get_ring().map_in(self.to.integer_ring().get_ring(), int_cast(value, self.to.integer_ring(), self.from.integer_ring()), &self.to_from_int)
    }

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.to
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.from
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use super::*;
    use crate::primitive_int::{StaticRingBase, StaticRing};

    pub fn test_zn_axioms<R: RingStore>(R: R)
        where R::Type: ZnRing,
            <R::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<StaticRingBase<i128>> + CanIsoFromTo<StaticRingBase<i32>>
    {
        let ZZ = R.integer_ring();
        let n = R.modulus();

        assert!(R.is_zero(&R.coerce(ZZ, ZZ.clone_el(n))));
        assert!(R.is_field() == algorithms::miller_rabin::is_prime(ZZ, n, 10));

        let mut k = ZZ.one();
        while ZZ.is_lt(&k, &n) {
            assert!(!R.is_zero(&R.coerce(ZZ, ZZ.clone_el(&k))));
            ZZ.add_assign(&mut k, ZZ.one());
        }

        let all_elements = R.elements().collect::<Vec<_>>();
        assert_eq!(int_cast(ZZ.clone_el(n), &StaticRing::<i128>::RING, &ZZ) as usize, all_elements.len());
        for (i, x) in all_elements.iter().enumerate() {
            for (j, y) in all_elements.iter().enumerate() {
                assert!(i == j || !R.eq_el(x, y));
            }
        }
    }

    pub fn test_map_in_large_int<R: RingStore>(R: R)
        where <R as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
    {
        let ZZ_big = BigIntRing::RING;
        let n = ZZ_big.power_of_two(1000);
        let x = R.coerce(&ZZ_big, n);
        assert!(R.eq_el(&R.pow(R.int_hom().map(2), 1000), &x));
    }
}

#[test]
fn test_reduction_map_large_value() {
    let ring1 = zn_64::Zn64B::new(1 << 42);
    let ring2 = zn_big::ZnGB::new(BigIntRing::RING, BigIntRing::RING.power_of_two(666));
    let reduce = ZnReductionMap::new(&ring2, ring1).unwrap();
    assert_el_eq!(ring1, ring1.zero(), reduce.map(ring2.pow(ring2.int_hom().map(2), 665)));
}

#[test]
fn test_reduction_map() {
    let ring1 = zn_64::Zn64B::new(257);
    let ring2 = zn_big::ZnGB::new(StaticRing::<i128>::RING, 257 * 7);

    crate::homomorphism::generic_tests::test_homomorphism_axioms(ZnReductionMap::new(&ring2, &ring1).unwrap(), ring2.elements().step_by(8));

    let ring1 = zn_big::ZnGB::new(StaticRing::<i16>::RING, 3);
    let ring2 = zn_big::ZnGB::new(BigIntRing::RING, BigIntRing::RING.int_hom().map(65537 * 3));

    crate::homomorphism::generic_tests::test_homomorphism_axioms(ZnReductionMap::new(&ring2, &ring1).unwrap(), ring2.elements().step_by(1024));
}

#[test]
fn test_generic_impl_checked_div_min() {
    let ring = zn_64::Zn64B::new(5 * 7 * 11 * 13);
    let actual = ring.annihilator(&ring.int_hom().map(1001));
    let expected = ring.int_hom().map(5);
    assert!(ring.checked_div(&expected, &actual).is_some());
    assert!(ring.checked_div(&actual, &expected).is_some());

    let actual = ring.annihilator(&ring.zero());
    let expected = ring.one();
    assert!(ring.checked_div(&expected, &actual).is_some());
    assert!(ring.checked_div(&actual, &expected).is_some());
}