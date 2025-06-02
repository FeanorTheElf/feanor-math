use crate::algorithms;
use crate::reduce_lift::poly_factor_gcd::IntegerPolyGCDRing;
use crate::reduce_lift::poly_eval::EvalPolyLocallyRing;
use crate::divisibility::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::pid::*;
use crate::ordered::*;

///
/// Type alias for the current default used big integer ring implementation.
/// 
/// The type this points to may change when features or other compilation parameters
/// change.
///
#[cfg(feature = "mpir")]
pub type BigIntRing = crate::rings::mpir::MPZ;
///
/// Type alias for the current default used big integer ring implementation.
/// 
/// The type this points to may change when features or other compilation parameters
/// change.
/// 
#[cfg(not(feature = "mpir"))]
pub type BigIntRing = crate::rings::rust_bigint::RustBigintRing;
///
/// Type alias for the current default used big integer ring implementation.
/// 
/// The type this points to may change when features or other compilation parameters
/// change.
/// 
#[cfg(feature = "mpir")]
pub type BigIntRingBase = crate::rings::mpir::MPZBase;
///
/// Type alias for the current default used big integer ring implementation.
/// 
/// The type this points to may change when features or other compilation parameters
/// change.
/// 
#[cfg(not(feature = "mpir"))]
pub type BigIntRingBase = crate::rings::rust_bigint::RustBigintRingBase;

///
/// Trait for rings that are isomorphic to the ring of integers `ZZ = { ..., -2, -1, 0, 1, 2, ... }`.
/// 
/// Some of the functionality in this trait refers to the binary expansion of
/// a positive integer. While this is not really general, it is often required
/// for fast operations with integers.
/// 
/// As an additional requirement, the euclidean division (i.e. [`EuclideanRing::euclidean_div_rem()`] and
/// [`IntegerRing::euclidean_div_pow_2()`]) are additionally expected to round towards zero.
/// 
/// Currently [`IntegerRing`] is a subtrait of the unstable traits [`IntegerPolyGCDRing`] and,
/// [`EvalPolyLocallyRing`] so it is at the moment impossible to implement [`IntegerRing`] for a
/// custom ring type without enabling unstable features. Sorry.
/// 
pub trait IntegerRing: Domain + EuclideanRing + OrderedRing + HashableElRing + IntegerPolyGCDRing + EvalPolyLocallyRing {

    ///
    /// Computes a float value that is "close" to the given integer.
    /// 
    /// However, no guarantees are made on how close it must be, in particular,
    /// this function may also always return `0.` (this is just an example - 
    /// it's not a good idea).
    /// 
    /// Some use cases include:
    ///  - Estimating control parameters (e.g. bounds for prime numbers
    ///    during factoring algorithms)
    ///  - First performing some operation on floating point numbers, and
    ///    then refining it to integers.
    /// 
    /// Note that a high-quality implementation of this function can vastly improve
    /// performance in some cases, e.g. of [`crate::algorithms::int_bisect::root_floor()`] or 
    /// [`crate::algorithms::lll::lll_exact()`].
    /// 
    /// # Example
    /// 
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::integer::*;
    /// let ZZ = BigIntRing::RING;
    /// let x = ZZ.power_of_two(1023);
    /// assert!(ZZ.to_float_approx(&x) > 2f64.powi(1023) * 0.99999);
    /// assert!(ZZ.to_float_approx(&x) < 2f64.powi(1023) * 1.000001);
    /// ```
    /// If the value is too large for the exponent of a `f64`, infinity is returned.
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::integer::*;
    /// let ZZ = BigIntRing::RING;
    /// let x = ZZ.power_of_two(1024);
    /// assert!(ZZ.to_float_approx(&x).is_infinite());
    /// ```
    /// 
    fn to_float_approx(&self, value: &Self::Element) -> f64;

    ///
    /// Computes a value that is "close" to the given float. However, no guarantees
    /// are made on the definition of close, in particular, this does not have to be
    /// the closest integer to the given float, and cannot be used to compute rounding.
    /// It is also implementation-defined when to return `None`, although this is usually
    /// the case on infinity and NaN.
    /// 
    /// For information when to use (or not use) this, see its counterpart [`IntegerRing::to_float_approx()`].
    /// 
    fn from_float_approx(&self, value: f64) -> Option<Self::Element>;

    ///
    /// Return whether the `i`-th bit in the two-complements representation of `abs(value)`
    /// is `1`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(false, StaticRing::<i32>::RING.abs_is_bit_set(&4, 1));
    /// assert_eq!(true, StaticRing::<i32>::RING.abs_is_bit_set(&4, 2));
    /// assert_eq!(true, StaticRing::<i32>::RING.abs_is_bit_set(&-4, 2));
    /// ```
    /// 
    fn abs_is_bit_set(&self, value: &Self::Element, i: usize) -> bool;

    ///
    /// Returns the index of the highest set bit in the two-complements representation of `abs(value)`,
    /// or `None` if the value is zero.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(None, StaticRing::<i32>::RING.abs_highest_set_bit(&0));
    /// assert_eq!(Some(0), StaticRing::<i32>::RING.abs_highest_set_bit(&-1));
    /// assert_eq!(Some(2), StaticRing::<i32>::RING.abs_highest_set_bit(&4));
    /// ```
    /// 
    fn abs_highest_set_bit(&self, value: &Self::Element) -> Option<usize>;

    ///
    /// Returns the index of the lowest set bit in the two-complements representation of `abs(value)`,
    /// or `None` if the value is zero.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(None, StaticRing::<i32>::RING.abs_lowest_set_bit(&0));
    /// assert_eq!(Some(0), StaticRing::<i32>::RING.abs_lowest_set_bit(&1));
    /// assert_eq!(Some(0), StaticRing::<i32>::RING.abs_lowest_set_bit(&-3));
    /// assert_eq!(Some(2), StaticRing::<i32>::RING.abs_lowest_set_bit(&4));
    /// ```
    /// 
    fn abs_lowest_set_bit(&self, value: &Self::Element) -> Option<usize>;

    ///
    /// Computes the euclidean division by a power of two, always rounding to zero (note that
    /// euclidean division requires that `|remainder| < |divisor|`, and thus would otherwise
    /// leave multiple possible results).
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// let mut value = -7;
    /// StaticRing::<i32>::RING.euclidean_div_pow_2(&mut value, 1);
    /// assert_eq!(-3, value);
    /// ```
    /// 
    fn euclidean_div_pow_2(&self, value: &mut Self::Element, power: usize);

    ///
    /// Multiplies the element by a power of two.
    /// 
    fn mul_pow_2(&self, value: &mut Self::Element, power: usize);

    ///
    /// Computes a uniformly random integer in `[0, 2^log_bound_exclusive - 1]`, assuming that
    /// `rng` provides uniformly random values in the whole range of `u64`.
    /// 
    fn get_uniformly_random_bits<G: FnMut() -> u64>(&self, log2_bound_exclusive: usize, rng: G) -> Self::Element;

    ///
    /// Computes the rounded division, i.e. rounding to the closest integer.
    /// In the case of a tie (i.e. `round(0.5)`), we round towards `+/- infinity`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(2, StaticRing::<i32>::RING.rounded_div(7, &3));
    /// assert_eq!(-2, StaticRing::<i32>::RING.rounded_div(-7, &3));
    /// assert_eq!(-2, StaticRing::<i32>::RING.rounded_div(7, &-3));
    /// assert_eq!(2, StaticRing::<i32>::RING.rounded_div(-7, &-3));
    /// 
    /// assert_eq!(3, StaticRing::<i32>::RING.rounded_div(8, &3));
    /// assert_eq!(-3, StaticRing::<i32>::RING.rounded_div(-8, &3));
    /// assert_eq!(-3, StaticRing::<i32>::RING.rounded_div(8, &-3));
    /// assert_eq!(3, StaticRing::<i32>::RING.rounded_div(-8, &-3));
    /// 
    /// assert_eq!(4, StaticRing::<i32>::RING.rounded_div(7, &2));
    /// assert_eq!(-4, StaticRing::<i32>::RING.rounded_div(-7, &2));
    /// assert_eq!(-4, StaticRing::<i32>::RING.rounded_div(7, &-2));
    /// assert_eq!(4, StaticRing::<i32>::RING.rounded_div(-7, &-2));
    /// ```
    /// 
    fn rounded_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut rhs_half = self.abs(self.clone_el(rhs));
        self.euclidean_div_pow_2(&mut rhs_half, 1);
        if self.is_neg(&lhs) {
            return self.euclidean_div(self.sub(lhs, rhs_half), rhs);
        } else {
            return self.euclidean_div(self.add(lhs, rhs_half), rhs);
        }
    }

    ///
    /// Computes the division `lhs / rhs`, rounding towards `+ infinity`.
    /// 
    /// In particular, if `rhs` is positive, this gives the smallest
    /// integer `quo` such that `quo * rhs >= lhs`. On the other hand, if
    /// `rhs` is negative, this computes the largest integer `quo` such that
    /// `quo * rhs <= lhs`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(3, StaticRing::<i32>::RING.ceil_div(7, &3));
    /// assert_eq!(-2, StaticRing::<i32>::RING.ceil_div(-7, &3));
    /// assert_eq!(-2, StaticRing::<i32>::RING.ceil_div(7, &-3));
    /// assert_eq!(3, StaticRing::<i32>::RING.ceil_div(-7, &-3));
    /// ```
    /// 
    fn ceil_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        if self.is_zero(&lhs) {
            return self.zero();
        }
        let one = self.one();
        return match (self.is_pos(&lhs), self.is_pos(rhs)) {
            (true, true) => self.add(self.euclidean_div(self.sub_ref_snd(lhs, &one), rhs), one),
            (false, true) => self.euclidean_div(lhs, rhs),
            (true, false) => self.euclidean_div(lhs, rhs),
            (false, false) => self.add(self.euclidean_div(self.add_ref_snd(lhs, &one), rhs), one)
        };
    }

    ///
    /// Computes the division `lhs / rhs`, rounding towards `- infinity`.
    /// 
    /// In particular, if `rhs` is positive, this gives the largest
    /// integer `quo` such that `quo * rhs <= lhs`. On the other hand, if
    /// `rhs` is negative, this computes the smallest integer `quo` such that
    /// `quo * rhs >= lhs`.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::ring::*;
    /// assert_eq!(2, StaticRing::<i32>::RING.floor_div(7, &3));
    /// assert_eq!(-3, StaticRing::<i32>::RING.floor_div(-7, &3));
    /// assert_eq!(-3, StaticRing::<i32>::RING.floor_div(7, &-3));
    /// assert_eq!(2, StaticRing::<i32>::RING.floor_div(-7, &-3));
    /// ```
    /// 
    fn floor_div(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.negate(self.ceil_div(self.negate(lhs), rhs))
    }

    ///
    /// Returns the value `2^power` in this integer ring.
    /// 
    fn power_of_two(&self, power: usize) -> Self::Element {
        let mut result = self.one();
        self.mul_pow_2(&mut result, power);
        return result;
    }

    ///
    /// Returns `n` such that this ring can represent at least `[-2^n, ..., 2^n - 1]`.
    /// Returning `None` means that the size of representable integers is unbounded.
    /// 
    fn representable_bits(&self) -> Option<usize>;

    ///
    /// Parses the given string as a number.
    /// 
    /// Returns `Err(())` if it is not a valid number w.r.t. base, i.e. if the string
    /// is not a sequence of digit characters, optionally beginning with `+` or `-`. 
    /// To denote digits larger than `9`, the same characters as in [`u64::from_str_radix()`]
    /// should be used.
    /// 
    fn parse(&self, string: &str, base: u32) -> Result<Self::Element, ()> {
        generic_impls::parse(RingRef::new(self), string, base)
    }
}

impl<I, J> CanHomFrom<I> for J
    where I: ?Sized + IntegerRing, J: ?Sized + IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, _: &I) -> Option<Self::Homomorphism> {
        Some(())
    }

    fn map_in(&self, from: &I, el: <I as RingBase>::Element, _: &Self::Homomorphism) -> Self::Element {
        int_cast(el, &RingRef::new(self), &RingRef::new(from))
    }

    default fn map_in_ref(&self, from: &I, el: &<I as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        <J as CanHomFrom<I>>::map_in(self, from, from.clone_el(el), hom)
    }
}

impl<I, J> CanIsoFromTo<I> for J
    where I: ?Sized + IntegerRing, J: ?Sized + IntegerRing
{
    type Isomorphism = ();

    fn has_canonical_iso(&self, _: &I) -> Option<Self::Isomorphism> {
        Some(())
    }

    fn map_out(&self, from: &I, el: Self::Element, _: &Self::Isomorphism) -> <I as RingBase>::Element {
        int_cast(el, &RingRef::new(from), &RingRef::new(self))
    }
}

///
/// Helper trait to simplify conversion between ints.
/// 
/// More concretely, `IntCast` defines a conversion between two
/// integer rings, and is default-implemented for all integer rings
/// using a double-and-and technique. All implementors of integer
/// rings are encouraged to provide specializations for improved performance.
/// 
/// # Why yet another conversion trait?
/// 
/// It is a common requirement to convert between arbitrary (i.e. generic)
/// integer rings. To achieve this, we require a blanket implementation
/// anyway.
/// 
/// Now it would be possible to just provide a blanket implementation of
/// [`CanHomFrom`] and specialize it for all integer rings. However, specialization
/// with default types is currently a pain in the ass. Furthermore, this trait is simpler.
/// 
pub trait IntCast<F: ?Sized + IntegerRing>: IntegerRing {

    ///
    /// Maps the given integer into this ring.
    /// 
    /// For the difference to [`RingStore::coerce()`] or [`RingStore::can_hom()`],
    /// see the documentation of [`IntCast`].
    /// 
    fn cast(&self, from: &F, value: F::Element) -> Self::Element;
}

impl<F: ?Sized + IntegerRing, T: ?Sized + IntegerRing> IntCast<F> for T {

    default fn cast(&self, from: &F, value: F::Element) -> Self::Element {
        let result = algorithms::sqr_mul::generic_abs_square_and_multiply(self.one(), &value, RingRef::new(from), |a| self.add_ref(&a, &a), |a, b| self.add_ref_fst(a, b), self.zero());
        if from.is_neg(&value) {
            return self.negate(result);
        } else {
            return result;
        }
    }
}

///
/// Conversion of elements between two rings representing the integers `ZZ`.
/// 
/// The underlying conversion functionality is the same as provided by [`IntCast`], and
/// indirectly also by [`CanHomFrom`] and [`CanIsoFromTo`].
/// 
/// # Example
/// 
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::assert_el_eq;
/// let ZZi64 = StaticRing::<i64>::RING;
/// let ZZbig = BigIntRing::RING;
/// let ZZi8 = StaticRing::<i8>::RING;
/// assert_eq!(7, int_cast(7, ZZi64, ZZi8));
/// assert_eq!(65536, int_cast(ZZbig.power_of_two(16), ZZi64, ZZbig));
/// assert_el_eq!(ZZbig, ZZbig.power_of_two(16), int_cast(65536, ZZbig, ZZi64));
///  ```
/// 
pub fn int_cast<T: RingStore, F: RingStore>(value: El<F>, to: T, from: F) -> El<T>
    where T::Type: IntegerRing, F::Type: IntegerRing
{
    <T::Type as IntCast<F::Type>>::cast(to.get_ring(), from.get_ring(), value)
}

///
/// Computes the binomial coefficient of `n` and `k`, defined as `n(n - 1)...(n - k + 1)/k!`.
/// 
/// The above definition works for any `n` and `k >= 0`. If `k < 0`, we define the binomial coefficient
/// to be zero. This function will not overflow, if the integer rings supports number up to 
/// `binomial(n, k) * k`.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::iters::*;
/// # use feanor_math::primitive_int::*;
/// // the binomial coefficient is equal to the number of combinations of fixed size
/// assert_eq!(
///     binomial(10, &3, StaticRing::<i64>::RING) as usize,
///     multiset_combinations(&[1; 10], 3, |_| ()).count()
/// );
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn binomial<I>(n: El<I>, k: &El<I>, ring: I) -> El<I>
    where I: RingStore + Copy,
        I::Type: IntegerRing
{
    if ring.is_neg(&n) {
        let mut result = binomial(ring.sub(ring.sub_ref_fst(&k, n), ring.one()), k, ring);
        if !ring.is_even(k) {
            ring.negate_inplace(&mut result);
        }
        return result;
    } else if ring.is_neg(k) || ring.is_gt(k, &n) {
        return ring.zero();
    } else {
        // this formula works always, and is guaranteed not to overflow if k <= n/2 and `binomial(n, k) * k` 
        // fits into an integer; thus distinguish this case that k > n/2
        let n_neg_k = ring.sub_ref(&n, &k);
        if ring.is_lt(&n_neg_k, k) {
            return binomial(n, &n_neg_k, ring);
        }
        let mut result = ring.one();
        let mut i = ring.one();
        while ring.is_leq(&i, &k) {
            ring.mul_assign(&mut result, ring.sub_ref_snd(ring.add_ref_fst(&n, ring.one()), &i));
            result = ring.checked_div(&result, &i).unwrap();
            ring.add_assign(&mut i, ring.one());
        }
        return result;
    }
}

#[stability::unstable(feature = "enable")]
pub fn factorial<I>(n: &El<I>, ring: I) -> El<I>
    where I: RingStore + Copy,
        I::Type: IntegerRing
{
    let mut current = ring.zero();
    let one = ring.one();
    return ring.prod((0..).map_while(|_| {
        if ring.is_lt(&current, &n) {
            ring.add_assign_ref(&mut current, &one);
            return Some(ring.clone_el(&current));
        } else {
            return None;
        }
    }));
}

///
/// Trait for [`RingStore`]s that store [`IntegerRing`]s. Mainly used
/// to provide a convenient interface to the `IntegerRing`-functions.
/// 
pub trait IntegerRingStore: RingStore
    where Self::Type: IntegerRing
{
    delegate!{ IntegerRing, fn to_float_approx(&self, value: &El<Self>) -> f64 }
    delegate!{ IntegerRing, fn from_float_approx(&self, value: f64) -> Option<El<Self>> }
    delegate!{ IntegerRing, fn abs_is_bit_set(&self, value: &El<Self>, i: usize) -> bool }
    delegate!{ IntegerRing, fn abs_highest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ IntegerRing, fn abs_lowest_set_bit(&self, value: &El<Self>) -> Option<usize> }
    delegate!{ IntegerRing, fn euclidean_div_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ IntegerRing, fn mul_pow_2(&self, value: &mut El<Self>, power: usize) -> () }
    delegate!{ IntegerRing, fn power_of_two(&self, power: usize) -> El<Self> }
    delegate!{ IntegerRing, fn rounded_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ IntegerRing, fn floor_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ IntegerRing, fn ceil_div(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ IntegerRing, fn parse(&self, string: &str, base: u32) -> Result<El<Self>, ()> }

    ///
    /// Using the randomness of the given rng, samples a uniformly random integer
    /// from the set `{ 0, 1, ..., bound_exclusive - 1 }`.
    /// 
    /// Uses rejection sampling on top of [`IntegerRing::get_uniformly_random_bits()`].
    /// 
    fn get_uniformly_random<G: FnMut() -> u64>(&self, bound_exclusive: &El<Self>, mut rng: G) -> El<Self> {
        assert!(self.is_gt(bound_exclusive, &self.zero()));
        let log2_ceil_bound = self.abs_highest_set_bit(bound_exclusive).unwrap() + 1;
        let mut result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        while self.is_geq(&result, bound_exclusive) {
            result = self.get_ring().get_uniformly_random_bits(log2_ceil_bound, || rng());
        }
        return result;
    }

    ///
    /// Computes `floor(log2(abs(value)))`, and returns `None` if the argument is 0.
    /// 
    /// This is equivalent to [`IntegerRingStore::abs_highest_set_bit`].
    /// 
    fn abs_log2_floor(&self, value: &El<Self>) -> Option<usize> {
        self.abs_highest_set_bit(value)
    }

    ///
    /// Computes `ceil(log2(abs(value)))`, and returns `None` if the argument is 0.
    /// 
    fn abs_log2_ceil(&self, value: &El<Self>) -> Option<usize> {
        let highest_bit = self.abs_highest_set_bit(value)?;
        if self.abs_lowest_set_bit(value).unwrap() == highest_bit {
            return Some(highest_bit);
        } else {
            return Some(highest_bit + 1);
        }
    }

    ///
    /// Returns true if the given integer is divisible by 2.
    /// 
    fn is_even(&self, value: &El<Self>) -> bool {
        !self.abs_is_bit_set(value, 0)
    }

    ///
    /// Returns true if the given integer is not divisible by 2.
    /// 
    fn is_odd(&self, value: &El<Self>) -> bool {
        !self.is_even(value)
    }

    ///
    /// Assumes the given integer is even, and computes its quotient by 2.
    /// 
    fn half_exact(&self, mut value: El<Self>) -> El<Self> {
        assert!(self.is_even(&value));
        self.euclidean_div_pow_2(&mut value, 1);
        return value;
    }
}

impl<R> IntegerRingStore for R
    where R: RingStore,
        R::Type: IntegerRing
{}

pub mod generic_impls {
    use crate::ring::*;
    use crate::primitive_int::*;
    use super::*;
    
    #[stability::unstable(feature = "enable")]
    pub fn parse<I>(ring: I, string: &str, base: u32) -> Result<El<I>, ()>
        where I: RingStore, I::Type: IntegerRing
    {
        let (negative, rest) = if string.chars().next() == Some('-') {
            (true, string.split_at(1).1)
        } else if string.chars().next() == Some('+') {
            (false, string.split_at(1).1)
        } else {
            (false, string)
        };
        assert!(base >= 2);

        let bits_per_chunk = u32::BITS as usize;
        assert!(bits_per_chunk <= i64::BITS as usize - 2);
        let chunk_size = (bits_per_chunk as f32 / (base as f32).log2()).floor() as usize;
        let it = <str as AsRef<[u8]>>::as_ref(rest)
            .rchunks(chunk_size)
            .rev()
            .map(std::str::from_utf8)
            .map(|chunk| chunk.map_err(|_| ()))
            .map(|chunk| chunk.and_then(|n| 
                u64::from_str_radix(n, base).map_err(|_| ()))
            );
        let chunk_base = ring.pow(int_cast(base as i64, &ring, StaticRing::<i64>::RING), chunk_size);
        let result = it.fold(Ok(ring.zero()), |current, next| {
            current.and_then(|mut current| next.map(|next| {
                ring.mul_assign_ref(&mut current, &chunk_base);
                ring.add(current, int_cast(next as i64, &ring, StaticRing::<i64>::RING))
            }))
        });
        if negative {
            return result.map(|result| ring.negate(result));
        } else {
            return result;
        }
    }
}

#[cfg(test)]
use crate::primitive_int::*;

#[allow(missing_docs)]
#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::ring::El;
    use super::*;
        
    pub fn test_integer_get_uniformly_random<R: RingStore>(ring: R) 
        where R::Type: IntegerRing
    {
        for b in [15, 16] {
            let bound = ring.int_hom().map(b);
            let mut rng = oorandom::Rand64::new(1);
            let elements: Vec<El<R>> = (0..1000).map(|_| ring.get_uniformly_random(&bound, || rng.rand_u64())).collect();
            for i in 0..b {
                assert!(elements.iter().any(|x| ring.eq_el(x, &ring.int_hom().map(i))))
            }
            for x in &elements {
                assert!(ring.is_lt(x, &bound));
            }
        }
    }

    pub fn test_integer_axioms<R: IntegerRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I) 
        where R::Type: IntegerRing
    {
        let elements = edge_case_elements.collect::<Vec<_>>();

        // test abs_highest_set_bit on standard values
        assert_eq!(None, ring.abs_highest_set_bit(&ring.int_hom().map(0)));
        assert_eq!(Some(0), ring.abs_highest_set_bit(&ring.int_hom().map(1)));
        assert_eq!(Some(1), ring.abs_highest_set_bit(&ring.int_hom().map(2)));

        // generic test of mul_pow_2 resp. euclidean_div_pow_2
        for a in &elements {
            let mut ceil_pow_2 = ring.int_hom().map(2);
            ring.mul_pow_2(&mut ceil_pow_2, ring.abs_highest_set_bit(a).unwrap_or(0));
            assert!(ring.is_lt(a, &ceil_pow_2));
            assert!(ring.is_lt(&ring.negate(ring.clone_el(a)), &ceil_pow_2));
            
            for i in 0..ring.abs_highest_set_bit(a).unwrap_or(0) {
                let mut pow_2 = ring.one();
                ring.mul_pow_2(&mut pow_2, i);
                let mut b = ring.clone_el(a);
                ring.mul_pow_2(&mut b, i);
                assert_el_eq!(ring, b, ring.mul(ring.clone_el(a), ring.clone_el(&pow_2)));
                ring.euclidean_div_pow_2(&mut b, i);
                assert_el_eq!(ring, b, a);
                ring.euclidean_div_pow_2(&mut b, i);
                assert_el_eq!(ring, b, ring.euclidean_div(ring.clone_el(a), &pow_2));
            }
        }

        // test euclidean div round to zero
        let d = ring.int_hom().map(8);
        for k in -10..=10 {
            let mut a = ring.int_hom().map(k);
            assert_el_eq!(ring, ring.int_hom().map(k / 8), ring.euclidean_div(ring.clone_el(&a), &d));
            ring.euclidean_div_pow_2(&mut a, 3);
            assert_el_eq!(ring, ring.int_hom().map(k / 8), a);
        }
        let d = ring.int_hom().map(-8);
        for k in -10..=10 {
            let a = ring.int_hom().map(k);
            assert_el_eq!(ring, ring.int_hom().map(k / -8), ring.euclidean_div(ring.clone_el(&a), &d));
        }

        // test rounded_div
        assert_el_eq!(ring, ring.int_hom().map(2), ring.rounded_div(ring.int_hom().map(7), &ring.int_hom().map(3)));
        assert_el_eq!(ring, ring.int_hom().map(-2), ring.rounded_div(ring.int_hom().map(-7), &ring.int_hom().map(3)));
        assert_el_eq!(ring, ring.int_hom().map(-2), ring.rounded_div(ring.int_hom().map(7), &ring.int_hom().map(-3)));
        assert_el_eq!(ring, ring.int_hom().map(2), ring.rounded_div(ring.int_hom().map(-7), &ring.int_hom().map(-3)));

        assert_el_eq!(ring, ring.int_hom().map(3), ring.rounded_div(ring.int_hom().map(8), &ring.int_hom().map(3)));
        assert_el_eq!(ring, ring.int_hom().map(-3), ring.rounded_div(ring.int_hom().map(-8), &ring.int_hom().map(3)));
        assert_el_eq!(ring, ring.int_hom().map(-3), ring.rounded_div(ring.int_hom().map(8), &ring.int_hom().map(-3)));
        assert_el_eq!(ring, ring.int_hom().map(3), ring.rounded_div(ring.int_hom().map(-8), &ring.int_hom().map(-3)));

        assert_el_eq!(ring, ring.int_hom().map(4), ring.rounded_div(ring.int_hom().map(7), &ring.int_hom().map(2)));
        assert_el_eq!(ring, ring.int_hom().map(-4), ring.rounded_div(ring.int_hom().map(-7), &ring.int_hom().map(2)));
        assert_el_eq!(ring, ring.int_hom().map(-4), ring.rounded_div(ring.int_hom().map(7), &ring.int_hom().map(-2)));
        assert_el_eq!(ring, ring.int_hom().map(4), ring.rounded_div(ring.int_hom().map(-7), &ring.int_hom().map(-2)));
    }
}

#[test]
fn test_int_div_assumption() {
    assert_eq!(-1, -10 / 8);
    assert_eq!(-1, 10 / -8);
    assert_eq!(1, 10 / 8);
    assert_eq!(1, -10 / -8);
}

#[test]
fn test_rounded_div() {
    let ZZ = StaticRing::<i32>::RING;
    assert_el_eq!(ZZ, 3, ZZ.rounded_div(20, &7));
    assert_el_eq!(ZZ, -3, ZZ.rounded_div(-20, &7));
    assert_el_eq!(ZZ, -3, ZZ.rounded_div(20, &-7));
    assert_el_eq!(ZZ, 3, ZZ.rounded_div(-20, &-7));
    assert_el_eq!(ZZ, 3, ZZ.rounded_div(22, &7));
    assert_el_eq!(ZZ, -3, ZZ.rounded_div(-22, &7));
    assert_el_eq!(ZZ, -3, ZZ.rounded_div(22, &-7));
    assert_el_eq!(ZZ, 3, ZZ.rounded_div(-22, &-7));
}

#[test]
fn test_binomial() {
    let ZZ = StaticRing::<i32>::RING;
    assert_eq!(0, binomial(-4, &-1, ZZ));
    assert_eq!(1, binomial(-4, &0, ZZ));
    assert_eq!(-4, binomial(-4, &1, ZZ));
    assert_eq!(10, binomial(-4, &2, ZZ));
    assert_eq!(-20, binomial(-4, &3, ZZ));
    assert_eq!(35, binomial(-4, &4, ZZ));
    assert_eq!(-56, binomial(-4, &5, ZZ));

    assert_eq!(0, binomial(3, &-1, ZZ));
    assert_eq!(1, binomial(3, &0, ZZ));
    assert_eq!(3, binomial(3, &1, ZZ));
    assert_eq!(3, binomial(3, &2, ZZ));
    assert_eq!(1, binomial(3, &3, ZZ));
    assert_eq!(0, binomial(3, &4, ZZ));
    
    assert_eq!(0, binomial(8, &-1, ZZ));
    assert_eq!(1, binomial(8, &0, ZZ));
    assert_eq!(8, binomial(8, &1, ZZ));
    assert_eq!(28, binomial(8, &2, ZZ));
    assert_eq!(56, binomial(8, &3, ZZ));
    assert_eq!(70, binomial(8, &4, ZZ));

    // a naive computation would overflow
    assert_eq!(145422675, binomial(30, &14, ZZ));
}

#[test]
fn test_factorial() {
    let ZZ = StaticRing::<i32>::RING;
    assert_eq!(1, factorial(&0, ZZ));
    assert_eq!(1, factorial(&1, ZZ));
    assert_eq!(2, factorial(&2, ZZ));
    assert_eq!(6, factorial(&3, ZZ));
    assert_eq!(24, factorial(&4, ZZ));
}

#[test]
fn test_ceil_floor_div() {
    let ZZ = StaticRing::<i32>::RING;
    for rhs in [-10, -3, -2, -1, 1, 2, 3, 10] {
        for lhs in [-10, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 10] {
            let result = ZZ.ceil_div(lhs, &rhs);
            assert_eq!(i32::div_ceil(lhs, rhs), result);
            assert_eq!((lhs as f64 / rhs as f64).ceil() as i32, result);

            let result = ZZ.floor_div(lhs, &rhs);
            assert_eq!(i32::div_floor(lhs, rhs), result);
            assert_eq!((lhs as f64 / rhs as f64).floor() as i32, result);
        }
    }
}

#[test]
fn test_parse() {
    let ZZbig = BigIntRing::RING;
    assert_el_eq!(&ZZbig, &ZZbig.int_hom().map(3), ZZbig.parse("3", 10).unwrap());
    assert_el_eq!(&ZZbig, &ZZbig.power_of_two(100), ZZbig.parse("1267650600228229401496703205376", 10).unwrap());
    assert_el_eq!(&ZZbig, &ZZbig.power_of_two(100), ZZbig.parse("+1267650600228229401496703205376", 10).unwrap());
    assert_el_eq!(&ZZbig, &ZZbig.negate(ZZbig.power_of_two(100)), ZZbig.parse("-1267650600228229401496703205376", 10).unwrap());
    assert_el_eq!(&ZZbig, &ZZbig.mul(ZZbig.pow(ZZbig.int_hom().map(11), 26), ZZbig.int_hom().map(10)), ZZbig.parse("a00000000000000000000000000", 11).unwrap());
    assert!(ZZbig.parse("238597a", 10).is_err());
}