use std::cmp::min;

use zn_64::Zn;

use crate::algorithms::int_factor::is_prime_power;
use crate::algorithms::miller_rabin::is_prime;
use crate::field::*;
use crate::homomorphism::CanHomFrom;
use crate::homomorphism::CanHomRef;
use crate::homomorphism::Homomorphism;
use crate::integer::*;
use crate::local::*;
use crate::divisibility::*;
use crate::pid::{EuclideanRing, PrincipalIdealRing};
use crate::ring::*;
use crate::rings::zn::*;
use crate::rings::local::AsLocalPIR;
use crate::primitive_int::StaticRing;

///
/// An approximate ring that computes with approximate p-adic numbers.
/// **This ring is only approximate but tries to implement equality, make sure you understand its behavior exactly before use!** 
/// 
/// Remember that every p-adic number can be written as a p-adic decomposition
/// ```text
/// a = sum_(i >= e) a_i p^i
/// ```
/// where `i` ranges from `e` (can be negative) to infinity.
/// Elements in this ring are represented by storing the first `n` nonzero
/// summands of this sum (the case of zero is an exception).
/// 
/// # Tracking of Precision
/// 
/// More concretely, for each nonzero number `a`, we store its *exponent* `e`
/// and its *precision*/*number of significant digits* `n` together with the value
/// ```text
/// sum_(e <= i < e + n) a_i p^i + O(p^(e + n))
/// ```
/// where the `O(p^k)` means for an unknown number divisible by `p^k` that represents
/// the difference to the ideal exact number.
/// For zero, we just store it as `O(p^n)`, i.e. only with a precision value `n`.
/// Arithmetic operations now follow the basic rules for computing with `O(.)`, e.g.
/// ```text
/// (p^e a + O(p^i)) + (p^f b + O(p^j)) = p^min(e, f) (p^(e - min(e, f)) a + p^(f - min(e, f)) b) + O(p^min(i, j))
/// (p^e a + O(p^i))(p^f b + O(p^j)) = p^(e + f) a b + O(p^min(e + j, f + i))
/// ```
/// This is directly reflected by the implementation:
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::padic::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::local::*;
/// # use crate::feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::primitive_int::StaticRing;
/// let QQ5 = PAdicNumbers::new(5, 10);
/// let a = QQ5.int_hom().map(5);
/// let b = QQ5.int_hom().map(2);
/// let sum = QQ5.add_ref(&a, &b);
/// assert_eq!("5^0 * 7 + O(5^10)", format!("{}", QQ5.format(&sum)));
/// // by subtracting (a + b) and b, we loose one digit of precision
/// let a_again = QQ5.sub(sum, b);
/// assert_eq!("5^1 * 1 + O(5^10)", format!("{}", QQ5.format(&a_again)));
/// assert_eq!("5^1 * 1 + O(5^11)", format!("{}", QQ5.format(&a)));
/// ```
/// Note that the output precision of operations may be lower than given by these
/// formula, since the operations themselves are only performed up to `O(p^max_precision)`.
/// 
/// ## Equality
/// 
/// Equality is clearly the big problem with any kind of approximate ring, since we
/// can be sure that two elements are actuall equal. To increase usability, we do provide
/// an implementation of [`RingBase::eq_el()`], as opposed to e.g. [`crate::rings::float_real::Real64`].
/// The situation is somewhat better than in the floating point case, since errors don't accumulate
/// when we use a non-archimedean valuation. Nevertheless, you should always keep in mind
/// that equality in [`PAdicNumbersBase`] is only an approximation.
/// 
/// Concretely, we allow comparison of two values `p^e a + O(p^i)` and `p^f b + O(p^j)` if
/// they there exist `min_significant_digits_for_compare` significant digits that are jointly nonzero
/// (i.e. digits of `p^k` with `e, f <= k < i, j`) or on `min_absolute_precise_digits_for_compare` jointly significant
/// digits with nonnegative valuation (i.e. digits of `p^k` with `0 <= k < i, j`).
/// This corresponds to the two ways of comparing floats, either absolutely (say `|a - b| < small constant`) or relatively
/// (say `|a - b| < small constant * (|a| + |b|)`).
/// If this is satisfied, we say the numbers are equal when they agree on all these digits.
/// 
/// The values `min_significant_digits_for_compare` and `min_absolute_precise_digits_for_compare` both default to 1 and
/// are currently not yet configurable.
/// 
/// # More examples
/// 
/// Note that the tracking of precision fails as soon as we introduce external logic, for example by
/// case distinctions whether elements are equal or zero. In the following example, we make the assumption 
/// that "equal" elements can be used interchangeably, which however circumvents the internal precision 
/// tracking, and leads to a result that claims to be more precise than it reasonably is.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::padic::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::local::*;
/// # use crate::feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::primitive_int::StaticRing;
/// # use feanor_math::field::*;
/// fn some_computation(QQ5: &PAdicNumbers, a: &El<PAdicNumbers>, b: &El<PAdicNumbers>) -> El<PAdicNumbers> {
///     QQ5.sub_ref(a, b)
/// }
/// let QQ5 = PAdicNumbers::new(5, 6);
/// let five_precise = QQ5.int_hom().map(5);
/// let five_less_precision = QQ5.sub(QQ5.int_hom().map(6), QQ5.one());
/// // we do some computation, and get a result with a certain precision
/// let actual_result = some_computation(&QQ5, &five_precise, &five_less_precision);
/// assert!(QQ5.is_zero(&actual_result));
/// assert_eq!(5, QQ5.highest_precise_digit(&actual_result));
/// 
/// let almost_five = QQ5.add_ref(&five_precise, &QQ5.int_hom().map(5 * 5 * 5 * 5 * 5 * 5));
/// // note that we have "equality" here
/// assert!(QQ5.eq_el(&five_less_precision, &almost_five));
/// // hence we should be able to use these two values interchangeably...
/// let equivalent_result = some_computation(&QQ5, &five_precise, &almost_five);
/// // indeed, equivalent_result and actual_result are equivalent up to 5 significant digits
/// assert!(QQ5.eq_el(&actual_result, &equivalent_result));
/// // However, equivalent_result has precision set to 6 significant digits now, which can be considered
/// // "incorrect" since the 6th digit has nothing to do with actual_result
/// assert_eq!(6, QQ5.highest_precise_digit(&equivalent_result));
/// assert_eq!(1, QQ5.significant_digits(&equivalent_result));
/// assert!(!QQ5.is_zero(&equivalent_result));
/// ```
/// It also shows that equality is not transitive anymore, which definitely should make you worry.
/// 
#[stability::unstable(feature = "enable")]
#[derive(Clone)]
pub struct PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    max_significant_digits: usize,
    min_significant_digits_for_compare: usize,
    min_absolute_precise_digits_for_compare: usize,
    ring: R
}

#[stability::unstable(feature = "enable")]
pub struct PAdicNumbersEl<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    // represents the element `sum_i a_i p^i` where `i` ranges from `exponent` (inclusive) to `exponent + precision` (exclusive);
    // the number is normalized if `a_exponent in Zp*`
    exponent: i64,
    // this can also be negative if there are no significant digits, but we still know that the value is in `O(p^(exponent + precision))`
    significant_digits: i64,
    el: El<R>
}

#[stability::unstable(feature = "enable")]
pub type PAdicNumbers<R = AsLocalPIR<Zn>> = RingValue<PAdicNumbersBase<R>>;

impl PAdicNumbers {
    
    #[stability::unstable(feature = "enable")]
    pub fn new(prime: i64, max_significant_digits: usize) -> Self {
        assert!(is_prime(&StaticRing::<i64>::RING, &prime, 10));
        Self::from(PAdicNumbersBase {
            min_significant_digits_for_compare: 1,
            min_absolute_precise_digits_for_compare: 1,
            max_significant_digits: max_significant_digits,
            ring: AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(prime, max_significant_digits) as u64)).unwrap()
        })
    }

}

impl<R> PAdicNumbers<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    #[stability::unstable(feature = "enable")]
    pub fn new_with(computations_in: R) -> Self {
        let (_p, e) = is_prime_power(computations_in.integer_ring(), computations_in.modulus()).unwrap();
        Self::from(PAdicNumbersBase {
            min_significant_digits_for_compare: 1,
            min_absolute_precise_digits_for_compare: 1,
            max_significant_digits: e,
            ring: computations_in
        })
    }

    ///
    /// See [`PAdicNumbersBase::significant_digits()`]
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn significant_digits(&self, el: &El<Self>) -> u64 {
        self.get_ring().significant_digits(el)
    }

    ///
    /// See [`PAdicNumbersBase::valuation()`]
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn valuation(&self, el: &El<Self>) -> Option<i64> {
        self.get_ring().valuation(el)
    }

    ///
    /// See [`PAdicNumbersBase::highest_precise_digit()`]
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn highest_precise_digit(&self, el: &El<Self>) -> i64 {
        self.get_ring().highest_precise_digit(el)
    }

    ///
    /// See [`PAdicNumbersBase::max_significant_digits()`]
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn max_significant_digits(&self) -> u64 {
        self.get_ring().max_significant_digits() as u64
    }
}

impl<R> Copy for PAdicNumbersBase<R>
    where R: RingStore + Copy,
        R::Type: ZnRing + PrincipalLocalRing,
        El<R>: Copy
{}

impl<R> PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    ///
    /// Returns the precision/number of significiant digits for the current element.
    /// 
    /// In other words, we expect that many leading p-adic coefficients of the element to 
    /// be equal to the ones the "true" result of the computation with p-adic numbers would 
    /// give. However, this is still only an estimate, and in certain cases, this 
    /// overestimates the precision. Note that tracking the actual precision is impossible
    /// in some situations. For more details and examples, see the struct-level doc
    /// [`PAdicNumbersBase`].
    ///  
    #[stability::unstable(feature = "enable")]
    pub fn significant_digits(&self, el: &<Self as RingBase>::Element) -> u64 {
        debug_assert!(self.ring.is_zero(&el.el) || self.is_normalized(el));
        if el.significant_digits < 0 { 0 } else { el.significant_digits as u64 }
    }

    ///
    /// Returns the p-adic valuation of `el`, or `None` if the value is considere to be zero up
    /// to the available precision.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn valuation(&self, el: &<Self as RingBase>::Element) -> Option<i64> {
        if self.is_zero(el) {
            None
        } else {
            debug_assert!(self.is_normalized(el));
            Some(el.exponent)
        }
    }

    ///
    /// Returns `i` such that we expect `c p^i` to be the last precise summand in the p-adic
    /// decomposition. For the problem of determining the actual precision, see 
    /// [`PAdicNumbersBase::precision_estimate()`] and the struct-level doc [`PAdicNumbersBase`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn highest_precise_digit(&self, el: &<Self as RingBase>::Element) -> i64 {
        debug_assert!(self.ring.is_zero(&el.el) || self.is_normalized(el));
        el.exponent + el.significant_digits - 1
    }

    #[stability::unstable(feature = "enable")]
    pub fn max_significant_digits(&self) -> u64 {
        self.max_significant_digits as u64
    }

    fn denormalize(&self, el: &mut PAdicNumbersEl<R>, target_exp: i64) -> i64 {
        assert!(el.exponent >= target_exp);
        self.ring.mul_assign(&mut el.el, self.ring.pow(self.ring.clone_el(self.ring.max_ideal_gen()), (el.exponent - target_exp) as usize));
        let new_precision = min(el.significant_digits + el.exponent - target_exp, self.max_significant_digits as i64);
        el.exponent = target_exp;
        let precision_change = new_precision - el.significant_digits;
        el.significant_digits = new_precision;
        return precision_change;
    }

    fn normalize(&self, el: &mut PAdicNumbersEl<R>) {
        if let Some(valuation) = self.ring.valuation(&el.el) {
            // here new_precision can become negative; works fine nevertheless
            el.significant_digits = el.significant_digits - valuation as i64;
            el.exponent += valuation as i64;
            el.el = self.ring.checked_div(&el.el, &self.ring.pow(self.ring.clone_el(&self.ring.max_ideal_gen()), valuation)).unwrap();
        } else {
            // cannot normalize zero
        }
    }

    fn is_normalized(&self, el: &PAdicNumbersEl<R>) -> bool {
        self.ring.is_unit(&el.el)
    }

    fn zero_with_prec(&self, prec: i64) -> PAdicNumbersEl<R> {
        PAdicNumbersEl {
            significant_digits: prec,
            exponent: i64::MAX - prec,
            el: self.ring.zero()
        }
    }

    fn map_in_from_int<I, H>(&self, mut value: I::Element, hom: H) -> <Self as RingBase>::Element
        where I: ?Sized + IntegerRing,
            H: Homomorphism<I, R::Type>
    {
        let from = hom.domain().get_ring();
        if from.is_zero(&value) {
            return self.zero_with_prec(self.max_significant_digits as i64);
        }
        let result = if from.representable_bits().is_none() || from.representable_bits().unwrap() > self.ring.integer_ring().abs_log2_ceil(self.ring.modulus()).unwrap() {
            let mut exponent = 0;
            let p_e = int_cast(self.ring.integer_ring().clone_el(self.ring.modulus()), hom.domain(), self.ring.integer_ring());
            let p = int_cast(self.ring.smallest_positive_lift(self.ring.clone_el(self.ring.max_ideal_gen())), hom.domain(), self.ring.integer_ring());
            while self.ring.is_zero(&hom.map_ref(&value)) {
                value = from.checked_left_div(&value, &p_e).unwrap();
                exponent += self.max_significant_digits;
            }
            let valuation = self.ring.valuation(&hom.map_ref(&value)).unwrap();
            value = from.checked_left_div(&value, &hom.domain().pow(p, valuation)).unwrap();
            exponent += valuation;
            PAdicNumbersEl {
                exponent: exponent as i64,
                significant_digits: self.max_significant_digits as i64,
                el: hom.map(value)
            }
        } else {
            let mut exponent = 0;
            let p = int_cast(self.ring.smallest_positive_lift(self.ring.clone_el(self.ring.max_ideal_gen())), hom.domain(), self.ring.integer_ring());
            while let Some(new_value) = from.checked_left_div(&value, &p) {
                exponent += 1;
                value = new_value;
            }
            PAdicNumbersEl {
                exponent: exponent as i64,
                significant_digits: self.max_significant_digits as i64,
                el: hom.map(value)
            }
        };
        debug_assert!(self.is_normalized(&result));
        return result;
    }
    
}

impl<R> PartialEq for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring()
    }
}

impl<R> RingBase for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    type Element = PAdicNumbersEl<R>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        PAdicNumbersEl {
            exponent: val.exponent,
            significant_digits: val.significant_digits.clone(),
            el: self.ring.clone_el(&val.el)
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, mut rhs: Self::Element) {
        if self.is_zero(&rhs) {
            lhs.significant_digits = min(lhs.significant_digits, (rhs.exponent + rhs.significant_digits).checked_sub(lhs.exponent).unwrap_or(i64::MAX));
            return;
        } else if self.is_zero(lhs) {
            rhs.significant_digits = min(rhs.significant_digits, (lhs.exponent + lhs.significant_digits).checked_sub(rhs.exponent).unwrap_or(i64::MAX));
            *lhs = rhs;
            return;
        }
        debug_assert!(self.is_normalized(lhs));
        debug_assert!(self.is_normalized(&rhs));
        if lhs.exponent > rhs.exponent {
            self.denormalize(lhs, rhs.exponent);
            self.ring.add_assign(&mut lhs.el, rhs.el);
            lhs.significant_digits = min(lhs.significant_digits, rhs.significant_digits);
        } else if rhs.exponent > lhs.exponent {
            self.denormalize(&mut rhs, lhs.exponent);
            self.ring.add_assign(&mut lhs.el, rhs.el);
            lhs.significant_digits = min(lhs.significant_digits, rhs.significant_digits);
        } else {
            self.ring.add_assign(&mut lhs.el, rhs.el);
            lhs.significant_digits = min(lhs.significant_digits, rhs.significant_digits);
            self.normalize(lhs);
        }
        debug_assert!(self.ring.is_zero(&lhs.el) || self.is_normalized(lhs));
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(self.ring.is_zero(&lhs.el) || self.is_normalized(lhs));
        debug_assert!(self.ring.is_zero(&rhs.el) || self.is_normalized(&rhs));
        lhs.significant_digits = min(lhs.significant_digits, rhs.significant_digits);
        lhs.exponent = min(i64::MAX.saturating_sub(lhs.significant_digits), lhs.exponent.checked_add(rhs.exponent).unwrap_or(i64::MAX));
        self.ring.mul_assign(&mut lhs.el, rhs.el);
        debug_assert!(self.ring.is_zero(&lhs.el) || self.is_normalized(lhs));
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.ring.negate_inplace(&mut lhs.el);
    }

    fn from_int(&self, value: i32) -> Self::Element {
        self.map_in_from_int(value, self.ring.int_hom())
    }

    fn zero(&self) -> Self::Element {
        return self.zero_with_prec(self.max_significant_digits as i64);
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        debug_assert!(self.ring.is_zero(&lhs.el) || self.is_normalized(lhs));
        debug_assert!(self.ring.is_zero(&rhs.el) || self.is_normalized(&rhs));
        if lhs.exponent >= rhs.exponent {
            // compare p-adic coefficients from `rhs.exponent` (inclusive) to `rhs.exponent + precision` (exclusive)
            let precision = min(rhs.significant_digits, (lhs.significant_digits + lhs.exponent).checked_sub(rhs.exponent).unwrap_or(i64::MAX));
            assert!(precision >= self.min_significant_digits_for_compare as i64 || precision + rhs.exponent >= self.min_absolute_precise_digits_for_compare as i64, "values {} and {} don't have enough precision for a comparison", RingRef::new(self).format(lhs), RingRef::new(self).format(rhs));
            debug_assert!(precision <= self.max_significant_digits as i64);
            let lhs_scale = min((self.max_significant_digits as i64 - precision).checked_add(lhs.exponent - rhs.exponent).unwrap_or(i64::MAX), self.max_significant_digits as i64);
            let lhs_denom = self.ring.mul_ref_fst(&lhs.el, self.ring.pow(self.ring.clone_el(self.ring.max_ideal_gen()), lhs_scale as usize));
            let rhs_denom = self.ring.mul_ref_fst(&rhs.el, self.ring.pow(self.ring.clone_el(self.ring.max_ideal_gen()), (self.max_significant_digits as i64 - precision) as usize));
            return self.ring.eq_el(&lhs_denom, &rhs_denom);
        } else {
            return self.eq_el(rhs, lhs);
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        debug_assert!(self.ring.is_zero(&value.el) || self.is_normalized(value));
        assert!(value.significant_digits >= self.min_significant_digits_for_compare as i64 || value.significant_digits + value.exponent >= self.min_absolute_precise_digits_for_compare as i64, "values {} doesn't have enough precision for a comparison with zero", RingRef::new(self).format(value));
        let power = self.max_significant_digits as i64 - value.significant_digits;
        self.ring.is_zero(&self.ring.mul_ref_fst(&value.el, self.ring.pow(self.ring.clone_el(self.ring.max_ideal_gen()), power as usize)))
    }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn is_approximate(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if self.is_zero(value) {
            return write!(out, "O({}^{})", self.ring.format(self.ring.max_ideal_gen()), value.significant_digits + value.exponent);
        }
        if env > EnvBindingStrength::Sum {
            write!(out, "(")?;
        }
        write!(out, "{}^{} * ", self.ring.format(self.ring.max_ideal_gen()), value.exponent)?;
        self.ring.get_ring().dbg_within(&value.el, out, EnvBindingStrength::Product)?;
        write!(out, " + O({}^{})", self.ring.format(self.ring.max_ideal_gen()), value.exponent + value.significant_digits)?;
        if env > EnvBindingStrength::Sum {
            write!(out, ")")?;
        }
        return Ok(());
    }

    fn characteristic<I: RingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        Some(ZZ.zero())
    }
}

impl<R> DivisibilityRing for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(rhs) && self.is_zero(lhs) {
            return Some(self.zero_with_prec(min(lhs.significant_digits, rhs.significant_digits)));
        } else if self.is_zero(rhs) {
            return None;
        }
        assert!(self.is_normalized(rhs));
        return Some(self.mul_ref_fst(lhs, PAdicNumbersEl {
            significant_digits: rhs.significant_digits.clone(),
            exponent: -rhs.exponent,
            el: self.ring.invert(&rhs.el).unwrap()
        }));
    }
}

impl<R> PrincipalIdealRing for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }
}

impl<R> EuclideanRing for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        if let Some(quo) = self.checked_left_div(&lhs, rhs) {
            return (quo, self.zero());
        } else {
            return (self.zero(), lhs);
        }
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }
}

impl<R> Domain for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{}

impl<R> Field for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing
{}

impl<I, R> CanHomFrom<I> for PAdicNumbersBase<R>
    where R: RingStore,
        R::Type: ZnRing + PrincipalLocalRing + CanHomFrom<I>,
        I: ?Sized + IntegerRing
{
    type Homomorphism = <R::Type as CanHomFrom<I>>::Homomorphism;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        self.ring.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &I, value: <I as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_from_int(value, CanHomRef::from_raw_parts(RingRef::new(from), &self.ring, hom))
    }
}

#[cfg(test)]
use crate::algorithms::eea::signed_gcd;

#[test]
fn test_precision() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let a = field.int_hom().map(3);
    let b = field.int_hom().map(9);
    let c = field.int_hom().map(12);
    assert_eq!(10, field.significant_digits(&a));
    assert_eq!(10, field.highest_precise_digit(&a));
    assert_eq!(10, field.significant_digits(&b));
    assert_eq!(11, field.highest_precise_digit(&b));
    assert_eq!(10, field.significant_digits(&c));
    assert_eq!(10, field.highest_precise_digit(&c));
    
    let e = field.sub_ref(&a, &a);
    assert_eq!(10, field.significant_digits(&e));
    assert_eq!(10, field.highest_precise_digit(&e));
    let z = field.zero();
    assert_eq!(10, field.significant_digits(&z));
    assert!(field.eq_el(&z, &e));
    assert_eq!(10, field.significant_digits(&e));
    assert_eq!(10, field.highest_precise_digit(&e));
    assert_eq!(10, field.significant_digits(&z));

    assert_eq!(10, field.significant_digits(&field.sub_ref(&b, &a)));
    assert_eq!(10, field.highest_precise_digit(&field.sub_ref(&b, &a)));

    let d = field.sub_ref(&c, &a);
    assert_eq!(9, field.significant_digits(&d));
    assert_eq!(10, field.highest_precise_digit(&d));

    assert!(field.eq_el(&b, &d));
    assert_eq!(11, field.highest_precise_digit(&b));
    assert_eq!(10, field.significant_digits(&b));

    assert!(!field.eq_el(&a, &b));
    assert_eq!(10, field.significant_digits(&a));
    assert_eq!(10, field.highest_precise_digit(&a));
}

#[test]
fn test_approx_eq() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let nine_precise = field.int_hom().map(9);
    let nine_imprecise = field.sub(field.int_hom().map(10), field.one());
    let high_valuation = field.int_hom().map(StaticRing::<i32>::RING.pow(3, 10));

    assert_eq!(8, field.significant_digits(&nine_imprecise));
    assert!(!field.is_zero(&high_valuation));

    let zero_imprecise = field.sub_ref(&nine_imprecise, &nine_precise);
    let almost_zero_imprecise = field.add_ref(&zero_imprecise, &high_valuation);
    assert!(field.is_zero(&almost_zero_imprecise));

    let almost_three = field.int_hom().map(3 + StaticRing::<i32>::RING.pow(3, 10));
    let imprecise_three = field.sub(field.int_hom().map(4), field.one());
    let should_be_zero = field.sub(almost_three, imprecise_three);
    assert!(field.is_zero(&should_be_zero));
    assert_eq!(0, field.significant_digits(&should_be_zero));

    let almost_nine = field.int_hom().map(9 + StaticRing::<i32>::RING.pow(3, 11));
    let should_be_zero = field.sub(almost_nine, nine_imprecise);
    assert!(field.is_zero(&should_be_zero));
    assert_eq!(0, field.significant_digits(&should_be_zero));
}

#[test]
fn test_add_zero() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let zero_high_exp = field.zero();
    let zero_low_exp = field.sub(field.one(), field.one());
    let high_valuation = field.int_hom().map(StaticRing::<i32>::RING.pow(3, 10));

    assert!(field.eq_el(&zero_low_exp, &high_valuation));
    assert!(!field.eq_el(&zero_high_exp, &high_valuation));
    assert!(field.eq_el(&field.add_ref(&zero_low_exp, &zero_high_exp), &high_valuation));
    assert!(field.eq_el(&field.add_ref(&zero_high_exp, &zero_low_exp), &high_valuation));
}

#[test]
fn test_ring_axioms() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let elements = (-3..10).flat_map(|num| (1..4).map(move |den| field.div(&field.int_hom().map(num), &field.int_hom().map(den))));
    crate::ring::generic_tests::test_ring_axioms(&field, elements);
}

#[test]
fn test_divisibility_axioms() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let elements = (-6..10).flat_map(|num| (1..4).map(move |den| field.div(&field.int_hom().map(num), &field.int_hom().map(den))));
    crate::divisibility::generic_tests::test_divisibility_axioms(&field, elements);
}

#[test]
fn test_field_axioms() {
    let field = PAdicNumbers::new_with(AsLocalPIR::from_zn(Zn::new(StaticRing::<i64>::RING.pow(3, 10) as u64)).unwrap());
    let elements = (-6..10).flat_map(|num| (1..4)
        .filter(move |den| signed_gcd(*den, num, &StaticRing::<i32>::RING) == 1)
        .map(move |den| field.div(&field.int_hom().map(num), &field.int_hom().map(den)))
    );
    crate::field::generic_tests::test_field_axioms(&field, elements);
}