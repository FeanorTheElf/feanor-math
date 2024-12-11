
use std::fmt::Debug;

use crate::ring::*;

///
/// Trait for rings that support checking divisibility, i.e.
/// whether for `x, y` there is `k` such that `x = ky`.
/// 
pub trait DivisibilityRing: RingBase {

    ///
    /// Additional data associated to a fixed ring element that can be used
    /// to speed up division by this ring element. 
    /// 
    /// See also [`DivisibilityRing::prepare_divisor()`].
    /// 
    #[stability::unstable(feature = "enable")]
    type PreparedDivisorData = ();

    ///
    /// Checks whether there is an element `x` such that `rhs * x = lhs`, and
    /// returns it if it exists. 
    /// 
    /// Note that this does not have to be unique, if rhs is a left zero-divisor. 
    /// In particular, this function will return any element in the ring if `lhs = rhs = 0`.
    /// In rings with many zero-divisors, this can sometimes lead to unintuitive behavior.
    /// See also [`crate::pid::PrincipalIdealRing::checked_div_min()`] for a function that,
    /// if available, might sometimes behave more intuitively.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::divisibility::*;
    /// let ZZ = StaticRing::<i64>::RING;
    /// assert_eq!(Some(3), ZZ.checked_left_div(&6, &2));
    /// assert_eq!(None, ZZ.checked_left_div(&6, &4));
    /// ```
    /// In rings that have zero-divisors, there are usually multiple possible results.
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::divisibility::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// let ring = Zn::new(6);
    /// let four_over_four = ring.checked_left_div(&ring.int_hom().map(4), &ring.int_hom().map(4)).unwrap();
    /// assert!(ring.eq_el(&four_over_four, &ring.int_hom().map(1)) || ring.eq_el(&four_over_four, &ring.int_hom().map(4)));
    /// // note that the output 4 might be unexpected, since it is a zero-divisor itself!
    /// ```
    /// 
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element>;

    ///
    /// Returns whether there is an element `x` such that `rhs * x = lhs`.
    /// If you need such an element, consider using [`DivisibilityRing::checked_left_div()`].
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::divisibility::*;
    /// let ZZ = StaticRing::<i64>::RING;
    /// assert_eq!(true, ZZ.divides_left(&6, &2));
    /// assert_eq!(false, ZZ.divides_left(&6, &4));
    /// ```
    /// 
    fn divides_left(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        self.checked_left_div(lhs, rhs).is_some()
    }

    ///
    /// Same as [`DivisibilityRing::divides_left()`], but requires a commutative ring.
    /// 
    fn divides(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert!(self.is_commutative());
        self.divides_left(lhs, rhs)
    }

    ///
    /// Same as [`DivisibilityRing::checked_left_div()`], but requires a commutative ring.
    /// 
    fn checked_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        assert!(self.is_commutative());
        self.checked_left_div(lhs, rhs)
    }

    ///
    /// Returns whether the given element is a unit, i.e. has an inverse.
    /// 
    fn is_unit(&self, x: &Self::Element) -> bool {
        self.checked_left_div(&self.one(), x).is_some()
    }

    ///
    /// Function that computes a "balancing" factor of a sequence of ring elements.
    /// The only use of the balancing factor is to increase performance, in particular,
    /// dividing all elements in the sequence by this factor should make them 
    /// "smaller" resp. cheaper to process.
    /// 
    /// Note that the balancing factor must always be a non-zero divisor.
    /// 
    /// Standard cases are reducing fractions (where the sequence would be exactly two
    /// elements), or polynomials over fields (where we often want to scale the polynomial
    /// to make all denominators 1).
    /// 
    /// If balancing does not make sense (as in the case of finite fields) or is not
    /// supported by the implementation, it is valid to return `None`.
    /// 
    fn balance_factor<'a, I>(&self, _elements: I) -> Option<Self::Element>
        where I: Iterator<Item = &'a Self::Element>,
            Self: 'a
    {
        None
    }

    ///
    /// "Prepares" an element of this ring for division.
    /// 
    /// The returned [`DivisibilityRing::PreparedDivisor`] can then be used in calls
    /// to [`DivisibilityRing::checked_left_div_prepared()`] and other "prepared" division
    /// functions, which can be faster than for an "unprepared" element.
    /// 
    /// See also [`DivisibilityRing::prepare_divisor()`].
    /// 
    /// # Caveat
    /// 
    /// Previously, this was its own trait, but that caused problems, since using this properly 
    /// would require fully-fledged specialization. Hence, we now inlude it in [`DivisibilityRing`]
    /// but provide defaults for all `*_prepared()` functions. 
    /// 
    /// This is not perfect, and in particular, if you specialize [`DivisibilityRing::PreparedDivisorData`]
    /// and not [`DivisibilityRing::prepare_divisor()`], this will currently not cause a compile error, but 
    /// panic at runtime when calling [`DivisibilityRing::prepare_divisor()`] (unfortunately). However,
    /// it seems like the most usable solution, and does not require unsafe code.
    /// 
    /// TODO: at the next breaking release, remove default implementation of `prepare_divisor()`.
    /// 
    /// # Example
    /// 
    /// Assume we want to go through all positive integers `<= 1000` that are divisible by `257`. The naive 
    /// way would be the following
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::divisibility::*;
    /// # use feanor_math::primitive_int::*;
    /// let ring = StaticRing::<i128>::RING;
    /// for integer in 0..1000 {
    ///     if ring.divides(&integer, &257) {
    ///         assert!(integer == 0 || integer == 257 || integer == 514 || integer == 771);
    ///     }
    /// }
    /// ```
    /// It can be faster to instead prepare the divisor `257` once and use this "prepared" divisor for
    /// all checks (of course, it will be much faster to iterate over `(0..10000).step_by(257)`, but
    /// for the sake of this example, let's use individual divisibility checks).
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::divisibility::*;
    /// # use feanor_math::primitive_int::*;
    /// # let ring = StaticRing::<i128>::RING;
    /// let prepared_257 = ring.get_ring().prepare_divisor(257);
    /// for integer in 0..1000 {
    ///     if ring.get_ring().divides_left_prepared(&integer, &prepared_257) {
    ///         assert!(integer == 0 || integer == 257 || integer == 514 || integer == 771);
    ///     }
    /// }
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    fn prepare_divisor(&self, x: Self::Element) -> PreparedDivisor<Self> {
        struct ProduceUnitType;
        trait ProduceValue<T> {
            fn produce() -> T;
        }
        impl<T> ProduceValue<T> for ProduceUnitType {
            default fn produce() -> T {
                panic!("if you specialize DivisibilityRing::PreparedDivisorData, you must also specialize DivisibilityRing::prepare_divisor()")
            }
        }
        impl ProduceValue<()> for ProduceUnitType {
            fn produce() -> () {}
        }
        PreparedDivisor {
            element: x,
            data: <ProduceUnitType as ProduceValue<Self::PreparedDivisorData>>::produce()
        }
    }

    ///
    /// Same as [`DivisibilityRing::checked_left_div()`] but for a prepared divisor.
    /// 
    /// See also [`DivisibilityRing::prepare_divisor()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn checked_left_div_prepared(&self, lhs: &Self::Element, rhs: &PreparedDivisor<Self>) -> Option<Self::Element> {
        self.checked_left_div(lhs, &rhs.element)
    }

    ///
    /// Same as [`DivisibilityRing::divides_left()`] but for a prepared divisor.
    /// 
    /// See also [`DivisibilityRing::prepare_divisor()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn divides_left_prepared(&self, lhs: &Self::Element, rhs: &PreparedDivisor<Self>) -> bool {
        self.divides_left(lhs, &rhs.element)
    }

    ///
    /// Same as [`DivisibilityRing::is_unit()`] but for a prepared divisor.
    /// 
    /// See also [`DivisibilityRing::prepare_divisor()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn is_unit_prepared(&self, x: &PreparedDivisor<Self>) -> bool {
        self.is_unit(&x.element)
    }
}

///
/// Struct for ring elements that are stored with associated data to
/// enable faster divisions.
/// 
/// For details, see [`DivisibilityRing::prepare_divisor()`].
/// 
pub struct PreparedDivisor<R>
    where R: ?Sized + RingBase + DivisibilityRing
{
    pub element: R::Element,
    pub data: R::PreparedDivisorData
}

impl<R> Debug for PreparedDivisor<R>
    where R: ?Sized + RingBase + DivisibilityRing,
        R::Element: Debug,
        R::PreparedDivisorData: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PreparedDivisor {{ element: {:?}, data: {:?} }}", &self.element, &self.data)
    }
}

impl<R> Clone for PreparedDivisor<R>
    where R: ?Sized + RingBase + DivisibilityRing,
        R::Element: Clone,
        R::PreparedDivisorData: Clone
{
    fn clone(&self) -> Self {
        Self {
            element: self.element.clone(),
            data: self.data.clone()
        }
    }
}

impl<R> Copy for PreparedDivisor<R>
    where R: ?Sized + RingBase + DivisibilityRing,
        R::Element: Copy,
        R::PreparedDivisorData: Copy
{}

///
/// Trait for rings that are integral, i.e. have no zero divisors.
/// 
/// A zero divisor is a nonzero element `a` such that there is a nonzero
/// element `b` with `ab = 0`.
/// 
pub trait Domain: DivisibilityRing {}

///
/// Trait for [`RingStore`]s that store [`DivisibilityRing`]s. Mainly used
/// to provide a convenient interface to the `DivisibilityRing`-functions.
/// 
pub trait DivisibilityRingStore: RingStore
    where Self::Type: DivisibilityRing
{
    delegate!{ DivisibilityRing, fn checked_left_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
    delegate!{ DivisibilityRing, fn divides_left(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ DivisibilityRing, fn is_unit(&self, x: &El<Self>) -> bool }
    delegate!{ DivisibilityRing, fn checked_div(&self, lhs: &El<Self>, rhs: &El<Self>) -> Option<El<Self>> }
    delegate!{ DivisibilityRing, fn divides(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }

    fn invert(&self, el: &El<Self>) -> Option<El<Self>> {
        self.checked_div(&self.one(), el)
    }
}

impl<R> DivisibilityRingStore for R
    where R: RingStore, R::Type: DivisibilityRing
{}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use crate::ring::El;
    use super::*;

    pub fn test_divisibility_axioms<R: DivisibilityRingStore, I: Iterator<Item = El<R>>>(ring: R, edge_case_elements: I)
        where R::Type: DivisibilityRing
    {
        let elements = edge_case_elements.collect::<Vec<_>>();

        for a in &elements {
            for b in &elements {
                let ab = ring.mul(ring.clone_el(a), ring.clone_el(b));
                let c = ring.checked_left_div(&ab, &a);
                assert!(c.is_some(), "Divisibility existence failed: there should exist b = {} such that {} = b * {}, but none was found", ring.format(b), ring.format(&ab), ring.format(&a));
                assert!(ring.eq_el(&ab, &ring.mul_ref_snd(ring.clone_el(a), c.as_ref().unwrap())), "Division failed: {} * {} != {} but {} = checked_div({}, {})", ring.format(a), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(&a));

                if !ring.is_unit(a) {
                    let ab_plus_one = ring.add(ring.clone_el(&ab), ring.one());
                    let c = ring.checked_left_div(&ab_plus_one, &a);
                    assert!(c.is_none(), "Unit check failed: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_plus_one), ring.format(a), ring.format(c.as_ref().unwrap()));

                    let ab_minus_one = ring.sub(ring.clone_el(&ab), ring.one());
                    let c = ring.checked_left_div(&ab_minus_one, &a);
                    assert!(c.is_none(), "Unit check failed: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_minus_one), ring.format(a), ring.format(c.as_ref().unwrap()));
                } else {
                    let inv = ring.checked_left_div(&ring.one(), a);
                    assert!(inv.is_some(), "Unit check failed: is_unit({}) is true but checked_div({}, {}) is None", ring.format(a), ring.format(&ring.one()), ring.format(&a));
                    let prod = ring.mul_ref(a, inv.as_ref().unwrap());
                    assert!(ring.eq_el(&ring.one(), &prod), "Division failed: {} != {} * {} but checked_div({}, {}) = {}", ring.format(&ring.one()), ring.format(a), ring.format(inv.as_ref().unwrap()), ring.format(&ring.one()), ring.format(a), ring.format(c.as_ref().unwrap()));
                }
            }
        }

        for a in &elements {
            let a_prepared_divisor = ring.get_ring().prepare_divisor(ring.clone_el(a));
            for b in &elements {
                let ab = ring.mul(ring.clone_el(a), ring.clone_el(b));
                let c = ring.get_ring().checked_left_div_prepared(&ab, &a_prepared_divisor);
                assert!(c.is_some(), "Divisibility existence failed for prepared divisor: there should exist b = {} such that {} = b * {}, but none was found", ring.format(b), ring.format(&ab), ring.format(&a));
                assert!(ring.eq_el(&ab, &ring.mul_ref_snd(ring.clone_el(a), c.as_ref().unwrap())), "Division failed: {} * {} != {} but {} = checked_div({}, {})", ring.format(a), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(c.as_ref().unwrap()), ring.format(&ab), ring.format(&a));

                if !ring.get_ring().is_unit_prepared(&a_prepared_divisor) {
                    let ab_plus_one = ring.add(ring.clone_el(&ab), ring.one());
                    let c = ring.get_ring().checked_left_div_prepared(&ab_plus_one, &a_prepared_divisor);
                    assert!(c.is_none(), "Unit check failed for prepared divisor: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_plus_one), ring.format(a), ring.format(c.as_ref().unwrap()));

                    let ab_minus_one = ring.sub(ring.clone_el(&ab), ring.one());
                    let c = ring.get_ring().checked_left_div_prepared(&ab_minus_one, &a_prepared_divisor);
                    assert!(c.is_none(), "Unit check failed for prepared divisor: is_unit({}) is false but checked_div({}, {}) = {}", ring.format(a), ring.format(&ab_minus_one), ring.format(a), ring.format(c.as_ref().unwrap()));
                } else {
                    let inv = ring.get_ring().checked_left_div_prepared(&ring.one(), &a_prepared_divisor);
                    assert!(inv.is_some(), "Unit check failed for prepared divisor: is_unit({}) is true but checked_div({}, {}) is None", ring.format(a), ring.format(&ring.one()), ring.format(&a));
                    let prod = ring.mul_ref(a, inv.as_ref().unwrap());
                    assert!(ring.eq_el(&ring.one(), &prod), "Division failed for prepared divisor: {} != {} * {} but checked_div({}, {}) = {}", ring.format(&ring.one()), ring.format(a), ring.format(inv.as_ref().unwrap()), ring.format(&ring.one()), ring.format(a), ring.format(c.as_ref().unwrap()));
                }
            }
        }
    }
}