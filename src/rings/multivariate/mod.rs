use crate::homomorphism::Homomorphism;
use crate::ring::*;
use crate::seq::VectorFn;
use crate::wrapper::RingElementWrapper;

///
/// Contains an implementation [`multivariate_impl::MultivariatePolyRingImpl`] of a multivariate polynomial
/// ring.
/// 
pub mod multivariate_impl;

use std::any::Any;
use std::cmp::{max, Ordering};

///
/// Type of coefficients of multivariate polynomials of the given ring.
/// 
pub type PolyCoeff<P> = El<<<P as RingStore>::Type as RingExtension>::BaseRing>;

///
/// Type of monomials of multivariate polynomials of the given ring.
/// 
pub type PolyMonomial<P> = <<P as RingStore>::Type as MultivariatePolyRing>::Monomial;

///
/// Trait for multivariate polynomial rings.
/// 
pub trait MultivariatePolyRing: RingExtension {

    type Monomial;
    type TermIter<'a>: Iterator<Item = (&'a El<Self::BaseRing>, &'a Self::Monomial)>
        where Self: 'a;

    ///
    /// Returns the number of variables of this polynomial ring, i.e. the transcendence degree
    /// of the base ring.
    /// 
    fn indeterminate_count(&self) -> usize;

    ///
    /// Creates a monomial with the given exponents.
    /// 
    /// Note that when building a polynomial, the most convenient method is usually
    /// to use [`MultivariatePolyRingStore::with_wrapped_indeterminates()`].
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::rings::multivariate::*;
    /// # use feanor_math::rings::multivariate::multivariate_impl::*;
    /// let poly_ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 3);
    /// let x_as_monomial = poly_ring.create_monomial([1, 0, 0]);
    /// let x_as_poly = poly_ring.create_term(1, x_as_monomial);
    /// assert_eq!("X0", format!("{}", poly_ring.format(&x_as_poly)));
    /// ```
    /// 
    fn create_monomial<I>(&self, exponents: I) -> Self::Monomial
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator;

    ///
    /// Multiplies the given polynomial with the given monomial.
    /// 
    fn mul_assign_monomial(&self, f: &mut Self::Element, monomial: Self::Monomial);

    ///
    /// Returns the coefficient corresponding to the given monomial in the given polynomial.
    /// If the polynomial does not contain a term with that monomial, zero is returned.
    /// 
    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, m: &Self::Monomial) -> &'a El<Self::BaseRing>;

    ///
    /// Returns the power of the `var_index`-th variable in the given monomial.
    /// In other words, this maps `X1^i1 ... Xm^im` to `i(var_index)`.
    /// 
    fn exponent_at(&self, m: &Self::Monomial, var_index: usize) -> usize;

    ///
    /// Writes the powers of each variable in the given monomial into the given 
    /// output slice. 
    /// 
    /// This is equivalent to performing `out[i] = self.exponent_at(m, i)` for
    /// every `i` in `0..self.indeterminate_count()`.
    /// 
    fn expand_monomial_to(&self, m: &Self::Monomial, out: &mut [usize]) {
        assert_eq!(out.len(), self.indeterminate_count());
        for i in 0..self.indeterminate_count() {
            out[i] = self.exponent_at(m, i)
        }
    }

    ///
    /// Returns an iterator over all nonzero terms of the given polynomial.
    /// 
    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermIter<'a>;

    ///
    /// Creates a new single-term polynomial.
    /// 
    fn create_term(&self, coeff: El<Self::BaseRing>, monomial: Self::Monomial) -> Self::Element {
        let mut result = self.from(coeff);
        self.mul_assign_monomial(&mut result, monomial);
        return result;
    }

    ///
    /// Returns the **L**eading **T**erm of `f`, i.e. the term whose monomial is largest w.r.t. the given order.
    /// 
    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        self.terms(f).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
    }

    ///
    /// Returns the term of `f` whose monomial is largest (w.r.t. the given order) among all monomials smaller than `lt_than`.
    /// 
    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a Self::Element, order: O, lt_than: &Self::Monomial) -> Option<(&'a El<Self::BaseRing>, &'a Self::Monomial)> {
        self.terms(f).filter(|(_, m)| order.compare(RingRef::new(self), m, lt_than) == Ordering::Less).max_by(|l, r| order.compare(RingRef::new(self), &l.1, &r.1))
    }

    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, Self::Monomial)>
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.into_iter().map(|(c, m)| self.create_term(c, m))
        ));
    }

    ///
    /// Applies the given homomorphism `R -> S` to each coefficient of the given polynomial
    /// in `R[X1, ..., Xm]` to produce a monomial in `S[X1, ..., Xm]`.
    /// 
    fn map_terms<P, H>(&self, from: &P, el: &P::Element, hom: H) -> Self::Element
        where P: ?Sized + MultivariatePolyRing,
            H: Homomorphism<<P::BaseRing as RingStore>::Type, <Self::BaseRing as RingStore>::Type>
    {
        assert!(self.base_ring().get_ring() == hom.codomain().get_ring());
        assert!(from.base_ring().get_ring() == hom.domain().get_ring());
        assert_eq!(self.indeterminate_count(), from.indeterminate_count());
        let mut exponents_storage = (0..self.indeterminate_count()).map(|_| 0).collect::<Vec<_>>();
        return RingRef::new(self).from_terms(from.terms(el).map(|(c, m)| {
            from.expand_monomial_to(m, &mut exponents_storage);
            (hom.map_ref(c), self.create_monomial(exponents_storage.iter().copied()))
        }));
    }

    fn clone_monomial(&self, mon: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.indeterminate_count()).map(|i| self.exponent_at(mon, i)))
    }

    ///
    /// Returns a list of all variables appearing in the given polynomial.
    /// Associated with each variable is the highest degree in which it
    /// appears in some term.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::primitive_int::*;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::multivariate::*;
    /// # use feanor_math::rings::multivariate::multivariate_impl::*;
    /// let poly_ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 2);
    /// let [f, g] = poly_ring.with_wrapped_indeterminates(|[X, Y]| [1 + X + X.pow_ref(2) * Y, X.pow_ref(3)]);
    /// assert_eq!(vec![(0, 2), (1, 1)], poly_ring.appearing_indeterminates(&f));
    /// assert_eq!(vec![(0, 3)], poly_ring.appearing_indeterminates(&g));
    /// ```
    /// 
    fn appearing_indeterminates(&self, f: &Self::Element) -> Vec<(usize, usize)> {
        let mut result = (0..self.indeterminate_count()).map(|_| 0).collect::<Vec<_>>();
        for (_, m) in self.terms(f) {
            for i in 0..self.indeterminate_count() {
                result[i] = max(result[i], self.exponent_at(m, i));
            }
        }
        return result.into_iter().enumerate().filter(|(_, e)| *e > 0).collect();
    }

    ///
    /// Multiplies two monomials.
    /// 
    fn monomial_mul(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.indeterminate_count()).map(|i| self.exponent_at(&lhs, i) + self.exponent_at(rhs, i)))
    }

    ///
    /// Returns the degree of a monomial, i.e. the sum of the exponents of all variables.
    /// 
    fn monomial_deg(&self, mon: &Self::Monomial) -> usize {
        (0..self.indeterminate_count()).map(|i| self.exponent_at(mon, i)).sum()
    }

    ///
    /// Returns the least common multiple of two monomials.
    /// 
    fn monomial_lcm(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Self::Monomial {
        self.create_monomial((0..self.indeterminate_count()).map(|i| max(self.exponent_at(&lhs, i), self.exponent_at(rhs, i))))
    }

    ///
    /// Computes the quotient of two monomials.
    /// 
    /// If `lhs` does not divide `rhs`, this returns `Result::Err` with the monomial
    /// `lhs / gcd(rhs, lhs)`.
    /// 
    fn monomial_div(&self, lhs: Self::Monomial, rhs: &Self::Monomial) -> Result<Self::Monomial, Self::Monomial> {
        let mut failed = false;
        let result = self.create_monomial((0..self.indeterminate_count()).map(|i| {
            if let Some(res) = self.exponent_at(&lhs, i).checked_sub(self.exponent_at(rhs, i)) {
                res
            } else {
                failed = true;
                0
            }
        }));
        if failed {
            Err(result)
        } else {
            Ok(result)
        }
    }

    ///
    /// Evaluates the given polynomial at the given values.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::multivariate::*;
    /// # use feanor_math::rings::multivariate::multivariate_impl::*;
    /// # use feanor_math::primitive_int::*; 
    /// # use feanor_math::seq::*; 
    /// let poly_ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 2);
    /// let [f] = poly_ring.with_wrapped_indeterminates(|[X, Y]| [1 + X + X.pow_ref(2) * Y]);
    /// assert_eq!(1 + 5 + 5 * 5 * 8, poly_ring.evaluate(&f, [5, 8].into_ring_el_fn(StaticRing::<i64>::RING), &poly_ring.base_ring().identity()));
    /// ```
    /// 
    fn evaluate<R, V, H>(&self, f: &Self::Element, value: V, hom: H) -> R::Element
        where R: ?Sized + RingBase,
            H: Homomorphism<<Self::BaseRing as RingStore>::Type, R>,
            V: VectorFn<R::Element>
    {
        assert_eq!(self.indeterminate_count(), value.len());
        assert!(hom.domain().get_ring() == self.base_ring().get_ring());
        hom.codomain().sum(self.terms(f).map(|(c, m)| hom.mul_map(
            hom.codomain().prod((0..self.indeterminate_count()).map(|i| hom.codomain().pow(value.at(i), self.exponent_at(m, i)))),
            hom.domain().clone_el(c)
        )))
    }

    ///
    /// Replaces the given indeterminate in the given polynomial by the value `val`.
    /// 
    /// Conceptually, this is similar to [`MultivariatePolyRing::evaluate()`], but less general,
    /// which can allow a faster implementation sometimes. In particular, this only replaces a single
    /// indeterminate, and does not change the ring.
    /// 
    fn specialize(&self, f: &Self::Element, var: usize, val: &Self::Element) -> Self::Element {
        assert!(var < self.indeterminate_count());
        let mut parts = Vec::new();
        for (c, m) in self.terms(f) {
            while self.exponent_at(m, var) as usize >= parts.len() {
                parts.push(Vec::new());
            }
            let new_m = self.create_monomial((0..self.indeterminate_count()).map(|i| if i == var { 0 } else { self.exponent_at(m, i) }));
            parts[self.exponent_at(m, var)].push((self.base_ring().clone_el(c), new_m));
        }
        if let Some(first) = parts.pop() {
            let mut current = self.zero();
            self.add_assign_from_terms(&mut current, first);
            while let Some(new) = parts.pop() {
                let mut next = self.zero();
                self.add_assign_from_terms(&mut next, new);
                self.mul_assign_ref(&mut current, val);
                self.add_assign(&mut current, next);
            }
            return current;
        } else {
            return self.zero();
        }
    }
}

///
/// [`RingStore`] for [`MultivariatePolyRing`]
/// 
pub trait MultivariatePolyRingStore: RingStore
    where Self::Type: MultivariatePolyRing
{
    delegate!{ MultivariatePolyRing, fn indeterminate_count(&self) -> usize }
    delegate!{ MultivariatePolyRing, fn create_term(&self, coeff: PolyCoeff<Self>, monomial: PolyMonomial<Self>) -> El<Self> }
    delegate!{ MultivariatePolyRing, fn exponent_at(&self, m: &PolyMonomial<Self>, var_index: usize) -> usize }
    delegate!{ MultivariatePolyRing, fn expand_monomial_to(&self, m: &PolyMonomial<Self>, out: &mut [usize]) -> () }
    delegate!{ MultivariatePolyRing, fn monomial_mul(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> PolyMonomial<Self> }
    delegate!{ MultivariatePolyRing, fn monomial_lcm(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> PolyMonomial<Self> }
    delegate!{ MultivariatePolyRing, fn monomial_div(&self, lhs: PolyMonomial<Self>, rhs: &PolyMonomial<Self>) -> Result<PolyMonomial<Self>, PolyMonomial<Self>> }
    delegate!{ MultivariatePolyRing, fn monomial_deg(&self, val: &PolyMonomial<Self>) -> usize }
    delegate!{ MultivariatePolyRing, fn mul_assign_monomial(&self, f: &mut El<Self>, monomial: PolyMonomial<Self>) -> () }
    delegate!{ MultivariatePolyRing, fn appearing_indeterminates(&self, f: &El<Self>) -> Vec<(usize, usize)> }
    delegate!{ MultivariatePolyRing, fn specialize(&self, f: &El<Self>, var: usize, val: &El<Self>) -> El<Self> }

    fn expand_monomial(&self, m: &PolyMonomial<Self>) -> Vec<usize> {
        let mut result = (0..self.indeterminate_count()).map(|_| 0).collect::<Vec<_>>();
        self.expand_monomial_to(m, &mut result);
        return result;
    }

    ///
    /// Returns the term of `f` whose monomial is largest (w.r.t. the given order) among all monomials smaller than `lt_than`.
    /// 
    fn largest_term_lt<'a, O: MonomialOrder>(&'a self, f: &'a El<Self>, order: O, lt_than: &PolyMonomial<Self>) -> Option<(&'a PolyCoeff<Self>, &'a PolyMonomial<Self>)> {
        self.get_ring().largest_term_lt(f, order, lt_than)
    }
    
    ///
    /// Returns the **L**eading **T**erm of `f`, i.e. the term whose monomial is largest w.r.t. the given order.
    /// 
    fn LT<'a, O: MonomialOrder>(&'a self, f: &'a El<Self>, order: O) -> Option<(&'a PolyCoeff<Self>, &'a PolyMonomial<Self>)> {
        self.get_ring().LT(f, order)
    }

    ///
    /// Creates a new monomial with given exponents.
    /// 
    /// For details, see [`MultivariatePolyRing::create_monomial()`].
    /// 
    fn create_monomial<I>(&self, exponents: I) -> PolyMonomial<Self>
        where I: IntoIterator<Item = usize>,
            I::IntoIter: ExactSizeIterator
    {
        self.get_ring().create_monomial(exponents)
    }

    fn clone_monomial(&self, mon: &PolyMonomial<Self>) -> PolyMonomial<Self> {
        self.get_ring().clone_monomial(mon)
    }

    ///
    /// Returns the coefficient of a polynomial corresponding to a monomial.
    /// 
    /// For details, see [`MultivariatePolyRing::coefficient_at()`].
    /// 
    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, m: &PolyMonomial<Self>) -> &'a PolyCoeff<Self> {
        self.get_ring().coefficient_at(f, m)
    }

    ///
    /// Returns an iterator over all nonzero terms of the polynomial.
    /// 
    /// For details, see [`MultivariatePolyRing::terms()`].
    /// 
    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as MultivariatePolyRing>::TermIter<'a> {
        self.get_ring().terms(f)
    }

    ///
    /// Creates a new polynomial by summing up all the given terms.
    /// 
    fn from_terms<I>(&self, terms: I) -> El<Self>
        where I: IntoIterator<Item = (PolyCoeff<Self>, PolyMonomial<Self>)>
    {
        let mut result = self.zero();
        self.get_ring().add_assign_from_terms(&mut result, terms);
        return result;
    }

    ///
    /// Evaluates the polynomial at the given values.
    /// 
    /// For details, see [`MultivariatePolyRing::evaluate()`].
    /// 
    fn evaluate<R, V, H>(&self, f: &El<Self>, value: V, hom: H) -> R::Element
        where R: ?Sized + RingBase,
            H: Homomorphism<<<Self::Type as RingExtension>::BaseRing as RingStore>::Type, R>,
            V: VectorFn<R::Element>
    {
        self.get_ring().evaluate(f, value, hom)
    }
    
    ///
    /// Returns the homomorphism `R[X1, ..., Xm] -> S[X1, ..., Xm]` that is induced by
    /// applying the given homomorphism `R -> S` coefficient-wise.
    /// 
    fn into_lifted_hom<P, H>(self, from: P, hom: H) -> CoefficientHom<P, Self, H>
        where P: RingStore,
            P::Type: MultivariatePolyRing,
            H: Homomorphism<<<P::Type as RingExtension>::BaseRing as RingStore>::Type, <<Self::Type as RingExtension>::BaseRing as RingStore>::Type>
    {
        CoefficientHom {
            from: from,
            to: self,
            hom: hom
        }
    }

    ///
    /// Returns the homomorphism `R[X1, ..., Xm] -> S[X1, ..., Xm]` that is induced by
    /// applying the given homomorphism `R -> S` coefficient-wise.
    /// 
    /// If the ownership of this ring should be transferred to the homomorphism, consider
    /// using [`MultivariatePolyRingStore::into_lifted_hom()`].
    /// 
    fn lifted_hom<'a, P, H>(&'a self, from: P, hom: H) -> CoefficientHom<P, &'a Self, H>
        where P: RingStore,
            P::Type: MultivariatePolyRing,
            H: Homomorphism<<<P::Type as RingExtension>::BaseRing as RingStore>::Type, <<Self::Type as RingExtension>::BaseRing as RingStore>::Type>
    {
        self.into_lifted_hom(from, hom)
    }

    ///
    /// Invokes the function with a wrapped version of the indeterminates of this poly ring.
    /// Use for convenient creation of polynomials.
    /// 
    /// Note however that [`MultivariatePolyRingStore::from_terms()`] might be more performant.
    /// 
    /// # Example
    /// ```
    /// use feanor_math::assert_el_eq;
    /// use feanor_math::homomorphism::*;
    /// use feanor_math::ring::*;
    /// use feanor_math::rings::multivariate::*;
    /// use feanor_math::rings::zn::zn_64::*;
    /// use feanor_math::rings::multivariate::multivariate_impl::*;
    /// let base_ring = Zn::new(7);
    /// let poly_ring = MultivariatePolyRingImpl::new(base_ring, 3);
    /// let f_version1 = poly_ring.from_terms([(base_ring.int_hom().map(3), poly_ring.create_monomial([0, 0, 0])), (base_ring.int_hom().map(2), poly_ring.create_monomial([0, 1, 1])), (base_ring.one(), poly_ring.create_monomial([2, 0, 0]))].into_iter());
    /// let f_version2 = poly_ring.with_wrapped_indeterminates_dyn(|[x, y, z]| [3 + 2 * y * z + x.pow_ref(2)]).into_iter().next().unwrap();
    /// let [f_version3] = poly_ring.with_wrapped_indeterminates(|[x, y, z]| [3 + 2 * y * z + x.pow_ref(2)]);
    /// assert_el_eq!(poly_ring, f_version1, f_version2);
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    fn with_wrapped_indeterminates_dyn<'a, F, T, const N: usize>(&'a self, f: F) -> Vec<El<Self>>
        where F: FnOnce([&RingElementWrapper<&'a Self>; N]) -> T,
            T: IntoIterator<Item = RingElementWrapper<&'a Self>>
    {
        assert_eq!(self.indeterminate_count(), N);
        let wrapped_indets: [_; N] = std::array::from_fn(|i| RingElementWrapper::new(self, self.create_term(self.base_ring().one(), self.create_monomial((0..N).map(|j| if i == j { 1 } else { 0 })))));
        f(std::array::from_fn(|i| &wrapped_indets[i])).into_iter().map(|f| f.unwrap()).collect()
    }

    ///
    /// Same as [`MultivariatePolyRingStore::with_wrapped_indeterminates_dyn()`], but returns result
    /// as an array to allow pattern matching.
    /// 
    /// # Example
    /// ```
    /// use feanor_math::assert_el_eq;
    /// use feanor_math::homomorphism::*;
    /// use feanor_math::ring::*;
    /// use feanor_math::rings::multivariate::*;
    /// use feanor_math::rings::zn::zn_64::*;
    /// use feanor_math::rings::multivariate::multivariate_impl::*;
    /// let base_ring = Zn::new(7);
    /// let poly_ring = MultivariatePolyRingImpl::new(base_ring, 3);
    /// let f_version1 = poly_ring.from_terms([(base_ring.int_hom().map(3), poly_ring.create_monomial([0, 0, 0])), (base_ring.int_hom().map(2), poly_ring.create_monomial([0, 1, 1])), (base_ring.one(), poly_ring.create_monomial([2, 0, 0]))].into_iter());
    /// let f_version2 = poly_ring.with_wrapped_indeterminates_dyn(|[x, y, z]| [3 + 2 * y * z + x.pow_ref(2)]).into_iter().next().unwrap();
    /// let [f_version3] = poly_ring.with_wrapped_indeterminates(|[x, y, z]| [3 + 2 * y * z + x.pow_ref(2)]);
    /// assert_el_eq!(poly_ring, f_version1, f_version2);
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    fn with_wrapped_indeterminates<'a, F, const N: usize, const M: usize>(&'a self, f: F) -> [El<Self>; M]
        where F: FnOnce([&RingElementWrapper<&'a Self>; N]) -> [RingElementWrapper<&'a Self>; M]
    {
        assert_eq!(self.indeterminate_count(), N);
        let wrapped_indets: [_; N] = std::array::from_fn(|i| RingElementWrapper::new(self, self.create_term(self.base_ring().one(), self.create_monomial((0..N).map(|j| if i == j { 1 } else { 0 })))));
        let mut result_it = f(std::array::from_fn(|i| &wrapped_indets[i])).into_iter().map(|f| f.unwrap());
        let result = std::array::from_fn(|_| result_it.next().unwrap());
        debug_assert!(result_it.next().is_none());
        return result;
    }
}

impl<P> MultivariatePolyRingStore for P
    where P: RingStore,
        P::Type: MultivariatePolyRing {}

///
/// Trait for monomial orders, i.e. orderings on the monomials `X1^i1 ... Xm^im` of a multivariate
/// polynomial ring that are compatible with multiplication. These are also sometimes called term
/// orders.
/// 
/// To be precise, a total order on monomials is a monomial order, if for all `u, v, w` with `u <= v`
/// also `uw <= vw`.
/// 
pub trait MonomialOrder: Clone {

    fn compare<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> Ordering
        where P: RingStore,
            P::Type: MultivariatePolyRing;

    ///
    /// Checks whether two monomials are equal.
    /// 
    /// This may be faster than [`MonomialOrder::compare()`], but clearly must have
    /// a compatible behavior.
    /// 
    fn eq_mon<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> bool
        where P: RingStore,
            P::Type: MultivariatePolyRing
    {
        self.compare(ring, lhs, rhs) == Ordering::Equal
    }

    ///
    /// Whether this order is the same as the given other order, i.e. [`MonomialOrder::compare()`]
    /// gives the same output on all inputs.
    /// 
    /// Many monomial orders are likely to be implemented as zero-sized types with only a single
    /// instance. In this case, the default implementation is sufficient.
    /// 
    fn is_same<O>(&self, rhs: &O) -> bool
        where O: MonomialOrder
    {
        assert!(self.as_any().is_some());
        assert!(std::mem::size_of::<Self>() == 0);
        if let Some(rhs_as_any) = rhs.as_any() {
            self.as_any().unwrap().type_id() == rhs_as_any.type_id()
        } else {
            false
        }
    }

    ///
    /// Upcasts this reference to `&dyn Any`, which is sometimes required to compare monomial order
    /// objects of different types.
    /// 
    fn as_any(&self) -> Option<&dyn Any>;
}

///
/// Trait for [`MonomialOrder`]s that are graded, i.e. for `v, w` with `deg(v) < deg(w)`
/// they always satisfy `v < w`.
/// 
pub trait GradedMonomialOrder: MonomialOrder {}

///
/// The graded reverse lexicographic order. This is the most important monomial order, since
/// computing a Groebner basis w.r.t. this order is usually more efficient than for other orders.
/// Also sometimes referred to as "grevlex".
/// 
/// Concretely, this is the ordering of monomials we get by first comparing monomial degrees, and
/// in case of a tie reverse the outcome of a lexicographic comparison, using a reversed order
/// of variables.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::multivariate::*;
/// # use feanor_math::rings::multivariate::multivariate_impl::*;
/// # use feanor_math::primitive_int::*; 
/// # use std::cmp::Ordering;
/// let poly_ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, 3);
/// let monomials_descending = [
///     [2, 0, 0], // x1^2
///     [1, 1, 0], // x1 x2
///     [0, 2, 0], // x2^2
///     [1, 0, 1], // x1 x3
///     [0, 1, 1], // x2 x3
///     [0, 0, 2], // x3^2
/// ].into_iter().map(|m| poly_ring.create_monomial(m)).collect::<Vec<_>>();
/// for i in 1..6 {
///     assert!(DegRevLex.compare(&poly_ring, &monomials_descending[i - 1], &monomials_descending[i]) == Ordering::Greater);
/// }
/// ```
/// 
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DegRevLex;

impl MonomialOrder for DegRevLex {

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self as &dyn Any)
    }

    fn compare<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> Ordering
        where P: RingStore,
            P::Type:MultivariatePolyRing
    {
        let lhs_deg = ring.monomial_deg(lhs);
        let rhs_deg = ring.monomial_deg(rhs);
        if lhs_deg < rhs_deg {
            return Ordering::Less;
        } else if lhs_deg > rhs_deg {
            return Ordering::Greater;
        } else {
            for i in (0..ring.indeterminate_count()).rev() {
                if ring.exponent_at(lhs, i) > ring.exponent_at(rhs, i) {
                    return Ordering::Less
                } else if ring.exponent_at(lhs, i) < ring.exponent_at(rhs, i) {
                    return Ordering::Greater;
                }
            }
            return Ordering::Equal;
        }
    }
}

impl GradedMonomialOrder for DegRevLex {}

///
/// Lexicographic ordering of monomials.
/// 
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Lex;

impl MonomialOrder for Lex {

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self as &dyn Any)
    }

    fn compare<P>(&self, ring: P, lhs: &PolyMonomial<P>, rhs: &PolyMonomial<P>) -> Ordering
        where P:RingStore,
            P::Type:MultivariatePolyRing
    {
        for i in 0..ring.indeterminate_count() {
            match ring.exponent_at(lhs, i).cmp(&ring.exponent_at(rhs, i)) {
                Ordering::Less => { return Ordering::Less; },
                Ordering::Greater => { return Ordering::Greater; },
                Ordering::Equal => {}
            }
        }
        return Ordering::Equal;
    }
}

pub struct CoefficientHom<PFrom, PTo, H>
    where PFrom: RingStore,
        PTo: RingStore,
        PFrom::Type: MultivariatePolyRing,
        PTo::Type: MultivariatePolyRing,
        H: Homomorphism<<<PFrom::Type as RingExtension>::BaseRing as RingStore>::Type, <<PTo::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    from: PFrom,
    to: PTo,
    hom: H
}

impl<PFrom, PTo, H> Homomorphism<PFrom::Type, PTo::Type> for CoefficientHom<PFrom, PTo, H>
    where PFrom: RingStore,
        PTo: RingStore,
        PFrom::Type: MultivariatePolyRing,
        PTo::Type: MultivariatePolyRing,
        H: Homomorphism<<<PFrom::Type as RingExtension>::BaseRing as RingStore>::Type, <<PTo::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    type DomainStore = PFrom;
    type CodomainStore = PTo;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.to
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.from
    }

    fn map(&self, x: <PFrom::Type as RingBase>::Element) -> <PTo::Type as RingBase>::Element {
        self.map_ref(&x)
    }

    fn map_ref(&self, x: &<PFrom::Type as RingBase>::Element) -> <PTo::Type as RingBase>::Element {
        self.to.get_ring().map_terms(self.from.get_ring(), x, &self.hom)
    }
}

pub mod generic_impls {
    use std::fmt::{Formatter, Result};
    use super::*;

    #[stability::unstable(feature = "enable")]
    pub fn print<P>(ring: P, poly: &El<P>, out: &mut Formatter, env: EnvBindingStrength) -> Result
        where P: RingStore,
            P::Type: MultivariatePolyRing
    {
        if ring.is_zero(poly) {
            ring.base_ring().get_ring().dbg_within(&ring.base_ring().zero(), out, env)?;
        } else {
            if env >= EnvBindingStrength::Product {
                write!(out, "(")?;
            }
            
            let mut print_term = |c: &PolyCoeff<P>, m: &PolyMonomial<P>, with_plus: bool| {
                if with_plus {
                    write!(out, " + ")?;
                }
                let is_constant_term = ring.monomial_deg(m) == 0;
                if !ring.base_ring().is_one(c) || is_constant_term {
                    ring.base_ring().get_ring().dbg_within(c, out, if is_constant_term { EnvBindingStrength::Sum } else { EnvBindingStrength::Product })?;
                    if !is_constant_term {
                        write!(out, " * ")?;
                    }
                }
                let mut needs_space = false;
                for i in 0..ring.indeterminate_count() {
                    if ring.exponent_at(m, i) > 0 {
                        if needs_space {
                            write!(out, " * ")?;
                        }
                        write!(out, "X{}", i)?;
                        needs_space = true;
                    }
                    if ring.exponent_at(m, i) > 1 {
                        write!(out, "^{}", ring.exponent_at(m, i))?;
                    }
                }
                return Ok::<(), std::fmt::Error>(());
            };
            
            for (i, (c, m)) in ring.terms(poly).enumerate() {
                print_term(c, m, i != 0)?;
            }
            if env >= EnvBindingStrength::Product {
                write!(out, ")")?;
            }
        }

        return Ok(());
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use super::*;
    use crate::seq::*;

    #[stability::unstable(feature = "enable")]
    pub fn test_poly_ring_axioms<P: RingStore, I: Iterator<Item = PolyCoeff<P>>>(ring: P, interesting_base_ring_elements: I)
        where P::Type: MultivariatePolyRing
    {
        let elements = interesting_base_ring_elements.collect::<Vec<_>>();
        let n = ring.indeterminate_count();
        let base_ring = ring.base_ring();

        // test multiplication of variables
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                let xj = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 })));
                let xixj = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == i && k == j { 2 } else if k == j || k == i { 1 } else { 0 })));
                assert_el_eq!(ring, xixj, ring.mul(xi, xj));
            }
        }

        // test monomial operations
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 }));
                let xj = ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 }));
                let xixj_lcm = ring.create_monomial((0..n).map(|k| if k == j || k == i { 1 } else { 0 }));
                assert_el_eq!(ring, ring.create_term(base_ring.one(), xixj_lcm), ring.create_term(base_ring.one(), ring.monomial_lcm(xi, &xj)));
            }
        }

        // all monomials should be different
        for i in 0..n {
            for j in 0..n {
                let xi = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                let xj = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == j { 1 } else { 0 })));
                assert!((i == j) == ring.eq_el(&xi, &xj));
            }
        }

        // monomials shouldn't be zero divisors
        for i in 0..n {
            for a in &elements {
                let xi = ring.create_term(base_ring.one(), ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 })));
                assert!(base_ring.is_zero(a) == ring.is_zero(&ring.inclusion().mul_ref_map(&xi, a)));
            }
        }

        // test add_assign_from_terms
        for i in 0..n {
            let xi = ring.create_monomial((0..n).map(|k| if k == i { 1 } else { 0 }));
            let mut a = ring.create_term(base_ring.int_hom().map(3), ring.create_monomial((0..n).map(|_| 0)));
            let terms_with_multiples = [
                (base_ring.one(), ring.clone_monomial(&xi)),
                (base_ring.one(), ring.clone_monomial(&xi)),
                (base_ring.one(), ring.create_monomial((0..n).map(|_| 0))),
                (base_ring.one(), ring.create_monomial((0..n).map(|_| 0))),
                (base_ring.one(), ring.clone_monomial(&xi)),
                (base_ring.one(), ring.create_monomial((0..n).map(|_| 0))),
                (base_ring.one(), ring.clone_monomial(&xi)),
                (base_ring.one(), ring.create_monomial((0..n).map(|_| 0))),
            ];
            ring.get_ring().add_assign_from_terms(&mut a, terms_with_multiples);
            assert_el_eq!(&ring, ring.from_terms([
                (base_ring.int_hom().map(7), ring.create_monomial((0..n).map(|_| 0))),
                (base_ring.int_hom().map(4), xi),
            ]), a);
        }

        if n >= 2 {
            let one = ring.create_monomial((0..n).map(|_| 0));
            let x0 = ring.create_monomial((0..n).map(|k| if k == 0 { 1 } else { 0 }));
            let x1 = ring.create_monomial((0..n).map(|k| if k == 1 { 1 } else { 0 }));
            let x0_v = ring.create_term(base_ring.one(), ring.clone_monomial(&x0));
            let x1_v = ring.create_term(base_ring.one(), ring.clone_monomial(&x1));
            let x0_2 = ring.create_monomial((0..n).map(|k| if k == 0 { 2 } else { 0 }));
            let x0_3 = ring.create_monomial((0..n).map(|k| if k == 0 { 3 } else { 0 }));
            let x0_4 = ring.create_monomial((0..n).map(|k| if k == 0 { 4 } else { 0 }));
            let x0x1 = ring.create_monomial((0..n).map(|k| if k == 0 || k == 1 { 1 } else { 0 }));
            let x0_3x1 = ring.create_monomial((0..n).map(|k| if k == 0 { 3 } else if k == 1 { 1 } else { 0 }));
            let x1_2 = ring.create_monomial((0..n).map(|k| if k == 1 { 2 } else { 0 }));

            // test product
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let f = ring.add(ring.inclusion().mul_ref_map(&x0_v, a), ring.inclusion().mul_ref_map(&x1_v, b));
                        let g = ring.add(ring.inclusion().mul_ref_map(&x0_v, c), ring.clone_el(&x1_v));
                        let h = ring.from_terms([
                            (base_ring.mul_ref(a, c), ring.clone_monomial(&x0_2)),
                            (base_ring.add_ref_snd(base_ring.mul_ref(b, c), a), ring.clone_monomial(&x0x1)),
                            (base_ring.clone_el(b), ring.clone_monomial(&x1_2)),
                        ].into_iter());
                        assert_el_eq!(ring, h, ring.mul(f, g));
                    }
                }
            }

            // test sum
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let f = ring.from_terms([
                            (base_ring.clone_el(a), ring.clone_monomial(&one)),
                            (base_ring.clone_el(c), ring.clone_monomial(&x0_2)),
                            (base_ring.one(), ring.clone_monomial(&x0x1))
                        ]);
                        let g = ring.from_terms([
                            (base_ring.clone_el(b), ring.clone_monomial(&x0)),
                            (base_ring.one(), ring.clone_monomial(&x0_2)),
                        ]);
                        let h = ring.from_terms([
                            (base_ring.clone_el(a), ring.clone_monomial(&one)),
                            (base_ring.clone_el(b), ring.clone_monomial(&x0)),
                            (base_ring.add_ref_fst(c, base_ring.one()), ring.clone_monomial(&x0_2)),
                            (base_ring.one(), ring.clone_monomial(&x0x1))
                        ]);
                        assert_el_eq!(ring, h, ring.add(f, g));
                    }
                }
            }

            // test mul_assign_monomial
            for a in &elements {
                for b in &elements {
                    for c in &elements {
                        let mut f = ring.from_terms([
                            (base_ring.clone_el(a), ring.clone_monomial(&one)),
                            (base_ring.clone_el(b), ring.clone_monomial(&x0)),
                            (base_ring.clone_el(c), ring.clone_monomial(&x0_2)),
                            (base_ring.one(), ring.clone_monomial(&x0x1))
                        ]);
                        let h = ring.from_terms([
                            (base_ring.clone_el(a), ring.clone_monomial(&x0_2)),
                            (base_ring.clone_el(b), ring.clone_monomial(&x0_3)),
                            (base_ring.clone_el(c), ring.clone_monomial(&x0_4)),
                            (base_ring.one(), ring.clone_monomial(&x0_3x1))
                        ]);
                        let m = ring.clone_monomial(&x0_2);
                        ring.mul_assign_monomial(&mut f, m);
                        assert_el_eq!(ring, h, f);
                    }
                }
            }

            // test evaluate
            for a in &elements {
                for b in &elements {
                    let f = ring.from_terms([
                        (base_ring.int_hom().map(3), ring.clone_monomial(&one)),
                        (base_ring.int_hom().map(10), ring.clone_monomial(&x0)),
                        (base_ring.neg_one(), ring.clone_monomial(&x0_2)),
                        (base_ring.one(), ring.clone_monomial(&x0x1))
                    ]);
                    let expected = <_ as RingStore>::sum(base_ring, [
                        base_ring.int_hom().map(3),
                        base_ring.int_hom().mul_ref_map(a, &10),
                        base_ring.negate(base_ring.pow(base_ring.clone_el(a), 2)),
                        base_ring.mul_ref(a, b)
                    ]);
                    let values = [base_ring.clone_el(a), base_ring.clone_el(b)].into_iter().chain((0..(ring.indeterminate_count() - 2)).map(|_| base_ring.zero())).collect::<Vec<_>>();
                    assert_el_eq!(&base_ring, &expected, &ring.evaluate(&f, values.as_ring_el_fn(base_ring), &base_ring.identity()));
                }
            }
        }
    }
}