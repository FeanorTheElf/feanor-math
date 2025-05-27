use std::collections::HashMap;

use crate::divisibility::*;
use crate::ring::*;
use crate::homomorphism::*;
use crate::wrapper::RingElementWrapper;

///
/// Contains [`dense_poly::DensePolyRing`], an implementation of univariate polynomials
/// based on dense coefficient storage.
/// 
pub mod dense_poly;
///
/// Contains [`sparse_poly::SparsePolyRing`], an implementation of univariate polynomials
/// based on sparse coefficient storage.
/// 
pub mod sparse_poly;

///
/// Trait for all rings that represent the polynomial ring `R[X]` with
/// any base ring R.
/// 
/// Currently, the two implementations of this type of ring are [`dense_poly::DensePolyRing`]
/// and [`sparse_poly::SparsePolyRing`].
/// 
pub trait PolyRing: RingExtension {

    type TermsIterator<'a>: Iterator<Item = (&'a El<Self::BaseRing>, usize)>
        where Self: 'a;

    ///
    /// Returns the indeterminate `X` generating this polynomial ring.
    /// 
    fn indeterminate(&self) -> Self::Element;

    ///
    /// Returns all the nonzero terms of the given polynomial.
    /// 
    /// If the base ring is only approximate, it is valid to return "zero" terms,
    /// whatever that actually means.
    /// 
    fn terms<'a>(&'a self, f: &'a Self::Element) -> Self::TermsIterator<'a>;
    
    ///
    /// Adds the given terms to the given polynomial.
    /// 
    fn add_assign_from_terms<I>(&self, lhs: &mut Self::Element, rhs: I)
        where I: IntoIterator<Item = (El<Self::BaseRing>, usize)>
    {
        let self_ring = RingRef::new(self);
        self.add_assign(lhs, self_ring.sum(
            rhs.into_iter().map(|(c, i)| self.mul(self.from(c), self_ring.pow(self.indeterminate(), i)))
        ));
    }

    ///
    /// Multiplies the given polynomial with `X^rhs_power`.
    /// 
    fn mul_assign_monomial(&self, lhs: &mut Self::Element, rhs_power: usize) {
        self.mul_assign(lhs, RingRef::new(self).pow(self.indeterminate(), rhs_power));
    }

    ///
    /// Returns the coefficient of `f` that corresponds to the monomial `X^i`.
    /// 
    fn coefficient_at<'a>(&'a self, f: &'a Self::Element, i: usize) -> &'a El<Self::BaseRing>;

    ///
    /// Returns the degree of the polynomial `f`, i.e. the value `d` such that `f` can be written as
    /// `f(X) = a0 + a1 * X + a2 * X^2 + ... + ad * X^d`. Returns `None` if `f` is zero.
    /// 
    fn degree(&self, f: &Self::Element) -> Option<usize>;

    ///
    /// Compute the euclidean division by a monic polynomial `rhs`.
    /// 
    /// Concretely, if `rhs` is a monic polynomial (polynomial with highest coefficient equal to 1), then
    /// there exist unique `q, r` such that `lhs = rhs * q + r` and `deg(r) < deg(rhs)`. These are returned.
    /// This function panics if `rhs` is not monic.
    /// 
    fn div_rem_monic(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element);
    
    fn map_terms<P, H>(&self, from: &P, el: &P::Element, hom: H) -> Self::Element
        where P: ?Sized + PolyRing,
            H: Homomorphism<<P::BaseRing as RingStore>::Type, <Self::BaseRing as RingStore>::Type>
    {
        assert!(self.base_ring().get_ring() == hom.codomain().get_ring());
        assert!(from.base_ring().get_ring() == hom.domain().get_ring());
        RingRef::new(self).from_terms(from.terms(el).map(|(c, i)| (hom.map_ref(c), i)))
    }
    
    ///
    /// Possibly divides all coefficients of the polynomial by a common factor,
    /// in order to make them "smaller". 
    /// 
    /// In cases where we mainly care about polynomials up to scaling by non-zero 
    /// divisors of the base ring, balancing intermediate polynomials can improve performance.
    /// 
    /// For more details on balancing, see [`DivisibilityRing::balance_factor()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn balance_poly(&self, f: &mut Self::Element) -> Option<El<Self::BaseRing>>
        where <Self::BaseRing as RingStore>::Type: DivisibilityRing
    {
        let balance_factor = self.base_ring().get_ring().balance_factor(self.terms(f).map(|(c, _)| c));
        if let Some(balance_factor) = &balance_factor {
            *f = RingRef::new(self).from_terms(self.terms(f).map(|(c, i)| (self.base_ring().checked_div(c, &balance_factor).unwrap(), i)));
        }
        return balance_factor;
    }

    ///
    /// Evaluates the given polynomial at the given values.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::poly::*;
    /// # use feanor_math::primitive_int::*; 
    /// let ring = dense_poly::DensePolyRing::new(StaticRing::<i32>::RING, "X");
    /// let x = ring.indeterminate();
    /// let poly = ring.add(ring.clone_el(&x), ring.pow(x, 2));
    /// assert_eq!(12, ring.evaluate(&poly, &3, &StaticRing::<i32>::RING.identity()));
    /// ```
    /// 
    fn evaluate<R, H>(&self, f: &Self::Element, value: &R::Element, hom: H) -> R::Element
        where R: ?Sized + RingBase,
            H: Homomorphism<<Self::BaseRing as RingStore>::Type, R>
    {
        hom.codomain().sum(self.terms(f).map(|(c, i)| {
            let result = hom.codomain().pow(hom.codomain().clone_el(value), i);
            hom.mul_ref_snd_map(result, c)
        }))
    }
}

///
/// [`RingStore`] corresponding to [`PolyRing`].
/// 
pub trait PolyRingStore: RingStore
    where Self::Type: PolyRing
{
    delegate!{ PolyRing, fn indeterminate(&self) -> El<Self> }
    delegate!{ PolyRing, fn degree(&self, f: &El<Self>) -> Option<usize> }
    delegate!{ PolyRing, fn mul_assign_monomial(&self, lhs: &mut El<Self>, rhs_power: usize) -> () }
    delegate!{ PolyRing, fn div_rem_monic(&self, lhs: El<Self>, rhs: &El<Self>) -> (El<Self>, El<Self>) }

    ///
    /// See [`PolyRing::coefficient_at()`].
    /// 
    fn coefficient_at<'a>(&'a self, f: &'a El<Self>, i: usize) -> &'a El<<Self::Type as RingExtension>::BaseRing> {
        self.get_ring().coefficient_at(f, i)
    }

    ///
    /// See [`PolyRing::terms()`].
    /// 
    fn terms<'a>(&'a self, f: &'a El<Self>) -> <Self::Type as PolyRing>::TermsIterator<'a> {
        self.get_ring().terms(f)
    }

    ///
    /// Computes the polynomial from the given terms.
    /// 
    /// If the iterator gives a term for the same monomial `X^i` multiple times,
    /// the corresponding coefficients will be summed up.
    /// 
    fn from_terms<I>(&self, iter: I) -> El<Self>
        where I: IntoIterator<Item = (El<<Self::Type as RingExtension>::BaseRing>, usize)>,
    {
        let mut result = self.zero();
        self.get_ring().add_assign_from_terms(&mut result, iter);
        return result;
    }

    fn try_from_terms<E, I>(&self, iter: I) -> Result<El<Self>, E>
        where I: IntoIterator<Item = Result<(El<<Self::Type as RingExtension>::BaseRing>, usize), E>>,
    {
        let mut result = self.zero();
        let mut error = None;
        self.get_ring().add_assign_from_terms(&mut result, iter.into_iter().map(|t| match t {
            Ok(t) => Some(t),
            Err(e) => { error = Some(e); None }
        }).take_while(|t| t.is_some()).map(|t| t.unwrap()));
        if let Some(e) = error {
            return Err(e);
        } else {
            return Ok(result);
        }
    }

    ///
    /// Returns a reference to the leading coefficient of the given polynomial, or `None` if the
    /// polynomial is zero.
    /// 
    fn lc<'a>(&'a self, f: &'a El<Self>) -> Option<&'a El<<Self::Type as RingExtension>::BaseRing>> {
        Some(self.coefficient_at(f, self.degree(f)?))
    }
    
    ///
    /// Divides each coefficient of this polynomial by its leading coefficient, thus making it monic.
    /// 
    /// Panics if the leading coefficient is not a unit.
    /// 
    /// # Example
    /// ```rust
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::poly::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::rings::poly::dense_poly::*;
    /// # use feanor_math::primitive_int::*;
    /// let P = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    /// let f = P.from_terms([(6, 0), (3, 1)].into_iter());
    /// assert_el_eq!(P, P.from_terms([(2, 0), (1, 1)].into_iter()), P.normalize(f));
    /// ```
    /// 
    fn normalize(&self, mut f: El<Self>) -> El<Self>
        where <<Self::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
    {
        if self.is_zero(&f) {
            return f;
        } else if let Some(inv_lc) = self.base_ring().invert(self.lc(&f).unwrap()) {
            self.inclusion().mul_assign_ref_map(&mut f, &inv_lc);
            return f;
        } else {
            let lc = self.lc(&f).unwrap();
            return self.from_terms(self.terms(&f).map(|(c, i)| (self.base_ring().checked_div(c, &lc).unwrap(), i)));
        }
    }

    ///
    /// See [`PolyRing::evaluate()`].
    /// 
    fn evaluate<R, H>(&self, f: &El<Self>, value: &R::Element, hom: H) -> R::Element
        where R: ?Sized + RingBase,
            H: Homomorphism<<<Self::Type as RingExtension>::BaseRing as RingStore>::Type, R>
    {
        self.get_ring().evaluate(f, value, hom)
    }

    fn into_lifted_hom<P, H>(self, from: P, hom: H) -> CoefficientHom<P, Self, H>
        where P: RingStore,
            P::Type: PolyRing,
            H: Homomorphism<<<P::Type as RingExtension>::BaseRing as RingStore>::Type, <<Self::Type as RingExtension>::BaseRing as RingStore>::Type>
    {
        CoefficientHom {
            from: from,
            to: self,
            hom: hom
        }
    }

    fn lifted_hom<'a, P, H>(&'a self, from: P, hom: H) -> CoefficientHom<P, &'a Self, H>
        where P: RingStore,
            P::Type: PolyRing,
            H: Homomorphism<<<P::Type as RingExtension>::BaseRing as RingStore>::Type, <<Self::Type as RingExtension>::BaseRing as RingStore>::Type>
    {
        self.into_lifted_hom(from, hom)
    }
    
    ///
    /// Invokes the function with a wrapped version of the indeterminate of this poly ring.
    /// Use for convenient creation of polynomials.
    /// 
    /// Note however that [`PolyRingStore::from_terms()`] might be more performant.
    /// 
    /// # Example
    /// ```rust
    /// use feanor_math::assert_el_eq;
    /// use feanor_math::ring::*;
    /// use feanor_math::rings::poly::*;
    /// use feanor_math::homomorphism::*;
    /// use feanor_math::rings::zn::zn_64::*;
    /// use feanor_math::rings::poly::dense_poly::*;
    /// let base_ring = Zn::new(7);
    /// let poly_ring = DensePolyRing::new(base_ring, "X");
    /// let f_version1 = poly_ring.from_terms([(base_ring.int_hom().map(3), 0), (base_ring.int_hom().map(2), 1), (base_ring.one(), 3)].into_iter());
    /// let f_version2 = poly_ring.with_wrapped_indeterminate_dyn(|x| [3 + 2 * x + x.pow_ref(3)]).into_iter().next().unwrap();
    /// assert_el_eq!(poly_ring, f_version1, f_version2);
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    fn with_wrapped_indeterminate_dyn<'a, F, T>(&'a self, f: F) -> Vec<El<Self>>
        where F: FnOnce(&RingElementWrapper<&'a Self>) -> T,
            T: IntoIterator<Item = RingElementWrapper<&'a Self>>
    {
        let wrapped_indet = RingElementWrapper::new(self, self.indeterminate());
        f(&wrapped_indet).into_iter().map(|f| f.unwrap()).collect()
    }

    fn with_wrapped_indeterminate<'a, F, const M: usize>(&'a self, f: F) -> [El<Self>; M]
        where F: FnOnce(&RingElementWrapper<&'a Self>) -> [RingElementWrapper<&'a Self>; M]
    {
        let wrapped_indet = RingElementWrapper::new(self, self.indeterminate());
        let mut result_it = f(&wrapped_indet).into_iter();
        return std::array::from_fn(|_| result_it.next().unwrap().unwrap());
    }
    
    #[stability::unstable(feature = "enable")]
    fn balance_poly(&self, f: &mut El<Self>) -> Option<El<<Self::Type as RingExtension>::BaseRing>>
        where <<Self::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing
    {
        self.get_ring().balance_poly(f)
    }
}

pub struct CoefficientHom<PFrom, PTo, H>
    where PFrom: RingStore,
        PTo: RingStore,
        PFrom::Type: PolyRing,
        PTo::Type: PolyRing,
        H: Homomorphism<<<PFrom::Type as RingExtension>::BaseRing as RingStore>::Type, <<PTo::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    from: PFrom,
    to: PTo,
    hom: H
}

impl<PFrom, PTo, H> Homomorphism<PFrom::Type, PTo::Type> for CoefficientHom<PFrom, PTo, H>
    where PFrom: RingStore,
        PTo: RingStore,
        PFrom::Type: PolyRing,
        PTo::Type: PolyRing,
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

impl<R: RingStore> PolyRingStore for R
    where R::Type: PolyRing
{}

///
/// Computes the formal derivative of a polynomial.
/// 
/// The formal derivative is the linear map defined by
/// ```text
///   X^k  ->  k * X^(k - 1)
/// ```
/// 
pub fn derive_poly<P>(poly_ring: P, poly: &El<P>) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing
{
    poly_ring.from_terms(poly_ring.terms(poly)
        .filter(|(_, i)| *i > 0)
        .map(|(c, i)| (poly_ring.base_ring().int_hom().mul_ref_fst_map(c, i.try_into().unwrap()), i - 1))
    )
}

pub mod generic_impls {
    use crate::ring::*;
    use super::PolyRing;
    use crate::homomorphism::*;

    #[allow(type_alias_bounds)]
    #[stability::unstable(feature = "enable")]
    pub type Homomorphism<P1: PolyRing, P2: PolyRing> = <<P2::BaseRing as RingStore>::Type as CanHomFrom<<P1::BaseRing as RingStore>::Type>>::Homomorphism;

    #[stability::unstable(feature = "enable")]
    pub fn has_canonical_hom<P1: PolyRing, P2: PolyRing>(from: &P1, to: &P2) -> Option<Homomorphism<P1, P2>> 
        where <P2::BaseRing as RingStore>::Type: CanHomFrom<<P1::BaseRing as RingStore>::Type>
    {
        to.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    #[stability::unstable(feature = "enable")]
    pub fn map_in<P1: PolyRing, P2: PolyRing>(from: &P1, to: &P2, el: P1::Element, hom: &Homomorphism<P1, P2>) -> P2::Element
        where <P2::BaseRing as RingStore>::Type: CanHomFrom<<P1::BaseRing as RingStore>::Type>
    {
        let mut result = to.zero();
        to.add_assign_from_terms(&mut result, from.terms(&el).map(|(c, i)| (to.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(c), hom), i)));
        return result;
    }

    #[allow(type_alias_bounds)]
    #[stability::unstable(feature = "enable")]
    pub type Isomorphism<P1: PolyRing, P2: PolyRing> = <<P2::BaseRing as RingStore>::Type as CanIsoFromTo<<P1::BaseRing as RingStore>::Type>>::Isomorphism;

    #[stability::unstable(feature = "enable")]
    pub fn map_out<P1: PolyRing, P2: PolyRing>(from: &P1, to: &P2, el: P2::Element, iso: &Isomorphism<P1, P2>) -> P1::Element
        where <P2::BaseRing as RingStore>::Type: CanIsoFromTo<<P1::BaseRing as RingStore>::Type>
    {
        let mut result = from.zero();
        from.add_assign_from_terms(&mut result, to.terms(&el).map(|(c, i)| (to.base_ring().get_ring().map_out(from.base_ring().get_ring(), to.base_ring().clone_el(c), iso), i)));
        return result;
    }

    #[stability::unstable(feature = "enable")]
    pub fn dbg_poly<P: PolyRing>(ring: &P, el: &P::Element, out: &mut std::fmt::Formatter, unknown_name: &str, env: EnvBindingStrength) -> std::fmt::Result {
        let mut terms = ring.terms(el).fuse();
        let first_term = terms.next();
        if first_term.is_none() {
            return ring.base_ring().get_ring().dbg_within(&ring.base_ring().zero(), out, env);
        }
        let second_term = terms.next();
        let use_parenthesis = (env > EnvBindingStrength::Sum && second_term.is_some()) || 
            (env > EnvBindingStrength::Product && !(ring.base_ring().is_one(&first_term.as_ref().unwrap().0) && first_term.as_ref().unwrap().1 == 1)) ||
            env == EnvBindingStrength::Strongest;
        let mut terms = first_term.into_iter().chain(second_term.into_iter()).chain(terms);
        if use_parenthesis {
            write!(out, "(")?;
        }
        let print_unknown = |i: usize, out: &mut std::fmt::Formatter| {
            if i == 0 {
                // print nothing
                Ok(())
            } else if i == 1 {
                write!(out, "{}", unknown_name)
            } else {
                write!(out, "{}^{}", unknown_name, i)
            }
        };
        if let Some((c, i)) = terms.next() {
            if !ring.base_ring().is_one(c) || i == 0 {
                ring.base_ring().get_ring().dbg_within(c, out, if i == 0 { EnvBindingStrength::Sum } else { EnvBindingStrength:: Product })?;
            }
            print_unknown(i, out)?;
        } else {
            write!(out, "0")?;
        }
        while let Some((c, i)) = terms.next() {
            write!(out, " + ")?;
            if !ring.base_ring().is_one(c) || i == 0 {
                ring.base_ring().get_ring().dbg_within(c, out, if i == 0 { EnvBindingStrength::Sum } else { EnvBindingStrength:: Product })?;
            }
            print_unknown(i, out)?;
        }
        if use_parenthesis {
            write!(out, ")")?;
        }
        return Ok(());
    }
}

pub fn transpose_indeterminates<P1, P2, H>(from: P1, to: P2, base_hom: H) -> impl Homomorphism<P1::Type, P2::Type>
    where P1: RingStore, P1::Type: PolyRing,
        P2: RingStore, P2::Type: PolyRing,
        <<P1::Type as RingExtension>::BaseRing as RingStore>::Type: PolyRing,
        <<P2::Type as RingExtension>::BaseRing as RingStore>::Type: PolyRing,
        H: Homomorphism<<<<<P1::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type,
            <<<<P2::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    LambdaHom::new(from, to, move |from, to, x| {
        let mut result_terms: HashMap<usize, Vec<(_, usize)>> = HashMap::new();
        for (f, i) in from.terms(x) {
            for (c, j) in from.base_ring().terms(f) {
                match result_terms.entry(j) {
                    std::collections::hash_map::Entry::Occupied(mut e) => { e.get_mut().push((base_hom.map_ref(c), i)); },
                    std::collections::hash_map::Entry::Vacant(e) => { _ = e.insert(vec![(base_hom.map_ref(c), i)]); }
                }
            }
        }
        return to.from_terms(result_terms.into_iter().map(|(j, f)| (to.base_ring().from_terms(f.into_iter()), j)));
    })
}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {
    use super::*;

    pub fn test_poly_ring_axioms<R: PolyRingStore, I: Iterator<Item = El<<R::Type as RingExtension>::BaseRing>>>(ring: R, interesting_base_ring_elements: I)
        where R::Type: PolyRing
    {    
        let x = ring.indeterminate();
        let elements = interesting_base_ring_elements.collect::<Vec<_>>();
        
        // test linear independence of X
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    for d in &elements {
                        let a_bx = ring.add(ring.inclusion().map_ref(a), ring.mul_ref_snd(ring.inclusion().map_ref(b), &x));
                        let c_dx = ring.add(ring.inclusion().map_ref(c), ring.mul_ref_snd(ring.inclusion().map_ref(d), &x));
                        assert!(ring.eq_el(&a_bx, &c_dx) == (ring.base_ring().eq_el(a, c) && ring.base_ring().eq_el(b, d)));
                    }
                }
            }
        }
        
        // elementwise addition follows trivially from the ring axioms

        // technically, convoluted multiplication follows from the ring axioms too, but test it anyway
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    for d in &elements {
                        let a_bx = ring.add(ring.inclusion().map_ref(a), ring.mul_ref_snd(ring.inclusion().map_ref(b), &x));
                        let c_dx = ring.add(ring.inclusion().map_ref(c), ring.mul_ref_snd(ring.inclusion().map_ref(d), &x));
                        let result = <_ as RingStore>::sum(&ring, [
                            ring.mul(ring.inclusion().map_ref(a), ring.inclusion().map_ref(c)),
                            ring.mul(ring.inclusion().map_ref(a), ring.mul_ref_snd(ring.inclusion().map_ref(d), &x)),
                            ring.mul(ring.inclusion().map_ref(b), ring.mul_ref_snd(ring.inclusion().map_ref(c), &x)),
                            ring.mul(ring.inclusion().map_ref(b), ring.mul(ring.inclusion().map_ref(d), ring.pow(ring.clone_el(&x), 2)))
                        ].into_iter());
                        assert_el_eq!(ring, result, ring.mul(a_bx, c_dx));
                    }
                }
            }
        }

        // test terms(), from_terms()
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    let f = <_ as RingStore>::sum(&ring, [
                        ring.inclusion().map_ref(a),
                        ring.mul_ref_snd(ring.inclusion().map_ref(b), &x),
                        ring.mul(ring.inclusion().map_ref(c), ring.pow(ring.clone_el(&x), 3))
                    ].into_iter());
                    let actual = ring.from_terms([(ring.base_ring().clone_el(a), 0), (ring.base_ring().clone_el(c), 3), (ring.base_ring().clone_el(b), 1)].into_iter());
                    assert_el_eq!(ring, f, actual);
                    assert_el_eq!(ring, f, ring.from_terms(ring.terms(&f).map(|(c, i)| (ring.base_ring().clone_el(c), i))));
                }
            }
        }

        // test div_rem_monic()
        for a in &elements {
            for b in &elements {
                for c in &elements {
                    let f = ring.from_terms([(ring.base_ring().clone_el(a), 0), (ring.base_ring().clone_el(b), 3)].into_iter());
                    let g = ring.from_terms([(ring.base_ring().negate(ring.base_ring().clone_el(c)), 0), (ring.base_ring().one(), 1)].into_iter());

                    let (quo, rem) = ring.div_rem_monic(ring.clone_el(&f), &g);
                    assert_el_eq!(
                        &ring,
                        &ring.from_terms([(ring.base_ring().add_ref_fst(a, ring.base_ring().mul_ref_fst(b, ring.base_ring().pow(ring.base_ring().clone_el(c), 3))), 0)].into_iter()),
                        &rem
                    );
                    assert_el_eq!(
                        &ring,
                        &f,
                        &ring.add(rem, ring.mul(quo, g))
                    );
                }
            }
        }

        // test evaluate()
        let hom = ring.base_ring().int_hom();
        let base_ring = hom.codomain();
        let f = ring.from_terms([(hom.map(1), 0), (hom.map(3), 1), (hom.map(7), 3)].into_iter());
        for a in &elements {
            assert_el_eq!(
                &base_ring,
                &base_ring.add(base_ring.one(), base_ring.add(base_ring.mul_ref_snd(hom.map(3), a), base_ring.mul(hom.map(7), base_ring.pow(base_ring.clone_el(a), 3)))),
                &ring.evaluate(&f, a, &base_ring.identity())
            )
        }
    }
}

#[cfg(test)]
use std::fmt::{Display, Formatter};
#[cfg(test)]
use dense_poly::DensePolyRing;
#[cfg(test)]
use crate::primitive_int::StaticRing;

#[test]
fn test_dbg_poly() {
    let ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let [f1, f2, f3, f4] = ring.with_wrapped_indeterminate(|X| [X.clone(), X + 1, 2 * X + 1, 2 * X]);

    fn to_str(ring: &DensePolyRing<StaticRing<i64>>, f: &El<DensePolyRing<StaticRing<i64>>>, env: EnvBindingStrength) -> String {
        struct DisplayEl<'a> {
            ring: &'a DensePolyRing<StaticRing<i64>>,
            f:  &'a El<DensePolyRing<StaticRing<i64>>>,
            env: EnvBindingStrength
        }
        impl<'a> Display for DisplayEl<'a> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                generic_impls::dbg_poly(self.ring.get_ring(), self.f, f, "X", self.env)
            }
        }
        return format!("{}", DisplayEl { ring, f, env });
    }

    assert_eq!("X", to_str(&ring, &f1, EnvBindingStrength::Sum));
    assert_eq!("X", to_str(&ring, &f1, EnvBindingStrength::Product));
    assert_eq!("X", to_str(&ring, &f1, EnvBindingStrength::Power));
    assert_eq!("1 + X", to_str(&ring, &f2, EnvBindingStrength::Sum));
    assert_eq!("(1 + X)", to_str(&ring, &f2, EnvBindingStrength::Product));
    assert_eq!("(1 + X)", to_str(&ring, &f2, EnvBindingStrength::Power));
    assert_eq!("1 + 2X", to_str(&ring, &f3, EnvBindingStrength::Sum));
    assert_eq!("(1 + 2X)", to_str(&ring, &f3, EnvBindingStrength::Product));
    assert_eq!("(1 + 2X)", to_str(&ring, &f3, EnvBindingStrength::Power));
    assert_eq!("2X", to_str(&ring, &f4, EnvBindingStrength::Sum));
    assert_eq!("2X", to_str(&ring, &f4, EnvBindingStrength::Product));
    assert_eq!("(2X)", to_str(&ring, &f4, EnvBindingStrength::Power));
}