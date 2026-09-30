//! Algorithms for computing Groebner bases of ideals in multivariate polynomial rings.
//!
//! The main entry point is [`buchberger()`] (or [`buchberger_with_strategy()`] for more
//! control), which computes a Groebner basis using a variant of Buchberger's algorithm.
//! Given a Groebner basis, [`multivariate_division()`] computes normal forms, and thus
//! can be used to decide ideal membership.
//!
//! # Base rings
//!
//! The algorithms support polynomial rings over fields, and more generally over local
//! principal ideal rings, i.e. principal ideal rings whose ideals are totally ordered
//! by inclusion. Apart from fields, this includes discrete valuation rings and rings
//! like `Z/p^eZ`. Over such rings, all computed Groebner bases are strong Groebner bases,
//! i.e. the leading term of every element of the ideal is divisible by the leading term
//! of some basis element.
//!
//! Over non-local rings (e.g. `Z` or `Z/6Z`), strong Groebner bases would require
//! additional "G-polynomials", which are currently not implemented. For `Z/nZ` with
//! composite `n`, callers should instead decompose the ring into its local components
//! `Z/p^eZ` using the Chinese Remainder Theorem, and compute a Groebner basis over each
//! component.

use crate::prelude::*;
use crate::ring_impls::multivariate::*;
use crate::ring_properties::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::ring_properties::pid::{PrincipalIdealRing, PrincipalIdealRingStore};

/// Contains Buchberger's algorithm for computing Groebner basis.
pub mod buchberger;
/// Contains critical pairs, the Gebauer-Moeller criteria and the basis data structure
/// that manages both.
mod pairs;
/// Contains multivariate polynomial division, i.e. reduction of polynomials modulo a
/// set of polynomials.
mod reduction;

pub use buchberger::buchberger;
#[allow(unused_imports)]
#[stability::unstable(feature = "enable")]
pub use buchberger::buchberger_with_strategy;
#[allow(unused_imports)]
#[stability::unstable(feature = "enable")]
pub use reduction::multivariate_division;

// origin (w.r.t. `algorithms/buchberger.rs`): new
/// A term `c * m` of a multivariate polynomial, where `c` is a coefficient and `m` a
/// monomial, stored together with the exponent vector of `m`.
///
/// Caching the exponent vector makes monomial divisibility checks cheap, which is the
/// most frequent operation during the search for reducers.
pub(super) struct CachedTerm<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    pub coeff: PolyCoeff<P>,
    pub monomial: PolyMonomial<P>,
    pub exponents: Box<[usize]>,
}

impl<P> CachedTerm<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): new
    pub fn new(ring: P, coeff: PolyCoeff<P>, monomial: PolyMonomial<P>) -> Self {
        let exponents = ring.expand_monomial(&monomial).into_boxed_slice();
        Self {
            coeff,
            monomial,
            exponents,
        }
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns the leading term of `f`, which must be nonzero.
    pub fn leading_term_of<O>(ring: P, f: &El<P>, order: O) -> Self
    where
        O: MonomialOrder,
    {
        let (c, m) = ring.LT(f, order).expect("leading term of zero polynomial");
        let (c, m) = (c.clone(), ring.clone_monomial(m));
        Self::new(ring, c, m)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    pub fn deg(&self) -> usize { self.exponents.iter().sum() }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    pub fn monomial_divides(&self, rhs_exponents: &[usize]) -> bool {
        self.exponents.iter().zip(rhs_exponents.iter()).all(|(l, r)| l <= r)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Checks whether this term divides the given term, i.e. whether both the monomial
    /// and the coefficient divide the respective counterpart.
    pub fn divides(&self, ring: P, rhs: &Self) -> bool
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        self.monomial_divides(&rhs.exponents) && ring.base_ring().divides(&rhs.coeff, &self.coeff)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Checks whether this term and the given term divide each other, i.e. are equal up
    /// to multiplication by a unit.
    pub fn is_associated(&self, ring: P, rhs: &Self) -> bool
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        self.exponents == rhs.exponents
            && ring.base_ring().divides(&rhs.coeff, &self.coeff)
            && ring.base_ring().divides(&self.coeff, &rhs.coeff)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Checks whether the monomials of this term and the given term are coprime.
    pub fn monomial_coprime(&self, rhs: &Self) -> bool {
        self.exponents
            .iter()
            .zip(rhs.exponents.iter())
            .all(|(l, r)| *l == 0 || *r == 0)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `term_lcm`
    /// Computes the least common multiple of both terms.
    pub fn lcm(&self, ring: P, rhs: &Self) -> Self
    where
        P: Copy,
        <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    {
        let exponents: Box<[usize]> = self
            .exponents
            .iter()
            .zip(rhs.exponents.iter())
            .map(|(l, r)| *l.max(r))
            .collect();
        Self {
            coeff: ring.base_ring().ideal_intersect(&self.coeff, &rhs.coeff),
            monomial: ring.create_monomial(exponents.iter().copied()),
            exponents,
        }
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `term_xlcm`
    /// Returns the term `q` such that `q * self = rhs`, assuming it exists.
    ///
    /// If the base ring has zero-divisors, `q` is not unique; in this case, any one of
    /// the possible values is returned.
    pub fn quotient_of(&self, ring: P, rhs: &Self) -> (PolyCoeff<P>, PolyMonomial<P>)
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        (
            ring.base_ring().checked_div(&rhs.coeff, &self.coeff).unwrap(),
            ring.create_monomial(self.exponents.iter().zip(rhs.exponents.iter()).map(|(l, r)| r - l)),
        )
    }
}

// origin (w.r.t. `algorithms/buchberger.rs`): new: replaces the tuples `(El<P>, ExpandedMonomial)`
/// A polynomial, together with its (cached) leading term and its sugar degree.
///
/// The sugar degree is used to decide in which order critical pairs are processed.
/// It is the degree the polynomial would have if we had homogenized the input
/// system and performed the same computation.
pub(super) struct BasisPoly<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    pub poly: El<P>,
    pub lt: CachedTerm<P>,
    pub sugar: usize,
}

impl<P> BasisPoly<P>
where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `expand_lm`
    /// Creates a new [`BasisPoly`], computing the leading term of `poly`, which must be
    /// nonzero.
    pub fn new<O>(ring: P, poly: El<P>, sugar: usize, order: O) -> Self
    where
        O: MonomialOrder,
    {
        Self {
            lt: CachedTerm::leading_term_of(ring, &poly, order),
            poly,
            sugar,
        }
    }
}

// origin (w.r.t. `algorithms/buchberger.rs`): new
/// Returns the total degree of `f`, i.e. the maximal degree of any of its monomials.
fn total_degree<P>(ring: P, f: &El<P>) -> usize
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    ring.terms(f).map(|(_, m)| ring.monomial_deg(m)).max().unwrap_or(0)
}

// origin (w.r.t. `algorithms/buchberger.rs`): new
/// A read-only view of the current state of a Groebner basis computation.
///
/// This is passed to user-provided callbacks, e.g. to decide whether to abort a
/// computation early.
#[stability::unstable(feature = "enable")]
pub struct BasisView<'a, P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    basis: &'a pairs::Basis<P>,
}

impl<'a, P> BasisView<'a, P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns an iterator over the current basis polynomials.
    ///
    /// These generate the same ideal as the input, and their leading terms are pairwise
    /// not divisible by each other. However, unless the computation is finished, they
    /// do not have to form a Groebner basis yet.
    #[stability::unstable(feature = "enable")]
    pub fn polys(&self) -> impl Iterator<Item = &'a El<P>> + use<'a, P> { self.basis.active().map(|(_, f)| &f.poly) }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns the leading monomials of the current basis polynomials, in the same order
    /// as [`BasisView::polys()`].
    #[stability::unstable(feature = "enable")]
    pub fn leading_monomials(&self) -> impl Iterator<Item = &'a PolyMonomial<P>> + use<'a, P> {
        self.basis.active().map(|(_, f)| &f.lt.monomial)
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns the number of current basis polynomials.
    #[stability::unstable(feature = "enable")]
    pub fn len(&self) -> usize { self.basis.active().count() }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns the number of critical pairs that are still open.
    #[stability::unstable(feature = "enable")]
    pub fn open_pair_count(&self) -> usize { self.basis.open_pair_count() }
}
