use std::fmt::Debug;

use tracing::instrument;

use super::{BasisPoly, CachedTerm};
use crate::homomorphism::Homomorphism;
use crate::prelude::*;
use crate::ring_impls::multivariate::*;
use crate::ring_properties::divisibility::DivisibilityRingStore;
use crate::ring_properties::pid::{PrincipalIdealRing, PrincipalIdealRingStore};

// origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `SPoly`; standard pairs store
// their lcm, annihilator pairs don't store the annihilated monomial
/// The different kinds of critical pairs; see [`CriticalPair`].
pub(super) enum CriticalPairKind<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    /// The standard S-polynomial `S(i, j)` of the basis polynomials `f_i` and `f_j`, i.e.
    /// `a * m * f_i - b * n * f_j` where `a * m * LT(f_i) = b * n * LT(f_j) = lcm`.
    Standard { i: usize, j: usize, lcm: CachedTerm<P> },
    /// The polynomial `annihilator * f_i`, where `annihilator` generates the annihilator
    /// of the leading coefficient of `f_i`.
    Annihilator { i: usize, annihilator: PolyCoeff<P> },
}

// origin (w.r.t. `algorithms/buchberger.rs`): new
/// A critical pair, i.e. a polynomial in the ideal whose leading term is cancelled by
/// construction, and which must be reduced to zero by a Groebner basis.
///
/// Over a local principal ideal ring, a basis is a strong Groebner basis if and only if all
/// standard S-polynomials and all annihilator polynomials reduce to zero.
pub(super) struct CriticalPair<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    pub kind: CriticalPairKind<P>,
    /// The sugar degree of the pair polynomial
    pub sugar: usize,
    /// The degree of the lcm of the leading monomials, for standard S-polynomials, and
    /// the degree of the leading monomial of `annihilator * f_i`, for annihilator polynomials
    pub deg: usize,
}

impl<P> Debug for CriticalPair<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `impl Debug for SPoly`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            CriticalPairKind::Standard { i, j, .. } => write!(f, "S({}, {})", i, j),
            CriticalPairKind::Annihilator { i, .. } => write!(f, "A({})", i),
        }
    }
}

impl<P> CriticalPair<P>
where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: from `SPoly::poly`
    /// Computes the polynomial represented by this critical pair.
    #[instrument(skip_all, level = "trace")]
    pub fn poly(&self, ring: P, basis: &Basis<P>) -> El<P> {
        match &self.kind {
            CriticalPairKind::Standard { i, j, lcm } => {
                let scaled = |f: &BasisPoly<P>| {
                    let (quo_c, quo_m) = f.lt.quotient_of(ring, lcm);
                    let mut result = f.poly.clone();
                    ring.mul_assign_monomial(&mut result, quo_m);
                    ring.inclusion().mul_assign_map(&mut result, quo_c);
                    result
                };
                ring.sub(scaled(basis.element(*i)), scaled(basis.element(*j)))
            }
            CriticalPairKind::Annihilator { i, annihilator } => {
                let mut result = basis.element(*i).poly.clone();
                ring.inclusion().mul_assign_ref_map(&mut result, annihilator);
                result
            }
        }
    }
}

// origin (w.r.t. `algorithms/buchberger.rs`): new
/// The current state of a Groebner basis computation, consisting of the basis polynomials
/// generated so far, and the critical pairs that still have to be considered.
///
/// Basis polynomials are only ever appended, so indices into the basis (as used by critical
/// pairs) remain valid. When a new polynomial is added whose leading term divides the leading
/// term of an existing polynomial, the latter is marked as redundant. Redundant polynomials
/// are not used to form new critical pairs, and are not part of the final result, but
/// critical pairs that have been formed with them before are still processed.
///
/// New critical pairs are filtered using the Gebauer-Moeller criteria, adapted to local
/// principal ideal rings.
pub(super) struct Basis<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    elements: Vec<BasisPoly<P>>,
    redundant: Vec<bool>,
    open: Vec<CriticalPair<P>>,
    filtered_pair_count: usize,
}

impl<P> Basis<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    // origin (w.r.t. `algorithms/buchberger.rs`): new
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            redundant: Vec::new(),
            open: Vec::new(),
            filtered_pair_count: 0,
        }
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    pub fn element(&self, i: usize) -> &BasisPoly<P> { &self.elements[i] }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Iterates over all non-redundant basis polynomials, together with their index.
    pub fn active(&self) -> impl Iterator<Item = (usize, &BasisPoly<P>)> {
        self.elements.iter().enumerate().filter(|(i, _)| !self.redundant[*i])
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Returns all non-redundant basis polynomials, in the order they were added.
    pub fn into_active(self) -> Vec<BasisPoly<P>> {
        let redundant = self.redundant;
        self.elements
            .into_iter()
            .zip(redundant)
            .filter(|(_, redundant)| !redundant)
            .map(|(f, _)| f)
            .collect()
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// The number of critical pairs that are still open.
    pub fn open_pair_count(&self) -> usize { self.open.len() }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// The number of critical pairs that have been discarded without reduction, since some
    /// criterion showed that they are unnecessary.
    pub fn filtered_pair_count(&self) -> usize { self.filtered_pair_count }

    // origin (w.r.t. `algorithms/buchberger.rs`): new
    /// Gives access to the list of open critical pairs, e.g. to remove pairs that should be
    /// reduced next.
    pub fn open_pairs_mut(&mut self) -> &mut Vec<CriticalPair<P>> { &mut self.open }

    // origin (w.r.t. `algorithms/buchberger.rs`): major changes: from `update_basis` and
    // `filter_spoly`; Gebauer-Moeller criteria, one annihilator polynomial per element, relaxed product
    // criterion, redundancy flags
    /// Adds `h` to the basis, and updates the set of open critical pairs.
    ///
    /// The leading term of `h` must not be divisible by the leading term of any
    /// non-redundant basis polynomial.
    ///
    /// This follows the procedure "Update" of Becker and Weispfenning ("Groebner bases",
    /// 1993), which implements the Gebauer-Moeller criteria:
    ///  - Old pairs `S(i, j)` are removed if `LT(h)` divides `lcm(i, j)`, and `lcm(i, j)` is
    ///    different from both `lcm(i, h)` and `lcm(j, h)`. They then follow from `S(i, h)` and
    ///    `S(j, h)`.
    ///  - New pairs `S(i, h)` are removed if there is another new pair `S(j, h)` whose lcm divides
    ///    `lcm(i, h)`. If the lcms are equal, exactly one of them is kept.
    ///  - New pairs `S(i, h)` are removed if the leading monomials of `f_i` and `h` are coprime,
    ///    and their leading coefficients are comaximal (product criterion). If this is the case,
    ///    all pairs with the same lcm are removed as well.
    ///  - New pairs `S(i, h)` are removed if the lcm of the leading coefficients is zero, since
    ///    they then are a combination of annihilator polynomials. (Note that this cannot happen
    ///    over local rings, but is cheap to check.)
    ///
    /// Over a ring with zero-divisors, we also add the annihilator polynomial `a * h`, where `a`
    /// generates the annihilator of `LC(h)`. We only need this one, since every other element of
    /// `Ann(LC(h))` is a multiple of `a`.
    #[instrument(skip_all, level = "trace")]
    pub fn insert<O>(&mut self, ring: P, h: BasisPoly<P>, order: O)
    where
        P: Copy,
        <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
        O: MonomialOrder + Copy,
    {
        let h_index = self.elements.len();
        debug_assert!(self.active().all(|(_, f)| !f.lt.divides(ring, &h.lt)));

        // we compute the lcms also with redundant polynomials, as they are needed to apply the
        // criterion to old pairs
        let mut lcms = self
            .elements
            .iter()
            .map(|f| Some(f.lt.lcm(ring, &h.lt)))
            .collect::<Vec<_>>();
        let lcm = |i: usize| lcms[i].as_ref().unwrap();

        // criterion for old pairs
        let open_count_before = self.open.len();
        self.open.retain(|pair| match &pair.kind {
            CriticalPairKind::Standard { i, j, lcm: lcm_ij } => {
                !h.lt.divides(ring, lcm_ij)
                    || lcm(*i).is_associated(ring, lcm_ij)
                    || lcm(*j).is_associated(ring, lcm_ij)
            }
            CriticalPairKind::Annihilator { .. } => true,
        });
        self.filtered_pair_count += open_count_before - self.open.len();

        // criteria for new pairs
        let candidates = self
            .active()
            .map(|(i, _)| i)
            .filter(|i| !ring.base_ring().is_zero(&lcm(*i).coeff))
            .collect::<Vec<_>>();
        let is_coprime = |i: usize| {
            let f_lt = &self.elements[i].lt;
            f_lt.monomial_coprime(&h.lt)
                && ring
                    .base_ring()
                    .is_unit(&ring.base_ring().ideal_gen(&f_lt.coeff, &h.lt.coeff))
        };
        let mut kept: Vec<(usize, bool)> = Vec::new();
        for (k, i) in candidates.iter().enumerate() {
            let coprime = is_coprime(*i);
            if coprime
                || !candidates[(k + 1)..]
                    .iter()
                    .chain(kept.iter().map(|(j, _)| j))
                    .any(|j| lcm(*j).divides(ring, lcm(*i)))
            {
                kept.push((*i, coprime));
            }
        }
        self.filtered_pair_count += self.active().count() - kept.len();

        for (i, coprime) in kept {
            if coprime {
                self.filtered_pair_count += 1;
                continue;
            }
            let f = &self.elements[i];
            let lcm = lcms[i].take().unwrap();
            let deg = lcm.deg();
            let sugar = (f.sugar + deg - f.lt.deg()).max(h.sugar + deg - h.lt.deg());
            self.open.push(CriticalPair {
                kind: CriticalPairKind::Standard { i, j: h_index, lcm },
                sugar,
                deg,
            });
        }

        // annihilator polynomial
        if !ring.base_ring().is_unit(&h.lt.coeff) {
            let annihilator = ring.base_ring().annihilator(&h.lt.coeff);
            let leading_monomial = ring
                .terms(&h.poly)
                .filter(|(c, _)| !ring.base_ring().is_zero(&ring.base_ring().mul_ref(c, &annihilator)))
                .map(|(_, m)| m)
                .max_by(|l, r| order.compare(ring, l, r));
            if let Some(m) = leading_monomial {
                self.open.push(CriticalPair {
                    deg: ring.monomial_deg(m),
                    kind: CriticalPairKind::Annihilator {
                        i: h_index,
                        annihilator,
                    },
                    sugar: h.sugar,
                });
            }
        }

        for i in 0..self.elements.len() {
            if !self.redundant[i] && h.lt.divides(ring, &self.elements[i].lt) {
                self.redundant[i] = true;
            }
        }
        self.elements.push(h);
        self.redundant.push(false);
    }
}
