use std::fmt::Debug;

use tracing::instrument;

use crate::homomorphism::Homomorphism;
use crate::prelude::*;
use crate::ring_impls::multivariate::*;
use crate::ring_properties::divisibility::DivisibilityRingStore;
use crate::ring_properties::pid::{PrincipalIdealRing, PrincipalIdealRingStore};

#[stability::unstable(feature = "enable")]
pub struct ExpandedTerm<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    pub coeff: PolyCoeff<P>,
    pub monomial: PolyMonomial<P>,
    pub exponents: Box<[usize]>,
}

impl<P> ExpandedTerm<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    #[stability::unstable(feature = "enable")]
    pub fn new(ring: P, coeff: PolyCoeff<P>, monomial: PolyMonomial<P>) -> Self {
        let exponents = ring.expand_monomial(&monomial).into_boxed_slice();
        Self {
            coeff,
            monomial,
            exponents,
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn lt<O>(ring: P, f: &El<P>, order: O) -> Self
    where
        O: MonomialOrder,
    {
        let (coeff, monomial) = ring.LT(f, order).expect("leading term of zero polynomial");
        let exponents = ring.expand_monomial(&monomial).into_boxed_slice();
        Self {
            coeff: coeff.clone(),
            monomial: monomial.clone(),
            exponents,
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn deg(&self) -> usize { self.exponents.iter().sum() }

    #[stability::unstable(feature = "enable")]
    pub fn divides(ring: P, lhs: &Self, rhs: &Self) -> bool
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        lhs.exponents.iter().zip(&rhs.exponents).all(|(l, r)| l >= r)
            && ring.base_ring().divides(&lhs.coeff, &rhs.coeff)
    }

    #[stability::unstable(feature = "enable")]
    pub fn checked_div(ring: P, lhs: &Self, rhs: &Self) -> Option<Self>
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        let coeff = ring.base_ring().checked_div(&lhs.coeff, &rhs.coeff)?;
        let mut fail = false;
        let mon = ring.create_monomial(lhs.exponents.iter().zip(&rhs.exponents).map(|(l, r)| {
            fail |= l < r;
            l.saturating_sub(*r)
        }));
        if fail {
            return None;
        } else {
            return Some(Self::new(ring, coeff, mon));
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn is_associated(ring: P, lhs: &Self, rhs: &Self) -> bool
    where
        <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing,
    {
        lhs.exponents == rhs.exponents
            && ring.base_ring().divides(&rhs.coeff, &lhs.coeff)
            && ring.base_ring().divides(&lhs.coeff, &rhs.coeff)
    }

    #[stability::unstable(feature = "enable")]
    pub fn lcm(ring: P, lhs: &Self, rhs: &Self) -> Self
    where
        P: Copy,
        <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    {
        let exponents: Box<[usize]> = lhs
            .exponents
            .iter()
            .zip(rhs.exponents.iter())
            .map(|(l, r)| *l.max(r))
            .collect();
        Self {
            coeff: ring.base_ring().ideal_intersect(&lhs.coeff, &rhs.coeff),
            monomial: ring.create_monomial(exponents.iter().copied()),
            exponents,
        }
    }
}

#[stability::unstable(feature = "enable")]
pub enum CriticalPairKind<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    /// The standard S-polynomial `S(i, j)` of the basis polynomials `f_i` and `f_j`, i.e.
    /// `a * m * f_i - b * n * f_j` where `a * m * LT(f_i) = b * n * LT(f_j) = lcm`.
    Standard { i: usize, j: usize, lcm: ExpandedTerm<P> },
    /// The polynomial `annihilator * f_i`, where `annihilator` generates the annihilator
    /// of the leading coefficient of `f_i`.
    Annihilator { i: usize, annihilator: PolyCoeff<P> },
}

#[stability::unstable(feature = "enable")]
pub struct CriticalPair<P>
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
    #[stability::unstable(feature = "enable")]
    #[instrument(skip_all, level = "trace")]
    pub fn poly(&self, ring: P, basis: &[BasisPoly<P>]) -> El<P> {
        match &self.kind {
            CriticalPairKind::Standard { i, j, lcm } => {
                let scaled = |f: &BasisPoly<P>| {
                    let quo = ExpandedTerm::checked_div(ring, lcm, &f.lt).unwrap();
                    let mut result = f.poly.clone();
                    ring.mul_assign_monomial(&mut result, quo.monomial);
                    ring.inclusion().mul_assign_map(&mut result, quo.coeff);
                    result
                };
                ring.sub(scaled(&basis[*i]), scaled(&basis[*j]))
            }
            CriticalPairKind::Annihilator { i, annihilator } => {
                let mut result = basis[*i].poly.clone();
                ring.inclusion().mul_assign_ref_map(&mut result, annihilator);
                result
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub struct BasisPoly<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    pub poly: El<P>,
    pub lt: ExpandedTerm<P>,
    pub sugar_degree: usize,
    pub redundant: bool,
}
