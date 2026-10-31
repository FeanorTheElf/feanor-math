use elsa::sync::FrozenVec;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tracing::{Level, Span, event, instrument, span};

use super::reduction::reduce;
use super::spoly::CriticalPair;
use crate::algorithms::groebner::spoly::{BasisPoly, CriticalPairKind, ExpandedTerm};
use crate::prelude::*;
use crate::ring_impls::multivariate::*;
use crate::ring_properties::pid::PrincipalIdealRing;

#[stability::unstable(feature = "enable")]
pub trait BuchbergerStrategy<P>
where
    P: RingStore,
    P::Ring: MultivariatePolyRing,
{
    /// Removes and returns the critical pairs that should be reduced next.
    ///
    /// The default implementation uses the sugar strategy, i.e. we select all pairs of minimal
    /// sugar degree. For homogeneous input, this is the same as selecting all pairs of minimal
    /// degree.
    fn select_batch(
        &mut self,
        _ring: P,
        _basis: &[BasisPoly<P>],
        open: &mut Vec<CriticalPair<P>>,
    ) -> Vec<CriticalPair<P>> {
        let Some(min_sugar) = open.iter().map(|pair| pair.sugar).min() else {
            return Vec::new();
        };
        let (mut batch, rest): (Vec<_>, Vec<_>) = std::mem::take(open)
            .into_iter()
            .partition(|pair| pair.sugar == min_sugar);
        *open = rest;
        batch.sort_by_key(|pair| pair.deg);
        return batch;
    }

    /// Is regularly called by [`buchberger()`] and [`buchberger_with_strategy()`]; if it returns
    /// true, the Groebner basis computation is stopped and current basis is returned.
    ///
    /// The default implementation always returns `false`.
    fn should_abort(&mut self, _ring: P, _basis: &[BasisPoly<P>], _open: &[CriticalPair<P>]) -> bool { false }
}

/// Adds `h` to the basis, and updates the set of open critical pairs.
///
/// The leading term of `h` must not be divisible by the leading term of any
/// non-redundant basis polynomial.
///
/// This follows the procedure "Update" of Becker and Weispfenning ("Groebner bases",
/// 1993), which implements the Gebauer-Moeller criteria:
///  - Old pairs `S(i, j)` are removed if `LT(h)` divides `lcm(i, j)`, and `lcm(i, j)` is different
///    from both `lcm(i, h)` and `lcm(j, h)`. They then follow from `S(i, h)` and `S(j, h)`.
///  - New pairs `S(i, h)` are removed if there is another new pair `S(j, h)` whose lcm divides
///    `lcm(i, h)`. If the lcms are equal, exactly one of them is kept.
///  - New pairs `S(i, h)` are removed if the leading monomials of `f_i` and `h` are coprime, and
///    their leading coefficients are comaximal (product criterion). If this is the case, all pairs
///    with the same lcm are removed as well.
///  - New pairs `S(i, h)` are removed if the lcm of the leading coefficients is zero, since they
///    then are a combination of annihilator polynomials. (Note that this cannot happen over local
///    rings, but is cheap to check.)
///
/// Over a ring with zero-divisors, we also add the annihilator polynomial `a * h`, where `a`
/// generates the annihilator of `LC(h)`. We only need this one, since every other element of
/// `Ann(LC(h))` is a multiple of `a`.
#[instrument(skip_all, level = "trace")]
fn insert_and_update_basis<P, O>(
    ring: P,
    h: BasisPoly<P>,
    basis: &mut Vec<BasisPoly<P>>,
    open: &mut Vec<CriticalPair<P>>,
    filtered_pair_count: &mut usize,
    order: O,
) where
    P: Copy + RingStore,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    O: MonomialOrder + Copy,
{
    assert!(!h.redundant);
    let h_index = basis.len();
    debug_assert!(
        basis
            .iter()
            .all(|b| b.redundant || !ExpandedTerm::divides(ring, &h.lt, &b.lt))
    );

    // we compute the lcms also with redundant polynomials, as they are needed to apply the
    // criterion to old pairs
    let mut lcms = basis
        .iter()
        .map(|b| Some(ExpandedTerm::lcm(ring, &b.lt, &h.lt)))
        .collect::<Vec<_>>();
    let lcm = |i: usize| lcms[i].as_ref().unwrap();

    // criterion for old pairs
    let open_count_before = open.len();
    open.retain(|pair| match &pair.kind {
        CriticalPairKind::Standard { i, j, lcm: lcm_ij } => {
            !ExpandedTerm::divides(ring, lcm_ij, &h.lt)
                || ExpandedTerm::is_associated(ring, lcm_ij, lcm(*i))
                || ExpandedTerm::is_associated(ring, lcm_ij, lcm(*j))
        }
        CriticalPairKind::Annihilator { .. } => true,
    });
    *filtered_pair_count += open_count_before - open.len();

    // criteria for new pairs
    let candidates = basis
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.redundant)
        .map(|(i, _)| i)
        .filter(|i| !ring.base_ring().is_zero(&lcm(*i).coeff))
        .collect::<Vec<_>>();
    let is_coprime = |i: usize| {
        basis[i]
            .lt
            .exponents
            .iter()
            .zip(&h.lt.exponents)
            .all(|(l, r)| *l == 0 || *r == 0)
            && ring
                .base_ring()
                .is_unit(&ring.base_ring().ideal_gen(&basis[i].lt.coeff, &h.lt.coeff))
    };
    let mut kept: Vec<(usize, bool)> = Vec::new();
    for (k, i) in candidates.iter().enumerate() {
        let coprime = is_coprime(*i);
        if coprime
            || !candidates[(k + 1)..]
                .iter()
                .chain(kept.iter().map(|(j, _)| j))
                .any(|j| ExpandedTerm::divides(ring, lcm(*i), lcm(*j)))
        {
            kept.push((*i, coprime));
        }
    }
    *filtered_pair_count += basis.iter().filter(|b| !b.redundant).count() - kept.len();

    for (i, coprime) in kept {
        if coprime {
            *filtered_pair_count += 1;
            continue;
        }
        let f = &basis[i];
        let lcm = lcms[i].take().unwrap();
        let deg = lcm.deg();
        let sugar = (f.sugar_degree + deg - f.lt.deg()).max(h.sugar_degree + deg - h.lt.deg());
        open.push(CriticalPair {
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
            open.push(CriticalPair {
                deg: ring.monomial_deg(m),
                kind: CriticalPairKind::Annihilator {
                    i: h_index,
                    annihilator,
                },
                sugar: h.sugar_degree,
            });
        }
    }
    for i in 0..basis.len() {
        if !basis[i].redundant && ExpandedTerm::divides(ring, &basis[i].lt, &h.lt) {
            basis[i].redundant = true;
        }
    }
    basis.push(h);
}

/// Reduces `f` modulo the current basis and, if the result is nonzero, adds it to the basis.
fn reduce_and_insert<P, O>(
    ring: P,
    mut h: El<P>,
    mut sugar_degree: usize,
    basis: &mut Vec<BasisPoly<P>>,
    open: &mut Vec<CriticalPair<P>>,
    filtered_pair_count: &mut usize,
    order: O,
) where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    O: MonomialOrder + Copy,
{
    reduce(
        ring,
        &mut h,
        &mut sugar_degree,
        || basis.iter().filter(|b| !b.redundant),
        order,
        false,
    );
    if !ring.is_zero(&h) {
        insert_and_update_basis(
            ring,
            BasisPoly {
                lt: ExpandedTerm::lt(ring, &h, order),
                poly: h,
                redundant: false,
                sugar_degree,
            },
            basis,
            open,
            filtered_pair_count,
            order,
        );
    }
}

/// Reduces the tails of the given polynomials, whose leading terms must be pairwise not
/// divisible by each other. This does not change the leading terms.
#[instrument(skip_all, level = "trace")]
fn tail_reduce<P, O>(ring: P, basis: Vec<BasisPoly<P>>, order: O) -> Vec<El<P>>
where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    O: MonomialOrder + Copy + Send + Sync,
{
    (0..basis.len())
        .into_par_iter()
        .map(|i| {
            let mut f = basis[i].poly.clone();
            reduce(
                ring,
                &mut f,
                &mut 0,
                || basis[..i].iter().chain(basis[(i + 1)..].iter()),
                order,
                true,
            );
            f
        })
        .collect()
}

/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given monomial
/// order.
///
/// The base ring must be local (i.e. its ideals must be totally ordered by inclusion), as
/// reported by [`PrincipalIdealRing::is_local()`]. This includes fields, discrete valuation rings
/// and rings like `Z/p^eZ`. Over such rings, the result is a strong Groebner basis. For `Z/nZ`
/// with composite `n`, decompose the ring into its components `Z/p^eZ` using the Chinese Remainder
/// Theorem, and compute Groebner bases over each of them.
///
/// The returned basis is reduced, in the sense that the leading term of no basis polynomial
/// divides any term of another basis polynomial. However, the coefficients are not normalized,
/// i.e. leading coefficients may be any unit (or, over rings that are not fields, any element).
///
/// See also [`buchberger()`], which runs the algorithm with the default strategy.
///
/// # Panics
///
/// Panics if the base ring is not local.
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn buchberger_with_strategy<P, O, S>(ring: P, input_basis: Vec<El<P>>, order: O, mut strategy: S) -> Vec<El<P>>
where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    O: MonomialOrder + Copy + Send + Sync,
    S: BuchbergerStrategy<P>,
{
    assert!(
        ring.base_ring().get_ring().is_local(),
        "Groebner basis computations are only supported over local rings"
    );
    event!(Level::TRACE, var_count = ring.indeterminate_count());

    let mut input_basis = input_basis
        .into_iter()
        .filter(|f| !ring.is_zero(f))
        .map(|f| {
            let sugar_degree = ring.terms(&f).map(|(_, m)| ring.monomial_deg(m)).max().unwrap();
            BasisPoly {
                lt: ExpandedTerm::lt(ring, &f, order),
                poly: f,
                redundant: false,
                sugar_degree,
            }
        })
        .collect::<Vec<_>>();
    input_basis.sort_by(|l, r| order.compare(ring, &l.lt.monomial, &r.lt.monomial));

    let mut basis = Vec::new();
    let mut open = Vec::new();
    let mut filtered_pair_count = 0;
    for f in input_basis {
        reduce_and_insert(
            ring,
            f.poly,
            f.sugar_degree,
            &mut basis,
            &mut open,
            &mut filtered_pair_count,
            order,
        );
    }

    loop {
        event!(
            Level::TRACE,
            basis_poly_count = basis.iter().filter(|b| !b.redundant).count(),
            open_pair_count = open.len(),
            filtered_pair_count = filtered_pair_count
        );

        let batch = strategy.select_batch(ring, &basis, &mut open);
        if batch.is_empty() {
            break;
        }
        event!(Level::TRACE, batch_size = batch.len(), sugar = batch[0].sugar);

        // we sort reducers, as that will speed up the multivariate division
        let mut reducers = basis.iter().filter(|b| !b.redundant).collect::<Vec<_>>();
        reducers.sort_by(|l, r| {
            order
                .compare(ring, &l.lt.monomial, &r.lt.monomial)
                .then_with(|| ring.terms(&l.poly).count().cmp(&ring.terms(&r.poly).count()))
        });

        // in effect, this is the same as the matrix reduction step during F4; new polynomials
        // can be used for reduction immediately, even by other threads; we also reduce tails
        // here, since (at least for DegRevLex) this makes later reductions much cheaper
        let new_polys: FrozenVec<Box<BasisPoly<P>>> = FrozenVec::new();
        let outer_span = Span::current();
        batch.par_iter().panic_fuse().for_each(|pair| {
            span!(parent: &outer_span, Level::TRACE, "reduce_pair").in_scope(|| {
                let mut f = pair.poly(ring, &basis);
                let mut sugar = pair.sugar;
                reduce(
                    ring,
                    &mut f,
                    &mut sugar,
                    || reducers.iter().copied().chain(new_polys.iter()),
                    order,
                    true,
                );
                if !ring.is_zero(&f) {
                    _ = new_polys.push(Box::new(BasisPoly {
                        lt: ExpandedTerm::lt(ring, &f, order),
                        poly: f,
                        sugar_degree: sugar,
                        redundant: false,
                    }));
                }
            })
        });
        drop(reducers);

        let mut new_polys = new_polys.into_vec();
        event!(
            Level::TRACE,
            reduced_to_zero = batch.len() - new_polys.len(),
            new_basis_polys = new_polys.len()
        );
        drop(batch);

        // since the new polynomials were reduced concurrently, they might still be reducible
        // by each other
        new_polys.sort_by(|l, r| order.compare(ring, &l.lt.monomial, &r.lt.monomial));
        for f in new_polys {
            reduce_and_insert(
                ring,
                f.poly,
                f.sugar_degree,
                &mut basis,
                &mut open,
                &mut filtered_pair_count,
                order,
            );
        }

        if strategy.should_abort(ring, &basis, &open) {
            event!(Level::TRACE, "early_abort");
            return basis.into_iter().filter(|b| !b.redundant).map(|b| b.poly).collect();
        }
    }

    return tail_reduce(ring, basis.into_iter().filter(|b| !b.redundant).collect(), order);
}

/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given monomial
/// order.
///
/// The returned basis is reduced, in the sense that the leading term of no basis polynomial
/// divides any term of another basis polynomial. However, the coefficients are not normalized,
/// i.e. basis polynomials are not necessarily monic.
///
/// The base ring must be local (i.e. its ideals must be totally ordered by inclusion), as
/// reported by [`PrincipalIdealRing::is_local()`]. This includes fields, discrete valuation rings
/// and rings like `Z/p^eZ`. Over such rings, the result is a strong Groebner basis. For `Z/nZ`
/// with composite `n`, decompose the ring into its components `Z/p^eZ` using the Chinese Remainder
/// Theorem, and compute Groebner bases over each of them.
///
/// For a variant of this function that allows for more configuration, see
/// [`buchberger_with_strategy()`].
///
/// # Example
/// ```rust
/// # use feanor_math::prelude::*;
/// # use feanor_math::algorithms::groebner::*;
/// # use feanor_math::ring_impls::multivariate::*;
/// # use feanor_math::ring_impls::multivariate::multivariate_impl::*;
/// # use feanor_math::ring_impls::rational::*;
/// let QQ = RationalField::new(ZZbig);
/// let ring = MultivariatePolyRingImpl::new(&QQ, 2);
/// let [f, g] =
///     ring.with_wrapped_indeterminates(|[X, Y]| [X.pow_ref(2) + Y.pow_ref(2) - 1, X - Y]);
/// let gb = buchberger(&ring, vec![f, g], Lex);
/// // the ideal contains `2 Y^2 - 1`, which is a univariate polynomial in `Y`
/// assert!(
///     gb.iter()
///         .any(|f| ring.appearing_indeterminates(f) == vec![(1, 2)])
/// );
/// ```
///
/// # Panics
///
/// Panics if the base ring is not local.
pub fn buchberger<P, O>(ring: P, input_basis: Vec<El<P>>, order: O) -> Vec<El<P>>
where
    P: RingStore + Copy,
    P::Ring: MultivariatePolyRing,
    <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
    O: MonomialOrder + Copy + Send + Sync,
{
    struct DefaultStrategy;

    impl<P> BuchbergerStrategy<P> for DefaultStrategy
    where
        P: RingStore,
        P::Ring: MultivariatePolyRing,
    {
    }

    buchberger_with_strategy(ring, input_basis, order, DefaultStrategy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::groebner::multivariate_division;
    use crate::homomorphism::Homomorphism;
    use crate::ring_impls::multivariate::multivariate_impl::MultivariatePolyRingImpl;
    use crate::ring_impls::poly::{PolyRingStore, dense_poly};
    use crate::ring_impls::rational::RationalField;
    use crate::ring_impls::zn::zn_static;
    use crate::ring_properties::divisibility::DivisibilityRingStore;
    use crate::ring_properties::pid::PrincipalIdealRingStore;

    fn assert_is_strong_gb<P, O>(ring: P, gb: &[El<P>], order: O)
    where
        P: RingStore + Copy,
        P::Ring: MultivariatePolyRing,
        <BaseRingStore<P> as RingStore>::Ring: PrincipalIdealRing,
        O: MonomialOrder + Copy,
    {
        let scale = |f: &El<P>, c: PolyCoeff<P>, m: PolyMonomial<P>| {
            let mut result = f.clone();
            ring.mul_assign_monomial(&mut result, m);
            ring.inclusion().mul_assign_map(&mut result, c);
            result
        };
        for i in 0..gb.len() {
            let (c_i, m_i) = ring.LT(&gb[i], order).unwrap();
            let apoly = scale(
                &gb[i],
                ring.base_ring().annihilator(c_i),
                ring.create_monomial((0..ring.indeterminate_count()).map(|_| 0)),
            );
            assert!(ring.is_zero(&multivariate_division(ring, apoly, gb.iter(), order)));
            for j in (i + 1)..gb.len() {
                let (c_j, m_j) = ring.LT(&gb[j], order).unwrap();
                let lcm_c = ring.base_ring().ideal_intersect(c_i, c_j);
                let lcm_m = ring.monomial_lcm(m_i.clone(), m_j);
                let spoly = ring.sub(
                    scale(
                        &gb[i],
                        ring.base_ring().checked_div(&lcm_c, c_i).unwrap(),
                        ring.monomial_div(lcm_m.clone(), m_i).ok().unwrap(),
                    ),
                    scale(
                        &gb[j],
                        ring.base_ring().checked_div(&lcm_c, c_j).unwrap(),
                        ring.monomial_div(lcm_m.clone(), m_j).ok().unwrap(),
                    ),
                );
                assert!(ring.is_zero(&multivariate_division(ring, spoly, gb.iter(), order)));
            }
        }
    }

    #[test]
    fn test_gb_local_ring_many_annihilators() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Zn::<16>::RING;
        let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);
        let [f] = ring.with_wrapped_indeterminates(|[X]| [8 * X.pow_ref(3) + 4 * X.pow_ref(2) + 2 * X + 1]);
        let gb = buchberger(&ring, vec![f], DegRevLex);

        assert_eq!(1, gb.len());
        let (c, m) = ring.LT(&gb[0], DegRevLex).unwrap();
        assert_eq!(0, ring.monomial_deg(m));
        assert!(base.is_unit(c));
    }

    #[test]
    fn test_gb_local_ring_nontrivial() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Zn::<27>::RING;
        let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 2);
        let system = ring.with_wrapped_indeterminates(|[X, Y]| {
            [
                3 * X.pow_ref(2) * Y + 9 * X + 1 * Y.pow_ref(2),
                9 * X * Y.pow_ref(2) + 3 * Y + X,
                X.pow_ref(3) + 3 * X * Y + 9,
            ]
        });
        for order in [DegRevLex] {
            let gb = buchberger(&ring, system.iter().cloned().collect(), order);
            assert_is_strong_gb(&ring, &gb, order);
            for f in &system {
                assert!(ring.is_zero(&multivariate_division(&ring, f.clone(), gb.iter(), order)));
            }
        }
        let gb = buchberger(&ring, system.iter().cloned().collect(), Lex);
        assert_is_strong_gb(&ring, &gb, Lex);
        for f in &system {
            assert!(ring.is_zero(&multivariate_division(&ring, f.clone(), gb.iter(), Lex)));
        }
    }

    #[test]
    #[should_panic(expected = "only supported over local rings")]
    fn test_gb_non_local_ring_zn() {
        let base = zn_static::Zn::<6>::RING;
        let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);
        let [f, g] = ring.with_wrapped_indeterminates(|[X]| [2 * X, 3 * X]);
        _ = buchberger(&ring, vec![f, g], DegRevLex);
    }

    #[test]
    fn test_gb_abort_early() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<65537>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 4);
        let system = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t]| {
            [
                x + y + z + t,
                x * y + y * z + z * t + t * x,
                x * y * z + y * z * t + z * t * x + t * x * y,
                x * y * z * t - 1,
            ]
        });

        struct TestStrategy(usize);

        impl<'a, P> BuchbergerStrategy<P> for &'a mut TestStrategy
        where
            P: RingStore,
            P::Ring: MultivariatePolyRing,
        {
            fn should_abort(&mut self, _ring: P, basis: &[BasisPoly<P>], _open: &[CriticalPair<P>]) -> bool {
                self.0 += 1;
                assert!(basis.len() >= 4);
                true
            }
        }

        let mut strategy = TestStrategy(0);
        let gb = buchberger_with_strategy(&ring, system.clone(), DegRevLex, &mut strategy);
        assert_eq!(1, strategy.0);
        for f in &system {
            assert!(ring.is_zero(&multivariate_division(&ring, f.clone(), gb.iter(), DegRevLex)));
        }
    }

    #[test]
    fn test_buchberger_small() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::F17;
        let ring = MultivariatePolyRingImpl::new(base, 2);

        let f1 = ring.from_terms(
            [
                (1, ring.create_monomial([2, 0])),
                (1, ring.create_monomial([0, 2])),
                (16, ring.create_monomial([0, 0])),
            ]
            .into_iter(),
        );
        let f2 = ring.from_terms([(1, ring.create_monomial([1, 1])), (15, ring.create_monomial([0, 0]))].into_iter());

        let actual = buchberger(&ring, vec![f1.clone(), f2.clone()], DegRevLex);
        assert_is_strong_gb(&ring, &actual, DegRevLex);

        let expected = ring.from_terms(
            [
                (16, ring.create_monomial([0, 3])),
                (15, ring.create_monomial([1, 0])),
                (1, ring.create_monomial([0, 1])),
            ]
            .into_iter(),
        );

        assert_eq!(3, actual.len());
        assert_el_eq!(
            ring,
            ring.zero(),
            multivariate_division(&ring, f1, actual.iter(), DegRevLex)
        );
        assert_el_eq!(
            ring,
            ring.zero(),
            multivariate_division(&ring, f2, actual.iter(), DegRevLex)
        );
        assert_el_eq!(
            ring,
            ring.zero(),
            multivariate_division(&ring, expected, actual.iter(), DegRevLex)
        );
    }

    #[test]
    fn test_buchberger_larger() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::F17;
        let ring = MultivariatePolyRingImpl::new(base, 3);

        let f1 = ring.from_terms(
            [
                (1, ring.create_monomial([2, 1, 1])),
                (1, ring.create_monomial([0, 2, 0])),
                (1, ring.create_monomial([1, 0, 1])),
                (2, ring.create_monomial([1, 0, 0])),
                (1, ring.create_monomial([0, 0, 0])),
            ]
            .into_iter(),
        );
        let f2 = ring.from_terms(
            [
                (1, ring.create_monomial([0, 3, 1])),
                (1, ring.create_monomial([0, 0, 3])),
                (1, ring.create_monomial([1, 1, 0])),
            ]
            .into_iter(),
        );
        let f3 = ring.from_terms(
            [
                (1, ring.create_monomial([1, 0, 2])),
                (1, ring.create_monomial([1, 0, 1])),
                (2, ring.create_monomial([0, 1, 1])),
                (7, ring.create_monomial([0, 0, 0])),
            ]
            .into_iter(),
        );

        let actual = buchberger(&ring, vec![f1.clone(), f2.clone(), f3.clone()], DegRevLex);
        assert_is_strong_gb(&ring, &actual, DegRevLex);

        let g1 = ring.from_terms(
            [
                (1, ring.create_monomial([0, 4, 0])),
                (8, ring.create_monomial([0, 3, 1])),
                (12, ring.create_monomial([0, 1, 3])),
                (6, ring.create_monomial([0, 0, 4])),
                (1, ring.create_monomial([0, 3, 0])),
                (13, ring.create_monomial([0, 2, 1])),
                (11, ring.create_monomial([0, 1, 2])),
                (10, ring.create_monomial([0, 0, 3])),
                (11, ring.create_monomial([0, 2, 0])),
                (12, ring.create_monomial([0, 1, 1])),
                (6, ring.create_monomial([0, 0, 2])),
                (6, ring.create_monomial([0, 1, 0])),
                (13, ring.create_monomial([0, 0, 1])),
                (9, ring.create_monomial([0, 0, 0])),
            ]
            .into_iter(),
        );

        assert_el_eq!(
            ring,
            ring.zero(),
            multivariate_division(&ring, g1, actual.iter(), DegRevLex)
        );
    }

    #[test]
    #[ignore]
    fn test_generic_computation() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::F17;
        let ring = MultivariatePolyRingImpl::new(base, 6);
        let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

        let var_i = |i: usize| {
            ring.create_term(
                base.one(),
                ring.create_monomial((0..ring.indeterminate_count()).map(|j| if i == j { 1 } else { 0 })),
            )
        };
        let X1 = poly_ring.mul(
            poly_ring.from_terms([(var_i(0), 0), (ring.one(), 1)].into_iter()),
            poly_ring.from_terms([(var_i(1), 0), (ring.one(), 1)].into_iter()),
        );
        let X2 = poly_ring.mul(
            poly_ring.add(
                X1.clone(),
                poly_ring.from_terms([(var_i(2), 0), (var_i(3), 1)].into_iter()),
            ),
            poly_ring.add(
                X1.clone(),
                poly_ring.from_terms([(var_i(4), 0), (var_i(5), 1)].into_iter()),
            ),
        );
        let basis = vec![
            ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 0)),
            ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 1)),
            ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 2)),
        ];

        let gb1 = buchberger(&ring, basis.iter().map(|f| f.clone()).collect(), DegRevLex);
        assert_is_strong_gb(&ring, &gb1, DegRevLex);
        assert_eq!(11, gb1.len());
    }

    #[test]
    fn test_gb_local_ring() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Zn::<16>::RING;
        let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);

        let f = ring.from_terms(
            [
                (base.int_hom().map(4), ring.create_monomial([1])),
                (base.one(), ring.create_monomial([0])),
            ]
            .into_iter(),
        );
        let gb = buchberger(&ring, vec![f], DegRevLex);

        assert_eq!(1, gb.len());
        assert!(ring.is_one(&gb[0]));
    }

    #[test]
    fn test_gb_lex() {
        feanor_tracing::DelayedLogger::init_test();
        let ZZ = ZZbig;
        let QQ = RationalField::new(ZZ);
        let QQYX = MultivariatePolyRingImpl::new(&QQ, 2);
        let [f, g] = QQYX.with_wrapped_indeterminates(|[Y, X]| {
            [
                1 + X.pow_ref(2) + 2 * Y + (1 + X) * Y.pow_ref(2),
                3 + X + (2 + X) * Y + (1 + X + X.pow_ref(2)) * Y.pow_ref(2),
            ]
        });
        let expected = QQYX.with_wrapped_indeterminates(|[Y, X]| {
            [
                X.pow_ref(8) + 2 * X.pow_ref(7) + 3 * X.pow_ref(6)
                    - 5 * X.pow_ref(5)
                    - 10 * X.pow_ref(4)
                    - 7 * X.pow_ref(3)
                    + 8 * X.pow_ref(2)
                    + 8 * X
                    + 4,
                2 * Y + X.pow_ref(6) + 3 * X.pow_ref(5) + 6 * X.pow_ref(4) + X.pow_ref(3)
                    - 7 * X.pow_ref(2)
                    - 12 * X
                    - 2,
            ]
        });

        let mut gb = buchberger(&QQYX, vec![f, g], Lex);

        assert_eq!(2, gb.len());
        gb.sort_unstable_by_key(|f| QQYX.appearing_indeterminates(f).len());
        for (mut f, mut e) in gb.into_iter().zip(expected.into_iter()) {
            let f_lc_inv = QQ.invert(QQYX.LT(&f, Lex).unwrap().0).unwrap();
            QQYX.inclusion().mul_assign_map(&mut f, f_lc_inv);
            let e_lc_inv = QQ.invert(QQYX.LT(&e, Lex).unwrap().0).unwrap();
            QQYX.inclusion().mul_assign_map(&mut e, e_lc_inv);
            assert_el_eq!(QQYX, e, f);
        }
    }

    #[ignore]
    #[test]
    fn test_expensive_gb_1() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Zn::<16>::RING;
        let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 12);

        let system = ring.with_wrapped_indeterminates(|[Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11]| {
            [
                Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4.pow_ref(2)
                    + 4 * Y0 * Y1 * Y2 * Y3 * Y4 * Y4 * Y8
                    + Y0 * Y1 * Y2 * Y5.pow_ref(2) * Y8.pow_ref(2)
                    + Y0 * Y2 * Y3 * Y4 * Y6
                    + Y0 * Y1 * Y3 * Y4 * Y7
                    + Y0 * Y2 * Y5 * Y6 * Y8
                    + Y0 * Y1 * Y5 * Y7 * Y8
                    + Y0 * Y2 * Y3 * Y5 * Y10
                    + Y0 * Y1 * Y3 * Y5 * Y11
                    + Y0 * Y6 * Y7
                    + Y3 * Y5 * Y9
                    - 4,
                2 * Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4 * Y5
                    + 2 * Y0 * Y1 * Y2 * Y3 * Y5.pow_ref(2) * Y8
                    + Y0 * Y2 * Y3 * Y5 * Y6
                    + Y0 * Y1 * Y3 * Y5 * Y7
                    + 8,
                Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y5.pow_ref(2) - 5,
            ]
        });

        let part_of_result =
            ring.with_wrapped_indeterminates(|[_Y0, Y1, Y2, _Y3, _Y4, _Y5, Y6, Y7, _Y8, _Y9, _Y10, _Y11]| {
                [
                    4 * Y2.pow_ref(2) * Y6.pow_ref(2) - 4 * Y1.pow_ref(2) * Y7.pow_ref(2),
                    8 * Y2 * Y6 + 8 * Y1 * Y7.clone(),
                ]
            });

        let gb = buchberger(&ring, system.iter().map(|f| f.clone()).collect(), DegRevLex);

        for f in &part_of_result {
            assert!(ring.is_zero(&multivariate_division(&ring, f.clone(), gb.iter(), DegRevLex)));
        }
        assert_eq!(108, gb.len());
    }

    #[test]
    #[ignore]
    fn test_expensive_gb_2() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<7>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 7);

        let basis = ring.with_wrapped_indeterminates_dyn(|[X0, X1, X2, X3, X4, X5, X6]| {
            [
                6 + 2 * X5
                    + 2 * X4
                    + X6
                    + 4 * X0
                    + 5 * X6 * X5
                    + X6 * X4
                    + 3 * X0 * X4
                    + 6 * X0 * X6
                    + 2 * X0 * X3
                    + X0 * X2
                    + 4 * X0 * X1
                    + 2 * X3 * X4 * X5
                    + 4 * X0 * X6 * X5
                    + 6 * X0 * X2 * X5
                    + 5 * X0 * X6 * X4
                    + 2 * X0 * X3 * X4
                    + 4 * X0 * X1 * X4
                    + X0 * X6.pow_ref(2)
                    + 3 * X0 * X3 * X6
                    + 5 * X0 * X2 * X6
                    + 2 * X0 * X1 * X6
                    + X0 * X3.pow_ref(2)
                    + 2 * X0 * X2 * X3
                    + 3 * X0 * X3 * X4 * X5
                    + 4 * X0 * X3 * X6 * X5
                    + 3 * X0 * X1 * X6 * X5
                    + 3 * X0 * X2 * X3 * X5
                    + 3 * X0 * X3 * X6 * X4
                    + 2 * X0 * X1 * X6 * X4
                    + 2 * X0 * X3.pow_ref(2) * X4
                    + 2 * X0 * X2 * X3 * X4
                    + 3 * X0 * X3.pow_ref(2) * X4 * X5
                    + 4 * X0 * X1 * X3 * X4 * X5
                    + X0 * X3.pow_ref(2) * X4.pow_ref(2),
                5 + 4 * X0
                    + 6 * X4 * X5
                    + 3 * X6 * X5
                    + 4 * X0 * X4
                    + 3 * X0 * X6
                    + 6 * X0 * X3
                    + 6 * X0 * X2
                    + 6 * X6 * X4 * X5
                    + 2 * X0 * X4 * X5
                    + 4 * X0 * X6 * X5
                    + 3 * X0 * X2 * X5
                    + 3 * X0 * X6 * X4
                    + 5 * X0 * X3 * X4
                    + 6 * X0 * X2 * X4
                    + 4 * X0 * X6.pow_ref(2)
                    + 3 * X0 * X3 * X6
                    + 3 * X0 * X2 * X6
                    + 2 * X0 * X6 * X4 * X5
                    + 6 * X0 * X3 * X4 * X5
                    + 5 * X0 * X1 * X4 * X5
                    + 6 * X0 * X6.pow_ref(2) * X5
                    + 2 * X0 * X3 * X6 * X5
                    + 2 * X0 * X2 * X6 * X5
                    + 6 * X0 * X1 * X6 * X5
                    + 6 * X0 * X2 * X3 * X5
                    + 6 * X0 * X3 * X4.pow_ref(2)
                    + 4 * X0 * X6.pow_ref(2) * X4
                    + 6 * X0 * X3 * X6 * X4
                    + 3 * X0 * X2 * X6 * X4
                    + 4 * X0 * X3 * X6 * X4 * X5
                    + 5 * X0 * X1 * X6 * X4 * X5
                    + 6 * X0 * X3.pow_ref(2) * X4 * X5
                    + 5 * X0 * X2 * X3 * X4 * X5
                    + 3 * X0 * X3 * X6 * X4.pow_ref(2)
                    + 6 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.clone(),
                2 + 2 * X0
                    + 4 * X0 * X4
                    + 2 * X0 * X6
                    + 5 * X0 * X4 * X5
                    + 2 * X0 * X6 * X5
                    + 4 * X0 * X2 * X5
                    + 2 * X0 * X4.pow_ref(2)
                    + 4 * X0 * X6 * X4
                    + 4 * X0 * X6.pow_ref(2)
                    + 2 * X6 * X4 * X5.pow_ref(2)
                    + 4 * X0 * X6 * X4 * X5
                    + X0 * X3 * X4 * X5
                    + X0 * X2 * X4 * X5
                    + 3 * X0 * X6.pow_ref(2) * X5
                    + 2 * X0 * X3 * X6 * X5
                    + 4 * X0 * X2 * X6 * X5
                    + 2 * X0 * X6 * X4.pow_ref(2)
                    + X0 * X6.pow_ref(2) * X4
                    + 3 * X0 * X6 * X4 * X5.pow_ref(2)
                    + 2 * X0 * X6.pow_ref(2) * X5.pow_ref(2)
                    + 3 * X0 * X2 * X6 * X5.pow_ref(2)
                    + X0 * X3 * X4.pow_ref(2) * X5
                    + X0 * X6.pow_ref(2) * X4 * X5
                    + X0 * X3 * X6 * X4 * X5
                    + 6 * X0 * X2 * X6 * X4 * X5
                    + 4 * X0 * X6.pow_ref(2) * X4.pow_ref(2)
                    + 6 * X0 * X3 * X6 * X4 * X5.pow_ref(2)
                    + 4 * X0 * X1 * X6 * X4 * X5.pow_ref(2)
                    + 4 * X0 * X2 * X3 * X4 * X5.pow_ref(2)
                    + 6 * X0 * X3 * X6 * X4.pow_ref(2) * X5
                    + 2 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2),
                4 + 5 * X0 * X4 * X5
                    + 6 * X0 * X6 * X5
                    + 5 * X0 * X4.pow_ref(2) * X5
                    + 3 * X0 * X6 * X4 * X5
                    + 3 * X0 * X6.pow_ref(2) * X5
                    + 6 * X0 * X6 * X4 * X5.pow_ref(2)
                    + 5 * X0 * X2 * X4 * X5.pow_ref(2)
                    + X0 * X6.pow_ref(2) * X5.pow_ref(2)
                    + 6 * X0 * X2 * X6 * X5.pow_ref(2)
                    + 4 * X0 * X6 * X4.pow_ref(2) * X5
                    + 2 * X0 * X6.pow_ref(2) * X4 * X5
                    + 5 * X0 * X3 * X4.pow_ref(2) * X5.pow_ref(2)
                    + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2)
                    + 5 * X0 * X3 * X6 * X4 * X5.pow_ref(2)
                    + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(2)
                    + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5
                    + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(2),
                4 + 4 * X0 * X4.pow_ref(2) * X5.pow_ref(2)
                    + X0 * X6 * X4 * X5.pow_ref(2)
                    + X0 * X6.pow_ref(2) * X5.pow_ref(2)
                    + 5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(2)
                    + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2)
                    + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3)
                    + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(3)
                    + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2)
                    + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(3),
                5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(3)
                    + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3)
                    + 5 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(3),
                2 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(4),
            ]
        });

        let gb = buchberger(&ring, basis, DegRevLex);
        assert_eq!(130, gb.len());
    }

    #[test]
    #[ignore]
    fn test_groebner_cyclic6() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<65537>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 6);

        let cyclic6 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v]| {
            [
                x + y + z + t + u + v,
                x * y + y * z + z * t + t * u + x * v + u * v,
                x * y * z + y * z * t + z * t * u + x * y * v + x * u * v + t * u * v,
                x * y * z * t + y * z * t * u + x * y * z * v + x * y * u * v + x * t * u * v + z * t * u * v,
                x * y * z * t * u
                    + x * y * z * t * v
                    + x * y * z * u * v
                    + x * y * t * u * v
                    + x * z * t * u * v
                    + y * z * t * u * v,
                x * y * z * t * u * v - 1,
            ]
        });

        let gb = buchberger(&ring, cyclic6, DegRevLex);
        assert_eq!(45, gb.len());
    }

    // origin (w.r.t. `algorithms/buchberger.rs`): minor changes: adapted to new API
    #[test]
    #[ignore]
    fn test_groebner_cyclic7() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<65537>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 7);

        let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v, w]| {
            [
                x + y + z + t + u + v + w,
                x * y + y * z + z * t + t * u + u * v + x * w + v * w,
                x * y * z + y * z * t + z * t * u + t * u * v + x * y * w + x * v * w + u * v * w,
                x * y * z * t
                    + y * z * t * u
                    + z * t * u * v
                    + x * y * z * w
                    + x * y * v * w
                    + x * u * v * w
                    + t * u * v * w,
                x * y * z * t * u
                    + y * z * t * u * v
                    + x * y * z * t * w
                    + x * y * z * v * w
                    + x * y * u * v * w
                    + x * t * u * v * w
                    + z * t * u * v * w,
                x * y * z * t * u * v
                    + x * y * z * t * u * w
                    + x * y * z * t * v * w
                    + x * y * z * u * v * w
                    + x * y * t * u * v * w
                    + x * z * t * u * v * w
                    + y * z * t * u * v * w,
                x * y * z * t * u * v * w - 1,
            ]
        });

        let gb = buchberger(&ring, cyclic7, DegRevLex);
        assert_eq!(209, gb.len());
    }

    #[test]
    #[ignore]
    fn test_groebner_cyclic8() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<65537>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 8);

        let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, s, t, u, v, w]| {
            [
                x + y + z + s + t + u + v + w,
                x * y + y * z + z * s + s * t + t * u + u * v + x * w + v * w,
                x * y * z + y * z * s + z * s * t + s * t * u + t * u * v + x * y * w + x * v * w + u * v * w,
                x * y * z * s
                    + y * z * s * t
                    + z * s * t * u
                    + s * t * u * v
                    + x * y * z * w
                    + x * y * v * w
                    + x * u * v * w
                    + t * u * v * w,
                x * y * z * s * t
                    + y * z * s * t * u
                    + z * s * t * u * v
                    + x * y * z * s * w
                    + x * y * z * v * w
                    + x * y * u * v * w
                    + x * t * u * v * w
                    + s * t * u * v * w,
                x * y * z * s * t * u
                    + y * z * s * t * u * v
                    + x * y * z * s * t * w
                    + x * y * z * s * v * w
                    + x * y * z * u * v * w
                    + x * y * t * u * v * w
                    + x * s * t * u * v * w
                    + z * s * t * u * v * w,
                x * y * z * s * t * u * v
                    + x * y * z * s * t * u * w
                    + x * y * z * s * t * v * w
                    + x * y * z * s * u * v * w
                    + x * y * z * t * u * v * w
                    + x * y * s * t * u * v * w
                    + x * z * s * t * u * v * w
                    + y * z * s * t * u * v * w,
                x * y * z * s * t * u * v * w - 1,
            ]
        });

        let gb = buchberger(&ring, cyclic7, DegRevLex);
        assert_eq!(372, gb.len());
    }

    #[test]
    #[ignore]
    fn test_groebner_dense_lex() {
        feanor_tracing::DelayedLogger::init_test();
        let base = zn_static::Fp::<65537>::RING;
        let ring: MultivariatePolyRingImpl<_> = crate::ring::RingArc::from(std::sync::Arc::new(
            crate::ring_impls::multivariate::multivariate_impl::MultivariatePolyRingImplBase::new_with_mult_table(
                base,
                2,
                2000,
                (6, 8),
                std::alloc::Global,
            ),
        ));
        // two dense polynomials of degree `d` with pseudo-random coefficients
        let d = 6;
        let mut rng = oorandom::Rand64::new(1);
        let system = (0..2)
            .map(|_| {
                ring.from_terms(
                    (0..=d)
                        .flat_map(|i| (0..=(d - i)).map(move |j| [i, j]))
                        .map(|e| {
                            (
                                base.int_hom().map((rng.rand_u64() % 65537) as i32),
                                ring.create_monomial(e),
                            )
                        })
                        .collect::<Vec<_>>()
                        .into_iter(),
                )
            })
            .collect::<Vec<_>>();
        let gb = buchberger(&ring, system, Lex);
        assert_eq!(2, gb.len());
    }
}
