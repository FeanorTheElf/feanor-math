use append_only_vec::AppendOnlyVec;

use crate::computation::*;
use crate::delegate::{UnwrapHom, WrapHom};
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::local::{PrincipalLocalRing, PrincipalLocalRingStore};
use crate::pid::PrincipalIdealRingStore;
use crate::ring::*;
use crate::seq::*;
use crate::rings::multivariate::*;

use std::cmp::min;
use std::fmt::Debug;

#[stability::unstable(feature = "enable")]
#[derive(PartialEq, Clone, Eq, Hash)]
pub enum SPoly {
    Standard(usize, usize), Nilpotent(/* poly index */ usize, /* power-of-p multiplier */ usize)
}

impl Debug for SPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SPoly::Standard(i, j) => write!(f, "S({}, {})", i, j),
            SPoly::Nilpotent(i, k) => write!(f, "p^{} F({})", k, i)
        }
    }
}

fn term_xlcm<P>(ring: P, (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>), (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>)) -> ((PolyCoeff<P>, PolyMonomial<P>), (PolyCoeff<P>, PolyMonomial<P>), (PolyCoeff<P>, PolyMonomial<P>))
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring.base_ring().mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (
        (l_factor, ring.monomial_div(ring.clone_monomial(&m_m), &l_m).ok().unwrap()),
        (r_factor, ring.monomial_div(ring.clone_monomial(&m_m), &r_m).ok().unwrap()),
        (m_c, m_m)
    );
}

fn term_lcm<P>(ring: P, (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>), (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>)) -> (PolyCoeff<P>, PolyMonomial<P>)
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring.base_ring().mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (m_c, m_m);
}

impl SPoly {

    #[stability::unstable(feature = "enable")]
    pub fn lcm_term<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> (PolyCoeff<P>, PolyMonomial<P>)
        where P: RingStore + Copy,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => term_lcm(&ring, ring.LT(&basis[*i], order).unwrap(), ring.LT(&basis[*j], order).unwrap()),
            Self::Nilpotent(i, k) => {
                let (c, m) = ring.LT(&basis[*i], order).unwrap();
                (ring.base_ring().mul_ref_fst(c, ring.base_ring().pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k)), ring.clone_monomial(m))
            }
        }
    }

    
    #[stability::unstable(feature = "enable")]
    pub fn poly<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> El<P>
        where P: RingStore + Copy,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => {
                let (f1_factor, f2_factor, _) = term_xlcm(&ring, ring.LT(&basis[*i], order).unwrap(), ring.LT(&basis[*j], order).unwrap());
                let mut f1_scaled = ring.clone_el(&basis[*i]);
                ring.mul_assign_monomial(&mut f1_scaled, f1_factor.1);
                ring.inclusion().mul_assign_map(&mut f1_scaled, f1_factor.0);
                let mut f2_scaled = ring.clone_el(&basis[*j]);
                ring.mul_assign_monomial(&mut f2_scaled, f2_factor.1);
                ring.inclusion().mul_assign_map(&mut f2_scaled, f2_factor.0);
                return ring.sub(f1_scaled, f2_scaled);
            },
            SPoly::Nilpotent(i, k) => {
                let mut result = ring.clone_el(&basis[*i]);
                ring.inclusion().mul_assign_map(&mut result, ring.base_ring().pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k));
                return result;
            }
        }
    }
}

#[inline(never)]
fn find_reducer<'a, 'b, P, O, I>(ring: P, f: &El<P>, reducers: I, order: O) -> Option<(usize, &'a El<P>, PolyCoeff<P>, PolyMonomial<P>)>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        I: Iterator<Item = (&'a El<P>, &'b ExpandedMonomial)>
{
    if ring.is_zero(&f) {
        return None;
    }
    let (f_lc, f_lm) = ring.LT(f, order).unwrap();
    let f_lm_expanded = ring.expand_monomial(f_lm);
    reducers.enumerate().filter_map(|(i, (reducer, reducer_lm_expanded))| {
        if (0..ring.indeterminate_count()).all(|j| reducer_lm_expanded[j] <= f_lm_expanded[j]) {
            let (r_lc, r_lm) = ring.LT(reducer, order).unwrap();
            let quo_m = ring.monomial_div(ring.clone_monomial(f_lm), r_lm).ok().unwrap();
            if let Some(quo_c) = ring.base_ring().checked_div(f_lc, r_lc) {
                return Some((i, reducer, quo_c, quo_m));
            }
        }
        return None;
    }).next()
}

#[inline(never)]
fn filter_spoly<P, O>(ring: P, new_spoly: SPoly, basis: &[El<P>], order: O) -> Option<usize>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy
{
    match new_spoly {
        SPoly::Standard(i, k) => {
            assert!(i < k);
            let (bi_c, bi_m) = ring.LT(&basis[i], order).unwrap();
            let (bk_c, bk_m) = ring.LT(&basis[k], order).unwrap();
            let (S_c, S_m) = term_lcm(ring, (bi_c, bi_m), (bk_c, bk_m));
            let S_c_val = ring.base_ring().valuation(&S_c).unwrap();

            if S_c_val == 0 && order.eq_mon(ring, &ring.monomial_div(ring.clone_monomial(&S_m), &bi_m).ok().unwrap(), &bk_m) {
                return Some(usize::MAX);
            }

            (0..k).filter_map(|j| {
                if j == i {
                    return None;
                }
                // more experiments needed - for some weird reason, replacing "properly divides" with "divides" (assuming
                // I didn't make a mistake) leads to terrible performance
                let (bj_c, bj_m) = ring.LT(&basis[j], order).unwrap();
                let (f_c, f_m) = term_lcm(ring, (bj_c, bj_m), (bk_c, bk_m));
                let f_c_val = ring.base_ring().valuation(&f_c).unwrap();

                if j < i && order.eq_mon(ring, &f_m, &S_m) && f_c_val <= S_c_val {
                    return Some(j);
                }
                if let Ok(quo) = ring.monomial_div(ring.clone_monomial(&S_m), &f_m) {
                    if f_c_val <= S_c_val && (f_c_val < S_c_val || ring.monomial_deg(&quo) > 0) {
                        return Some(j);
                    }
                }
                return None;
            }).next()
        },
        SPoly::Nilpotent(i, k) => {
            let nilpotent_power = ring.base_ring().nilpotent_power().unwrap();
            let f = &basis[i];

            let mut smallest_elim_coeff_valuation = usize::MAX;
            let mut current = ring.LT(f, order).unwrap();
            while ring.base_ring().valuation(&current.0).unwrap() + k >= nilpotent_power {
                smallest_elim_coeff_valuation = min(smallest_elim_coeff_valuation, ring.base_ring().valuation(&current.0).unwrap());
                let next = ring.largest_term_lt(f, order, &current.1);
                if next.is_none() {
                    return Some(usize::MAX);
                }
                current = next.unwrap();
            }
            assert!(smallest_elim_coeff_valuation == usize::MAX || smallest_elim_coeff_valuation + k >= nilpotent_power);
            if smallest_elim_coeff_valuation == usize::MAX || smallest_elim_coeff_valuation + k > nilpotent_power {
                return Some(usize::MAX);
            } else {
                return None;
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn default_sort_fn<P, O>(ring: P, order: O) -> impl FnMut(&mut [SPoly], &[El<P>])
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy
{
    move |open, basis| open.sort_by_key(|spoly| {
        let (lc, lm) = spoly.lcm_term(ring, &basis, order);
        (-(ring.base_ring().valuation(&lc).unwrap_or(0) as i64), -(ring.monomial_deg(&lm) as i64))
    })
}

#[stability::unstable(feature = "enable")]
pub type ExpandedMonomial = Vec<usize>;

fn augment_lm<P, O>(ring: P, f: El<P>, order: O) -> (El<P>, ExpandedMonomial)
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder
{
    let exponents = ring.expand_monomial(ring.LT(&f, order).unwrap().1);
    return (f, exponents);
}

///
/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given term ordering.
/// 
/// For a variant of this function that uses sensible defaults for most parameters, see [`buchberger_simple()`].
/// 
/// The algorithm proceeds F4-style, i.e. reduces multiple S-polynomials before adding them to the basis.
/// When using a fast polynomial ring implementation (e.g. [`crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl`]),
/// this makes the algorithm as efficient as standard F4. Furthermore, the behavior can be modified by passing custom functions for
/// `sort_spolys` and `abort_early_if`.
/// 
/// - `sort_spolys` should permute the given list of S-polynomials w.r.t. the given basis; this can be used to customize in which
///   order S-polynomials are reduced, which can have huge impact on performance.
///   Note that S-polynomials that are supposed to be reduced first should be put at the end of the list.
/// - `abort_early_if` takes the current basis (unfortunately, currently with some additional information that can be ignored), and
///   can return `true` to abort the GB computation, yielding the current basis. In this case, the basis will in general not be a GB,
///   but can still be useful (e.g. `abort_early_if` might decide that a GB up to a fixed degree is sufficient).
/// 
/// # Explanation of logging output
/// 
/// If the passed computation controller accepts the logging, it will receive the following symbols:
///  - `-` means an S-polynomial was reduced to zero
///  - `s` means an S-polynomial reduced to a nonzero value and will be added to the basis at the next opportunity
///  - `b(n)` means that the list of all generated basis polynomials has length `n`
///  - `r(n)` means that the current basis of the ideal has length `n`
///  - `S(n)` means that the algorithm still has to reduce `n` more S-polynomials
///  - `f(n)` means that `n` S-polynomials have, in total, been discarded by using the Buchberger criteria
///  - `{n}` means that the algorithm is currently reducing S-polynomials of degree `n`
///  - `!` means that the algorithm decided to discard all current S-polynomial, and restart the computation with the current basis
/// 
#[stability::unstable(feature = "enable")]
pub fn buchberger<P, O, Controller, SortFn, AbortFn>(ring: P, input_basis: Vec<El<P>>, order: O, mut sort_spolys: SortFn, mut abort_early_if: AbortFn, controller: Controller) -> Result<Vec<El<P>>, Controller::Abort>
    where P: RingStore + Copy + Send + Sync,
        El<P>: Send + Sync,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy + Send + Sync,
        PolyCoeff<P>: Send + Sync,
        Controller: ComputationController,
        SortFn: FnMut(&mut [SPoly], &[El<P>]),
        AbortFn: FnMut(&[(El<P>, ExpandedMonomial)]) -> bool
{
    controller.run_computation(format_args!("buchberger(len={}, vars={})", input_basis.len(), ring.indeterminate_count()), |controller| {

        // this are the basis polynomials we generated; we only append to this, such that the S-polys remain valid
        let input_basis = inter_reduce(&ring, input_basis.into_iter().map(|f| augment_lm(ring, f, order)).collect(), order).into_iter().map(|(f, _)| f).collect::<Vec<_>>();
        debug_assert!(input_basis.iter().all(|f| !ring.is_zero(f)));

        let nilpotent_power = ring.base_ring().nilpotent_power().and_then(|e| if e != 0 { Some(e) } else { None });
        assert!(nilpotent_power.is_none() || ring.base_ring().is_zero(&ring.base_ring().pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), nilpotent_power.unwrap())));

        let sort_reducers = |reducers: &mut [(El<P>, ExpandedMonomial)]| {
            // I have no idea why, but this order seems to give the best results
            reducers.sort_by(|(lf, _), (rf, _)| order.compare(ring, &ring.LT(lf, order).unwrap().1, &ring.LT(rf, order).unwrap().1).then_with(|| ring.terms(lf).count().cmp(&ring.terms(rf).count())))
        };

        // invariant: `(reducers) = (basis)` and there exists a reduction to zero for every `f` in `basis` modulo `reducers`;
        // reducers are always stored with an expanded version of their leading monomial, in order to simplify divisibility checks
        let mut reducers: Vec<(El<P>, ExpandedMonomial)> = input_basis.iter().map(|f| augment_lm(ring, ring.clone_el(f), order)).collect::<Vec<_>>();
        sort_reducers(&mut reducers);

        let mut open = Vec::new();
        let mut basis = Vec::new();
        update_basis(ring, input_basis.into_iter(), &mut basis, &mut open, order, nilpotent_power, &mut 0, &mut sort_spolys);

        let mut current_deg = 0;
        let mut filtered_spolys = 0;
        let mut changed = false;
        loop {

            // reduce all known S-polys of minimal lcm degree; in effect, this is the same as 
            // the matrix reduction step during F4
            let spolys_to_reduce_index = open.iter().enumerate().rev().filter(|(_, spoly)| ring.monomial_deg(&spoly.lcm_term(ring, &basis, order).1) > current_deg).next().map(|(i, _)| i + 1).unwrap_or(0);
            let spolys_to_reduce = &open[spolys_to_reduce_index..];

            let computation = ShortCircuitingComputation::new();
            let new_polys = AppendOnlyVec::new();
            let new_polys_ref = &new_polys;
            let basis_ref = &basis;
            let reducers_ref = &reducers;

            computation.handle(controller.clone()).join_many(spolys_to_reduce.as_fn().map_fn(move |spoly| move |handle: ShortCircuitingComputationHandle<(), _>| {
                let mut f = spoly.poly(ring, basis_ref, order);
                
                reduce_poly(ring, &mut f, || reducers_ref.iter().chain(new_polys_ref.iter()).map(|(f, lmf)| (f, lmf)), order);

                if !ring.is_zero(&f) {
                    log_progress!(handle, "s");
                    _ = new_polys_ref.push(augment_lm(ring, f, order));
                } else {
                    log_progress!(handle, "-");
                }

                checkpoint!(handle);
                return Ok(None);
            }));

            drop(open.drain(spolys_to_reduce_index..));
            let new_polys = new_polys.into_vec();
            _ = computation.finish()?;

            // process the generated new polynomials
            if new_polys.len() == 0 && open.len() == 0 {
                if changed {
                    log_progress!(controller, "!");
                    // this seems necessary, as the invariants for `reducers` don't imply that it already is a GB;
                    // more concretely, reducers contains polys of basis that are reduced with eath other, but the
                    // S-polys between two of them might not have been considered
                    return buchberger::<P, O, _, _, _>(ring, reducers.into_iter().map(|(f, _)| f).collect(), order, sort_spolys, abort_early_if, controller);
                } else {
                    return Ok(reducers.into_iter().map(|(f, _)| f).collect());
                }
            } else if new_polys.len() == 0 {
                current_deg = ring.monomial_deg(&open.last().unwrap().lcm_term(ring, &basis, order).1);
                log_progress!(controller, "{{{}}}", current_deg);
            } else {
                changed = true;
                current_deg = 0;
                update_basis(ring, new_polys.iter().map(|(f, _)| ring.clone_el(f)), &mut basis, &mut open, order, nilpotent_power, &mut filtered_spolys, &mut sort_spolys);
                log_progress!(controller, "(b={})(S={})(f={})", basis.len(), open.len(), filtered_spolys);

                reducers.extend(new_polys.into_iter());
                reducers = inter_reduce(ring, reducers, order);
                sort_reducers(&mut reducers);
                log_progress!(controller, "(r={})", reducers.len());
                if abort_early_if(&reducers) {
                    log_progress!(controller, "(early_abort)");
                    return Ok(reducers.into_iter().map(|(f, _)| f).collect());
                }
            }

            // less S-polys if we restart from scratch with reducers
            if open.len() + filtered_spolys > reducers.len() * reducers.len() / 2 + reducers.len() * nilpotent_power.unwrap_or(0) + 1 {
                log_progress!(controller, "!");
                return buchberger::<P, O, _, _, _>(ring, reducers.into_iter().map(|(f, _)| f).collect(), order, sort_spolys, abort_early_if, controller);
            }
        }
    })
}

fn update_basis<I, P, O, SortFn>(ring: P, new_polys: I, basis: &mut Vec<El<P>>, open: &mut Vec<SPoly>, order: O, nilpotent_power: Option<usize>, filtered_spolys: &mut usize, sort_spolys: &mut SortFn)
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy,
        SortFn: FnMut(&mut [SPoly], &[El<P>]),
        I: Iterator<Item = El<P>>
{
    for new_poly in new_polys {
        basis.push(new_poly);
        for i in 0..(basis.len() - 1) {
            let spoly = SPoly::Standard(i, basis.len() - 1);
            if filter_spoly(ring, spoly.clone(), &*basis, order).is_none() {
                open.push(spoly);
            } else {
                *filtered_spolys += 1;
            }
        }
        if let Some(e) = nilpotent_power {
            for k in 1..e {
                let spoly = SPoly::Nilpotent(basis.len() - 1, k);
                if filter_spoly(ring, spoly.clone(), &*basis, order).is_none() {
                    open.push(spoly);
                } else {
                    *filtered_spolys += 1;
                }
            }
        }
    }
    sort_spolys(&mut *open, &*basis);
}

fn reduce_poly<'a, 'b, F, I, P, O>(ring: P, to_reduce: &mut El<P>, mut reducers: F, order: O)
    where P: 'a + RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        F: FnMut() -> I,
        I: Iterator<Item = (&'a El<P>, &'b ExpandedMonomial)>
{
    while let Some((_, reducer, quo_c, quo_m)) = find_reducer(ring, to_reduce, reducers(), order) {
        let prev_lm = ring.clone_monomial(&ring.LT(to_reduce, order).unwrap().1);
        let mut scaled_reducer = ring.clone_el(reducer);
        ring.mul_assign_monomial(&mut scaled_reducer, ring.clone_monomial(&quo_m));
        ring.inclusion().mul_assign_ref_map(&mut scaled_reducer, &quo_c);
        debug_assert!(order.compare(ring, ring.LT(&scaled_reducer, order).unwrap().1, ring.LT(&to_reduce, order).unwrap().1) == std::cmp::Ordering::Equal);
        ring.sub_assign(to_reduce, scaled_reducer);
        debug_assert!(ring.is_zero(&to_reduce) || order.compare(ring, &ring.LT(&to_reduce, order).unwrap().1, &prev_lm) == std::cmp::Ordering::Less);
    }
}

#[stability::unstable(feature = "enable")]
pub fn multivariate_division<'a, P, O, I>(ring: P, mut f: El<P>, reducers: I, order: O) -> El<P>
    where P: 'a + RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        I: Clone + Iterator<Item = &'a El<P>>
{
    let lms = reducers.clone().map(|f| ring.expand_monomial(ring.LT(f, order).unwrap().1)).collect::<Vec<_>>();
    reduce_poly(ring, &mut f, || reducers.clone().zip(lms.iter()), order);
    return f;
}

fn inter_reduce<P, O>(ring: P, mut polys: Vec<(El<P>, ExpandedMonomial)>, order: O) -> Vec<(El<P>, ExpandedMonomial)>
    where P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy
{
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < polys.len() {
            let last_i = polys.len() - 1;
            polys.swap(i, last_i);
            let (reducers, to_reduce) = polys.split_at_mut(last_i);
            let to_reduce = &mut to_reduce[0];

            reduce_poly(ring, &mut to_reduce.0, || reducers.iter().map(|(f, lmf)| (f, lmf)), order);

            // undo swap so that the outer loop still iterates over every poly
            if !ring.is_zero(&to_reduce.0) {
                to_reduce.1 = ring.expand_monomial(ring.LT(&to_reduce.0, order).unwrap().1);
                polys.swap(i, last_i);
                i += 1;
            } else {
                _ = polys.pop().unwrap();
            }
        }
    }
    return polys;
}

use crate::rings::local::AsLocalPIR;
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;

///
/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given term ordering.
/// 
/// For a variant of this function that allows for more configuration, see [`buchberger()`].
/// 
pub fn buchberger_simple<P, O>(ring: P, input_basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: RingStore + Copy + Send + Sync,
        El<P>: Send + Sync,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy + Send + Sync,
        PolyCoeff<P>: Send + Sync
{
    let as_local_pir = AsLocalPIR::from_field(ring.base_ring());
    let new_poly_ring = MultivariatePolyRingImpl::new(&as_local_pir, ring.indeterminate_count());
    let from_ring = new_poly_ring.lifted_hom(ring, WrapHom::new(as_local_pir.get_ring()));
    let result = buchberger::<_, _, _, _, _>(
        &new_poly_ring, 
        input_basis.into_iter().map(|f| from_ring.map(f)).collect(), 
        order, 
        default_sort_fn(&new_poly_ring, order), 
        |_| false, 
        DontObserve
    ).unwrap_or_else(no_error);
    let to_ring = ring.lifted_hom(&new_poly_ring, UnwrapHom::new(as_local_pir.get_ring()));
    return result.into_iter().map(|f| to_ring.map(f)).collect();
}

#[cfg(test)]
use crate::rings::poly::{dense_poly, PolyRingStore};
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::rational::RationalField;

#[test]
fn test_buchberger_small() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);

    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 0])),
        (1, ring.create_monomial([0, 2])),
        (16, ring.create_monomial([0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([1, 1])),
        (15, ring.create_monomial([0, 0]))
    ].into_iter());

    let actual = buchberger(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS).unwrap_or_else(no_error);

    let expected = ring.from_terms([
        (16, ring.create_monomial([0, 3])),
        (15, ring.create_monomial([1, 0])),
        (1, ring.create_monomial([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, f1, actual.iter(), DegRevLex));
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, f2, actual.iter(), DegRevLex));
    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, expected, actual.iter(), DegRevLex));
}

#[test]
fn test_buchberger_larger() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 3);

    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 1, 1])),
        (1, ring.create_monomial([0, 2, 0])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([1, 0, 0])),
        (1, ring.create_monomial([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([0, 3, 1])),
        (1, ring.create_monomial([0, 0, 3])),
        (1, ring.create_monomial([1, 1, 0]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, ring.create_monomial([1, 0, 2])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([0, 1, 1])),
        (7, ring.create_monomial([0, 0, 0]))
    ].into_iter());

    let actual = buchberger(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS).unwrap_or_else(no_error);

    let g1 = ring.from_terms([
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
        (9, ring.create_monomial([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, g1, actual.iter(), DegRevLex));
}

#[test]
fn test_generic_computation() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 6);
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let var_i = |i: usize| ring.create_term(base.one(), ring.create_monomial((0..ring.indeterminate_count()).map(|j| if i == j { 1 } else { 0 })));
    let X1 = poly_ring.mul(
        poly_ring.from_terms([(var_i(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(var_i(1), 0), (ring.one(), 1)].into_iter())
    );
    let X2 = poly_ring.mul(
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(var_i(2), 0), (var_i(3), 1)].into_iter())),
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(var_i(4), 0), (var_i(5), 1)].into_iter()))
    );
    let basis = vec![
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 0)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 1)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = buchberger(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(11, gb1.len());
}

#[test]
fn test_gb_local_ring() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);
    
    let f = ring.from_terms([(base.int_hom().map(4), ring.create_monomial([1])), (base.one(), ring.create_monomial([0]))].into_iter());
    let gb = buchberger(&ring, vec![f], DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS).unwrap_or_else(no_error);

    assert_eq!(1, gb.len());
    assert_el_eq!(ring, ring.one(), gb[0]);
}

#[test]
fn test_gb_lex() {
    let ZZ = BigIntRing::RING;
    let QQ = AsLocalPIR::from_field(RationalField::new(ZZ));
    let QQYX = MultivariatePolyRingImpl::new(&QQ, 2);
    let [f, g] = QQYX.with_wrapped_indeterminates(|[Y, X]| [ 1 + X.pow_ref(2) + 2 * Y + (1 + X) * Y.pow_ref(2), 3 + X + (2 + X) * Y + (1 + X + X.pow_ref(2)) * Y.pow_ref(2) ]);
    let expected = QQYX.with_wrapped_indeterminates(|[Y, X]| [ 
        X.pow_ref(8) + 2 * X.pow_ref(7) + 3 * X.pow_ref(6) - 5 * X.pow_ref(5) - 10 * X.pow_ref(4) - 7 * X.pow_ref(3) + 8 * X.pow_ref(2) + 8 * X + 4,
        2 * Y + X.pow_ref(6) + 3 * X.pow_ref(5) + 6 * X.pow_ref(4) + X.pow_ref(3) - 7 * X.pow_ref(2) - 12 * X - 2, 
    ]);

    let mut gb = buchberger_simple(&QQYX, vec![f, g], Lex);

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

#[cfg(test)]
#[cfg(feature = "parallel")]
static TEST_COMPUTATION_CONTROLLER: ExecuteMultithreaded<LogProgress> = RunMultithreadedLogProgress;
#[cfg(test)]
#[cfg(not(feature = "parallel"))]
static TEST_COMPUTATION_CONTROLLER: LogProgress = TEST_LOG_PROGRESS;

#[ignore]
#[test]
fn test_expensive_gb_1() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 12);

    let system = ring.with_wrapped_indeterminates(|[Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11]| [
        Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4.pow_ref(2) + 4 * Y0 * Y1 * Y2 * Y3 * Y4 * Y4 * Y8 + Y0 * Y1 * Y2 * Y5.pow_ref(2) * Y8.pow_ref(2) + Y0 * Y2 * Y3 * Y4 * Y6 + Y0 * Y1 * Y3 * Y4 * Y7 + Y0 * Y2 * Y5 * Y6 * Y8 + Y0 * Y1 * Y5 * Y7 * Y8 + Y0 * Y2 * Y3 * Y5 * Y10 + Y0 * Y1 * Y3 * Y5 * Y11 + Y0 * Y6 * Y7 + Y3 * Y5 * Y9 -  4,
        2 * Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4 * Y5 + 2 * Y0 * Y1 * Y2 * Y3 * Y5.pow_ref(2) * Y8 + Y0 * Y2 * Y3 * Y5 * Y6 + Y0 * Y1 * Y3 * Y5 * Y7 + 8,
        Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y5.pow_ref(2) - 5
    ]);

    let part_of_result = ring.with_wrapped_indeterminates(|[_Y0, Y1, Y2, _Y3, _Y4, _Y5, Y6, Y7, _Y8, _Y9, _Y10, _Y11]| [
        4 * Y2.pow_ref(2) * Y6.pow_ref(2) -  4 * Y1.pow_ref(2) * Y7.pow_ref(2),
        8 * Y2 * Y6 + 8 * Y1 * Y7.clone()
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger(&ring, system.iter().map(|f| ring.clone_el(f)).collect(), DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_COMPUTATION_CONTROLLER).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    for f in &part_of_result {
        assert!(ring.is_zero(&multivariate_division(&ring, ring.clone_el(f), gb.iter(), DegRevLex)));
    }

    assert_eq!(108, gb.len());
}

#[test]
#[ignore]
fn test_expensive_gb_2() {
    let base = zn_static::Fp::<7>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 7);

    let basis = ring.with_wrapped_indeterminates_dyn(|[X0, X1, X2, X3, X4, X5, X6]| [
        6 + 2 * X5 + 2 * X4 + X6 + 4 * X0 + 5 * X6 * X5 + X6 * X4 + 3 * X0 * X4 + 6 * X0 * X6 + 2 * X0 * X3 + X0 * X2 + 4 * X0 * X1 + 2 * X3 * X4 * X5 + 4 * X0 * X6 * X5 + 6 * X0 * X2 * X5 + 5 * X0 * X6 * X4 + 2 * X0 * X3 * X4 + 4 * X0 * X1 * X4 + X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 5 * X0 * X2 * X6 + 2 * X0 * X1 * X6 + X0 * X3.pow_ref(2) + 2 * X0 * X2 * X3 + 3 * X0 * X3 * X4 * X5 + 4 * X0 * X3 * X6 * X5 + 3 * X0 * X1 * X6 * X5 + 3 * X0 * X2 * X3 * X5 + 3 * X0 * X3 * X6 * X4 + 2 * X0 * X1 * X6 * X4 + 2 * X0 * X3.pow_ref(2) * X4 + 2 * X0 * X2 * X3 * X4 + 3 * X0 * X3.pow_ref(2) * X4 * X5 + 4 * X0 * X1 * X3 * X4 * X5 + X0 * X3.pow_ref(2) * X4.pow_ref(2),
        5 + 4 * X0 + 6 * X4 * X5 + 3 * X6 * X5 + 4 * X0 * X4 + 3 * X0 * X6 + 6 * X0 * X3 + 6 * X0 * X2 + 6 * X6 * X4 * X5 + 2 * X0 * X4 * X5 + 4 * X0 * X6 * X5 + 3 * X0 * X2 * X5 + 3 * X0 * X6 * X4 + 5 * X0 * X3 * X4 + 6 * X0 * X2 * X4 + 4 * X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 3 * X0 * X2 * X6 + 2 * X0 * X6 * X4 * X5 + 6 * X0 * X3 * X4 * X5 + 5 * X0 * X1 * X4 * X5 + 6 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 2 * X0 * X2 * X6 * X5 + 6 * X0 * X1 * X6 * X5 + 6 * X0 * X2 * X3 * X5 + 6 * X0 * X3 * X4.pow_ref(2) + 4 * X0 * X6.pow_ref(2) * X4 + 6 * X0 * X3 * X6 * X4 + 3 * X0 * X2 * X6 * X4 + 4 * X0 * X3 * X6 * X4 * X5 + 5 * X0 * X1 * X6 * X4 * X5 + 6 * X0 * X3.pow_ref(2) * X4 * X5 + 5 * X0 * X2 * X3 * X4 * X5 + 3 * X0 * X3 * X6 * X4.pow_ref(2) + 6 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.clone(),
        2 + 2 * X0 + 4 * X0 * X4 + 2 * X0 * X6 + 5 * X0 * X4 * X5 + 2 * X0 * X6 * X5 + 4 * X0 * X2 * X5 + 2 * X0 * X4.pow_ref(2) + 4 * X0 * X6 * X4 + 4 * X0 * X6.pow_ref(2) + 2 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X6 * X4 * X5 + X0 * X3 * X4 * X5 + X0 * X2 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 4 * X0 * X2 * X6 * X5 + 2 * X0 * X6 * X4.pow_ref(2) + X0 * X6.pow_ref(2) * X4 + 3 * X0 * X6 * X4 * X5.pow_ref(2) + 2 * X0 * X6.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X2 * X6 * X5.pow_ref(2) + X0 * X3 * X4.pow_ref(2) * X5 + X0 * X6.pow_ref(2) * X4 * X5 + X0 * X3 * X6 * X4 * X5 + 6 * X0 * X2 * X6 * X4 * X5 + 4 * X0 * X6.pow_ref(2) * X4.pow_ref(2) + 6 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X1 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X3 * X4 * X5.pow_ref(2) + 6 * X0 * X3 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 5 * X0 * X4 * X5 + 6 * X0 * X6 * X5 + 5 * X0 * X4.pow_ref(2) * X5 + 3 * X0 * X6 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 6 * X0 * X6 * X4 * X5.pow_ref(2) + 5 * X0 * X2 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X2 * X6 * X5.pow_ref(2) + 4 * X0 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X6.pow_ref(2) * X4 * X5 + 5 * X0 * X3 * X4.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 5 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5 + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 4 * X0 * X4.pow_ref(2) * X5.pow_ref(2) + X0 * X6 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2) + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(3),
        5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 5 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(3),
        2 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(4)
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger(&ring, basis, DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_COMPUTATION_CONTROLLER).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
}

#[test]
#[ignore]
fn test_groebner_cyclic6() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 6);

    let cyclic6 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v]| {
        [x + y + z + t + u + v, x*y + y*z + z*t + t*u + x*v + u*v, x*y*z + y*z*t + z*t*u + x*y*v + x*u*v + t*u*v, x*y*z*t + y*z*t*u + x*y*z*v + x*y*u*v + x*t*u*v + z*t*u*v, x*y*z*t*u + x*y*z*t*v + x*y*z*u*v + x*y*t*u*v + x*z*t*u*v + y*z*t*u*v, x*y*z*t*u*v - 1]
    });

    let start = std::time::Instant::now();
    let gb = buchberger(&ring, cyclic6, DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_COMPUTATION_CONTROLLER).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(45, gb.len());

}

#[test]
#[ignore]
fn test_groebner_cyclic7() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 7);

    let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v, w]| [
        x + y + z + t + u + v + w, x*y + y*z + z*t + t*u + u*v + x*w + v*w, x*y*z + y*z*t + z*t*u + t*u*v + x*y*w + x*v*w + u*v*w, x*y*z*t + y*z*t*u + z*t*u*v + x*y*z*w + x*y*v*w + x*u*v*w + t*u*v*w, 
        x*y*z*t*u + y*z*t*u*v + x*y*z*t*w + x*y*z*v*w + x*y*u*v*w + x*t*u*v*w + z*t*u*v*w, x*y*z*t*u*v + x*y*z*t*u*w + x*y*z*t*v*w + x*y*z*u*v*w + x*y*t*u*v*w + x*z*t*u*v*w + y*z*t*u*v*w, x*y*z*t*u*v*w - 1
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger(&ring, cyclic7, DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_COMPUTATION_CONTROLLER).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(209, gb.len());
}

#[test]
#[ignore]
fn test_groebner_cyclic8() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 8);

    let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, s, t, u, v, w]| [
        x + y + z + s + t + u + v + w, x*y + y*z + z*s + s*t + t*u + u*v + x*w + v*w, x*y*z + y*z*s + z*s*t + s*t*u + t*u*v + x*y*w + x*v*w + u*v*w, 
        x*y*z*s + y*z*s*t + z*s*t*u + s*t*u*v + x*y*z*w + x*y*v*w + x*u*v*w + t*u*v*w, x*y*z*s*t + y*z*s*t*u + z*s*t*u*v + x*y*z*s*w + x*y*z*v*w + x*y*u*v*w + x*t*u*v*w + s*t*u*v*w, x*y*z*s*t*u + y*z*s*t*u*v + x*y*z*s*t*w + x*y*z*s*v*w + x*y*z*u*v*w + x*y*t*u*v*w + x*s*t*u*v*w + z*s*t*u*v*w, 
        x*y*z*s*t*u*v + x*y*z*s*t*u*w + x*y*z*s*t*v*w + x*y*z*s*u*v*w + x*y*z*t*u*v*w + x*y*s*t*u*v*w + x*z*s*t*u*v*w + y*z*s*t*u*v*w, x*y*z*s*t*u*v*w - 1
    ]);

    let start = std::time::Instant::now();
    let gb = buchberger(&ring, cyclic7, DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_COMPUTATION_CONTROLLER).unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(372, gb.len());
}