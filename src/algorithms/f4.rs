use std::io::Write;
use std::marker::PhantomData;
use std::cmp::{min, Ordering};

use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::local::PrincipalLocalRing;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::ring::*;
use crate::rings::finite::FiniteRing;
use crate::rings::local::AsLocalPIRBase;
use crate::primitive_int::StaticRing;
use crate::rings::multivariate::*;
use crate::rings::zn::{ZnRing, zn_64, ZnRingStore, zn_static};
use crate::algorithms;

#[allow(type_alias_bounds)]
type Mon<P: MultivariatePolyRingStore> = Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>;

#[allow(type_alias_bounds)]
type Coeff<P: MultivariatePolyRingStore> = El<<P::Type as RingExtension>::BaseRing>;

#[allow(type_alias_bounds)]
type LT<P: MultivariatePolyRingStore> = (Coeff<P>, Mon<P>);

#[allow(non_camel_case_types)]
#[allow(type_alias_bounds)]
type LT_ref<'a, P: MultivariatePolyRingStore> = (&'a Coeff<P>, &'a Mon<P>);

struct MonomialMap<P, O, K = ()>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore,
        O: MonomialOrder + Copy
{
    data: Vec<(Mon<P>, K)>,
    order: O
}

enum AddResult<K = ()> {
    Present(usize, K), Added(usize)
}

impl<P, O, K> MonomialMap<P, O, K>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore,
        O: MonomialOrder + Copy
{
    fn new(order: O) -> Self {
        Self {
            data: Vec::new(),
            order: order
        }
    }

    fn add(&mut self, m: Mon<P>, k: K) -> AddResult<K> {
        match self.data.binary_search_by(|(x, _)| self.order.compare(x, &m).reverse()) {
            Ok(i) => {
                let old = std::mem::replace(&mut self.data[i].1, k);
                AddResult::Present(i, old)
            },
            Err(i) => {
                self.data.insert(i, (m, k));
                AddResult::Added(i)
            }
        }
    }

    fn at_index(&self, i: usize) -> &Mon<P> {
        &self.data[i].0
    }
}

///
/// Computes `(s, t, lcm(x, y))` such that `s * x = t * y = lcm(x, y)`.
/// 
fn coeff_lcm<R>(ring: R, lhs: &El<R>, rhs: &El<R>) -> (El<R>, El<R>, El<R>)
    where R: PrincipalIdealRingStore,
        R::Type: PrincipalIdealRing
{
    let gcd = ring.ideal_gen(lhs, rhs);
    let s = ring.checked_div(rhs, &gcd).unwrap();
    return (ring.clone_el(&s), ring.checked_div(lhs, &gcd).unwrap(), ring.mul(s, gcd));
}

fn mon_lcm<P>(ring: &P, lhs: &Mon<P>, rhs: &Mon<P>) -> (Mon<P>, Mon<P>, Mon<P>)
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing
{
    let gcd = ring.clone_monomial(lhs).gcd(rhs);
    return (ring.clone_monomial(rhs).div(&gcd), ring.clone_monomial(lhs).div(&gcd), ring.clone_monomial(lhs).mul(rhs).div(&gcd));
}

fn lt_lcm<'a, P>(ring: &P, lhs: LT_ref<'a, P>, rhs: LT_ref<'a, P>) -> (LT<P>, LT<P>, LT<P>)
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing
{
    let (coeff_s, coeff_t, coeff_lcm) = coeff_lcm(ring.base_ring(), lhs.0, rhs.0);
    let (mon_s, mon_t, mon_lcm) = mon_lcm(ring, lhs.1, rhs.1);
    return ((coeff_s, mon_s), (coeff_t, mon_t), (coeff_lcm, mon_lcm));
}

fn lt_divides<'a, P>(ring: &P, lhs: LT_ref<'a, P>, rhs: LT_ref<'a, P>) -> bool
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing
{
    lhs.1.divides(rhs.1) && ring.base_ring().checked_div(rhs.0, lhs.0).is_some()
}

fn p_valuation<R>(ring: R, p: &El<R>, mut val: El<R>) -> usize
    where R: RingStore,
        R::Type: DivisibilityRing
{
    assert!(!ring.is_zero(&val));
    let mut result = 0;
    while let Some(new) = ring.checked_div(&val, p) {
        val = new;
        result += 1;
    }
    return result;
}

#[inline(never)]
fn reduce_S_matrix<P, O>(ring: P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, S_polys: &mut Vec<SPoly>, basis: &[El<P>], order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore + Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        O: MonomialOrder + Copy,
        Coeff<P>: Send + Sync
{
    let mut columns: MonomialMap<P, O> = MonomialMap::new(order);

    let mut monomial_column = |m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>, open: &mut Vec<Monomial<_>>, A: &mut SparseMatrix<_>| match columns.add(ring.clone_monomial(m), ()) {
        AddResult::Present(i, ()) => i,
        AddResult::Added(i) => {
            A.add_col(i); 
            open.push(ring.clone_monomial(m)); 
            i
        }
    };

    // all monomials for which we have to add a row to A to enable eliminating them
    let mut open: Vec<Monomial<_>> = Vec::new();
    let mut A = algorithms::sparse_invert::matrix::SparseMatrix::new(&ring.base_ring());

    while let Some(S_poly) = S_polys.pop() {
        let S_poly = S_poly.poly(&ring, ring_info, basis, order);
        let i = A.row_count();
        A.add_zero_row(i);
        for (c, m) in ring.terms(&S_poly) {
            let j = monomial_column(m, &mut open, &mut A);
            A.set(i, j, ring.base_ring().clone_el(c));
        }
        while let Some(m) = open.pop() {
            if let Some(reducer) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m))
                .min_by_key(|f| p_valuation(ring.base_ring(), &ring_info.extended_ideal_generator, ring.base_ring().clone_el(ring.lt(f, order).unwrap().0)))
            {
                let div_monomial = ring.clone_monomial(&m).div(ring.lm(reducer, order).unwrap());
                let i = 0;
                A.add_zero_row(0);
                for (c, m) in ring.terms(&reducer) {
                    let j = monomial_column(&ring.clone_monomial(m).mul(&div_monomial), &mut open, &mut A);
                    A.set(i, j, ring.base_ring().clone_el(c));
                }
            }
        }
    }

    let entries = algorithms::sparse_invert::gb_sparse_row_echelon::<_, true>(ring.base_ring(), A, 256);

    let mut result = Vec::new();
    for i in 0..entries.len() {
        if let Some(j) = entries[i].iter().inspect(|(_, c)| assert!(!ring.base_ring().is_zero(c))).map(|(j, _)| *j).min() {
            if basis.iter().all(|f| !ring.lm(f, order).unwrap().divides(columns.at_index(j)) || ring.base_ring().checked_div(&entries[i][0].1, ring.lt(f, order).unwrap().0).is_none()) {
                let f = ring.from_terms(entries[i].iter().map(|(j, c)| (ring.base_ring().clone_el(c), ring.clone_monomial(columns.at_index(*j)))));
                if ring.is_zero(&f) {
                    println!();
                    ring.println(&f);
                    for (j, c) in &entries[i] {
                        print!("({}, {}, {}), ", j, ring.format(&ring.monomial(columns.at_index(*j))), ring.base_ring().format(c));
                    }
                    println!();
                    panic!()
                }
                result.push(f)
            }
        }
    }
    return result;
}

fn sym_tuple(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (b, a)
    } else {
        (a, b)
    }
}

///
/// Data associated to a ring that is required when computing Groebner basis.
/// 
pub struct RingInfo<R: ?Sized>
    where R: RingBase
{
    ring: PhantomData<R>,
    extended_ideal_generator: R::Element,
    annihilating_power: Option<usize>
}

impl<R: ?Sized + RingBase> RingInfo<R> {

    ///
    /// Creates a new [`RingInfo`].
    /// 
    /// It is currently only valid to create [`RingInfo`]s for discrete valuation rings.
    /// In this case, `extended_ideal_generator` should be a generator of the maximal ideal
    /// (or 0 if the ring is a field) and annihilating_power should be the smallest integer
    /// such that `extended_ideal_generator^annihilating_power == 0`. Obviously, this currently
    /// only works for finite rings.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(extended_ideal_generator: R::Element, annihilating_power: Option<usize>) -> Self {
        Self {
            extended_ideal_generator: extended_ideal_generator,
            annihilating_power: annihilating_power,
            ring: PhantomData
        }
    }
}

impl<R: ?Sized + RingBase> RingInfo<R> {

    fn required_s_polys<'a, P>(&'a self, _ring: &'a P, basis: &'a [El<P>]) -> impl 'a + Iterator<Item = SPoly>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = R>
    {
        (0..basis.len()).flat_map(|i| (0..i).map(move |j: usize| SPoly::Standard(i, j)))
            .chain((0..basis.len()).flat_map(|i| (1..self.annihilating_power.unwrap_or(0)).map(move |k: usize| SPoly::Nilpotent(i, k))))
    }
}

#[derive(PartialEq, Clone)]
enum SPoly {
    Standard(usize, usize), Nilpotent(/* poly index */ usize, /* power-of-p multiplier */ usize)
}

impl SPoly {

    fn is_zero<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> bool
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.expected_lt(ring, ring_info, basis, order).is_none()
    }

    fn expected_lm<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.expected_lt(ring, ring_info, basis, order).unwrap().1
    }

    fn expected_lt<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> Option<LT<P>>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        fn get_scaled_poly_largest_term_lt<P, O>(ring: P, poly: &El<P>, order: O, lt_than: Option<&Mon<P>>, scaling: &LT<P>) -> Option<LT<P>>
            where P: MultivariatePolyRingStore,
                P::Type: MultivariatePolyRing,
                <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
                O: MonomialOrder + Copy
        {
            let scale = |(c, mon): LT_ref<P>| (ring.base_ring().mul_ref(c, &scaling.0), ring.clone_monomial(mon).mul(&scaling.1));
            let lt_than = lt_than.map(|lt_than| ring.clone_monomial(lt_than).div(&scaling.1));
            let mut current = if let Some(lt_than) = lt_than {
                ring.get_ring().max_term_lt(poly, &lt_than, order)
            } else {
                ring.lt(poly, order)
            }?;
            while ring.base_ring().is_zero(&scale(current).0) {
                current = ring.get_ring().max_term_lt(poly, current.1, order)?;
            }
            return Some(scale(current))
        }

        let do_computation = || match self {
            SPoly::Standard(i, j) => {
                let (fi_factor, fj_factor, _lcm) = lt_lcm(ring, ring.lt(&basis[*i], order).unwrap(), ring.lt(&basis[*j], order).unwrap());

                let fi_lt = get_scaled_poly_largest_term_lt(&ring, &basis[*i], order, None, &fi_factor).unwrap();
                let fj_lt = get_scaled_poly_largest_term_lt(&ring, &basis[*j], order, None, &fj_factor).unwrap();
                assert!(order.compare(&fi_lt.1, &fj_lt.1) == Ordering::Equal);
                assert!(ring.base_ring().eq_el(&fi_lt.0, &fj_lt.0));
                
                let mut current_lm = fi_lt.1;
                loop {
                    let fi_candidate = get_scaled_poly_largest_term_lt(&ring, &basis[*i], order, Some(&current_lm), &fi_factor);
                    let fj_candidate = get_scaled_poly_largest_term_lt(&ring, &basis[*j], order, Some(&current_lm), &fj_factor);
                    match (fi_candidate, fj_candidate) {
                        (None, None) => { return None; }
                        (Some(res), None) => { return Some(res); },
                        (None, Some(res)) => { return Some((ring.base_ring().negate(res.0), res.1)); },
                        (Some(fi_candidate), Some(fj_candidate)) if order.compare(&fi_candidate.1, &fj_candidate.1) == Ordering::Less => { return Some((ring.base_ring().negate(fj_candidate.0), fj_candidate.1)); },
                        (Some(fi_candidate), Some(fj_candidate)) if order.compare(&fi_candidate.1, &fj_candidate.1) == Ordering::Greater => { return Some(fi_candidate); },
                        (Some(fi_candidate), Some(fj_candidate)) if !ring.base_ring().eq_el(&fi_candidate.0, &fj_candidate.0) => { return Some((ring.base_ring().sub(fi_candidate.0, fj_candidate.0), fi_candidate.1)); },
                        (Some(fi_candidate), Some(_fj_candidate)) => { current_lm = fi_candidate.1; }
                    }
                }
            },
            SPoly::Nilpotent(i, k) => {
                let multiplier = ring.base_ring().pow(ring.base_ring().clone_el(&ring_info.extended_ideal_generator), *k);
                let (mut lc, mut lm) = ring.lt(&basis[*i], order)?;
                while ring.base_ring().is_zero(&ring.base_ring().mul_ref(lc, &multiplier)) {
                    (lc, lm) = ring.get_ring().max_term_lt(&basis[*i], lm, order)?;
                }
                return Some((ring.base_ring().mul_ref(lc, &multiplier), ring.clone_monomial(lm)));
            }
        };
        let result = do_computation();
        assert!(result.is_none() || !ring.base_ring().is_zero(&result.as_ref().unwrap().0));
        debug_assert!(
            (result.is_none() && ring.is_zero(&self.poly(ring, ring_info, basis, order))) || 
            (ring.base_ring().eq_el(&result.as_ref().unwrap().0, ring.lt(&self.poly(ring, ring_info, basis, order), order).unwrap().0) && result.as_ref().unwrap().1 == *ring.lt(&self.poly(ring, ring_info, basis, order), order).unwrap().1)
        );
        return result;
    }

    fn expected_lc_valuation<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> usize
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        p_valuation(ring.base_ring(), &ring_info.extended_ideal_generator, self.expected_lt(ring, ring_info, basis, order).unwrap().0)
    }

    fn poly<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> El<P>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => {
                let (f1_factor, f2_factor, _) = lt_lcm(&ring, ring.lt(&basis[*i], order).unwrap(), ring.lt(&basis[*j], order).unwrap());
                let mut f1_scaled = ring.clone_el(&basis[*i]);
                ring.mul_monomial(&mut f1_scaled, &f1_factor.1);
                ring.inclusion().mul_assign_map(&mut f1_scaled, f1_factor.0);
                let mut f2_scaled = ring.clone_el(&basis[*j]);
                ring.mul_monomial(&mut f2_scaled, &f2_factor.1);
                ring.inclusion().mul_assign_map(&mut f2_scaled, f2_factor.0);
                return ring.sub(f1_scaled, f2_scaled);
            },
            SPoly::Nilpotent(i, k) => {
                let mut result = ring.clone_el(&basis[*i]);
                ring.inclusion().mul_assign_map(&mut result, ring.base_ring().pow(ring.base_ring().clone_el(&ring_info.extended_ideal_generator), *k));
                return result;
            }
        }
    }

    ///
    /// The chain criterion says that an S-poly of `f` and `g` reduces to zero, if there are
    /// `f0 = f, f1, f2, ..., fk = g` such that the leading terms of each `fi` divide `lcm(lt(f), lt(g))`
    /// and the S-polys of each pair `(fi, f(i + 1))` have already been considered (and possibly included
    /// in the basis).
    /// 
    /// For efficiency reasons, we only check length-3 chains, i.e. where `k = 2`.
    /// 
    #[inline(never)]
    fn filter_chain_criterion<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O, reduced_pairs: &[(usize, usize)]) -> bool
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => {
                let lcm = lt_lcm(ring, ring.lt(&basis[*i], order).unwrap(), ring.lt(&basis[*j], order).unwrap()).2;
                (0..basis.len())
                    .filter(|k| lt_divides(ring, ring.lt(&basis[*k], order).unwrap(), (&lcm.0, &lcm.1)))
                    .any(|k| reduced_pairs.binary_search(&sym_tuple(*i, k)).is_ok() && reduced_pairs.binary_search(&sym_tuple(k, *j)).is_ok())
            },
            SPoly::Nilpotent(i, k) => self.is_zero(ring, ring_info, basis, order) || (*k > 0 && self.expected_lm(&ring, ring_info, &basis, order) == SPoly::Nilpotent(*i, *k - 1).expected_lm(&ring, ring_info, &basis, order))
        }
    }

    ///
    /// The product criterion says that an S-poly of two polynomials with coprime leading terms always
    /// reduces to zero.
    /// 
    #[inline(never)]
    fn filter_product_criterion<P, O>(&self, ring: &P, _ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> bool
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => {
                let fi_lm = ring.lm(&basis[*i], order).unwrap();
                let fj_lm = ring.lm(&basis[*j], order).unwrap();
                fi_lm.is_coprime(fj_lm) && ring.base_ring().is_unit(&ring.base_ring().extended_ideal_gen(ring.coefficient_at(&basis[*i], fi_lm), ring.coefficient_at(&basis[*j], fj_lm)).2)
            },
            _ => false
        }
    }
}

pub trait GBRingDescriptorRing: Sized + PrincipalIdealRing {

    fn create_ring_info(&self) -> RingInfo<Self>;
}

fn ring_info_local_zn_ring<R>(ring: R) -> Option<RingInfo<R::Type>>
    where R: ZnRingStore,
        R::Type: ZnRing
{
    let factorization = algorithms::int_factor::factor(ring.integer_ring(), ring.integer_ring().clone_el(ring.modulus()));
    if factorization.len() > 1 {
        return None;
    }
    let (p, e) = factorization.into_iter().next().unwrap();
    return Some(RingInfo {
        annihilating_power: Some(e),
        ring: PhantomData,
        extended_ideal_generator: ring.can_hom(ring.integer_ring()).unwrap().map(p)
    });
}

impl<F: Field> GBRingDescriptorRing for F {

    fn create_ring_info(&self) -> RingInfo<Self> {
        RingInfo {
            annihilating_power: None,
            ring: PhantomData,
            extended_ideal_generator: self.zero()
        }
    }
}

impl GBRingDescriptorRing for zn_64::ZnBase {

    fn create_ring_info(&self) -> RingInfo<Self> {
        ring_info_local_zn_ring(RingRef::new(self)).expect("Currently GBRingDescriptorRing only works for local rings Z/nZ, i.e. Z/p^eZ with p prime")
    }
}

impl<const N: u64> GBRingDescriptorRing for zn_static::ZnBase<N, false> {

    fn create_ring_info(&self) -> RingInfo<Self> {
        ring_info_local_zn_ring(RingRef::new(self)).expect("Currently GBRingDescriptorRing only works for local rings Z/nZ, i.e. Z/p^eZ with p prime")
    }
}

impl<R> GBRingDescriptorRing for AsLocalPIRBase<R>
    where R: RingStore,
        R::Type: DivisibilityRing + FiniteRing
{
    fn create_ring_info(&self) -> RingInfo<Self> {
        let max_ideal_gen = self.max_ideal_gen();
        let annihilating_power = int_bisect::find_root_floor(&StaticRing::<i64>::RING, 0, |x| if *x < 0 || !self.is_zero(&RingRef::new(self).pow(self.clone_el(max_ideal_gen), *x as usize)) { -1 } else { 1 });
        RingInfo {
            annihilating_power: Some(annihilating_power as usize),
            ring: PhantomData,
            extended_ideal_generator: self.clone_el(max_ideal_gen)
        }
    }
}

///
/// A simple implementation of the F4 algorithm for computing Groebner basis.
/// This implementation cannot (yet ?) compete with highly optimized implementations 
/// (Singular, Macaulay2, Magma etc).
/// 
/// This algorithm will only consider S-polynomials with leading monomial of degree smaller than 
/// the given bound.
/// Ignoring S-polynomials this way might cause the resulting basis not to be a 
/// Groebner basis, but can drastically speed up computatations. If you are unsure which bound to 
/// use, set it to `u16::MAX` to get an actual GB.
/// 
/// Note that Groebner basis algorithms are still the subject of ongoing research, and
/// whether the Groebner basis of even a simple example can be efficiently computed is
/// hard to predict from the example itself. 
/// 
/// As an additional note, if you want to compute a GB w.r.t. a term ordering that is
/// not degrevlex, it might be faster to first compute a degrevlex-GB and use that as input
/// for a new invocation of f4.
/// 
/// # Example
/// 
/// ```
/// #![feature(generic_const_exprs)]
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::multivariate::*;
/// # use feanor_math::algorithms::f4::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::rings::zn::zn_static;
/// let order = DegRevLex;
/// let base = zn_static::F17;
/// let ring: ordered::MultivariatePolyRingImpl<_, _, 2> = ordered::MultivariatePolyRingImpl::new(base, order);
/// 
/// // the classical GB example: x^2 + y^2 - 1, xy - 2
/// let f1 = ring.from_terms([
///     (1, Monomial::new([2, 0])),
///     (1, Monomial::new([0, 2])),
///     (16, Monomial::new([0, 0]))
/// ].into_iter());
/// let f2 = ring.from_terms([
///     (1, Monomial::new([1, 1])),
///     (15, Monomial::new([0, 0]))
/// ].into_iter());
/// 
/// let gb = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order, u16::MAX);
/// 
/// let in_ideal = ring.from_terms([
///     (16, Monomial::new([0, 3])),
///     (15, Monomial::new([1, 0])),
///     (1, Monomial::new([0, 1])),
/// ].into_iter());
/// 
/// assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, in_ideal, gb.iter(), order));
/// ```
/// 
pub fn f4<P, O, const LOG: bool>(ring: P, mut basis: Vec<El<P>>, order: O, S_poly_degree_bound: u16) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: GBRingDescriptorRing,
        O: MonomialOrder + Copy,
        Coeff<P>: Send + Sync
{
    assert!(basis.iter().all(|f| !ring.is_zero(f)));
    
    let ring_info = ring.base_ring().get_ring().create_ring_info();

    basis = reduce(&ring, basis, order);

    let select = |s_poly: &SPoly, basis: &[El<P>], degree_bound: (u16, usize), filtered_out_degree: &mut bool, filtered_out_valuation: &mut bool| 
        if s_poly.expected_lm(&ring, &ring_info, basis, order).deg() > degree_bound.0 {
            *filtered_out_degree |= true;
            None
        } else if s_poly.expected_lc_valuation(&ring, &ring_info, basis, order) > degree_bound.1 {
            *filtered_out_valuation |= true;
            None
        } else {
            Some(s_poly.clone()) 
        };

    let update_degree_bound = |degree_bound: &mut (u16, usize), _filtered_out_degree: bool, _filtered_out_valuation: bool| {
        degree_bound.0 = min(degree_bound.0 + 5, S_poly_degree_bound);
        degree_bound.1 += 1;
    };

    let mut chain_criterion_reduced_pairs = Vec::new();
    let mut product_criterion_skipped = 0;
    let mut chain_criterion_skipped = 0;

    let mut open = ring_info.required_s_polys(&ring, &basis[..]).collect::<Vec<_>>();

    let mut degree_bound = (1, 0);
    while open.len() > 0 {
        if LOG {
            print!("S({})", open.len());
            std::io::stdout().flush().unwrap();
        }

        let mut new_reduced_pairs = Vec::new();
        let mut S_polys = Vec::new();
        let mut filtered_out_degree = false;
        let mut filtered_out_valuation = false;

        open.retain(|S_poly| {
            if S_poly.is_zero(&ring, &ring_info, &basis, order) {
                false
            } else if S_poly.filter_product_criterion(&ring, &ring_info, &basis[..], order) {
                product_criterion_skipped += 1;
                false
            } else if S_poly.filter_chain_criterion(&ring, &ring_info, &basis[..], order, &chain_criterion_reduced_pairs[..]) {
                chain_criterion_skipped += 1;
                false
            } else if let Some(poly) = select(S_poly, &basis[..], degree_bound, &mut filtered_out_degree, &mut filtered_out_valuation) { 
                S_polys.push(poly);
                if let SPoly::Standard(i, j) = S_poly {
                    new_reduced_pairs.push((*i, *j));
                }
                false
            } else {
                true
            }
        });

        let mut new_polys: Vec<_> = Vec::new();
        // usually, reduce_S_matrix will consume all S_polys and there will only be one loop execution;
        // however, if there is a danger of not fitting the S matrix into memory, we will split it. Note
        // that the performance will degrade, as we cannot reduce all S polys with each other
        while S_polys.len() > 10 {
            new_polys.extend(reduce_S_matrix(&ring, &ring_info, &mut S_polys, &basis, order));
        }
        if S_polys.len() > 0 {
            let start = std::time::Instant::now();
            new_polys.extend(S_polys.into_iter().map(|f| multivariate_division(&ring, f.poly(&ring, &ring_info, &basis, order), basis.iter(), order)).filter(|f| !ring.is_zero(f)));
            let end = std::time::Instant::now();
            if LOG {
                print!("[{}ms]", (end - start).as_millis());
                std::io::stdout().flush().unwrap();
            }
        }

        chain_criterion_reduced_pairs.extend(new_reduced_pairs.into_iter());
        chain_criterion_reduced_pairs.sort_unstable();

        if new_polys.len() == 0 {
            if degree_bound.0 == S_poly_degree_bound && degree_bound.1 >= ring_info.annihilating_power.unwrap_or(0) {
                if LOG {
                    println!();
                    println!("S-poly degree bound exceeded, aborting GB computation");
                    println!("Redundant S-pairs: {} (prod), {} (chain)", product_criterion_skipped, chain_criterion_skipped);
                }
                return basis;
            }

            update_degree_bound(&mut degree_bound, filtered_out_degree, filtered_out_valuation);

            if LOG {
                print!("{{{:?}}}", degree_bound);
                std::io::stdout().flush().unwrap();
            }
        } else {

            degree_bound = (1, 0);
            basis.extend(new_polys.into_iter());
            basis = reduce(&ring, basis, order);
            chain_criterion_reduced_pairs = Vec::new();

            open = ring_info.required_s_polys(&ring, &basis[..]).collect::<Vec<_>>();
    
            if LOG {
                print!("b({})", basis.len());
                std::io::stdout().flush().unwrap();
            }
        }
    }
    if LOG {
        println!();
        println!("Redundant S-pairs: {} (prod), {} (chain)", product_criterion_skipped, chain_criterion_skipped);
    }
    return basis;
}

#[stability::unstable(feature = "enable")]
pub fn reduce<P, O>(ring: P, mut polys: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: DivisibilityRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy
{
    let mut changed = true;
    while changed {
        changed = false;
        polys.sort_by(|l, r| order.compare(ring.lm(l, order).unwrap(), ring.lm(r, order).unwrap()));
        let mut i = 0;
        while i < polys.len() {
            let reduced = multivariate_division(&ring, ring.clone_el(&polys[i]), polys[..i].iter().chain(polys[(i + 1)..].iter()), order);
            if !ring.is_zero(&reduced) {
                if !ring.eq_el(&reduced, &polys[i]) {
                    changed = true;
                    polys[i] = reduced;
                }
                i += 1;
            } else {
                polys.remove(i);
                changed = true;
            }
        }
    }
    for b1 in &polys {
        for b2 in &polys {
            if b1 as *const _ != b2 as *const _ {
                let b1_lm = ring.lm(b1, order).unwrap();
                let b2_lm = ring.lm(b2, order).unwrap();
                assert!(!b1_lm.divides(b2_lm) || ring.base_ring().checked_div(ring.coefficient_at(b2, b2_lm), ring.coefficient_at(b1, b1_lm)).is_none());
            }
        }
    }
    return polys;
}

#[stability::unstable(feature = "enable")]
pub fn multivariate_division<'a, P, I, O>(ring: P, mut f: El<P>, set: I, order: O) -> El<P>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: DivisibilityRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        El<P>: 'a,
        I: Clone + Iterator<Item = &'a El<P>>
{
    if ring.is_zero(&f) {
        return f;
    }
    let mut f_lm = ring.clone_monomial(ring.lm(&f, order).unwrap());
    let mut f_lc = ring.base_ring().clone_el(ring.coefficient_at(&f, &f_lm));
    let incl = ring.inclusion();
    let mut changed = true;
    while changed {
        changed = false;
        while let Some((quo, g)) = set.clone()
            .filter(|g| ring.lm(g, order).unwrap().divides(&f_lm))
            .filter_map(|g| ring.base_ring().checked_div(&f_lc, ring.coefficient_at(g, ring.lm(g, order).unwrap())).map(|quo| (quo, g))).next()
        {
            changed = true;
            let g_lm = ring.lm(g, order).unwrap();
            let div_monomial = f_lm.div(&g_lm);
            let mut g_scaled = ring.clone_el(g);
            ring.mul_monomial(&mut g_scaled, &div_monomial);
            incl.mul_assign_map_ref(&mut g_scaled, &quo);
            ring.sub_assign(&mut f, g_scaled);
            if let Some(m) = ring.lm(&f, order) {
                f_lm = ring.clone_monomial(m);
                f_lc = ring.base_ring().clone_el(ring.coefficient_at(&f, &f_lm))
            } else {
                return f;
            }
        }
    }
    return f;
}

#[cfg(test)]
use crate::rings::multivariate::ordered::*;
#[cfg(test)]
use crate::rings::poly::*;
#[cfg(test)]
use crate::seq::*;

use super::int_bisect;
use super::sparse_invert::matrix::SparseMatrix;

#[test]
fn test_f4_small() {
    let order = DegRevLex;
    let base: RingValue<zn_static::ZnBase<17, true>> = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, 2> = MultivariatePolyRingImpl::new(base, order);

    let f1 = ring.from_terms([
        (1, Monomial::new([2, 0])),
        (1, Monomial::new([0, 2])),
        (16, Monomial::new([0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, Monomial::new([1, 1])),
        (15, Monomial::new([0, 0]))
    ].into_iter());

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order, u16::MAX);

    let expected = ring.from_terms([
        (16, Monomial::new([0, 3])),
        (15, Monomial::new([1, 0])),
        (1, Monomial::new([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(ring, f2, actual.at(0));
    assert_el_eq!(ring, f1, actual.at(1));
    assert_el_eq!(ring, ring.negate(expected), actual.at(2));
}

#[test]
fn test_f4_larger() {
    let order = DegRevLex;
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(base, order);

    let f1 = ring.from_terms([
        (1, Monomial::new([2, 1, 1])),
        (1, Monomial::new([0, 2, 0])),
        (1, Monomial::new([1, 0, 1])),
        (2, Monomial::new([1, 0, 0])),
        (1, Monomial::new([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, Monomial::new([0, 3, 1])),
        (1, Monomial::new([0, 0, 3])),
        (1, Monomial::new([1, 1, 0]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, Monomial::new([1, 0, 2])),
        (1, Monomial::new([1, 0, 1])),
        (2, Monomial::new([0, 1, 1])),
        (7, Monomial::new([0, 0, 0]))
    ].into_iter());

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order, u16::MAX);

    let g1 = ring.from_terms([
        (1, Monomial::new([0, 4, 0])),
        (8, Monomial::new([0, 3, 1])),
        (12, Monomial::new([0, 1, 3])),
        (6, Monomial::new([0, 0, 4])),
        (1, Monomial::new([0, 3, 0])),
        (13, Monomial::new([0, 2, 1])),
        (11, Monomial::new([0, 1, 2])),
        (10, Monomial::new([0, 0, 3])),
        (11, Monomial::new([0, 2, 0])),
        (12, Monomial::new([0, 1, 1])),
        (6, Monomial::new([0, 0, 2])),
        (6, Monomial::new([0, 1, 0])),
        (13, Monomial::new([0, 0, 1])),
        (9, Monomial::new([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, g1, actual.iter(), order));
}

#[test]
fn test_f4_larger_elim() {
    let order = BlockLexDegRevLex::new(..1, 1..);
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, 3> = MultivariatePolyRingImpl::new(base, order);

    let f1 = ring.from_terms([
        (1, Monomial::new([2, 1, 1])),
        (1, Monomial::new([0, 2, 0])),
        (1, Monomial::new([1, 0, 1])),
        (2, Monomial::new([1, 0, 0])),
        (1, Monomial::new([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, Monomial::new([1, 1, 0])),
        (1, Monomial::new([0, 3, 1])),
        (1, Monomial::new([0, 0, 3]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, Monomial::new([1, 0, 2])),
        (1, Monomial::new([1, 0, 1])),
        (2, Monomial::new([0, 1, 1])),
        (7, Monomial::new([0, 0, 0]))
    ].into_iter());

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order, u16::MAX);

    let g1 = ring.from_terms([
        (1, Monomial::new([0, 4, 0])),
        (8, Monomial::new([0, 3, 1])),
        (12, Monomial::new([0, 1, 3])),
        (6, Monomial::new([0, 0, 4])),
        (1, Monomial::new([0, 3, 0])),
        (13, Monomial::new([0, 2, 1])),
        (11, Monomial::new([0, 1, 2])),
        (10, Monomial::new([0, 0, 3])),
        (11, Monomial::new([0, 2, 0])),
        (12, Monomial::new([0, 1, 1])),
        (6, Monomial::new([0, 0, 2])),
        (6, Monomial::new([0, 1, 0])),
        (13, Monomial::new([0, 0, 1])),
        (9, Monomial::new([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(ring, ring.zero(), multivariate_division(&ring, g1, actual.iter(), order));
}

#[test]
fn test_gb_local_ring() {
    let order = DegRevLex;
    let base = zn_static::Zn::<16>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 1> = MultivariatePolyRingImpl::new(base, order);
    
    let f = ring.from_terms([(4, Monomial::new([1])), (1, Monomial::new([0]))].into_iter());
    let gb = f4::<_, _, true>(&ring, vec![f], order, u16::MAX);

    assert_eq!(1, gb.len());
    assert_el_eq!(ring, ring.one(), gb[0]);
}

#[test]
fn test_generic_computation() {
    let order = DegRevLex;
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, 6> = MultivariatePolyRingImpl::new(base, order);
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let X1 = poly_ring.mul(
        poly_ring.from_terms([(ring.indeterminate(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(ring.indeterminate(1), 0), (ring.one(), 1)].into_iter())
    );
    let X2 = poly_ring.mul(
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(ring.indeterminate(2), 0), (ring.indeterminate(3), 1)].into_iter())),
        poly_ring.add(poly_ring.clone_el(&X1), poly_ring.from_terms([(ring.indeterminate(4), 0), (ring.indeterminate(5), 1)].into_iter()))
    );
    let basis = vec![
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 0)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 1)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = f4::<_, _, true>(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), order, u16::MAX);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(11, gb1.len());
}

#[ignore]
#[test]
fn test_expensive_gb_1() {
    let order = DegRevLex;
    let base = zn_static::Zn::<16>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 12> = MultivariatePolyRingImpl::new(base, order);

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
    let gb = f4::<_, _, true>(&ring, system, order, u16::MAX);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    for f in &part_of_result {
        assert!(ring.is_zero(&multivariate_division(&ring, ring.clone_el(f), gb.iter(), order)));
    }

    assert_eq!(93, gb.len());
}

#[test]
#[ignore]
fn test_expensive_gb_2() {
    let order = DegRevLex;
    let base = zn_static::Fp::<7>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 7> = MultivariatePolyRingImpl::new(base, order);

    let basis = ring.with_wrapped_indeterminates(|[X0, X1, X2, X3, X4, X5, X6]| [
        6 + 2 * X5 + 2 * X4 + X6 + 4 * X0 + 5 * X6 * X5 + X6 * X4 + 3 * X0 * X4 + 6 * X0 * X6 + 2 * X0 * X3 + X0 * X2 + 4 * X0 * X1 + 2 * X3 * X4 * X5 + 4 * X0 * X6 * X5 + 6 * X0 * X2 * X5 + 5 * X0 * X6 * X4 + 2 * X0 * X3 * X4 + 4 * X0 * X1 * X4 + X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 5 * X0 * X2 * X6 + 2 * X0 * X1 * X6 + X0 * X3.pow_ref(2) + 2 * X0 * X2 * X3 + 3 * X0 * X3 * X4 * X5 + 4 * X0 * X3 * X6 * X5 + 3 * X0 * X1 * X6 * X5 + 3 * X0 * X2 * X3 * X5 + 3 * X0 * X3 * X6 * X4 + 2 * X0 * X1 * X6 * X4 + 2 * X0 * X3.pow_ref(2) * X4 + 2 * X0 * X2 * X3 * X4 + 3 * X0 * X3.pow_ref(2) * X4 * X5 + 4 * X0 * X1 * X3 * X4 * X5 + X0 * X3.pow_ref(2) * X4.pow_ref(2),
        5 + 4 * X0 + 6 * X4 * X5 + 3 * X6 * X5 + 4 * X0 * X4 + 3 * X0 * X6 + 6 * X0 * X3 + 6 * X0 * X2 + 6 * X6 * X4 * X5 + 2 * X0 * X4 * X5 + 4 * X0 * X6 * X5 + 3 * X0 * X2 * X5 + 3 * X0 * X6 * X4 + 5 * X0 * X3 * X4 + 6 * X0 * X2 * X4 + 4 * X0 * X6.pow_ref(2) + 3 * X0 * X3 * X6 + 3 * X0 * X2 * X6 + 2 * X0 * X6 * X4 * X5 + 6 * X0 * X3 * X4 * X5 + 5 * X0 * X1 * X4 * X5 + 6 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 2 * X0 * X2 * X6 * X5 + 6 * X0 * X1 * X6 * X5 + 6 * X0 * X2 * X3 * X5 + 6 * X0 * X3 * X4.pow_ref(2) + 4 * X0 * X6.pow_ref(2) * X4 + 6 * X0 * X3 * X6 * X4 + 3 * X0 * X2 * X6 * X4 + 4 * X0 * X3 * X6 * X4 * X5 + 5 * X0 * X1 * X6 * X4 * X5 + 6 * X0 * X3.pow_ref(2) * X4 * X5 + 5 * X0 * X2 * X3 * X4 * X5 + 3 * X0 * X3 * X6 * X4.pow_ref(2) + 6 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.clone(),
        2 + 2 * X0 + 4 * X0 * X4 + 2 * X0 * X6 + 5 * X0 * X4 * X5 + 2 * X0 * X6 * X5 + 4 * X0 * X2 * X5 + 2 * X0 * X4.pow_ref(2) + 4 * X0 * X6 * X4 + 4 * X0 * X6.pow_ref(2) + 2 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X6 * X4 * X5 + X0 * X3 * X4 * X5 + X0 * X2 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 2 * X0 * X3 * X6 * X5 + 4 * X0 * X2 * X6 * X5 + 2 * X0 * X6 * X4.pow_ref(2) + X0 * X6.pow_ref(2) * X4 + 3 * X0 * X6 * X4 * X5.pow_ref(2) + 2 * X0 * X6.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X2 * X6 * X5.pow_ref(2) + X0 * X3 * X4.pow_ref(2) * X5 + X0 * X6.pow_ref(2) * X4 * X5 + X0 * X3 * X6 * X4 * X5 + 6 * X0 * X2 * X6 * X4 * X5 + 4 * X0 * X6.pow_ref(2) * X4.pow_ref(2) + 6 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X1 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X3 * X4 * X5.pow_ref(2) + 6 * X0 * X3 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 5 * X0 * X4 * X5 + 6 * X0 * X6 * X5 + 5 * X0 * X4.pow_ref(2) * X5 + 3 * X0 * X6 * X4 * X5 + 3 * X0 * X6.pow_ref(2) * X5 + 6 * X0 * X6 * X4 * X5.pow_ref(2) + 5 * X0 * X2 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X2 * X6 * X5.pow_ref(2) + 4 * X0 * X6 * X4.pow_ref(2) * X5 + 2 * X0 * X6.pow_ref(2) * X4 * X5 + 5 * X0 * X3 * X4.pow_ref(2) * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 5 * X0 * X3 * X6 * X4 * X5.pow_ref(2) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5 + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(2),
        4 + 4 * X0 * X4.pow_ref(2) * X5.pow_ref(2) + X0 * X6 * X4 * X5.pow_ref(2) + X0 * X6.pow_ref(2) * X5.pow_ref(2) + 5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(2) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2) + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2) + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(3),
        5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(3) + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3) + 5 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(3),
        2 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(4)
    ]);

    let start = std::time::Instant::now();
    let gb = f4::<_, _, true>(ring, basis, order, u16::MAX);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
}

#[test]
#[ignore]
fn test_expensive_gb_3_incomplete() {
    let order = DegRevLex;
    let base = zn_static::Zn::<32768>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 14> = MultivariatePolyRingImpl::new(base, order);

    let basis = ring.with_wrapped_indeterminates(|[X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13]| [
        22528 + 18432 * X3 * X8 * X13,
        2560 + 10752 * X3 * X8 * X13 + 19200 * X2 * X8 * X13 + 19200 * X3 * X7 * X13,
        384 + 18304 * X3 * X8 * X13 + 7744 * X2 * X8 * X13 + 1728 * X1 * X8 * X13 + 7744 * X3 * X7 * X13 + 1728 * X3 * X6 * X13,
        29568 + 14208 * X3 * X8 * X13 + 10368 * X2 * X8 * X13 + 31776 * X1 * X8 * X13 + 10368 * X3 * X7 * X13 + 10880 * X2 * X7 * X13 + 13216 * X1 * X7 * X13 + 31776 * X3 * X6 * X13 + 13216 * X2 * X6 * X13,
        17920 + 27536 * X12 + 27536 * X2 * X9 * X13 + 20992 * X3 * X8 * X13 + 27504 * X2 * X8 * X13 + 15320 * X1 * X8 * X13 + 27536 * X4 * X7 * X13 + 27504 * X3 * X7 * X13 + 28128 * X2 * X7 * X13 + 1864 * X1 * X7 * X13 + 15320 * X3 * X6 * X13 + 1864 * X2 * X6 * X13 + 27536 * X1 * X6 * X13,
        11136 + 16608 * X12 + 31242 * X10 + 16608 * X2 * X9 * X13 + 31242 * X0 * X9 * X13 + 17536 * X3 * X8 * X13 + 6400 * X2 * X8 * X13 + 30656 * X1 * X8 * X13 + 16608 * X4 * X7 * X13 + 6400 * X3 * X7 * X13 + 27520 * X2 * X7 * X13 + 960 * X1 * X7 * X13 + 30656 * X3 * X6 * X13 + 960 * X2 * X6 * X13 + 16608 * X1 * X6 * X13 + 31242 * X4 * X5 * X13,
        1024 + 28816 * X12 + 14470 * X10 + 28816 * X2 * X9 * X13 + 14470 * X0 * X9 * X13 + 5632 * X3 * X8 * X13 + 10416 * X2 * X8 * X13 + 24248 * X1 * X8 * X13 + 4336 * X0 * X8 * X13 + 28816 * X4 * X7 * X13 + 10416 * X3 * X7 * X13 + 27232 * X2 * X7 * X13 + 8296 * X1 * X7 * X13 + 24248 * X3 * X6 * X13 + 8296 * X2 * X6 * X13 + 28816 * X1 * X6 * X13 + 14470 * X4 * X5 * X13 + 4336 * X3 * X5 * X13,
        21632 + 21064 * X12 + 18806 * X10 + 21064 * X2 * X9 * X13 + 18806 * X0 * X9 * X13 + 8320 * X3 * X8 * X13 + 9848 * X2 * X8 * X13 + 11500 * X1 * X8 * X13 + 4908 * X0 * X8 * X13 + 21064 * X4 * X7 * X13 + 9848 * X3 * X7 * X13 + 28656 * X2 * X7 * X13 + 1124 * X1 * X7 * X13 + 27972 * X0 * X7 * X13 + 11500 * X3 * X6 * X13 + 1124 * X2 * X6 * X13 + 21064 * X1 * X6 * X13 + 18806 * X4 * X5 * X13 + 4908 * X3 * X5 * X13 + 27972 * X2 * X5 * X13,
        768 + 11720 * X12 + 12877 * X10 + 11720 * X2 * X9 * X13 + 12877 * X0 * X9 * X13 + 12032 * X3 * X8 * X13 + 17912 * X2 * X8 * X13 + 29388 * X1 * X8 * X13 + 30515 * X0  * X8 * X13 + 11720 * X4 * X7 * X13 + 17912 * X3 * X7 * X13 + 22384 * X2 * X7 * X13 + 30916 * X1 * X7 * X13 + 4795 * X0 * X7 * X13 + 29388 * X3 * X6 * X13 + 30916 * X2 * X6 * X13 + 11720 * X1 * X6 * X13 + 32767 * X0 * X6 * X13 + 12877 * X4 * X5 * X13 + 30515 * X3 * X5 * X13 + 4795 * X2 * X5 * X13 + 32767 * X1 * X5 * X13,
        29440 + 6968 * X12 + 6970 * X11 + 14875 * X10 + 6968 * X2 * X9 * X13 + 6970 * X1 * X9 * X13 + 14875 * X0 * X9 * X13 + 24832 * X3 * X8 * X13 + 25904 * X2 * X8 * X13 + 20792 * X1 * X8 * X13 + 11290 * X0 * X8 * X13 + 6968 * X4 * X7 * X13 + 25904 * X3 * X7 * X13 + 6368 * X2 * X7 * X13 + 6920 * X1 * X7 * X13 + 25786 * X0 * X7 * X13 + 6970 * X4 * X6 * X13 + 20792 * X3 * X6 * X13 + 6920 * X2 * X6 * X13 + 6968 * X1 * X6 * X13 + 25798 * X0 * X6 * X13 + 14875 * X4 * X5 * X13 + 11290 * X3 * X5 * X13 + 25786 * X2 * X5 * X13 + 25798 * X1 * X5 * X13 + 6970 * X0 * X5 * X13,
    ]);

    let start = std::time::Instant::now();
    let gb = f4::<_, _, true>(ring, basis, order, 8);
    let end = std::time::Instant::now();

    println!("Computed incomplete GB in {} ms", (end - start).as_millis());
    std::hint::black_box(gb);
}