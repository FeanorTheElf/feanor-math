use std::io::Write;
use std::marker::PhantomData;
use std::cmp::min;

use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::ring::*;
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

    #[allow(unused)]
    fn remove(&mut self, m: &Mon<P>) -> Option<K> {
        match self.data.binary_search_by(|(x, _)| self.order.compare(x, m).reverse()) {
            Ok(i) => {
                let (_, result) = self.data.remove(i);
                return Some(result);
            },
            Err(_) => return None
        }
    }

    #[allow(unused)]
    fn contains<'a>(&'a self, m: &Mon<P>) -> Option<&'a K> {
        match self.data.binary_search_by(|(x, _)| self.order.compare(x, m).reverse()) {
            Ok(i) => Some(&self.data[i].1),
            Err(_) => None
        }
    }

    fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a (Mon<P>, K)> {
        self.data.iter()
    }

    fn at_index(&self, i: usize) -> &Mon<P> {
        &self.data[i].0
    }

    fn index_of(&self, m: &Mon<P>) -> Option<usize> {
        self.data.binary_search_by(|(x, _)| self.order.compare(x, &m).reverse()).ok()
    }

    fn len(&self) -> usize {
        self.data.len()
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

fn reduce_S_matrix<P, O>(ring: P, p: &Coeff<P>, S_polys: &[El<P>], basis: &[El<P>], order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore + Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        O: MonomialOrder + Copy,
        Coeff<P>: Send + Sync
{
    if S_polys.iter().all(|S_poly| ring.is_zero(S_poly)) {
        return Vec::new();
    }

    let ring_ref = &ring;
    let mut columns: MonomialMap<P, O> = MonomialMap::new(order);
    for b in S_polys {
        for (_, b_m) in ring.terms(b) {
            columns.add(ring.clone_monomial(&b_m), ());
        }
    }
    let columns_ref = &columns;
    let mut A = algorithms::sparse_invert::matrix::SparseMatrix::new(&ring.base_ring());
    for j in 0..columns.len() {
        A.add_col(j);
    }
    for (i, S_poly) in S_polys.iter().enumerate() {
        A.add_row(i, ring.terms(S_poly).map(move |(c, m)| {
            let col = columns_ref.index_of(m).unwrap();
            return (col, ring_ref.base_ring().clone_el(c));
        }));
    }

    // all monomials for which we have to add a row to A to enable eliminating them
    let mut open: Vec<Monomial<_>> = columns.iter().map(|(m, _)| ring.clone_monomial(m)).collect();
    
    while let Some(m) = open.pop() {
        if let Some(f) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m))
            .min_by_key(|f| p_valuation(ring.base_ring(), p, ring.base_ring().clone_el(ring.lt(f, order).unwrap().0)))
        {
            let div_monomial = ring.clone_monomial(&m).div(ring.lm(f, order).unwrap());
            A.add_zero_row(0);
            for (c, f_m) in ring.terms(f) {
                let final_monomial = ring.clone_monomial(&div_monomial).mul(f_m);
                let col = match columns.add(ring.clone_monomial(&final_monomial), ()) {
                    AddResult::Added(i) => { A.add_col(i); open.push(final_monomial); i },
                    AddResult::Present(i, _) => i
                };

                A.set(0, col, ring.base_ring().clone_el(c));
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

#[derive(PartialEq)]
enum SPoly {
    Standard(usize, usize), Nilpotent(usize, usize)
}

impl SPoly {

    ///
    /// The terms are not ordered, and may contain the same monomial multiple times!
    /// 
    fn terms<'a, P, O>(&'a self, ring: &'a P, ring_info: &'a RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &'a [El<P>], order: O) -> Box<dyn 'a + Iterator<Item = (El<<P::Type as RingExtension>::BaseRing>, Mon<P>)>>
        where P: 'a + MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        match self {
            SPoly::Standard(i, j) => {
                let (f1_factor, f2_factor, lcm) = lt_lcm(ring, ring.lt(&basis[*i], order).unwrap(), ring.lt(&basis[*j], order).unwrap());
                return Box::new(ring.terms(&basis[*i])
                    .map(move |(c, m)| (ring.base_ring().mul_ref(c, &f1_factor.0), ring.clone_monomial(m).mul(&f1_factor.1)))
                    .chain(
                        ring.terms(&basis[*j])
                            .map(move |(c, m)| (ring.base_ring().negate(ring.base_ring().mul_ref(c, &f2_factor.0)), ring.clone_monomial(m).mul(&f2_factor.1))))
                    .filter(move |(_, m)| *m != lcm.1)
                    .filter(move |(c, _)| !ring.base_ring().is_zero(c)));
            },
            SPoly::Nilpotent(i, k) => {
                assert!(ring_info.annihilating_power.is_some());
                let factor = ring.base_ring().pow(ring.base_ring().clone_el(&ring_info.extended_ideal_generator), *k);
                return Box::new(ring.terms(&basis[*i]).map(move |(c, m)| (ring.base_ring().mul_ref(c, &factor), ring.clone_monomial(m))).filter(move |(c, _)| !ring.base_ring().is_zero(c)));
            }
        }
    }
    
    fn is_zero<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> bool
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.terms(ring, ring_info, basis, order).next().is_none()
    }

    fn expected_max_deg<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> u16
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.terms(ring, ring_info, basis, order).map(|(_, m)| m.deg()).max().unwrap()
    }

    fn expected_lm<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.terms(ring, ring_info, basis, order).max_by(|(_, l), (_, r)| order.compare(l, r)).map(|(_, m)| m).unwrap()
    }

    fn expected_lc_valuation<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> usize
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        self.terms(ring, ring_info, basis, order).max_by(|(_, l), (_, r)| order.compare(l, r)).map(|(c, _)| p_valuation(ring.base_ring(), &ring_info.extended_ideal_generator, c)).unwrap()
    }

    fn poly<P, O>(&self, ring: &P, ring_info: &RingInfo<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>, basis: &[El<P>], order: O) -> El<P>
        where P: MultivariatePolyRingStore,
            P::Type: MultivariatePolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
            O: MonomialOrder + Copy
    {
        let result = ring.from_terms(self.terms(ring, ring_info, basis, order));
        match self {
            SPoly::Standard(i, j) => {
                let (f1_factor, f2_factor, _) = lt_lcm(&ring, ring.lt(&basis[*i], order).unwrap(), ring.lt(&basis[*j], order).unwrap());
                let mut f1_scaled = ring.clone_el(&basis[*i]);
                ring.mul_monomial(&mut f1_scaled, &f1_factor.1);
                ring.inclusion().mul_assign_map(&mut f1_scaled, f1_factor.0);
                let mut f2_scaled = ring.clone_el(&basis[*j]);
                ring.mul_monomial(&mut f2_scaled, &f2_factor.1);
                ring.inclusion().mul_assign_map(&mut f2_scaled, f2_factor.0);
                assert_el_eq!(ring, result, ring.sub(f1_scaled, f2_scaled));
            },
            _ => {}
        }
        return result;
    }

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

impl<F: Field> GBRingDescriptorRing for F {

    fn create_ring_info(&self) -> RingInfo<Self> {
        RingInfo {
            annihilating_power: None,
            ring: PhantomData,
            extended_ideal_generator: self.zero()
        }
    }
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

///
/// A simple implementation of the F4 algorithm for computing Groebner basis.
/// This implementation cannot (yet ?) compete with highly optimized implementations 
/// (Singular, Macaulay2, Magma etc).
/// 
/// This algorithm will only consider S-polynomials of degree smaller than the given bound.
/// Ignoring S-polynomials this way might cause the resulting basis not to be a Groebner basis,
/// but can drastically speed up computatations. If you are unsure which bound to use, set
/// it to `u16::MAX` to get an actual GB.
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

    let select = |s_poly: &SPoly, basis: &[El<P>], degree_bound: (u16, usize)| 
        if s_poly.expected_max_deg(&ring, &ring_info, basis, order) < degree_bound.0 && s_poly.expected_lc_valuation(&ring, &ring_info, basis, order) <= degree_bound.1 {
            Some(s_poly.poly(&ring, &ring_info, basis, order))
        } else {
            None 
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

        open.retain(|S_poly| {
            if S_poly.is_zero(&ring, &ring_info, &basis, order) {
                false
            } else if S_poly.filter_product_criterion(&ring, &ring_info, &basis[..], order) {
                product_criterion_skipped += 1;
                false
            } else if S_poly.filter_chain_criterion(&ring, &ring_info, &basis[..], order, &chain_criterion_reduced_pairs[..]) {
                chain_criterion_skipped += 1;
                false
            } else if let Some(poly) = select(S_poly, &basis[..], degree_bound) { 
                S_polys.push(poly);
                if let SPoly::Standard(i, j) = S_poly {
                    new_reduced_pairs.push((*i, *j));
                }
                false
            } else {
                true
            }
        });

        let new_polys: Vec<_> = if S_polys.len() > 0 {
            
            if S_polys.len() > 20 {
                reduce_S_matrix(&ring, &ring_info.extended_ideal_generator, &S_polys, &basis, order)
            } else {
                let start = std::time::Instant::now();
                let result = S_polys.into_iter().map(|f| multivariate_division(&ring, f, basis.iter(), order)).filter(|f| !ring.is_zero(f)).collect();
                let end = std::time::Instant::now();
                if LOG {
                    print!("[{}ms]", (end - start).as_millis());
                    std::io::stdout().flush().unwrap();
                }
                result
            }

        } else {
            Vec::new()
        };

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
            degree_bound.0 = min(degree_bound.0 + 5, S_poly_degree_bound);
            degree_bound.1 += 1;
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
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::wrapper::RingElementWrapper;
#[cfg(test)]
use crate::seq::*;

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
fn test_gb_local_ring_large() {
    let order = DegRevLex;
    let base = zn_static::Zn::<16>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 12> = MultivariatePolyRingImpl::new(base, order);

    let Y0 = RingElementWrapper::new(&ring, ring.indeterminate(0));
    let Y1 = RingElementWrapper::new(&ring, ring.indeterminate(1));
    let Y2 = RingElementWrapper::new(&ring, ring.indeterminate(2));
    let Y3 = RingElementWrapper::new(&ring, ring.indeterminate(3));
    let Y4 = RingElementWrapper::new(&ring, ring.indeterminate(4));
    let Y5 = RingElementWrapper::new(&ring, ring.indeterminate(5));
    let Y6 = RingElementWrapper::new(&ring, ring.indeterminate(6));
    let Y7 = RingElementWrapper::new(&ring, ring.indeterminate(7));
    let Y8 = RingElementWrapper::new(&ring, ring.indeterminate(8));
    let Y9 = RingElementWrapper::new(&ring, ring.indeterminate(9));
    let Y10 = RingElementWrapper::new(&ring, ring.indeterminate(10));
    let Y11 = RingElementWrapper::new(&ring, ring.indeterminate(11));
    let scalar = |x: i32| RingElementWrapper::new(&ring, ring.int_hom().map(x));

    let system = [
        Y0.clone() * Y1.clone() * Y2.clone() * Y3.clone().pow(2) * Y4.clone().pow(2) + scalar(4) * Y0.clone() * Y1.clone() * Y2.clone() * Y3.clone() * Y4.clone() * Y5.clone() * Y8.clone() + Y0.clone() * Y1.clone() * Y2.clone() * Y5.clone().pow(2) * Y8.clone().pow(2) + Y0.clone() * Y2.clone() * Y3.clone() * Y4.clone() * Y6.clone() + Y0.clone() * Y1.clone() * Y3.clone() * Y4.clone() * Y7.clone() + Y0.clone() * Y2.clone() * Y5.clone() * Y6.clone() * Y8.clone() + Y0.clone() * Y1.clone() * Y5.clone() * Y7.clone() * Y8.clone() + Y0.clone() * Y2.clone() * Y3.clone() * Y5.clone() * Y10.clone() + Y0.clone() * Y1.clone() * Y3.clone() * Y5.clone() * Y11.clone() + Y0.clone() * Y6.clone() * Y7.clone() + Y3.clone() * Y5.clone() * Y9.clone() - scalar(4),
        scalar(2) * Y0.clone() * Y1.clone() * Y2.clone() * Y3.clone().pow(2) * Y4.clone() * Y5.clone() + scalar(2) * Y0.clone() * Y1.clone() * Y2.clone() * Y3.clone() * Y5.clone().pow(2) * Y8.clone() + Y0.clone() * Y2.clone() * Y3.clone() * Y5.clone() * Y6.clone() + Y0.clone() * Y1.clone() * Y3.clone() * Y5.clone() * Y7.clone() + scalar(8),
        Y0.clone() * Y1.clone() * Y2.clone() * Y3.clone().pow(2) * Y5.clone().pow(2) - scalar(5)
    ].into_iter().map(|f| f.unwrap()).collect::<Vec<_>>();

    let part_of_result = [
        scalar(4) * Y2.clone().pow(2) * Y6.clone().pow(2) - scalar(4) * Y1.clone().pow(2) * Y7.clone().pow(2),
        scalar(8) * Y2.clone() * Y6.clone() + scalar(8) * Y1.clone() * Y7.clone()
    ];

    let start = std::time::Instant::now();
    let gb = f4::<_, _, true>(&ring, system, order, u16::MAX);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    for f in &part_of_result {
        assert!(ring.is_zero(&multivariate_division(&ring, f.clone().unwrap(), gb.iter(), order)));
    }

    assert_eq!(93, gb.len());
}

#[test]
#[ignore]
fn test_difficult_gb() {
    let order = DegRevLex;
    let base = zn_static::Fp::<7>::RING;
    let ring: MultivariatePolyRingImpl<_, _, 7> = MultivariatePolyRingImpl::new(base, order);

    let X0 = RingElementWrapper::new(&ring, ring.indeterminate(0));
    let X1 = RingElementWrapper::new(&ring, ring.indeterminate(1));
    let X2 = RingElementWrapper::new(&ring, ring.indeterminate(2));
    let X3 = RingElementWrapper::new(&ring, ring.indeterminate(3));
    let X4 = RingElementWrapper::new(&ring, ring.indeterminate(4));
    let X5 = RingElementWrapper::new(&ring, ring.indeterminate(5));
    let X6 = RingElementWrapper::new(&ring, ring.indeterminate(6));

    let i = |x: i64| RingElementWrapper::new(&ring, ring.inclusion().map(ring.base_ring().coerce(&StaticRing::<i64>::RING, x)));

    let basis = vec![
        i(6) + i(2) * X5.clone() + i(2) * X4.clone() + X6.clone() + i(4) * X0.clone() + i(5) * X6.clone() * X5.clone() + X6.clone() * X4.clone() + i(3) * X0.clone() * X4.clone() + i(6) * X0.clone() * X6.clone() + i(2) * X0.clone() * X3.clone() + X0.clone() * X2.clone() + i(4) * X0.clone() * X1.clone() + i(2) * X3.clone() * X4.clone() * X5.clone() + i(4) * X0.clone() * X6.clone() * X5.clone() + i(6) * X0.clone() * X2.clone() * X5.clone() + i(5) * X0.clone() * X6.clone() * X4.clone() + i(2) * X0.clone() * X3.clone() * X4.clone() + i(4) * X0.clone() * X1.clone() * X4.clone() + X0.clone() * X6.clone().pow(2) + i(3) * X0.clone() * X3.clone() * X6.clone() + i(5) * X0.clone() * X2.clone() * X6.clone() + i(2) * X0.clone() * X1.clone() * X6.clone() + X0.clone() * X3.clone().pow(2) + i(2) * X0.clone() * X2.clone() * X3.clone() + i(3) * X0.clone() * X3.clone() * X4.clone() * X5.clone() + i(4) * X0.clone() * X3.clone() * X6.clone() * X5.clone() + i(3) * X0.clone() * X1.clone() * X6.clone() * X5.clone() + i(3) * X0.clone() * X2.clone() * X3.clone() * X5.clone() + i(3) * X0.clone() * X3.clone() * X6.clone() * X4.clone() + i(2) * X0.clone() * X1.clone() * X6.clone() * X4.clone() + i(2) * X0.clone() * X3.clone().pow(2) * X4.clone() + i(2) * X0.clone() * X2.clone() * X3.clone() * X4.clone() + i(3) * X0.clone() * X3.clone().pow(2) * X4.clone() * X5.clone() + i(4) * X0.clone() * X1.clone() * X3.clone() * X4.clone() * X5.clone() + X0.clone() * X3.clone().pow(2) * X4.clone().pow(2),
        i(5) + i(4) * X0.clone() + i(6) * X4.clone() * X5.clone() + i(3) * X6.clone() * X5.clone() + i(4) * X0.clone() * X4.clone() + i(3) * X0.clone() * X6.clone() + i(6) * X0.clone() * X3.clone() + i(6) * X0.clone() * X2.clone() + i(6) * X6.clone() * X4.clone() * X5.clone() + i(2) * X0.clone() * X4.clone() * X5.clone() + i(4) * X0.clone() * X6.clone() * X5.clone() + i(3) * X0.clone() * X2.clone() * X5.clone() + i(3) * X0.clone() * X6.clone() * X4.clone() + i(5) * X0.clone() * X3.clone() * X4.clone() + i(6) * X0.clone() * X2.clone() * X4.clone() + i(4) * X0.clone() * X6.clone().pow(2) + i(3) * X0.clone() * X3.clone() * X6.clone() + i(3) * X0.clone() * X2.clone() * X6.clone() + i(2) * X0.clone() * X6.clone() * X4.clone() * X5.clone() + i(6) * X0.clone() * X3.clone() * X4.clone() * X5.clone() + i(5) * X0.clone() * X1.clone() * X4.clone() * X5.clone() + i(6) * X0.clone() * X6.clone().pow(2) * X5.clone() + i(2) * X0.clone() * X3.clone() * X6.clone() * X5.clone() + i(2) * X0.clone() * X2.clone() * X6.clone() * X5.clone() + i(6) * X0.clone() * X1.clone() * X6.clone() * X5.clone() + i(6) * X0.clone() * X2.clone() * X3.clone() * X5.clone() + i(6) * X0.clone() * X3.clone() * X4.clone().pow(2) + i(4) * X0.clone() * X6.clone().pow(2) * X4.clone() + i(6) * X0.clone() * X3.clone() * X6.clone() * X4.clone() + i(3) * X0.clone() * X2.clone() * X6.clone() * X4.clone() + i(4) * X0.clone() * X3.clone() * X6.clone() * X4.clone() * X5.clone() + i(5) * X0.clone() * X1.clone() * X6.clone() * X4.clone() * X5.clone() + i(6) * X0.clone() * X3.clone().pow(2) * X4.clone() * X5.clone() + i(5) * X0.clone() * X2.clone() * X3.clone() * X4.clone() * X5.clone() + i(3) * X0.clone() * X3.clone() * X6.clone() * X4.clone().pow(2) + i(6) * X0.clone() * X3.clone().pow(2) * X4.clone().pow(2) * X5.clone(),
        i(2) + i(2) * X0.clone() + i(4) * X0.clone() * X4.clone() + i(2) * X0.clone() * X6.clone() + i(5) * X0.clone() * X4.clone() * X5.clone() + i(2) * X0.clone() * X6.clone() * X5.clone() + i(4) * X0.clone() * X2.clone() * X5.clone() + i(2) * X0.clone() * X4.clone().pow(2) + i(4) * X0.clone() * X6.clone() * X4.clone() + i(4) * X0.clone() * X6.clone().pow(2) + i(2) * X6.clone() * X4.clone() * X5.clone().pow(2) + i(4) * X0.clone() * X6.clone() * X4.clone() * X5.clone() + X0.clone() * X3.clone() * X4.clone() * X5.clone() + X0.clone() * X2.clone() * X4.clone() * X5.clone() + i(3) * X0.clone() * X6.clone().pow(2) * X5.clone() + i(2) * X0.clone() * X3.clone() * X6.clone() * X5.clone() + i(4) * X0.clone() * X2.clone() * X6.clone() * X5.clone() + i(2) * X0.clone() * X6.clone() * X4.clone().pow(2) + X0.clone() * X6.clone().pow(2) * X4.clone() + i(3) * X0.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(2) * X0.clone() * X6.clone().pow(2) * X5.clone().pow(2) + i(3) * X0.clone() * X2.clone() * X6.clone() * X5.clone().pow(2) + X0.clone() * X3.clone() * X4.clone().pow(2) * X5.clone() + X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone() + X0.clone() * X3.clone() * X6.clone() * X4.clone() * X5.clone() + i(6) * X0.clone() * X2.clone() * X6.clone() * X4.clone() * X5.clone() + i(4) * X0.clone() * X6.clone().pow(2) * X4.clone().pow(2) + i(6) * X0.clone() * X3.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(4) * X0.clone() * X1.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(4) * X0.clone() * X2.clone() * X3.clone() * X4.clone() * X5.clone().pow(2) + i(6) * X0.clone() * X3.clone() * X6.clone() * X4.clone().pow(2) * X5.clone() + i(2) * X0.clone() * X3.clone().pow(2) * X4.clone().pow(2) * X5.clone().pow(2),
        i(4) + i(5) * X0.clone() * X4.clone() * X5.clone() + i(6) * X0.clone() * X6.clone() * X5.clone() + i(5) * X0.clone() * X4.clone().pow(2) * X5.clone() + i(3) * X0.clone() * X6.clone() * X4.clone() * X5.clone() + i(3) * X0.clone() * X6.clone().pow(2) * X5.clone() + i(6) * X0.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(5) * X0.clone() * X2.clone() * X4.clone() * X5.clone().pow(2) + X0.clone() * X6.clone().pow(2) * X5.clone().pow(2) + i(6) * X0.clone() * X2.clone() * X6.clone() * X5.clone().pow(2) + i(4) * X0.clone() * X6.clone() * X4.clone().pow(2) * X5.clone() + i(2) * X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone() + i(5) * X0.clone() * X3.clone() * X4.clone().pow(2) * X5.clone().pow(2) + i(3) * X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone().pow(2) + i(5) * X0.clone() * X3.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(4) * X0.clone() * X2.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + i(6) * X0.clone() * X6.clone().pow(2) * X4.clone().pow(2) * X5.clone() + i(4) * X0.clone() * X3.clone() * X6.clone() * X4.clone().pow(2) * X5.clone().pow(2),
        i(4) + i(4) * X0.clone() * X4.clone().pow(2) * X5.clone().pow(2) + X0.clone() * X6.clone() * X4.clone() * X5.clone().pow(2) + X0.clone() * X6.clone().pow(2) * X5.clone().pow(2) + i(5) * X0.clone() * X6.clone() * X4.clone().pow(2) * X5.clone().pow(2) + i(6) * X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone().pow(2) + i(3) * X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone().pow(3) + i(4) * X0.clone() * X2.clone() * X6.clone() * X4.clone() * X5.clone().pow(3) + i(6) * X0.clone() * X6.clone().pow(2) * X4.clone().pow(2) * X5.clone().pow(2) + i(4) * X0.clone() * X3.clone() * X6.clone() * X4.clone().pow(2) * X5.clone().pow(3),
        i(5) * X0.clone() * X6.clone() * X4.clone().pow(2) * X5.clone().pow(3) + i(6) * X0.clone() * X6.clone().pow(2) * X4.clone() * X5.clone().pow(3) + i(5) * X0.clone() * X6.clone().pow(2) * X4.clone().pow(2) * X5.clone().pow(3),
        i(2) * X0.clone() * X6.clone().pow(2) * X4.clone().pow(2) * X5.clone().pow(4)
    ].into_iter().map(|f| f.unwrap()).collect();

    let start = std::time::Instant::now();
    let gb = f4::<_, _, true>(ring, basis, order, u16::MAX);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
    std::hint::black_box(gb);
}