use std::io::Write;

use crate::algorithms::miller_rabin::is_prime;
use crate::divisibility::{DivisibilityRingStore, DivisibilityRing};
use crate::field::{FieldStore, Field};
use crate::homomorphism::Homomorphism;
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::rings::zn::{ZnRingStore, ZnRing};
use crate::vector::*;

use super::int_factor::factor;
use super::sparse_invert::*;

struct MonomialMap<P, O, K = ()>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore,
        O: MonomialOrder + Copy
{
    data: Vec<(Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>, K)>,
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

    fn add(&mut self, m: Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>, k: K) -> AddResult<K> {
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
    fn remove(&mut self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> Option<K> {
        match self.data.binary_search_by(|(x, _)| self.order.compare(x, m).reverse()) {
            Ok(i) => {
                let (_, result) = self.data.remove(i);
                return Some(result);
            },
            Err(_) => return None
        }
    }

    #[allow(unused)]
    fn contains<'a>(&'a self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> Option<&'a K> {
        match self.data.binary_search_by(|(x, _)| self.order.compare(x, m).reverse()) {
            Ok(i) => Some(&self.data[i].1),
            Err(_) => None
        }
    }

    fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a (Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>, K)> {
        self.data.iter()
    }

    fn at_index(&self, i: usize) -> &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector> {
        &self.data[i].0
    }

    fn index_of(&self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> Option<usize> {
        self.data.binary_search_by(|(x, _)| self.order.compare(x, &m).reverse()).ok()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

fn S<P, O>(ring: P, f1: &El<P>, f2: &El<P>, order: O) -> El<P> 
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        O: MonomialOrder + Copy
{
    let f1_lm = ring.lm(f1, order).unwrap();
    let f2_lm = ring.lm(f2, order).unwrap();
    let mon_lc = ring.clone_monomial(&f1_lm).lcm(&f2_lm);
    let f1_factor = ring.clone_monomial(&mon_lc).div(&f1_lm);
    let f2_factor = mon_lc.div(&f2_lm);
    let f1_lc = ring.coefficient_at(f1, f1_lm);
    let f2_lc = ring.coefficient_at(f2, f2_lm);
    let coeff_gcd = ring.base_ring().ideal_gen(f1_lc, f2_lc).2;
    let mut f1_scaled = ring.clone_el(f1);
    ring.mul_monomial(&mut f1_scaled, &f1_factor);
    ring.inclusion().mul_assign_map(&mut f1_scaled, ring.base_ring().checked_div(f2_lc, &coeff_gcd).unwrap());
    let mut f2_scaled = ring.clone_el(f2);
    ring.mul_monomial(&mut f2_scaled, &f2_factor);
    ring.inclusion().mul_assign_map(&mut f2_scaled, ring.base_ring().checked_div(f1_lc, &coeff_gcd).unwrap());
    return ring.sub(f1_scaled, f2_scaled);
}

fn S_deg<P, O>(ring: P, f1: &El<P>, f2: &El<P>, order: O) -> u16
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy
{
    let f1_lm = ring.lm(f1, order).unwrap();
    let f2_lm = ring.lm(f2, order).unwrap();
    let lcm = ring.clone_monomial(&f1_lm).lcm(&f2_lm);
    let f1_factor = lcm.deg() - f1_lm.deg();
    let f2_factor = lcm.deg() - f2_lm.deg();
    return ring.terms(f1).filter(|(_, m)| *m != f1_lm).map(|(_, m)| m.deg() + f1_factor)
        .chain(ring.terms(f2).filter(|(_, m)| *m != f2_lm).map(|(_, m)| m.deg() + f2_factor))
        .max().unwrap_or(0);
}

fn nil_S<P>(ring: P, p: &El<<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as ZnRing>::Integers>, e: usize, f: &El<P>, k: usize) -> El<P> 
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{
    debug_assert!(is_prime(ring.base_ring().integer_ring(), p, 10));
    debug_assert!(ring.base_ring().integer_ring().eq_el(ring.base_ring().modulus(), &ring.base_ring().integer_ring().pow(ring.base_ring().integer_ring().clone_el(p), e)));
    let mut result = ring.clone_el(f);
    let modulo = ring.base_ring().can_hom(ring.base_ring().integer_ring()).unwrap();
    ring.inclusion().mul_assign_map(&mut result, ring.base_ring().pow(modulo.map_ref(p), k));
    return result;
}

fn nil_S_deg<P>(ring: P, p: &El<<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as ZnRing>::Integers>, e: usize, f: &El<P>, k: usize) -> u16
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{
    debug_assert!(is_prime(ring.base_ring().integer_ring(), p, 10));
    debug_assert!(ring.base_ring().integer_ring().eq_el(ring.base_ring().modulus(), &ring.base_ring().integer_ring().pow(ring.base_ring().integer_ring().clone_el(p), e)));
    let annihilator_gen = ring.base_ring().integer_ring().checked_div(ring.base_ring().modulus(), &ring.base_ring().integer_ring().pow(ring.base_ring().integer_ring().clone_el(p), k)).unwrap();
    let modulo = ring.base_ring().can_hom(ring.base_ring().integer_ring()).unwrap();
    let annihilator_gen = modulo.map(annihilator_gen);
    return ring.terms(f).filter(|(c, _)| ring.base_ring().checked_div(c, &annihilator_gen).is_none())
        .map(|(_, m)| m.deg()).max().unwrap_or(0);
}

pub fn reduce_S_matrix<P, O>(ring: P, S_polys: &[El<P>], basis: &[El<P>], order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: RingStore + Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        O: MonomialOrder + Copy,
        El<<P::Type as RingExtension>::BaseRing>: Send + Sync
{
    if S_polys.len() == 0 {
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
    let mut A = SparseMatrixBuilder::new(&ring.base_ring());
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
    let mut open = columns.iter().map(|(m, _)| ring.clone_monomial(m)).collect::<Vec<_>>();
    
    while let Some(m) = open.pop() {
        if let Some(f) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m)).next() {
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

    let entries = gb_sparse_row_echelon::<_, true>(ring.base_ring(), A, 256);

    let mut result = Vec::new();
    for i in 0..entries.len() {
        if let Some(j) = entries[i].iter().inspect(|(_, c)| assert!(!ring.base_ring().is_zero(c))).map(|(j, _)| *j).min() {
            if basis.iter().all(|f| !ring.lm(f, order).unwrap().divides(columns.at_index(j))) {
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
/// A simple implementation of the F4 algorithm for computing Groebner basis.
/// This implementation does currently not include the Buchberger criteria, and thus
/// cannot compete with highly optimized implementations (Singular, Macaulay2, Magma etc).
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
/// # use feanor_math::{assert_el_eq, default_memory_provider};
/// # use feanor_math::rings::zn::zn_static;
/// 
/// let order = DegRevLex;
/// let base = zn_static::F17;
/// let ring: ordered::MultivariatePolyRingImpl<_, _, _, 2> = ordered::MultivariatePolyRingImpl::new(base, order, default_memory_provider!());
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
/// let gb = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order);
/// 
/// let in_ideal = ring.from_terms([
///     (16, Monomial::new([0, 3])),
///     (15, Monomial::new([1, 0])),
///     (1, Monomial::new([0, 1])),
/// ].into_iter());
/// 
/// assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, in_ideal, &gb, order));
/// ```
/// 
pub fn f4<P, O, const LOG: bool>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore + Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy,
        El<<P::Type as RingExtension>::BaseRing>: Send + Sync
{
    basis = reduce(&ring, basis, order);

    let filter_product_criterion = |f_i: usize, g_i: usize, basis: &[El<P>]| {
        ring.lm(&basis[f_i], order).unwrap().is_coprime(ring.lm(&basis[g_i], order).unwrap())
    };

    let mut chain_criterion_reduced_pairs = Vec::new();
    let filter_chain_criterion = |f_i: usize, g_i: usize, basis: &[El<P>], reduced_pairs: &[(usize, usize)]| {
        let m = ring.clone_monomial(ring.lm(&basis[f_i], order).unwrap()).lcm(ring.lm(&basis[g_i], order).unwrap());
        (0..basis.len()).filter(|k| ring.lm(&basis[*k], order).unwrap().divides(&m))
            .any(|k| reduced_pairs.binary_search(&sym_tuple(f_i, k)).is_ok() && reduced_pairs.binary_search(&sym_tuple(k, g_i)).is_ok())
    };

    let select = |ring: &P, f: &El<P>, g: &El<P>, degree_bound: usize| {
        if S_deg(ring, f, g, order) as usize <= degree_bound {
            true
        } else {
            false
        }
    };

    let mut product_criterion_skipped = 0;
    let mut chain_criterion_skipped = 0;

    let mut open = (0..basis.len()).flat_map(|i| (0..i).map(move |j: usize| (i, j))).collect::<Vec<_>>();
    
    let mut degree_bound = 1;
    while open.len() > 0 {
        if LOG {
            print!("S({})", open.len());
            std::io::stdout().flush().unwrap();
        }

        let mut S_polys = Vec::new();
        let mut new_reduced_pairs = Vec::new();

        open.retain(|(i, j)| {
            if filter_product_criterion(*i, *j, &basis) {
                product_criterion_skipped += 1;
                false
            } else if filter_chain_criterion(*i, *j, &basis, &chain_criterion_reduced_pairs[..]) {
                chain_criterion_skipped += 1;
                false
            } else if select(&ring, &basis[*i], &basis[*j], degree_bound) { 
                S_polys.push(S(&ring, &basis[*i], &basis[*j], order));
                new_reduced_pairs.push(sym_tuple(*i, *j));
                false
            } else {
                true
            }
        });

        if S_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
                std::io::stdout().flush().unwrap();
            }
            continue;
        }

        let new_polys = if S_polys.len() > 20 {
            reduce_S_matrix(&ring, &S_polys, &basis, order)
        } else {
            let start = std::time::Instant::now();
            let result = S_polys.into_iter().map(|f| multivariate_division(&ring, f, &basis, order)).filter(|f| !ring.is_zero(f)).collect();
            let end = std::time::Instant::now();
            print!("[{}ms]", (end - start).as_millis());
            result
        };

        chain_criterion_reduced_pairs.extend(new_reduced_pairs.into_iter());
        chain_criterion_reduced_pairs.sort_unstable();

        if new_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
                std::io::stdout().flush().unwrap();
            }
        } else {

            degree_bound = 0;
            chain_criterion_reduced_pairs = Vec::new();
            basis.extend(new_polys.into_iter());
            basis = reduce(&ring, basis, order);

            open = (0..basis.len()).flat_map(|i| (0..i).map(move |j: usize| (i, j)))
                .collect::<Vec<_>>();
    
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

///
/// Works on rings `Z/p^eZ` for `p` a prime.
/// 
pub fn f4_local<P, O, const LOG: bool>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: ZnRingStore + Sync,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing,
        O: MonomialOrder + Copy,
        El<<P::Type as RingExtension>::BaseRing>: Send + Sync
{
    let factorization = factor(ring.base_ring().integer_ring(), ring.base_ring().integer_ring().clone_el(ring.base_ring().modulus()));
    assert!(factorization.len() == 1);
    let p = &factorization[0].0;
    let e = factorization[0].1;

    basis = reduce(&ring, basis, order);

    enum SPoly {
        Standard(usize, usize), Nilpotent(usize, usize)
    }

    let select = |s_poly: &SPoly, basis: &[El<P>], degree_bound: u16| match s_poly {
        SPoly::Standard(i, j) => if S_deg(&ring, &basis[*i], &basis[*j], order) <= degree_bound { Some(S(&ring, &basis[*i], &basis[*j], order)) } else { None },
        SPoly::Nilpotent(i, k) => if nil_S_deg(&ring, p, e, &basis[*i], *k) <= degree_bound { Some(nil_S(&ring, p, e, &basis[*i], *k)) } else { None }
    };

    let mut open = (0..basis.len()).flat_map(|i| (0..i).map(move |j: usize| SPoly::Standard(i, j)))
        .chain((0..basis.len()).flat_map(|i| (0..e).map(move |k: usize| SPoly::Nilpotent(i, k))))
        .collect::<Vec<_>>();
    
    let mut degree_bound = 1;
    while open.len() > 0 {
        if LOG {
            print!("S({})", open.len());
            std::io::stdout().flush().unwrap();
        }

        let mut S_polys = Vec::new();

        open.retain(|S_poly| {
            if let Some(poly) = select(&S_poly, &basis[..], degree_bound) { 
                S_polys.push(poly);
                false
            } else {
                true
            }
        });

        if S_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
                std::io::stdout().flush().unwrap();
            }
            continue;
        }

        let new_polys: Vec<_> = if S_polys.len() > 20 {
            reduce_S_matrix(&ring, &S_polys, &basis, order)
        } else {
            let start = std::time::Instant::now();
            let result = S_polys.into_iter().map(|f| multivariate_division(&ring, f, &basis, order)).filter(|f| !ring.is_zero(f)).collect();
            let end = std::time::Instant::now();
            print!("[{}ms]", (end - start).as_millis());
            result
        };

        if new_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
                std::io::stdout().flush().unwrap();
            }
        } else {

            degree_bound = 0;
            basis.extend(new_polys.into_iter());
            basis = reduce(&ring, basis, order);

            open = (0..basis.len()).flat_map(|i| (0..i).map(move |j: usize| SPoly::Standard(i, j)))
                .chain((0..basis.len()).flat_map(|i| (0..e).map(move |k: usize| SPoly::Nilpotent(i, k))))
                .collect::<Vec<_>>();
    
            if LOG {
                print!("b({})", basis.len());
                std::io::stdout().flush().unwrap();
            }
        }
    }
    return basis;
}

fn reduce<P, O>(ring: P, mut polys: Vec<El<P>>, order: O) -> Vec<El<P>>
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
            let reduced = multivariate_division(&ring, ring.clone_el(&polys[i]), (&polys[..i]).chain(&polys[(i + 1)..]), order);
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

pub fn multivariate_division<P, V, O>(ring: P, mut f: El<P>, set: V, order: O) -> El<P>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: DivisibilityRingStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        O: MonomialOrder + Copy,
        V: VectorView<El<P>>
{
    if ring.is_zero(&f) {
        return f;
    }
    let mut f_lm = ring.clone_monomial(ring.lm(&f, order).unwrap());
    let mut f_lc = ring.base_ring().clone_el(ring.coefficient_at(&f, &f_lm));
    let incl = ring.inclusion();
    while let Some((quo, g)) = set.iter()
        .filter(|g| ring.lm(g, order).unwrap().divides(&f_lm))
        .filter_map(|g| ring.base_ring().checked_div(&f_lc, ring.coefficient_at(g, ring.lm(g, order).unwrap())).map(|quo| (quo, g)))
        .next()
    {
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
    return f;
}

#[cfg(test)]
use crate::rings::multivariate::ordered::*;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::default_memory_provider;
#[cfg(test)]
use crate::rings::poly::*;
#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::wrapper::RingElementWrapper;

#[test]
fn test_f4_small() {
    let order = DegRevLex;
    let base: RingValue<zn_static::ZnBase<17, true>> = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, _, 2> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

    let f1 = ring.from_terms([
        (1, Monomial::new([2, 0])),
        (1, Monomial::new([0, 2])),
        (16, Monomial::new([0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, Monomial::new([1, 1])),
        (15, Monomial::new([0, 0]))
    ].into_iter());

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order);

    let expected = ring.from_terms([
        (16, Monomial::new([0, 3])),
        (15, Monomial::new([1, 0])),
        (1, Monomial::new([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(&ring, &f2, actual.at(0));
    assert_el_eq!(&ring, &f1, actual.at(1));
    assert_el_eq!(&ring, &ring.negate(expected), actual.at(2));
}

#[test]
fn test_f4_larger() {
    let order = DegRevLex;
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

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

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);

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

    assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, g1, &actual, order));
}

#[test]
fn test_f4_larger_elim() {
    let order = BlockLexDegRevLex::new(..1, 1..);
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

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

    let actual = f4::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);

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

    assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, g1, &actual, order));
}

#[test]
fn test_gb_local_ring() {
    let order = DegRevLex;
    let base = zn_static::Zn::<16>::RING;
    let ring: MultivariatePolyRingImpl<_, _, _, 1> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());
    
    let f = ring.from_terms([(4, Monomial::new([1])), (1, Monomial::new([0]))].into_iter());
    let gb = f4_local::<_, _, true>(&ring, vec![f], order);

    assert_eq!(1, gb.len());
    assert_el_eq!(&ring, &ring.one(), &gb[0]);
}

#[test]
fn test_gb_local_ring_large() {
    let order = DegRevLex;
    let base = zn_static::Zn::<16>::RING;
    let ring: MultivariatePolyRingImpl<_, _, _, 11> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

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

    let gb = f4_local::<_, _, true>(ring, system, order);
}

#[test]
fn test_generic_computation() {
    let order = DegRevLex;
    let base = zn_static::F17;
    let ring: MultivariatePolyRingImpl<_, _, _, 6> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let X1 = poly_ring.mul(
        poly_ring.from_terms([(ring.indeterminate(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(ring.indeterminate(1), 0), (ring.one(), 1)].into_iter())
    );
    let X2 = poly_ring.mul(
        poly_ring.add(poly_ring.clone_el(&X1.clone()), poly_ring.from_terms([(ring.indeterminate(2), 0), (ring.indeterminate(3), 1)].into_iter())),
        poly_ring.add(poly_ring.clone_el(&X1.clone()), poly_ring.from_terms([(ring.indeterminate(4), 0), (ring.indeterminate(5), 1)].into_iter()))
    );
    let basis = vec![
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2.clone(), 0)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2.clone(), 1)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2.clone(), 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = f4::<_, _, true>(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), order);
    std::hint::black_box(&gb1);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    for f in &gb1 {
        println!("{}", ring.format(f));
    }
    assert_eq!(11, gb1.len());
}

#[test]
#[ignore]
fn test_difficult_gb() {
    let order = DegRevLex;
    let base = zn_static::Fp::<7>::RING;
    let ring: MultivariatePolyRingImpl<_, _, _, 7> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

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
    let gb = f4::<_, _, true>(ring, basis, order);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
    std::hint::black_box(gb);
}