use crate::field::{FieldStore, Field};
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::vector::*;

use super::sparse_invert::{SparseMatrix, gb_rowrev_sparse_row_echelon};

struct MonomialSet<P, O>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    data: Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>>,
    order: O
}

enum AddResult {
    Present(usize), Added(usize)
}

impl<P, O> MonomialSet<P, O>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    fn new(order: O) -> Self {
        Self {
            data: Vec::new(),
            order: order
        }
    }

    fn add(&mut self, m: Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> AddResult {
        match self.data.binary_search_by(|x| self.order.compare(x, &m).reverse()) {
            Ok(i) => AddResult::Present(i),
            Err(i) => {
                self.data.insert(i, m);
                AddResult::Added(i)
            }
        }
    }

    #[allow(unused)]
    fn remove(&mut self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> bool {
        match self.data.binary_search_by(|x| self.order.compare(x, m).reverse()) {
            Ok(i) => {
                self.data.remove(i);
                return true;
            },
            Err(_) => return false
        }
    }

    #[allow(unused)]
    fn contains(&self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> bool {
        match self.data.binary_search_by(|x| self.order.compare(x, m).reverse()) {
            Ok(_) => true,
            Err(_) => false
        }
    }

    fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>> {
        self.data.iter()
    }

    fn at_index(&self, i: usize) -> &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector> {
        &self.data[i]
    }

    fn index_of(&self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> Option<usize> {
        self.data.binary_search_by(|x| self.order.compare(x, &m).reverse()).ok()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

fn S<P, O>(ring: P, f1: &El<P>, f2: &El<P>, order: O) -> El<P> 
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy
{
    let f1_lm = ring.lm(f1, order).unwrap();
    let f2_lm = ring.lm(f2, order).unwrap();
    let lcm = f1_lm.clone().lcm(&f2_lm);
    let f1_factor = lcm.clone().div(&f1_lm);
    let f2_factor = lcm.div(&f2_lm);
    let mut f1_scaled = ring.clone_el(f1);
    ring.mul_monomial(&mut f1_scaled, &f1_factor);
    let mut f2_scaled = ring.clone_el(f2);
    ring.mul_monomial(&mut f2_scaled, &f2_factor);
    return ring.sub(f1_scaled, f2_scaled);
}

fn S_deg<P, O>(ring: P, f1: &El<P>, f2: &El<P>, order: O) -> u16
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy
{
    let f1_lm = ring.lm(f1, order).unwrap();
    let f2_lm = ring.lm(f2, order).unwrap();
    let lcm = f1_lm.clone().lcm(&f2_lm);
    let f1_factor = lcm.deg() - f1_lm.deg();
    let f2_factor = lcm.deg() - f2_lm.deg();
    return ring.terms(f1).filter(|(_, m)| *m != f1_lm).map(|(_, m)| m.deg() + f1_factor)
        .chain(ring.terms(f2).filter(|(_, m)| *m != f2_lm).map(|(_, m)| m.deg() + f2_factor))
        .max().unwrap_or(0);
}

pub fn reduce_S_matrix<P, O>(ring: P, S_polys: &[El<P>], basis: &[El<P>], order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    if S_polys.len() == 0 {
        return Vec::new();
    }

    let ring_ref = &ring;
    let mut columns: MonomialSet<P, O> = MonomialSet::new(order);
    for b in S_polys {
        for (_, b_m) in ring.terms(b) {
            columns.add(b_m.clone());
        }
    }
    let columns_ref = &columns;
    let mut A = SparseMatrix::new(
        ring.base_ring(),
        S_polys.len(),
        columns.len(),
        S_polys.iter().enumerate().flat_map(move |(i, f)|
            ring_ref.terms(f).map(move |(c, m)| {
                let row = i;
                let col = columns_ref.index_of(m).unwrap();
                return (row, col, ring_ref.base_ring().clone_el(c));
            })
        )
    );

    // all monomials for which we have to add a row to A to enable eliminating them
    let mut open = columns.iter().cloned().collect::<Vec<_>>();
    
    while let Some(m) = open.pop() {
        if let Some(f) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m)).next() {
            let div_monomial = m.clone().div(ring.lm(f, order).unwrap());
            A.add_row(0);
            for (c, f_m) in ring.terms(f) {
                let final_monomial = div_monomial.clone().mul(f_m);
                let col = match columns.add(final_monomial.clone()) {
                    AddResult::Added(i) => { A.add_column(i); open.push(final_monomial); i },
                    AddResult::Present(i) => i
                };

                A.set(0, col, ring.base_ring().clone_el(c));
            }
        }
    }

    gb_rowrev_sparse_row_echelon(&mut A);

    let mut result = Vec::new();
    for i in 0..A.row_count() {
        if let Some(j) = A.get_row(i).nontrivial_entries().map(|(j, _)| j).min() {
            if basis.iter().all(|f| !ring.lm(f, order).unwrap().divides(columns.at_index(j))) {
                result.push(ring.from_terms(
                    A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), columns.at_index(j)))
                ))
            }
        }
    }
    return result;
}

///
/// A relatively simple implementation of the F4 algorithm to compute a Groebner Basis.
/// This currently does not include the Buchberger 
/// 
pub fn f4_base<P, O, const LOG: bool>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let filter_S_pair = |f: &El<P>, g: &El<P>| {
        f as *const _ != g as *const _ &&
            !ring.lm(f, order).unwrap().is_coprime(ring.lm(g, order).unwrap())
    };

    let select = |ring: &P, f: &El<P>, g: &El<P>, degree_bound: usize| {
        if S_deg(ring, f, g, order) as usize <= degree_bound {
            true
        } else {
            false
        }
    };

    let mut open = (0..basis.len()).flat_map(|i| (0..i).map(move |j| (i, j)))
        .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
        .collect::<Vec<_>>();
    
    let mut degree_bound = 1;
    while open.len() > 0 {
        let mut process = Vec::new();

        open.retain(|(i, j)| 
            if select(&ring, &basis[*i], &basis[*j], degree_bound) { 
                process.push((*i, *j));
                false
            } else {
                true
            }
        );

        let S_polys: Vec<_> = process.iter().cloned().map(|(i, j)| S(&ring, &basis[i], &basis[j], order)).collect();

        if S_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
            }
            continue;
        }

        let new_polys = reduce_S_matrix(&ring, &S_polys, &basis, order);

        if new_polys.len() == 0 {
            degree_bound += 1;
            if LOG {
                print!("{{{}}}", degree_bound);
            }
            continue;
        }

        basis.extend(new_polys
            .into_iter()
            .inspect(|f| if LOG { print!("a{:?}", ring.lm(f, order).unwrap()); })
        );
        basis = reduce(&ring, basis, order).0;
        open = (0..basis.len()).flat_map(|i| (0..i).map(move |j| (i, j)))
            .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
            .collect::<Vec<_>>();

        if LOG {
            print!("({})", open.len());
        }
    }
    if LOG {
        println!();
    }
    return basis;
}

fn reduce<P, O>(ring: P, mut polys: Vec<El<P>>, order: O) -> (Vec<El<P>>, bool)
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    polys.sort_by(|l, r| order.compare(ring.lm(l, order).unwrap(), ring.lm(r, order).unwrap()));
    let mut i = 0;
    let mut removed_any = false;
    while i < polys.len() {
        let reduced = multivariate_division(&ring, ring.clone_el(&polys[i]), &polys[..i], order);
        if !ring.is_zero(&reduced) {
            polys[i] = reduced;
            i += 1;
        } else {
            polys.remove(i);
            removed_any = true;
        }
    }
    return (polys, removed_any);
}

pub fn multivariate_division<P, V, O>(ring: P, mut f: El<P>, set: V, order: O) -> El<P>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy,
        V: VectorView<El<P>>
{
    if ring.is_zero(&f) {
        return f;
    }
    let mut f_lm = ring.lm(&f, order).unwrap().clone();
    while let Some(g) = set.iter().filter(|g| ring.lm(g, order).unwrap().divides(&f_lm)).next() {
        let g_lm = ring.lm(g, order).unwrap();
        let f_lc = ring.coefficient_at(&f, &f_lm);
        let g_lc = ring.coefficient_at(&g, &g_lm);
        let div_monomial = f_lm.div(&g_lm);
        let div_coeff = ring.base_ring().div(&f_lc, &g_lc);
        let mut g_scaled = ring.clone_el(g);
        ring.mul_monomial(&mut g_scaled, &div_monomial);
        ring.mul_assign_base(&mut g_scaled, &div_coeff);
        ring.sub_assign(&mut f, g_scaled);
        if let Some(m) = ring.lm(&f, order) {
            f_lm = m.clone();
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

#[test]
fn test_f4_small() {
    let order = DegRevLex;
    let base = zn_static::Z17;
    let ring: MultivariatePolyRingImpl<_, _, _, 2> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

    let f1 = ring.from_terms([
        (1, &Monomial::new([2, 0])),
        (1, &Monomial::new([0, 2])),
        (16, &Monomial::new([0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, &Monomial::new([1, 1])),
        (15, &Monomial::new([0, 0]))
    ].into_iter());

    ring.println(&f1);
    ring.println(&f2);

    let actual = f4_base::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order);

    let expected = ring.from_terms([
        (16, &Monomial::new([0, 3])),
        (15, &Monomial::new([1, 0])),
        (1, &Monomial::new([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(&ring, &f2, actual.at(0));
    assert_el_eq!(&ring, &f1, actual.at(1));
    assert_el_eq!(&ring, &expected, actual.at(2));
}

#[test]
fn test_f4_larger() {
    let order = DegRevLex;
    let base = zn_static::Z17;
    let ring: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

    let f1 = ring.from_terms([
        (1, &Monomial::new([2, 1, 1])),
        (1, &Monomial::new([0, 2, 0])),
        (1, &Monomial::new([1, 0, 1])),
        (2, &Monomial::new([1, 0, 0])),
        (1, &Monomial::new([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, &Monomial::new([0, 3, 1])),
        (1, &Monomial::new([0, 0, 3])),
        (1, &Monomial::new([1, 1, 0]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, &Monomial::new([1, 0, 2])),
        (1, &Monomial::new([1, 0, 1])),
        (2, &Monomial::new([0, 1, 1])),
        (7, &Monomial::new([0, 0, 0]))
    ].into_iter());

    let actual = f4_base::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);

    let g1 = ring.from_terms([
        (1, &Monomial::new([0, 4, 0])),
        (8, &Monomial::new([0, 3, 1])),
        (12, &Monomial::new([0, 1, 3])),
        (6, &Monomial::new([0, 0, 4])),
        (1, &Monomial::new([0, 3, 0])),
        (13, &Monomial::new([0, 2, 1])),
        (11, &Monomial::new([0, 1, 2])),
        (10, &Monomial::new([0, 0, 3])),
        (11, &Monomial::new([0, 2, 0])),
        (12, &Monomial::new([0, 1, 1])),
        (6, &Monomial::new([0, 0, 2])),
        (6, &Monomial::new([0, 1, 0])),
        (13, &Monomial::new([0, 0, 1])),
        (9, &Monomial::new([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, g1, &actual, order));
}

#[test]
fn test_f4_larger_elim() {
    let order = BlockLexDegRevLex::new(..1, 1..);
    let base = zn_static::Z17;
    let ring: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());

    let f1 = ring.from_terms([
        (1, &Monomial::new([2, 1, 1])),
        (1, &Monomial::new([0, 2, 0])),
        (1, &Monomial::new([1, 0, 1])),
        (2, &Monomial::new([1, 0, 0])),
        (1, &Monomial::new([0, 0, 0]))
    ].into_iter());
    let f2 = ring.from_terms([
        (1, &Monomial::new([1, 1, 0])),
        (1, &Monomial::new([0, 3, 1])),
        (1, &Monomial::new([0, 0, 3]))
    ].into_iter());
    let f3 = ring.from_terms([
        (1, &Monomial::new([1, 0, 2])),
        (1, &Monomial::new([1, 0, 1])),
        (2, &Monomial::new([0, 1, 1])),
        (7, &Monomial::new([0, 0, 0]))
    ].into_iter());

    let actual = f4_base::<_, _, true>(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);
    
    for f in &actual {
        println!("{}, ", ring.format(f));
    }

    let g1 = ring.from_terms([
        (1, &Monomial::new([0, 4, 0])),
        (8, &Monomial::new([0, 3, 1])),
        (12, &Monomial::new([0, 1, 3])),
        (6, &Monomial::new([0, 0, 4])),
        (1, &Monomial::new([0, 3, 0])),
        (13, &Monomial::new([0, 2, 1])),
        (11, &Monomial::new([0, 1, 2])),
        (10, &Monomial::new([0, 0, 3])),
        (11, &Monomial::new([0, 2, 0])),
        (12, &Monomial::new([0, 1, 1])),
        (6, &Monomial::new([0, 0, 2])),
        (6, &Monomial::new([0, 1, 0])),
        (13, &Monomial::new([0, 0, 1])),
        (9, &Monomial::new([0, 0, 0]))
    ].into_iter());

    assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, g1, &actual, order));
}

#[test]
#[ignore]
fn test_generic_computation() {
    let order = DegRevLex;
    let base = zn_static::Z17;
    let ring: MultivariatePolyRingImpl<_, _, _, 6> = MultivariatePolyRingImpl::new(base, order, default_memory_provider!());
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let x1 = poly_ring.mul(
        poly_ring.from_terms([(ring.indeterminate(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(ring.indeterminate(1), 0), (ring.one(), 1)].into_iter())
    );
    let x2 = poly_ring.mul(
        poly_ring.add(poly_ring.clone_el(&x1), poly_ring.from_terms([(ring.indeterminate(2), 0), (ring.indeterminate(3), 1)].into_iter())),
        poly_ring.add(poly_ring.clone_el(&x1), poly_ring.from_terms([(ring.indeterminate(4), 0), (ring.indeterminate(5), 1)].into_iter()))
    );
    let basis = vec![
        ring.sub_ref_snd(ring.from_int(1), poly_ring.coefficient_at(&x2, 0)),
        ring.sub_ref_snd(ring.from_int(1), poly_ring.coefficient_at(&x2, 1)),
        ring.sub_ref_snd(ring.from_int(1), poly_ring.coefficient_at(&x2, 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = f4_base::<_, _, true>(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), order);
    std::hint::black_box(&gb1);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
}