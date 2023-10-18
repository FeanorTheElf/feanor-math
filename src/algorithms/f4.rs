use crate::field::{FieldStore, Field};
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::vector::*;

use super::sparse_invert::{SparseMatrix, gb_rowrev_sparse_row_echelon};

pub struct MonomialSet<P, O>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    data: Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>>,
    order: O
}

impl<P, O> MonomialSet<P, O>
where P: MultivariatePolyRingStore,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: FieldStore,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
    O: MonomialOrder + Copy
{
    pub fn new(order: O) -> Self {
        Self {
            data: Vec::new(),
            order: order
        }
    }

    fn add(&mut self, m: Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> bool {
        match self.data.binary_search_by(|x| self.order.compare(x, &m).reverse()) {
            Ok(_) => return true,
            Err(i) => {
                self.data.insert(i, m);
                return false;
            }
        }
    }

    fn remove(&mut self, m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>) -> bool {
        match self.data.binary_search_by(|x| self.order.compare(x, m).reverse()) {
            Ok(i) => {
                self.data.remove(i);
                return true;
            },
            Err(_) => return false
        }
    }

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

pub fn reduce_S_matrix<P, O>(ring: P, S_polys: &[El<P>], basis: &[El<P>], order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let ring_ref = &ring;
    let mut columns: Vec<_> = Vec::new();
    for b in S_polys {
        for (_, b_m) in ring.terms(b) {
            for m in b_m.clone().dividing_monomials() {
                if let Err(index) = columns.binary_search_by(|x| order.compare(&x, &m)) {
                    columns.insert(index, m);
                }
            }
        }
    }
    let monomials_ref = &columns;
    let mut A = SparseMatrix::new(
        ring.base_ring(),
        S_polys.len(),
        columns.len(),
        S_polys.iter().enumerate().flat_map(move |(i, f)|
            ring_ref.terms(f).map(move |(c, m)| {
                let row = i;
                let col = monomials_ref.binary_search_by(|x| order.compare(x, m)).unwrap();
                return (row, col, ring_ref.base_ring().clone_el(c));
            })
        )
    );
    // all monomials m for which we find a basis polynomial f such that lm(f) | m
    // rows with these leading monomials will not yield new basis elements
    let mut reduction_monomials = Vec::new();

    // all monomials for which we have to add a row to A to enable eliminating them
    let mut open = columns.clone();
    
    while let Some(m) = open.pop() {
        if let Some(f) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m)).next() {
            reduction_monomials.push(m.clone());
            let div_monomial = m.clone().div(ring.lm(f, order).unwrap());
            let row = A.row_count();
            A.add_row(row);
            for (c, f_m) in ring.terms(f) {
                let final_monomial = div_monomial.clone().mul(f_m);
                let index = match columns.binary_search_by(|x| order.compare(x, &final_monomial)) {
                    Err(index) => {
                        columns.insert(index, final_monomial.clone());
                        open.push(final_monomial);
                        A.add_column(index);
                        index
                    },
                    Ok(index) => index
                };
                A.set(row, index, ring.base_ring().clone_el(c));
            }
        }
    }
    reduction_monomials.sort_by(|l, r| order.compare(l, r));

    // we have ordered the monomials in ascending order, but we want to reduce towards smaller ones
    A.reverse_cols();
    gb_rowrev_sparse_row_echelon(&mut A);
    A.reverse_cols();

    let mut result = Vec::new();
    for i in 0..A.row_count() {
        if let Some((j, _)) = A.get_row(i).nontrivial_entries().max_by(|(j1, _), (j2, _)| order.compare(&columns[*j1], &columns[*j2])) {
            if reduction_monomials.binary_search_by(|x| order.compare(x, &columns[j])).is_err() {
                result.push(ring.from_terms(
                    A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), &columns[j]))
                ))
            }
        }
    }
    return result;
}

fn leading_monomial<'a, P, O>(
    i: usize,
    A: &'a SparseMatrix<&<P::Type as RingExtension>::BaseRing>,
    columns: &'a MonomialSet<P, O>
) -> Option<&'a Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    A.get_row(i).nontrivial_entries().map(|(j, _)| j).min().map(|j| columns.at_index(j))
}

///
/// Requires that A is in lower right triangle form
/// 
fn leading_monomial_row<'a, P, O>(
    m: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>,
    A: &'a SparseMatrix<&<P::Type as RingExtension>::BaseRing>,
    columns: &'a MonomialSet<P, O>,
    order: O,
    ignore_last: usize
) -> Result<usize, usize>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let mut start = 0;
    let mut end = A.row_count() - ignore_last;
    while start + 1 < end {
        let mid = (start + end) / 2;
        match order.compare(m, leading_monomial(mid, A, columns).unwrap()) {
            std::cmp::Ordering::Equal => return Ok(mid),
            std::cmp::Ordering::Greater => start = mid + 1,
            std::cmp::Ordering::Less => end = mid,
        }
    }
    if start < end {
        println!("{:?}, {:?}, {:?}", m, leading_monomial(start, A, columns).unwrap(), order.compare(m, leading_monomial(start, A, columns).unwrap()));
        match order.compare(m, leading_monomial(start, A, columns).unwrap()) {
            std::cmp::Ordering::Equal => Ok(start),
            std::cmp::Ordering::Less => Err(start),
            std::cmp::Ordering::Greater => Err(start + 1)
        }
    } else {
        Err(start)
    }
}

fn add_reduction_rows_for_poly<P, O>(
    ring: &P, 
    poly: &El<P>, 
    A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>, 
    columns: &mut MonomialSet<P, O>, 
    do_not_reduce_with_self: bool,
    order: O,
    ignore_last: usize
)
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let ensure_column_exists = |m: &Monomial<_>, columns: &mut MonomialSet<_, _>, A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>| {
        if columns.add(m.clone()) {
            false
        } else {
            A.add_column(columns.index_of(m).unwrap());
            true
        }
    };

    let mut new_monomials = Vec::new();

    for (_, m) in ring.terms(&poly) {
        ensure_column_exists(&m, columns, A);
        new_monomials.push(m.clone());
    }

    while let Some(m) = new_monomials.pop() {

        if let Err(insert_at) = leading_monomial_row(&m, A, columns, order, ignore_last) {

            let mut reduction_rows = (0..(A.row_count() - ignore_last))
                .filter(|i| if let Some(i_m) = leading_monomial(*i, &A, &columns) {
                    i_m.divides(&m) && (*i_m != m || !do_not_reduce_with_self)
                } else {
                    false
                });

            if let Some(i) = reduction_rows.next() {
                
                assert!(i < insert_at);
                A.add_row(insert_at);

                let quo_m = m.clone().div(leading_monomial(i, &A, &columns).unwrap());
                let A_row = A.get_row(i).nontrivial_entries().map(|(j, c)| (columns.at_index(j).clone(), ring.base_ring().clone_el(c))).collect::<Vec<_>>();
                for (base_m, c) in A_row.into_iter() {
                    let new_m = base_m.mul(&quo_m);
                    ensure_column_exists(&new_m, columns, A);
                    new_monomials.push(new_m.clone());
                    A.set(insert_at, columns.index_of(&new_m).unwrap(), c);
                }
            }
        }
    }
}

fn add_poly_to_matrix<P, O>(
    ring: &P, 
    poly: El<P>, 
    A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>, 
    columns: &mut MonomialSet<P, O>,
    order: O,
    ignore_last: &mut usize
) -> usize
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{    
    add_reduction_rows_for_poly(ring, &poly, A, columns, false, order, *ignore_last);

    let new_row_index = A.row_count();
    A.add_row(new_row_index);
    *ignore_last += 1;

    for (c, m) in ring.terms(&poly) {
        A.set(new_row_index, columns.index_of(m).unwrap(), ring.base_ring().clone_el(c));
    }

    return new_row_index;
}

pub fn reduce_basis_polys<P, O>(
    ring: &P, 
    current_monomial_basis: &mut MonomialSet<P, O>,
    A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>, 
    columns: &mut MonomialSet<P, O>, 
    order: O
) -> bool
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let row_poly = |i: usize, A: &SparseMatrix<&<P::Type as RingExtension>::BaseRing>, columns: &MonomialSet<P, O>| ring.from_terms(
        A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), columns.at_index(j)))
    );
    let mut deleted_any = false;
    let mut index = 0;
    while index < current_monomial_basis.len() {
        let m = current_monomial_basis.at_index(index);
        if current_monomial_basis.iter().any(|x| x.divides(m) && x != m) {
            deleted_any = true;
            println!("del {:?} of {:?}", m, current_monomial_basis.iter().filter(|x| x.divides(m) && *x != m).collect::<Vec<_>>());
            let row = leading_monomial_row(m, A, columns, order, 0).unwrap();
            current_monomial_basis.remove(&m.clone());
            add_reduction_rows_for_poly(ring, &row_poly(row, &A, &columns), A, columns, true, order, 0);
        } else {
            index += 1;
        }
    }
    return deleted_any;
}

pub fn add_S_polys<P, O>(
    ring: &P, 
    to_process: &mut Vec<(Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>, Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>)>,
    current_monomial_basis: &MonomialSet<P, O>,
    A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>, 
    columns: &mut MonomialSet<P, O>, 
    order: O,
    degree_bound: usize
) -> bool
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let row_poly = |i: usize, A: &SparseMatrix<&<P::Type as RingExtension>::BaseRing>, columns: &MonomialSet<P, O>| ring.from_terms(
        A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), columns.at_index(j)))
    );
    let mut added_S_polys = 0;
    to_process.retain(|(m1, m2)| {
        if current_monomial_basis.contains(m1) && current_monomial_basis.contains(m2) {
            if (m1.clone().lcm(&m2).deg() as usize) < degree_bound {
                println!("S-poly for {:?}, {:?}", m1, m2);
                let i1 = leading_monomial_row(m1, A, columns, order, added_S_polys).unwrap();
                let i2 = leading_monomial_row(m2, A, columns, order, added_S_polys).unwrap();
                let S_poly = S(&ring, &row_poly(i1, &A, &columns), &row_poly(i2, &A, &columns), order);
                ring.println(&S_poly);
                add_poly_to_matrix(ring, S_poly, A, columns, order, &mut added_S_polys);
                return false;
            } else {
                return true;
            }
        }
        return false;
    });
    return added_S_polys > 0;
}

pub fn f4<P, O>(ring: P, basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let row_poly = |i: usize, A: &SparseMatrix<&<P::Type as RingExtension>::BaseRing>, columns: &MonomialSet<P, O>| ring.from_terms(
        A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), columns.at_index(j)))
    );

    let mut A = SparseMatrix::new(ring.base_ring(), 0, 0, None.into_iter());

    // the monomials belonging to the columns of A, in the same order
    let mut columns = MonomialSet::new(order);

    // the S-polynomials we still have to add
    let mut to_process = basis.iter().map(|f| ring.lm(f, order).unwrap()).flat_map(|m1| basis.iter().map(|f| (m1.clone(), ring.lm(f, order).unwrap().clone()))).collect::<Vec<_>>();

    let mut reduced_basis = reduce(&ring, basis, order).0;
    reduced_basis.sort_by(|l, r| order.compare(ring.lm(l, order).unwrap(), ring.lm(r, order).unwrap()));
    for b in reduced_basis {
        let mut len = A.row_count();
        add_poly_to_matrix(&ring, b, &mut A, &mut columns, order, &mut len);
    }
    
    // A set of monomials such that the rows with these leading monomials already suffice to reduce all other rows to 0 via multivariate division
    // A is in echelon form, so there is at most one row for any leading monomial
    let mut current_monomial_basis: MonomialSet<P, O> = MonomialSet::new(order);

    let mut degree_bound = 0;

    while to_process.len() != 0 {

        #[cfg(debug_assertions)]
        {
            for i in 1..A.row_count() {
                assert!(order.compare(leading_monomial(i - 1, &A, &columns).unwrap(), leading_monomial(i, &A, &columns).unwrap()) == std::cmp::Ordering::Less);
            }
        }

        println!("Current basis");
        for i in 0..A.row_count() {
            if let Some(m) = leading_monomial(i, &A, &columns) {
                if let Some(_) = current_monomial_basis.index_of(m) {
                    println!("{}, ", ring.format(&row_poly(i, &A, &columns)));
                }
            }
        }
        println!();

        // we either reduce the basis or add S-polys, doing both causes polys to be forgotten
        if !reduce_basis_polys(&ring, &mut current_monomial_basis, &mut A, &mut columns, order) {
            if !add_S_polys(&ring, &mut to_process, &current_monomial_basis, &mut A, &mut columns, order, degree_bound) {
                println!("Increase degree bound to {}", degree_bound);
                degree_bound += 1;
            }
        }

        gb_rowrev_sparse_row_echelon(&mut A);
        A.drop_zero_rows();

        for i in 0..A.row_count() {
            if let Some(m) = leading_monomial(i, &A, &columns) {
                println!("{:?}", m);
                if current_monomial_basis.iter().all(|x| !x.divides(&m)) {
                    println!("add {:?}", m);
                    to_process.extend(current_monomial_basis.iter().cloned()
                        .map(|other_m| (m.clone(), other_m))
                        .filter(|(m1, m2)| m1 != m2 && !m1.is_coprime(m2))
                    );
                    current_monomial_basis.add(m.clone());
                }
            }
        }
    }

    let mut result = Vec::new();
    for i in 0..A.row_count() {
        if let Some(m) = leading_monomial(i, &A, &columns) {
            debug_assert!(result.iter().all(|f| *ring.lm(f, order).unwrap() != *m));
            if current_monomial_basis.iter().any(|x| x == m) {
                result.push(row_poly(i, &A, &columns));
            }
        }
    }
    return result;
}

pub fn f4_old<P, O>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
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
        if ring.lm(f, order).unwrap().clone().lcm(ring.lm(g, order).unwrap()).deg() as usize <= degree_bound {
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

        let new_polys = reduce_S_matrix(&ring, &S_polys, &basis, order);

        if new_polys.len() == 0 {
            degree_bound += 1;
        } else {
            basis.extend(new_polys.into_iter());
            basis = reduce(&ring, basis, order).0;
            open = (0..basis.len()).flat_map(|i| (0..i).map(move |j| (i, j)))
                .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
                .collect::<Vec<_>>();
        }
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
    let mut changed_last_iter = true;
    let mut changed_at_all = false;
    while changed_last_iter {
        changed_last_iter = false;
        for i in 0..polys.len() {
            let f = ring.clone_el(&polys[i]);
            let reduced = multivariate_division(&ring, f, (&polys[..i]).chain(&polys[(i + 1)..]), order);
            if !ring.eq_el(&polys[i], &reduced) {
                changed_last_iter = true;
                changed_at_all = true;
                if ring.is_zero(&reduced) {
                    polys.remove(i);
                } else {
                    polys[i] = reduced;
                }
                break;
            }
        }
    }
    return (polys, changed_at_all);
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

    let actual = f4(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)], order);

    let expected = ring.from_terms([
        (16, &Monomial::new([0, 3])),
        (15, &Monomial::new([1, 0])),
        (1, &Monomial::new([0, 1])),
    ].into_iter());

    assert_eq!(3, actual.len());
    assert_el_eq!(&ring, &f2, actual.at(0));
    assert_el_eq!(&ring, &f1, actual.at(1));
    assert_el_eq!(&ring, &ring.negate(expected), actual.at(2));
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

    let actual = f4(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);

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

    let actual = f4(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);
    
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
    let order = BlockLexDegRevLex::new(..1, 1..);
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

    for f in &basis {
        println!("{}, ", ring.format(f));
    }

    let start = std::time::Instant::now();
    let gb1 = f4(&ring, basis.iter().map(|f| ring.clone_el(f)).collect(), order);
    std::hint::black_box(&gb1);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    let start = std::time::Instant::now();
    let gb2 = f4_old(&ring, basis, order);
    let end = std::time::Instant::now();

    println!("Old algorithm took {} ms", (end - start).as_millis());

    for f in &gb1 {
        assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, ring.clone_el(f), &gb2, order));
    }
    for f in &gb2 {
        assert_el_eq!(&ring, &ring.zero(), &multivariate_division(&ring, ring.clone_el(f), &gb1, order));
    }
}