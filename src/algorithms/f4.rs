use crate::field::{FieldStore, Field};
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::vector::*;

use super::sparse_invert::{SparseMatrix, RowColTrackingMatrix, gb_sparse_row_echelon};

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
    gb_sparse_row_echelon(&mut A);
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

fn add_poly_to_matrix<P, O>(
    ring: &P, 
    poly: El<P>, 
    A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>, 
    columns: &mut Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>>, 
    order: O
) -> usize
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let monomial_column = |m: &Monomial<_>, columns: &[Monomial<_>]| columns.binary_search_by(|x: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>| order.compare(x, m).reverse()).unwrap();
    let column_monomial = |j: usize, columns: &[Monomial<_>]| columns[j].clone();
    let leading_monomial = |i: usize, A: &SparseMatrix<_>, columns: &[Monomial<_>]| A.get_row(i).nontrivial_entries().map(|(j, _)| j).min().map(|j| column_monomial(j, columns));
    let ensure_column_exists = |m: &Monomial<_>, columns: &mut Vec<Monomial<_>>, A: &mut SparseMatrix<&<P::Type as RingExtension>::BaseRing>| match columns.binary_search_by(|x: &Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>| order.compare(x, m).reverse()) {
        Ok(_) => false,
        Err(i) => {
            A.add_column(i);
            columns.insert(i, m.clone());
            true
        }
    };

    let mut new_monomials = Vec::new();

    for (_, m) in ring.terms(&poly) {
        if ensure_column_exists(&m, columns, A) {
            new_monomials.push(m.clone());
        }
    }

    while let Some(m) = new_monomials.pop() {
        if let Some(i) = (0..A.row_count()).filter(|i| leading_monomial(*i, &A, &columns[..]).map(|lm| lm.divides(&m)).unwrap_or(false)).next() {
            let quo_m = m.clone().div(&leading_monomial(i, &A, &columns[..]).unwrap());

            let new_row_index = A.row_count();
            A.add_row(new_row_index);

            let A_row = A.get_row(i).nontrivial_entries().map(|(j, c)| (column_monomial(j, &columns[..]), ring.base_ring().clone_el(c))).collect::<Vec<_>>();
            for (base_m, c) in A_row.into_iter() {
                let new_m = base_m.mul(&quo_m);
                if ensure_column_exists(&new_m, columns, A) {
                    new_monomials.push(new_m.clone());
                }
                A.set(new_row_index, monomial_column(&new_m, &columns[..]), c);
            }
        }
    }

    let new_row_index = A.row_count();
    A.add_row(new_row_index);

    for (c, m) in ring.terms(&poly) {
        A.set(new_row_index, monomial_column(m, &columns[..]), ring.base_ring().clone_el(c));
    }

    return new_row_index;
}

pub fn f4<P, O>(ring: P, basis: Vec<El<P>>, order: O) -> Vec<El<P>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: FieldStore,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
        O: MonomialOrder + Copy
{
    let leading_monomial = |i: usize, A: &SparseMatrix<&<P::Type as RingExtension>::BaseRing>, columns: &[Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>]| A.get_row(i).nontrivial_entries().map(|(j, _)| j).min().map(|j| columns[j].clone());
    let row_poly = |i: usize, A: &SparseMatrix<&<P::Type as RingExtension>::BaseRing>, columns: &[Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>]| ring.from_terms(
        A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), &columns[j]))
    );

    let mut A = SparseMatrix::new(ring.base_ring(), 0, 0, None.into_iter());
    let mut columns: Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>> = Vec::new();

    let mut new_element_row_indices = Vec::new();
    for b in basis {
        new_element_row_indices.push(add_poly_to_matrix(&ring, b, &mut A, &mut columns, order));
    }
    
    // A set of monomials such that the rows with these leading monomials already suffice to reduce all other rows to 0 via multivariate division
    let mut current_basis_monomials: Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>> = Vec::new();

    while new_element_row_indices.len() != 0 {

        // this basically reduces the current basis with each other
        let mut deleted_monomials = Vec::new();
        let mut index1 = 0;
        while index1 < current_basis_monomials.len() {
            let potential_divisor = current_basis_monomials[index1].clone();
            current_basis_monomials.retain(|m| {
                if *m != potential_divisor && potential_divisor.divides(m) {
                    println!("Removing {} from basis", ring.format(&ring.monomial(m)));
                    deleted_monomials.push(m.clone());
                    false
                } else {
                    true
                }
            });
            index1 += 1;
        }

        let mut basis = Vec::new();
        let original_row_count = A.row_count();
        for i in 0..original_row_count {
            if let Some(m) = leading_monomial(i, &A, &columns[..]) {
                if current_basis_monomials.iter().any(|x| *x == m) {
                    basis.push(i);
                    for deleted_m in deleted_monomials.iter().filter(|x| m.divides(x)) {
                        let quo_m = deleted_m.clone().div(&m);
                        let mut poly = row_poly(i, &A, &columns[..]);
                        ring.mul_monomial(&mut poly, &quo_m);
                        add_poly_to_matrix(&ring, poly, &mut A, &mut columns, order);
                    }
                }
            }
        }
        
        // now add S-polys
        for new_i in &new_element_row_indices {
            for old_i in &basis {
                if new_i != old_i {
                    let S_poly = S(&ring, &row_poly(*new_i, &A, &columns[..]), &row_poly(*old_i, &A, &columns[..]), order);
                    add_poly_to_matrix(&ring, S_poly, &mut A, &mut columns, order);
                }
            }
        }

        gb_sparse_row_echelon(&mut A);
        A.drop_zero_rows();
        new_element_row_indices.clear();

        // find the nontrivial new rows
        println!("New polys");
        for i in 0..A.row_count() {
            if let Some(m) = leading_monomial(i, &A, &columns[..]) {
                if current_basis_monomials.iter().all(|x: &Monomial<_>| !x.divides(&m)) {
                    new_element_row_indices.push(i);
                    current_basis_monomials.push(m.clone());
                    ring.println(&row_poly(i, &A, &columns[..]));
                }
            }
        }
        println!();
    }
    
    let mut result = Vec::new();
    for i in (0..A.row_count()).rev() {
        if let Some(m) = leading_monomial(i, &A, &columns[..]) {
            if current_basis_monomials.iter().any(|x| *x == m) {
                result.push(row_poly(i, &A, &columns[..]));
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

    for f in &actual {
        ring.println(f);
        println!();
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
    let gb = f4(&ring, basis, order);
    for f in &gb {
        ring.println(f);
    }
}