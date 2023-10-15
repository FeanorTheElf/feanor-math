use std::collections::BTreeSet;

use crate::field::{FieldStore, Field};
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::vector::*;

use super::sparse_invert::{SparseBaseMatrix, SparseWorkMatrix, gb_sparse_row_echelon};

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

fn extract_monomials<'a, P, O, I>(ring: P, polys: I, order: O) -> Vec<Monomial<<P::Type as MultivariatePolyRing>::MonomialVector>>
    where P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        O: MonomialOrder + Copy,
        I: Iterator<Item = &'a El<P>>,
        P: 'a
{
    let mut result = BTreeSet::new();
    for b in polys {
        for (_, m) in ring.terms(b) {
            for divisor in m.clone().dividing_monomials() {
                result.insert(FixedOrderMonomial::new(divisor, order));
            }
        }
    }
    return result.into_iter().rev().map(|x| x.into()).collect();
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
                if let Err(index) = columns.binary_search_by(|x| order.cmp(&x, &m)) {
                    columns.insert(index, m);
                }
            }
        }
    }
    let monomials_ref = &columns;
    let mut A = SparseBaseMatrix::new(
        ring.base_ring(),
        S_polys.len(),
        columns.len(),
        S_polys.iter().enumerate().flat_map(move |(i, f)|
        ring_ref.terms(f).map(move |(c, m)| {
                let row = i;
                let col = monomials_ref.binary_search_by(|x| order.cmp(x, m)).unwrap();
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
                let index = match columns.binary_search_by(|x| order.cmp(x, &final_monomial)) {
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
    reduction_monomials.sort_by(|l, r| order.cmp(l, r));

    // we have ordered the monomials in ascending order, but we want to reduce towards smaller ones
    A.reverse_cols();
    gb_sparse_row_echelon(SparseWorkMatrix::new(&mut A));
    A.reverse_cols();

    let mut result = Vec::new();
    for i in 0..A.row_count() {
        if let Some((j, _)) = A.get_row(i).nontrivial_entries().max_by(|(j1, _), (j2, _)| order.cmp(&columns[*j1], &columns[*j2])) {
            if reduction_monomials.binary_search_by(|x| order.cmp(x, &columns[j])).is_err() {
                result.push(ring.from_terms(
                    A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), &columns[j]))
                ))
            }
        }
    }
    return result;
}

pub fn f4<P, O>(ring: P, mut basis: Vec<El<P>>, order: O) -> Vec<El<P>>
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

        println!("Found {} new polynomials", new_polys.len());
        for p in &new_polys {
            ring.println(p);
        }
        println!();

        if new_polys.len() == 0 {
            degree_bound += 1;
        } else {
            let old_len = basis.len();
            basis.extend(new_polys.into_iter());
            open.extend((0..basis.len()).flat_map(|i| (old_len..basis.len()).map(move |j| (i, j)))
                .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
            );
        }
    }
    return basis;
}

#[cfg(test)]
use crate::rings::multivariate::ordered::*;
#[cfg(test)]
use crate::rings::zn::zn_static;
#[cfg(test)]
use crate::default_memory_provider;

#[test]
fn test_f4() {
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
    assert_el_eq!(&ring, &f1, actual.at(0));
    assert_el_eq!(&ring, &f2, actual.at(1));
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

    let actual = f4(&ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)], order);

    for f in &actual {
        println!("{},", ring.format(f));
    }
}