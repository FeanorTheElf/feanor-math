use std::collections::BTreeSet;

use crate::field::{FieldStore, Field};
use crate::{ring::*, default_memory_provider};
use crate::rings::multivariate::*;
use crate::vector::*;

use super::sparse_invert::{SparseBaseMatrix, sparse_row_echelon, SparseWorkMatrix};

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
    let mut monomials: BTreeSet<FixedOrderMonomial<_, O>> = BTreeSet::new();
    for b in S_polys {
        for (_, m) in ring.terms(b) {
            for divisor in m.clone().dividing_monomials() {
                monomials.insert(FixedOrderMonomial::new(divisor, order));
            }
        }
    }
    let mut open = monomials.iter().map(|m| m.clone().into()).collect::<Vec<_>>();
    while let Some(m) = open.pop() {
        if let Some(f) = basis.iter().filter(|f| ring.terms)
    }
//     let columns = extract_monomials(&ring, S_polys.iter().chain(basis.iter()), order);
//     let columns_ref = &columns;
//     let ring_ref = &ring;
//     let mut A = SparseBaseMatrix::new(
//         ring.base_ring(),
//         S_polys.len(),
//         columns.len(),
//         S_polys.iter().enumerate().flat_map(move |(i, f)|
//             ring_ref.terms(f).map(move |(c, m)| {
//                 let row = i;
//                 let col = columns_ref.binary_search_by(|x| order.cmp(x, m).reverse()).unwrap();
//                 return (row, col, ring_ref.base_ring().clone_el(c));
//             })
//         )
//     );
//     let mut open = columns.clone();
//     while let Some(m) = open.pop() {
//         if let Some(f) = basis.iter().filter(|f| ring.lm(f, order).unwrap().divides(&m)).next() {
//             let i = A.row_count();
//             A.add_row(i);

//         }
//     }
//     for j in 0..A.col_count() {
//         print!("{:?}, ", columns[j]);
//     }
//     println!();
//     for i in 0..A.row_count() {
//         for j in 0..A.col_count() {
//             print!("{}, ", ring.base_ring().format(A.at(i, j)));
//         }
//         println!();
//     }
//     println!();
//     sparse_row_echelon(SparseWorkMatrix::new(&mut A));
//     let mut result = Vec::new();
//     for i in 0..A.row_count() {
//         if let Some((j, _)) = A.get_row(i).nontrivial_entries().min_by_key(|(j, _)| *j) {
//             if basis.iter().all(|f| ring.lm(f, order).unwrap() != &columns[j]) {
//                 result.push(ring.from_terms(
//                     A.get_row(i).nontrivial_entries().map(|(j, c)| (ring.base_ring().clone_el(c), &columns[j]))
//                 ))
//             }
//         }
//     }
//     return result;
    unimplemented!()
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
    let mut open = (0..basis.len()).flat_map(|i| (0..i).map(move |j| (i, j)))
        .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
        .collect::<Vec<_>>();
    let mut i = 0;
    while open.len() > 0 {
        let S_polys: Vec<_> = open.iter().cloned().map(|(i, j)| S(&ring, &basis[i], &basis[j], order)).collect();
        println!("Have {} S-polys", S_polys.len());
        for p in &S_polys {
            ring.println(p);
        }
        println!();
        open.clear();
        let new_polys = reduce_S_matrix(&ring, &S_polys, &basis, order);
        println!("Found {} new polynomials", new_polys.len());
        for p in &new_polys {
            ring.println(p);
        }
        println!();
        let old_len = basis.len();
        basis.extend(new_polys.into_iter());
        open.extend((0..basis.len()).flat_map(|i| (old_len..basis.len()).map(move |j| (i, j)))
            .filter(|(i, j)| filter_S_pair(&basis[*i], &basis[*j]))
        );
        i += 1;
        if i > 4 {
            return basis;
        }
    }
    return basis;
}

#[cfg(test)]
use crate::rings::multivariate::ordered::*;
#[cfg(test)]
use crate::rings::zn::zn_static;

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

    f4(ring, vec![f1, f2], order);
}