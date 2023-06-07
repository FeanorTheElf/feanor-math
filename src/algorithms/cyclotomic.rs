use crate::divisibility::*;
use crate::primitive_int::StaticRing;
use crate::rings::poly::*;
use crate::ring::*;
use crate::algorithms;

pub fn cyclotomic_polynomial<P>(P: P, n: usize) -> El<P>
    where P: PolyRingStore, P::Type: PolyRing + DivisibilityRing
{
    let mut current = P.sub(P.indeterminate(), P.one());
    let ZZ = StaticRing::<i64>::RING;
    for (p, e) in algorithms::int_factor::factor(&ZZ, n as i64) {
        let pe = ZZ.pow(p, e) as usize;
        let p_e_minus_one = ZZ.pow(p, e - 1) as usize;
        current = P.checked_div(
            &P.from_terms(P.terms(&current).map(|(c, d)| (P.base_ring().clone_el(c), d * pe))), 
            &P.from_terms(P.terms(&current).map(|(c, d)| (P.base_ring().clone_el(c), d * p_e_minus_one))), 
        ).unwrap();
    }
    return current;
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_static::Zn;

#[test]
pub fn test_cyclotomic_polynomial() {
    let poly_ring = DensePolyRing::new(Zn::<7>::RING, "X");
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 2)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 3)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 4)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 5)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 2), (6, 1), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 6)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([(1, 6), (6, 3), (1, 0)].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 18)
    ));
    assert!(poly_ring.eq_el(
        &poly_ring.from_terms([
            (1, 48), (1, 47), (1, 46), (6, 43), (6, 42), (5, 41), (6, 40), (6, 39), (1, 36), (1, 35), (1, 34), (1, 33), (1, 32), (1, 31), (6, 28), (6, 26), (6, 24), 
            (6, 22), (6, 20), (1, 17), (1, 16), (1, 15), (1, 14), (1, 13), (1, 12), (6, 9), (6, 8), (5, 7), (6, 6), (6, 5), (1, 2), (1, 1), (1, 0)
        ].into_iter()),
        &cyclotomic_polynomial(&poly_ring, 105)
    ));
}