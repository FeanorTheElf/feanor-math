use crate::homomorphism::Homomorphism;
use crate::integer::IntegerRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::poly::{PolyRing, PolyRingStore};

fn invert_mod_xn<P>(poly_ring: P, poly: &El<P>, n: usize) -> El<P>
where
    P: RingStore,
    P::Type: PolyRing,
{
    let mut current = poly_ring.one();
    for i in 1..=StaticRing::<i64>::RING.abs_log2_ceil(&(n as i64)).unwrap() {
        let mut square = poly_ring.pow(poly_ring.clone_el(&current), 2);
        poly_ring.truncate_monomials(&mut square, 1 << i);
        let mut subtract = poly_ring.mul_ref_fst(poly, square);
        poly_ring.truncate_monomials(&mut subtract, 1 << i);
        poly_ring.int_hom().mul_assign_map(&mut current, 2);
        poly_ring.sub_assign(&mut current, subtract);
    }
    return current;
}

/// Computes the evaluation of the given polynomial at each of the given points.
///
/// This uses a multi-point evaluation algorithm, and will run in time `O(T(max(d, l)) log(max(d,
/// l)))`, where `T(d)` is the time required to multiply two degree-`d` polynomials in the given
/// polynomial ring.
#[stability::unstable(feature = "enable")]
pub fn multipointeval<P>(
    poly_ring: P,
    poly: &El<P>,
    points: &[El<<P::Type as RingExtension>::BaseRing>],
) -> Vec<El<<P::Type as RingExtension>::BaseRing>>
where
    P: RingStore,
    P::Type: PolyRing,
{
    let base_ring = poly_ring.base_ring();
    let mut point_polys = vec![
        points
            .iter()
            .map(|x| poly_ring.from_terms([(base_ring.one(), 0), (base_ring.negate(base_ring.clone_el(x)), 1)]))
            .collect::<Vec<_>>(),
    ];
    while point_polys.last().unwrap().len() > 1 {
        let new_polys = point_polys
            .last()
            .unwrap()
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    poly_ring.mul_ref(&chunk[0], &chunk[1])
                } else {
                    poly_ring.clone_el(&chunk[0])
                }
            })
            .collect::<Vec<_>>();
        point_polys.push(new_polys);
    }

    let deg_poly = poly_ring.degree(&poly).unwrap();
    let mut current = vec![(
        deg_poly,
        poly_ring.from_terms(
            poly_ring
                .terms(poly)
                .map(|(c, i)| (base_ring.clone_el(c), deg_poly - i)),
        ),
    )];
    while current.len() < points.len() {
        let polys = point_polys.pop().unwrap();
        assert!(polys.len() == 2 * current.len() || polys.len() == 2 * current.len() - 1);
        let new = polys
            .into_iter()
            .zip(current.iter().flat_map(|f| [f, f]))
            .map(|(modulus, (deg_poly, poly))| {
                let deg_modulus = 1 << point_polys.len();
                let deg_remainder = deg_modulus - 1;
                if deg_remainder > *deg_poly {
                    return (*deg_poly, poly_ring.clone_el(poly));
                }
                let poly_div_modulus = poly_ring.mul_ref_snd(invert_mod_xn(&poly_ring, &modulus, deg_poly + 1), poly);
                let deg_offset = deg_poly - deg_remainder;
                let mut remainder = poly_ring.from_terms((0..=deg_remainder).map(|i| {
                    (
                        base_ring.clone_el(poly_ring.coefficient_at(&poly_div_modulus, i + deg_offset)),
                        i,
                    )
                }));
                poly_ring.mul_assign(&mut remainder, modulus);
                poly_ring.truncate_monomials(&mut remainder, deg_remainder + 1);
                return (deg_remainder, remainder);
            })
            .collect();
        current = new;
    }
    assert_eq!(0, point_polys.len());
    assert_eq!(points.len(), current.len());
    return current
        .into_iter()
        .map(|(deg_rem, rem)| {
            assert!(deg_rem == 0);
            base_ring.clone_el(poly_ring.coefficient_at(&rem, 0))
        })
        .collect();
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;

#[test]
fn test_multipointeval() {
    let ZZX = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X + 1]);
    let result = multipointeval(&ZZX, &f, &[0, 1]);
    assert_eq!(1, result[0]);
    assert_eq!(2, result[1]);

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) + 2 * X + 1]);
    let result = multipointeval(&ZZX, &f, &[0, 1, 2, 3]);
    assert_eq!(1, result[0]);
    assert_eq!(4, result[1]);
    assert_eq!(21, result[2]);
    assert_eq!(88, result[3]);

    let FpX = DensePolyRing::new(Zn::new(65537), "X");
    let points = (0..99)
        .map(|x| FpX.base_ring().int_hom().map(x * x * x - 2 * x + 7))
        .collect::<Vec<_>>();
    let f = FpX.from_terms((0..100).map(|i| (FpX.base_ring().int_hom().map(i * i + 3 * i + 2), i as usize)));
    let result = multipointeval(&FpX, &f, &points);
    for i in 0..points.len() {
        assert_el_eq!(
            FpX.base_ring(),
            FpX.evaluate(&f, &points[i], FpX.base_ring().identity()),
            &result[i]
        );
    }
}
