use crate::divisibility::{DivisibilityRingStore, Domain};
use crate::pid::{PrincipalIdealRing, PrincipalIdealRingStore};
use crate::rings::poly::*;
use crate::algorithms;
use crate::ring::*;

pub fn resultant<P>(ring: P, mut f: El<P>, mut g: El<P>) -> El<<P::Type as RingExtension>::BaseRing>
    where P: PolyRingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Domain + PrincipalIdealRing
{
    let base_ring = ring.base_ring();
    let mut scale_den = base_ring.one();
    let mut scale_num = base_ring.one();

    if ring.is_zero(&g) || ring.degree(&g).unwrap() < ring.degree(&f).unwrap_or(0) {
        if ring.is_zero(&f) {
            return base_ring.zero();
        }
        base_ring.negate_inplace(&mut scale_num);
        std::mem::swap(&mut f, &mut g);
    }

    while ring.degree(&f).unwrap_or(0) >= 1 {
        // use here that `res(f, g) = a^(-deg(f)) lc(f)^(deg(g) - deg(ag - fh)) res(f, ag - fh)` if `deg(fh) <= deg(g)`
        let deg_g = ring.degree(&g).unwrap();
        let (_q, r, a) = algorithms::poly_div::poly_div_domain(ring, g, &f);
        let deg_r = ring.degree(&r).unwrap_or(0);

        // adjust the scaling factor - we cancel out gcd's to prevent excessive number growth
        base_ring.mul_assign(&mut scale_den, base_ring.pow(a, ring.degree(&f).unwrap()));
        base_ring.mul_assign(&mut scale_num, base_ring.pow(base_ring.clone_el(ring.lc(&f).unwrap()), deg_g - deg_r));
        let gcd = base_ring.ideal_gen(&scale_den, &scale_num).2;
        scale_den = base_ring.checked_div(&scale_den, &gcd).unwrap();
        scale_num = base_ring.checked_div(&scale_num, &gcd).unwrap();

        g = f;
        f = r;
    }

    if ring.is_zero(&f) {
        return base_ring.zero();
    } else {
        let mut result = base_ring.clone_el(&ring.coefficient_at(&f, 0));
        result = base_ring.pow(result, ring.degree(&g).unwrap());
        base_ring.mul_assign(&mut result, scale_num);
        return base_ring.checked_div(&result, &scale_den).unwrap();
    }
}

#[cfg(test)]
use self::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::rings::multivariate::ordered::MultivariatePolyRingImpl;
#[cfg(test)]
use crate::rings::multivariate::*;
#[cfg(test)]
use crate::algorithms::f4::f4;
#[cfg(test)]
use crate::field::FieldStore;
#[cfg(test)]
use crate::default_memory_provider;

#[test]
fn test_resultant() {
    let ZZ = StaticRing::<i64>::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    // a quadratic polynomial and its derivative - the resultant should be the discriminant
    let f = ZZX.from_terms([(3, 0), (-5, 1), (1, 2)].into_iter());
    let g = ZZX.from_terms([(-5, 0), (2, 1)].into_iter());

    assert_el_eq!(&ZZ, &13, &resultant(&ZZX, ZZX.clone_el(&f), ZZX.clone_el(&g)));
    assert_el_eq!(&ZZ, &-13, &resultant(&ZZX, g, f));

    // if f and g have common factors, this should be zero
    let f = ZZX.from_terms([(1, 0), (-2, 1), (1, 2)].into_iter());
    let g = ZZX.from_terms([(-1, 0), (1, 2)].into_iter());
    assert_el_eq!(&ZZ, &0, &resultant(&ZZX, f, g));

    // a slightly larger example
    let f = ZZX.from_terms([(5, 0), (-1, 1), (3, 2), (1, 4)].into_iter());
    let g = ZZX.from_terms([(-1, 0), (-1, 2), (1, 3), (4, 5)].into_iter());
    assert_el_eq!(&ZZ, &642632, &resultant(&ZZX, f, g));
}

#[test]
fn test_resultant_polynomial() {
    let ZZ = StaticRing::<i64>::RING;
    let QQ = RationalField::new(ZZ);
    let QQX = DensePolyRing::new(QQ, "X");
    let QQXY = DensePolyRing::new(QQX.clone(), "Y");
    let ZZ_to_QQ = QQ.inclusion();

    // 1 + X^2 + 2 Y + (1 + X) Y^2
    let f= QQXY.from_terms([
        (vec![(1, 0), (1, 2)], 0),
        (vec![(2, 0)], 1),
        (vec![(1, 0), (1, 1)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    // 3 + X + (2 + X) Y + (1 + X + X^2) Y^2
    let g = QQXY.from_terms([
        (vec![(3, 0), (1, 1)], 0),
        (vec![(2, 0), (1, 1)], 1),
        (vec![(1, 0), (1, 1), (1, 2)], 2)
    ].into_iter().map(|(v, i)| (QQX.from_terms(v.into_iter().map(|(c, j)| (ZZ_to_QQ.map(c), j))), i)));

    let mut actual = resultant(&QQXY, QQXY.clone_el(&f), QQXY.clone_el(&g));
    let actual_lc_inv = QQ.div(&QQ.one(), QQX.lc(&actual).unwrap());
    QQX.inclusion().mul_assign_map(&mut actual, actual_lc_inv);

    let QQYX: MultivariatePolyRingImpl<_, _, _, 2> = MultivariatePolyRingImpl::new(QQ, Lex, default_memory_provider!());
    let f = QQYX.from_terms([
        (1, Monomial::new([0, 0])),
        (1, Monomial::new([0, 2])),
        (2, Monomial::new([1, 0])),
        (1, Monomial::new([2, 0])),
        (1, Monomial::new([2, 1]))
    ].into_iter().map(|(c, m)| (ZZ_to_QQ.map(c), m)));

    let g = QQYX.from_terms([
        (3, Monomial::new([0, 0])),
        (1, Monomial::new([0, 1])),
        (2, Monomial::new([1, 0])),
        (1, Monomial::new([1, 1])),
        (1, Monomial::new([2, 0])),
        (1, Monomial::new([2, 1])),
        (1, Monomial::new([2, 2]))
    ].into_iter().map(|(c, m)| (ZZ_to_QQ.map(c), m)));

    let expected = f4::<_, _, false>(&QQYX, vec![f, g], Lex).into_iter().filter(|poly| QQYX.appearing_variables(&poly).len() == 1).collect::<Vec<_>>();
    assert!(expected.len() == 1);
    let mut expected = QQX.from_terms(QQYX.terms(&expected[0]).map(|(c, m)| (*c, m[1] as usize)));
    let expected_lc_inv = QQ.div(&QQ.one(), QQX.lc(&expected).unwrap());
    QQX.inclusion().mul_assign_map(&mut expected, expected_lc_inv);

    assert_el_eq!(&QQX, &expected, &actual);
}