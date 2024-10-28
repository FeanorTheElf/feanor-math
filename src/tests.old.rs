
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::finite::*;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::rings::extension::galois_field::GaloisField;

use super::poly_gcd::IntegerPolyGCDRing;
use super::poly_gcd::PolyGCDRing;

#[test]
fn random_test_factor_poly_galois_field() {
    let mut rng = oorandom::Rand64::new(1);
    let Fq = GaloisField::new(17, 5);
    let ring = DensePolyRing::new(&Fq, "X");

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f1 = ring.from_terms((0..8).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f2 = ring.from_terms((0..10).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f3 = ring.from_terms((0..10).map(|i| (Fq.random_element(|| rng.rand_u64()), i)));
        let f = ring.mul_ref_fst(&f1, ring.mul_ref(&f2, &f3));

        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&ring, &f);

        let product = ring.inclusion().mul_map(ring.prod(factorization.iter().map(|(g, e)| ring.pow(ring.clone_el(g), *e))), unit);
        assert_el_eq!(&ring, &f, &product);
        assert!(factorization.iter().map(|(_, e)| *e).sum::<usize>() >= 3);
        for (g, _) in &factorization {
            assert!(ring.checked_div(&f1, g).is_some() || ring.checked_div(&f2, g).is_some() || ring.checked_div(&f3, g).is_some());
        }
    }
}

#[test]
fn random_test_factor_rational_poly() {
    let mut rng = oorandom::Rand64::new(1);
    let ZZbig = BigIntRing::RING;
    let QQ = RationalField::new(ZZbig);
    let ring = DensePolyRing::new(&QQ, "X");
    let coeff_bound = ZZbig.int_hom().map(10);

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let f1 = ring.from_terms((0..8).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f2 = ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f3 = ring.from_terms((0..10).map(|i| (QQ.inclusion().map(ZZbig.get_uniformly_random(&coeff_bound, || rng.rand_u64())), i)));
        let f = ring.mul_ref_fst(&f1, ring.mul_ref(&f2, &f3));

        let (factorization, unit) = <_ as FactorPolyField>::factor_poly(&ring, &f);

        let product = ring.inclusion().mul_map(ring.prod(factorization.iter().map(|(g, e)| ring.pow(ring.clone_el(g), *e))), unit);
        assert_el_eq!(&ring, &f, &product);
        assert!(factorization.iter().map(|(_, e)| *e).sum::<usize>() >= 3);
        for (g, _) in &factorization {
            assert!(ring.checked_div(&f1, g).is_some() || ring.checked_div(&f2, g).is_some() || ring.checked_div(&f3, g).is_some());
        }
    }
}

#[bench]
fn bench_factor_random_poly(bencher: &mut Bencher) {
    let degree = 128;
    let Fp = zn_64::Zn::new(17).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(Fp, "X");
    let mut rng = oorandom::Rand64::new(1);
    let f = poly_ring.from_terms((0..degree).map(|d| (Fp.random_element(|| rng.rand_u64()), d)).chain([(Fp.one(), degree)].into_iter()));
    bencher.iter(|| {
        let (factors, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &f);
        assert_el_eq!(&Fp, unit, Fp.one());
        assert_el_eq!(poly_ring, &f, poly_ring.prod(factors.into_iter().map(|(g, e)| poly_ring.pow(g, e))));
    });
}

#[bench]
fn bench_factor_rational_poly_old(bencher: &mut Bencher) {
    let QQ = RationalField::new(BigIntRing::RING);
    let incl = QQ.int_hom();
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let f1 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 2), (incl.map(3), 4), (incl.map(1), 8)].into_iter());
    let f2 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(2), 1), (incl.map(1), 2), (incl.map(1), 4), (incl.map(1), 5), (incl.map(1), 10)].into_iter());
    let f3 = poly_ring.from_terms([(incl.map(1), 0), (incl.map(1), 1), (incl.map(-2), 5), (incl.map(1), 17)].into_iter());
    bencher.iter(|| {
        let (actual, unit) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly_ring.prod([poly_ring.clone_el(&f1), poly_ring.clone_el(&f1), poly_ring.clone_el(&f2), poly_ring.clone_el(&f3), poly_ring.int_hom().map(9)].into_iter()));
        assert_eq!(3, actual.len());
        assert_el_eq!(QQ, QQ.int_hom().map(9), unit);
        for (f, e) in actual.into_iter() {
            if poly_ring.eq_el(&f, &f1) {
                assert_el_eq!(poly_ring, f1, f);
                assert_eq!(2, e);
            } else if poly_ring.eq_el(&f, &f2) {
                assert_el_eq!(poly_ring, f2, f);
                assert_eq!(1, e);
           } else if poly_ring.eq_el(&f, &f3) {
               assert_el_eq!(poly_ring, f3, f);
               assert_eq!(1, e);
            } else {
                panic!("Factorization returned wrong factor {} of ({})^2 * ({}) * ({})", poly_ring.format(&f), poly_ring.format(&f1), poly_ring.format(&f2), poly_ring.format(&f3));
            }
        }
    });
}

#[cfg(test)]
use crate::rings::zn::*;

#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::rational::RationalField;

#[test]
fn test_factor_number_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let ZZ_to_QQ = QQ.int_hom();

    // a quadratic field

    let K = FreeAlgebraImpl::new(&QQ, 2, [ZZ_to_QQ.map(-1), ZZ_to_QQ.map(-1)]).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());

    let (factorization, unit) = poly_factor_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(2, factorization.len());
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a galois field of degree 3

    let K = FreeAlgebraImpl::new(&QQ, 3, [ZZ_to_QQ.map(1), ZZ_to_QQ.map(2), ZZ_to_QQ.map(-1)]).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());

    let (factorization, unit) = poly_factor_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(3, factorization.len());
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a non-galois field (not normal) of degree 3

    let K = FreeAlgebraImpl::new(&QQ, 3, [ZZ_to_QQ.map(2), ZZ_to_QQ.map(0), ZZ_to_QQ.map(0)]).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());
    
    let (factorization, unit) = poly_factor_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(2, factorization.len());
    for (_, e) in &factorization {
        assert_eq!(1, *e);
    }
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));
}

#[test]
fn test_poly_squarefree_part_global() {
    let ring = DensePolyRing::new(zn_static::Fp::<257>::RING, "X");
    let poly = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let expected = ring.prod([
        ring.from_terms([(4, 0), (1, 1)].into_iter()),
        ring.from_terms([(6, 0), (1, 1)].into_iter()),
        ring.from_terms([(255, 0), (1, 1)].into_iter()),
        ring.from_terms([(8, 0), (1, 1)].into_iter())
    ].into_iter());
    let actual = poly_squarefree_part_global(&ring, &poly);
    assert_el_eq!(ring, expected, actual);

    let QQ = RationalField::new(BigIntRing::RING);
    let poly_ring = DensePolyRing::new(&QQ, "X");
    let [mut f, mut expected] = poly_ring.with_wrapped_indeterminate(|X| [
        16 - 32 * X + 104 * X.pow_ref(2) - 8 * 11 * X.pow_ref(3) + 121 * X.pow_ref(4),
        4 - 4 * X + 11 * X.pow_ref(2)
    ]);
    poly_ring.inclusion().mul_assign_map(&mut f, QQ.div(&QQ.one(), &QQ.int_hom().map(121)));
    poly_ring.inclusion().mul_assign_map(&mut expected, QQ.div(&QQ.one(), &QQ.int_hom().map(11)));

    let actual = poly_squarefree_part_global(&poly_ring, &f);
    assert_el_eq!(poly_ring, expected, actual);
}

#[test]
fn test_find_factor_by_extension_finite_field() {
    let field = zn_64::Zn::new(2).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    assert!(<_ as FactorPolyField>::find_factor_by_extension(&poly_ring, FreeAlgebraImpl::new(field, 2, [field.one(), field.one()])).is_none());

    let poly = poly_ring.mul(
        poly_ring.from_terms([(field.one(), 0), (field.one(), 1), (field.one(), 3)].into_iter()),
        poly_ring.from_terms([(field.one(), 0), (field.one(), 2), (field.one(), 3)].into_iter()),
    );
    let factor = <_ as FactorPolyField>::find_factor_by_extension(&poly_ring, FreeAlgebraImpl::new(field, 6, [field.one(), field.one(), field.one(), field.one(), field.one(), field.one()])).unwrap();
    assert_eq!(3, poly_ring.degree(&factor).unwrap());
    assert!(poly_ring.checked_div(&poly, &factor).is_some());
}