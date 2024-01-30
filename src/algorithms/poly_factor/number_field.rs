use crate::algorithms;
use crate::algorithms::poly_factor::cantor_zassenhaus;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::divisibility::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRingStore;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::extension::FreeAlgebraStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::rational::*;
use crate::vector::VectorView;

fn factor_squarefree_over_number_field<'a, P, I>(KX: &'a P, f: El<P>) -> impl 'a + Iterator<Item = El<P>>
    where P: PolyRingStore,
        El<I>: 'a,
        P::Type: PolyRing + EuclideanRing,
        I: IntegerRingStore,
        I::Type: IntegerRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>
{
    let K = KX.base_ring();
    let QQ = K.base_ring();

    assert!(KX.base_ring().is_one(KX.lc(&f).unwrap()));

    let QQX = DensePolyRing::new(QQ, "X");
    // Y will take the role of the ring generator `theta` and `X` remains the indeterminate
    let QQXY = DensePolyRing::new(QQX.clone(), "Y");

    let N = |f: El<P>| {
        let f_over_QQY = <_ as RingStore>::sum(&QQXY,
            KX.terms(&f).map(|(c, i)| {
                let mut result = K.poly_repr(&QQXY, c, QQX.inclusion());
                QQXY.inclusion().mul_assign_map(&mut result, QQX.pow(QQX.indeterminate(), i));
                result
            })
        );
        let gen_poly = K.generating_poly(&QQXY, QQX.inclusion());
    
        algorithms::resultant::resultant(&QQXY, f_over_QQY, gen_poly)
    };

    for k in 0.. {
        let lin_transform = KX.from_terms([(K.mul(K.canonical_gen(), K.int_hom().map(k)), 0), (K.one(), 1)].into_iter());
        let f_transformed = KX.evaluate(&f, &lin_transform, &KX.inclusion());
        let norm_f_transformed = N(KX.clone_el(&f_transformed));
        let degree = QQX.degree(&norm_f_transformed).unwrap();
        let squarefree_part = cantor_zassenhaus::poly_squarefree_part(&QQX, norm_f_transformed);
        if QQX.degree(&squarefree_part).unwrap() == degree {
            let lin_transform_rev = KX.from_terms([(K.mul(K.canonical_gen(), K.int_hom().map(-k)), 0), (K.one(), 1)].into_iter());
            let (factorization, _unit) = <RationalFieldBase<_> as FactorPolyField>::factor_poly(&QQX, &squarefree_part);
            return factorization.into_iter().map(move |(factor, e)| {
                assert!(e == 1);
                let mut f_factor = KX.ideal_gen(&f_transformed, &KX.lifted_hom(&QQX, K.inclusion()).map(factor)).2;
                let lc_inv = K.div(&K.one(), KX.lc(&f_factor).unwrap());
                KX.inclusion().mul_assign_map(&mut f_factor, lc_inv);
                return KX.evaluate(&f_factor, &lin_transform_rev, &KX.inclusion());
            });
        }
    }
    unreachable!()
}

pub fn factor_over_number_field<P, I>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        I: IntegerRingStore,
        I::Type: IntegerRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>
{
    let KX = &poly_ring;
    let K = KX.base_ring();

    // We use the approach outlined in Cohen's "a course in computational algebraic number theory".
    //  - Use square-free reduction to assume wlog that `f` is square-free
    //  - Observe that the factorization of `f` is the product over `gcd(f, g)` where `g` runs
    //    through the factors of `N(f)` over `QQ[X]` - assuming that `N(f)` is square-free!
    //    Here `N(f)` is the "norm" of `f`, i.e.
    //    the product `prod_sigma sigma(f)` where `sigma` runs through the embeddings `K -> CC`.
    //  - It is now left to actually compute `N(f)`, which is not so simple as we do not known the
    //    `sigma`. As it turns out, this is the resultant of `f` and `MiPo(theta)` where `theta`
    //    generates `K`

    assert!(!KX.is_zero(f));
    let mut result: Vec<(El<P>, usize)> = Vec::new();
    let mut current = KX.clone_el(f);
    while !KX.is_unit(&current) {
        let mut squarefree_part = cantor_zassenhaus::poly_squarefree_part(KX, KX.clone_el(&current));
        let lc_inv = K.div(&K.one(), KX.lc(&squarefree_part).unwrap());
        KX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);
        current = KX.checked_div(&current, &squarefree_part).unwrap();

        for factor in factor_squarefree_over_number_field(KX, squarefree_part) {
            if let Some((i, _)) = result.iter().enumerate().filter(|(_, f)| KX.eq_el(&f.0, &factor)).next() {
                result[i].1 += 1;
            } else {
                result.push((factor, 1));
            }
        }
    }
    return (result, K.clone_el(KX.coefficient_at(&current, 0)));
}

#[cfg(test)]
use crate::default_memory_provider;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;

#[test]
fn test_factor_number_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let ZZ_to_QQ = QQ.int_hom();

    // a quadratic field

    let K = FreeAlgebraImpl::new(QQ, [ZZ_to_QQ.map(-1), ZZ_to_QQ.map(-1)], default_memory_provider!()).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());

    let (factorization, unit) = factor_over_number_field(&KX, &poly);
    assert_el_eq!(&K, &K.one(), &unit);
    assert_eq!(2, factorization.len());
    assert_el_eq!(&KX, &poly, &KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a galois field of degree 3

    let K = FreeAlgebraImpl::new(QQ, [ZZ_to_QQ.map(1), ZZ_to_QQ.map(2), ZZ_to_QQ.map(-1)], default_memory_provider!()).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());

    let (factorization, unit) = factor_over_number_field(&KX, &poly);
    assert_el_eq!(&K, &K.one(), &unit);
    assert_eq!(3, factorization.len());
    assert_el_eq!(&KX, &poly, &KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a non-galois field (not normal) of degree 3

    let K = FreeAlgebraImpl::new(QQ, [ZZ_to_QQ.map(2), ZZ_to_QQ.map(0), ZZ_to_QQ.map(0)], default_memory_provider!()).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());
    
    let (factorization, unit) = factor_over_number_field(&KX, &poly);
    assert_el_eq!(&K, &K.one(), &unit);
    assert_eq!(2, factorization.len());
    for (_, e) in &factorization {
        assert_eq!(1, *e);
    }
    assert_el_eq!(&KX, &poly, &KX.prod(factorization.into_iter().map(|(f, _e)| f)));
}