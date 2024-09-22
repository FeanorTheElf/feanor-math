use crate::algorithms;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::divisibility::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;
use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRingStore;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::extension::FreeAlgebraStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::integer::BigIntRing;

use super::poly_squarefree_part;

fn factor_squarefree_over_extension<'a, P>(LX: &'a P, f: El<P>) -> impl 'a + Iterator<Item = El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField,
{
    let L = LX.base_ring();
    let K = L.base_ring();

    assert!(LX.base_ring().is_one(LX.lc(&f).unwrap()));

    let KX = DensePolyRing::new(K, "X");
    // Y will take the role of the ring generator `theta` and `X` remains the indeterminate
    let KXY = DensePolyRing::new(KX.clone(), "Y");

    let N = |f: El<P>| {
        let f_over_KY = <_ as RingStore>::sum(&KXY,
            LX.terms(&f).map(|(c, i)| {
                let mut result = L.poly_repr(&KXY, c, KX.inclusion());
                KXY.inclusion().mul_assign_map(&mut result, KX.pow(KX.indeterminate(), i));
                result
            })
        );
        let gen_poly = L.generating_poly(&KXY, KX.inclusion());
    
        algorithms::resultant::resultant(&KXY, f_over_KY, gen_poly)
    };

    let characteristic = K.characteristic(&BigIntRing::RING).unwrap();

    for k in 0.. {
        println!("Trying {}", k);
        // TODO: change `k` to random for a certain set, and make a fixed characteristic assertion
        assert!(BigIntRing::RING.is_zero(&characteristic) || BigIntRing::RING.is_lt(&BigIntRing::RING.int_hom().map(k), &characteristic));
        let lin_transform = LX.from_terms([(L.mul(L.canonical_gen(), L.int_hom().map(k)), 0), (L.one(), 1)].into_iter());
        let f_transformed = LX.evaluate(&f, &lin_transform, &LX.inclusion());
        let norm_f_transformed = N(LX.clone_el(&f_transformed));
        let degree = KX.degree(&norm_f_transformed).unwrap();

        println!("Computing square-free part");
        let squarefree_part = poly_squarefree_part(&KX, norm_f_transformed);
        println!("done");

        if KX.degree(&squarefree_part).unwrap() == degree {
            let lin_transform_rev = LX.from_terms([(L.mul(L.canonical_gen(), L.int_hom().map(-k)), 0), (L.one(), 1)].into_iter());
            let (factorization, _unit) = <_ as FactorPolyField>::factor_poly(&KX, &squarefree_part);
            return factorization.into_iter().map(move |(factor, e)| {
                assert!(e == 1);
                let mut f_factor = LX.extended_ideal_gen(&f_transformed, &LX.lifted_hom(&KX, L.inclusion()).map(factor)).2;
                let lc_inv = L.div(&L.one(), LX.lc(&f_factor).unwrap());
                LX.inclusion().mul_assign_map(&mut f_factor, lc_inv);
                return LX.evaluate(&f_factor, &lin_transform_rev, &LX.inclusion());
            });
        }
    }
    unreachable!()
}

#[stability::unstable(feature = "enable")]
pub fn factor_over_extension<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField,
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
        println!("Extracting square-free part");
        let mut squarefree_part = poly_squarefree_part(KX, KX.clone_el(&current));
        println!("done");
        let lc_inv = K.div(&K.one(), KX.lc(&squarefree_part).unwrap());
        KX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);
        current = KX.checked_div(&current, &squarefree_part).unwrap();

        for factor in factor_squarefree_over_extension(KX, squarefree_part) {
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

    let (factorization, unit) = factor_over_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(2, factorization.len());
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a galois field of degree 3

    let K = FreeAlgebraImpl::new(&QQ, 3, [ZZ_to_QQ.map(1), ZZ_to_QQ.map(2), ZZ_to_QQ.map(-1)]).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());

    let (factorization, unit) = factor_over_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(3, factorization.len());
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));

    // the case of a non-galois field (not normal) of degree 3

    let K = FreeAlgebraImpl::new(&QQ, 3, [ZZ_to_QQ.map(2), ZZ_to_QQ.map(0), ZZ_to_QQ.map(0)]).as_field().ok().unwrap();
    let KX = DensePolyRing::new(&K, "X");
    let poly = K.generating_poly(&KX, K.inclusion());
    
    let (factorization, unit) = factor_over_extension(&KX, &poly);
    assert_el_eq!(K, K.one(), unit);
    assert_eq!(2, factorization.len());
    for (_, e) in &factorization {
        assert_eq!(1, *e);
    }
    assert_el_eq!(KX, poly, KX.prod(factorization.into_iter().map(|(f, _e)| f)));
}