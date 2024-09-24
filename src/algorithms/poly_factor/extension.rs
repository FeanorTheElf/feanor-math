use crate::algorithms;
use crate::algorithms::poly_squarefree::poly_squarefree_part_global;
use crate::algorithms::poly_factor::FactorPolyField;
use crate::compute_locally::InterpolationBaseRing;
use crate::divisibility::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::ordered::OrderedRingStore;
use crate::perfect::PerfectField;
use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRingStore;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::extension::FreeAlgebraStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::integer::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;

fn factor_squarefree_over_extension<P>(LX: P, f: El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FactorPolyField + InterpolationBaseRing
{
    let L = LX.base_ring();
    let K = L.base_ring();

    assert!(LX.base_ring().is_one(LX.lc(&f).unwrap()));

    let KX = DensePolyRing::new(K, "X");
    // Y will take the role of the ring generator `theta` and `X` remains the indeterminate
    let KXY = DensePolyRing::new(KX.clone(), "Y");

    let Norm = |f: El<P>| {
        let f_over_KY = <_ as RingStore>::sum(&KXY,
            LX.terms(&f).map(|(c, i)| {
                let mut result = L.poly_repr(&KXY, c, KX.inclusion());
                KXY.inclusion().mul_assign_map(&mut result, KX.pow(KX.indeterminate(), i));
                result
            })
        );
        let gen_poly = L.generating_poly(&KXY, KX.inclusion());
    
        return algorithms::resultant::resultant_local::<&DensePolyRing<_>>(&KXY, f_over_KY, gen_poly);
    };

    // we want to find `k` such that `N(f(X + k theta))` remains square-free, where `theta` generates `L`;
    // there are only finitely many `k` that don't work, by the following idea:
    // Write `f = f1 ... fr` for the factorization of `f`, where `r <= d`. Then `N(f(X + k theta))` is not
    // square-free, iff there exist `i != j` and `sigma` such that `fi(X + k theta) = sigma(fj(X + k theta)) = sigma(fj)(X + k sigma(theta))`.
    // Now note that for given `i, j, sigma`, `fi(X + k theta) = sigma(fj)(X + k sigma(theta))` for at most one `k`, as otherwise
    // `fi(a (k1 - k2) theta) sigma(theta)) = sigma(fj)(a (k1 - k2) sigma(theta))` for all `a in Z` (evaluation is ring hom).
    //
    // Now note that `fi(X) - sigma(fj(X))` is a linear combination of `X^m` and `sigma(X^m)`, i.e. of `deg(fi) + deg(fj) <= d`
    // basic functions. The vectors `(a1 theta)^m, ..., (al theta)^m` for `m <= deg(fi)` and `sigma(a1 theta)^m, ..., sigma(al theta)^m`
    // for `m <= deg(fj)` are all linearly independent (assuming `a1, ..., al` distinct and `l >= deg(fi) + deg(fj) + 2`), 
    // thus `char(K) >= d + 2` (or `char(K) = 0`) implies that `fi = fj = 0`, a contradiction.
    //
    // Thus we find that there are at most `d(d + 1)/2 * [L : K]` many `k` such that `N(f(X + k theta))` is not squarefree.
    // This would even work if `k` is not an integer, but any element of `K`. Note that we still require `char(K) >= d + 2` for
    // the previous paragraph to work.

    let ZZbig = BigIntRing::RING;
    let characteristic = K.characteristic(&ZZbig).unwrap();
    // choose bound about twice as large as necessary, so the probability of succeeding is almost 1/2
    let bound = LX.degree(&f).unwrap() * LX.degree(&f).unwrap() * L.rank();
    assert!(ZZbig.is_zero(&characteristic) || ZZbig.is_geq(&characteristic, &int_cast(bound as i64, ZZbig, StaticRing::<i64>::RING)));

    let mut rng = oorandom::Rand64::new(1);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let k = StaticRing::<i32>::RING.get_uniformly_random(&(bound as i32), || rng.rand_u64());
        let lin_transform = LX.from_terms([(L.mul(L.canonical_gen(), L.int_hom().map(k)), 0), (L.one(), 1)].into_iter());
        let f_transformed = LX.evaluate(&f, &lin_transform, &LX.inclusion());

        let norm_f_transformed = Norm(LX.clone_el(&f_transformed));
        let degree = KX.degree(&norm_f_transformed).unwrap();
        let squarefree_part = poly_squarefree_part_global(&KX, norm_f_transformed);

        if KX.degree(&squarefree_part).unwrap() == degree {
            let lin_transform_rev = LX.from_terms([(L.mul(L.canonical_gen(), L.int_hom().map(-k)), 0), (L.one(), 1)].into_iter());
            let (factorization, _unit) = <_ as FactorPolyField>::factor_poly(&KX, &squarefree_part);
            return factorization.into_iter().map(|(factor, e)| {
                assert!(e == 1);
                let mut f_factor = LX.extended_ideal_gen(&f_transformed, &LX.lifted_hom(&KX, L.inclusion()).map(factor)).2;
                let lc_inv = L.div(&L.one(), LX.lc(&f_factor).unwrap());
                LX.inclusion().mul_assign_map(&mut f_factor, lc_inv);
                return LX.evaluate(&f_factor, &lin_transform_rev, &LX.inclusion());
            }).collect();
        }
    }
    unreachable!()
}

///
/// Factors a polynomial with coefficients in a field `K` that is a simple, finite-degree
/// field extension of a base field that supports polynomial factorization. 
/// 
#[stability::unstable(feature = "enable")]
pub fn factor_over_extension<P>(poly_ring: P, f: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FreeAlgebra + Field + PerfectField + SpecializeToFiniteField + SpecializeToFiniteRing,
        <<<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + FactorPolyField + InterpolationBaseRing
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
        let mut squarefree_part = poly_squarefree_part_global(KX, KX.clone_el(&current));
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

use super::SpecializeToFiniteField;
use super::SpecializeToFiniteRing;

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