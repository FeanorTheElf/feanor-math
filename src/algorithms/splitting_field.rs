use std::alloc::Global;
use std::cmp::min;

use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::divisibility::DivisibilityRing;
use crate::homomorphism::{Homomorphism, LambdaHom};
use crate::matrix::OwnedMatrix;
use crate::ordered::OrderedRingStore;
use crate::pid::EuclideanRing;
use crate::primitive_int::StaticRing;
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
use crate::rings::extension::FreeAlgebraStore;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::finite::FiniteRingStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::{ring::*, MAX_PROBABILISTIC_REPETITIONS};
use crate::rings::rational::RationalField;
use crate::integer::*;
use crate::rings::zn::zn_64::Zn;
use crate::rings::zn::ZnRingStore;
use crate::seq::VectorFn;
use crate::algorithms::linsolve::LinSolveRingStore;

use super::linsolve::LinSolveRing;
use super::poly_factor::FactorPolyField;

#[stability::unstable(feature = "enable")]
pub fn splitting_field<P>(poly_ring: P, poly: El<P>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: FactorPolyField,
        for<'a> AsFieldBase<FreeAlgebraImpl<&'a <P::Type as RingExtension>::BaseRing, Vec<El<<P::Type as RingExtension>::BaseRing>>>>: FactorPolyField
{
    let base_ring = poly_ring.base_ring();
    let (mut factorization, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &poly);

}

type ThisPolyRing<'a, 'b, R> = DensePolyRing<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>;

fn extend_splitting_field<'a, 'b, R>(poly_ring: &ThisPolyRing<'a, 'b, R>, mut remaining_factors: Vec<(El<ThisPolyRing<'a, 'b, R>>, usize)>)
    where R: RingStore,
        R::Type: LinSolveRing + FactorPolyField,
        for<'c> AsFieldBase<FreeAlgebraImpl<&'c R, Vec<El<R>>>>: FactorPolyField
{
    if remaining_factors.len() == 0 {
        return;
    }
    let (factor, _) = remaining_factors.swap_remove(remaining_factors.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(&f.0).unwrap()).unwrap().0);
    let (mut sub_factorization, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &factor);
    let (largest_factor, _) = sub_factorization.swap_remove(sub_factorization.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(&f.0).unwrap()).unwrap().0);

    let (extension_embedding, root_of_new_poly) = extend_field(poly_ring, largest_factor);
    let new_ring = RingRef::new(extension_embedding.codomain().get_ring());
    let new_poly_ring = DensePolyRing::new(new_ring, "X");
    let lifted_hom = new_poly_ring.lifted_hom(poly_ring, &extension_embedding);
    let new_factorization = remaining_factors.into_iter().chain(sub_factorization.into_iter()).map(|(f, e)| (lifted_hom.map(f), e)).collect::<Vec<_>>();

    return extend_splitting_field(&new_poly_ring, new_factorization);
}

///
/// Builds the extension field. Assumes that `irred_poly` is irreducible, without checking.
/// 
fn extend_field<'a, 'b, 'c, R>(poly_ring: &'c ThisPolyRing<'a, 'b, R>, irred_poly: El<ThisPolyRing<'a, 'b, R>>) -> (
    impl 'c + Homomorphism<AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>,
    El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>
)
    where R: RingStore,
        R::Type: LinSolveRing + FactorPolyField,
        for<'d> AsFieldBase<FreeAlgebraImpl<&'d R, Vec<El<R>>>>: FactorPolyField,
        'a: 'b,
        'b: 'c
{
    assert!(!poly_ring.is_zero(&irred_poly) && poly_ring.degree(&irred_poly).unwrap() > 0);
    let base_ring = poly_ring.base_ring().base_ring();
    let ring = *poly_ring.base_ring();
    let extension_ring = FreeAlgebraImpl::new_with(
        ring, poly_ring.degree(&irred_poly).unwrap(), 
        (0..poly_ring.degree(&irred_poly).unwrap()).map(|i| ring.negate(ring.clone_el(poly_ring.coefficient_at(&irred_poly, i)))).collect::<Vec<_>>(),
        "X",
        Global,
        STANDARD_CONVOLUTION
    );
    
    let total_rank = ring.rank() * extension_ring.rank();

    // the main task is to find a primitive element of `extension_ring`, i.e. that generates it over `base_ring`.
    // we use the following approach:
    //  - Consider generator `a` of `K = ring` and `b` of `L = extension_ring` over `K`; Call the base ring `k`
    //  - Consider the set `A_n = { b, b + a, b + 2a, ..., b + (n - 1)a }` where `n <= char(k)`
    //  - Consider also the subset `B_n = { x in A_n | k[x] != L }`
    //  - The idea is now to choose a random element from `A_n`, and hope that it is not in `B_n`
    //  - To estimate the probability, consider the map `B_n -> { maximal proper subfields of L }` that maps any `x`
    //    to some maximal proper subfield containing `k(x)`
    //  - Then this map is injective, as any field containing `a + ib` and `a + jb`, `j > i` must contain `a, b`, thus be `L`
    //  - In other words, to find `n`, we need a bound on the maximal proper subfields
    //  - I believe the degree `[L : k]` is such a bound (in the Galois case it is, at least)

    // take `A` twice as large, so that we find a good element with probability >= 1/2
    let size_of_A = (2 * total_rank) as i32;
    let characteristic = base_ring.characteristic(&BigIntRing::RING).unwrap();

    if !BigIntRing::RING.is_zero(&characteristic) && BigIntRing::RING.is_lt(&characteristic, &int_cast(size_of_A as i64, BigIntRing::RING, StaticRing::<i64>::RING)) {
        // note further that there are some weird field extensions that are not simple, i.e. do not have a primitive element at all;
        // TODO: can we restrict this via traits?
        unimplemented!("The case that 2 * [extension_ring : base_ring] > char(base_ring) is currently not implemented");
    }

    let mut rng = oorandom::Rand64::new(1);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let a = StaticRing::<i32>::RING.get_uniformly_random(&size_of_A, || rng.rand_u64());
        let potential_generator = extension_ring.add(extension_ring.canonical_gen(), extension_ring.inclusion().map(ring.int_hom().mul_map(ring.canonical_gen(), a)));

        let mut lhs = OwnedMatrix::zero(total_rank, total_rank, base_ring);
        let mut current = extension_ring.one();
        for j in 0..total_rank {
            let current_wrt_basis = extension_ring.wrt_canonical_basis(&current);
            for i1 in 0..extension_ring.rank() {
                let c = current_wrt_basis.at(i1);
                let c_wrt_basis = ring.wrt_canonical_basis(&c);
                for i2 in 0..ring.rank() {
                    *lhs.at_mut(i1 * ring.rank() + i2, j) = c_wrt_basis.at(i2);
                }
            }
            extension_ring.mul_assign_ref(&mut current, &potential_generator);
        }

        let mut rhs = OwnedMatrix::zero(total_rank, 3, base_ring);
        let current_wrt_basis = extension_ring.wrt_canonical_basis(&current);
        for i1 in 0..extension_ring.rank() {
            let c = current_wrt_basis.at(i1);
            let c_wrt_basis = ring.wrt_canonical_basis(&c);
            for i2 in 0..ring.rank() {
                *rhs.at_mut(i1 * ring.rank() + i2, 0) = c_wrt_basis.at(i2);
            }
        }
        *rhs.at_mut(1, 1) = base_ring.one();
        *rhs.at_mut(ring.rank(), 2) = base_ring.one();

        let mut sol = OwnedMatrix::zero(total_rank, 3, base_ring);
    
        if base_ring.solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).is_solved() {

            let solution_modulus = (0..total_rank).map(|i| base_ring.clone_el(sol.at(i, 0))).collect::<Vec<_>>();
            let potential_result = FreeAlgebraImpl::new(*base_ring, total_rank, solution_modulus);
            let base_poly_ring = DensePolyRing::new(base_ring, "X");

            if let Ok(result) = potential_result.as_field() {

                let old_gen = result.sum((0..total_rank).map(|i| result.inclusion().mul_ref_snd_map(result.pow(result.canonical_gen(), i), sol.at(i, 1))));
                debug_assert!(result.is_zero(&base_poly_ring.evaluate(&ring.generating_poly(&base_poly_ring, &base_ring.identity()), &old_gen, &result.inclusion())));
                let embedding = LambdaHom::new(
                    ring,
                    result,
                    move |from, to, x| base_poly_ring.evaluate(&from.poly_repr(&base_poly_ring, &x, &base_ring.identity()), &old_gen, &to.inclusion())
                );
                    let new_gen = embedding.codomain().sum((0..total_rank).map(|i| embedding.codomain().inclusion().mul_ref_snd_map(embedding.codomain().pow(embedding.codomain().canonical_gen(), i), sol.at(i, 2))));
                debug_assert!(embedding.codomain().is_zero(&poly_ring.evaluate(&irred_poly, &new_gen, &embedding)));
                return (embedding, new_gen);
            }
        }
    }
    unreachable!()
}

#[test]
fn test_extend_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let ring = FreeAlgebraImpl::new_with(&QQ, 2, vec![QQ.neg_one(), QQ.zero()], "i", Global, STANDARD_CONVOLUTION).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(RingRef::new(ring.get_ring()), "X");

    // extend `QQ[i]` by `X^4 - i`
    let irred_poly = poly_ring.sub(poly_ring.pow(poly_ring.indeterminate(), 4), poly_ring.inclusion().map(ring.canonical_gen()));
    let (extension_field_embedding, x) = extend_field(&poly_ring, irred_poly);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(8, ext_field.rank());
    assert!(ext_field.get_ring().clone().unwrap_self().as_field().is_ok());
    assert_el_eq!(ext_field, ext_field.neg_one(), ext_field.pow(extension_field_embedding.map(ring.canonical_gen()), 2));
    assert_el_eq!(ext_field, extension_field_embedding.map(ring.canonical_gen()), ext_field.pow(x, 4));

    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding, 
        [(0, 0), (0, 1), (1, 0), (2, 0), (-1, 0), (0, -1), (-1, -1)].into_iter().map(|(a, b)| ring.from_canonical_basis([QQ.int_hom().map(a), QQ.int_hom().map(b)]))
    );

    // extend `QQ[i]` by `X^3 - 2`
    let [irred_poly] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 2]);
    let (extension_field_embedding, x) = extend_field(&poly_ring, irred_poly);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(6, ext_field.rank());
    assert!(ext_field.get_ring().clone().unwrap_self().as_field().is_ok());
    assert_el_eq!(ext_field, ext_field.neg_one(), ext_field.pow(extension_field_embedding.map(ring.canonical_gen()), 2));
    assert_el_eq!(ext_field, ext_field.int_hom().map(2), ext_field.pow(x, 3));
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding, 
        [(0, 0), (0, 1), (1, 0), (2, 0), (-1, 0), (0, -1), (-1, -1)].into_iter().map(|(a, b)| ring.from_canonical_basis([QQ.int_hom().map(a), QQ.int_hom().map(b)]))
    );
}

#[test]
fn test_extend_field_finite_field() {
    let Fp = Zn::new(31).as_field().ok().unwrap();
    let ring = FreeAlgebraImpl::new(&Fp, 3, vec![Fp.int_hom().map(3), Fp.neg_one()]).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(RingRef::new(ring.get_ring()), "X");

    // extend `GF(31^3)` by `X^5 + 7X + 28`
    let [irred_poly] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(5) + 7 * X + 28]);let (extension_field_embedding, x) = extend_field(&poly_ring, irred_poly);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(15, ext_field.rank());
    assert!(ext_field.get_ring().clone().unwrap_self().as_field().is_ok());
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding,
        ring.elements().step_by(1000)
    );
}