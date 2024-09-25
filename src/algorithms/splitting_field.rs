use std::alloc::*;

use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::compute_locally::InterpolationBaseRing;
use crate::homomorphism::*;
use crate::matrix::OwnedMatrix;
use crate::ordered::OrderedRingStore;
use crate::field::*;
use crate::pid::EuclideanRing;
use crate::pid::PrincipalIdealRing;
use crate::primitive_int::StaticRing;
use crate::rings::extension::extension_impl::*;
use crate::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::finite::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::specialization::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::integer::*;
use crate::divisibility::*;
use crate::seq::*;
use crate::ring::*;
use crate::algorithms::linsolve::LinSolveRingStore;
use crate::delegate::DelegateRing;

use super::linsolve::LinSolveRing;
use super::poly_factor::FactorPolyField;
use super::unity_root::get_prim_root_of_unity_gen;

///
/// Computes the splitting field of `poly` over `poly_ring.base_ring()`.
/// 
/// The splitting field is a finite extension of the base field such that the polynomial
/// splits completely in it. The roots of polynomial are returned as elements of the splitting
/// field, and with multiplicity.
/// 
#[stability::unstable(feature = "enable")]
pub fn splitting_field<'a, P>(poly_ring: &'a P, poly: El<P>) -> (
    AsField<FreeAlgebraImpl<&'a <P::Type as RingExtension>::BaseRing, Vec<El<<P::Type as RingExtension>::BaseRing>>>>,
    Vec<(El<AsField<FreeAlgebraImpl<&'a <P::Type as RingExtension>::BaseRing, Vec<El<<P::Type as RingExtension>::BaseRing>>>>>, usize)>
)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PerfectField + InterpolationBaseRing + LinSolveRing + FactorPolyField + SpecializeToFiniteRing + SpecializeToFiniteField,
        for<'c> <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as InterpolationBaseRing>::ExtendedRingBase<'c>: Domain + PrincipalIdealRing
{
    let trivial_extension = AsField::from(AsFieldBase::promise_is_perfect_field(FreeAlgebraImpl::new(poly_ring.base_ring(), 1, vec![poly_ring.base_ring().one()])));
    let new_poly_ring = DensePolyRing::new(RingRef::new(trivial_extension.get_ring()), "X");
    let hom = new_poly_ring.lifted_hom(poly_ring, trivial_extension.inclusion());
    return extend_splitting_field(&new_poly_ring, vec![(hom.map(poly), 1)], Vec::new());
}

type ThisPolyRing<'a, 'b, R> = DensePolyRing<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>;

#[stability::unstable(feature = "enable")]
pub fn extend_splitting_field<'a, 'b, R>(poly_ring: &ThisPolyRing<'a, 'b, R>, mut remaining_factors: Vec<(El<ThisPolyRing<'a, 'b, R>>, usize)>, mut list_of_roots: Vec<(El<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>, usize)>) -> (
    AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>,
    Vec<(El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, usize)>
)
    where R: RingStore,
        R::Type: PerfectField + InterpolationBaseRing + LinSolveRing + FactorPolyField + SpecializeToFiniteField + SpecializeToFiniteRing,
        for<'c> <R::Type as InterpolationBaseRing>::ExtendedRingBase<'c>: Domain + PrincipalIdealRing
{
    let (factor, multiplicity_outer) = remaining_factors.swap_remove(remaining_factors.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(&f.0).unwrap()).unwrap().0);

    assert!(!poly_ring.is_zero(&factor));
    assert!(poly_ring.degree(&factor).unwrap() > 0);

    let (mut sub_factorization, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &factor);
    let (largest_factor, multiplicity_inner) = sub_factorization.swap_remove(sub_factorization.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(&f.0).unwrap()).unwrap().0);
    let multiplicity = multiplicity_outer * multiplicity_inner;

    if poly_ring.degree(&largest_factor).unwrap() == 1 {
        remaining_factors.extend(sub_factorization.into_iter().map(|(f, i)| (f, i * multiplicity_outer)));
        let root = poly_ring.base_ring().negate(poly_ring.base_ring().div(poly_ring.coefficient_at(&largest_factor, 0), poly_ring.coefficient_at(&largest_factor, 1)));
        list_of_roots.push((root, multiplicity));

        if remaining_factors.len() == 0 {
            let result = poly_ring.base_ring().get_ring().get_delegate();
            let clone_of_result = FreeAlgebraImpl::new(*result.base_ring(), result.rank(), result.x_pow_rank().as_iter().map(|a| result.base_ring().clone_el(a)).collect::<Vec<_>>());
            return (AsField::from(AsFieldBase::promise_is_perfect_field(clone_of_result)), list_of_roots);
        } else {
            return extend_splitting_field(poly_ring, remaining_factors, list_of_roots);
        }
    }

    let (extension_embedding, root_of_new_poly) = extend_field(poly_ring, &largest_factor);

    let new_ring = RingRef::new(extension_embedding.codomain().get_ring());
    let new_poly_ring = DensePolyRing::new(new_ring, "X");
    let lifted_hom = new_poly_ring.lifted_hom(poly_ring, &extension_embedding);

    let mut new_factorization = remaining_factors.into_iter()
        .chain(sub_factorization.into_iter())
        .map(|(f, e)| (lifted_hom.map(f), e))
        .collect::<Vec<_>>();
    new_factorization.push((new_poly_ring.checked_div(&lifted_hom.map(largest_factor), &new_poly_ring.from_terms([(new_ring.negate(new_ring.clone_el(&root_of_new_poly)), 0), (new_ring.one(), 1)])).unwrap(), multiplicity));

    let new_list_of_roots = list_of_roots.into_iter().map(|(a, i)| (extension_embedding.map(a), i)).chain([(root_of_new_poly, multiplicity)].into_iter()).collect::<Vec<_>>();

    if new_factorization.len() == 0 {
        return (extension_embedding.into_domain_codomain().1, new_list_of_roots);
    }

    return extend_splitting_field(&new_poly_ring, new_factorization, new_list_of_roots);
}

struct FiniteFieldCase<'a, 'b, 'c, R>
    where R: RingStore,
        R::Type: PerfectField,
        'a: 'b,
        'b: 'c
{
    extension_ring: &'c AsField<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, Vec<El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>>>
}

impl<'a, 'b, 'c, R> FiniteFieldOperation<AsFieldBase<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, Vec<El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>>>> for FiniteFieldCase<'a, 'b, 'c, R>
    where R: RingStore,
        R::Type: PerfectField,
        'a: 'b,
        'b: 'c
{
    type Output<'d> = El<AsField<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, Vec<El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>>>>
        where Self: 'd;

    fn execute<'d, F>(self, field: F) -> Self::Output<'d>
        where Self: 'd,
            F: 'd + RingStore,
            F::Type: FiniteRing + Field + LinSolveRing + CanIsoFromTo<AsFieldBase<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, Vec<El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>>>>
    {
        let unit_group_order = BigIntRing::RING.sub(field.size(&BigIntRing::RING).unwrap(), BigIntRing::RING.one());
        return field.can_iso(self.extension_ring).unwrap().map(get_prim_root_of_unity_gen(&field, &unit_group_order, BigIntRing::RING).unwrap());
    }
}

///
/// Constructs the field `F[X]/(f(X))` that is isomorphic to `(F[X]/(g(X))[Y]/(h(Y))`
/// where `F[X]/(g(X))` is the base ring of `poly_ring` and `h` is the given irreducible
/// polynomial over `F[X]/(g(X))`. **Warning**: `h` is assumed to be irreducible, without
/// this being checked! 
/// 
#[stability::unstable(feature = "enable")]
pub fn extend_field<'a, 'b, 'c, R>(poly_ring: &'c ThisPolyRing<'a, 'b, R>, irred_poly: &El<ThisPolyRing<'a, 'b, R>>) -> (
    LambdaHom<
        RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, 
        AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>, 
        impl 'c + Fn(&RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, &AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>, &El<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>>) -> El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>
    >,
    El<AsField<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>
)
    where R: RingStore,
        R::Type: PerfectField + LinSolveRing + FactorPolyField + SpecializeToFiniteRing + SpecializeToFiniteField,
        'a: 'b,
        'b: 'c
{
    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);

    let base_ring = poly_ring.base_ring().base_ring();
    let ring: RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>> = *poly_ring.base_ring();

    let extension_ring = AsField::<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, _>>::from(
        AsFieldBase::<FreeAlgebraImpl<RingRef<'b, AsFieldBase<FreeAlgebraImpl<&'a R, Vec<El<R>>>>>, _>>::promise_is_perfect_field(
            FreeAlgebraImpl::new_with(
                ring, 
                poly_ring.degree(&irred_poly).unwrap(), 
                (0..poly_ring.degree(&irred_poly).unwrap()).map(|i| ring.negate(ring.clone_el(poly_ring.coefficient_at(&irred_poly, i)))).collect::<Vec<_>>(),
                "X",
                Global,
                STANDARD_CONVOLUTION
            )
        )
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
    

    let mut rng = oorandom::Rand64::new(1);
    let mut solution = None;
    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {

        let potential_primitive_element = if let Ok(finite_field_result) = extension_ring.get_ring().specialize_finite_field(FiniteFieldCase { extension_ring: &extension_ring }) {
            finite_field_result
        } else if !BigIntRing::RING.is_zero(&characteristic) && BigIntRing::RING.is_lt(&characteristic, &int_cast(size_of_A as i64, BigIntRing::RING, StaticRing::<i64>::RING)) {
            panic!("The case that 2 * [extension_ring : base_ring] > char(base_ring) for an infinite field base_ring is currently not supported")
        } else {
            let a = StaticRing::<i32>::RING.get_uniformly_random(&size_of_A, || rng.rand_u64());
            extension_ring.add(extension_ring.canonical_gen(), extension_ring.inclusion().map(ring.int_hom().mul_map(ring.canonical_gen(), a)))
        };
    
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
            extension_ring.mul_assign_ref(&mut current, &potential_primitive_element);
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
        if ring.rank() > 1 {
            *rhs.at_mut(1, 1) = base_ring.one();
        } else {
            *rhs.at_mut(0, 1) = ring.wrt_canonical_basis(&ring.canonical_gen()).at(0);
        }
        *rhs.at_mut(ring.rank(), 2) = base_ring.one();
    
        let mut sol = OwnedMatrix::zero(total_rank, 3, base_ring);
        if base_ring.solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).is_solved() {
    
            let solution_modulus = (0..total_rank).map(|i| base_ring.clone_el(sol.at(i, 0))).collect::<Vec<_>>();
            let potential_result = FreeAlgebraImpl::new(*base_ring, total_rank, solution_modulus);
    
            if let Ok(result) = potential_result.as_field() {
                let old_gen = result.sum((0..total_rank).map(|i| result.inclusion().mul_ref_snd_map(result.pow(result.canonical_gen(), i), sol.at(i, 1))));
                let new_gen = result.sum((0..total_rank).map(|i| result.inclusion().mul_ref_snd_map(result.pow(result.canonical_gen(), i), sol.at(i, 2))));
                solution = Some((result, old_gen, new_gen));
                break;
            }
        }
    }

    let (result_ring, old_gen, new_gen) = solution.unwrap();
    let base_poly_ring = DensePolyRing::new(base_ring, "X");
    let generating_poly = ring.generating_poly(&base_poly_ring, &base_ring.identity());

    debug_assert!(result_ring.is_zero(&base_poly_ring.evaluate(&generating_poly, &old_gen, &result_ring.inclusion())));

    let embedding = LambdaHom::new(
        ring,
        result_ring,
        move |from, to, x| base_poly_ring.evaluate(&from.poly_repr(&base_poly_ring, &x, &base_ring.identity()), &old_gen, &to.inclusion())
    );
    debug_assert!(embedding.codomain().is_zero(&poly_ring.evaluate(&irred_poly, &new_gen, &embedding)));
    return (embedding, new_gen);
}

#[cfg(test)]
use crate::rings::zn::zn_64::*;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;
#[cfg(test)]
use crate::rings::rational::*;

#[test]
fn test_extend_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let ring = FreeAlgebraImpl::new_with(&QQ, 2, vec![QQ.neg_one(), QQ.zero()], "i", Global, STANDARD_CONVOLUTION).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(RingRef::new(ring.get_ring()), "X");

    // extend `QQ[i]` by `X^4 - i`
    let irred_poly = poly_ring.sub(poly_ring.pow(poly_ring.indeterminate(), 4), poly_ring.inclusion().map(ring.canonical_gen()));
    let (extension_field_embedding, x) = extend_field(&poly_ring, &irred_poly);
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
    let (extension_field_embedding, x) = extend_field(&poly_ring, &irred_poly);
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
    // extend `GF(2^3)` by `X^2 + X + 1`
    let Fp = Zn::new(2).as_field().ok().unwrap();
    let ring = FreeAlgebraImpl::new(&Fp, 2, vec![Fp.one(), Fp.one()]).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(RingRef::new(ring.get_ring()), "X");

    let [irred_poly] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) + X + 1]);
    let (extension_field_embedding, x) = extend_field(&poly_ring, &irred_poly);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(6, ext_field.rank());
    assert!(ext_field.get_ring().clone().unwrap_self().as_field().is_ok());
    assert_el_eq!(ext_field, ext_field.zero(), poly_ring.evaluate(&irred_poly, &x, &extension_field_embedding));
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding,
        ring.elements()
    );

    // extend `GF(31^3)` by `X^5 + 7X + 28`
    let Fp = Zn::new(31).as_field().ok().unwrap();
    let ring = FreeAlgebraImpl::new(&Fp, 3, vec![Fp.int_hom().map(3), Fp.neg_one()]).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(RingRef::new(ring.get_ring()), "X");

    let [irred_poly] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(5) + 7 * X + 28]);
    let (extension_field_embedding, x) = extend_field(&poly_ring, &irred_poly);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(15, ext_field.rank());
    assert!(ext_field.get_ring().clone().unwrap_self().as_field().is_ok());
    assert_el_eq!(ext_field, ext_field.zero(), poly_ring.evaluate(&irred_poly, &x, &extension_field_embedding));
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding,
        ring.elements().step_by(1000)
    );
}

#[test]
fn test_splitting_field() {
    let base_field = Zn::new(5).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(&base_field, "X");
    let [f] = poly_ring.with_wrapped_indeterminate(|X| [1 + 3 * X + 2 * X.pow_ref(3) + 3 * X.pow_ref(4) + X.pow_ref(5) + X.pow_ref(7)]);

    let (extension, roots) = splitting_field(&poly_ring, poly_ring.clone_el(&f));
    assert_eq!(6, extension.rank());
    assert_eq!(7, roots.iter().map(|(_, i)| i).sum::<usize>());
    assert_eq!(7, roots.len());

    for (x, _) in &roots {
        assert_el_eq!(&extension, extension.zero(), poly_ring.evaluate(&f, x, &extension.inclusion()));
    }
    
    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(7) + 3 * X.pow_ref(6) + X.pow_ref(5) + 3 * X.pow_ref(4) + 3 * X.pow_ref(3) + X.pow_ref(2) + 3 * X + 1]);

    let (extension, roots) = splitting_field(&poly_ring, poly_ring.clone_el(&f));
    assert_eq!(2, extension.rank());
    assert_eq!(7, roots.iter().map(|(_, i)| i).sum::<usize>());
    assert_eq!(5, roots.len());

    for (x, _) in &roots {
        assert_el_eq!(&extension, extension.zero(), poly_ring.evaluate(&f, x, &extension.inclusion()));
    }
}

#[ignore]
#[test]
fn test_splitting_field_rationals() {
    let base_field = RationalField::new(BigIntRing::RING);
    let poly_ring = DensePolyRing::new(&base_field, "X");
    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(6) + 10]);

    let (extension, roots) = splitting_field(&poly_ring, poly_ring.clone_el(&f));
    assert_eq!(12, extension.rank());
    assert_eq!(6, roots.iter().map(|(_, i)| i).sum::<usize>());
    assert_eq!(6, roots.len());

    for (x, _) in &roots {
        assert_el_eq!(&extension, extension.zero(), poly_ring.evaluate(&f, x, &extension.inclusion()));
    }
}