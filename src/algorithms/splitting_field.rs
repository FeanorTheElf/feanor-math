use std::alloc::*;
use std::borrow::Borrow;

use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::homomorphism::*;
use crate::matrix::OwnedMatrix;
use crate::rings::extension::galois_field::*;
use crate::rings::extension::number_field::*;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::rings::extension::extension_impl::*;
use crate::rings::extension::*;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::finite::FiniteRingStore;
use crate::rings::multivariate::{MultivariatePolyRing, MultivariatePolyRingStore};
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::divisibility::*;
use crate::field::*;
use crate::seq::sparse::SparseMapVector;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::integer::*;
use crate::seq::*;
use crate::ring::*;
use crate::algorithms::linsolve::*;
use super::poly_factor::FactorPolyField;
use super::poly_gcd::PolyTFracGCDRing;

///
/// Given a number field `K` and an irreducible polynomial `f`, computes a representation of
/// the number field `L = K[X]/(f)`. The result is returned by the inclusion `K -> L` and 
/// the element that corresponds to the coset of `X`, i.e. a root of `f` in `L`. Note that the 
/// canonical generator of `L` does not have to be a root of `f` (this might even be impossible,
/// e.g. if `f in ZZ[X]` but `K != QQ`).
/// 
/// The number field `K` is taken to be the base ring of the given polynomial ring.
/// 
/// As opposed to [`extend_number_field_promise_is_irreducible()`], this checks that `f` is 
/// indeed irreducible.
/// 
#[stability::unstable(feature = "enable")]
pub fn extend_number_field<K>(poly_ring: DensePolyRing<K>, irred_poly: &El<DensePolyRing<K>>) -> (
    FreeAlgebraHom<K, NumberField>,
    El<NumberField>
)
    where K: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);
    assert!(<NumberFieldBase<_, _> as FactorPolyField>::is_irred(&poly_ring, irred_poly));

    extend_number_field_promise_is_irreducible(poly_ring, irred_poly)
}

///
/// If the powers of `potential_primitive_element` up to `[L : K] [K : k]` generate `L`,
/// this returns the minimal polynomial of `potential_primitive_element`, as well as polynomials
/// `f, g` such that `f(potential_primitive_element)` and `f(potential_primitive_element)` give
/// the canonical generators of `K` resp. `L`.
/// 
fn test_primitive_element<R>(L: R, potential_primitive_element: El<R>) -> Option<(
    DensePolyRing<<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing>, 
    El<DensePolyRing<<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing>>,
    El<DensePolyRing<<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing>>,
    El<DensePolyRing<<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing>>
)>
    where R: RingStore,
        R::Type: FreeAlgebra,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: FreeAlgebra,
        <<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing: Clone,
        <<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing
{
    let K = L.base_ring();
    let k = K.base_ring();
    let total_rank = L.rank() * K.rank();

    // lhs is the matrix whose columns are the powers of `potential_primitive_element`, w.r.t. the tensor basis
    let mut lhs = OwnedMatrix::zero(total_rank, total_rank, k);
    let mut current = L.one();
    for j in 0..total_rank {
        let current_wrt_basis = L.wrt_canonical_basis(&current);
        for i1 in 0..L.rank() {
            let c = current_wrt_basis.at(i1);
            let c_wrt_basis = K.wrt_canonical_basis(&c);
            for i2 in 0..K.rank() {
                *lhs.at_mut(i1 * K.rank() + i2, j) = c_wrt_basis.at(i2);
            }
        }
        drop(current_wrt_basis);
        L.mul_assign_ref(&mut current, &potential_primitive_element);
    }

    // rhs has three columns: 
    // - first the `total_rank`-th power of `x + ka` -> this will later give us the minpoly
    // - second the generator of `K` -> this will give us a repr of this generator in the result field
    // - third the generator of `L` -> this will give us a repr of this generator in the result field
    let mut rhs = OwnedMatrix::zero(total_rank, 3, k);
    let current_wrt_basis = L.wrt_canonical_basis(&current);
    for i1 in 0..L.rank() {
        let c = current_wrt_basis.at(i1);
        let c_wrt_basis = K.wrt_canonical_basis(&c);
        for i2 in 0..K.rank() {
            *rhs.at_mut(i1 * K.rank() + i2, 0) = c_wrt_basis.at(i2);
        }
    }
    if K.rank() > 1 {
        *rhs.at_mut(1, 1) = k.one();
    } else {
        *rhs.at_mut(0, 1) = K.wrt_canonical_basis(&K.canonical_gen()).at(0);
    }
    *rhs.at_mut(K.rank(), 2) = k.one();

    let mut sol = OwnedMatrix::zero(total_rank, 3, k);
    let has_sol = k.solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).is_solved();

    if has_sol {
        let kX = DensePolyRing::new(k.clone(), "X");
        let gen_poly = kX.from_terms((0..total_rank).map(|i| (k.negate(k.clone_el(sol.at(i, 0))), i)).chain([(k.one(), total_rank)].into_iter()));
        let K_generator = kX.from_terms((0..total_rank).map(|i| (k.clone_el(sol.at(i, 1)), i)));
        let L_generator = kX.from_terms((0..total_rank).map(|i| (k.clone_el(sol.at(i, 2)), i)));
        return Some((
            kX,
            gen_poly,
            K_generator,
            L_generator
        ));
    } else {
        return None;
    }
}

#[stability::unstable(feature = "enable")]
pub fn extend_galois_field<K>(poly_ring: DensePolyRing<K>, irred_poly: &El<DensePolyRing<K>>) -> (
    FreeAlgebraHom<K, GaloisField>,
    El<GaloisField>
)
    where K: RingStore<Type = GaloisFieldBase<DefaultGaloisFieldImpl>>
{
    
    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);

    // I wonder what is the better method: Either factoring the polynomial (taking `O(d^3 log(q))`
    // operations in `GF(q)`, or up to `d/log(d)` less with better convolution algorithms), or solving
    // the linear system (taking `O(d^3)` operations in `GF(q)`, or up to `d^(3 - omega)` less with
    // better matrix inversion algirhtms). I decide for solving the system, it feels better.

    let K = poly_ring.base_ring();
    let Fp = K.base_ring();
    
    let L = AsField::from(
        AsFieldBase::promise_is_perfect_field(
            FreeAlgebraImpl::new_with(
                K, 
                poly_ring.degree(&irred_poly).unwrap(), 
                (0..poly_ring.degree(&irred_poly).unwrap()).map(|i| K.negate(K.clone_el(poly_ring.coefficient_at(&irred_poly, i)))).collect::<Vec<_>>(),
                "X",
                Global,
                STANDARD_CONVOLUTION
            )
        )
    );
    let total_rank = L.rank() * K.rank();

    let mut rng = oorandom::Rand64::new(1);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
        let potential_primitive_element = L.random_element(|| rng.rand_u64());

        if let Some((FpX, gen_poly, K_gen, L_gen)) = test_primitive_element(&L, potential_primitive_element) {

            let mut modulus = SparseMapVector::new(total_rank, Fp.clone());
            for (x, i) in FpX.terms(&gen_poly).filter(|(_, i)| *i < total_rank) {
                *modulus.at_mut(i) = Fp.clone_el(x);
            }
            _ = modulus.at_mut(0);
            let potential_result = FreeAlgebraImpl::new(
                Fp.clone(),
                total_rank,
                modulus
            );

            if let Ok(result) = potential_result.as_field() {
                let result = GaloisField::create(result);

                // note that `sol` contains coefficients w.r.t. `x` and not `result.canonical_gen()`
                let K_generator = result.from_canonical_basis((0..total_rank).map(|i| Fp.clone_el(FpX.coefficient_at(&K_gen, i))));
                let L_generator = result.from_canonical_basis((0..total_rank).map(|i| Fp.clone_el(FpX.coefficient_at(&L_gen, i))));

                let result = FreeAlgebraHom::new(poly_ring.into().into_base_ring(), result, K_generator);
                return (result, L_generator);
            }
        }
    }
    unreachable!()
}

///
/// Given a number field `K` and an irreducible polynomial `f`, computes a representation of
/// the number field `L = K[X]/(f)`. The result is returned by the inclusion `K -> L` and 
/// the element that corresponds to the coset of `X`, i.e. a root of `f` in `L`. Note that the 
/// canonical generator of `L` does not have to be a root of `f` (this might even be impossible,
/// e.g. if `f in ZZ[X]` but `K != QQ`).
/// 
/// The number field `K` is taken to be the base ring of the given polynomial ring.
/// 
/// This function assumes that the given polynomial is irreducible. If it is not, the results
/// may be nonsensical (but of course not UB).
/// 
#[stability::unstable(feature = "enable")]
pub fn extend_number_field_promise_is_irreducible<K>(poly_ring: DensePolyRing<K>, irred_poly: &El<DensePolyRing<K>>) -> (
    FreeAlgebraHom<K, NumberField>,
    El<NumberField>
)
    where K: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    static_assert_impls!(NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>: FactorPolyField);
    static_assert_impls!(NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>: PolyTFracGCDRing);

    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);

    let K = poly_ring.base_ring();

    let L = AsField::from(
        AsFieldBase::promise_is_perfect_field(
            FreeAlgebraImpl::new_with(
                K, 
                poly_ring.degree(&irred_poly).unwrap(), 
                (0..poly_ring.degree(&irred_poly).unwrap()).map(|i| K.negate(K.clone_el(poly_ring.coefficient_at(&irred_poly, i)))).collect::<Vec<_>>(),
                "X",
                Global,
                STANDARD_CONVOLUTION
            )
        )
    );
    
    let total_rank = K.rank() * L.rank();

    // the main task is to find a primitive element of `L`, i.e. that generates it over `K`.
    // we use the following approach:
    //  - Consider generator `a` of `K` and `b` of `L` over `K`
    //  - Consider the set `A_n = { b, b + a, b + 2a, ..., b + (n - 1)a }`
    //  - Consider also the subset `B_n = { x in A_n | QQ[x] != L }`
    //  - The idea is now to choose a random element from `A_n`, and hope that it is not in `B_n`
    //  - To estimate the probability, consider the map `B_n -> { maximal proper subfields of L }` that maps any `x`
    //    to some maximal proper subfield containing `QQ[x]`
    //  - Then this map is injective, as any field containing `a + ib` and `a + jb`, `j > i` must contain `a, b`, thus be `L`
    //  - In other words, to find `n`, we need a bound on the maximal proper subfields
    //  - I believe the degree `[L : QQ]` is such a bound (in the Galois case it is, at least)

    // take `A` twice as large, so that we find a good element with probability >= 1/2
    let size_of_A: i32 = (2 * total_rank).try_into().unwrap();

    let mut rng = oorandom::Rand64::new(1);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {

        let a = StaticRing::<i32>::RING.get_uniformly_random(&size_of_A, || rng.rand_u64());
        let potential_primitive_element = L.add(L.canonical_gen(), L.inclusion().map(K.int_hom().mul_map(K.canonical_gen(), a)));
    
        if let Some((QQX, gen_poly, K_gen, L_gen)) = test_primitive_element(&L, potential_primitive_element) {
            if let Some((result, x)) = NumberField::try_adjoin_root(&QQX, &gen_poly) {

                println!("Ring generated by a root of {}", QQX.format(&gen_poly));

                // note that `sol` contains coefficients w.r.t. `x` and not `result.canonical_gen()`
                let K_generator = QQX.evaluate(&K_gen, &x, result.inclusion());
                let L_generator = QQX.evaluate(&L_gen, &x, result.inclusion());

                let result = FreeAlgebraHom::new(poly_ring.into().into_base_ring(), result, K_generator);
                return (result, L_generator);
            }
        }
    }

    unreachable!()
}

///
/// Given a polynomial `f in K[X]` over a field `K`, this computes an 
/// extension `L` of `K` such that `f` splits in `L`.
/// 
/// This function returns the embedding `K -> L` and the list of roots of `f`.
/// 
/// The closure `create_field` should, when given a field `K'` and an irreducible
/// polynomial `g in K'[X]`, compute an extension field `K''`, and return both the
/// embedding `K' -> K''` and a root `a in K''` of `g`.
/// 
#[stability::unstable(feature = "enable")]
pub fn splitting_field<K, F>(
    mut poly_ring: DensePolyRing<K>, 
    f: El<DensePolyRing<K>>, 
    mut create_field: F
) -> (
    FreeAlgebraHom<K, K>,
    Vec<(El<K>, usize)>
)
    where K: RingStore + Clone,
        K::Type: FactorPolyField + FreeAlgebra,
        F: for<'a> FnMut(DensePolyRing<&'a K>, El<DensePolyRing<&'a K>>) -> (FreeAlgebraHom<&'a K, K>, El<K>)
{
    assert!(!poly_ring.is_zero(&f));
    let mut to_split = vec![(f, 1)];
    let mut roots = Vec::new();
    let mut base_field = None;
    let mut image_of_base_can_gen = poly_ring.base_ring().canonical_gen();

    while let Some((next_to_split, multiplicity)) = to_split.pop() {
        let (mut factorization, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &next_to_split);
        println!("factorization:");
        for (f, _) in &factorization {
            println!("{}", poly_ring.format(f));
        }
        let extend_idx = factorization.iter().enumerate().max_by_key(|(_, (f, _))| poly_ring.degree(f).unwrap()).unwrap().0;
        
        let extend_with_poly = if poly_ring.degree(&factorization[extend_idx].0).unwrap() > 1 {
            Some(factorization.remove(extend_idx))
        } else {
            None
        };

        for (g, e) in factorization.into_iter() {
            if poly_ring.degree(&g).unwrap() > 1 {
                to_split.push((g, e * multiplicity));
            } else {
                roots.push((poly_ring.base_ring().negate(poly_ring.base_ring().div(poly_ring.coefficient_at(&g, 0), poly_ring.coefficient_at(&g, 1))), e * multiplicity));
            }
        }

        if let Some((extend_with_poly, e)) = extend_with_poly {
            let ref_poly_ring = DensePolyRing::new(poly_ring.base_ring(), "X");
            let ref_extend_with_poly = ref_poly_ring.lifted_hom(&poly_ring, poly_ring.base_ring().identity()).map_ref(&extend_with_poly);
            let (into_new_field, root) = create_field(ref_poly_ring, ref_extend_with_poly);
            let (old_field, new_field, image) = into_new_field.destruct();
            let new_poly_ring = DensePolyRing::new(new_field, "X");
            let into_new_field = FreeAlgebraHom::new(old_field, new_poly_ring.base_ring(), image);
            
            image_of_base_can_gen = into_new_field.map(image_of_base_can_gen);
            roots = roots.into_iter().map(|(r, e)| (into_new_field.map(r), e)).collect();
            let lifted_hom = new_poly_ring.lifted_hom(&poly_ring, &into_new_field);
            to_split = to_split.into_iter().map(|(f, e)| (lifted_hom.map(f), e)).collect();

            to_split.push((new_poly_ring.checked_div(
                &lifted_hom.map(extend_with_poly), 
                &new_poly_ring.sub(new_poly_ring.indeterminate(), new_poly_ring.inclusion().map_ref(&root))
            ).unwrap(), multiplicity * e));
            roots.push((root, multiplicity * e));

            if base_field.is_none() {
                base_field = Some(poly_ring.into().into_base_ring());
            }
            poly_ring = new_poly_ring;
        }
    }

    println!("computed splitting field");
    return (
        FreeAlgebraHom::new(
            base_field.unwrap_or_else(|| poly_ring.clone().into().into_base_ring()),
            poly_ring.into().into_base_ring(),
            image_of_base_can_gen
        ),
        roots
    );
}

///
/// The closure `create_field` should, when given a field `K'` and an irreducible
/// polynomial `g in K'[X]`, compute an extension field `K''`, and return both the
/// embedding `K' -> K''` and a root `a in K''` of `g`.
/// 
#[stability::unstable(feature = "enable")]
pub fn variety_from_lex_gb<K, P, F>(
    poly_ring: P, 
    lex_gb: &[El<P>], 
    mut create_field: F
) -> (FreeAlgebraHom<K, K>, Vec<Vec<El<K>>>)
    where P: RingStore,
        K: RingStore + Clone,
        K::Type: FactorPolyField + FreeAlgebra,
        P::Type: MultivariatePolyRing,
        <P::Type as RingExtension>::BaseRing: Borrow<K> + RingStore<Type = K::Type>,
        F: for<'a> FnMut(DensePolyRing<&'a K>, El<DensePolyRing<&'a K>>) -> (FreeAlgebraHom<&'a K, K>, El<K>)
{
    let n = poly_ring.indeterminate_count();
    let constant_monomial = poly_ring.create_monomial((0..n).map(|_| 0));
    if lex_gb.iter().any(|f| poly_ring.appearing_indeterminates(f).len() == 0 && !poly_ring.base_ring().is_zero(poly_ring.coefficient_at(f, &constant_monomial))) {
        return (FreeAlgebraHom::identity(poly_ring.base_ring().borrow().clone()), Vec::new());
    }

    let mut relevant_indeterminates = Vec::new();
    for f in lex_gb {
        relevant_indeterminates.extend(poly_ring.appearing_indeterminates(f).into_iter().map(|(v, _)| v));
    }
    relevant_indeterminates.sort_unstable();
    relevant_indeterminates.dedup();
    let contains_indeterminate = |f, indet| poly_ring.appearing_indeterminates(f).iter().any(|(v, _)| v == indet);
    let has_no_indeterminate_before = |f, indet| poly_ring.appearing_indeterminates(f).iter().all(|(v, _)| v >= indet);
    
    let polys_for_indeterminate = relevant_indeterminates.iter().map(|indet| {
        let relevant_polys = lex_gb.iter().filter(|f| contains_indeterminate(f, indet) && has_no_indeterminate_before(f, indet)).collect::<Vec<_>>();
        assert!(relevant_polys.len() > 0, "basis is either not a lex-GB or does not generate a zero-dimensional ideal");
        return relevant_polys;
    }).collect::<Vec<_>>();

    let mut current_embedding: FreeAlgebraHom<_, K> = FreeAlgebraHom::identity(poly_ring.base_ring().borrow().clone());
    let mut current_roots: Vec<(usize, Vec<El<K>>)> = vec![(0, (0..relevant_indeterminates.len()).map(|_| current_embedding.codomain().zero()).collect())];
    let mut final_roots: Vec<Vec<El<K>>> = Vec::new();

    while let Some((set_indeterminates, current_root)) = current_roots.pop() {
        let next_indet_idx = relevant_indeterminates.len() - set_indeterminates - 1;
        println!("Extend {:?}", current_root[(next_indet_idx + 1)..].iter().map(|x| current_embedding.codomain().format(x)).collect::<Vec<_>>());
        let next_indet = relevant_indeterminates[next_indet_idx];
        
        let (base_field, current_field, base_field_gen) = current_embedding.destruct();
        let KX = DensePolyRing::new(current_field, "X");
        let ref_current_embedding = FreeAlgebraHom::new(&base_field, KX.base_ring(), base_field_gen);
        let next_roots_from: El<DensePolyRing<K>> = polys_for_indeterminate[next_indet_idx].iter().map(|f| 
            poly_ring.evaluate(
                f, 
                (0..n).map_fn(|i| if i == next_indet {
                    KX.indeterminate()
                } else {
                    KX.inclusion().map_ref(&current_root[i])
                }),
                KX.inclusion().compose(&ref_current_embedding)
            )
        ).fold(KX.zero(), |current, next| KX.ideal_gen(&current, &next));

        assert!(!KX.is_zero(&next_roots_from), "basis is either not a lex-GB or does not generate a zero-dimensional ideal");
        if KX.degree(&next_roots_from).unwrap() == 0 {
            println!("Found unsatisfyiable partial root combination");
            let (_, _, base_field_gen) = ref_current_embedding.destruct();
            current_embedding = FreeAlgebraHom::new(base_field, KX.into().into_base_ring(), base_field_gen);
            // continue
        } else if KX.degree(&next_roots_from).unwrap() == 1 {
            let mut new_root = current_root;
            new_root[next_indet_idx] = KX.base_ring().negate(KX.base_ring().div(KX.coefficient_at(&next_roots_from, 0), KX.coefficient_at(&next_roots_from, 1)));
            if set_indeterminates + 1 < relevant_indeterminates.len() {
                current_roots.push((set_indeterminates + 1, new_root));
            } else {
                final_roots.push(new_root);
            }
            let (_, _, base_field_gen) = ref_current_embedding.destruct();
            current_embedding = FreeAlgebraHom::new(base_field, KX.into().into_base_ring(), base_field_gen);
        } else {
            println!("Extend_with {:?}", KX.format(&next_roots_from));
            let (_, _, base_field_gen) = ref_current_embedding.destruct();
            let (into_new_field, roots) = splitting_field(KX, next_roots_from, &mut create_field);

            final_roots = final_roots.into_iter().map(|root| root.into_iter().map(|x| into_new_field.map(x)).collect()).collect();
            current_roots = current_roots.into_iter().map(|(l, root)| (l, root.into_iter().map(|x| into_new_field.map(x)).collect())).collect();
            for (r, _) in roots.into_iter() {
                let mut new_root: Vec<El<K>> = current_root.iter().map(|x| into_new_field.map_ref(x)).collect();
                new_root[next_indet_idx] = r;
                if set_indeterminates + 1 < relevant_indeterminates.len() {
                    current_roots.push((set_indeterminates + 1, new_root));
                } else {
                    final_roots.push(new_root);
                }
            }
            let base_field_gen = into_new_field.map(base_field_gen);
            let (_, new_field, _) = into_new_field.destruct();
            current_embedding = FreeAlgebraHom::new(base_field, new_field, base_field_gen);
        }
    }
    return (current_embedding, final_roots);
}

#[cfg(test)]
use crate::wrapper::RingElementWrapper;
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;

#[test]
fn test_extend_field() {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);
    let ZZX = DensePolyRing::new(&ZZ, "X");
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(ZZX, &f);
    let KX = DensePolyRing::new(&K, "X");

    // extend `QQ[i]` by `X^4 - i`
    let [g] = KX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - RingElementWrapper::new(&KX, KX.inclusion().map(K.canonical_gen()))]);
    let (extension_field_embedding, x) = extend_number_field(KX.clone(), &g);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(8, ext_field.rank());
    assert_el_eq!(ext_field, ext_field.neg_one(), ext_field.pow(extension_field_embedding.map(K.canonical_gen()), 2));
    assert_el_eq!(ext_field, extension_field_embedding.map(K.canonical_gen()), ext_field.pow(x, 4));

    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding, 
        [(0, 0), (0, 1), (1, 0), (2, 0), (-1, 0), (0, -1), (-1, -1)].into_iter().map(|(a, b)| K.from_canonical_basis([QQ.int_hom().map(a), QQ.int_hom().map(b)]))
    );

    // extend `QQ[i]` by `X^3 - 2`
    let [g] = KX.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 2]);
    let (extension_field_embedding, x) = extend_number_field(KX, &g);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(6, ext_field.rank());
    assert_el_eq!(ext_field, ext_field.neg_one(), ext_field.pow(extension_field_embedding.map(K.canonical_gen()), 2));
    assert_el_eq!(ext_field, ext_field.int_hom().map(2), ext_field.pow(x, 3));
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding, 
        [(0, 0), (0, 1), (1, 0), (2, 0), (-1, 0), (0, -1), (-1, -1)].into_iter().map(|(a, b)| K.from_canonical_basis([QQ.int_hom().map(a), QQ.int_hom().map(b)]))
    );
}

#[test]
fn test_variety_from_lex_gb() {
    let ZZX = DensePolyRing::new(BigIntRing::RING, "X");
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X - 1]);
    let QQ = NumberField::new(ZZX, &f);
    let QQXYZ = MultivariatePolyRingImpl::new(&QQ, 3);
    let intersection_unitsphere_twistedcubic = QQXYZ.with_wrapped_indeterminates(|[x, y, z]| [
        x.pow_ref(2) + y.pow_ref(2) + z.pow_ref(2) - 1, 
        y - x.pow_ref(2), 
        z - x.pow_ref(2)
    ]);
    let lex_gb = QQXYZ.with_wrapped_indeterminates(|[x, y, z]| [
        z.pow_ref(6) - 5 * z.pow_ref(4) + 7 * z.pow_ref(2) - 1,
        y - z.pow_ref(4) + 3 * z.pow_ref(2) - 1,
        x + z.pow_ref(3) - 2 * z
    ]);
    let (into_L, variety) = variety_from_lex_gb(&QQXYZ, &lex_gb, |poly_ring, poly| extend_number_field_promise_is_irreducible(poly_ring, &poly));
    let L = into_L.codomain();

    assert_eq!(6, variety.len());
    for point in &variety {
        assert_el_eq!(L, L.zero(), QQXYZ.evaluate(&intersection_unitsphere_twistedcubic[0], point.clone_ring_els(L), &into_L));
        assert_el_eq!(L, L.zero(), QQXYZ.evaluate(&intersection_unitsphere_twistedcubic[1], point.clone_ring_els(L), &into_L));
        assert_el_eq!(L, L.zero(), QQXYZ.evaluate(&intersection_unitsphere_twistedcubic[2], point.clone_ring_els(L), &into_L));
    }
}