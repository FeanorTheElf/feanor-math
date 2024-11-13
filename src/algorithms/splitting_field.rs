use std::alloc::*;
use std::marker::PhantomData;

use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::homomorphism::*;
use crate::matrix::OwnedMatrix;
use crate::field::*;
use crate::rings::extension::number_field::*;
use crate::rings::rational::*;
use crate::pid::*;
use crate::primitive_int::StaticRing;
use crate::rings::extension::extension_impl::*;
use crate::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::integer::*;
use crate::seq::*;
use crate::ring::*;
use crate::algorithms::linsolve::*;
use super::poly_factor::FactorPolyField;
use super::poly_gcd::PolyGCDRing;

#[stability::unstable(feature = "enable")]
pub struct NumberFieldHom<R1, Impl1, I1, R2, Impl2, I2>
    where R1: RingStore<Type = NumberFieldBase<Impl1, I1>>,
        Impl1: RingStore,
        Impl1::Type: Field + FreeAlgebra,
        <Impl1::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I1>>,
        I1: RingStore,
        I1::Type: IntegerRing,
        R2: RingStore<Type = NumberFieldBase<Impl2, I2>>,
        Impl2: RingStore,
        Impl2::Type: Field + FreeAlgebra,
        <Impl2::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I2>>,
        I2: RingStore,
        I2::Type: IntegerRing
{
    from: R1,
    to: R2,
    from_int: PhantomData<I1>,
    to_int: PhantomData<I2>,
    generator_image: El<NumberField<Impl2, I2>>,
}

impl<R1, Impl1, I1, R2, Impl2, I2> Homomorphism<NumberFieldBase<Impl1, I1>, NumberFieldBase<Impl2, I2>> for NumberFieldHom<R1, Impl1, I1, R2, Impl2, I2>
    where R1: RingStore<Type = NumberFieldBase<Impl1, I1>>,
        Impl1: RingStore,
        Impl1::Type: Field + FreeAlgebra,
        <Impl1::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I1>>,
        I1: RingStore,
        I1::Type: IntegerRing,
        R2: RingStore<Type = NumberFieldBase<Impl2, I2>>,
        Impl2: RingStore,
        Impl2::Type: Field + FreeAlgebra,
        <Impl2::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I2>>,
        I2: RingStore,
        I2::Type: IntegerRing
{
    type DomainStore = R1;
    type CodomainStore = R2;

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.from
    }

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.to
    }

    fn map(&self, x: <NumberFieldBase<Impl1, I1> as RingBase>::Element) -> <NumberFieldBase<Impl2, I2> as RingBase>::Element {
        self.map_ref(&x)
    }

    fn map_ref(&self, x: &<NumberFieldBase<Impl1, I1> as RingBase>::Element) -> <NumberFieldBase<Impl2, I2> as RingBase>::Element {
        let poly_ring = DensePolyRing::new(self.to.base_ring(), "X");
        return poly_ring.evaluate(
            &self.from.poly_repr(&poly_ring, &x, self.to.base_ring().can_hom(self.from.base_ring()).unwrap()),
            &self.generator_image,
            self.to.inclusion()
        )
    }
}

///
/// Constructs the field `F[X]/(f(X))` that is isomorphic to `(F[X]/(g(X))[Y]/(h(Y))`
/// where `F[X]/(g(X))` is the base ring of `poly_ring` and `h` is the given irreducible
/// polynomial over `F[X]/(g(X))`. 
/// 
#[stability::unstable(feature = "enable")]
pub fn extend_number_field<P>(poly_ring: P, irred_poly: &El<P>) -> (
    NumberFieldHom<<<P as RingStore>::Type as RingExtension>::BaseRing, DefaultNumberFieldImpl, BigIntRing, NumberField, DefaultNumberFieldImpl, BigIntRing>,
    El<NumberField>
)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: Clone + RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);
    assert!(<NumberFieldBase<_, _> as FactorPolyField>::is_irred(&poly_ring, irred_poly));

    extend_number_field_promise_is_irreducible(poly_ring, irred_poly)
}

///
/// Given a number field `K` and an irreducible polynomial `f`, computes a representation of
/// the number field `L = K[X]/(f)`. The result is returned by the inclusion `K -> L` and 
/// the element that corresponds to the coset of `X`, i.e. a root of `f` in `L`. Note that the 
/// canonical generator of `L` does not have to be a root of `f` (this might even be impossible,
/// e.g. if `f in ZZ[X]` but `K != QQ`).
/// 
#[stability::unstable(feature = "enable")]
pub fn extend_number_field_promise_is_irreducible<P>(poly_ring: P, irred_poly: &El<P>) -> (
    NumberFieldHom<<<P as RingStore>::Type as RingExtension>::BaseRing, DefaultNumberFieldImpl, BigIntRing, NumberField, DefaultNumberFieldImpl, BigIntRing>,
    El<NumberField>
)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: Clone + RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    static_assert_impls!(NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>: FactorPolyField);
    static_assert_impls!(NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>: PolyGCDRing);

    assert!(!poly_ring.is_zero(&irred_poly));
    assert!(poly_ring.degree(&irred_poly).unwrap() > 1);

    let K = poly_ring.base_ring();
    let QQ = K.base_ring();

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

    // the main task is to find a primitive element of `extension_ring`, i.e. that generates it over `base_ring`.
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
    let size_of_A = (2 * total_rank) as i32;

    let mut rng = oorandom::Rand64::new(1);

    for _ in 0..MAX_PROBABILISTIC_REPETITIONS {

        let a = StaticRing::<i32>::RING.get_uniformly_random(&size_of_A, || rng.rand_u64());
        let potential_primitive_element = L.add(L.canonical_gen(), L.inclusion().map(K.int_hom().mul_map(K.canonical_gen(), a)));
    
        // lhs is the matrix whose columns are the powers of `b + ka`, w.r.t. the tensor basis
        let mut lhs = OwnedMatrix::zero(total_rank, total_rank, QQ);
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
            L.mul_assign_ref(&mut current, &potential_primitive_element);
        }
    
        // rhs has three columns: 
        // - first the `total_rank`-th power of `x + ka` -> this will later give us the minpoly
        // - second the generator of `K` -> this will give us a repr of this generator in the result field
        // - third the generator of `L` -> this will give us a repr of this generator in the result field
        let mut rhs = OwnedMatrix::zero(total_rank, 3, QQ);
        let current_wrt_basis = L.wrt_canonical_basis(&current);
        for i1 in 0..L.rank() {
            let c = current_wrt_basis.at(i1);
            let c_wrt_basis = K.wrt_canonical_basis(&c);
            for i2 in 0..K.rank() {
                *rhs.at_mut(i1 * K.rank() + i2, 0) = c_wrt_basis.at(i2);
            }
        }
        if K.rank() > 1 {
            *rhs.at_mut(1, 1) = QQ.one();
        } else {
            *rhs.at_mut(0, 1) = K.wrt_canonical_basis(&K.canonical_gen()).at(0);
        }
        *rhs.at_mut(K.rank(), 2) = QQ.one();
    
        let mut sol = OwnedMatrix::zero(total_rank, 3, QQ);
        let has_sol = QQ.solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).is_solved();

        if has_sol {
    
            let QQX = DensePolyRing::new(QQ, "X");
            let gen_poly = QQX.from_terms((0..total_rank).map(|i| (QQ.negate(QQ.clone_el(sol.at(i, 0))), i)).chain([(QQ.one(), total_rank)].into_iter()));
    
            if let Some((result, x)) = NumberField::try_adjoin_root(&QQX, &gen_poly) {

                // note that `sol` contains coefficients w.r.t. `x` and not `result.canonical_gen()`
                let K_generator = <_ as RingStore>::sum(&result, (0..total_rank).map(|i| result.inclusion().mul_map(result.pow(result.clone_el(&x), i), QQ.clone_el(sol.at(i, 1)))));
                let L_generator = <_ as RingStore>::sum(&result, (0..total_rank).map(|i| result.inclusion().mul_map(result.pow(result.clone_el(&x), i), QQ.clone_el(sol.at(i, 2)))));

                let result = NumberFieldHom {
                    from: K.clone(),
                    to: result,
                    from_int: PhantomData,
                    to_int: PhantomData,
                    generator_image: K_generator
                };

                return (result, L_generator);
            }
        }
    }

    unreachable!()
}

#[cfg(test)]
use crate::wrapper::RingElementWrapper;

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
    let (extension_field_embedding, x) = extend_number_field(&KX, &g);
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
    let (extension_field_embedding, x) = extend_number_field(&KX, &g);
    let ext_field = extension_field_embedding.codomain();
    assert_eq!(6, ext_field.rank());
    assert_el_eq!(ext_field, ext_field.neg_one(), ext_field.pow(extension_field_embedding.map(K.canonical_gen()), 2));
    assert_el_eq!(ext_field, ext_field.int_hom().map(2), ext_field.pow(x, 3));
    
    crate::homomorphism::generic_tests::test_homomorphism_axioms(
        &extension_field_embedding, 
        [(0, 0), (0, 1), (1, 0), (2, 0), (-1, 0), (0, -1), (-1, -1)].into_iter().map(|(a, b)| K.from_canonical_basis([QQ.int_hom().map(a), QQ.int_hom().map(b)]))
    );
}
