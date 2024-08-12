use crate::algorithms;
use crate::algorithms::unity_root::get_prim_root_of_unity;
use crate::divisibility::DivisibilityRingStore;
use crate::pid::*;
use crate::field::{Field, FieldStore};
use crate::integer::*;
use crate::ring::*;
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
use crate::rings::field::{AsField, AsFieldBase};
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::rings::finite::{FiniteRing, FiniteRingStore};
use crate::homomorphism::Homomorphism;
use crate::seq::VectorFn;
use crate::rings::poly::dense_poly::DensePolyRing;

use oorandom;

///
/// As [`distinct_degree_factorization()`], but takes `f` in the form of the ring `R[X]/(f)`, which is the internal
/// representation that is used to actually compute the factorization.
/// 
#[stability::unstable(feature = "enable")]
pub fn distinct_degree_factorization_base<P, R>(poly_ring: P, mod_f_ring: R) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        R: RingStore,
        R::Type: FreeAlgebra,
        <<R as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(poly_ring.base_ring().get_ring() == mod_f_ring.base_ring().get_ring());
    let ZZ = BigIntRing::RING;
    let q = poly_ring.base_ring().size(&ZZ).unwrap();
    debug_assert!(ZZ.eq_el(&algorithms::int_factor::is_prime_power(&ZZ, &q).unwrap().0, &poly_ring.base_ring().characteristic(&ZZ).unwrap()));

    let mut total_deg = 0;
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());

    let mut result = Vec::new();
    let mut current_deg = 0;
    result.push(poly_ring.one());
    let mut x_power_Q_mod_f = mod_f_ring.canonical_gen();
    while total_deg < mod_f_ring.rank() {
        current_deg += 1;
        x_power_Q_mod_f = mod_f_ring.pow_gen(x_power_Q_mod_f, &q, ZZ);
        let fq_defining_poly_mod_f = poly_ring.sub(mod_f_ring.poly_repr(&poly_ring, &x_power_Q_mod_f, &poly_ring.base_ring().identity()), poly_ring.indeterminate());
        let mut deg_i_factor = poly_ring.normalize(poly_ring.ideal_gen(&f, &fq_defining_poly_mod_f));
        for prev_deg in 1..current_deg {
            if current_deg % prev_deg == 0 {
                deg_i_factor = poly_ring.checked_div(&deg_i_factor, &result[prev_deg]).unwrap();
            }
        }
        total_deg += poly_ring.degree(&deg_i_factor).unwrap();
        result.push(deg_i_factor);
    }
    return result;
}

///
/// Computes the distinct-degree factorization of a polynomial over a finite
/// field. The given polynomial must be square-free.
/// 
/// Concretely, if a univariate polynomial `f(X)` factors uniquely as
/// `f(X) = f1(X) ... fr(X)`, then the `d`-th distinct-degree factor of `f` is
/// `prod_i fi(X)` where `i` runs through all indices with `deg(fi(X)) = d`.
/// This function returns a list whose `d`-th entry is the `d`-th distinct degree
/// factor. Once the list ends, all further `d`-th distinct degree factors should
/// be considered to be `1`.
/// 
/// To get a square-free polynomial, consider using [`super::poly_squarefree_part()`].
/// 
/// # Example
/// 
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::divisibility::*;
/// # use crate::feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::algorithms::poly_factor::cantor_zassenhaus::*;
/// let Fp = Zn::new(3).as_field().ok().unwrap();
/// let FpX = DensePolyRing::new(Fp, "X");
/// // f = (X^3 + 2 X^2 + 1) (X^3 + 2 X + 1) (X^2 + 1)
/// let f = FpX.prod([
///     FpX.from_terms([(Fp.one(), 0), (Fp.one(), 2)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 1), (Fp.one(), 3)].into_iter()),
///     FpX.from_terms([(Fp.one(), 0), (Fp.int_hom().map(2), 2), (Fp.one(), 3)].into_iter())
/// ].into_iter());
/// let factorization = distinct_degree_factorization(&FpX, f);
/// assert_eq!(4, factorization.len());
/// assert!(FpX.is_unit(&factorization[0]));
/// assert!(FpX.is_unit(&factorization[1]));
/// assert!(!FpX.is_unit(&factorization[2])); // factorization[2] is some scalar multiple of (X^2 + 1)
/// assert!(!FpX.is_unit(&factorization[3])); // factorization[3] is some scalar multiple of (X^3 + 2 X^2 + 1) (X^3 + 2 X + 1)
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn distinct_degree_factorization<P>(poly_ring: P, mut f: El<P>) -> Vec<El<P>>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: FieldStore + FiniteRingStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    let lc = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lc_inv = poly_ring.base_ring().invert(&lc).unwrap();
    poly_ring.inclusion().mul_assign_map(&mut f, lc_inv);

    let f_coeffs = (0..poly_ring.degree(&f).unwrap()).map(|i| poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&f, i)))).collect::<Vec<_>>();
    let mod_f_ring = FreeAlgebraImpl::new(poly_ring.base_ring(), &f_coeffs[..]);

    let mut result = distinct_degree_factorization_base(&poly_ring, mod_f_ring);
    poly_ring.inclusion().mul_assign_map(&mut result[0], lc);
    return result;
}

///
/// As [`cantor_zassenhaus()`], but takes `f` in the form of the ring `R[X]/(f)`, which is the internal
/// representation that is used to actually compute the factorization.
/// 
#[stability::unstable(feature = "enable")]
pub fn cantor_zassenhaus_base<P, R>(poly_ring: P, mod_f_ring: R, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        R: RingStore,
        R::Type: FreeAlgebra,
        <<R as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(poly_ring.base_ring().get_ring() == mod_f_ring.base_ring().get_ring());
    let ZZ = BigIntRing::RING;
    let q = poly_ring.base_ring().size(&ZZ).unwrap();
    debug_assert!(ZZ.eq_el(&algorithms::int_factor::is_prime_power(&ZZ, &q).unwrap().0, &poly_ring.base_ring().characteristic(&ZZ).unwrap()));
    assert!(ZZ.is_odd(&q));
    assert!(mod_f_ring.rank() % d == 0);
    assert!(mod_f_ring.rank() > d);
    let mut rng = oorandom::Rand64::new(ZZ.default_hash(&q) as u128);
    let exp = ZZ.half_exact(ZZ.sub(ZZ.pow(q, d), ZZ.one()));
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());

    loop {
        let T = mod_f_ring.from_canonical_basis((0..mod_f_ring.rank()).map(|_| poly_ring.base_ring().random_element(|| rng.rand_u64())));
        let G = mod_f_ring.sub(mod_f_ring.pow_gen(T, &exp, ZZ), mod_f_ring.one());
        let g = poly_ring.ideal_gen(&f, &mod_f_ring.poly_repr(&poly_ring, &G, &poly_ring.base_ring().identity()));
        if !poly_ring.is_unit(&g) && poly_ring.checked_div(&g, &f).is_none() {
            return g;
        }
    }
}

///
/// Uses the Cantor-Zassenhaus algorithm to find a nontrivial, factor of a polynomial f
/// over a finite field, that is squarefree and consists only of irreducible factors of 
/// degree d.
/// 
/// # Algorithm
/// 
/// The algorithm relies on the fact that for some monic polynomial T over Fq have
/// ```text
/// T^Q - T = T (T^((Q - 1)/2) + 1) (T^((Q - 1)/2) - 1)
/// ```
/// where `Q = q^d`. Furthermore, the three factors are pairwise coprime.
/// Since `X^Q - X` divides `T^Q - T`, and f is squarefree (so divides `X^Q - X`), 
/// we see that `f` also divides `T^Q - T` and so
/// ```text
/// f = gcd(T, f) gcd((T^((Q - 1)/2) + 1, f) gcd(T^((Q - 1)/2) - 1, f)
/// ```
/// The idea is now to choose a random T and check whether `gcd(T^((Q - 1)/2) - 1, f)`
/// gives a nontrivial factor of f. When f has two irreducible factors, with roots a, b
/// in FQ, then this works if exactly one of them maps to zero under the polynomial
/// `T^((Q - 1)/2) - 1`. Now observe that this is the case if and only if `T(a)` resp.
/// `T(b)` is a square in FQ. Now apparently, for a polynomial chosen uniformly at random
/// among all monic polynomials of degree d in `Fq[X]`, the values `T(a)` and `T(b)` are close
/// to independent and uniform on FQ, and thus the probability that one is a square and
/// the other is not is approximately 1/2.
/// 
///
#[stability::unstable(feature = "enable")]
pub fn cantor_zassenhaus<P>(poly_ring: P, mut f: El<P>, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: FiniteRingStore + FieldStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    let lc = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lc_inv = poly_ring.base_ring().invert(&lc).unwrap();
    poly_ring.inclusion().mul_assign_map(&mut f, lc_inv);

    let f_coeffs = (0..poly_ring.degree(&f).unwrap()).map(|i| poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&f, i)))).collect::<Vec<_>>();
    let mod_f_ring = FreeAlgebraImpl::new(poly_ring.base_ring(), &f_coeffs[..]);

    let result = cantor_zassenhaus_base(&poly_ring, mod_f_ring, d);
    return result;
}

///
/// Same as [`cantor_zassenhaus_even_base()`], but assumes that the base ring contains a 3rd root of unity.
/// 
/// Note that when using this in [`cantor_zassenhaus_even()`], some factors that might be returned by this
/// function don't work. In this case, we repeat, but clearly the randomness must be new. Thus, we allow passing
/// a seed.
/// 
fn cantor_zassenhaus_even_base_with_root_of_unity<P, R>(poly_ring: P, mod_f_ring: R, d: usize, seed: u64) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        R: RingStore,
        R::Type: FreeAlgebra,
        <<R as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(poly_ring.base_ring().get_ring() == mod_f_ring.base_ring().get_ring());
    let ZZ = BigIntRing::RING;
    let Fq = poly_ring.base_ring();
    let q = Fq.size(&ZZ).unwrap();
    let e = ZZ.abs_log2_ceil(&q).unwrap();
    assert_el_eq!(ZZ, ZZ.power_of_two(e), q);

    let mut rng = oorandom::Rand64::new((ZZ.default_hash(&q) as u128) | ((seed as u128) << u64::BITS));
    let zeta3 = get_prim_root_of_unity(&Fq, 3).unwrap();
    let exp = if d % 2 == 0 {
        ZZ.checked_div(&ZZ.sub(ZZ.power_of_two(d), ZZ.one()), &ZZ.int_hom().map(3)).unwrap()
    } else {
        ZZ.checked_div(&ZZ.sub(ZZ.power_of_two(2 * d), ZZ.one()), &ZZ.int_hom().map(3)).unwrap()
    };
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());
    
    // as in the standard case, we consider a random polynomial `T` and use the factorization `T^(2^d') - T = T (T^e + 1) (T^e + zeta) (T^e + zeta^2)`;
    // here `d'` is either `d` or `2d`, and `e = (2^d' - 1) / 3`
    
    loop {
        let T = mod_f_ring.from_canonical_basis((0..mod_f_ring.rank()).map(|_| poly_ring.base_ring().random_element(|| rng.rand_u64())));
        let T_pow_exp = mod_f_ring.pow_gen(T, &exp, ZZ);
        let T_pow_exp_poly = mod_f_ring.poly_repr(&poly_ring, &T_pow_exp, &Fq.identity());
        let g = poly_ring.ideal_gen(&f, &poly_ring.add_ref_fst(&T_pow_exp_poly, poly_ring.one()));
        if !poly_ring.is_unit(&g) && poly_ring.degree(&g) != poly_ring.degree(&f) {
            return g;
        }
        let g = poly_ring.ideal_gen(&f, &poly_ring.add(T_pow_exp_poly, poly_ring.inclusion().map_ref(&zeta3)));
        if !poly_ring.is_unit(&g) && poly_ring.degree(&g) != poly_ring.degree(&f) {
            return g;
        }
    }
}

///
/// Finds a nontrivial factor of a square-free polynomial over `Fq` for `q` a power of two,
/// assuming all its irreducible factors have degree `d`.
/// 
#[stability::unstable(feature = "enable")]
pub fn cantor_zassenhaus_even_base<P, R>(poly_ring: P, mod_f_ring: R, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        R: RingStore,
        R::Type: FreeAlgebra,
        <<R as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    assert!(poly_ring.base_ring().get_ring() == mod_f_ring.base_ring().get_ring());
    let ZZ = BigIntRing::RING;
    let Fq = poly_ring.base_ring();
    let q = Fq.size(&ZZ).unwrap();
    let e = ZZ.abs_log2_ceil(&q).unwrap();
    assert_el_eq!(ZZ, ZZ.power_of_two(e), q);
    assert!(mod_f_ring.rank() % d == 0);
    assert!(mod_f_ring.rank() > d);
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());
    
    if e % 2 != 0 {
        // adjoin a third root of unity, this will enable use to use the main idea
        // use `promise_as_field()`, since `as_field().unwrap()` can cause infinite generic expansion (always adding a `&`)
        let new_base_ring = AsField::from(AsFieldBase::promise_is_field(FreeAlgebraImpl::new(Fq, [Fq.neg_one(), Fq.neg_one()])));
        let new_x_pow_rank = mod_f_ring.wrt_canonical_basis(&mod_f_ring.pow(mod_f_ring.canonical_gen(), mod_f_ring.rank())).into_iter().map(|x| new_base_ring.inclusion().map(x)).collect::<Vec<_>>();
        // once we have any kind of tensoring operation, maybe we can find a way to do this that preserves e.g. sparse implementations?
        let new_mod_f_ring = FreeAlgebraImpl::new(&new_base_ring, &new_x_pow_rank);
        let new_poly_ring = DensePolyRing::new(&new_base_ring, "X");

        // it might happen that cantor_zassenhaus gives a nontrivial factor over the extension, but that factor only
        // induces a trivial factor over the base ring; in this case repeat
        for seed in 0..u64::MAX {
            let factor = new_poly_ring.normalize(cantor_zassenhaus_even_base_with_root_of_unity(&new_poly_ring, &new_mod_f_ring, d, seed));

            if new_poly_ring.terms(&factor).all(|(c, _)| Fq.is_zero(&new_base_ring.wrt_canonical_basis(c).at(1))) {
                // factor already lives in Fq
                return poly_ring.from_terms(new_poly_ring.terms(&factor).map(|(c, i)| (new_base_ring.wrt_canonical_basis(c).at(0), i)));
            } else {
                assert!(d % 2 == 0);
                // if d is even, the factor might only live in `new_base_ring`, but we can just use its norm;
                // the automorphism is X -> X^2
                let factor_conjugate = new_poly_ring.from_terms(new_poly_ring.terms(&factor).map(|(c, i)| {
                    let c_vec = new_base_ring.wrt_canonical_basis(c);
                    let new_c = new_base_ring.from_canonical_basis([Fq.sub(c_vec.at(0), c_vec.at(1)), Fq.negate(c_vec.at(1))].into_iter());
                    (new_c, i)
                }));
                let factor_norm = new_poly_ring.mul(factor, factor_conjugate);
                let factor_norm_Fq = poly_ring.from_terms(new_poly_ring.terms(&factor_norm).map(|(c, i)| (new_base_ring.wrt_canonical_basis(c).at(0), i)));
                let potential_result = poly_ring.ideal_gen(&f, &factor_norm_Fq);
                if poly_ring.degree(&potential_result).unwrap() < mod_f_ring.rank() {
                    return potential_result;
                }
                // we are unlucky, and got `factor` that contains exactly one factor over the extension ring of each irreducible factor of `f`;
                // in this case, `factor` is a nontrivial factor of `f`, but `N(factor)` is `f` up to units
            }
        }
        unreachable!()
    } else {
        return cantor_zassenhaus_even_base_with_root_of_unity(poly_ring, mod_f_ring, d, 0);
    }
}

///
/// Finds a nontrivial factor of a square-free polynomial over `Fq` for `q` a power of two,
/// assuming all its irreducible factors have degree `d`.
/// 
#[stability::unstable(feature = "enable")]
pub fn cantor_zassenhaus_even<P>(poly_ring: P, mut f: El<P>, d: usize) -> El<P>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P as RingStore>::Type as RingExtension>::BaseRing: FiniteRingStore + FieldStore,
        <<<P as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type: FiniteRing + Field
{
    let lc = poly_ring.base_ring().clone_el(poly_ring.lc(&f).unwrap());
    let lc_inv = poly_ring.base_ring().invert(&lc).unwrap();
    poly_ring.inclusion().mul_assign_map(&mut f, lc_inv);

    let f_coeffs = (0..poly_ring.degree(&f).unwrap()).map(|i| poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&f, i)))).collect::<Vec<_>>();
    let mod_f_ring = FreeAlgebraImpl::new(poly_ring.base_ring(), &f_coeffs[..]);

    let result = cantor_zassenhaus_even_base(&poly_ring, &mod_f_ring, d);
    return result;
}

#[cfg(test)]
use crate::rings::zn::zn_static::Fp;

#[test]
fn test_distinct_degree_factorization() {
    let field = Fp::<2>::RING;
    let ring = DensePolyRing::new(field, "X");
    let a0 = ring.one();
    let a1 = ring.mul(ring.indeterminate(), ring.from_terms([(1, 0), (1, 1)].into_iter()));
    let a2 = ring.from_terms([(1, 0), (1, 1), (1, 2)].into_iter());
    let a3 = ring.mul(ring.from_terms([(1, 0), (1, 1), (1, 3)].into_iter()), ring.from_terms([(1, 0), (1, 2), (1, 3)].into_iter()));
    let a = ring.prod([&a0, &a1, &a2, &a3].into_iter().map(|x| ring.clone_el(x)));
    let expected = vec![a0, a1, a2, a3];
    let distinct_degree_factorization = distinct_degree_factorization(&ring, a);
    assert_eq!(expected.len(), distinct_degree_factorization.len());
    for (f, e) in distinct_degree_factorization.into_iter().zip(expected.into_iter()) {
        assert_el_eq!(ring, e, ring.normalize(f));
    }
}

#[test]
fn test_cantor_zassenhaus() {
    let ring = DensePolyRing::new(Fp::<7>::RING, "X");
    let f = ring.from_terms([(1, 0), (1, 2)].into_iter());
    let g = ring.from_terms([(3, 0), (1, 1), (1, 2)].into_iter());
    let p = ring.mul_ref(&f, &g);
    let factor = ring.normalize(cantor_zassenhaus(&ring, p, 2));
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
}

#[test]
fn test_cantor_zassenhaus_even() {
    let ring = DensePolyRing::new(Fp::<2>::RING, "X");
    // (X^3 + X + 1) (X^3 + X^2 + 1)
    let f = ring.from_terms([(1, 0), (1, 1), (1, 3)].into_iter());
    let g = ring.from_terms([(1, 0), (1, 2), (1, 3)].into_iter());
    let h = ring.mul_ref(&f, &g);
    let factor = ring.normalize(cantor_zassenhaus_even(&ring, h, 3));
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
    
    // (X^4 + X + 1) (X^4 + X^3 + 1)
    let f = ring.from_terms([(1, 0), (1, 1), (1, 4)].into_iter());
    let g = ring.from_terms([(1, 0), (1, 3), (1, 4)].into_iter());
    let h = ring.mul_ref(&f, &g);
    let factor = ring.normalize(cantor_zassenhaus_even(&ring, h, 4));
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
}

#[test]
fn test_cantor_zassenhaus_even_extension_field() {

    let Fq = FreeAlgebraImpl::new(Fp::<2>::RING, [1, 1, 0, 0]).as_field().ok().unwrap();
    let ring = DensePolyRing::new(&Fq, "X");

    // (X^3 + X + 1) (X^3 + X^2 + 1)
    let f = ring.from_terms([(Fq.one(), 0), (Fq.one(), 1), (Fq.one(), 3)].into_iter());
    let g = ring.from_terms([(Fq.one(), 0), (Fq.one(), 2), (Fq.one(), 3)].into_iter());
    let h = ring.mul_ref(&f, &g);
    let factor = ring.normalize(cantor_zassenhaus_even(&ring, h, 3));
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
    
    // (X^4 + X + 1) = (X + a) (X + a + 1) (X + a^2) (X + a^2 + 1)
    let f1 = ring.from_terms([(Fq.canonical_gen(), 0), (Fq.one(), 1)].into_iter());
    let f2 = ring.from_terms([(Fq.add(Fq.canonical_gen(), Fq.one()), 0), (Fq.one(), 1)].into_iter());
    let f3 = ring.from_terms([(Fq.pow(Fq.canonical_gen(), 2), 0), (Fq.one(), 1)].into_iter());
    let f4 = ring.from_terms([(Fq.add(Fq.pow(Fq.canonical_gen(), 2), Fq.one()), 0), (Fq.one(), 1)].into_iter());
    let h = ring.from_terms([(Fq.one(), 0), (Fq.one(), 1), (Fq.one(), 4)].into_iter());
    let factor = ring.normalize(cantor_zassenhaus_even(&ring, h, 1));
    assert!(ring.eq_el(&factor, &f1) || ring.eq_el(&factor, &f2) || ring.eq_el(&factor, &f3) || ring.eq_el(&factor, &f4));

    let Fq = FreeAlgebraImpl::new(Fp::<2>::RING, [1, 1, 0]).as_field().ok().unwrap();
    let ring = DensePolyRing::new(&Fq, "X");
    
    // (X^4 + X + 1) (X^4 + X^3 + 1)
    let f = ring.from_terms([(Fq.one(), 0), (Fq.one(), 1), (Fq.one(), 4)].into_iter());
    let g = ring.from_terms([(Fq.one(), 0), (Fq.one(), 3), (Fq.one(), 4)].into_iter());
    let h = ring.mul_ref(&f, &g);
    let factor = ring.normalize(cantor_zassenhaus_even(&ring, h, 4));
    assert!(ring.eq_el(&factor, &f) || ring.eq_el(&factor, &g));
}