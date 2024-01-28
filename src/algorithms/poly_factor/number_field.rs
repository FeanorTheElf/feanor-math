use crate::algorithms::poly_factor::cantor_zassenhaus;
use crate::divisibility::*;
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::{IntegerRing, IntegerRingStore};
use crate::pid::EuclideanRing;
use crate::ring::*;
use crate::rings::extension::FreeAlgebra;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::{PolyRing, PolyRingStore};
use crate::rings::rational::RationalFieldBase;

fn factor_squarefree_over_number_field<P, I>(KX: P, f: El<P>) -> Vec<(El<P>, usize)>
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        I: IntegerRingStore,
        I::Type: IntegerRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>
{
    let K = KX.base_ring();
    let QQ = K.base_ring();
    let ZZ: &I = QQ.base_ring();

    let d = KX.degree(&f).unwrap();
    assert!(KX.base_ring().is_one(KX.lc(&f).unwrap()));

    let QQX = DensePolyRing::new(QQ, "X");

    unimplemented!()
}

pub fn factor_over_number_field<P, I>(poly_ring: P, f: &El<P>)
    where P: PolyRingStore,
        P::Type: PolyRing + EuclideanRing,
        I: IntegerRingStore,
        I::Type: IntegerRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field + FreeAlgebra,
        <<<P::Type as RingExtension>::BaseRing as RingStore>::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>
{
    let KX = &poly_ring;
    let K = KX.base_ring();
    let QQ = K.base_ring();
    let ZZ: &I = QQ.base_ring();

    // We use the approach outlined in Cohen's "a course in computational algebraic number theory", 
    // with some notable changes. We proceed as follows:
    //  - Use square-free reduction to assume wlog that `f` is square-free
    //  - Observe that the factorization of `f` is the product over `gcd(f, g)` where `g` runs
    //    through the factors of `N(f)` over `QQ[X]`. Here `N(f)` is the "norm" of `f`, i.e.
    //    the product `prod_sigma sigma(f)` where `sigma` runs through the embeddings `K -> CC`.
    //  - It is now left to actually compute `N(f)`, which is not so simple as we do not known the
    //    `sigma`. Instead, write the coefficients of `f` as polynomials of a fixed generator `a` of `K`.
    //    We still cannot find `sigma(a)` for various `sigma`, but observe that the coefficients of `N(f)`
    //    depend only on the value of symmetric polynomials evaluated at `a, sigma_2(a), ..., sigma_n(a)`.
    //    We can find those values, as they are the coefficients of the generating polynomial.

    assert!(!KX.is_zero(f));
    let mut current = KX.clone_el(f);
    while !KX.is_unit(&current) {
        let mut squarefree_part = cantor_zassenhaus::poly_squarefree_part(KX, KX.clone_el(&current));
        let lc_inv = K.div(&K.one(), KX.lc(&squarefree_part).unwrap());
        KX.inclusion().mul_assign_map(&mut squarefree_part, lc_inv);
        current = KX.checked_div(&current, &squarefree_part).unwrap();

        factor_squarefree_over_number_field(KX, squarefree_part);
    }
}