use tracing::{Level, event, instrument};

use super::IntegerRing;
use crate::algorithms::poly_factor::factor_locally::poly_factor_integer;
use crate::homomorphism::*;
use crate::prelude::*;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::poly::*;
use crate::ring_impls::rational::RationalFieldBase;
use crate::ring_impls::zn::zn_64b::*;
use crate::ring_properties::pid::{EuclideanRing, PrincipalIdealRingStore};

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_factor_rational<'a, P, I>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, El<BaseRingStore<P>>)
where
    P: RingStore,
    P::Ring: PolyRing + EuclideanRing,
    BaseRingStore<P>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
    Zn64BBase: CanHomFrom<I::Ring>,
{
    assert!(!poly_ring.is_zero(poly));

    event!(Level::TRACE, poly_deg = poly_ring.degree(poly).unwrap());

    let QQX = &poly_ring;
    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();

    let den_lcm = QQX
        .terms(poly)
        .map(|(c, _)| QQ.get_ring().den(c))
        .fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, b));

    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| {
        (
            ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c))
                .unwrap(),
            i,
        )
    }));
    let mut factorization = poly_factor_integer(&ZZX, f);
    factorization.sort_unstable_by_key(|(factor, e)| (ZZX.degree(factor).unwrap(), *e));

    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    return (
        factorization
            .into_iter()
            .map(|(f, e)| (QQX.normalize(ZZX_to_QQX.map(f)), e))
            .collect(),
        QQX.lc(poly).unwrap().clone(),
    );
}
