
use tracing::{instrument, Level, span};

use crate::algorithms::poly_factor::factor_locally::poly_factor_integer;
use crate::pid::PrincipalIdealRingStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::divisibility::*;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::zn_64b::*;
use crate::pid::EuclideanRing;

use super::IntegerRing;

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_factor_rational<'a, P, I>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, El<BaseRing<P>>)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        BaseRing<P>: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing,
        Zn64BBase: CanHomFrom<I::Type>
{
    assert!(!poly_ring.is_zero(poly));

    span!(Level::INFO, "factor_poly_QQ", poly_deg = poly_ring.degree(poly).unwrap()).in_scope(|| {

        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();

        let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, b));
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let mut factorization = poly_factor_integer(&ZZX, f);
        factorization.sort_unstable_by_key(|(factor, e)| (ZZX.degree(factor).unwrap(), *e));

        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
        return (
            factorization.into_iter().map(|(f, e)| (QQX.normalize(ZZX_to_QQX.map(f)), e)).collect(),
            QQ.clone_el(QQX.lc(poly).unwrap())
        );
    })
}