
use crate::algorithms::poly_factor::factor_locally::poly_factor_integer;
use crate::computation::*;
use crate::pid::PrincipalIdealRingStore;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::divisibility::*;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::zn_64::*;
use crate::pid::EuclideanRing;

use super::IntegerRing;

#[stability::unstable(feature = "enable")]
pub fn poly_factor_rational<'a, P, I, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing,
        Zn64BBase: CanHomFrom<I::Type>,
        Controller: ComputationController
{
    assert!(!poly_ring.is_zero(poly));
    let QQX = &poly_ring;
    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();

    let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, b));
    
    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
    let mut factorization = poly_factor_integer(&ZZX, f, controller);
    factorization.sort_unstable_by_key(|(factor, e)| (ZZX.degree(factor).unwrap(), *e));

    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    return (
        factorization.into_iter().map(|(f, e)| (QQX.normalize(ZZX_to_QQX.map(f)), e)).collect(),
        QQ.clone_el(QQX.lc(poly).unwrap())
    );
}