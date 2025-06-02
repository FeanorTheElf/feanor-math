
use crate::algorithms::poly_gcd::factor::poly_factor_integer;
use crate::computation::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::*;
use crate::divisibility::*;
use crate::rings::rational::RationalFieldBase;
use crate::rings::zn::zn_64::*;
use crate::algorithms::eea::signed_lcm;
use crate::pid::EuclideanRing;

use super::IntegerRing;

#[stability::unstable(feature = "enable")]
pub fn poly_factor_rational<'a, P, I>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing,
        ZnBase: CanHomFrom<I::Type>
{
    assert!(!poly_ring.is_zero(poly));
    let QQX = &poly_ring;
    let QQ = QQX.base_ring();
    let ZZ = QQ.base_ring();

    let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
    
    let ZZX = DensePolyRing::new(ZZ, "X");
    let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
    let mut factorization = poly_factor_integer(&ZZX, f, LOG_PROGRESS);
    factorization.sort_unstable_by_key(|(factor, e)| (ZZX.degree(factor).unwrap(), *e));

    let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    return (
        factorization.into_iter().map(|(f, e)| (QQX.normalize(ZZX_to_QQX.map(f)), e)).collect(),
        QQ.clone_el(QQX.lc(poly).unwrap())
    );
}