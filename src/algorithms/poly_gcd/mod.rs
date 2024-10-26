use gcd::poly_gcd_local;
use global::poly_power_decomposition_finite_field;
use local::IntegerPolyGCDRing;
use squarefree_part::poly_power_decomposition_local;

use crate::algorithms::eea::signed_lcm;
use crate::computation::DontObserve;
use crate::divisibility::*;
use crate::homomorphism::*;
use crate::integer::*;
use crate::pid::*;
use crate::rings::rational::RationalFieldBase;
use crate::ring::*;
use crate::rings::poly::dense_poly::*;
use crate::rings::poly::*;
use crate::rings::finite::*;
use crate::field::*;

use super::eea::gcd;

pub mod global;
pub mod local;
pub mod hensel;
pub mod squarefree_part;
pub mod gcd;
pub mod factor;

const INCREASE_EXPONENT_PER_ATTEMPT_CONSTANT: f64 = 1.5;

pub trait PolyGCDRing {

    ///
    /// Computes the square-free part of a polynomial `f`, which is the largest-degree squarefree
    /// polynomial `d` such that `d | a f` for some non-zero-divisor `a` of this ring.
    /// 
    /// This value is unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that it be monic, which makes it unique.
    /// 
    fn squarefree_part<P>(poly_ring: P, poly: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        poly_ring.prod(Self::power_decomposition(poly_ring, poly).into_iter().map(|(f, _)| f))
    }

    ///
    /// Compute square-free polynomials `f1, f2, ...` such that `a f = f1 f2^2 f3^3 ...`
    /// for some non-zero-divisor `a` of this ring. They are returned as tuples `(fi, i)`
    /// where `deg(fi) > 0`.
    /// 
    /// These values are unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that all `fi` be monic, which makes them unique.
    /// 
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;
    
    ///
    /// Computes the greatest common divisor of two polynomials `f, g` over the fraction field,
    /// which is the largest-degree polynomial `d` such that `d | a f, a g` for some non-zero-divisor
    /// `a` of this ring.
    /// 
    /// This value is unique up to multiplication by units. If the base ring is a field,
    /// we impose the additional constraint that it be monic, which makes it unique.
    /// 
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>;
}

///
/// Computes the map
/// ```text
///   R[X] -> R[X],  f(X) -> a^(deg(f) - 1) f(X / a)
/// ```
/// that can be used to make polynomials over a domain monic (when setting `a = lc(f)`).
/// 
fn evaluate_aX<P>(poly_ring: P, f: &El<P>, a: &El<<P::Type as RingExtension>::BaseRing>) -> El<P>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(f) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&f).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| if i == d { (ring.checked_div(c, a).unwrap(), d) } else { (ring.mul_ref_fst(c, ring.pow(ring.clone_el(a), d - i - 1)), i) }));
    return result;
}

///
/// Computes the inverse to [`evaluate_aX()`].
/// 
fn unevaluate_aX<P>(poly_ring: P, g: &El<P>, a: &El<<P::Type as RingExtension>::BaseRing>) -> El<P>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(g) {
        return poly_ring.zero();
    }
    let ring = poly_ring.base_ring();
    let d = poly_ring.degree(&g).unwrap();
    let result = poly_ring.from_terms(poly_ring.terms(g).map(|(c, i)| if i == d { (ring.clone_el(a), d) } else { (ring.checked_div(c, &ring.pow(ring.clone_el(a), d - i - 1)).unwrap(), i) }));
    return result;
}

///
/// Given a polynomial `f` over a PID, returns `(f/cont(f), cont(f))`, where `cont(f)`
/// is the content of `f`, i.e. the gcd of all coefficients of `f`.
/// 
#[stability::unstable(feature = "enable")]
pub fn make_primitive<P>(poly_ring: P, f: &El<P>) -> (El<P>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing + Domain
{
    if poly_ring.is_zero(f) {
        return (poly_ring.zero(), poly_ring.base_ring().one());
    }
    let ring = poly_ring.base_ring();
    let content = poly_ring.terms(f).map(|(c, _)| c).fold(ring.zero(), |a, b| ring.ideal_gen(&a, b));
    let result = poly_ring.from_terms(poly_ring.terms(f).map(|(c, i)| (ring.checked_div(c, &content).unwrap(), i)));
    return (result, content);
}

///
/// A weaker version of [`make_primitive()`] that just divides out the "balance factor" of
/// all coefficients of `f`. The definition of the balance factor is completely up to the
/// underlying ring, see [`DivisibilityRing::balance_factor()`].
/// 
fn balance_poly<P>(poly_ring: P, f: El<P>) -> (El<P>, El<<P::Type as RingExtension>::BaseRing>)
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    if poly_ring.is_zero(&f) {
        return (poly_ring.zero(), poly_ring.base_ring().one());
    }
    let ring = poly_ring.base_ring();
    let factor = ring.get_ring().balance_factor(poly_ring.terms(&f).map(|(c, _)| c));
    let result = poly_ring.from_terms(poly_ring.terms(&f).map(|(c, i)| (ring.checked_div(c, &factor).unwrap(), i)));
    return (result, factor);
}

///
/// Checks whether there exists a polynomial `g` such that `g^k = f`, and if yes,
/// returns `g`.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::algorithms::poly_gcd::*;
/// let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let [f, f_sqrt] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 2 * X + 1, X + 1]);
/// assert_el_eq!(&poly_ring, f_sqrt, poly_root(&poly_ring, &f, 2).unwrap());
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn poly_root<P>(poly_ring: P, f: &El<P>, k: usize) -> Option<El<P>>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain
{
    assert!(poly_ring.degree(&f).unwrap() % k == 0);
    let d = poly_ring.degree(&f).unwrap() / k;
    let ring = poly_ring.base_ring();
    let k_in_ring = ring.int_hom().map(k as i32);

    let mut result_reversed = Vec::new();
    result_reversed.push(ring.one());
    for i in 1..=d {
        let g = poly_ring.pow(poly_ring.from_terms((0..i).map(|j| (ring.clone_el(&result_reversed[j]), j))), k);
        let partition_sum = poly_ring.coefficient_at(&g, i);
        let next_coeff = ring.checked_div(&ring.sub_ref(poly_ring.coefficient_at(&f, k * d - i), partition_sum), &k_in_ring)?;
        result_reversed.push(next_coeff);
    }

    let result = poly_ring.from_terms(result_reversed.into_iter().enumerate().map(|(i, c)| (c, d - i)));
    if poly_ring.eq_el(&f, &poly_ring.pow(poly_ring.clone_el(&result), k)) {
        return Some(result);
    } else {
        return None;
    }
}

impl<R> PolyGCDRing for R
    where R: FiniteRing + Field
{
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let new_poly_ring = DensePolyRing::new(poly_ring.base_ring(), "X");
        let new_poly = new_poly_ring.from_terms(poly_ring.terms(poly).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i)));
        poly_power_decomposition_finite_field(&new_poly_ring, &new_poly).into_iter().map(|(f, k)| 
            (poly_ring.from_terms(new_poly_ring.terms(&f).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i))), k)
        ).collect()
    }
    
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let new_poly_ring = DensePolyRing::new(poly_ring.base_ring(), "X");
        let new_lhs = new_poly_ring.from_terms(poly_ring.terms(lhs).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i)));
        let new_rhs = new_poly_ring.from_terms(poly_ring.terms(rhs).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i)));
        let new_gcd = gcd(new_lhs, new_rhs, &new_poly_ring);
        poly_ring.from_terms(new_poly_ring.terms(&new_gcd).map(|(c, i)| (poly_ring.base_ring().clone_el(c), i)))
    }
}

impl<I> PolyGCDRing for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerPolyGCDRing
{
    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        assert!(!poly_ring.is_zero(poly));
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm = QQX.terms(poly).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let f = ZZX.from_terms(QQX.terms(poly).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let power_decomp = poly_power_decomposition_local(&ZZX, f, DontObserve);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return power_decomp.into_iter().map(|(f, k)| (QQX.normalize(ZZX_to_QQX.map(f)), k)).collect();
    }
    
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        if poly_ring.is_zero(lhs) {
            return poly_ring.clone_el(rhs);
        } else if poly_ring.is_zero(rhs) {
            return poly_ring.clone_el(lhs);
        }
        let QQX = &poly_ring;
        let QQ = QQX.base_ring();
        let ZZ = QQ.base_ring();
    
        let den_lcm_lhs = QQX.terms(lhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        let den_lcm_rhs = QQX.terms(rhs).map(|(c, _)| QQ.get_ring().den(c)).fold(ZZ.one(), |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ));
        
        let ZZX = DensePolyRing::new(ZZ, "X");
        let lhs = ZZX.from_terms(QQX.terms(lhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_lhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let rhs = ZZX.from_terms(QQX.terms(rhs).map(|(c, i)| (ZZ.checked_div(&ZZ.mul_ref(&den_lcm_rhs, QQ.get_ring().num(c)), QQ.get_ring().den(c)).unwrap(), i)));
        let result = poly_gcd_local(&ZZX, lhs, rhs, DontObserve);
        let ZZX_to_QQX = QQX.lifted_hom(&ZZX, QQ.inclusion());
    
        return QQX.normalize(ZZX_to_QQX.map(result));
    }
}

#[test]
fn test_poly_root() {
    let ring = BigIntRing::RING;
    let poly_ring = DensePolyRing::new(ring, "X");
    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(7) + X.pow_ref(6) + X.pow_ref(5) + X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1]);
    for k in 1..5 {
        assert_el_eq!(&poly_ring, &f, poly_root(&poly_ring, &poly_ring.pow(poly_ring.clone_el(&f), k), k).unwrap());
    }

    let [f] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(5) + 2 * X.pow_ref(4) + 3 * X.pow_ref(3) + 4 * X.pow_ref(2) + 5 * X + 6]);
    for k in 1..5 {
        assert_el_eq!(&poly_ring, &f, poly_root(&poly_ring, &poly_ring.pow(poly_ring.clone_el(&f), k), k).unwrap());
    }
}