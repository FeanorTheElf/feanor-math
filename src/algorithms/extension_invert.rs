use crate::homomorphism::*;
use crate::local::*;
use crate::divisibility::*;
use crate::primitive_int::StaticRing;
use crate::rings::extension::*;
use crate::ring::*;
use crate::rings::field::AsFieldBase;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::zn::*;
use crate::integer::*;
use crate::seq::*;
use crate::rings::poly::*;
use crate::algorithms::linsolve::LinSolveRing;
use crate::algorithms::poly_gcd::hensel::local_zn_ring_bezout_identity;

///
/// Computes the inverse of a unit `a` in the ring `(Z/p^eZ)[X]/(f(X))` with a
/// monic irreducible polynomial `f(X)`. Returns `None` if `a` is not a unit.
/// 
/// # Algorithm
/// 
/// This function will compute an inverse of `a` modulo `p` using EEA and
/// then lift it to `Z/p^eZ` using Hensel's lemma. In particular, the
/// complexity is `O(deg(f)^2)` and not cubic (which would be the complexity
/// of using matrix inversion to compute `a^-1`).
/// 
#[stability::unstable(feature = "enable")]
pub fn invert_over_local_zn<S>(ring: S, el: &El<S>) -> Option<El<S>>
    where S: RingStore,
        S::Type: FreeAlgebra,
        <<S::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing + ZnRing + PrincipalLocalRing + FromModulusCreateableZnRing,
        AsFieldBase<RingValue<<<S::Type as RingExtension>::BaseRing as RingStore>::Type>>: CanIsoFromTo<<<S::Type as RingExtension>::BaseRing as RingStore>::Type> + SelfIso
{
    let base_ring = ring.base_ring();
    let poly_ring = DensePolyRing::new(base_ring, "X");
    let modulus = ring.generating_poly(&poly_ring, base_ring.identity());

    // we consider the element as a polynomial `f(X)` of degree `< rank` and 
    // write `f(X) = g(X) + p h(X)` where `g(X)` has unit coefficients. This allows
    // us to normalize `g(X)`, which is required for `local_zn_ring_bezout_identity()`
    let mut nilpotent_part = Vec::new();
    let mut possibly_invertible_part = Vec::new();
    for (i, c) in ring.wrt_canonical_basis(el).iter().enumerate() {
        if base_ring.divides(&c, base_ring.max_ideal_gen()) {
            nilpotent_part.push((c, i));
        } else {
            possibly_invertible_part.push((c, i));
        }
    }
    let nilpotent_part = poly_ring.from_terms(nilpotent_part.into_iter());
    let nilpotent_part = ring.from_canonical_basis((0..ring.rank()).map(|i| base_ring.clone_el(poly_ring.coefficient_at(&nilpotent_part, i))));
    let mut possibly_invertible_part = poly_ring.from_terms(possibly_invertible_part.into_iter());
    let lc_inv = base_ring.invert(poly_ring.lc(&possibly_invertible_part)?).unwrap();
    poly_ring.inclusion().mul_assign_ref_map(&mut possibly_invertible_part, &lc_inv);

    let (mut inverse, _) = local_zn_ring_bezout_identity(&poly_ring, &possibly_invertible_part, &modulus)?;
    poly_ring.inclusion().mul_assign_map(&mut inverse, lc_inv);

    // `inverse` is now the inverse of `possibly_invertible_part`, and we can
    // annihilate `nilpotent_part` by the third binomial formula, since a sufficiently
    // large power of it is zero
    let inverse = ring.from_canonical_basis((0..ring.rank()).map(|i| base_ring.clone_el(poly_ring.coefficient_at(&inverse, i))));

    let mut nilpotent_correction = ring.negate(ring.mul_ref_snd(nilpotent_part, &inverse));
    let mut result = inverse;
    for _ in 0..StaticRing::<i64>::RING.abs_log2_ceil(&base_ring.nilpotent_power().unwrap_or(1).try_into().unwrap()).unwrap() {
        ring.mul_assign(&mut result, ring.add_ref_snd(ring.one(), &nilpotent_correction));
        ring.square(&mut nilpotent_correction);
    }
    return Some(result);
}

#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::local::AsLocalPIR;

#[test]
fn test_invert_over_local_zn() {
    let base_ring = AsLocalPIR::from_zn(zn_64::Zn::new(27)).unwrap();
    let array = |data: [i32; 4]| std::array::from_fn::<_, 4, _>(|i| base_ring.int_hom().map(data[i]));
    let ring = FreeAlgebraImpl::new(base_ring, 4, array([1, 0, 0, 1]));

    let a = ring.from_canonical_basis(array([1, 0, 0, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([1, 1, 0, 1]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([1, 3, 0, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([1, 2, 3, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([3, 2, 9, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([1, 3, 9, 9]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([3, 1, 0, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([0, 3, 9, 9]));
    assert!(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).is_none());

    let base_ring = zn_64::Zn::new(257).as_field().ok().unwrap();
    let array = |data: [i32; 2]| std::array::from_fn::<_, 2, _>(|i| base_ring.int_hom().map(data[i]));
    let ring = FreeAlgebraImpl::new(base_ring, 2, array([1, 0]));

    let a = ring.from_canonical_basis(array([-1, 2]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([0, 2]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([2, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));
}