use tracing::instrument;

use crate::homomorphism::*;
use crate::rings::extension::*;
use crate::ring::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::zn::*;
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
#[instrument(skip_all, level = "trace")]
pub fn invert_over_local_zn<S>(ring: S, el: &El<S>) -> Option<El<S>>
    where S: RingStore,
        S::Type: FreeAlgebra,
        <<S::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing + SelfIso + ZnRing + FromModulusCreateableZnRing + Clone
{
    let base_ring = ring.base_ring();
    let poly_ring = DensePolyRing::new(base_ring, "X");
    let modulus = ring.generating_poly(&poly_ring, base_ring.identity());
    let poly = ring.poly_repr(&poly_ring, el, base_ring.identity());

    let (inverse, _) = local_zn_ring_bezout_identity(&poly_ring, &poly_ring.add_ref_fst(&modulus, poly), &modulus)?;
    return Some(ring.from_canonical_basis_extended((0..=poly_ring.degree(&inverse).unwrap()).map(|i| base_ring.clone_el(poly_ring.coefficient_at(&inverse, i)))));
}

#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::local::AsLocalPIR;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_invert_over_local_zn() {
    LogAlgorithmSubscriber::init_test();
    let base_ring = AsLocalPIR::from_zn(zn_64b::Zn64B::new(27)).unwrap();
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

    let base_ring = zn_64b::Zn64B::new(257).as_field().ok().unwrap();
    let array = |data: [i32; 2]| std::array::from_fn::<_, 2, _>(|i| base_ring.int_hom().map(data[i]));
    let ring = FreeAlgebraImpl::new(base_ring, 2, array([1, 0]));

    let a = ring.from_canonical_basis(array([-1, 2]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([0, 2]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));

    let a = ring.from_canonical_basis(array([2, 0]));
    assert_el_eq!(&ring, ring.one(), ring.mul(invert_over_local_zn(RingRef::new(ring.get_ring()), &a).unwrap(), a));
}