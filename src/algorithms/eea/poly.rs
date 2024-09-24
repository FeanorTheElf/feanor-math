use std::mem::swap;

use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::divisibility::*;
use crate::rings::poly::*;
use crate::ring::*;
use crate::pid::*;

#[stability::unstable(feature = "enable")]
pub fn poly_eea_global<R>(fst: El<R>, snd: El<R>, ring: R) -> (El<R>, El<R>, El<R>) 
    where R: RingStore,
        R::Type: PolyRing + EuclideanRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Field
{
    let (mut a, mut b) = (ring.clone_el(&fst), ring.clone_el(&snd));

    let a_balance_factor = ring.get_ring().balance_element(&mut a);
    let b_balance_factor = ring.get_ring().balance_element(&mut b);

    let (mut sa, mut ta) = (ring.inclusion().map(ring.base_ring().invert(&a_balance_factor).unwrap()), ring.zero());
    let (mut sb, mut tb) = (ring.zero(), ring.inclusion().map(ring.base_ring().invert(&b_balance_factor).unwrap()));

    while !ring.is_zero(&b) {
        let balance_factor = ring.get_ring().balance_element(&mut b);
        let inv_balance_factor = ring.base_ring().invert(&balance_factor).unwrap();
        ring.inclusion().mul_assign_ref_map(&mut tb, &inv_balance_factor);
        ring.inclusion().mul_assign_ref_map(&mut sb, &inv_balance_factor);

        let scale_factor = ring.base_ring().pow(ring.base_ring().clone_el(ring.lc(&b).unwrap()), ring.degree(&b).unwrap());
        ring.inclusion().mul_assign_ref_map(&mut a, &scale_factor);
        ring.inclusion().mul_assign_ref_map(&mut ta, &scale_factor);
        ring.inclusion().mul_assign_ref_map(&mut sa, &scale_factor);

        let (quo, rem) = ring.euclidean_div_rem(a, &b);
        ta = ring.sub(ta, ring.mul_ref(&quo, &tb));
        sa = ring.sub(sa, ring.mul_ref(&quo, &sb));
        a = rem;
        
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return (sa, ta, a);
}

#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::rational::RationalField;

#[test]
fn test_polynomial_eea_global() {
    let ring = DensePolyRing::new(RationalField::new(BigIntRing::RING), "X");
    let [f, g, expected_gcd] = ring.with_wrapped_indeterminate(|X| [
        (X.pow_ref(2) + 1) * (X.pow_ref(3) + 2),
        (X.pow_ref(2) + 1) * (2 * X + 1),
        X.pow_ref(2) + 1
    ]);

    let (s, t, actual_gcd) = poly_eea_global(ring.clone_el(&f), ring.clone_el(&g), &ring);
    assert_el_eq!(ring, &expected_gcd, actual_gcd);
    assert_el_eq!(ring, &expected_gcd, ring.add(ring.mul_ref(&s, &f), ring.mul_ref(&t, &g)));
}