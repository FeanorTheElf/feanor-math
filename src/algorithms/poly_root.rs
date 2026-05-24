use tracing::instrument;

use crate::prelude::*;
use crate::ring_impls::poly::*;

/// Checks whether there exists a polynomial `g` such that `g^k = f`, and if yes,
/// returns `g`.
///
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::prelude::*;
/// # use feanor_math::ring_impls::poly::*;
/// # use feanor_math::ring_impls::poly::dense_poly::*;
/// # use feanor_math::algorithms::poly_gcd::*;
/// # use feanor_math::algorithms::poly_root::poly_root;
/// let poly_ring = DensePolyRing::new(ZZi64, "X");
/// let [f, f_sqrt] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 2 * X + 1, X + 1]);
/// assert_el_eq!(&poly_ring, f_sqrt, poly_root(&poly_ring, &f, 2).unwrap());
/// ```
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn poly_root<P>(poly_ring: P, f: &El<P>, k: usize) -> Option<El<P>>
where
    P: RingStore,
    P::Ring: PolyRing,
    <BaseRingStore<P> as RingStore>::Ring: DivisibilityRing + Domain,
{
    assert!(poly_ring.degree(&f).unwrap() % k == 0);
    let d = poly_ring.degree(&f).unwrap() / k;
    let ring = poly_ring.base_ring();
    let k_in_ring = ring.int_hom().map(k.try_into().unwrap());

    let mut result_reversed = Vec::new();
    result_reversed.push(ring.one());
    for i in 1..=d {
        let g = poly_ring.pow(poly_ring.from_terms((0..i).map(|j| (result_reversed[j].clone(), j))), k);
        let partition_sum = poly_ring.coefficient_at(&g, i);
        let next_coeff = ring.checked_div(
            &ring.sub_ref(poly_ring.coefficient_at(&f, k * d - i), partition_sum),
            &k_in_ring,
        )?;
        result_reversed.push(next_coeff);
    }

    let result = poly_ring.from_terms(result_reversed.into_iter().enumerate().map(|(i, c)| (c, d - i)));
    if poly_ring.eq_el(&f, &poly_ring.pow(result.clone(), k)) {
        return Some(result);
    } else {
        return None;
    }
}

#[cfg(test)]
use crate::ring_impls::poly::dense_poly::DensePolyRing;

#[test]
fn test_poly_root() {
    feanor_tracing::DelayedLogger::init_test();
    let ring = ZZbig;
    let poly_ring = DensePolyRing::new(ring, "X");
    let [f] = poly_ring.with_wrapped_indeterminate(|X| {
        [X.pow_ref(7) + X.pow_ref(6) + X.pow_ref(5) + X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1]
    });
    for k in 1..5 {
        assert_el_eq!(
            &poly_ring,
            &f,
            poly_root(&poly_ring, &poly_ring.pow(f.clone(), k), k).unwrap()
        );
    }

    let [f] = poly_ring.with_wrapped_indeterminate(|X| {
        [X.pow_ref(5) + 2 * X.pow_ref(4) + 3 * X.pow_ref(3) + 4 * X.pow_ref(2) + 5 * X + 6]
    });
    for k in 1..5 {
        assert_el_eq!(
            &poly_ring,
            &f,
            poly_root(&poly_ring, &poly_ring.pow(f.clone(), k), k).unwrap()
        );
    }
}
