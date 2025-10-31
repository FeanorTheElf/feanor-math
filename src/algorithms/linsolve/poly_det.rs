use std::alloc::Allocator;
use std::collections::BTreeMap;
use std::cmp::max;
use std::alloc::Global;

use crate::algorithms::interpolate::interpolate_multivariate;
use crate::algorithms::linsolve::smith;
use crate::homomorphism::Homomorphism;
use crate::integer::int_cast;
use crate::integer::BigIntRing;
use crate::iters::multi_cartesian_product;
use crate::matrix::AsPointerToSlice;
use crate::matrix::OwnedMatrix;
use crate::matrix::Submatrix;
use crate::pid::*;
use crate::seq::VectorView;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::ordered::OrderedRingStore;
use crate::seq::subvector::SubvectorFn;
use crate::seq::SelfSubvectorFn;
use crate::seq::VectorFn;
use crate::algorithms::int_factor::factor;

fn determinant_poly_matrix_base<P, V, A, I>(A: Submatrix<V, El<P>>, poly_ring: P, allocator: A, total_max_degrees: BTreeMap<usize, usize>, interpolation_points: I) -> El<P>
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        V: AsPointerToSlice<El<P>>,
        A: Allocator,
        I: VectorFn<El<<P::Type as RingExtension>::BaseRing>>,
{
    assert_eq!(A.row_count(), A.col_count());
    let n = A.row_count();
    let R = poly_ring.base_ring();

    let interpolation_grid_dims = total_max_degrees.iter().map(|(_var, exp)| *exp as i64 + 1).collect::<Vec<_>>();
    let interpolation_grid_size = StaticRing::<i64>::RING.prod(interpolation_grid_dims.iter().copied());
    let mut determinants: Vec<El<<P::Type as RingExtension>::BaseRing>, &A> = Vec::with_capacity_in(interpolation_grid_size as usize, &allocator);
    let mut evaluated_matrix = OwnedMatrix::zero_in(n, n, R, &allocator);

    for _ in multi_cartesian_product((0..total_max_degrees.len()).map(|i| interpolation_points.iter().take(interpolation_grid_dims[i] as usize)), |assignment| {
        for i in 0..n {
            for j in 0..n {
                *evaluated_matrix.at_mut(i, j) = poly_ring.evaluate(A.at(i, j), assignment.into_ring_el_fn(R), &R.identity());
            }
        }
        determinants.push(smith::determinant_using_pre_smith(R, evaluated_matrix.data_mut(), Global));
    }, |_, x| R.clone_el(x)) {}

    let result = interpolate_multivariate(
        poly_ring, 
        (0..total_max_degrees.len()).map_fn(|i| SubvectorFn::new(&interpolation_points).restrict(0..(interpolation_grid_dims[i] as usize))), 
        determinants, 
        &allocator
    ).unwrap();

    return result;
}

///
/// Computes the determinant in exponential time using the naive Laplace formula.
/// 
#[stability::unstable(feature = "enable")]
pub fn naive_det<R, V, A>(A: Submatrix<V, El<R>>, ring: R, allocator: A) -> El<R>
    where R: RingStore + Copy,
        V: AsPointerToSlice<El<R>>,
        A: Allocator + Copy
{
    assert_eq!(A.row_count(), A.col_count());
    let n = A.row_count();
    if n == 1 {
        return ring.clone_el(A.at(0, 0));
    }
    let mut tmp_matrix = OwnedMatrix::zero_in(n - 1, n - 1, ring, allocator);

    <_ as RingStore>::sum(&ring, (0..n).map(|i| {
        if ring.is_zero(A.at(i, 0)) {
            return ring.zero();
        }
        for k in 0..(n - 1) {
            if k < i {
                for j in 0..(n - 1) {
                    *tmp_matrix.at_mut(k, j) = ring.clone_el(A.at(k, j + 1));
                }
            } else {
                for j in 0..(n - 1) {
                    *tmp_matrix.at_mut(k, j) = ring.clone_el(A.at(k + 1, j + 1));
                }
            }
        }
        let result = ring.mul_ref_fst(A.at(i, 0), naive_det(tmp_matrix.data(), ring, allocator));
        if i % 2 == 0 {
            return result;
        } else {
            return ring.negate(result);
        }
    }))
}

#[stability::unstable(feature = "enable")]
pub fn determinant_poly_matrix<P, V, A>(A: Submatrix<V, El<P>>, poly_ring: P, allocator: A) -> El<P>
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalIdealRing,
        V: AsPointerToSlice<El<P>>,
        A: Allocator
{
    assert_eq!(A.row_count(), A.col_count());
    let n = A.row_count();
    let mut total_degs = (0..poly_ring.indeterminate_count()).map(|var| (var, 0)).collect::<BTreeMap<_, _>>();

    for i in 0..n {
        let mut max_degs = (0..poly_ring.indeterminate_count()).map(|var| (var, 0)).collect::<BTreeMap<_, _>>();
        for j in 0..n {
            for (var, exp) in poly_ring.appearing_indeterminates(A.at(i, j)) {
                let old = max_degs.insert(var, max(max_degs[&var], exp));
                debug_assert!(old.is_some());
            }
        }
        for (var, exp) in max_degs {
            let old = total_degs.insert(var, exp + total_degs[&var]);
            debug_assert!(old.is_some());
        }
    }
    // check if we need an extension field
    let max_interpolation_point_count = total_degs.iter().map(|(_, exp)| *exp as i64).max().unwrap_or(0) + 1;
    assert!(max_interpolation_point_count <= i32::MAX as i64);

    let characteristic = poly_ring.base_ring().characteristic(&BigIntRing::RING).unwrap();
    if !BigIntRing::RING.is_zero(&characteristic) {
        let characteristic_smallest_factor = factor(&BigIntRing::RING, characteristic).into_iter().map(|(p, _)| p).min_by(|l, r| BigIntRing::RING.cmp(l, r)).unwrap();
        if BigIntRing::RING.is_lt(&characteristic_smallest_factor, &int_cast(max_interpolation_point_count, &BigIntRing::RING, &StaticRing::<i64>::RING)) {
            // we need an extension field
            return unimplemented!();
        } else {
            return determinant_poly_matrix_base(A, &poly_ring, allocator, total_degs, (0..(max_interpolation_point_count as usize)).map_fn(|x| poly_ring.base_ring().int_hom().map(x.try_into().unwrap())));
        }
    } else {
        return determinant_poly_matrix_base(A, &poly_ring, allocator, total_degs, (0..(max_interpolation_point_count as usize)).map_fn(|x| poly_ring.base_ring().int_hom().map(x.try_into().unwrap())));
    }
}

#[cfg(test)]
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;

#[test]
fn test_determinant_poly_matrix() {
    LogAlgorithmSubscriber::init_test();
    let poly_ring: MultivariatePolyRingImpl<_, _> = MultivariatePolyRingImpl::new(StaticRing::<i128>::RING, 3);

    let det2 = |A: Submatrix<_, _>| poly_ring.sub(poly_ring.mul_ref(A.at(0, 0), A.at(1, 1)), poly_ring.mul_ref(A.at(0, 1), A.at(1, 0)));
    let det3 = |A: Submatrix<_, _>| poly_ring.sum([
        poly_ring.mul_ref_fst(A.at(0, 0), det2(A.submatrix(1..3, 1..3))),
        poly_ring.negate(poly_ring.mul_ref_fst(A.at(1, 0), det2(Submatrix::<Vec<_>, _>::from_2d(&(0..2).map(|i| (0..2).map(|j| if i == 0 { poly_ring.clone_el(A.at(0, j + 1)) } else { poly_ring.clone_el(A.at(2, j + 1)) }).collect::<Vec<_>>()).collect::<Vec<_>>())))),
        poly_ring.mul_ref_fst(A.at(2, 0), det2(A.submatrix(0..2, 1..3)))
    ].into_iter());
    let det4 = |A: Submatrix<_, _>| poly_ring.sum([
        poly_ring.mul_ref_fst(A.at(0, 0), det3(A.submatrix(1..4, 1..4))),
        poly_ring.negate(poly_ring.mul_ref_fst(A.at(1, 0), det3(Submatrix::<Vec<_>, _>::from_2d(&(0..3).map(|i| (0..3).map(|j| if i == 0 { poly_ring.clone_el(A.at(0, j + 1)) } else { poly_ring.clone_el(A.at(i + 1, j + 1)) }).collect::<Vec<_>>()).collect::<Vec<_>>())))),
        poly_ring.mul_ref_fst(A.at(2, 0), det3(Submatrix::<Vec<_>, _>::from_2d(&(0..3).map(|i| (0..3).map(|j| if i == 2 { poly_ring.clone_el(A.at(3, j + 1)) } else { poly_ring.clone_el(A.at(i, j + 1)) }).collect::<Vec<_>>()).collect::<Vec<_>>()))),
        poly_ring.negate(poly_ring.mul_ref_fst(A.at(3, 0), det3(A.submatrix(0..3, 1..4))))
    ].into_iter());

    let matrix = [
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, _]| [x0.clone(), x1.clone()]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, _, x2]| [x2.clone(), x0.clone()])
    ];
    let matrix = Submatrix::<Vec<_>, _>::from_2d(&matrix[..]);
    let expected = poly_ring.with_wrapped_indeterminates(|[x0, x1, x2]| [x0 * x0 - x1 * x2]).into_iter().next().unwrap();
    assert_el_eq!(&poly_ring, &expected, naive_det(matrix, &poly_ring, Global));
    assert_el_eq!(&poly_ring, &expected, determinant_poly_matrix(matrix, &poly_ring, Global));

    let matrix = [
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone(), x1.clone(), x2.clone()]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x2.clone(), x0.clone(), x1.clone()]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x1.clone(), x2.clone(), x0.clone()])
    ];
    let matrix = Submatrix::<Vec<_>, _>::from_2d(&matrix[..]);
    assert_el_eq!(&poly_ring, det3(matrix), determinant_poly_matrix(matrix, &poly_ring, Global));
    assert_el_eq!(&poly_ring, det3(matrix), naive_det(matrix, &poly_ring, Global));

    let matrix = [
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone(), x1.clone(), x2.clone()]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone(), x1.clone(), x2.clone()]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone(), x1.clone(), x2.clone()])
    ];
    let matrix = Submatrix::<Vec<_>, _>::from_2d(&matrix[..]);
    assert_el_eq!(&poly_ring, &poly_ring.zero(), determinant_poly_matrix(matrix, &poly_ring, Global));
    assert_el_eq!(&poly_ring, &poly_ring.zero(), naive_det(matrix, &poly_ring, Global));

    let matrix = [
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone(),             x0 + x1,                x0 + x1 + x2,           x0.clone().pow(2)]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0 * x1,                x0 * x2,                x1 * x2,                x1.clone().pow(2)]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x1 * 2,                 x2.clone(),             x0 + 1,                 x2.clone().pow(2)]),
        poly_ring.with_wrapped_indeterminates_dyn(|[x0, x1, x2]| [x0.clone().pow(2) - x1, x1.clone().pow(2) - x2, x2.clone().pow(2) - x0, x0.clone()])
    ];
    let matrix = Submatrix::<Vec<_>, _>::from_2d(&matrix[..]);
    assert_el_eq!(&poly_ring, det4(matrix), determinant_poly_matrix(matrix, &poly_ring, Global));
    assert_el_eq!(&poly_ring, det4(matrix), naive_det(matrix, &poly_ring, Global));
}