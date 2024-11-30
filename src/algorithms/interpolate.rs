use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::integer::IntegerRingStore;
use crate::iters::multi_cartesian_product;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::rings::poly::*;
use crate::seq::*;
use crate::homomorphism::Homomorphism;
use crate::rings::poly::dense_poly::DensePolyRing;

use std::alloc::Allocator;
use std::cmp::min;
use std::ops::Range;

#[allow(unused)]
#[stability::unstable(feature = "enable")]
fn invert_many<R>(ring: R, values: &[El<R>], out: &mut [El<R>]) -> Result<(), ()>
    where R: RingStore,
        R::Type: DivisibilityRing
{
    assert_eq!(out.len(), values.len());
    out[0] = ring.clone_el(&values[0]);
    for i in 1..out.len() {
        out[i] = ring.mul_ref(&values[i], &out[i - 1]);
    }
    let joint_inv = ring.invert(&out[out.len() - 1]).ok_or(())?;
    out[out.len() - 1] = joint_inv;
    for i in (1..out.len()).rev() {
        let (fst, snd) = out.split_at_mut(i);
        ring.mul_assign_ref(&mut fst[i - 1], &snd[0]);
        ring.mul_assign_ref(&mut snd[0], &values[i]);
        std::mem::swap(&mut fst[i - 1], &mut snd[0]);
    }
    return Ok(());
}

///
/// Computes `out[i] = prod_(j != i) values[j]`.
/// 
/// This algorithm recursively halfes the input, and thus it implicitly pads the input to the
/// next power-of-two. The time complexity is then `O(n sum_(0 <= i <= log n) T(2^i))`
/// where `T(d)` is the complexity of multiplying two products of `d` input elements. If the cost
/// of multiplication is constant, this becomes `O(n log n T)`.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::algorithms::interpolate::*;
/// # use feanor_math::seq::*;
/// let ring = StaticRing::<i64>::RING;
/// let mut result = [0; 6];
/// product_except_one(ring, (1..7).map_fn(|x| x as i64), &mut result);
/// let factorial_6 = 6 * 5 * 4 * 3 * 2 * 1;
/// // `product_except_one()` computes exactly these values, but without using any divisions
/// let expected = [factorial_6 / 1, factorial_6 / 2, factorial_6 / 3, factorial_6 / 4, factorial_6 / 5, factorial_6 / 6];
/// assert_eq!(expected, result);
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn product_except_one<V, R>(ring: R, values: V, out: &mut [El<R>])
    where R: RingStore,
        V: VectorFn<El<R>>
{
    assert_eq!(values.len(), out.len());
    let n = values.len();
    let log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n <= (1 << log2_n));
    if n % 2 == 0 {
        for i in 0..n {
            out[i] = values.at(i ^ 1);
        }
    } else {
        for i in 0..(n - 1) {
            out[i] = values.at(i ^ 1);
        }
        out[n - 1] = ring.one();
    }
    for s in 1..log2_n {
        for j in 0..(1 << (log2_n - s - 1)) {
            let block_index = j << (s + 1);
            if block_index + (1 << s) < n {
                let (fst, snd) = (&mut out[block_index..min(n, block_index + (1 << (s + 1)))]).split_at_mut(1 << s);
                let snd_block_prod = ring.mul_ref_fst(&snd[0], values.at(block_index + (1 << s)));
                let fst_block_prod = ring.mul_ref_fst(&fst[0], values.at(block_index));
                for i in 0..(1 << s) {
                    ring.mul_assign_ref(&mut fst[i], &snd_block_prod);
                }
                for i in 0..snd.len() {
                    ring.mul_assign_ref(&mut snd[i], &fst_block_prod);
                }
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum InterpolationError {
    NotInvertible
}

///
/// Uses Lagrange interpolation to compute the interpolation polynomial of the given values.
/// Concretely, this is the univariate polynomial `f` of degree `< x.len()` such that `f(x[i]) = y[i]`
/// for all `i`.
/// 
/// If no such polynomial exists (this is only possible if the base ring is not a field), an 
/// error is returned.
/// 
/// # Example
/// ```
/// # #![feature(allocator_api)]
/// # use std::alloc::Global;
/// # use feanor_math::ring::*;
/// # use feanor_math::seq::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::algorithms::interpolate::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// let ZZX = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let [expected] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
/// let actual = interpolate(&ZZX, [1, 2, 6].copy_els(), [2, 5, 37].copy_els(), Global).unwrap();
/// assert_el_eq!(&ZZX, expected, actual);
/// ```
/// In some cases the interpolation polynomial does not exist.
/// ```
/// # #![feature(allocator_api)]
/// # use std::alloc::Global;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::seq::*;
/// # use feanor_math::algorithms::interpolate::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// let ZnX = DensePolyRing::new(Zn::new(25), "X");
/// let ZZ_to_Zn = ZnX.base_ring().int_hom();
/// // since `1 = 6 mod 5`, the values of any polynomial at 1 resp. 6 must be the same modulo 5
/// let actual = interpolate(&ZnX, [ZZ_to_Zn.map(1), ZZ_to_Zn.map(6)].copy_els(), [ZZ_to_Zn.map(1), ZZ_to_Zn.map(2)].copy_els(), Global);
/// assert!(actual.is_err());
/// ```
/// 
#[stability::unstable(feature = "enable")]
pub fn interpolate<P, V1, V2, A: Allocator>(poly_ring: P, x: V1, y: V2, allocator: A) -> Result<El<P>, InterpolationError>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        V1: VectorFn<El<<P::Type as RingExtension>::BaseRing>>,
        V2: VectorFn<El<<P::Type as RingExtension>::BaseRing>>
{
    assert_eq!(x.len(), y.len());
    let mut nums = Vec::with_capacity_in(x.len(), &allocator);
    nums.resize_with(x.len(), || poly_ring.zero());
    let R = poly_ring.base_ring();
    product_except_one(&poly_ring, (0..x.len()).map_fn(|i| poly_ring.from_terms([(R.negate(x.at(i)), 0), (R.one(), 1)].into_iter())), &mut nums[..]);

    let mut denoms = Vec::with_capacity_in(x.len(), &allocator);
    denoms.extend((0..x.len()).map(|i| poly_ring.evaluate(&nums[i], &x.at(i), &R.identity())));
    let mut factors = Vec::with_capacity_in(x.len(), &allocator);
    factors.resize_with(x.len(), || R.zero());
    product_except_one(R, (&denoms[..]).into_clone_ring_els(R), &mut factors);
    let denominator = R.mul_ref(&factors[0], &denoms[0]);
    for i in 0..x.len() {
        R.mul_assign(&mut factors[i], y.at(i));
    }

    if let Some(inv) = R.invert(&denominator) {
        return Ok(poly_ring.inclusion().mul_map(<_ as RingStore>::sum(&poly_ring, nums.into_iter().zip(factors.into_iter()).map(|(num, c)| poly_ring.inclusion().mul_map(num, c))), inv));
    } else {
        let scaled_result = <_ as RingStore>::sum(&poly_ring, nums.into_iter().zip(factors.into_iter()).map(|(num, c)| poly_ring.inclusion().mul_map(num, c)));
        let mut failed_division = false;
        let result = poly_ring.from_terms(poly_ring.terms(&scaled_result).map_while(|(c, i)| match R.checked_div(&c, &denominator) {
            Some(c) => Some((c, i)),
            None => {
                failed_division = true;
                None
            }
        }));
        if failed_division {
            return Err(InterpolationError::NotInvertible);
        } else {
            return Ok(result);
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn interpolate_multivariate<P, V1, V2, A, A2>(poly_ring: P, interpolation_points: V1, mut values: Vec<El<<P::Type as RingExtension>::BaseRing>, A2>, allocator: A) -> Result<El<P>, InterpolationError>
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
        V1: VectorFn<V2>,
        V2: VectorFn<El<<P::Type as RingExtension>::BaseRing>>,
        A: Allocator,
        A2: Allocator
{
    let dim_prod = |range: Range<usize>| <_ as RingStore>::prod(&StaticRing::<i64>::RING, range.map(|i| interpolation_points.at(i).len() as i64)) as usize;
    assert_eq!(interpolation_points.len(), poly_ring.indeterminate_count());
    let n = poly_ring.indeterminate_count();
    assert_eq!(values.len(), dim_prod(0..n));

    let uni_poly_ring = DensePolyRing::new_with(poly_ring.base_ring(), "X", &allocator, STANDARD_CONVOLUTION);

    for i in (0..n).rev() {
        let leading_dim = dim_prod((i + 1)..n);
        let outer_block_count = dim_prod(0..i);
        let len = interpolation_points.at(i).len();
        let outer_block_size = leading_dim * len;
        for outer_block_index in 0..outer_block_count {
            for inner_block_index in 0..leading_dim {
                let block_start = inner_block_index + outer_block_index * outer_block_size;
                let poly = interpolate(&uni_poly_ring, interpolation_points.at(i), (&values[..]).into_clone_ring_els(poly_ring.base_ring()).restrict(block_start..(block_start + outer_block_size + 1 - leading_dim)).step_by_fn(leading_dim), &allocator)?;
                for j in 0..len {
                    values[block_start + leading_dim * j] = poly_ring.base_ring().clone_el(uni_poly_ring.coefficient_at(&poly, j));
                }
            }
        }
    }
    return Ok(poly_ring.from_terms(
        multi_cartesian_product((0..n).map(|i| 0..interpolation_points.at(i).len()), |idxs| poly_ring.get_ring().create_monomial(idxs.iter().map(|e| *e)), |_, x| *x)
            .zip(values.into_iter())
            .map(|(m, c)| (c, m))
    ));
}

#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use multivariate_impl::MultivariatePolyRingImpl;

use super::convolution::STANDARD_CONVOLUTION;

#[test]
fn test_product_except_one() {
    let ring = StaticRing::<i64>::RING;
    let data = [2, 3, 5, 7, 11, 13, 17, 19];
    let mut actual = [0; 8];
    let expected = [
        3 * 5 * 7 * 11 * 13 * 17 * 19,
        2 * 5 * 7 * 11 * 13 * 17 * 19,
        2 * 3 * 7 * 11 * 13 * 17 * 19,
        2 * 3 * 5 * 11 * 13 * 17 * 19,
        2 * 3 * 5 * 7 * 13 * 17 * 19,
        2 * 3 * 5 * 7 * 11 * 17 * 19,
        2 * 3 * 5 * 7 * 11 * 13 * 19,
        2 * 3 * 5 * 7 * 11 * 13 * 17
    ];
    product_except_one(&ring, (&data[..]).clone_els_by(|x| *x), &mut actual);
    assert_eq!(expected, actual);

    let data = [2, 3, 5, 7, 11, 13, 17];
    let mut actual = [0; 7];
    let expected = [
        3 * 5 * 7 * 11 * 13 * 17,
        2 * 5 * 7 * 11 * 13 * 17,
        2 * 3 * 7 * 11 * 13 * 17,
        2 * 3 * 5 * 11 * 13 * 17,
        2 * 3 * 5 * 7 * 13 * 17,
        2 * 3 * 5 * 7 * 11 * 17,
        2 * 3 * 5 * 7 * 11 * 13
    ];
    product_except_one(&ring, (&data[..]).clone_els_by(|x| *x), &mut actual);
    assert_eq!(expected, actual);

    let data = [2, 3, 5, 7, 11, 13];
    let mut actual = [0; 6];
    let expected = [
        3 * 5 * 7 * 11 * 13,
        2 * 5 * 7 * 11 * 13,
        2 * 3 * 7 * 11 * 13,
        2 * 3 * 5 * 11 * 13,
        2 * 3 * 5 * 7 * 13,
        2 * 3 * 5 * 7 * 11
    ];
    product_except_one(&ring, (&data[..]).clone_els_by(|x| *x), &mut actual);
    assert_eq!(expected, actual);
}

#[test]
fn test_invert_many() {
    let ring = Zn::new(17);
    let data = ring.elements().skip(1).collect::<Vec<_>>();
    let mut actual = (0..16).map(|_| ring.zero()).collect::<Vec<_>>();
    let expected = ring.elements().skip(1).map(|x| ring.invert(&x).unwrap()).collect::<Vec<_>>();
    invert_many(&ring, &data, &mut actual).unwrap();
    for i in 0..16 {
        assert_el_eq!(&ring, &expected[i], &actual[i]);
    }
}

#[test]
fn test_interpolate() {
    let ring = StaticRing::<i64>::RING;
    let poly_ring = DensePolyRing::new(ring, "X");
    let poly = poly_ring.from_terms([(3, 0), (1, 1), (-1, 3), (2, 4), (1, 5)].into_iter());
    let actual = interpolate(&poly_ring, (0..6).map_fn(|x| x as i64), (0..6).map_fn(|x| poly_ring.evaluate(&poly, &(x as i64), &ring.identity())), Global).unwrap();
    assert_el_eq!(&poly_ring, &poly, &actual);

    let ring = Zn::new(25);
    let poly_ring = DensePolyRing::new(ring, "X");
    let poly = interpolate(&poly_ring, (0..5).map_fn(|x| ring.int_hom().map(x as i32)), (0..5).map_fn(|x| if x == 3 { ring.int_hom().map(6) } else { ring.zero() }), Global).unwrap();
    for x in 0..5 {
        if x == 3 {
            assert_el_eq!(ring, ring.int_hom().map(6), poly_ring.evaluate(&poly, &ring.int_hom().map(x), &ring.identity()));
        } else {
            assert_el_eq!(ring, ring.zero(), poly_ring.evaluate(&poly, &ring.int_hom().map(x), &ring.identity()));
        }
    }
}

#[test]
fn test_interpolate_multivariate() {
    let ring = Zn::new(25);
    let poly_ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(ring, 2);

    let interpolation_points = (0..2).map_fn(|_| (0..5).map_fn(|x| ring.int_hom().map(x as i32)));
    let values = (0..25).map(|x| ring.int_hom().map(x & 1)).collect::<Vec<_>>();
    let poly = interpolate_multivariate(&poly_ring, &interpolation_points, values, Global).unwrap();

    for x in 0..5 {
        for y in 0..5 {
            let expected = (x * 5 + y) & 1;
            assert_el_eq!(ring, ring.int_hom().map(expected), poly_ring.evaluate(&poly, [ring.int_hom().map(x), ring.int_hom().map(y)].into_clone_ring_els(&ring), &ring.identity()));
        }
    }

    let poly_ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(ring, 3);

    let interpolation_points = (0..3).map_fn(|i| (0..(i + 2)).map_fn(|x| ring.int_hom().map(x as i32)));
    let values = (0..24).map(|x| ring.int_hom().map(x / 2)).collect::<Vec<_>>();
    let poly = interpolate_multivariate(&poly_ring, &interpolation_points, values, Global).unwrap();

    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                let expected = (x * 12 + y * 4 + z) / 2;
                assert_el_eq!(ring, ring.int_hom().map(expected), poly_ring.evaluate(&poly, [ring.int_hom().map(x), ring.int_hom().map(y), ring.int_hom().map(z)].into_clone_ring_els(&ring), &ring.identity()));
            }
        }
    }
}