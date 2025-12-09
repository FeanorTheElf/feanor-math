use crate::algorithms::matmul::ComputeInnerProduct;
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore, Domain};
use crate::integer::IntegerRingStore;
use crate::iters::multi_cartesian_product;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::rings::poly::*;
use crate::seq::*;
use crate::homomorphism::Homomorphism;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::algorithms::convolution::STANDARD_CONVOLUTION;

use tracing::instrument;

use std::alloc::Allocator;
use std::cmp::min;
use std::ops::Range;

///
/// Computes `out[i] = prod_(j != i) values[j]`.
/// 
/// This algorithm recursively halfes the input, and thus it implicitly pads the input to the
/// next power-of-two. The time complexity is then `O(n sum_(0 <= i <= log n) T(2^i))`
/// where `T(d)` is the complexity of multiplying two products of `d` input elements. If the cost
/// of multiplication is constant, this becomes `O(n log n T)`. In the other common case, where
/// the input elements are degree-1 polynomials, this becomes `O(n sum_(0 <= i <= log n) 2^(i c)) = O(n^(c + 1))`
/// where `c` is the multiplication exponent, e.g. `c = 1.58...` for Karatsuba multiplication.
/// Unfortunately, this means that a single complete multiplication together with `n` polynomial
/// divisions by the monic linear factors is faster.
/// 
/// # Example
/// ```rust
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
#[instrument(skip_all, level = "trace")]
pub fn product_except_one<V, R>(ring: R, values: V, out: &mut [El<R>])
    where R: RingStore,
        V: VectorFn<El<R>>
{
    assert_eq!(values.len(), out.len());
    let n = values.len();
    let log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&n.try_into().unwrap()).unwrap();
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
/// The complexity is `O(n T(n))` where `T(n)` is the cost of multiplying two degree-`n` polynomials.
/// 
/// # Example
/// ```rust
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
/// ```rust
/// # #![feature(allocator_api)]
/// # use std::alloc::Global;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::StaticRing;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::seq::*;
/// # use feanor_math::algorithms::interpolate::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// let ZnX = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let actual = interpolate(&ZnX, [-2, 0, 2].copy_els(), [1, 0, 1].copy_els(), Global);
/// assert!(actual.is_err());
/// ```
/// 
#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn interpolate<P, V1, V2, A: Allocator>(poly_ring: P, x: V1, y: V2, allocator: A) -> Result<El<P>, InterpolationError>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain,
        V1: VectorFn<El<<P::Type as RingExtension>::BaseRing>>,
        V2: VectorFn<El<<P::Type as RingExtension>::BaseRing>>
{
    assert_eq!(x.len(), y.len());
    let base_ring = poly_ring.base_ring();
    let null_poly = poly_ring.prod(x.iter().map(|x| poly_ring.from_terms([(base_ring.one(), 1), (base_ring.negate(x), 0)])));
    let mut nums = Vec::with_capacity_in(x.len(), &allocator);
    let div_linear = |poly: &El<P>, a: &El<<P::Type as RingExtension>::BaseRing>| if let Some(d) = poly_ring.degree(poly) {
        poly_ring.from_terms((0..d).rev().scan(base_ring.zero(), |current, i| {
            base_ring.add_assign_ref(current, poly_ring.coefficient_at(poly, i + 1));
            let result = base_ring.clone_el(current);
            base_ring.mul_assign_ref(current, a);
            return Some((result, i));
        }))
    } else { poly_ring.zero() };
    nums.extend(x.iter().map(|x| div_linear(&null_poly, &x)));
    let nums = nums;

    let mut denoms = Vec::with_capacity_in(x.len(), &allocator);
    denoms.extend((0..x.len()).map(|i| poly_ring.evaluate(&nums[i], &x.at(i), &base_ring.identity())));
    let denoms = denoms;

    let denoms_inv = denoms.iter().map(|den| base_ring.invert(den).ok_or(())).collect::<Result<Vec<_>, _>>();
    if let Ok(denoms_inv) = denoms_inv {
        return Ok(poly_ring.from_terms((0..x.len()).map(|i| (
            <_ as ComputeInnerProduct>::inner_product_ref_fst(base_ring.get_ring(), (0..x.len()).map(|j| (poly_ring.coefficient_at(&nums[j], i), base_ring.mul_ref_snd(y.at(j), &denoms_inv[j])))),
            i
        ))));
    } else {
        let mut factors = Vec::with_capacity_in(x.len(), &allocator);
        factors.resize_with(x.len(), || base_ring.zero());
        product_except_one(base_ring, (&denoms[..]).into_clone_ring_els(base_ring), &mut factors);
        let denominator = base_ring.mul_ref(&factors[0], &denoms[0]);
        for (factor, y) in factors.iter_mut().zip(y.iter()) {
            base_ring.mul_assign(factor, y);
        }

        let mut scaled_result = poly_ring.zero();
        for (num, c) in nums.into_iter().zip(factors.into_iter()) {
            scaled_result = poly_ring.inclusion().fma_map(&num, &c, scaled_result);
        }
        return poly_ring.try_from_terms(poly_ring.terms(&scaled_result).map(|(c, i)| base_ring.checked_div(&c, &denominator).map(|c| (c, i)).ok_or(InterpolationError::NotInvertible)));
    }
}

#[stability::unstable(feature = "enable")]
pub fn interpolate_multivariate<P, V1, V2, A, A2>(poly_ring: P, interpolation_points: V1, mut values: Vec<El<<P::Type as RingExtension>::BaseRing>, A2>, allocator: A) -> Result<El<P>, InterpolationError>
    where P: RingStore,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing + Domain,
        V1: VectorFn<V2>,
        V2: VectorFn<El<<P::Type as RingExtension>::BaseRing>>,
        A: Allocator + Send + Sync,
        A2: Allocator
{
    let dim_prod = |range: Range<usize>| <_ as RingStore>::prod(&StaticRing::<i64>::RING, range.map(|i| interpolation_points.at(i).len().try_into().unwrap())) as usize;
    assert_eq!(interpolation_points.len(), poly_ring.indeterminate_count());
    let n = poly_ring.indeterminate_count();
    assert_eq!(values.len(), dim_prod(0..n));

    let uni_poly_ring = DensePolyRing::new_with_convolution(poly_ring.base_ring(), "X", &allocator, STANDARD_CONVOLUTION);

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
use crate::rings::zn::zn_64b::Zn64B;
#[cfg(test)]
use crate::rings::zn::ZnRingStore;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use multivariate_impl::MultivariatePolyRingImpl;
#[cfg(test)]
use crate::rings::fraction::FractionFieldStore;
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_product_except_one() {
    LogAlgorithmSubscriber::init_test();
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
fn test_interpolate() {
    LogAlgorithmSubscriber::init_test();
    let ring = StaticRing::<i64>::RING;
    let poly_ring = DensePolyRing::new(ring, "X");
    let poly = poly_ring.from_terms([(3, 0), (1, 1), (-1, 3), (2, 4), (1, 5)].into_iter());
    let x = (0..6).map_fn(|i| i.try_into().unwrap());
    let actual = interpolate(&poly_ring, x.clone(), x.map_fn(|x| poly_ring.evaluate(&poly, &x, &ring.identity())), Global).unwrap();
    assert_el_eq!(&poly_ring, &poly, &actual);

    let ring = RationalField::new(StaticRing::<i64>::RING);
    let poly_ring = DensePolyRing::new(ring, "X");
    let x = (0..4).map_fn(|i| ring.from_fraction(i.try_into().unwrap(), 1));
    let y = (0..4).map_fn(|_| ring.zero());
    let actual = interpolate(&poly_ring, x.clone(), y, Global).unwrap();
    for i in 0..4 {
        assert_el_eq!(ring, ring.zero(), poly_ring.evaluate(&actual, &ring.from_fraction(i, 1), ring.identity()));
    }

    let ring = Zn64B::new(29).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(ring, "X");
    let x = (0..5).map_fn(|i| ring.int_hom().map(i as i32));
    let y = (0..5).map_fn(|i| if i == 3 { ring.int_hom().map(6) } else { ring.zero() });
    let poly = interpolate(&poly_ring, x.clone(), y.clone(), Global).unwrap();
    for i in 0..5 {
        assert_el_eq!(ring, y.at(i), poly_ring.evaluate(&poly, &x.at(i), &ring.identity()));
    }
}

#[test]
fn test_interpolate_multivariate() {
    LogAlgorithmSubscriber::init_test();
    let ring = Zn64B::new(29).as_field().ok().unwrap();
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

#[test]
#[ignore]
fn large_polynomial_interpolation() {
    LogAlgorithmSubscriber::init_test();
    let field = Zn64B::new(65537).as_field().ok().unwrap();
    let poly_ring = DensePolyRing::new(field, "X");
    let hom = poly_ring.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    let actual = interpolate(&poly_ring, (0..65536).map_fn(|x| hom.map(x as i64)), (0..65536).map_fn(|x| hom.map(x as i64)), Global).unwrap();
    assert_el_eq!(&poly_ring, poly_ring.indeterminate(), actual);
}