use crate::algorithms::matmul::{STANDARD_MATMUL, MatmulAlgorithm};
use crate::field::*;
use crate::homomorphism::*;
use crate::integer::generic_impls::map_from_integer_ring;
use crate::integer::BigIntRing;
use crate::matrix::*;
use crate::matrix::transform::{DuplicateTransforms, OffsetTransformIndex, TransformCols, TransformRows, TransformTarget};
use crate::ordered::*;
use crate::ring::*;
use crate::rings::approx_real::{ApproxRealField, NotEnoughPrecision, SqrtRing};

///
/// Stores the Gram matrix, its (partial) floating-point Cholesky decomposition,
/// and an error bound on the latter. These values are jointly modified during the
/// LLL algorithm
/// 
struct GSOMatrix<'a, I, R, V1, V2, V3>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    /// The quadratic form to be reduced
    quadratic_form: SubmatrixMut<'a, V1, I::Element>,
    /// Cholesky-decomposition of the upper left part of `quadratic_form`;
    /// stored in the upper triangle
    cholesky: SubmatrixMut<'a, V2, R::Element>,
    /// Bound on the absolute error of the Cholesky decomposition,;
    /// in other words, `E[i, j]` is an upper bound on `|C[i, j] - C*[i, j]|`
    /// for `j >= i`, where `C*` is the real Cholesky decomposition; only
    /// the upper triangle is used
    error_bound: SubmatrixMut<'a, V3, R::Element>
}

impl<'a, I, R, V1, V2, V3> TransformTarget<I> for GSOMatrix<'a, I, R, V1, V2, V3>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    fn transform<S: Copy + RingStore<Type = I>>(&mut self, ring: S, i: usize, j: usize, transform: &[<I as RingBase>::Element; 4]) {
        TransformRows(self.quadratic_form.reborrow(), ring.get_ring()).transform(ring, i, j, transform);
        TransformCols(self.quadratic_form.reborrow(), ring.get_ring()).transform(ring, i, j, transform);
    }

    fn swap<S: Copy + RingStore<Type = I>>(&mut self, ring: S, i: usize, j: usize) {
        TransformRows(self.quadratic_form.reborrow(), ring.get_ring()).swap(ring, i, j);
        TransformCols(self.quadratic_form.reborrow(), ring.get_ring()).swap(ring, i, j);
    }

    fn subtract<S: Copy + RingStore<Type = I>>(&mut self, ring: S, src: usize, dst: usize, factor: &<I as RingBase>::Element) {
        TransformRows(self.quadratic_form.reborrow(), ring.get_ring()).subtract(ring, src, dst, factor);
        TransformCols(self.quadratic_form.reborrow(), ring.get_ring()).subtract(ring, src, dst, factor);
    }
}

///
/// Computes the floating-point division `num / den` and also computes information on
/// the error of the result.
/// 
fn divide_with_error<R>(RR: &R, num: &R::Element, num_err: &R::Element, den: &R::Element, den_err: &R::Element) -> (R::Element, R::Element)
    where R: ?Sized + ApproxRealField
{
    assert!(!RR.is_neg(den));
    let result = RR.div(num, den);
    let pivot_inv_error = if RR.is_geq(den_err, den) {
        RR.infinity()
    } else {
        RR.div(
            den_err, 
            &RR.mul_ref_fst(
                den,
                RR.sub_ref(den, den_err)
            )
        )
    };
    assert!(!RR.is_neg(&pivot_inv_error));
    let result_err = RR.sum([
        RR.mul_ref_fst(num_err, RR.div(&RR.one(), &den)),
        RR.mul(RR.abs(RR.clone_el(num)), pivot_inv_error)
    ]);
    return (result, result_err);
}

///
/// Computes the column `C[..i, i]` of the Cholesky decomposition `C`
/// of `A` and the corresponding error bounds, assuming that `Q[..i, ..i]`
/// is already computed.
/// 
fn compute_cholesky_column_without_pivot<I, R, H, V1, V2, V3>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: &H
)
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    let RR = h.codomain();
    let eps = RR.get_ring().epsilon();
    for k in 0..i {
        let sum = RR.sub(
            h.map_ref(gso.quadratic_form.at(k, i)),
            RR.sum(
                (0..k).map(|l| RR.mul_ref(gso.cholesky.at(l, i), gso.cholesky.at(l, k)))
            )
        );
        let sum_error = RR.sum([
            RR.mul_ref_fst(eps, RR.abs(h.map_ref(gso.quadratic_form.at(k, i)))),
            RR.sum(
                (0..k).map(|l| RR.add(
                    RR.mul_ref_snd(RR.abs(RR.clone_el(gso.cholesky.at(l, i))), &gso.error_bound.at(l, k)),
                    RR.mul_ref_snd(RR.abs(RR.clone_el(gso.cholesky.at(l, k))), &gso.error_bound.at(l, i))
                ))
            )
        ]);
        assert!(!RR.is_neg(&sum_error));
        assert!(!RR.is_neg(gso.cholesky.at(k, k)));

        let (result, result_error) = divide_with_error(RR.get_ring(), &sum, &sum_error, gso.cholesky.at(k, k), gso.error_bound.at(k, k));
        *gso.cholesky.at_mut(k, i) = result;
        *gso.error_bound.at_mut(k, i) = result_error;
    }
}

///
/// Computes the entry `C[i, i]` of the Cholesky decomposition `C`
/// of `A` and the corresponding error bounds, assuming that
/// `Q[..i, ..=i]` is already computed.
/// 
fn compute_cholesky_pivot<I, R, H, V1, V2, V3>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: &H
)
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    let RR = h.codomain();
    let eps = RR.get_ring().epsilon();
    let sum = RR.sub(
        h.map_ref(gso.quadratic_form.at(i, i)),
        RR.sum(
            (0..i).map(|l| RR.pow(RR.clone_el(gso.cholesky.at(l, i)), 2))
        )
    );
    let sum_error = RR.sum([
        RR.mul_ref_fst(eps, RR.abs(h.map_ref(gso.quadratic_form.at(i, i)))),
        RR.int_hom().mul_map(RR.sum(
            (0..i).map(|l| RR.mul_ref_snd(RR.abs(RR.clone_el(gso.cholesky.at(l, i))), &gso.error_bound.at(l, i)))
        ), 2)
    ]);
    assert!(!RR.is_neg(&sum_error));

    let upper = RR.add_ref(&sum, &sum_error);
    assert!(!RR.is_neg(&upper), "Quadratic form is not positive semidefinite");
    let (value, error) = if !RR.is_pos(&RR.sub_ref(&sum, &sum_error)) {
        // we assume the matrix is positive semi-definite
        (RR.div(&upper, &RR.int_hom().map(2)), RR.div(&upper, &RR.int_hom().map(2)))
    } else {
        (sum, sum_error)
    };

    *gso.cholesky.at_mut(i, i) = RR.get_ring().sqrt(value);
    *gso.error_bound.at_mut(i, i) = RR.get_ring().sqrt(error);
}

///
/// Size-reduces the `i`-th basis vector, implicitly defined by
/// the quadratic form `A`. 
/// 
/// This will fill in the entries `C[..i, i]` in the partial
/// Cholesky decomposition `C` and the corresponding entries in
/// the error bound matrix.
/// 
fn size_reduce<I, R, H, V1, V2, V3, T>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: &H,
    eta: &R::Element,
    mut transform: T
) -> Result<(), NotEnoughPrecision>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>,
        T: TransformTarget<I>
{
    debug_assert!(i < gso.quadratic_form.row_count());
    let RR = h.codomain();

    // the largest value of `abs(mu) + error` we encountered in the last iteration;
    // this should decrease each iteration, otherwise we are not making progress
    let mut largest_max_mu = RR.get_ring().infinity();
    loop {
        compute_cholesky_column_without_pivot(gso, i, &h);

        // first, check precision and progress
        let new_largest_max_mu = (0..i).map(|k| RR.div(
            &RR.add_ref_snd(
                RR.abs(RR.clone_el(gso.cholesky.at(k, i))), 
                gso.error_bound.at(k, i)
            ),
            gso.cholesky.at(k, k)
        )).fold(RR.zero(), |l, r| if RR.is_leq(&l, &r) { r } else { l });
        if RR.is_leq(&new_largest_max_mu, &eta) {
            return Ok(());
        } else if RR.is_geq(&new_largest_max_mu, &largest_max_mu) {
            return Err(NotEnoughPrecision);
        } else {
            largest_max_mu = new_largest_max_mu;
        }

        for k in (0..i).rev() {
            let min_entry = RR.sub(
                RR.abs(RR.clone_el(gso.cholesky.at(k, i))), 
                // add epsilon here, to avoid `NotEnoughPrecision` error in the case that `mu = +/- 1/2` and `error = 0`
                RR.add_ref(
                    gso.error_bound.at(k, i),
                    RR.get_ring().epsilon()
                )
            );
            let min_entry = if RR.is_neg(&min_entry) {
                RR.zero()
            } else if RR.is_neg(gso.cholesky.at(k, i)) {
                RR.negate(min_entry)
            } else {
                min_entry
            };
            let min_mu = RR.div(&min_entry, gso.cholesky.at(k, k));

            let factor = RR.get_ring().round_to_integer(BigIntRing::RING, min_mu).ok_or(NotEnoughPrecision)?;
            if !BigIntRing::RING.is_zero(&factor) {
                let (mut target, rest) = gso.cholesky.reborrow().split_cols(i..(i + 1), 0..i);
                let factor = map_from_integer_ring(&BigIntRing::RING, h.domain(), factor);
                for l in 0..k {
                    RR.sub_assign(target.at_mut(l, 0), h.mul_ref_map(rest.at(l, k), &factor));
                }
                gso.subtract(h.domain(), k, i, &factor);
                transform.subtract(h.domain(), k, i, &factor);
            }
        }
    }
}

enum LovaszCondition {
    Satisfied, Swap
}

///
/// Checks whether the Lovasz-condition is satisfied for the `(i - 1)`-th
/// and `i`-th basis vectors, implicitly defined by the quadratic form `A`.
/// 
/// This will fill in the entry `C[i, i]` in the partial Cholesky decomposition
/// and the corresponding entry in the error bound matrix.
/// 
/// Note that here we handle numerical issues differently than in [`size_reduce()`].
/// In particular, even if the precision of the Cholesky decomposition is not
/// enough to determine for sure whether the Lovasz condition is satisfied or
/// not, we just work with the best guess. The reason is that, in the LLL book,
/// Damien Stehlé says, in practice, this almost never causes problems.
/// 
fn check_lovasz_condition<I, R, H, V1, V2, V3>(
    gso: &mut GSOMatrix<I, R, V1, V2, V3>,
    i: usize,
    h: &H,
    delta: &R::Element
) -> LovaszCondition
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    let RR = h.codomain();
    compute_cholesky_pivot(gso, i, h);

    let prev_norm_squared = RR.pow(RR.clone_el(gso.cholesky.at(i - 1, i - 1)), 2);
    let current_norm_squared = RR.pow(RR.clone_el(gso.cholesky.at(i, i)), 2);
    let mu = RR.div(gso.cholesky.at(i - 1, i), gso.cholesky.at(i - 1, i - 1));
    let factor = RR.sub_ref_fst(delta, RR.pow(mu, 2));
    assert!(!RR.is_neg(&factor));
    if RR.is_lt(
        &current_norm_squared,
        &RR.mul(factor, prev_norm_squared),
    ) {
        LovaszCondition::Swap
    } else {
        LovaszCondition::Satisfied
    }
}

///
/// Shrinks all matrices associated to the `gso` as long as the top left
/// element of the quadratic form is zero. Increments `zero_vector_count` for
/// each dimension removed this way.
/// 
/// Afterwards, the top left element of the Cholesky decomposition is properly
/// initialized, by taking the root of the 
/// 
fn remove_zero_vectors<'a, I, R, H, V1, V2, V3>(
    mut gso: GSOMatrix<'a, I, R, V1, V2, V3>,
    h: &H,
    zero_vector_count: &mut usize
) -> GSOMatrix<'a, I, R, V1, V2, V3>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    while gso.quadratic_form.row_count() > 1 && h.domain().is_zero(gso.quadratic_form.at(0, 0)) {
        let n = gso.quadratic_form.row_count();
        assert!(
            (1..n).all(|j| h.domain().is_zero(gso.quadratic_form.at(j, 0))),
            "Quadratic form is not positive semidefinite"
        );
        *zero_vector_count += 1;
        gso.quadratic_form = gso.quadratic_form.submatrix(1..n, 1..n);
        gso.error_bound = gso.error_bound.submatrix(1..n, 1..n);
        gso.cholesky = gso.cholesky.submatrix(1..n, 1..n);
    }
    let RR = h.codomain();
    assert!(
        RR.is_pos(&h.map_ref(gso.quadratic_form.at(0, 0))),
        "Quadratic form is not positive semidefinite"
    );
    *gso.cholesky.at_mut(0, 0) = RR.get_ring().sqrt(h.map_ref(gso.quadratic_form.at(0, 0)));
    *gso.error_bound.at_mut(0, 0) = RR.mul_ref(gso.cholesky.at(0, 0), RR.get_ring().epsilon());
    return gso;
}

///
/// Computes an `(delta, eta)`-LLL-reduced form of the given positive semidefinite
/// quadratic form, using a custom variant of the L^2 algorithm.
/// 
/// Note that the algorithm may return [`NotEnoughPrecision`], if the given precision
/// is not sufficient to prove that the result is `(delta, eta)`-LLL-reduced. However,
/// it will usually be quite reduced already, and may even be `(delta, eta)`-LLL-reduced.
/// 
/// The given quadratic form must be positive semidefinite.
/// 
/// More concretely, this function transforms the quadratic form into another quadratic
/// form `Q` by unimodular operations that are simultaneously applied to rows and columns,
/// such that
///  - (size-reduced) `|ei^T Q ej*| < eta (ej*^T Q ej*)` whenever `i > j`
///  - (Lovasz-condition) `(ek*^T Q ek*) >= delta (e(k-1)*^T Q e(k - 1)*) - (ek^T Q e(k-1)*) / (e(k-1)*^T Q e(k - 1)*)`
/// 
/// Here the `ei*` refer to the Gram-Schmidt orthogonalization of the unit vectors `ei`
/// w.r.t. the inner product defined by `Q`.
/// 
/// # Algorithm and numerical stability
/// 
/// The used algorithm is a custom variant of the L^2 algorithm by
/// Nguyen and Stehlé. More concretely, it internally computes with
/// floating point numbers, with the following notable features:
///  - The GSO coefficients are always recomputed after a column swap.
///    This is necessary, since continuously adjusting them fundamentally
///    has very bad numerical properties. For this purpose, a variant
///    of the Cholesky decomposition that keeps track of the current error
///    is used.
///  - It may happen that the error after computing the Cholesky decomposition
///    is too large to compute a `(delta, eta)`-LLL reduced basis. In that
///    case, [`NotEnoughPrecision`] is returned. If you are using arbitrary
///    precision floating point numbers, it can make sense to increase the precision
///    and try again.
///
#[stability::unstable(feature = "enable")]
pub fn lll_quadratic_form<I, R, H, V1, T>(
    quadratic_form: SubmatrixMut<V1, I::Element>,
    h: H,
    delta: &R::Element,
    eta: &R::Element,
    mut transform: T
) -> Result<(), NotEnoughPrecision>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        T: TransformTarget<I>
{
    assert!(!h.domain().get_ring().is_approximate());
    let RR = h.codomain();
    assert_eq!(quadratic_form.row_count(), quadratic_form.col_count());

    let half = RR.div(&RR.one(), &RR.int_hom().map(2));
    assert!(RR.is_gt(eta, &half));
    assert!(RR.is_lt(delta, &RR.one()));
    assert!(RR.is_gt(delta, &RR.mul_ref(eta, eta)));
    let strict_delta = RR.mul_ref_snd(RR.add_ref_fst(delta, RR.one()), &half);

    let mut C = OwnedMatrix::zero(quadratic_form.row_count(), quadratic_form.row_count(), RR);
    let mut E = OwnedMatrix::zero(quadratic_form.row_count(), quadratic_form.row_count(), RR);
    let mut gso = GSOMatrix {
        cholesky: C.data_mut(),
        error_bound: E.data_mut(),
        quadratic_form: quadratic_form
    };

    let mut i = 1;
    let mut zero_vector_count = 0;
    let mut remaining_swaps = gso.quadratic_form.row_count() * gso.quadratic_form.row_count() * 1000;
    gso = remove_zero_vectors(gso, &h, &mut zero_vector_count);
    while i < gso.quadratic_form.row_count() {
        assert!(i > 0);
        size_reduce(&mut gso, i, &h, eta, OffsetTransformIndex::new(&mut transform, zero_vector_count))?;
        match check_lovasz_condition(&mut gso, i, &h, &strict_delta) {
            LovaszCondition::Swap if remaining_swaps == 0 => {
                return Err(NotEnoughPrecision);
            }
            LovaszCondition::Swap => {
                remaining_swaps -= 1;
                gso.swap(h.domain(), i - 1, i);
                OffsetTransformIndex::new(&mut transform, zero_vector_count).swap(h.domain(), i - 1, i);
                i -= 1;
                if i == 0 {
                    gso = remove_zero_vectors(gso, &h, &mut zero_vector_count);
                    i = 1;
                }
            },
            LovaszCondition::Satisfied => {
                i += 1;
            }
        }
    }

    return Ok(());
}

///
/// LLL-reduces the given lattice basis, defined by the columns of the given
/// matrix, using a custom variant of the L^2 algorithm.
/// 
/// Note that the algorithm may return [`NotEnoughPrecision`], if it cannot
/// prove that the result is `(delta, eta)`-LLL-reduced. However, it will usually
/// be quite reduced already, and may even be `(delta, eta)`-LLL-reduced.
/// 
/// For more details, see [`lll_quadratic_form()`].
///
#[stability::unstable(feature = "enable")]
pub fn lll<I, R, H, V1, T>(
    basis: SubmatrixMut<V1, I::Element>,
    h: H,
    delta: &R::Element,
    eta: &R::Element,
    transform: T
) -> Result<(), NotEnoughPrecision>
    where I: ?Sized + RingBase,
        R: ?Sized + ApproxRealField + SqrtRing,
        H: Homomorphism<I, R>,
        V1: AsPointerToSlice<I::Element>,
        T: TransformTarget<I>
{
    let n = basis.col_count();
    let mut quadratic_form = OwnedMatrix::zero(n, n, h.domain());
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(basis.as_const()).transpose(),
        TransposableSubmatrix::from(basis.as_const()),
        TransposableSubmatrixMut::from(quadratic_form.data_mut()),
        h.domain()
    );

    lll_quadratic_form(
        quadratic_form.data_mut(),
        &h,
        delta,
        eta,
        DuplicateTransforms::new(TransformCols(basis, h.domain().get_ring()), transform)
    )?;

    return Ok(());
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::assert_matrix_eq;
#[cfg(test)]
use crate::algorithms::lll::{assert_lattice_isomorphic, norm_squared};
#[cfg(test)]
use crate::rings::approx_real::float::*;

#[test]
fn test_compute_cholesky_column_without_pivot() {
    let RR = Real64::RING;
    let mut quadratic_form = [
        DerefArray::from([4., 5.]),
        DerefArray::from([5., 12.])
    ];
    let mut cholesky = [
        DerefArray::from([2., 0.]),
        DerefArray::from([0., 0.])
    ];
    let mut errors = [
        DerefArray::from([0., 0.]),
        DerefArray::from([0., 0.])
    ];
    let mut gso = GSOMatrix {
        quadratic_form: SubmatrixMut::from_2d(&mut quadratic_form),
        cholesky: SubmatrixMut::from_2d(&mut cholesky),
        error_bound: SubmatrixMut::from_2d(&mut errors)
    };
    compute_cholesky_column_without_pivot(&mut gso, 1, &RR.identity());

    assert!(errors[0][1] <= 5. * f64::EPSILON);
    assert!((cholesky[0][1] - 2.5).abs() <= errors[0][1]);
    
    let mut quadratic_form = [
        DerefArray::from([16., 6., 4.]),
        DerefArray::from([6., 11., 2.5]),
        DerefArray::from([4., 2.5, 2.]),
    ];
    let mut cholesky = [
        DerefArray::from([4., 1.5, 0.]),
        DerefArray::from([0., 2.9580398915498080212836641, 0.]),
        DerefArray::from([0., 0., 0.]),
    ];
    let mut errors = [
        DerefArray::from([0., 0., 0.]),
        DerefArray::from([0., f64::EPSILON, 0.]),
        DerefArray::from([0., 0., 0.]),
    ];
    let mut gso = GSOMatrix {
        quadratic_form: SubmatrixMut::from_2d(&mut quadratic_form),
        cholesky: SubmatrixMut::from_2d(&mut cholesky),
        error_bound: SubmatrixMut::from_2d(&mut errors)
    };
    compute_cholesky_column_without_pivot(&mut gso, 2, &RR.identity());

    assert!(errors[0][2] <= 5. * f64::EPSILON);
    assert!((cholesky[0][2] - 1.).abs() <= errors[0][2]);
    assert!(errors[1][2] <= 10. * f64::EPSILON);
    assert!((cholesky[1][2] - 0.33806170189140663100384733).abs() <= errors[1][2]);
}

#[test]
fn test_lll_float_2d() {
    let ZZ = StaticRing::<i64>::RING;
    let RR = Real64::RING;
    let original = [
        DerefArray::from([146, 265]),
        DerefArray::from([265, 481])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transform_matrix = OwnedMatrix::identity(2, 2, ZZ);
    lll_quadratic_form(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.9, &0.55, TransformCols(transform_matrix.data_mut(), ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, OwnedMatrix::identity(2, 2, ZZ), reduced_matrix);

    let mut tmp = original;
    let mut tmp_matrix = SubmatrixMut::from_2d(&mut tmp);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(transform_matrix.data()).transpose(),
        TransposableSubmatrix::from(Submatrix::from_2d(&original)),
        TransposableSubmatrixMut::from(tmp_matrix.reborrow()),
        ZZ
    );
    let mut check = original;
    let mut check_matrix = SubmatrixMut::from_2d(&mut check);
    STANDARD_MATMUL.matmul(
        TransposableSubmatrix::from(tmp_matrix.as_const()),
        TransposableSubmatrix::from(transform_matrix.data()),
        TransposableSubmatrixMut::from(check_matrix.reborrow()),
        ZZ
    );
    assert_matrix_eq!(ZZ, reduced_matrix, check_matrix);
    
    let original = [
        DerefArray::from([10, 8]),
        DerefArray::from([27, 22])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.9, &0.55, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert_eq!(4, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
}

#[test]
fn test_lll_float_3d() {
    let ZZ = StaticRing::<i64>::RING;
    let RR = Real64::RING;
    // in this case, the shortest vector is shorter than half the second successive minimum,
    // so LLL will find it (for delta = 0.9 > 0.75)
    let original = [
        DerefArray::from([72, 0, 0]),
        DerefArray::from([0,  9, 0]),
        DerefArray::from([8432, 7344, 16864])
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::<DerefArray<_, 3>, _>::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert_eq!(144 * 144, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_eq!(72 * 72 + 279 * 279, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_eq!(72 * 72 * 2 + 272 * 272, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
}

#[test]
fn test_lll_precision() {
    let ZZ = StaticRing::<i128>::RING;
    let RR = Real64::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([65208, 1, 0, 0, 0]),
        DerefArray::from([0, 65208, 1, 0, 0]),
        DerefArray::from([0, 0, 65208, 1, 0]),
        DerefArray::from([0, 0, 0, 65208, 999769]),
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 200);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 400);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 500);
    
    let ZZ = StaticRing::<i64>::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([-3085729, 1, 0, 0, 0]),
        DerefArray::from([0, -3085729, 1, 0, 0]),
        DerefArray::from([0, 0, -3085729, 1, 0]),
        DerefArray::from([0, 0, 0, -3085729, 23068673]),
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);
    
    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 500);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 900);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 1200);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 1300);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 2000);

    
    let ZZ = StaticRing::<i128>::RING;
    let original = [
        DerefArray::from([1, 0, 0, 0, 0]),
        DerefArray::from([207432708, 1, 0, 0, 0]),
        DerefArray::from([0, 207432708, 1, 0, 0]),
        DerefArray::from([0, 0, 207432708, 1, 0]),
        DerefArray::from([0, 0, 0, 207432708, 447741953]),
    ];
    
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);
    
    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)) < 1800);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)) < 1800);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)) < 4600);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)) < 4600);
    assert!(norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)) < 5600);
}

#[test]
fn test_lll_generating_set() {
    let RR = Real64::RING;
    let ZZ = StaticRing::<i128>::RING;
    let original = [
        DerefArray::from([ -6,  -1,  6, 116, -2]),
        DerefArray::from([-14, -12,  8, 232, -2]),
        DerefArray::from([-10,   2, 12,   0,  2])
    ];

    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_el_eq!(ZZ, 5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
    assert_el_eq!(ZZ, 5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)));
    assert_el_eq!(ZZ, 12, norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)));

    let original = [
        DerefArray::from([-4,   8, -54,  -1,   42,   15,   -23,   -259]),
        DerefArray::from([-3,  10, -36,  18,  -48, -473, -1200,  -6493]),
        DerefArray::from([ 5, -13,  62, -15,   17,  398,  1043,   5721]),
        DerefArray::from([-8,  10, -68,  18,  -18, -434, -1118,  -6126]),
        DerefArray::from([11,  -5,  90,  26, -215, -910, -2227, -11637])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(5)));
    assert_el_eq!(ZZ, 5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(6)));
    assert_el_eq!(ZZ, 40, norm_squared(ZZ, &reduced_matrix.as_const().col_at(7)));

    let original = [
        DerefArray::from([  -60725263117,   -448122081513,  -218368759847,   2100701846793,   216156377534,   -3137996709827,   14835704835919,    67504381450573]),
        DerefArray::from([-1310716961940,  -9682451257943, -4729935920987,  45413204073392,  4667627712725,  -67791459966817,  320528485599331,  1458334256347773]),
        DerefArray::from([ 1159398893231,   8564380015666,  4183444050825, -40168532351902, -4128711773154,   59963582382418, -283516375460965, -1289940218804617]),
        DerefArray::from([-1236320093452,  -9132639612642, -4461079566742,  42833893239948,  4402644221490,  -63942205977848,  302328006373120,  1375528654002990]),
        DerefArray::from([-2344979577397, -17323890604545, -8464219959177,  81256383857324,  8351008895542, -121291584595649,  573488494361461,  2609233431319737])
    ];
    let mut reduced = original;
    let mut reduced_matrix = SubmatrixMut::from_2d(&mut reduced);
    let mut transformed = original;
    let transformed_matrix = SubmatrixMut::from_2d(&mut transformed);
    lll(reduced_matrix.reborrow(), RR.can_hom(&ZZ).unwrap(), &0.999, &0.51, TransformCols(transformed_matrix, ZZ.get_ring())).unwrap();
    assert_matrix_eq!(ZZ, reduced_matrix, transformed);

    assert_lattice_isomorphic(ZZ, BigIntRing::RING, Submatrix::from_2d(&original), reduced_matrix.as_const());
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(0)));
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(1)));
    assert_el_eq!(ZZ, 0, norm_squared(ZZ, &reduced_matrix.as_const().col_at(2)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(3)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(4)));
    assert_el_eq!(ZZ, 1, norm_squared(ZZ, &reduced_matrix.as_const().col_at(5)));
    assert_el_eq!(ZZ, 5, norm_squared(ZZ, &reduced_matrix.as_const().col_at(6)));
    assert_el_eq!(ZZ, 40, norm_squared(ZZ, &reduced_matrix.as_const().col_at(7)));
}
