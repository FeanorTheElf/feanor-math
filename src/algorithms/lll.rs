use crate::field::*;
use crate::homomorphism::Homomorphism;
use crate::integer::IntegerRing;
use crate::matrix::submatrix::{AsPointerToSlice, Submatrix, SubmatrixMut};
use crate::matrix::TransformTarget;
use crate::ordered::{OrderedRing, OrderedRingStore};
use crate::ring::*;

pub trait LLLRealField: OrderedRing + Field {

    type IntRing: IntegerRing;

    fn integer_ring(&self) -> &Self::IntRing;
    fn from_integer(&self, x: <Self::IntRing as RingBase>::Element) -> Self::Element;
    fn round_to_integer(&self, x: &Self::Element) -> <Self::IntRing as RingBase>::Element;
}

fn size_reduce<R, V, T>(ring: R, mut target: SubmatrixMut<V, El<R>>, target_j: usize, matrix: Submatrix<V, El<R>>, col_ops: &mut T)
    where R: RingStore,
        R::Type: LLLRealField,
        V: AsPointerToSlice<El<R>>,
        T: TransformTarget<<R::Type as LLLRealField>::IntRing>
{
    for j in (0..matrix.col_count()).rev() {
        let factor = ring.get_ring().round_to_integer(target.as_const().at(j, 0));
        col_ops.subtract(ring.get_ring().integer_ring(), target_j, j, &factor);
        let factor = ring.get_ring().from_integer(factor);
        ring.sub_assign_ref(target.at(j, 0), &factor);
        for k in 0..j {
            ring.sub_assign(target.at(k, 0), ring.mul_ref(matrix.at(k, j), &factor));
        }
    }
}

///
/// Updates the "gso"-matrix resulting from swapping cols i and i + 1.
/// 
/// This uses explicit formulas, in particular if `b_i` and `b_(i + 1)`
/// represent the vectors, `b_i*` and `b_(i + 1)*` represent their GS-orthogonalizations
/// and the corresponding `b'` values are the ones after swapping, then we find
/// then this gives
/// ```text
/// b'_i = b_(i + 1)
/// b'_(i + 1) = b_i
/// 
/// b'_i* = b_(i + 1)* + mu b_i*
/// b'_(i + 1) = (1 - gamma^2 mu^2) b_i* - mu * gamma^2 b_(i + 1)*
///     where gamma^2 = |b_i*|^2 / |b'_i*|^2
/// mu' = gamma^2 mu
/// ```
/// 
fn swap_gso_cols<R, V>(ring: R, mut gso: SubmatrixMut<V, El<R>>, i: usize, j: usize)
    where R: RingStore,
        R::Type: LLLRealField,
        V: AsPointerToSlice<El<R>>
{
    assert!(j == i + 1);

    // swap the columns
    let (mut col_i, mut col_i1) = gso.reborrow().restrict_cols(i..(i + 2)).split_cols(0..1, 1..2);
    for k in 0..i {
        std::mem::swap(col_i.at(k, 0), col_i1.at(k, 0));
    }

    // re-orthogonalize the triangle `i..(i + 2) x i..(i + 2)`

    // | b_i* |^2
    let bi_star_norm_sqr = ring.clone_el(gso.at(i, i));
    // | b_(i + 1)* |^2
    let bi1_star_norm_sqr = ring.clone_el(gso.at(i + 1, i + 1));
    // mu_(i + 1)i = <b_(i + 1), bi*> / <bi*, bi*>
    let mu = ring.clone_el(gso.at(i, i + 1));
    let mu_sqr = ring.pow(ring.clone_el(&mu), 2);

    let new_bi_star_norm_sqr = ring.add_ref_fst(&bi1_star_norm_sqr, ring.mul_ref(&mu_sqr, &bi_star_norm_sqr));
    // `|b_i*|^2 / |bnew_i*|^2`
    let gamma_sqr = ring.div(&bi_star_norm_sqr, &new_bi_star_norm_sqr);
    let new_bi1_star_norm_sqr = ring.mul_ref(&gamma_sqr, &bi1_star_norm_sqr);
    let new_mu = ring.mul_ref(&gamma_sqr, &mu);

    // we now update the `mu_ki` resp. `mu_k(i + 1)` by a linear transform
    let lin_transform_muki = [ring.mul_ref(&gamma_sqr, &mu), ring.sub(ring.one(), ring.mul_ref(&gamma_sqr, &mu_sqr))];
    let (row_i, row_i1) = gso.reborrow().restrict_rows(i..(i + 2)).split_rows(0..1, 1..2);
    for k in (i + 2)..gso.col_count() {
        std::mem::swap(row.at(0, k), row_i1.at(0, k));
        let mu_k_i = ring.clone_el(gso.at(i, k));
        let mu_k_i1 = ring.clone_el(gso.at(i + 1, k));
    }
}

///
/// gso contains on the diagonal the squared lengths of the GS-orthogonalized basis vectors `|bi*|^2`,
/// and above it the GS-coefficients `mu_ij = <bi, bj*> / <bj*, bj*>`.
/// 
fn lll_raw<R, V, T>(ring: R, mut gso: SubmatrixMut<V, El<R>>, mut col_ops: T, delta: &El<R>)
    where R: RingStore + Copy,
        R::Type: LLLRealField,
        V: AsPointerToSlice<El<R>>,
        T: TransformTarget<<R::Type as LLLRealField>::IntRing>
{
    let mut i = 0;
    while i + 1 < gso.col_count() {
        let (target, matrix) = gso.reborrow().split_cols(i..(i + 1), 0..i);
        size_reduce(ring, target, i, matrix.as_const(), &mut col_ops);
        if ring.is_gt(
            &ring.mul_ref_snd(
                ring.sub_ref_fst(delta, ring.mul_ref(gso.as_const().at(i, i + 1), gso.as_const().at(i, i + 1))),
                gso.as_const().at(i, i)
            ),
            gso.as_const().at(i + 1, i + 1)
        ) {
            col_ops.swap(ring.get_ring().integer_ring(), i, i + 1);
            swap_gso_cols(ring, gso.reborrow(), i, i + 1);
        } else {
            i += 1;
        }
    }
}