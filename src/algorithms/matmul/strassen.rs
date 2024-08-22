use crate::algorithms::matmul::ComputeInnerProduct;
use crate::matrix::*;
use crate::ring::*;
use crate::integer::*;
use crate::primitive_int::StaticRing;
use std::alloc::Allocator;
use std::ops::Range;

#[stability::unstable(feature = "enable")]
pub fn naive_matmul<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
    lhs: TransposableSubmatrix<V1, El<R>, T1>, 
    rhs: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    assert_eq!(lhs.row_count(), dst.row_count());
    assert_eq!(rhs.col_count(), dst.col_count());
    assert_eq!(lhs.col_count(), rhs.row_count());
    for i in 0..lhs.row_count() {
        for j in 0..rhs.col_count() {
            let inner_prod = <_ as ComputeInnerProduct>::inner_product_ref(ring.get_ring(), (0..lhs.col_count()).map(|k| (lhs.at(i, k), rhs.at(k, j))));
            if ADD_ASSIGN {
                ring.add_assign(dst.at_mut(i, j), inner_prod);
            } else {
                *dst.at_mut(i, j) = inner_prod;
            }
        }
    }
}

fn matrix_add_add_sub<R, V1, V2, V3, V4, const T1: bool, const T2: bool, const T3: bool, const T4: bool>(
    a: TransposableSubmatrix<V1, El<R>, T1>, 
    b: TransposableSubmatrix<V2, El<R>, T2>, 
    c: TransposableSubmatrix<V3, El<R>, T3>, 
    mut dst: TransposableSubmatrixMut<V4, El<R>, T4>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>,
        V4: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(a.row_count(), b.row_count());
    debug_assert_eq!(a.row_count(), c.row_count());
    debug_assert_eq!(a.row_count(), dst.row_count());
    debug_assert_eq!(a.col_count(), b.col_count());
    debug_assert_eq!(a.col_count(), c.col_count());
    debug_assert_eq!(a.col_count(), dst.col_count());
    
    for i in 0..a.row_count() {
        for j in 0..a.col_count() {
            *dst.at_mut(i, j) = ring.add_ref_snd(ring.sub_ref(a.at(i, j), c.at(i, j)), b.at(i, j));
        }
    }
}

fn matrix_sub_self_assign_add_sub<R, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
    a: TransposableSubmatrix<V1, El<R>, T1>, 
    b: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(a.row_count(), b.row_count());
    debug_assert_eq!(a.row_count(), dst.row_count());
    debug_assert_eq!(a.col_count(), b.col_count());
    debug_assert_eq!(a.col_count(), dst.col_count());
    
    for i in 0..a.row_count() {
        for j in 0..a.col_count() {
            ring.sub_self_assign(dst.at_mut(i, j), ring.sub_ref(a.at(i, j), b.at(i, j)));
        }
    }
}

fn matrix_add_assign_add_sub<R, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
    a: TransposableSubmatrix<V1, El<R>, T1>, 
    b: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(a.row_count(), b.row_count());
    debug_assert_eq!(a.row_count(), dst.row_count());
    debug_assert_eq!(a.col_count(), b.col_count());
    debug_assert_eq!(a.col_count(), dst.col_count());
    
    for i in 0..a.row_count() {
        for j in 0..a.col_count() {
            ring.add_assign(dst.at_mut(i, j), ring.sub_ref(a.at(i, j), b.at(i, j)));
        }
    }
}

fn matrix_sub<R, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
    a: TransposableSubmatrix<V1, El<R>, T1>, 
    b: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(a.row_count(), b.row_count());
    debug_assert_eq!(a.row_count(), dst.row_count());
    debug_assert_eq!(a.col_count(), b.col_count());
    debug_assert_eq!(a.col_count(), dst.col_count());
    
    for i in 0..a.row_count() {
        for j in 0..a.col_count() {
            *dst.at_mut(i, j) = ring.sub_ref(a.at(i, j), b.at(i, j));
        }
    }
}

fn matrix_add<R, V1, V2, V3, const T1: bool, const T2: bool, const T3: bool>(
    a: TransposableSubmatrix<V1, El<R>, T1>, 
    b: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(a.row_count(), b.row_count());
    debug_assert_eq!(a.row_count(), dst.row_count());
    debug_assert_eq!(a.col_count(), b.col_count());
    debug_assert_eq!(a.col_count(), dst.col_count());
    
    for i in 0..a.row_count() {
        for j in 0..a.col_count() {
            *dst.at_mut(i, j) = ring.add_ref(a.at(i, j), b.at(i, j));
        }
    }
}

fn matrix_add_assign<R, V1, V2, const T1: bool, const T2: bool>(
    val: TransposableSubmatrix<V1, El<R>, T1>, 
    mut dst: TransposableSubmatrixMut<V2, El<R>, T2>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(val.row_count(), dst.row_count());
    debug_assert_eq!(val.col_count(), dst.col_count());
    
    for i in 0..val.row_count() {
        for j in 0..val.col_count() {
            ring.add_assign_ref(dst.at_mut(i, j), val.at(i, j));
        }
    }
}

fn matrix_sub_self_assign<R, V1, V2, const T1: bool, const T2: bool>(
    val: TransposableSubmatrix<V1, El<R>, T1>, 
    mut dst: TransposableSubmatrixMut<V2, El<R>, T2>, 
    ring: R
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>
{
    debug_assert_eq!(val.row_count(), dst.row_count());
    debug_assert_eq!(val.col_count(), dst.col_count());
    
    for i in 0..val.row_count() {
        for j in 0..val.col_count() {
            ring.sub_self_assign_ref(dst.at_mut(i, j), val.at(i, j));
        }
    }
}

#[stability::unstable(feature = "enable")]
pub const fn strassen_mem_size(add_assign: bool, block_size_log2: usize, threshold_size_log2: usize) -> usize {
    // solve the recurrence
    //   mem(threshold_n) = 0
    //   mem(n) = 3 * (n/2)^2 + mem(n/2)
    //   mem_not_add_assign(n) = (n/2) + mem(n/2)
    // has the solution mem(n) = n^2 - t^2
    if block_size_log2 <= threshold_size_log2 {
        0
    } else if add_assign {
        (1 << (block_size_log2 * 2)) - (1 << (threshold_size_log2 * 2))
    } else {
        (1 << ((block_size_log2 - 1) * 2)) + strassen_mem_size(true, block_size_log2 - 1, threshold_size_log2)
    }
}

macro_rules! strassen_base_algorithm {
    ($R:expr, $V1:expr, $V2:expr, $V3:expr, $ADD_ASSIGN:expr, $T1:expr, $T2:expr, $T3:expr, $steps_left:expr, $block_size_log2:expr, $lhs:expr, $rhs:expr, $dst:expr, $ring:expr, $memory:expr, $smaller_strassen:ident) => {
        {
            let steps_left = $steps_left;
            let block_size_log2 = $block_size_log2;
            let lhs = $lhs;
            let dst = $dst;
            let rhs = $rhs;
            let ring = $ring;
            let memory = $memory;

            debug_assert_eq!(lhs.row_count(), 1 << block_size_log2);
            debug_assert_eq!(lhs.col_count(), 1 << block_size_log2);
            debug_assert_eq!(rhs.row_count(), 1 << block_size_log2);
            debug_assert_eq!(rhs.col_count(), 1 << block_size_log2);
            debug_assert_eq!(dst.row_count(), 1 << block_size_log2);
            debug_assert_eq!(dst.col_count(), 1 << block_size_log2);

            if steps_left == 0 {
                naive_matmul::<_, _, _, _, $ADD_ASSIGN, $T1, $T2, $T3>(lhs, rhs, dst, ring);
            } else {
                // we have something similar to the "Winograd form"
                // [ a  b ] [ a' b' ]   [ t + x,      w + v + y ]
                // [ c  d ] [ c' d' ] = [ w + u + z,  w + u + v ]
                // where
                // t = a a'
                // u = (c - a) (b' - d')
                // v = (c + d) (b' - a')
                // w = t + (c + d - a) (a' + d' - b')
                // x = b c'
                // y = (a + b - c - d) d'
                // z = d (c' + b' - a' - d')
                let n_half = 1 << (block_size_log2 - 1);
                let n = 1 << block_size_log2;
                let (a_lhs, b_lhs, c_lhs, d_lhs) = (lhs.submatrix(0..n_half, 0..n_half), lhs.submatrix(0..n_half, n_half..n), lhs.submatrix(n_half..n, 0..n_half), lhs.submatrix(n_half..n, n_half..n));
                let (a_rhs, b_rhs, c_rhs, d_rhs) = (rhs.submatrix(0..n_half, 0..n_half), rhs.submatrix(0..n_half, n_half..n), rhs.submatrix(n_half..n, 0..n_half), rhs.submatrix(n_half..n, n_half..n));
                let (ac_dst, bd_dst) = dst.split_cols(0..n_half, n_half..n);
                let (mut a_dst, mut c_dst) = ac_dst.split_rows(0..n_half, n_half..n);
                let (mut b_dst, mut d_dst) = bd_dst.split_rows(0..n_half, n_half..n);

                if ADD_ASSIGN {
                    // now find space for all temporary values; we won't use dst because we want to add results to dst
                    let (tmp0, memory) = memory.split_at_mut(n_half * n_half);

                    // implicitly add x = b c'
                    $smaller_strassen::<_, _, _, _, true, $T1, $T2, $T3>(block_size_log2 - 1, b_lhs, c_rhs, a_dst.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle t = a a'
                    let mut t = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp0, n_half, n_half));
                    $smaller_strassen::<_, _, _, _, false, $T1, $T2, false>(block_size_log2 - 1, a_lhs, a_rhs, t.reborrow(), ring, &mut *memory, steps_left - 1);
                    matrix_add_assign(t.as_const(), a_dst.reborrow(), ring);
                    
                    let (tmp1, memory) = memory.split_at_mut(n_half * n_half);
                    let (tmp2, memory) = memory.split_at_mut(n_half * n_half);

                    // handle w = t + (c + d - a) (a' + d' - b')
                    let mut c_d_neg_a_lhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp1, n_half, n_half));
                    matrix_add_add_sub(c_lhs, d_lhs, a_lhs, c_d_neg_a_lhs.reborrow(), ring);
                    let mut a_d_neg_b_rhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp2, n_half, n_half));
                    matrix_add_add_sub(a_rhs, d_rhs, b_rhs, a_d_neg_b_rhs.reborrow(), ring);
                    let mut w = t;
                    $smaller_strassen::<_, _, _, _, true, false, false, false>(block_size_log2 - 1, c_d_neg_a_lhs.as_const(), a_d_neg_b_rhs.as_const(), w.reborrow(), ring, &mut *memory, steps_left - 1);
                    matrix_add_assign(w.as_const(), b_dst.reborrow(), ring);
                    matrix_add_assign(w.as_const(), c_dst.reborrow(), ring);
                    matrix_add_assign(w.as_const(), d_dst.reborrow(), ring);

                    // handle y = (a + b - c - d) d'
                    let mut a_b_neg_c_d_lhs = c_d_neg_a_lhs;
                    matrix_sub_self_assign(b_lhs, a_b_neg_c_d_lhs.reborrow(), ring);
                    // implicitly add y to matrix
                    $smaller_strassen::<_, _, _, _, true, false, $T2, $T3>(block_size_log2 - 1, a_b_neg_c_d_lhs.as_const(), d_rhs, b_dst.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle z = d (c' + b' - a' - d')
                    let mut b_c_neg_a_d_rhs = a_d_neg_b_rhs;
                    matrix_sub_self_assign(c_rhs, b_c_neg_a_d_rhs.reborrow(), ring);
                    // implicitly add z to matrix
                    $smaller_strassen::<_, _, _, _, true, $T1, false, $T3>(block_size_log2 - 1, d_lhs, b_c_neg_a_d_rhs.as_const(), c_dst.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle u = (c - a) (b' - d')
                    let mut u = w;
                    let mut c_neg_a_lhs = a_b_neg_c_d_lhs;
                    matrix_sub(c_lhs, a_lhs, c_neg_a_lhs.reborrow(), ring);
                    let mut b_neg_d_rhs = b_c_neg_a_d_rhs;
                    matrix_sub(b_rhs, d_rhs, b_neg_d_rhs.reborrow(), ring);
                    $smaller_strassen::<_, _, _, _, false, false, false, false>(block_size_log2 - 1, c_neg_a_lhs.as_const(), b_neg_d_rhs.as_const(), u.reborrow(), ring, &mut *memory, steps_left - 1);
                    matrix_add_assign(u.as_const(), c_dst.reborrow(), ring);
                    matrix_add_assign(u.as_const(), d_dst.reborrow(), ring);
                    
                    // handle v = (c + d) (b' - a')
                    let mut v = u;
                    let mut c_d_lhs = c_neg_a_lhs;
                    matrix_add(c_lhs, d_lhs, c_d_lhs.reborrow(), ring);
                    let mut b_neg_a_rhs = b_neg_d_rhs;
                    matrix_sub(b_rhs, a_rhs, b_neg_a_rhs.reborrow(), ring);
                    $smaller_strassen::<_, _, _, _, false, false, false, false>(block_size_log2 - 1, c_d_lhs.as_const(), b_neg_a_rhs.as_const(), v.reborrow(), ring, &mut *memory, steps_left - 1);
                    matrix_add_assign(v.as_const(), b_dst.reborrow(), ring);
                    matrix_add_assign(v.as_const(), d_dst.reborrow(), ring);

                } else {
                    // same as before, but we require less memory since we can use parts of `dst`

                    // handle w* = (c + d - a) (a' + d' - b')
                    let mut c_d_neg_a_lhs = a_dst.reborrow();
                    matrix_add_add_sub(c_lhs, d_lhs, a_lhs, c_d_neg_a_lhs.reborrow(), ring);
                    let mut a_d_neg_b_rhs = c_dst.reborrow();
                    matrix_add_add_sub(a_rhs, d_rhs, b_rhs, a_d_neg_b_rhs.reborrow(), ring);
                    let mut w_star = b_dst.reborrow();
                    $smaller_strassen::<_, _, _, _, false, $T3, $T3, $T3>(block_size_log2 - 1, c_d_neg_a_lhs.as_const(), a_d_neg_b_rhs.as_const(), w_star.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle v = (c + d) (b' - a')
                    let mut v = d_dst.reborrow();
                    let mut c_d_lhs = a_dst.reborrow();
                    matrix_add(c_lhs, d_lhs, c_d_lhs.reborrow(), ring);
                    let mut b_neg_a_rhs = c_dst.reborrow();
                    matrix_sub(b_rhs, a_rhs, b_neg_a_rhs.reborrow(), ring);
                    $smaller_strassen::<_, _, _, _, false, $T3, $T3, $T3>(block_size_log2 - 1, c_d_lhs.as_const(), b_neg_a_rhs.as_const(), v.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle u = (c - a) (b' - d'); here we need temporary memory
                    let mut u = c_dst.reborrow();
                    let (tmp0, memory) = memory.split_at_mut(n_half * n_half);
                    let mut c_neg_a_lhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp0, n_half, n_half));
                    matrix_sub(c_lhs, a_lhs, c_neg_a_lhs.reborrow(), ring);
                    let mut b_neg_d_rhs = a_dst.reborrow();
                    matrix_sub(b_rhs, d_rhs, b_neg_d_rhs.reborrow(), ring);
                    $smaller_strassen::<_, _, _, _, false, false, $T3, $T3>(block_size_log2 - 1, c_neg_a_lhs.as_const(), b_neg_d_rhs.as_const(), u.reborrow(), ring, &mut *memory, steps_left - 1);

                    // now perform linear combinations; concretely, transform (w*, u, v) into (w* + v, w* + u, w* + u + v)
                    matrix_add_assign(w_star.as_const(), u.reborrow(), ring);
                    matrix_add_assign(v.as_const(), w_star.reborrow(), ring);
                    matrix_add_assign(u.as_const(), v.reborrow(), ring);

                    // now implicitly handle y = (a + b - c - d) d'
                    let mut a_b_neg_c_d_lhs = c_neg_a_lhs;
                    matrix_sub_self_assign_add_sub(b_lhs, d_lhs, a_b_neg_c_d_lhs.reborrow(), ring);
                    // implicitly add y to matrix
                    $smaller_strassen::<_, _, _, _, true, false, $T2, $T3>(block_size_log2 - 1, a_b_neg_c_d_lhs.as_const(), d_rhs, b_dst.reborrow(), ring, &mut *memory, steps_left - 1);

                    // now implicitly handle z = d (c' + b' - a' - d')
                    let mut b_c_neg_a_d_rhs = b_neg_d_rhs;
                    matrix_add_assign_add_sub(c_rhs, a_rhs, b_c_neg_a_d_rhs.reborrow(), ring);
                    // implicitly add z to matrix
                    $smaller_strassen::<_, _, _, _, true, $T1, $T3, $T3>(block_size_log2 - 1, d_lhs, b_c_neg_a_d_rhs.as_const(), c_dst.reborrow(), ring, &mut *memory, steps_left - 1);

                    // handle t = a a'
                    let mut t = a_dst.reborrow();
                    $smaller_strassen::<_, _, _, _, false, $T1, $T2, $T3>(block_size_log2 - 1, a_lhs, a_rhs, t.reborrow(), ring, &mut *memory, steps_left - 1);
                    matrix_add_assign(t.as_const(), b_dst.reborrow(), ring);
                    matrix_add_assign(t.as_const(), c_dst.reborrow(), ring);
                    matrix_add_assign(t.as_const(), d_dst.reborrow(), ring);

                    // handle x = b c'
                    $smaller_strassen::<_, _, _, _, true, $T1, $T2, $T3>(block_size_log2 - 1, b_lhs, c_rhs, a_dst.reborrow(), ring, &mut *memory, steps_left - 1);
                }
            }
        }
    }
}

#[allow(unused_macros)]
macro_rules! unrolled_strassen_impl {
    ($( ($num:literal, $fun:ident, $prev:ident) ),*) => {

        #[stability::unstable(feature = "enable")]
        pub fn dispatch_strassen_impl<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
            block_size_log2: usize, 
            threshold_size_log2: usize, 
            lhs: TransposableSubmatrix<V1, El<R>, T1>, 
            rhs: TransposableSubmatrix<V2, El<R>, T2>, 
            dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
            ring: R, 
            memory: &mut [El<R>]
        )
            where R: RingStore + Copy, 
                V1: AsPointerToSlice<El<R>>,
                V2: AsPointerToSlice<El<R>>,
                V3: AsPointerToSlice<El<R>>
        {
            $(
                fn $fun<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
                    block_size_log2: usize,
                    lhs: TransposableSubmatrix<V1, El<R>, T1>, 
                    rhs: TransposableSubmatrix<V2, El<R>, T2>, 
                    dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
                    ring: R, 
                    memory: &mut [El<R>],
                    _steps_left: usize
                )
                    where R: RingStore + Copy, 
                        V1: AsPointerToSlice<El<R>>,
                        V2: AsPointerToSlice<El<R>>,
                        V3: AsPointerToSlice<El<R>>
                {
                    strassen_base_algorithm!(R, V1, V2, V3, ADD_ASSIGN, T1, T2, T3, $num, block_size_log2, lhs, rhs, dst, ring, memory, $prev)
                }
            )*
            if block_size_log2 <= threshold_size_log2 {
                naive_matmul::<_, _, _, _, ADD_ASSIGN, T1, T2, T3>(lhs, rhs, dst, ring);
            } else {
                match block_size_log2 - threshold_size_log2 {
                    $(
                        $num => $fun::<_, _, _, _, ADD_ASSIGN, T1, T2, T3>(block_size_log2, lhs, rhs, dst, ring, memory, $num),
                    )*
                    _ => panic!()
                }
            }
        }
    };
}

#[cfg(feature = "unrolled_strassen")]
unrolled_strassen_impl!{
    (0, strassen_impl_0, strassen_impl_0),
    (1, strassen_impl_1, strassen_impl_0),
    (2, strassen_impl_2, strassen_impl_1),
    (3, strassen_impl_3, strassen_impl_2),
    (4, strassen_impl_4, strassen_impl_3),
    (5, strassen_impl_5, strassen_impl_4),
    (6, strassen_impl_6, strassen_impl_5),
    (7, strassen_impl_7, strassen_impl_6),
    (8, strassen_impl_8, strassen_impl_7),
    (9, strassen_impl_9, strassen_impl_8),
    (10, strassen_impl_10, strassen_impl_9),
    (11, strassen_impl_11, strassen_impl_10),
    (12, strassen_impl_12, strassen_impl_11),
    (13, strassen_impl_13, strassen_impl_12),
    (14, strassen_impl_14, strassen_impl_13),
    (15, strassen_impl_15, strassen_impl_14),
    (16, strassen_impl_16, strassen_impl_15)
}

#[cfg(not(feature = "unrolled_strassen"))]
#[stability::unstable(feature = "enable")]
pub fn dispatch_strassen_impl<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
    block_size_log2: usize, 
    threshold_size_log2: usize, 
    lhs: TransposableSubmatrix<V1, El<R>, T1>, 
    rhs: TransposableSubmatrix<V2, El<R>, T2>, 
    dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R, 
    memory: &mut [El<R>]
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>
{
    assert_eq!(1 << block_size_log2, lhs.row_count());
    assert_eq!(1 << block_size_log2, lhs.col_count());
    assert_eq!(1 << block_size_log2, rhs.row_count());
    assert_eq!(1 << block_size_log2, rhs.col_count());
    assert_eq!(1 << block_size_log2, dst.row_count());
    assert_eq!(1 << block_size_log2, dst.col_count());
    fn strassen_impl<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
        block_size_log2: usize, 
        lhs: TransposableSubmatrix<V1, El<R>, T1>, 
        rhs: TransposableSubmatrix<V2, El<R>, T2>, 
        dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
        ring: R, 
        memory: &mut [El<R>],
        steps_left: usize
    )
        where R: RingStore + Copy, 
            V1: AsPointerToSlice<El<R>>,
            V2: AsPointerToSlice<El<R>>,
            V3: AsPointerToSlice<El<R>>
    {
        strassen_base_algorithm!(R, V1, V2, V3, ADD_ASSIGN, T1, T2, T3, steps_left, block_size_log2, lhs, rhs, dst, ring, memory, strassen_impl)
    }

    if block_size_log2 <= threshold_size_log2 {
        naive_matmul::<_, _, _, _, ADD_ASSIGN, T1, T2, T3>(lhs, rhs, dst, ring);
    } else {
        let steps_left = block_size_log2 - threshold_size_log2;
        strassen_impl::<_, _, _, _, ADD_ASSIGN, T1, T2, T3>(block_size_log2, lhs, rhs, dst, ring, memory, steps_left)
        
    }
}

#[stability::unstable(feature = "enable")]
pub fn strassen<R, V1, V2, V3, A, const T1: bool, const T2: bool, const T3: bool>(
    add_assign: bool,
    threshold_log2: usize,
    lhs: TransposableSubmatrix<V1, El<R>, T1>, 
    rhs: TransposableSubmatrix<V2, El<R>, T2>, 
    mut dst: TransposableSubmatrixMut<V3, El<R>, T3>, 
    ring: R, 
    allocator: &A
)
    where R: RingStore + Copy, 
        V1: AsPointerToSlice<El<R>>,
        V2: AsPointerToSlice<El<R>>,
        V3: AsPointerToSlice<El<R>>,
        A: Allocator
{
    assert_eq!(lhs.row_count(), dst.row_count());
    assert_eq!(rhs.col_count(), dst.col_count());
    assert_eq!(lhs.col_count(), rhs.row_count());

    const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

    let max_block_size_log2 = [
        ZZ.abs_log2_floor(&(lhs.row_count() as i64)),
        ZZ.abs_log2_floor(&(lhs.col_count() as i64)),
        ZZ.abs_log2_floor(&(rhs.col_count() as i64))
    ].into_iter().min().unwrap();
    if max_block_size_log2.is_none() {
        // some matrix dimension is 0
        return;
    }
    let max_block_size_log2 = max_block_size_log2.unwrap();
    let memory_size = strassen_mem_size(add_assign || (lhs.col_count() >= (2 << max_block_size_log2)), max_block_size_log2, threshold_log2);
    let mut memory = Vec::with_capacity_in(memory_size, allocator);
    memory.resize_with(memory_size, || ring.zero());

    let mut matmul_part = |block_size_log2: usize, lhs_rows: Range<usize>, ks: Range<usize>, rhs_cols: Range<usize>, add_assign: bool| {
        let block_size = 1 << block_size_log2;
        debug_assert_eq!(lhs_rows.len() % block_size, 0);
        debug_assert_eq!(ks.len() % block_size, 0);
        debug_assert_eq!(rhs_cols.len() % block_size, 0);
        if lhs_rows.len() == 0 || ks.len() == 0 || rhs_cols.len() == 0 {
            return;
        }
        for lhs_row in lhs_rows.step_by(block_size) {
            for rhs_col in rhs_cols.clone().step_by(block_size) {
                for k in ks.clone().step_by(block_size) {
                    if add_assign || k > 0 {
                        dispatch_strassen_impl::<_, _, _, _, true, T1, T2, T3>(
                            block_size_log2, 
                            threshold_log2, 
                            lhs.submatrix(lhs_row..(lhs_row + block_size), k..(k + block_size)), 
                            rhs.submatrix(k..(k + block_size), rhs_col..(rhs_col + block_size)), 
                            dst.reborrow().submatrix(lhs_row..(lhs_row + block_size), rhs_col..(rhs_col + block_size)), 
                            ring, 
                            &mut memory
                        );
                    } else {
                        dispatch_strassen_impl::<_, _, _, _, false, T1, T2, T3>(
                            block_size_log2, 
                            threshold_log2, 
                            lhs.submatrix(lhs_row..(lhs_row + block_size), k..(k + block_size)), 
                            rhs.submatrix(k..(k + block_size), rhs_col..(rhs_col + block_size)), 
                            dst.reborrow().submatrix(lhs_row..(lhs_row + block_size), rhs_col..(rhs_col + block_size)), 
                            ring, 
                            &mut memory
                        );
                    }
                }
            }
        }
    };

    let mut lhs_included_rows = 0;
    let mut included_k = 0;
    let mut rhs_included_cols = 0;
    let mut current_block_size_log2 = max_block_size_log2;
    loop {
        // complete using naive algorithm, this is significantly faster than going down all the block sizes
        if current_block_size_log2 <= threshold_log2 {
            if add_assign {
                naive_matmul::<_, _, _, _, true, T1, T2, T3>(
                    lhs.submatrix(lhs_included_rows..lhs.row_count(), 0..included_k), 
                    rhs.submatrix(0..included_k, 0..rhs_included_cols), 
                    dst.reborrow().submatrix(lhs_included_rows..lhs.row_count(), 0..rhs_included_cols), 
                    ring
                );
                naive_matmul::<_, _, _, _, true, T1, T2, T3>(
                    lhs.submatrix(0..lhs.row_count(), 0..included_k), 
                    rhs.submatrix(0..included_k, rhs_included_cols..rhs.col_count()), 
                    dst.reborrow().submatrix(0..lhs.row_count(), rhs_included_cols..rhs.col_count()), 
                    ring
                );
            } else {
                naive_matmul::<_, _, _, _, false, T1, T2, T3>(
                    lhs.submatrix(lhs_included_rows..lhs.row_count(), 0..included_k), 
                    rhs.submatrix(0..included_k, 0..rhs_included_cols), 
                    dst.reborrow().submatrix(lhs_included_rows..lhs.row_count(), 0..rhs_included_cols), 
                    ring
                );
                naive_matmul::<_, _, _, _, false, T1, T2, T3>(
                    lhs.submatrix(0..lhs.row_count(), 0..included_k), 
                    rhs.submatrix(0..included_k, rhs_included_cols..rhs.col_count()), 
                    dst.reborrow().submatrix(0..lhs.row_count(), rhs_included_cols..rhs.col_count()), 
                    ring
                );
            }
            naive_matmul::<_, _, _, _, true, T1, T2, T3>(
                lhs.submatrix(0..lhs.row_count(), included_k..lhs.col_count()), 
                rhs.submatrix(included_k..rhs.row_count(), 0..rhs.col_count()), 
                dst.submatrix(0..lhs.row_count(), 0..rhs.col_count()), 
                ring
            );
            return;
        }
        let block_size = 1 << current_block_size_log2;
        if included_k + block_size <= lhs.col_count() {
            matmul_part(current_block_size_log2, 0..lhs_included_rows, included_k..(included_k + block_size), 0..rhs_included_cols, true);
            included_k += block_size;
        } else if rhs_included_cols + block_size <= rhs.col_count() {
            matmul_part(current_block_size_log2, 0..lhs_included_rows, 0..included_k, rhs_included_cols..(rhs_included_cols + block_size), add_assign);
            rhs_included_cols += block_size;
        } else if lhs_included_rows + block_size <= lhs.row_count() {
            matmul_part(current_block_size_log2, lhs_included_rows..(lhs_included_rows + block_size), 0..included_k, 0..rhs_included_cols, add_assign);
            lhs_included_rows += block_size;
        } else if current_block_size_log2 == 0 {
            return;
        } else {
            current_block_size_log2 -= 1;
        }
    }
}

#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::assert_matrix_eq;

#[test]
fn test_dispatch_strassen_one_level() {
    {
        let a = [DerefArray::from([ 1, 2 ]), DerefArray::from([ 3, 4 ])];
        let b = [DerefArray::from([ 2, 1 ]), DerefArray::from([ -1, -2 ])];
        let mut result = [DerefArray::from([i32::MIN, i32::MIN]), DerefArray::from([i32::MIN, i32::MIN])];
        let expected = [DerefArray::from([0, -3]), DerefArray::from([2, -5])];
        let mut memory = [i32::MIN; strassen_mem_size(false, 1, 0)];

        dispatch_strassen_impl::<_, _, _, _, false, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING, 
            &mut memory
        );

        assert_eq!(expected, result);
        // ensure that all of memory was used, i.e. `strassen_mem_size()` is correct
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
    {
        let a = [DerefArray::from([ 1, 0 ]), DerefArray::from([ 7, 2 ])];
        let b = [DerefArray::from([ -3, -3 ]), DerefArray::from([ 3, 1 ])];
        let mut result = [DerefArray::from([i32::MIN, i32::MIN]), DerefArray::from([i32::MIN, i32::MIN])];
        let expected = [DerefArray::from([-3, -3]), DerefArray::from([-15, -19])];
        let mut memory = [i32::MIN; strassen_mem_size(false, 1, 0)];

        dispatch_strassen_impl::<_, _, _, _, false, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING, 
            &mut memory
        );

        assert_eq!(expected, result);
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
}

#[test]
fn test_dispatch_strassen_add_assign_one_level() {
    {
        let a = [DerefArray::from([ 1, 2 ]), DerefArray::from([ 3, 4 ])];
        let b = [DerefArray::from([ 2, 1 ]), DerefArray::from([ -1, -2 ])];
        let mut result = [DerefArray::from([10, 20]), DerefArray::from([30, 40])];
        let expected = [DerefArray::from([10, 17]), DerefArray::from([32, 35])];
        let mut memory = [i32::MIN; strassen_mem_size(true, 1, 0)];

        dispatch_strassen_impl::<_, _, _, _, true, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING, 
            &mut memory
        );

        assert_eq!(expected, result);
        // ensure that all of memory was used, i.e. `strassen_mem_size()` is correct
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
    {
        let a = [DerefArray::from([ 1, 0 ]), DerefArray::from([ 7, 2 ])];
        let b = [DerefArray::from([ -3, -3 ]), DerefArray::from([ 3, 1 ])];
        let mut result = [DerefArray::from([100, 100]), DerefArray::from([0, 0])];
        let expected = [DerefArray::from([97, 97]), DerefArray::from([-15, -19])];
        let mut memory = [i32::MIN; strassen_mem_size(true, 1, 0)];

        dispatch_strassen_impl::<_, _, _, _, true, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING, 
            &mut memory
        );

        assert_eq!(expected, result);
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
}

#[test]
fn test_dispatch_strassen_more_levels() {
    let a = OwnedMatrix::from_fn_in(16, 16, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(16, 16, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(16, 16, |_, _| i64::MIN, Global);
    let mut memory = (0..strassen_mem_size(false, 4, 1)).map(|_| i64::MIN).collect::<Vec<_>>();

    dispatch_strassen_impl::<_, _, _, _, false, false, false, false>(
        4, 
        1, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING, 
        &mut memory
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::zero(16, 16, StaticRing::<i64>::RING);
    naive_matmul::<_, _, _, _, false, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING
    );

    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
    assert!(memory.iter().all(|x| *x != i64::MIN));
}

#[test]
fn test_dispatch_strassen_add_assign_more_levels() {
    let a = OwnedMatrix::from_fn_in(16, 16, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(16, 16, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(16, 16, |i, j| (i as i64) * (i as i64) + j as i64, Global);
    let mut memory = (0..strassen_mem_size(true, 4, 1)).map(|_| i64::MIN).collect::<Vec<_>>();

    dispatch_strassen_impl::<_, _, _, _, true, false, false, false>(
        4, 
        1, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING, 
        &mut memory
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(16, 16, |i, j| (i as i64) * (i as i64) + j as i64, Global);
    naive_matmul::<_, _, _, _, true, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING
    );

    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
    assert!(memory.iter().all(|x| *x != i64::MIN));
}

#[test]
fn test_strassen_non_power_of_two() {
    let a = OwnedMatrix::from_fn_in(15, 60, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(60, 17, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(15, 17, |_, _| i64::MIN, Global);

    strassen::<_, _, _, _, _, false, false, false>(
        false,
        2, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING, 
        &Global
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::zero(15, 17, StaticRing::<i64>::RING);
    naive_matmul::<_, _, _, _, false, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING
    );

    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
}

#[test]
fn test_strassen_non_power_of_two_add_assign() {
    let a = OwnedMatrix::from_fn_in(15, 60, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(60, 17, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(15, 17, |i, j| (i as i64) * (i as i64) + j as i64, Global);

    strassen::<_, _, _, _, _, false, false, false>(
        true,
        2, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING, 
        &Global
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(15, 17, |i, j| (i as i64) * (i as i64) + j as i64, Global);
    naive_matmul::<_, _, _, _, true, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING
    );
    
    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
}

#[test]
#[should_panic]
fn test_strassen() {
    let A: [DerefArray<i128, 17>; 17] = [
        DerefArray::from([ -50608095887528277,  73054695436629240, 135079650152849717, -44595544675692246, -56581055912953844,-137580940283414139, -60355163059471494,-101114868555246178,   4148770459060710,  94718832024766237, -57745983681626932,-125413720256188926,-120746648124780645, 132012670919130576,   3891221384999917,  29041393213424970, -59411124536656702]),
        DerefArray::from([ -24265593210352294, -65749293152983474, -22356971605668667, -83286941560364249,   4795548074880669,  12360973584288796,-133609907653405596, 131690715369640213,  70145420051574371, -79595493938175963,  18541460376433194,  30902433960721990,  30499919197170082,-113605552486423623,  75021063323094518, 111809902789033559, 111248762258599164]),
        DerefArray::from([ 134495900489110870,  68491289877669123, -91379896221700391, -58363640844726317, -24116463425464214, -36071488370903026, -24621026932728704,-129968721817529386, -38800591580918024,  14260664345423592,  -1121196272463430,  76875526529397047,   -105251634957889,   5231081704160681,   8565213318150924,-139740340136197267,-132338019763416784]),
        DerefArray::from([  48453414151089694,-132633132935429164,  50592988405033151, 100197723494674310, -37416490082132603, 104160393390186086, -52197695428332139, 101285359359682700,-131762221897458374,  70961594126540004, 136680115307456588,  97917284202470707,  17169491908836982,  83736856782730997,-137722441944262398, -63271345266319274,  36320079037924385]),
        DerefArray::from([-136778763503403968, 116305058643150590,  38666215191259280, 143565138826431898,  57885371962200838, 100675484324184341,  29511577434361479,  -8926644089369173, -18770716227490637,  26513847365274330, -74156233898401771, 143602588695337164,  18487027839424912, 116143419162520895,  43387062289215883,  -2200196997831066,  36974055678849824]),
        DerefArray::from([-120244774157854167, -26391194956861447,-124323102808943415, -24094360034098722,-116649338536466576,  35351550066463526, -78690279337528078,  -1668673863616890,   8476279403982034, -32994078605252304,  -4869391955574977, -90672697254337893,  57465686378887906,   8106142194399209,  -4697958723721398, 118680203794556123,-106670176461640944]),
        DerefArray::from([-138140233751943139, -25176771531487410, -33957182623050275, -48436615795891486,-103870370964017309,-127157202973185894,-132276614816859796,  11231048090840190, -48562369407651839,-116228046356167356, -84744826134524446, -37922544678082591, 100974079244743691, 129823816171813317,  36402942380206271, -74641722658056792, 119558480571973315]),
        DerefArray::from([  90961422910016098,  87742430922284919, -41871610706754353, -17198909721771420,   8197622796875557, -14832681135023454, 127350707488007841, -46002264949268147, 111819509956333247, -85726937146084648,-119073678126000782, -27531188412539881,  58911329079673218,  31806852514549269, -54250560359191577, -96969389089559641,  88883759664068364]),
        DerefArray::from([ -19694430495220835, 122624791826508138,  63118585771841776, 121938201072104559,  22762539643269097,  85324452867752792, 134488563789366767, 133388378156609005,  58030388334724238,  78933237741274010, -76787899047502880,-109482092688638797,  27445601610358901,-111017356969326981, 113932170860318701,  34274848712270414,  35378290212677866]),
        DerefArray::from([ -87350294515101358,  31806307716812734,-143423887516003712, -50171651202005610, 120666176226745600, 126836605692886000,  -6972685184326655,-135028943929988291,  36174492578083911, -56364227877855757,-141043614440764498,  59445591706100157,   7179298886907465,   1598810348664771,  27409994952843018,  42857633453430454, -54946800516863333]),
        DerefArray::from([  44046685656114523, -99299149272268292, 122706171439570893,-114946394742252603, -28473655088397246,   3962960299392184,  78726048193439655,   9150935189314008, 105407584164844220, -93441527341540357, -52564161434721237, -80167154637949523, -52201700995394951,  52650109266350840, 127522906678504696, -19184337356777204, -99287860487806781]),
        DerefArray::from([  31952490001227707,  47962690101476144,-127736552370763073,  90923364882371165,  26505874373831297, 118364667232825540, -52948428843991425,  65334459411900832,  78028377441474753,  41596612064929524,  21838180940407722,  84267286003886022,  -3300129156060183, 130047295735791255,  50868254715925852,  14162812965871843,  80312953771945043]),
        DerefArray::from([   5996756265912220,-131879878566935455,   -870235029013795,  78241360616233941, -71886618854508968,  77243785637695000,  95148541353103323,  83341298352028956,-113965031332164947,  91147667052480100,  17242307715872373,  -9158637485631415,  17855748934305576,-143533059281730246, -22208298172726749,-124818510843426185,  62039984855646810]),
        DerefArray::from([  43529480820477626, -33088974119666463, -10650539705111819,-116444460664324756, 114688323413001813,   1416924643419921,-106288481080308918, 106820834857549859, 130335787893841884,  82790190622186012, -44118632159555284,  78325892542199022, -59654571227527517,  95987613409557625,  17318315793770187, 141081243203112898,  91367246069705294]),
        DerefArray::from([ -88249178444218939,  32709502141680789, 128345340938520827,-143229372777619490, -63755710050623790, -17344677002469781, -67100271889345038,-119842114168815325,   7637827324324916,-127375047059426574, -30283701007680798,-122088686738370969,  89028094102285100, -45425551511521197, -69827396828445134,  39873220477389041,  51775449142981569]),
        DerefArray::from([ -83304215397661690, -42210539656445227, 123985129578447912,  68189222115767008,    -41717159964908,   5481666467044168, 102530872502713797, -72485386810589923,-107000587265125516, 139762418832317451,  86529201323481318,  87105671554680896,  85006316356746424, 128398169705810988, -89234144291701359,  63871052138770661,-139494565281002632]),
        DerefArray::from([  60931429096079057,  -9468289160993637, -25229673129419175,   9855499212491473, -95094082781899622,-116640655518330738,  89715497636174913,-118980993008399205,  49694942386534460,  90913954948220084, -66064921771103711,  35588019033862683, -11314395894373635,  74542413579801690, -24949833451332052, 130915059532734333,  41049990564631869])
    ];
    let a: [i128; 17] = [22181356992938787, 132895809188917162, 146748577833448181, 170231591496764369, 286521804176299793, 191945166239912094, 228160163204849550, 210641234879760983, 216990643972873096, 269633947087248691, 86959573437019910, 138498651078848707, 259227762938426409, 108169574406013449, 95894569122194734, 47629834633328439, 191170897374537609];
    let A = Submatrix::<DerefArray<_, 17>, _>::new(&A);
    let a = Submatrix::<AsFirstElement<_>, _>::new(&a, 17, 1);
    let mut b: [i128; 17] = [0; 17];
    let b = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut b, 17, 1);
    let mut mem = [0; strassen_mem_size(false, 4, 3)];

    dispatch_strassen_impl::<_, _, _, _, false, false, false, false>(4, 3, TransposableSubmatrix::from(A), TransposableSubmatrix::from(a), TransposableSubmatrixMut::from(b.reborrow()), &StaticRing::<i128>::RING, &mut mem);

}