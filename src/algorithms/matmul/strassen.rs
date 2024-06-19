use crate::algorithms::matmul::ComputeInnerProduct;
use crate::matrix::*;
use crate::ring::*;

#[stability::unstable(feature = "enable")]
pub fn naive_matmul<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
    lhs: TransposableSubmatrix<V1, R::Element, T1>, 
    rhs: TransposableSubmatrix<V2, R::Element, T2>, 
    mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    debug_assert_eq!(lhs.row_count(), dst.row_count());
    debug_assert_eq!(rhs.col_count(), dst.col_count());
    debug_assert_eq!(lhs.col_count(), rhs.row_count());
    for i in 0..lhs.row_count() {
        for j in 0..rhs.col_count() {
            let inner_prod = <_ as ComputeInnerProduct>::inner_product_ref(ring, (0..lhs.col_count()).map(|k| (lhs.at(i, k), rhs.at(k, j))));
            if ADD_ASSIGN {
                ring.add_assign(dst.at_mut(i, j), inner_prod);
            } else {
                *dst.at_mut(i, j) = inner_prod;
            }
        }
    }
}

fn matrix_add_add_sub<R, V1, V2, V3, V4, const T1: bool, const T2: bool, const T3: bool, const T4: bool>(
    a: TransposableSubmatrix<V1, R::Element, T1>, 
    b: TransposableSubmatrix<V2, R::Element, T2>, 
    c: TransposableSubmatrix<V3, R::Element, T3>, 
    mut dst: TransposableSubmatrixMut<V4, R::Element, T4>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>,
        V4: AsPointerToSlice<R::Element>
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
    a: TransposableSubmatrix<V1, R::Element, T1>, 
    b: TransposableSubmatrix<V2, R::Element, T2>, 
    mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
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
    a: TransposableSubmatrix<V1, R::Element, T1>, 
    b: TransposableSubmatrix<V2, R::Element, T2>, 
    mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
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
    a: TransposableSubmatrix<V1, R::Element, T1>, 
    b: TransposableSubmatrix<V2, R::Element, T2>, 
    mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
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
    a: TransposableSubmatrix<V1, R::Element, T1>, 
    b: TransposableSubmatrix<V2, R::Element, T2>, 
    mut dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
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
    val: TransposableSubmatrix<V1, R::Element, T1>, 
    mut dst: TransposableSubmatrixMut<V2, R::Element, T2>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>
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
    val: TransposableSubmatrix<V1, R::Element, T1>, 
    mut dst: TransposableSubmatrixMut<V2, R::Element, T2>, 
    ring: &R
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>
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
    assert!(block_size_log2 >= threshold_size_log2);
    if block_size_log2 == threshold_size_log2 {
        0
    } else if add_assign {
        (1 << (block_size_log2 * 2)) - (1 << (threshold_size_log2 * 2))
    } else {
        (1 << ((block_size_log2 - 1) * 2)) + strassen_mem_size(true, block_size_log2 - 1, threshold_size_log2)
    }
}

#[stability::unstable(feature = "enable")]
pub fn strassen<R, V1, V2, V3, const ADD_ASSIGN: bool, const T1: bool, const T2: bool, const T3: bool>(
    block_size_log2: usize, 
    threshold_size_log2: usize, 
    lhs: TransposableSubmatrix<V1, R::Element, T1>, 
    rhs: TransposableSubmatrix<V2, R::Element, T2>, 
    dst: TransposableSubmatrixMut<V3, R::Element, T3>, 
    ring: &R, 
    memory: &mut [R::Element]
)
    where R: ?Sized + RingBase, 
        V1: AsPointerToSlice<R::Element>,
        V2: AsPointerToSlice<R::Element>,
        V3: AsPointerToSlice<R::Element>
{
    assert_eq!(lhs.row_count(), 1 << block_size_log2);
    assert_eq!(lhs.col_count(), 1 << block_size_log2);
    assert_eq!(rhs.row_count(), 1 << block_size_log2);
    assert_eq!(rhs.col_count(), 1 << block_size_log2);
    assert_eq!(dst.row_count(), 1 << block_size_log2);
    assert_eq!(dst.col_count(), 1 << block_size_log2);

    let steps_left = block_size_log2 - threshold_size_log2;
    if steps_left == 0 {
        naive_matmul::<_, _, _, _, ADD_ASSIGN, T1, T2, T3>(lhs, rhs, dst, ring);
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
            strassen::<_, _, _, _, true, T1, T2, T3>(block_size_log2 - 1, threshold_size_log2, b_lhs, c_rhs, a_dst.reborrow(), ring, &mut *memory);

            // handle t = a a'
            let mut t = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp0, n_half, n_half));
            strassen::<_, _, _, _, false, T1, T2, false>(block_size_log2 - 1, threshold_size_log2, a_lhs, a_rhs, t.reborrow(), ring, &mut *memory);
            matrix_add_assign(t.as_const(), a_dst.reborrow(), ring);
            
            let (tmp1, memory) = memory.split_at_mut(n_half * n_half);
            let (tmp2, memory) = memory.split_at_mut(n_half * n_half);

            // handle w = t + (c + d - a) (a' + d' - b')
            let mut c_d_neg_a_lhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp1, n_half, n_half));
            matrix_add_add_sub(c_lhs, d_lhs, a_lhs, c_d_neg_a_lhs.reborrow(), ring);
            let mut a_d_neg_b_rhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp2, n_half, n_half));
            matrix_add_add_sub(a_rhs, d_rhs, b_rhs, a_d_neg_b_rhs.reborrow(), ring);
            let mut w = t;
            strassen::<_, _, _, _, true, false, false, false>(block_size_log2 - 1, threshold_size_log2, c_d_neg_a_lhs.as_const(), a_d_neg_b_rhs.as_const(), w.reborrow(), ring, &mut *memory);
            matrix_add_assign(w.as_const(), b_dst.reborrow(), ring);
            matrix_add_assign(w.as_const(), c_dst.reborrow(), ring);
            matrix_add_assign(w.as_const(), d_dst.reborrow(), ring);

            // handle y = (a + b - c - d) d'
            let mut a_b_neg_c_d_lhs = c_d_neg_a_lhs;
            matrix_sub_self_assign(b_lhs, a_b_neg_c_d_lhs.reborrow(), ring);
            // implicitly add y to matrix
            strassen::<_, _, _, _, true, false, T2, T3>(block_size_log2 - 1, threshold_size_log2, a_b_neg_c_d_lhs.as_const(), d_rhs, b_dst.reborrow(), ring, &mut *memory);

            // handle z = d (c' + b' - a' - d')
            let mut b_c_neg_a_d_rhs = a_d_neg_b_rhs;
            matrix_sub_self_assign(c_rhs, b_c_neg_a_d_rhs.reborrow(), ring);
            // implicitly add z to matrix
            strassen::<_, _, _, _, true, T1, false, T3>(block_size_log2 - 1, threshold_size_log2, d_lhs, b_c_neg_a_d_rhs.as_const(), c_dst.reborrow(), ring, &mut *memory);

            // handle u = (c - a) (b' - d')
            let mut u = w;
            let mut c_neg_a_lhs = a_b_neg_c_d_lhs;
            matrix_sub(c_lhs, a_lhs, c_neg_a_lhs.reborrow(), ring);
            let mut b_neg_d_rhs = b_c_neg_a_d_rhs;
            matrix_sub(b_rhs, d_rhs, b_neg_d_rhs.reborrow(), ring);
            strassen::<_, _, _, _, false, false, false, false>(block_size_log2 - 1, threshold_size_log2, c_neg_a_lhs.as_const(), b_neg_d_rhs.as_const(), u.reborrow(), ring, &mut *memory);
            matrix_add_assign(u.as_const(), c_dst.reborrow(), ring);
            matrix_add_assign(u.as_const(), d_dst.reborrow(), ring);
            
            // handle v = (c + d) (b' - a')
            let mut v = u;
            let mut c_d_lhs = c_neg_a_lhs;
            matrix_add(c_lhs, d_lhs, c_d_lhs.reborrow(), ring);
            let mut b_neg_a_rhs = b_neg_d_rhs;
            matrix_sub(b_rhs, a_rhs, b_neg_a_rhs.reborrow(), ring);
            strassen::<_, _, _, _, false, false, false, false>(block_size_log2 - 1, threshold_size_log2, c_d_lhs.as_const(), b_neg_a_rhs.as_const(), v.reborrow(), ring, &mut *memory);
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
            strassen::<_, _, _, _, false, T3, T3, T3>(block_size_log2 - 1, threshold_size_log2, c_d_neg_a_lhs.as_const(), a_d_neg_b_rhs.as_const(), w_star.reborrow(), ring, &mut *memory);

            // handle v = (c + d) (b' - a')
            let mut v = d_dst.reborrow();
            let mut c_d_lhs = a_dst.reborrow();
            matrix_add(c_lhs, d_lhs, c_d_lhs.reborrow(), ring);
            let mut b_neg_a_rhs = c_dst.reborrow();
            matrix_sub(b_rhs, a_rhs, b_neg_a_rhs.reborrow(), ring);
            strassen::<_, _, _, _, false, T3, T3, T3>(block_size_log2 - 1, threshold_size_log2, c_d_lhs.as_const(), b_neg_a_rhs.as_const(), v.reborrow(), ring, &mut *memory);

            // handle u = (c - a) (b' - d'); here we need temporary memory
            let mut u = c_dst.reborrow();
            let (tmp0, memory) = memory.split_at_mut(n_half * n_half);
            let mut c_neg_a_lhs = TransposableSubmatrixMut::from(SubmatrixMut::<AsFirstElement<_>, _>::new(tmp0, n_half, n_half));
            matrix_sub(c_lhs, a_lhs, c_neg_a_lhs.reborrow(), ring);
            let mut b_neg_d_rhs = a_dst.reborrow();
            matrix_sub(b_rhs, d_rhs, b_neg_d_rhs.reborrow(), ring);
            strassen::<_, _, _, _, false, false, T3, T3>(block_size_log2 - 1, threshold_size_log2, c_neg_a_lhs.as_const(), b_neg_d_rhs.as_const(), u.reborrow(), ring, &mut *memory);

            // now perform linear combinations; concretely, transform (w*, u, v) into (w* + v, w* + u, w* + u + v)
            matrix_add_assign(w_star.as_const(), u.reborrow(), ring);
            matrix_add_assign(v.as_const(), w_star.reborrow(), ring);
            matrix_add_assign(u.as_const(), v.reborrow(), ring);

            // now implicitly handle y = (a + b - c - d) d'
            let mut a_b_neg_c_d_lhs = c_neg_a_lhs;
            matrix_sub_self_assign_add_sub(b_lhs, d_lhs, a_b_neg_c_d_lhs.reborrow(), ring);
            // implicitly add y to matrix
            strassen::<_, _, _, _, true, false, T2, T3>(block_size_log2 - 1, threshold_size_log2, a_b_neg_c_d_lhs.as_const(), d_rhs, b_dst.reborrow(), ring, &mut *memory);

            // now implicitly handle z = d (c' + b' - a' - d')
            let mut b_c_neg_a_d_rhs = b_neg_d_rhs;
            matrix_add_assign_add_sub(c_rhs, a_rhs, b_c_neg_a_d_rhs.reborrow(), ring);
            // implicitly add z to matrix
            strassen::<_, _, _, _, true, T1, T3, T3>(block_size_log2 - 1, threshold_size_log2, d_lhs, b_c_neg_a_d_rhs.as_const(), c_dst.reborrow(), ring, &mut *memory);

            // handle t = a a'
            let mut t = a_dst.reborrow();
            strassen::<_, _, _, _, false, T1, T2, T3>(block_size_log2 - 1, threshold_size_log2, a_lhs, a_rhs, t.reborrow(), ring, &mut *memory);
            matrix_add_assign(t.as_const(), b_dst.reborrow(), ring);
            matrix_add_assign(t.as_const(), c_dst.reborrow(), ring);
            matrix_add_assign(t.as_const(), d_dst.reborrow(), ring);

            // handle x = b c'
            strassen::<_, _, _, _, true, T1, T2, T3>(block_size_log2 - 1, threshold_size_log2, b_lhs, c_rhs, a_dst.reborrow(), ring, &mut *memory);
        }
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::assert_matrix_eq;

#[test]
fn test_strassen_one_level() {
    {
        let a = [DerefArray::from([ 1, 2 ]), DerefArray::from([ 3, 4 ])];
        let b = [DerefArray::from([ 2, 1 ]), DerefArray::from([ -1, -2 ])];
        let mut result = [DerefArray::from([0, 0]), DerefArray::from([0, 0])];
        let expected = [DerefArray::from([0, -3]), DerefArray::from([2, -5])];
        let mut memory = [i32::MIN; strassen_mem_size(false, 1, 0)];

        strassen::<_, _, _, _, false, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING.get_ring(), 
            &mut memory
        );

        assert_eq!(expected, result);
        // ensure that all of memory was used, i.e. `strassen_mem_size()` is correct
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
    {
        let a = [DerefArray::from([ 1, 0 ]), DerefArray::from([ 7, 2 ])];
        let b = [DerefArray::from([ -3, -3 ]), DerefArray::from([ 3, 1 ])];
        let mut result = [DerefArray::from([1, 1]), DerefArray::from([1, 1])];
        let expected = [DerefArray::from([-3, -3]), DerefArray::from([-15, -19])];
        let mut memory = [i32::MIN; strassen_mem_size(false, 1, 0)];

        strassen::<_, _, _, _, false, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING.get_ring(), 
            &mut memory
        );

        assert_eq!(expected, result);
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
}

#[test]
fn test_strassen_add_assign_one_level() {
    {
        let a = [DerefArray::from([ 1, 2 ]), DerefArray::from([ 3, 4 ])];
        let b = [DerefArray::from([ 2, 1 ]), DerefArray::from([ -1, -2 ])];
        let mut result = [DerefArray::from([10, 20]), DerefArray::from([30, 40])];
        let expected = [DerefArray::from([10, 17]), DerefArray::from([32, 35])];
        let mut memory = [i32::MIN; strassen_mem_size(true, 1, 0)];

        strassen::<_, _, _, _, true, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING.get_ring(), 
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

        strassen::<_, _, _, _, true, false, false, false>(
            1, 
            0, 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&a)), 
            TransposableSubmatrix::from(Submatrix::<DerefArray<_, 2>, _>::new(&b)), 
            TransposableSubmatrixMut::from(SubmatrixMut::<DerefArray<_, 2>, _>::new(&mut result)), 
            StaticRing::<i32>::RING.get_ring(), 
            &mut memory
        );

        assert_eq!(expected, result);
        assert!(memory.iter().all(|x| *x != i32::MIN));
    }
}

#[test]
fn test_strassen_more_levels() {
    let a = OwnedMatrix::from_fn_in(16, 16, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(16, 16, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::zero(16, 16, StaticRing::<i64>::RING);
    let mut memory = (0..strassen_mem_size(false, 4, 1)).map(|_| i64::MIN).collect::<Vec<_>>();

    strassen::<_, _, _, _, false, false, false, false>(
        4, 
        1, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING.get_ring(), 
        &mut memory
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::zero(16, 16, StaticRing::<i64>::RING);
    naive_matmul::<_, _, _, _, false, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING.get_ring()
    );

    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
    assert!(memory.iter().all(|x| *x != i64::MIN));
}

#[test]
fn test_strassen_add_assign_more_levels() {
    let a = OwnedMatrix::from_fn_in(16, 16, |i, j| (i * j) as i64, Global);
    let b = OwnedMatrix::from_fn_in(16, 16, |i, j| i as i64 - (j as i64) * (j as i64), Global);
    let mut result: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(16, 16, |i, j| (i as i64) * (i as i64) + j as i64, Global);
    let mut memory = (0..strassen_mem_size(true, 4, 1)).map(|_| i64::MIN).collect::<Vec<_>>();

    strassen::<_, _, _, _, true, false, false, false>(
        4, 
        1, 
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(result.data_mut()), 
        StaticRing::<i64>::RING.get_ring(), 
        &mut memory
    );

    let mut expected: OwnedMatrix<i64> = OwnedMatrix::from_fn_in(16, 16, |i, j| (i as i64) * (i as i64) + j as i64, Global);
    naive_matmul::<_, _, _, _, true, false, false, false>(
        TransposableSubmatrix::from(a.data()), 
        TransposableSubmatrix::from(b.data()), 
        TransposableSubmatrixMut::from(expected.data_mut()), 
        StaticRing::<i64>::RING.get_ring()
    );

    assert_matrix_eq!(&StaticRing::<i64>::RING, &expected, &result);
    assert!(memory.iter().all(|x| *x != i64::MIN));
}