use tracing::instrument;

use crate::algorithms::matmul::ComputeInnerProduct;
use crate::ring::*;
use crate::seq::*;
use crate::integer::*;
use crate::primitive_int::*;

use std::alloc::Allocator;
use std::cmp::{max, min};

#[stability::unstable(feature = "enable")]
pub fn naive_assign_mul<R, V1, V2, V3, const ADD_ASSIGN: bool>(mut dst: V1, lhs: V2, rhs: V3, ring: R) 
    where R: RingStore, V1: VectorViewMut<El<R>>, V2: VectorView<El<R>>, V3: VectorView<El<R>>
{
    for i in 0..(lhs.len() + rhs.len()) {
        let from = max(i as isize - lhs.len() as isize + 1, 0) as usize;
        let to = min(rhs.len(), i + 1);
        let value = <_ as ComputeInnerProduct>::inner_product_ref(ring.get_ring(), (from..to)
            .map(|j| (lhs.at(i - j), rhs.at(j)))
        );
        if ADD_ASSIGN {
            ring.add_assign(dst.at_mut(i), value);
        } else {
            *dst.at_mut(i) = value;
        }
    }
}

fn slice_add_assign<R, V1, V2>(mut dst: V1, src: V2, ring: R) 
    where R: RingStore, V1: VectorViewMut<El<R>>, V2: VectorView<El<R>> 
{
    assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        ring.add_assign_ref(dst.at_mut(i), src.at(i));
    }
}

fn slice_assign<R, V1, V2>(mut dst: V1, src: V2, ring: R) 
    where R: RingStore, V1: VectorViewMut<El<R>>, V2: VectorView<El<R>> 
{
    assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        *dst.at_mut(i) = ring.clone_el(src.at(i));
    }
}

fn slice_zero<R, V1>(mut dst: V1, ring: R) 
    where R: RingStore, V1: VectorViewMut<El<R>>
{
    for i in 0..dst.len() {
        *dst.at_mut(i) = ring.zero();
    }
}

fn slice_sub_assign<R, V1, V2>(mut dst: V1, src: V2, ring: R) 
    where R: RingStore, V1: VectorViewMut<El<R>>, V2: VectorView<El<R>>
{
    assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        ring.sub_assign_ref(dst.at_mut(i), src.at(i));
    }
}

macro_rules! karatsuba_impl {
    ($( ($num:literal, $fun:ident, $prev:ident) ),*) => {
        fn dispatch_karatsuba_impl<R, V2, V3, const ADD_ASSIGN: bool>(
            block_size_log2: usize, threshold_size_log2: usize, dst: &mut [El<R>], lhs: V2, rhs: V3, mem: &mut [El<R>], ring: R
        )
            where R: RingStore + Copy, V2: SelfSubvectorView<El<R>> + Copy, V3: SelfSubvectorView<El<R>> + Copy
        {
            $(
                fn $fun<R, V1, V2, V3, const ADD_ASSIGN: bool>(block_size_log2: usize, dst: &mut [El<R>], lhs: V2, rhs: V3, mem: &mut [El<R>], ring: R) 
                    where R: RingStore + Copy, V2: SelfSubvectorView<El<R>> + Copy, V3: SelfSubvectorView<El<R>> + Copy
                {
                    const STEPS_LEFT: usize = $num;
                    let block_size: usize = 1 << block_size_log2;
                    debug_assert_eq!(block_size, lhs.len());
                    debug_assert_eq!(block_size, rhs.len());
                    debug_assert_eq!(2 * block_size, dst.len());
                    debug_assert!(STEPS_LEFT <= block_size_log2);
                
                    if STEPS_LEFT == 0 {
                        naive_assign_mul::<R, _, V2, V3, ADD_ASSIGN>(dst, lhs, rhs, ring);
                    } else {
                        let n: usize = block_size / 2;

                        let (lower, rest) = mem.split_at_mut(2 * n);
                        $prev::<R, &mut [El<R>], V2, V3, false>(block_size_log2 - 1, lower, lhs.restrict(..n), rhs.restrict(..n), rest, ring);
                        if ADD_ASSIGN {
                            slice_add_assign(dst.restrict(..(2 * n)), &lower, ring);
                        } else {
                            slice_assign(dst.restrict(..(2 * n)), &lower, ring);
                            slice_zero(dst.restrict((2 * n)..), ring);
                        }
                        slice_sub_assign(dst.restrict(n..(3 * n)), &lower, ring);
                
                        let upper = lower;
                        $prev::<R, &mut [El<R>], _, _, false>(block_size_log2 - 1, upper, lhs.restrict(n..(2 * n)), rhs.restrict(n..(2 * n)), rest, ring);
                        slice_add_assign(dst.restrict((2 * n)..(4 * n)), &upper, ring);
                        slice_sub_assign(dst.restrict(n..(3 * n)), &upper, ring);
                
                        let (lhs_combined, rhs_combined) = upper.split_at_mut(n);
                        for i in 0..n {
                            lhs_combined[i] = ring.add_ref(lhs.at(i), lhs.at(i + n));
                            rhs_combined[i] = ring.add_ref(rhs.at(i), rhs.at(i + n));
                        }
                        $prev::<R, &mut [El<R>], _, _, true>(block_size_log2 - 1, dst.restrict(n..(3 * n)), &lhs_combined[..], &rhs_combined[..], rest, ring);
                    }
                }
            )*
            if block_size_log2 <= threshold_size_log2 {
                naive_assign_mul::<R, _, _, _, ADD_ASSIGN>(dst, lhs, rhs, ring);
            } else {
                match block_size_log2 - threshold_size_log2 {
                    $(
                        $num => $fun::<R, &mut [El<R>], _, _, ADD_ASSIGN>(block_size_log2, dst, lhs, rhs, mem, ring),
                    )*
                    _ => panic!()
                }
            }
        }
    };
}

karatsuba_impl!{
    (0, karatsuba_impl_0, karatsuba_impl_0),
    (1, karatsuba_impl_1, karatsuba_impl_0),
    (2, karatsuba_impl_2, karatsuba_impl_1),
    (3, karatsuba_impl_3, karatsuba_impl_2),
    (4, karatsuba_impl_4, karatsuba_impl_3),
    (5, karatsuba_impl_5, karatsuba_impl_4),
    (6, karatsuba_impl_6, karatsuba_impl_5),
    (7, karatsuba_impl_7, karatsuba_impl_6),
    (8, karatsuba_impl_8, karatsuba_impl_7),
    (9, karatsuba_impl_9, karatsuba_impl_8),
    (10, karatsuba_impl_10, karatsuba_impl_9),
    (11, karatsuba_impl_11, karatsuba_impl_10),
    (12, karatsuba_impl_12, karatsuba_impl_11),
    (13, karatsuba_impl_13, karatsuba_impl_12),
    (14, karatsuba_impl_14, karatsuba_impl_13),
    (15, karatsuba_impl_15, karatsuba_impl_14),
    (16, karatsuba_impl_16, karatsuba_impl_15)
}

#[stability::unstable(feature = "enable")]
#[instrument(skip_all, level = "trace")]
pub fn karatsuba<R, V1, V2, A: Allocator>(threshold_size_log2: usize, dst: &mut [El<R>], lhs: V1, rhs: V2, ring: R, allocator: &A) 
    where R: RingStore + Copy, V1: SelfSubvectorView<El<R>> + Copy, V2: SelfSubvectorView<El<R>> + Copy
{
    if lhs.len() == 0 || rhs.len() == 0 {
        return;
    }
    assert!(dst.len() >= rhs.len() + lhs.len());
    if threshold_size_log2 == usize::MAX || lhs.len() < (1 << threshold_size_log2) || rhs.len() < (1 << threshold_size_log2) {
        naive_assign_mul::<R, _, _, _, true>(dst, lhs, rhs, ring);
        return;
    }

    let lhs_log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&(lhs.len() as i64)).unwrap();
    let rhs_log2_len = StaticRing::<i64>::RING.abs_log2_ceil(&(rhs.len() as i64)).unwrap();

    fn pad<'a, R, V, A>(data: V, len: usize, ring: R, allocator: &'a A) -> Vec<El<R>, &'a A>
        where R: RingStore + Copy, V: SelfSubvectorView<El<R>> + Copy, A: Allocator
    {
        let mut new = Vec::with_capacity_in(len, allocator);
        new.extend(data.clone_ring_els(ring).iter());
        if new.len() < len {
            new.resize_with(len, || ring.zero());
        }
        return new;
    }
    
    if lhs.len() != 1 << lhs_log2_len {
        if dst.len() < (1 << lhs_log2_len) + (1 << rhs_log2_len) {
            let mut new_dst = pad(&dst[..], (1 << lhs_log2_len) + (1 << rhs_log2_len), ring, allocator);
            karatsuba(threshold_size_log2, &mut new_dst, &pad(lhs, 1 << lhs_log2_len, ring, allocator)[..], rhs, ring, allocator);
            for (i, x) in new_dst.into_iter().enumerate().take(dst.len()) {
                dst[i] = x;
            }
        } else {
            karatsuba(threshold_size_log2, dst, &pad(lhs, 1 << lhs_log2_len, ring, allocator)[..], rhs, ring, allocator);
        }
        return;
    }
    if rhs.len() != 1 << rhs_log2_len {
        if dst.len() < (1 << lhs_log2_len) + (1 << rhs_log2_len) {
            let mut new_dst = pad(&dst[..], (1 << lhs_log2_len) + (1 << rhs_log2_len), ring, allocator);
            karatsuba(threshold_size_log2, &mut new_dst, lhs, &pad(rhs, 1 << rhs_log2_len, ring, allocator)[..], ring, allocator);
            for (i, x) in new_dst.into_iter().enumerate().take(dst.len()) {
                dst[i] = x;
            }
        } else {
            karatsuba(threshold_size_log2, dst, lhs, &pad(rhs, 1 << rhs_log2_len, ring, allocator)[..], ring, allocator);
        }
        return;
    }

    let block_size_log2 = min(lhs_log2_len, rhs_log2_len);
    let n = 1 << block_size_log2;

    let memory_size = karatsuba_mem_size(block_size_log2, threshold_size_log2);
    let mut memory = Vec::with_capacity_in(memory_size, allocator);
    memory.extend((0..memory_size).map(|_| ring.zero()));

    if lhs.len() == n {
        assert!(rhs.len() % n == 0);
        for i in 0..(rhs.len() / n) {
            dispatch_karatsuba_impl::<R, _, _, true>(
                block_size_log2,
                threshold_size_log2, 
                &mut dst[(i * n)..(i * n + 2 * n)], 
                lhs, 
                rhs.restrict((i * n)..(i * n + n)), 
                &mut memory[..], 
                ring
            );
        }
    } else {
        assert!(lhs.len() % n == 0);
        assert!(rhs.len() == n);
        for i in 0..(lhs.len() / n) {
            dispatch_karatsuba_impl::<R, _, _, true>(
                block_size_log2,
                threshold_size_log2, 
                &mut dst[(i * n)..(i * n + 2 * n)], 
                lhs.restrict((i * n)..(i * n + n)), 
                rhs, 
                &mut memory[..], 
                ring
            );
        }
    }
}

fn karatsuba_mem_size(block_size_log2: usize, threshold_size_log2: usize) -> usize {
    if block_size_log2 <= threshold_size_log2 {
        return 0;
    }
    return (2 << block_size_log2) - (2 << threshold_size_log2);
}

#[cfg(test)]
use std::alloc::Global;
#[cfg(test)]
use crate::tracing::LogAlgorithmSubscriber;

#[test]
fn test_karatsuba_impl() {
    LogAlgorithmSubscriber::init_test();
    let a = [1, 2, 3, 0];
    let b = [3, 4, 5, 0];
    let mut c = [0; 8];
    let mut tmp = [0; 4];
    dispatch_karatsuba_impl::<_, _, _, true>(2, 1, &mut c[..], &a[..], &b[..], &mut tmp[..], StaticRing::<i64>::RING);
    assert_eq!([3, 10, 22, 22, 15, 0, 0, 0], c);
}

#[test]
fn test_karatsuba_mul() {
    LogAlgorithmSubscriber::init_test();
    let mut c = vec![0, 0, 0, 0];
    karatsuba(0, &mut c[..], &[-1, 0][..], &[1, 0][..], StaticRing::<i64>::RING, &Global);
    assert_eq!(vec![-1, 0, 0, 0], c);

    let a = vec![1, 0, 1, 0, 1, 2, 3];
    let b = vec![3, 4];
    let mut c = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
    karatsuba(0, &mut c[..], &a[..], &b[..], StaticRing::<i64>::RING, &Global);
    assert_eq!(vec![3, 4, 3, 4, 3, 10, 17, 12, 0], c);
}
