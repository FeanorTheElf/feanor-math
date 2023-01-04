use crate::ring::*;

use std::cmp::{ max, min };

pub fn naive_assign_mul<R: RingWrapper, const ADD_ASSIGN: bool>(dst: &mut [El<R>], lhs: &[El<R>], rhs: &[El<R>], ring: R) {
    let n = lhs.len();
    assert_eq!(n, rhs.len());
    assert_eq!(2 * n, dst.len());
    for i in 0..(2 * n) {
        let from = max(i as isize - n as isize + 1, 0) as usize;
        let to = min(n, i + 1);
        let value = ring.sum((from..to)
            .map(|j| ring.mul_ref(&lhs[i - j], &rhs[j]))
        );
        if ADD_ASSIGN {
            ring.add_assign(&mut dst[i], value);
        } else {
            dst[i] = value;
        }
    }
}

fn slice_add_assign<R: RingWrapper>(dst: &mut [El<R>], src: &[El<R>], ring: R) {
    assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        ring.add_assign_ref(&mut dst[i], &src[i]);
    }
}

fn slice_sub_assign<R: RingWrapper>(dst: &mut [El<R>], src: &[El<R>], ring: R) {
    assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        ring.sub_assign_ref(&mut dst[i], &src[i]);
    }
}
macro_rules! karatsuba_impl {
    ($( ($num:literal, $fun:ident, $prev:ident) ),*) => {
        fn dispatch_karatsuba_impl<R: RingWrapper + Copy, const ADD_ASSIGN: bool>(
            block_size_log2: usize, threshold_size_log2: usize, dst: &mut [El<R>], lhs: &[El<R>], rhs: &[El<R>], mem: &mut [El<R>], ring: R
        ) {
            $(
                fn $fun<R: RingWrapper + Copy, const ADD_ASSIGN: bool>(block_size_log2: usize, dst: &mut [El<R>], lhs: &[El<R>], rhs: &[El<R>], mem: &mut [El<R>], ring: R) {
                    const STEPS_LEFT: usize = $num;
                    let block_size: usize = 1 << block_size_log2;
                    assert_eq!(block_size, lhs.len());
                    assert_eq!(block_size, rhs.len());
                    assert_eq!(2 * block_size, dst.len());
                    assert!(STEPS_LEFT <= block_size_log2);
                
                    if STEPS_LEFT == 0 {
                        naive_assign_mul::<R, ADD_ASSIGN>(dst, lhs, rhs, ring);
                    } else {
                        let n: usize = block_size / 2;

                        let (mut lower, rest) = mem.split_at_mut(2 * n);
                        $prev::<R, false>(block_size_log2 - 1, &mut lower, &lhs[..n], &rhs[..n], rest, ring);
                        if ADD_ASSIGN {
                            slice_add_assign(&mut dst[..(2 * n)], &lower, ring);
                        } else {
                            for i in 0..(2 * n) {
                                dst[i] = lower[i].clone();
                            }
                            for i in (2 * n)..(4 * n) {
                                dst[i] = ring.zero();
                            }
                        }
                        slice_sub_assign(&mut dst[n..(3 * n)], &lower, ring);
                
                        let mut upper = lower;
                        $prev::<R, false>(block_size_log2 - 1, &mut upper, &lhs[n..(2 * n)], &rhs[n..(2 * n)], rest, ring);
                        slice_add_assign(&mut dst[(2 * n)..(4 * n)], &upper, ring);
                        slice_sub_assign(&mut dst[n..(3 * n)], &upper, ring);
                
                        let (lhs_combined, rhs_combined) = upper.split_at_mut(n);
                        for i in 0..n {
                            lhs_combined[i] = ring.add_ref(&lhs[i], &lhs[i + n]);
                            rhs_combined[i] = ring.add_ref(&rhs[i], &rhs[i + n]);
                        }
                        $prev::<R, true>(block_size_log2 - 1, &mut dst[n..(3 * n)], &lhs_combined, &rhs_combined, rest, ring);
                    }
                }
            )*
            if block_size_log2 <= threshold_size_log2 {
                naive_assign_mul::<R, ADD_ASSIGN>(dst, lhs, rhs, ring);
            } else {
                match block_size_log2 - threshold_size_log2 {
                    $(
                        $num => $fun::<R, true>(block_size_log2, dst, lhs, rhs, mem, ring),
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

pub fn karatsuba<R: RingWrapper + Copy>(threshold_size_log2: usize, dst: &mut [El<R>], lhs: &[El<R>], rhs: &[El<R>], ring: R) {
    if lhs.len() == 0 || rhs.len() == 0 {
        return;
    }
    assert!(dst.len() >= rhs.len() + lhs.len());
    let block_size_log2 = (usize::BITS - 1 - max(lhs.len().leading_zeros(), rhs.len().leading_zeros())) as usize;
    let n = 1 << block_size_log2;
    assert!(lhs.len() >= n);
    assert!(rhs.len() >= n);
    assert!(lhs.len() < 2 * n || rhs.len() < 2 * n);

    let mut memory = Vec::new();
    memory.resize(karatsuba_mem_size(block_size_log2, threshold_size_log2), ring.zero());
    for i in (0..=(lhs.len() - n)).step_by(n) {
        for j in (0..=(rhs.len() - n)).step_by(n) {
            dispatch_karatsuba_impl::<R, true>(
                block_size_log2, 
                threshold_size_log2,
                &mut dst[(i + j)..(i + j + 2 * n)], 
                &lhs[i..(i + n)], 
                &rhs[j..(j + n)], 
                &mut memory[..], 
                ring
            );
        }
    }

    let mut lhs_rem = (lhs.len() / n) * n;
    let mut rhs_rem = (rhs.len() / n) * n;
    let mut rem_block_size_log2: isize = block_size_log2 as isize - 1;
    let mut rem_n = n / 2;
    while rem_block_size_log2 >= 0 {
        let n = rem_n;
        let block_size_log2: usize = rem_block_size_log2 as usize;
        if lhs.len() >= lhs_rem + n {
            for j in (0..=(rhs_rem - n)).step_by(n) {
                dispatch_karatsuba_impl::<R, true>(
                    block_size_log2, 
                    threshold_size_log2,
                    &mut dst[(lhs_rem + j)..(lhs_rem + j + 2 * n)], 
                    &lhs[lhs_rem..(lhs_rem + n)], 
                    &rhs[j..(j + n)], 
                    &mut memory[..], 
                    ring
                );
            }
            lhs_rem += n;
        }
        if rhs.len() >= rhs_rem + n {
            for i in (0..=(lhs.len() - n)).step_by(n) {
                dispatch_karatsuba_impl::<R, true>(
                    rem_block_size_log2 as usize,
                    threshold_size_log2, 
                    &mut dst[(rhs_rem + i)..(rhs_rem + i + 2 * n)], 
                    &lhs[i..(i + n)], 
                    &rhs[rhs_rem..(rhs_rem + n)], 
                    &mut memory[..], 
                    ring
                );
            }
            rhs_rem += n;
        }
        rem_n = rem_n / 2;
        rem_block_size_log2 = rem_block_size_log2 - 1;
    }
}

fn karatsuba_mem_size(block_size_log2: usize, threshold_size_log2: usize) -> usize {
    if block_size_log2 <= threshold_size_log2 {
        return 0;
    }
    return (2 << block_size_log2) - (2 << threshold_size_log2);
}

#[cfg(test)]
use crate::primitive::*;

#[test]
fn test_karatsuba_impl() {
    let a = [1, 2, 3, 0];
    let b = [3, 4, 5, 0];
    let mut c = [0; 8];
    let mut tmp = [0; 4];
    dispatch_karatsuba_impl::<_, true>(2, 1, &mut c[..], &a[..], &b[..], &mut tmp[..], StaticRing::<i64>::RING);
    assert_eq!([3, 10, 22, 22, 15, 0, 0, 0], c);
}

#[test]
fn test_karatsuba_mul() {
    let mut c = vec![0, 0, 0, 0];
    karatsuba(0, &mut c[..], &[-1, 0], &[1, 0], StaticRing::<i64>::RING);
    assert_eq!(vec![-1, 0, 0, 0], c);

    let a = vec![1, 0, 1, 0, 1, 2, 3];
    let b = vec![3, 4];
    let mut c = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
    karatsuba(0, &mut c[..], &a[..], &b[..], StaticRing::<i64>::RING);
    assert_eq!(vec![3, 4, 3, 4, 3, 10, 17, 12, 0], c);
}
