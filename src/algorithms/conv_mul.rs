use crate::ring::*;
use crate::mempool::*;
use super::karatsuba::*;

///
/// Helper trait that gives rings the ability to specify to which length naive multiplication
/// should be used (as opposed to karatsuba convolution).
/// 
/// This is default implemented for all rings, but if a ring provides very cheap multiplication,
/// it might be worth considering specializing `KaratsubaHint` with a larger value, to gain a 
/// performance benefit.
/// 
pub trait KaratsubaHint: RingBase {

    fn karatsuba_threshold(&self) -> usize;
}

impl<R: RingBase + ?Sized> KaratsubaHint for R {

    default fn karatsuba_threshold(&self) -> usize {
        0
    }
}

pub fn add_assign_convoluted_mul<R: RingStore + Copy, M: MemoryProvider<El<R>>>(dst: &mut [El<R>], lhs: &[El<R>], rhs: &[El<R>], ring: R, memory_provider: &M) {
    // checks are done by karatsuba()
    karatsuba(ring.get_ring().karatsuba_threshold(), dst, lhs, rhs, ring, memory_provider);
}

#[cfg(test)]
use test;
#[cfg(test)]
use crate::primitive_int::*;

#[bench]
fn bench_naive_mul(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(10, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &AllocatingMemoryProvider);
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}

#[bench]
fn bench_karatsuba_mul(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(4, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &AllocatingMemoryProvider);
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}