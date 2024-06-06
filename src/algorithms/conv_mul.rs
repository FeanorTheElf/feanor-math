use crate::ring::*;
use crate::mempool::*;
use super::karatsuba::*;

///
/// Trait to allow rings to provide specialized implementations for computing a convolution,
/// i.e. the sums `sum_i a[i] * b[j - i]` for all `j`.
/// 
pub trait ConvMulComputation: RingBase {

    ///
    /// Define a threshold from which on the default implementation of [`ConvMulComputation::add_assign_conv_mul()`]
    /// will use the Karatsuba algorithm.
    /// 
    /// Concretely, when this returns `k`, [`ConvMulComputation::add_assign_conv_mul()`] will reduce the 
    /// convolution down to ones on slices of size `2^k`, and compute their convolution naively. The default
    /// value is `0`, but if the considered rings have fast multiplication (compared to addition), then setting
    /// it higher may result in a performance gain.
    /// 
    fn karatsuba_threshold(&self) -> usize;

    ///
    /// Computes the convolution of `lhs` and `rhs`, and adds the result to `dst`.
    /// 
    /// In other words, computes `dst[j] += sum_i lhs[i] * rhs[j - i]` for all `j`,
    /// where `i` runs through `max(0, j - rhs.len() - 1), ..., min(j, lhs.len() - 1)`.
    /// Requires that `dst` is of length at least `lhs.len() + rhs.len() + 1`.
    /// 
    fn add_assign_conv_mul<M: MemoryProvider<Self::Element>>(&self, dst: &mut [Self::Element], lhs: &[Self::Element], rhs: &[Self::Element], memory_provider: &M);
}

impl<R: RingBase + ?Sized> ConvMulComputation for R {

    default fn karatsuba_threshold(&self) -> usize {
        0
    }

    fn add_assign_conv_mul<M: MemoryProvider<Self::Element>>(&self, dst: &mut [Self::Element], lhs: &[Self::Element], rhs: &[Self::Element], memory_provider: &M) {
        // checks are done by karatsuba()
        karatsuba(self.karatsuba_threshold(), dst, lhs, rhs, RingRef::new(self), memory_provider);
    }
}

#[cfg(test)]
use test;
#[cfg(test)]
use crate::primitive_int::*;
#[cfg(test)]
use crate::default_memory_provider;

#[bench]
fn bench_naive_mul_1024_bit(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(10, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &default_memory_provider!());
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}

#[bench]
fn bench_karatsuba_mul_1024_bit(bencher: &mut test::Bencher) {
    let a: Vec<i32> = (0..32).collect();
    let b: Vec<i32> = (0..32).collect();
    let mut c: Vec<i32> = (0..64).collect();
    bencher.iter(|| {
        c.clear();
        c.resize(64, 0);
        karatsuba(4, &mut c[..], &a[..], &b[..], StaticRing::<i32>::RING, &default_memory_provider!());
        assert_eq!(c[31], 31 * 31 * 32 / 2 - 31 * (31 + 1) * (31 * 2 + 1) / 6);
        assert_eq!(c[62], 31 * 31);
    });
}