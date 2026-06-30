use tracing::instrument;

use crate::prelude::*;

/// Abstracts the operation `mod f` for a monic polynomial `f(X)` of degree `d` over the ring `R`.
pub trait PolyModulus<R: RingStore> {
    /// Returns the ring over which the monic polynomial `f` is defined.
    fn ring(&self) -> &R;

    /// Returns the degree `d` of the monic polynomial `f`, for which this modulus represents
    /// the reduction modulo `f`.
    fn degree(&self) -> usize;

    /// Returns `a_0, ..., a_(d - 1)`, where this modulus represents the reduction modulo
    /// `f(X) = X^d - a_(d - 1) X^(d - 1) - ... - a_1 X - a_0`.
    fn x_pow_rank(&self) -> &[El<R>];

    /// Returns the largest degree `k` such that this modulus supports reduction of degree-`k`
    /// polynomials modulo `f`.
    fn supported_operand_degree(&self) -> usize;

    /// Given the coefficients of a polynomial `g` of degree `<= k`, computes the coefficients
    /// of `g mod f`, and overwrites the input with these.
    ///
    /// Values at index `>= d` will have an unspecified value after this call, they are not
    /// necessarily zero!
    fn perform_reduction(&self, operand: &mut [El<R>]);
}
pub struct SparsePolyModulus<R: RingStore> {
    ring: R,
    x_pow_rank_coeffs: Vec<(usize, El<R>)>,
    x_pow_rank: Vec<El<R>>,
}

impl<R: RingStore> SparsePolyModulus<R> {
    pub fn new(ring: R, x_pow_rank: Vec<El<R>>) -> Self {
        let mut x_pow_rank_coeffs = Vec::new();
        for i in 0..x_pow_rank.len() {
            if !ring.is_zero(&x_pow_rank[i]) {
                x_pow_rank_coeffs.push((i, x_pow_rank[i].clone()))
            }
        }
        Self {
            ring,
            x_pow_rank,
            x_pow_rank_coeffs,
        }
    }
}

impl<R: RingStore> Clone for SparsePolyModulus<R> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            x_pow_rank: self.x_pow_rank.clone(),
            x_pow_rank_coeffs: self.x_pow_rank_coeffs.clone(),
        }
    }
}

impl<R: RingStore> PolyModulus<R> for SparsePolyModulus<R> {
    fn degree(&self) -> usize { self.x_pow_rank.len() }

    fn ring(&self) -> &R { &self.ring }

    fn supported_operand_degree(&self) -> usize { usize::MAX }

    fn x_pow_rank(&self) -> &[El<R>] { &self.x_pow_rank }

    #[instrument(skip_all, level = "trace")]
    fn perform_reduction(&self, operand: &mut [El<R>]) {
        for i in (self.degree()..operand.len()).rev() {
            for (j, c) in &self.x_pow_rank_coeffs {
                let add = self.ring().mul_ref(c, &operand[i]);
                self.ring().add_assign(&mut operand[i - self.degree() + j], add);
            }
        }
    }
}
