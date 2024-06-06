use crate::ring::*;

///
/// Trait to allow rings to provide specialized implementations for inner products, i.e.
/// the sums `sum_i a[i] * b[i]`.
/// 
pub trait InnerProductComputation: RingBase {

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a;

    ///
    /// Computes the inner product `sum_i lhs[i] * rhs[i]`.
    /// 
    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element;
}

impl<R: ?Sized + RingBase> InnerProductComputation for R {

    default fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product(els.map(|(l, r)| (self.clone_el(l), r)))
    }

    default fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a
    {
        self.inner_product_ref_fst(els.map(|(l, r)| (l, self.clone_el(r))))
    }

    default fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        self.sum(els.map(|(l, r)| self.mul(l, r)))
    }
}