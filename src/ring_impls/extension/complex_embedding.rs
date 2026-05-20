use crate::algorithms::newton::absolute_error_of_poly_eval;
use crate::prelude::*;
use crate::ring_impls::extension::{FreeAlgebra, FreeAlgebraStore};
use crate::ring_impls::float_complex::{Complex64, Complex64Base};
use crate::ring_impls::poly::PolyRingStore;
use crate::ring_impls::poly::dense_poly::DensePolyRing;
use crate::ring_impls::rational::RationalFieldBase;

/// An embedding of a number field `K` into the complex numbers `CC`, represented
/// approximately via floating point numbers.
#[stability::unstable(feature = "enable")]
pub struct ComplexEmbedding<K, I>
where
    K: RingStore,
    K::Ring: FreeAlgebra,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    from: K,
    image_of_generator: El<Complex64>,
    absolute_error_image_of_generator: f64,
}

impl<K, I> ComplexEmbedding<K, I>
where
    K: RingStore,
    K::Ring: FreeAlgebra,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    #[stability::unstable(feature = "enable")]
    pub fn create(field: K, image_of_generator: El<Complex64>, absolute_error_image_of_generator: f64) -> Self {
        Self {
            from: field,
            image_of_generator,
            absolute_error_image_of_generator,
        }
    }

    /// Returns `epsilon > 0` such that when evaluating this homomorphism
    /// at point `x`, the given result is at most `epsilon` from the actual
    /// result (i.e. the result when computed with infinite precision).
    #[stability::unstable(feature = "enable")]
    pub fn absolute_error_bound_at(&self, x: &El<K>) -> f64 {
        let CC = Complex64::RING;
        let CCX = DensePolyRing::new(CC, "X");
        let f = self.from.poly_repr(&CCX, x, CC.can_hom(self.from.base_ring()).unwrap());
        return absolute_error_of_poly_eval(
            &CCX,
            &f,
            self.from.rank(),
            self.image_of_generator,
            self.absolute_error_image_of_generator / CC.abs(self.image_of_generator),
        );
    }
}

impl<K, I> Homomorphism<K::Ring, Complex64Base> for ComplexEmbedding<K, I>
where
    K: RingStore,
    K::Ring: FreeAlgebra,
    BaseRingStore<K>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    type DomainStore = K;
    type CodomainStore = Complex64;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore { &Complex64::RING }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore { &self.from }

    fn map_ref(&self, x: &El<K>) -> <Complex64Base as RingBase>::Element {
        let poly_ring = DensePolyRing::new(*self.codomain(), "X");
        let hom = self.codomain().can_hom(self.from.base_ring()).unwrap();
        return poly_ring.evaluate(
            &self.from.poly_repr(&poly_ring, &x, &hom),
            &self.image_of_generator,
            self.codomain().identity(),
        );
    }

    fn map(&self, x: El<K>) -> <Complex64Base as RingBase>::Element { self.map_ref(&x) }
}
