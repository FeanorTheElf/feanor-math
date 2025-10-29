
use std::fmt::Debug;
use std::hash::Hash;

use crate::function::no_error;
use crate::homomorphism::*;
use crate::field::*;
use crate::divisibility::*;
use crate::rings::extension::number_field::*;
use crate::rings::extension::*;
use crate::primitive_int::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::integer::*;
use crate::rings::rational::*;
use crate::ring::*;
use crate::seq::VectorFn;
use super::splitting_field::extend_number_field_promise_is_irreducible;
use super::splitting_field::splitting_field;
use super::sqr_mul::generic_pow_shortest_chain_table;

///
/// Computes the Galois closure of `field`.
///
/// If the field is already a Galois extension of `Q`, all conjugates of its canonical
/// generator are returned via [`Result::Err`] instead.
/// 
#[stability::unstable(feature = "enable")]
pub fn compute_galois_closure(field: NumberField) -> Result<
        FreeAlgebraHom<NumberField, NumberField>,
        (NumberField, Vec<El<NumberField>>)
    >
{
    let poly_ring = DensePolyRing::new(field, "X");
    let gen_poly = poly_ring.base_ring().generating_poly(&poly_ring, poly_ring.base_ring().inclusion());
    let poly_to_factor = poly_ring.checked_div(
        &gen_poly,
        &poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map(poly_ring.base_ring().canonical_gen()))
    ).unwrap();

    let mut extended_field = false;
    let (into_extension_field, roots) = splitting_field(
        poly_ring,
        poly_to_factor,
        |poly_ring, extend_with| {
            extended_field = true;
            extend_number_field_promise_is_irreducible(poly_ring, &extend_with)
        }
    );

    if extended_field {
        return Ok(into_extension_field);
    } else {
        let (field, field_again, _) = into_extension_field.destruct();
        debug_assert!(field.get_ring() == field_again.get_ring());
        return Err((field, [field_again.canonical_gen()].into_iter().chain(roots.into_iter().map(|(f, _)| f)).collect()));
    }
}

#[stability::unstable(feature = "enable")]
pub struct GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    field: K,
    image_of_canonical_gen_powers: Vec<El<K>>
}

impl<K, Impl, I> GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// Creates a new Galois automorphism, mapping the canonical generator
    /// of field to the given element.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new(field: K, image_of_canonical_gen: El<K>) -> Self {
        let poly_ring = DensePolyRing::new(field.base_ring(), "X");
        assert!(field.is_zero(&poly_ring.evaluate(&field.generating_poly(&poly_ring, field.base_ring().identity()), &image_of_canonical_gen, field.inclusion())));
        return Self {
            image_of_canonical_gen_powers: (0..field.rank()).map(|i| field.pow(field.clone_el(&image_of_canonical_gen), i)).collect(),
            field: field
        };
    }

    ///
    /// Returns the galois automorphism `x -> self(first(x))`
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn compose_gal(self, first: &Self) -> Self {
        assert!(self.field.get_ring() == first.field.get_ring());
        let new_image = self.map_ref(&first.image_of_canonical_gen_powers[1]);
        return Self::new(self.field, new_image);
    }
    
    ///
    /// Returns the galois automorphism `x -> self(...(self(x))...)`, where
    /// `self` is applied `k` times.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn pow(self, k: usize) -> Self {
        let field = self.field;
        let poly_ring = DensePolyRing::new(field.base_ring(), "X");
        let new_image = generic_pow_shortest_chain_table(
            field.clone_el(&self.image_of_canonical_gen_powers[1]), 
            &(k as i64), 
            StaticRing::<i64>::RING, 
            |x| Ok(poly_ring.evaluate(&field.poly_repr(&poly_ring, x, field.base_ring().identity()), x, field.inclusion())), 
            |x, y| Ok(poly_ring.evaluate(&field.poly_repr(&poly_ring, x, field.base_ring().identity()), y, field.inclusion())), 
            |x| field.clone_el(x), 
            field.canonical_gen()
        ).unwrap_or_else(no_error);
        return Self::new(field, new_image);
    }

    ///
    /// Returns true if this is the identity map.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn is_identity(&self) -> bool {
        self.field.eq_el(&self.image_of_canonical_gen_powers[1], &self.field.canonical_gen())
    }

    ///
    /// Returns the inverse of this automorphism
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn invert(self) -> Self {
        let k = self.field.rank() - 1;
        self.pow(k)
    }
}

impl<K, Impl, I> Homomorphism<NumberFieldBase<Impl, I>, NumberFieldBase<Impl, I>> for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    type DomainStore = K;
    type CodomainStore = K;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.field
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.field
    }

    fn map_ref(&self, x: &<NumberFieldBase<Impl, I> as RingBase>::Element) -> <NumberFieldBase<Impl, I> as RingBase>::Element {
        let coeffs_wrt_basis = self.field.wrt_canonical_basis(x);
        let hom = self.field.inclusion();
        return self.field.sum(self.image_of_canonical_gen_powers.iter().zip(coeffs_wrt_basis.iter()).map(|(x, y)| hom.mul_ref_map(x, &y)));
    }

    fn map(&self, x: <NumberFieldBase<Impl, I> as RingBase>::Element) -> <NumberFieldBase<Impl, I> as RingBase>::Element {
        self.map_ref(&x)
    }
}

impl<K, Impl, I> Debug for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaloisAutomorphism").field("gen_image", &self.field.formatted_el(&self.image_of_canonical_gen_powers[1])).finish()
    }
}

impl<K, Impl, I> Clone for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>> + Clone,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        Self {
            field: self.field.clone(),
            image_of_canonical_gen_powers: self.image_of_canonical_gen_powers.iter().map(|x| self.field.clone_el(x)).collect()
        }
    }
}

impl<K, Impl, I> PartialEq for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        assert!(self.field.get_ring() == other.field.get_ring());
        self.field.eq_el(&self.image_of_canonical_gen_powers[1], &other.image_of_canonical_gen_powers[1])
    }
}

impl<K, Impl, I> Eq for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

impl<K, Impl, I> Hash for GaloisAutomorphism<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra + HashableElRing,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.field.hash(&self.image_of_canonical_gen_powers[1], state)
    }
}

///
/// If the number field is galois, returns its Galois group.
/// 
/// Otherwise, returns its embedding into its Galois closure as [`Result::Err`].
/// 
#[stability::unstable(feature = "enable")]
pub fn compute_galois_group<K>(field: K) -> Result<
        Vec<GaloisAutomorphism<K, DefaultNumberFieldImpl, BigIntRing>>, 
        FreeAlgebraHom<K, NumberField>
    >
    where K: Clone + RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    match compute_galois_closure(RingValue::from_ref(field.get_ring()).clone()) {
        Ok(embedding) => {
            let (_, target, image) = embedding.destruct();
            return Err(FreeAlgebraHom::promise_is_well_defined(field, target, image));
        },
        Err((_, conjugates)) => {
            let mut result: Vec<_> = conjugates.into_iter().map(|x| GaloisAutomorphism::new(field.clone(), x)).collect();
            let id_idx = result.iter().enumerate().filter(|(_, g)| g.is_identity()).next().unwrap().0;
            result.swap(0, id_idx);
            return Ok(result);
        }
    }
}

///
/// Finds the Galois automorphism that will become complex conjugation under
/// the given embedding `K -> C`.
/// 
#[stability::unstable(feature = "enable")]
pub fn find_complex_conjugation<'a, K1, K2, K3>(
    field: K1, 
    complex_embedding: &ComplexEmbedding<K2, DefaultNumberFieldImpl, BigIntRing>, 
    galois_group: &'a [GaloisAutomorphism<K3, DefaultNumberFieldImpl, BigIntRing>]
) -> &'a GaloisAutomorphism<K3, DefaultNumberFieldImpl, BigIntRing>
    where K1: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>,
        K2: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>,
        K3: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>,
{
    assert!(complex_embedding.domain().get_ring() == field.get_ring());
    assert!(galois_group.iter().all(|g| g.domain().get_ring() == field.get_ring()));
    assert_eq!(field.rank(), galois_group.len());

    let CC = complex_embedding.codomain();
    let target = CC.conjugate(complex_embedding.map(field.canonical_gen()));
    let mut result = None;
    for g in galois_group {
        let conj = g.map(field.canonical_gen());
        let image = complex_embedding.map_ref(&conj);
        let dist = CC.abs(CC.sub(target, image));
        let error = complex_embedding.absolute_error_bound_at(&conj);
        if dist <= error {
            assert!(result.is_none(), "not enough precision to separate all complex roots");
            result = Some(g);
        }
    }

    return result.unwrap();
}

#[cfg(test)]
use crate::algorithms::poly_factor::FactorPolyField;

#[test]
fn test_compute_galois_closure() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2]);
    let number_field = NumberField::new(&ZZX, &f);
    let (number_field, conjugates) = compute_galois_closure(number_field).err().unwrap();
    assert_el_eq!(&number_field, number_field.canonical_gen(), &conjugates[0]);
    assert_el_eq!(&number_field, number_field.negate(number_field.canonical_gen()), &conjugates[1]);

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2]);
    let number_field = NumberField::new(&ZZX, &f);
    let into_closure = compute_galois_closure(number_field).ok().unwrap();
    let number_field = into_closure.domain();
    assert_eq!(8, into_closure.codomain().rank());
    crate::homomorphism::generic_tests::test_homomorphism_axioms(&into_closure, (0..8).map(|i| number_field.pow(number_field.canonical_gen(), i)));
    let KX = DensePolyRing::new(into_closure.codomain(), "X");
    let (factors, _) = <_ as FactorPolyField>::factor_poly(&KX, &number_field.generating_poly(&KX, KX.base_ring().inclusion()));
    assert_eq!(4, factors.len());
    
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(3) - X.pow_ref(2) + 1]);
    let number_field = NumberField::new(&ZZX, &f);
    let into_closure = compute_galois_closure(number_field).ok().unwrap();
    let number_field = into_closure.domain();
    assert_eq!(6, into_closure.codomain().rank());
    crate::homomorphism::generic_tests::test_homomorphism_axioms(&into_closure, (0..6).map(|i| number_field.pow(number_field.canonical_gen(), i)));
    let KX = DensePolyRing::new(into_closure.codomain(), "X");
    let (factors, _) = <_ as FactorPolyField>::factor_poly(&KX, &number_field.generating_poly(&KX, KX.base_ring().inclusion()));
    assert_eq!(3, factors.len());
}

#[test]
fn test_compute_galois_group() {
    tracing_subscriber::fmt()
        .compact()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ENTER | tracing_subscriber::fmt::format::FmtSpan::EXIT)
        .with_timer(tracing_subscriber::fmt::time())
        .with_file(false)
        .init();

    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");
    
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(3) - X.pow_ref(2) - 6 * X + 7]);
    let number_field = NumberField::new(&ZZX, &f);
    let galois_group = compute_galois_group(&number_field).ok().unwrap();
    assert_eq!(3, galois_group.len());
    assert!(galois_group[0].is_identity());
    let g = &galois_group[1];
    assert_eq!(&g.clone().pow(2), &galois_group[2]);
    assert_eq!(&g.clone().pow(3), &galois_group[0]);

    // the galois group in this case is C12
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(12) - X.pow_ref(11) + 3 * X.pow_ref(10) - 4 * X.pow_ref(9) + 9 * X.pow_ref(8) + 2 * X.pow_ref(7) + 12 * X.pow_ref(6) + X.pow_ref(5) + 25 * X.pow_ref(4) - 11 * X.pow_ref(3) + 5 * X.pow_ref(2) - 2 * X + 1]);
    let number_field = NumberField::new(&ZZX, &f);
    let galois_group = compute_galois_group(&number_field).ok().unwrap();
    assert_eq!(12, galois_group.len());
    let g = galois_group.iter().filter(|g| !(*g).clone().pow(4).is_identity() && !(*g).clone().pow(6).is_identity()).next().unwrap();
    assert!(g.clone().pow(12).is_identity());
    for i in 0..12 {
        assert!(galois_group.contains(&g.clone().pow(i)));
    }

    // the galois group in this case is D5
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(10) - 2 * X.pow_ref(8) - 9 * X.pow_ref(6) + 57 * X.pow_ref(4) - 69 * X.pow_ref(2) + 47]);
    let number_field = NumberField::new(&ZZX, &f);
    let galois_group = compute_galois_group(&number_field).ok().unwrap();
    assert_eq!(10, galois_group.len());
    let id = &galois_group[0];
    assert!(id.is_identity());
    let mut g1 = &galois_group[1];
    assert!(!g1.is_identity());
    let subgroup: Vec<_> = [id.clone()].into_iter().chain((1..).map(|i| g1.clone().pow(i)).take_while(|g| !g.is_identity())).collect();
    let mut g2 = galois_group.iter().filter(|g| !subgroup.contains(g)).next().unwrap();
    if g1.clone().pow(2).is_identity() {
        std::mem::swap(&mut g1, &mut g2);
    }
    // now g1 has order 5 and g2 has order 2, and together they generate the Dihedral group D5
    assert!(!g1.is_identity());
    assert!(!g2.is_identity());
    assert!(g1.clone().pow(5).is_identity());
    assert!(g2.clone().pow(2).is_identity());
    assert_eq!(g2.clone().compose_gal(&g1).compose_gal(&g2), g1.clone().invert());

}

#[test]
fn test_complex_conjugation() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");
    
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let number_field = NumberField::new(&ZZX, &f);
    let galois_group = compute_galois_group(&number_field).ok().unwrap();
    let complex_embedding = number_field.choose_complex_embedding();
    let conjugation = find_complex_conjugation(&number_field, &complex_embedding, &galois_group);
    assert_el_eq!(&number_field, number_field.negate(number_field.canonical_gen()), conjugation.map(number_field.canonical_gen()));
    
    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) + X.pow_ref(3) + X.pow_ref(2) + X + 1]);
    let number_field = NumberField::new(&ZZX, &f);
    let galois_group = compute_galois_group(&number_field).ok().unwrap();
    let complex_embedding = number_field.choose_complex_embedding();
    let conjugation = find_complex_conjugation(&number_field, &complex_embedding, &galois_group);
    assert_el_eq!(&number_field, number_field.invert(&number_field.canonical_gen()).unwrap(), conjugation.map(number_field.canonical_gen()));
}