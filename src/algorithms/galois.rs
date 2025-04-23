use crate::homomorphism::*;
use crate::field::*;
use crate::divisibility::*;
use crate::rings::extension::number_field::*;
use crate::rings::extension::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::*;
use crate::integer::*;
use crate::ring::*;
use super::poly_factor::FactorPolyField;
use super::splitting_field::extend_number_field_promise_is_irreducible;
use super::splitting_field::NumberFieldHom;

///
/// Computes the Galois closure of `field`.
///
/// If the field is already a Galois extension of `Q`, all conjugates of its canonical
/// generator are returned via [`Result::Err`] instead.
/// 
#[stability::unstable(feature = "enable")]
pub fn compute_galois_closure<K, >(field: K) -> Result<
        (
            NumberFieldHom<K, DefaultNumberFieldImpl, BigIntRing, NumberField<DefaultNumberFieldImpl, BigIntRing>, DefaultNumberFieldImpl, BigIntRing>,
            Vec<El<NumberField<DefaultNumberFieldImpl, BigIntRing>>>
        ),
        Vec<El<K>>
    >
    where K: RingStore<Type = NumberFieldBase<DefaultNumberFieldImpl, BigIntRing>>
{
    let poly_ring = DensePolyRing::new(&field, "X");
    let next_poly_to_factor = poly_ring.checked_div(
        &field.generating_poly(&poly_ring, field.inclusion()),
        &poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map(field.canonical_gen()))
    ).unwrap();
    let (factors, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &next_poly_to_factor);
    let mut roots = vec![field.canonical_gen()];
    let mut unfinished = Vec::new();
    for (f, _) in factors.into_iter() {
        if poly_ring.degree(&f).unwrap() == 1 {
            roots.push(poly_ring.base_ring().negate(poly_ring.base_ring().div(poly_ring.coefficient_at(&f, 0), poly_ring.coefficient_at(&f, 1))));
        } else {
            unfinished.push(f);
        }
    }

    if unfinished.len() == 0 {
        return Err(roots);
    } else {
        let extend_poly_idx = unfinished.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(f).unwrap()).unwrap().0;
        let (inclusion, new_root) = extend_number_field_promise_is_irreducible(&poly_ring, &unfinished[extend_poly_idx]);
        let (_, new_number_field, image_of_gen) = inclusion.destruct();
        let new_poly_ring = DensePolyRing::new(new_number_field, "X");
        let inclusion_ref = NumberFieldHom::new(poly_ring.base_ring(), new_poly_ring.base_ring(), image_of_gen);
        let poly_ring_inclusion = new_poly_ring.lifted_hom(&poly_ring, &inclusion_ref);

        let mut to_factor: Vec<_> = [new_poly_ring.checked_div(
            &poly_ring_inclusion.map_ref(&unfinished[extend_poly_idx]), 
            &new_poly_ring.sub(new_poly_ring.indeterminate(), new_poly_ring.inclusion().map_ref(&new_root))
        ).unwrap()].into_iter().chain(
            unfinished.into_iter().enumerate().filter(|(i, _)| *i != extend_poly_idx).map(|(_, f)| poly_ring_inclusion.map(f))
        ).collect();
        let mut current_roots = roots.into_iter().map(|r| inclusion_ref.map(r)).chain([new_root]).collect();
        let mut poly_ring = new_poly_ring;

        while to_factor.len() > 0 {
            (poly_ring, current_roots, to_factor) = compute_galois_closure_impl_step(poly_ring, current_roots, to_factor);
        }

        let final_number_field = poly_ring.into().into_base_ring();
        let image_gen = final_number_field.clone_el(&current_roots[0]);
        let embedding = NumberFieldHom::new(field, final_number_field, image_gen);
        return Ok((embedding, current_roots));
    }
}

fn compute_galois_closure_impl_step(
    poly_ring: DensePolyRing<NumberField<DefaultNumberFieldImpl, BigIntRing>>,
    mut current_roots: Vec<El<NumberField<DefaultNumberFieldImpl, BigIntRing>>>,
    mut to_factor: Vec<El<DensePolyRing<NumberField<DefaultNumberFieldImpl, BigIntRing>>>>
) -> (
    DensePolyRing<NumberField<DefaultNumberFieldImpl, BigIntRing>>,
    Vec<El<NumberField<DefaultNumberFieldImpl, BigIntRing>>>,
    Vec<El<DensePolyRing<NumberField<DefaultNumberFieldImpl, BigIntRing>>>>
) {
    to_factor.sort_unstable_by_key(|f| poly_ring.degree(f).unwrap());
    let next_poly_to_factor = to_factor.pop().unwrap();
    let (factors, _) = <_ as FactorPolyField>::factor_poly(&poly_ring, &next_poly_to_factor);
    let mut unfinished = Vec::new();
    for (f, _) in factors.into_iter() {
        if poly_ring.degree(&f).unwrap() == 1 {
            current_roots.push(poly_ring.base_ring().negate(poly_ring.base_ring().div(poly_ring.coefficient_at(&f, 0), poly_ring.coefficient_at(&f, 1))));
        } else {
            unfinished.push(f);
        }
    }

    if unfinished.len() > 0 {
        let extend_poly_idx = unfinished.iter().enumerate().max_by_key(|(_, f)| poly_ring.degree(f).unwrap()).unwrap().0;
        let (inclusion, new_root) = extend_number_field_promise_is_irreducible(&poly_ring, &unfinished[extend_poly_idx]);
        let (_, new_number_field, image_of_gen) = inclusion.destruct();
        let new_poly_ring = DensePolyRing::new(new_number_field, "X");
        let inclusion_ref = NumberFieldHom::new(poly_ring.base_ring(), new_poly_ring.base_ring(), image_of_gen);
        let poly_ring_inclusion = new_poly_ring.lifted_hom(&poly_ring, &inclusion_ref);

        to_factor = to_factor.into_iter().map(|f| poly_ring_inclusion.map(f)).chain(
            [new_poly_ring.checked_div(&poly_ring_inclusion.map_ref(&unfinished[extend_poly_idx]), &new_poly_ring.sub(new_poly_ring.indeterminate(), new_poly_ring.inclusion().map_ref(&new_root))).unwrap()]
        ).chain(
            unfinished.into_iter().enumerate().filter(|(i, _)| *i != extend_poly_idx).map(|(_, f)| poly_ring_inclusion.map(f))
        ).collect();
        current_roots = current_roots.into_iter().map(|r| inclusion_ref.map(r)).chain([new_root]).collect();

        return (new_poly_ring, current_roots, to_factor);
    } else {
        return (poly_ring, current_roots, to_factor);
    }
}

#[test]
fn test_compute_galois_closure() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(&ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2]);
    let number_field = NumberField::new(&ZZX, &f);
    let conjugates = compute_galois_closure(&number_field).err().unwrap();
    assert_el_eq!(&number_field, number_field.canonical_gen(), &conjugates[0]);
    assert_el_eq!(&number_field, number_field.negate(number_field.canonical_gen()), &conjugates[1]);

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2]);
    let number_field = NumberField::new(&ZZX, &f);
    let (into_closure, conjugates) = compute_galois_closure(&number_field).ok().unwrap();
}