use std::alloc::Allocator;
use std::alloc::Global;
use std::cmp::min;

use crate::algorithms::int_bisect::bisect_floor;
use crate::algorithms::linsolve::smith::determinant_using_pre_smith;
use crate::algorithms::linsolve::*;
use crate::rings::poly::*;
use crate::algorithms::matmul::ComputeInnerProduct;
use crate::divisibility::*;
use crate::homomorphism::Homomorphism;
use crate::matrix::OwnedMatrix;
use crate::pid::PrincipalIdealRing;
use crate::primitive_int::StaticRing;
use crate::ring::*;
use crate::rings::extension::*;
use crate::seq::*;
use crate::rings::poly::PolyRing;

///
/// Default impl for [`FreeAlgebra::from_canonical_basis_extended()`]
///  
#[stability::unstable(feature = "enable")]
pub fn from_canonical_basis_extended<R, V>(ring: &R, vec: V) -> R::Element
    where R: ?Sized + FreeAlgebra,
        V: IntoIterator<Item = El<<R as RingExtension>::BaseRing>>
{
    let mut data = vec.into_iter().collect::<Vec<_>>();
    let mut power_of_canonical_gen = ring.one();
    ring.mul_assign_gen_power(&mut power_of_canonical_gen, ring.rank());
    let mut current_power = ring.one();
    return <_ as ComputeInnerProduct>::inner_product(ring, (0..).map_while(|_| {
        if data.len() == 0 {
            return None;
        }
        let taken_elements = min(data.len(), ring.rank());
        let chunk = data.drain(..taken_elements).chain((taken_elements..ring.rank()).map(|_| ring.base_ring().zero()));
        let current = ring.from_canonical_basis(chunk);
        let result = (current, ring.clone_el(&current_power));
        ring.mul_assign_ref(&mut current_power, &power_of_canonical_gen);
        return Some(result);
    }));
}

#[stability::unstable(feature = "enable")]
pub fn charpoly<R, P, H>(ring: &R, el: &R::Element, poly_ring: P, hom: H) -> El<P>
    where R: ?Sized + FreeAlgebra,
        P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing,
        H: Homomorphism<<R::BaseRing as RingStore>::Type, <<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let minpoly = minpoly(ring, el, &poly_ring, hom);
    let power = StaticRing::<i64>::RING.checked_div(&(ring.rank() as i64), &(poly_ring.degree(&minpoly).unwrap() as i64)).unwrap() as usize;
    return poly_ring.pow(minpoly, power);
}
 
#[stability::unstable(feature = "enable")]
pub fn minpoly<R, P, H>(ring: &R, el: &R::Element, poly_ring: P, hom: H) -> El<P>
    where R: ?Sized + FreeAlgebra,
        P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: LinSolveRing,
        H: Homomorphism<<R::BaseRing as RingStore>::Type, <<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    assert!(!ring.is_zero(el));
    let base_ring = hom.codomain();

    let mut result = None;
    _ = bisect_floor(StaticRing::<i64>::RING, 1, ring.rank() as i64, |d| {
        let d = *d as usize;
        let mut lhs = OwnedMatrix::zero(ring.rank(), d, &base_ring);
        let mut current = ring.one();
        for j in 0..d {
            let wrt_basis = ring.wrt_canonical_basis(&current);
            for i in 0..ring.rank() {
                *lhs.at_mut(i, j) = hom.map(wrt_basis.at(i));
            }
            drop(wrt_basis);
            ring.mul_assign_ref(&mut current, el);
        }
        let mut rhs = OwnedMatrix::zero(ring.rank(), 1, &base_ring);
        let wrt_basis = ring.wrt_canonical_basis(&current);
        for i in 0..ring.rank() {
            *rhs.at_mut(i, 0) = base_ring.negate(hom.map(wrt_basis.at(i)));
        }
        let mut sol = OwnedMatrix::zero(d, 1, &base_ring);
        let sol_poly = |sol: &OwnedMatrix<El<<P::Type as RingExtension>::BaseRing>>| poly_ring.from_terms((0..d).map(|i| (base_ring.clone_el(sol.at(i, 0)), i)).chain([(base_ring.one(), d)].into_iter()));
        match <_ as LinSolveRingStore>::solve_right(base_ring, lhs.data_mut(), rhs.data_mut(), sol.data_mut()) {
            SolveResult::NoSolution => {
                -1
            },
            // I once thought that `FoundUniqueSolution` means we are immediately done;
            // however, that's wrong - it may be that the matrix has a nontrivial kernel,
            // but there still is only one solution to the given system 
            SolveResult::FoundUniqueSolution | SolveResult::FoundSomeSolution => {
                result = Some(sol_poly(&sol));
                1
            }
        }
    });

    return result.unwrap();
}

///
/// Default impl for [`FreeAlgebra::discriminant()`]
/// 
#[stability::unstable(feature = "enable")]
pub fn discriminant<R>(ring: &R) -> El<R::BaseRing>
    where R: ?Sized + FreeAlgebra,
        <R::BaseRing as RingStore>::Type: PrincipalIdealRing
{
    let mut current = ring.one();
    let generator = ring.canonical_gen();
    let traces = (0..(2 * ring.rank())).map(|_| {
        let result = ring.trace(ring.clone_el(&current));
        ring.mul_assign_ref(&mut current, &generator);
        return result;
    }).collect::<Vec<_>>();
    let mut matrix = OwnedMatrix::from_fn(ring.rank(), ring.rank(), |i, j| ring.base_ring().clone_el(&traces[i + j]));
    let result = determinant_using_pre_smith(ring.base_ring(), matrix.data_mut(), Global);
    return result;
}

#[stability::unstable(feature = "enable")]
pub fn create_multiplication_matrix<R: RingStore, A: Allocator>(ring: R, el: &El<R>, allocator: A) -> OwnedMatrix<El<<R::Type as RingExtension>::BaseRing>, A>
    where R::Type: FreeAlgebra
{
    let mut result = OwnedMatrix::zero_in(ring.rank(), ring.rank(), ring.base_ring(), allocator);
    let mut current = ring.clone_el(el);
    let g = ring.canonical_gen();
    for i in 0..ring.rank() {
        {
            let current_basis_repr = ring.wrt_canonical_basis(&current);
            for j in 0..ring.rank() {
                *result.at_mut(j, i) = current_basis_repr.at(j);
            }
        }
        ring.mul_assign_ref(&mut current, &g);
    }
    return result;
}

#[cfg(test)]
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::rings::poly::dense_poly::DensePolyRing;

#[test]
fn test_charpoly() {
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [2]);
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 2]);
    assert_el_eq!(&poly_ring, &expected, charpoly(ring.get_ring(), &ring.canonical_gen(), &poly_ring, &ring.base_ring().identity()));

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 4]);
    assert_el_eq!(&poly_ring, &expected, charpoly(ring.get_ring(), &ring.pow(ring.canonical_gen(), 2), &poly_ring, &ring.base_ring().identity()));

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 6 * X - 6]);
    assert_el_eq!(&poly_ring, &expected, charpoly(ring.get_ring(), &ring.add(ring.canonical_gen(), ring.pow(ring.canonical_gen(), 2)), &poly_ring, &ring.base_ring().identity()));
    
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 4, [2]);
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2]);
    assert_el_eq!(&poly_ring, &expected, charpoly(ring.get_ring(), &ring.canonical_gen(), &poly_ring, &ring.base_ring().identity()));
    
    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 4 * X.pow_ref(2) + 4]);
    assert_el_eq!(&poly_ring, &expected, charpoly(ring.get_ring(), &ring.pow(ring.canonical_gen(), 2), &poly_ring, &ring.base_ring().identity()));
}

#[test]
fn test_minpoly() {
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 6, [2]);
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(6) - 2]);
    assert_el_eq!(&poly_ring, &expected, minpoly(ring.get_ring(), &ring.canonical_gen(), &poly_ring, &ring.base_ring().identity()));

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) - 2]);
    assert_el_eq!(&poly_ring, &expected, minpoly(ring.get_ring(), &ring.pow(ring.canonical_gen(), 2), &poly_ring, &ring.base_ring().identity()));

    let [expected] = poly_ring.with_wrapped_indeterminate(|X| [X.pow_ref(2) - 2 * X - 1]);
    assert_el_eq!(&poly_ring, &expected, minpoly(ring.get_ring(), &ring.add(ring.one(), ring.pow(ring.canonical_gen(), 3)), &poly_ring, &ring.base_ring().identity()));
}

#[test]
fn test_trace() {
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [2, 0, 0]);

    assert_eq!(3, ring.trace(ring.from_canonical_basis([1, 0, 0])));
    assert_eq!(0, ring.trace(ring.from_canonical_basis([0, 1, 0])));
    assert_eq!(0, ring.trace(ring.from_canonical_basis([0, 0, 1])));
    assert_eq!(6, ring.trace(ring.from_canonical_basis([2, 0, 0])));
    assert_eq!(6, ring.trace(ring.from_canonical_basis([2, 1, 0])));
    assert_eq!(6, ring.trace(ring.from_canonical_basis([2, 0, 1])));
}

#[test]
fn test_discriminant() {
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [2, 0, 0]);
    assert_eq!(-108, discriminant(ring.get_ring()));

    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [2, 1, 0]);
    assert_eq!(-104, discriminant(ring.get_ring()));
    
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [3, 0, 0]);
    assert_eq!(-243, discriminant(ring.get_ring()));

    let base_ring = DensePolyRing::new(RationalField::new(StaticRing::<i64>::RING), "X");
    let [f] = base_ring.with_wrapped_indeterminate(|X| [X.pow_ref(3) + 1]);
    let ring = FreeAlgebraImpl::new(&base_ring, 2, [f, base_ring.zero()]);
    let [expected] = base_ring.with_wrapped_indeterminate(|X| [4 * X.pow_ref(3) + 4]);
    assert_el_eq!(&base_ring, expected, discriminant(ring.get_ring()));
}

#[test]
fn test_from_canonical_basis_extended() {
    let ring = FreeAlgebraImpl::new(StaticRing::<i64>::RING, 3, [2]);
    let actual = from_canonical_basis_extended(ring.get_ring(), [1, 2, 3, 4, 5, 6, 7]);
    let expected = ring.from_canonical_basis([37, 12, 15]);
    assert_el_eq!(&ring, expected, actual);
}