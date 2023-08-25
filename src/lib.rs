#![allow(soft_unstable)]
#![allow(non_snake_case)]
#![feature(test)]
#![feature(specialization)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_slice)]
#![feature(allocator_api)]
#![feature(new_uninit)] 
#![feature(core_intrinsics)]
#![feature(const_type_name)]

#![doc = include_str!("../Readme.md")]

extern crate test;

///
/// Struct that may be used to refer to a generic computation.
/// Use it together with [`Exists`] to ensure that a const generic
/// computation works. 
/// 
/// Use as
/// ```
/// #![feature(generic_const_exprs)]
/// # use feanor_math::*; 
/// pub struct Foo<const N: usize>
///     where Expr<{N * N}>: Exists;
/// ```
/// 
pub struct Expr<const VALUE: usize>;
///
/// Trait used to represent that a const generic computation works
/// 
/// Use together with [`Expr`] as
/// ```
/// #![feature(generic_const_exprs)]
/// # use feanor_math::*; 
/// pub struct Foo<const N: usize>
///     where Expr<{N * N}>: Exists;
/// ```
/// 
pub trait Exists {}
impl<T: ?Sized> Exists for T {}

///
/// Module containing different implementations of [`mempool::MemoryProvider`],
/// which can be used to tell algorithms and ring implementations how to allocate internally
/// used memory.
/// 
pub mod mempool;
#[macro_use]
///
/// This module contains the core traits of the library - [`ring::RingBase`] and [`ring::RingStore`],
/// as well as [`ring::CanonicalHom`] and [`ring::CanonicalIso`].
/// 
pub mod ring;
///
/// This module contains the trait [`delegate::DelegateRing`] that simplifies implementing the 
/// newtype-pattern for rings.
/// 
pub mod delegate;
///
/// This module contains the trait [`vector::VectorView`] for objects that provide access to 
/// some kind of linear container.
/// 
/// This module is currently slightly chaotic, as there is significant functionality overlap
/// between [`vector::VectorView`], [`vector::vec_fn::VectorFn`] and [`std::iter::Iterator`]
/// 
pub mod vector;
///
/// This module contains the trait [`divisibility::DivisibilityRing`] for rings that provide information
/// about divisibility of their elements.
/// 
pub mod divisibility;
///
/// This module contains the trait [`field::Field`] for rings that are fields.
/// 
pub mod field;
///
/// This module contains the trait [`euclidean::EuclideanRing`] for rings that provide euclidean division
/// between their elements.
/// 
pub mod euclidean;
///
/// This module contains the trait [`ordered::OrderedRing`] for rings with a total ordering that is compatible
/// with the ring operations.
/// 
pub mod ordered;
///
/// This module provides the ring implementation [`primitive_int::StaticRing`] that represents the integer ring
/// with arithmetic given by the primitive integer types ``i8` to `i128`.
/// 
pub mod primitive_int;
///
/// This module contains the trait [`integer::IntegerRing`] for rings that represent the ring of integers `Z`.
/// 
pub mod integer;
///
/// This module is a collection of all number-theoretic algorithms that are currently implemented in
/// this crate.
/// 
pub mod algorithms;
///
/// This module is a collection of various more complicated ring traits and implementations, in particular
/// arbitrary-precision integer rings, the integer quotients `Z/nZ` or polynomial rings.
/// 
pub mod rings;
///
/// This module contains the struct [`wrapper::RingElementWrapper`] that contains an element together with its ring,
/// and thus can provide ring operations without explicit access to the ring.
/// 
/// Using this is for example necessary if you want to use elements of a [`crate::ring::HashableElRing`]-ring
/// as elements in a [`std::collections::HashSet`].
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::integer::*;
/// # use std::collections::HashSet;
/// 
/// let mut set = HashSet::new();
/// set.insert(RingElementWrapper::new(BigIntRing::RING, BigIntRing::RING.from_int(3)));
/// assert!(set.contains(&RingElementWrapper::new(BigIntRing::RING, BigIntRing::RING.from_int(3))));
/// ```
/// 
pub mod wrapper;