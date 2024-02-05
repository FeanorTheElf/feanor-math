#![allow(soft_unstable)]
#![allow(non_snake_case)]
#![feature(test)]
#![feature(min_specialization)]
#![feature(maybe_uninit_slice)]
#![feature(allocator_api)]
#![feature(new_uninit)]
#![feature(const_type_name)]
#![feature(is_sorted)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(iter_advance_by)]
#![feature(non_null_convenience)]

#![doc = include_str!("../Readme.md")]

extern crate test;
extern crate libc;
extern crate oorandom;
#[cfg(feature = "parallel")]
extern crate rayon;
#[cfg(feature = "ndarray")]
extern crate ndarray;

///
/// Contains different implementations of [`mempool::MemoryProvider`],
/// which can be used to tell algorithms and ring implementations how to allocate internally
/// used memory.
/// 
pub mod mempool;
#[macro_use]
///
/// Contains the core traits of the library - [`ring::RingBase`] and [`ring::RingStore`],
/// as well as [`ring::CanHomFrom`] and [`ring::CanonicalIso`].
/// 
pub mod ring;
///
/// Contains the trait [`delegate::DelegateRing`] that simplifies implementing the 
/// newtype-pattern for rings.
/// 
pub mod delegate;
///
/// Contains the trait [`vector::VectorView`] for objects that provide access to 
/// some kind of linear container.
/// 
/// This module is currently slightly chaotic, as there is significant functionality overlap
/// between [`vector::VectorView`], [`vector::vec_fn::VectorFn`] and [`std::iter::Iterator`]
/// 
pub mod vector;
///
/// Contains the trait [`divisibility::DivisibilityRing`] for rings that provide information
/// about divisibility of their elements.
/// 
pub mod divisibility;
///
/// Contains the trait [`field::Field`] for rings that are fields.
/// 
pub mod field;
///
/// Contains the trait [`pid::PrincipalIdealRing`] for rings in whom every ideal is principal.
/// Also contains [`pid::EuclideanRing`], which is the simplest way how a ring can become a
/// principal idea ring.
/// 
pub mod pid;
///
/// Contains the trait [`matrix::Matrix`], which is a very minimalistic approach to implement
/// matrices in this library.
/// 
pub mod matrix;
///
/// Contains the trait [`ordered::OrderedRing`] for rings with a total ordering that is compatible
/// with the ring operations.
/// 
pub mod ordered;
///
/// Provides the ring implementation [`primitive_int::StaticRing`] that represents the integer ring
/// with arithmetic given by the primitive integer types ``i8` to `i128`.
/// 
pub mod primitive_int;
///
/// Contains the trait [`integer::IntegerRing`] for rings that represent the ring of integers `Z`.
/// 
pub mod integer;
///
/// A collection of all number-theoretic algorithms that are currently implemented in
/// this crate.
/// 
pub mod algorithms;
///
/// A collection of various more complicated ring traits and implementations, in particular
/// arbitrary-precision integer rings, the integer quotients `Z/nZ` or polynomial rings.
/// 
pub mod rings;
///
/// Ccontains the struct [`wrapper::RingElementWrapper`] that contains an element together with its ring,
/// and thus can provide ring operations without explicit access to the ring.
/// 
/// Using this is for example necessary if you want to use elements of a [`crate::ring::HashableElRing`]-ring
/// as elements in a [`std::collections::HashSet`].
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::integer::*;
/// # use std::collections::HashSet;
/// 
/// let mut set = HashSet::new();
/// set.insert(RingElementWrapper::new(BigIntRing::RING, BigIntRing::RING.int_hom().map(3)));
/// assert!(set.contains(&RingElementWrapper::new(BigIntRing::RING, BigIntRing::RING.int_hom().map(3))));
/// ```
/// 
pub mod wrapper;
///
/// Contains functions to conditionally enable parallel execution of some algorithms.
/// 
pub mod parallel;
///
/// Contains the trait [`homomorphism::Homomorphism`], [`homomorphism::CanHomFrom`] and
/// others that are the foundation of the homomorphism framework, that enables mapping
/// elements between different rings.
/// 
pub mod homomorphism;
///
/// Contains implementations of various iterators and combinators, like [`iters::powerset()`]
/// or [`iters::cartesian_product`].
/// 
pub mod iters;