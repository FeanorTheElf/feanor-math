#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(rustdoc::private_intra_doc_links)]
#![warn(
    // missing_debug_implementations,
    unused_extern_crates,
    unused_import_braces,
    // unused_qualifications,
    unused_results,
    // missing_docs
)]
#![feature(associated_type_defaults)]
#![feature(btree_cursors)]
#![feature(test)]
#![feature(min_specialization)]
#![feature(iter_array_chunks)]
#![feature(allocator_api)]
#![feature(cow_is_borrowed)]
#![feature(fn_traits)]
#![feature(iter_advance_by)]
#![feature(ptr_metadata)]
#![feature(mapped_lock_guards)]
#![feature(unboxed_closures)]
#![feature(ptr_alignment_type)]
#![feature(never_type)]
#![feature(doc_cfg)]
#![feature(int_roundings)]
#![feature(array_try_from_fn)]
#![feature(hasher_prefixfree_extras)]
#![feature(current_thread_id)]
#![feature(thread_id_value)]
#![doc = include_str!("../Readme.md")]

#[cfg(test)]
extern crate test;

pub mod prelude;

// ---- RING FRAMEWORK ----

/// Contains the traits [`ring::RingBase`] and [`ring::RingStore`], which are the foundation of the ring framework in this crate.
#[macro_use]
pub mod ring;
/// Contains the trait [`homomorphism::Homomorphism`], [`homomorphism::CanHomFrom`] and
/// others that are the foundation of the homomorphism framework, that enables mapping
/// elements between different rings.
pub mod homomorphism;
/// Contains the trait [`delegate::DelegateRing`] that simplifies implementing the
/// newtype-pattern for rings.
pub mod delegate;
pub mod ring_properties;
pub mod ring_impls;

// ---- ABELIAN GROUP FRAMEWORK ----

/// Contains the traits [`group::AbelianGroupBase`] and [`group::AbelianGroupStore`], which are the foundation of the group framework in this crate.
/// They are designed analogeously to [`ring::RingBase`] and [`ring::RingStore`].
pub mod group;
pub mod group_impls;

// ---- ALGORITHMS ----

/// A collection of all number-theoretic algorithms that are currently implemented in
/// this crate.
pub mod algorithms;

// ---- UTILITY ----

pub mod function;
/// Contains implementations of various iterators and combinators, like [`iters::powerset()`]
/// or [`iters::multi_cartesian_product`].
pub mod iters;
/// Contains the core of `feanor-math`'s (currently) minimalistic approach to matrices. In
/// particular, we use [`matrix::Submatrix`] and [`matrix::SubmatrixMut`] for matrices that don't
/// own their data.
pub mod matrix;
/// Contains different traits for sequences of elements, namely [`seq::VectorView`] and
/// [`seq::VectorFn`]. They all have some functional overlap with [`ExactSizeIterator`], but differ
/// in how they allow access to the elements of the sequence.
pub mod seq;
/// Contains the struct [`wrapper::RingElementWrapper`] that contains an element together with its
/// ring, and thus can provide ring operations without explicit access to the ring.
///
/// Using this is for example necessary if you want to use elements of a
/// [`crate::ring::HashableElRing`]-ring as elements in a [`std::collections::HashSet`].
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::integer::*;
/// # use std::collections::HashSet;
///
/// let mut set = HashSet::new();
/// set.insert(RingElementWrapper::new(
///     BigIntRing::RING,
///     BigIntRing::RING.int_hom().map(3),
/// ));
/// assert!(set.contains(&RingElementWrapper::new(
///     BigIntRing::RING,
///     BigIntRing::RING.int_hom().map(3)
/// )));
/// ```
pub mod wrapper;

// ---- PRIVATE HELPERS ----

mod cow;

const MAX_PROBABILISTIC_REPETITIONS: usize = 30;

#[cfg(test)]
const RANDOM_TEST_INSTANCE_COUNT: usize = 10;

/// Contains [`unstable_sealed::UnstableSealed`] to mark a trait "sealed" on stable.
#[stability::unstable(feature = "enable")]
pub mod unstable_sealed {

    /// Marks a trait as "sealed" on stable. In other words, using this trait
    /// as supertrait for another trait within `feanor-math` means that implementing
    /// the subtrait for new types is unstable, and only available when `unstable-enable`
    /// is active.
    pub trait UnstableSealed {}
}
