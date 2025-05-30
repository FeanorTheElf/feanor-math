#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
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
#![feature(maybe_uninit_slice)]
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
#![feature(array_chunks)]
#![feature(doc_cfg)]
#![feature(int_roundings)]
#![feature(array_try_from_fn)]
#![feature(hasher_prefixfree_extras)]

#![doc = include_str!("../Readme.md")]

#[cfg(test)]
extern crate test;

const MAX_PROBABILISTIC_REPETITIONS: usize = 30;
const DEFAULT_PROBABILISTIC_REPETITIONS: usize = 30;

#[cfg(test)]
const RANDOM_TEST_INSTANCE_COUNT: usize = 10;

macro_rules! static_assert_impls {
    ($type:ty: $trait:tt) => {
        {
            fn assert_impls<T>() where T: ?Sized + $trait {}
            assert_impls::<$type>();
        }
    };
}

///
/// Contains [`unstable_sealed::UnstableSealed`] to mark a trait "sealed" on stable.
/// 
#[stability::unstable(feature = "enable")]
pub mod unstable_sealed {

    ///
    /// Marks a trait as "sealed" on stable. In other words, using this trait
    /// as supertrait for another trait within `feanor-math` means that implementing
    /// the subtrait for new types is unstable, and only available when `unstable-enable`
    /// is active.
    /// 
    pub trait UnstableSealed {}
}

mod unsafe_any;
mod lazy;
mod cow;

///
/// Contains [`computation::ComputationController`] to observe long-running computations.
/// 
#[macro_use]
pub mod computation;
///
/// Contains the core traits of the library - [`ring::RingBase`] and [`ring::RingStore`].
/// 
#[macro_use]
pub mod ring;
///
/// Contains the trait [`delegate::DelegateRing`] that simplifies implementing the 
/// newtype-pattern for rings.
/// 
pub mod delegate;
///
/// Contains different traits for sequences of elements, namely [`seq::VectorView`] and [`seq::VectorFn`]. 
/// They all have some functional overlap with [`ExactSizeIterator`], but differ in how they allow
/// access to the elements of the sequence.
/// 
pub mod seq;
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
/// Contains the core of `feanor-math`'s (currently) minimalistic approach to matrices. In particular,
/// we use [`matrix::Submatrix`] and [`matrix::SubmatrixMut`] for matrices that don't own their data.
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
/// ```rust
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
/// Contains the trait [`homomorphism::Homomorphism`], [`homomorphism::CanHomFrom`] and
/// others that are the foundation of the homomorphism framework, that enables mapping
/// elements between different rings.
/// 
pub mod homomorphism;
///
/// Contains implementations of various iterators and combinators, like [`iters::powerset()`]
/// or [`iters::multi_cartesian_product`].
/// 
pub mod iters;
///
/// Contains the trait [`local::PrincipalLocalRing`] for principal ideal rings that additionally are local,
/// i.e. they have a unique maximal ideal (which then is generated by a single element).
/// 
pub mod local;
///
/// Contains the trait [`serialization::SerializableElementRing`] for rings whose elements can be serialized
/// by using `serde`. 
/// 
/// It also contains some utilities to simplify this, since it is usually not possible to use 
/// `#[derive(Serialize, Deserialize)]` to implement serialization - the reason is that serialization and
/// deserialization usually require access to the ring. Hence, we need to use [`serde::de::DeserializeSeed`],
/// but this is incompatible with `#[derive]`
/// 
pub mod serialization;
///
/// Contains a workaround for specialization.
/// 
pub mod specialization;
///
/// Contains the two traits [`reduce_lift::poly_eval::EvalPolyLocallyRing`] and
/// [`reduce_lift::poly_factor_gcd::PolyGCDLocallyDomain`] that formalize the assumptions required
/// to perform certain computations over a ring modulo prime ideals, and then reconstruct the element
/// from the resulting congruences.
/// 
pub mod reduce_lift;