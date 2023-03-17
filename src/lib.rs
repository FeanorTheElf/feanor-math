#![allow(soft_unstable)]
#![feature(test)]
#![feature(specialization)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]
#![feature(associated_type_bounds)]

extern crate test;

// used to represent that a const generic computation works
pub struct Expr<const VALUE: usize>;
pub trait Exists {}
impl<T: ?Sized> Exists for T {}

#[macro_use]
pub mod ring;
pub mod delegate;
pub mod vector;
pub mod vectors;
pub mod divisibility;
pub mod field;
pub mod euclidean;
pub mod ordered;
pub mod primitive_int;
pub mod integer;
pub mod algorithms;
pub mod rings;