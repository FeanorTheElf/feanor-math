#![allow(soft_unstable)]
#![feature(test)]
#![feature(specialization)]
#![feature(generic_const_exprs)]
#![feature(associated_type_bounds)]

extern crate test;

// used to represent that a const generic computation works
pub struct Expr<const VALUE: usize>;
pub trait Exists {}
impl<T: ?Sized> Exists for T {}

#[macro_use]
pub mod ring;
pub mod vector;
pub mod vectors;
pub mod divisibility;
pub mod field;
pub mod euclidean;
pub mod ordered;
pub mod primitive;
pub mod integer;
pub mod algorithms;
pub mod rings;