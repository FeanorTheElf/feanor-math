#![allow(soft_unstable)]
#![allow(non_snake_case)]
#![feature(test)]
#![feature(specialization)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]
#![feature(new_uninit)] 
#![feature(core_intrinsics)]

extern crate test;

// used to represent that a const generic computation works
pub struct Expr<const VALUE: usize>;
pub trait Exists {}
impl<T: ?Sized> Exists for T {}

pub mod mempool;
#[macro_use]
pub mod ring;
pub mod delegate;
pub mod vector;
pub mod divisibility;
pub mod field;
pub mod euclidean;
pub mod ordered;
pub mod primitive_int;
pub mod integer;
pub mod algorithms;
pub mod rings;
pub mod wrapper;