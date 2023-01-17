#![allow(soft_unstable)]
#![feature(test)]
#![feature(specialization)]
#![feature(associated_type_bounds)]

extern crate test;

#[macro_use]
pub mod ring;
pub mod divisibility;
pub mod euclidean;
pub mod ordered;
pub mod primitive;
pub mod integer;
pub mod vector;
pub mod algorithms;
pub mod rings;