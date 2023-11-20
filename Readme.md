
# feanor-math

This is a library for number theory, written completely in Rust. 
The idea is to provide a more modern alternative to projects like NTL or FLINT, however due to the large scope of those projects, the current implementation is still far away from that.
More concretely, we use modern language features - in particular the trait system - to provide a generic framework for rings, that makes it easy to nest them, or create custom rings, while still achieving high performance.
This is impossible in NTL, and while FLINT provides a framework for this, it is quite complicated and not very extensible.
From a user's point of view, we thus envision this library to be somewhat closer to high-level computer algebra systems like sagemath, but also have a strong static type system and very high performance.

## Current State

The current state is far away from this vision, as only a small set of most important algorithms have been implemented.
Furthermore, this library should still be considered to be in an alpha phase.
In particular, I will make changes to interfaces and implementations without warning, although I try to keep core interfaces (basically those in `crate::ring::*`) stable.
Furthermore, there might be bugs, and many implementations are not particularly optimized.
Nevertheless, I think this library can already be useful, and I regularly use it for various applications, including cryptography.

This library uses nightly Rust, and even unstable features like const-generics and specialization.

## A short introduction

The two fundamental traits in this crate are `RingBase` and `RingStore`.
The trait `RingBase` is designed for implementors, i.e. to define a ring structure as simply as possible.
The trait `RingStore` on the other hand is designed for using a ring, and is implemented by objects that provide access to an underlying `RingBase` object.
The reasons for this separation are explained further down this page.

## Features

The following rings are provided
 - The integer ring `Z`, as a trait `crate::integer::IntegerRing` with implementations for all primitive ints (`i8` to `i128`), an arbitrary-precision implementation `crate::rings::rust_bigint::RustBigintRing`, and an optional implementation using bindings to the heavily optimized library [mpir](https://github.com/wbhart/mpir)
 - The quotient ring `Z/nZ`, as a trait `crate::rings::zn::ZnRing` with four implementations. One where the modulus is small and known at compile-time `crate::rings::zn::zn_static::Zn`, an optimized implementation of Barett-reductions for moduli up to 41 bits `crate::rings::zn::zn_42::Zn`, a generic implementation of Barett-reductions for any modulus and any integer ring (including arbitrary-precision ones) `crate::rings::zn::zn_barett::Zn` and a residue-number-system implementation for highly composite moduli `crate::rings::zn::zn_rns::Zn`.
 - The polynomial ring `R[X]` over any base ring, as a trait `crate::rings::poly::PolyRing` with two implementations, one for densely filled polynomials `crate::rings::poly::dense_poly::DensePolyRing` and one for sparsely filled polynomials `crate::rings::poly::sparse_poly::SparsePolyRing`.
 - Finite-rank simple and free ring extensions, as a trait `crate::rings::extension::FreeAlgebra`, with an implementation based on polynomial division `crate::rings::extension::FreeAlgebraImpl`

The following algorithms are implemented
 - Fast Fourier transforms, including an optimized implementation of the Cooley-Tuckey algorithm for the power-of-two case, an implementation of the Bluestein algorithm for arbitrary lengths, and a factor FFT implementation (also based on the Cooley-Tuckey algorithm). The Fourier transforms work on all rings that have suitable roots of unity, in particular the complex numbers `C` and suitable finite rings `Fq`
 - An optimized variant of the Karatsuba algorithm for fast convolution
 - An implementation (currently not very optimized) of the Cantor-Zassenhaus algorithm to factor polynomials over finite fields
 - Lenstra's Elliptic Curve algorithm to factor integers (although the current implementation is very slow)
 - Miller-Rabin test to check primality of integers
 - A baby-step-giant-step and factorization-based algorithm to compute arbitrary discrete logarithms

## Using rings

As simple example of how to use the library, we implement Fermat primality test here
```rust
use feanor_math::homomorphism::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::rings::zn::zn_barett::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::finite::*;
use feanor_math::algorithms;
use oorandom;

fn fermat_is_prime(n: i64) -> bool {
    // the Fermat primality test is based on the observation that a^n == a mod n if n
    // is a prime; On the other hand, if n is not prime, we hope that there are many
    // a such that this is not the case. 
    // Note that this is not always the case, and so more advanced primality tests should 
    // be used in practice. This is just a proof of concept.

    let ZZ = StaticRing::<i64>::RING;
    let Zn = Zn::new(ZZ, n); // the ring Z/nZ

    // check for 6 random a whether a^n == a mod n
    let mut rng = oorandom::Rand64::new(0);
    for _ in 0..6 {
        let a = Zn.random_element(|| rng.rand_u64());
        let a_n = Zn.pow(Zn.clone_el(&a), n as usize);
        if !Zn.eq_el(&a, &a_n) {
            return false;
        }
    }
    return true;
}

assert!(algorithms::miller_rabin::is_prime(StaticRing::<i64>::RING, &91, 6) == fermat_is_prime(91));
```
If we want to support arbitrary rings of integers - e.g. `RustBigintRing::RING`, which is a simple
implementation of arbitrary-precision integers - we could make the function generic as

```rust
use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::zn_barett::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::finite::*;
use feanor_math::algorithms;

use oorandom;

fn fermat_is_prime<R>(ZZ: R, n: El<R>) -> bool 
    where R: RingStore, R::Type: IntegerRing
{
    // the Fermat primality test is based on the observation that a^n == a mod n if n
    // is a prime; On the other hand, if n is not prime, we hope that there are many
    // a such that this is not the case. 
    // Note that this is not always the case, and so more advanced primality tests should 
    // be used in practice. This is just a proof of concept.

    // ZZ is not guaranteed to be Copy anymore, so use reference instead
    let Zn = Zn::new(&ZZ, ZZ.clone_el(&n)); // the ring Z/nZ

    // check for 6 random a whether a^n == a mod n
    let mut rng = oorandom::Rand64::new(0);
    for _ in 0..6 {
        let a = Zn.random_element(|| rng.rand_u64());
        // use a generic square-and-multiply powering function that works with any implementation
        // of integers
        let a_n = Zn.pow_gen(Zn.clone_el(&a), &n, &ZZ);
        if !Zn.eq_el(&a, &a_n) {
            return false;
        }
    }
    return true;
}

// the miller-rabin primality test is implemented in feanor_math::algorithms, so we can
// check our implementation
let n = BigIntRing::RING.int_hom().map(91);
assert!(algorithms::miller_rabin::is_prime(BigIntRing::RING, &n, 6) == fermat_is_prime(BigIntRing::RING, n));
```
This function now works with any ring that implements `IntegerRing`, a subtrait of `RingBase`.

## Implementing rings

To implement a custom ring, just create a struct and add an `impl RingBase` and an `impl CanonicalIso<Self>` - that's it!
Assuming we want to provide our own implementation of the finite binary field F2, we could do it as follows.
```rust
use feanor_math::homomorphism::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;

#[derive(PartialEq)]
struct F2Base;

impl RingBase for F2Base {
   
    type Element = u8;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = (*lhs + rhs) % 2;
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        *lhs = (2 - *lhs) % 2;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = (*lhs * rhs) % 2;
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        // make sure that we handle negative numbers correctly
        (((value % 2) + 2) % 2) as u8
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        // elements are always represented by 0 or 1
        *lhs == *rhs
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", *value)
    }
}

// To properly use a ring, in addition to RingBase we have to implement CanonicalHom<Self> and
// CanonicalIso<Self>. This ensures that the ring works well with the canonical ring mapping
// framework, that later allows us to use functions like `cast()` or `coerce()`.
// In practice, we might also want to add implementations like `CanonicalHom<I> where I: IntegerRing`
// or CanonicalIso<feanor_math::rings::zn::zn_static::ZnBase<2, true>>.

impl CanonicalHom<F2Base> for F2Base {
    
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &Self) -> Option<Self::Homomorphism> {
        // a canonical homomorphism F -> F exists for all rings F of type F2Base, as
        // there is only one possible instance of F2Base
        Some(())
    }

    fn map_in(&self, from: &Self, el: Self::Element, hom: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl CanonicalIso<F2Base> for F2Base {
    
    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &Self) -> Option<Self::Isomorphism> {
        // a canonical isomorphism F -> F exists for all rings F of type F2Base, as
        // there is only one possible instance of F2Base
        Some(())
    }

    fn map_out(&self, from: &Self, el: Self::Element, hom: &Self::Homomorphism) -> Self::Element {
        el
    }
}

pub const F2: RingValue<F2Base> = RingValue::from(F2Base);

assert_el_eq!(&F2, &F2.int_hom().map(1), &F2.add(F2.one(), F2.zero()));
```

## `RingBase` vs `RingStore`

The trait `RingBase` is designed to provide a simple way of defining a ring structure.
As such, it provides a basic interface with all ring operations, like addition, multiplication and equality testing.
In many cases, variants of the basic operations are defined to make use of move-semantics and memory reuse.
However, they usually have default implementations, to simplify creating new rings.

On the other hand, a `RingStore` is any kind of object that gives access to an underlying `RingBase` object.
The trait comes with default implementations for all ring operations, that just delegate calls to the underlying `RingBase` object.
In normal circumstances, this trait should not be implemented for custom types.

The main power of this separation becomes apparent when we start nesting rings.
Say we have a ring type that builds on an underlying ring type, for example a polynomial ring `PolyRing<R>` over a base ring `R`.
In this case, `PolyRing` implements `RingBase`, but the underlying ring `R` is constrained to be `RingStore`.
As a result, types like `PolyRing<R>`, `PolyRing<&&R>` and `PolyRing<Box<R>>` can all be used equivalently, which provides a lot of flexibility, but still works both with expensive-to-clone rings and zero-sized rings.

# Performance

Generally speaking, I want performance to be a high priority in this crate.
However, I did not have the time so far to thoroughly optimize many of the algorithms.

## Tipps for achieving optimal performance

 - Use `lto = "fat"` in the `Cargo.toml` of your project. This is absolutely vital to enable inlining across crate boundaries, and can have a huge impact if you extensively use rings that have "simple" basic arithmetic - like `zn_42::Zn` or `primitive_int::StaticRing`.
 - Different parts of this library are at different stages of optimization. While I have spent some time on the FFT algorithms, for example integer factorization are currently relatively slow.
 - If you extensively use rings whose elements require dynamic memory allocation, be careful to choose good memory providers. This is currently still WIP. 
 - The default arbitrary-precision integer arithmetic is very slow currently. Use the feature "mpir" together with an installation of the [mpir](https://github.com/wbhart/mpir) library if you heavily use arbitrary-precision integers. 

# Design decisions

Here I document - mainly for myself - some of the more important design decisions.

## `RingStore`

Already talked about this.
Mainly, this is the result of the headache I got in the first version of this crate, when trying to map elements from `RingWrapper<Zn<IntRing>>` to `RingWrapper<Zn<&IntRing>>` or similar.

## Elements referencing the ring

It seems to be a reasonably common requirement that elements of a ring may contain references to the ring.
For example, this is the case if the element uses memory that is managed by a memory pool of the ring.
In other words, we would define `RingBase` as
```rust,ignore
trait RingBase {

    type Element<'a> where Self: 'a;

    ...
}
```
However, this conflicts with another design decision:
We want to be able to nest rings, and allow the nested rings to be owned (not just borrowed).
If we allow ring-referential elements, this now prevents us from defining rings that store the nested ring and some of its elements.
For example, an implementation of Z/qZ might look like
```rust,ignore
struct Zn<I: IntegerRingStore> {
    integer_ring: I,
    modulus: El<I>
}
```
If `El<I>` may contain a reference to `I`, then this struct is self-referential, causing untold trouble.

Now it seems somewhat more natural to forbid owning nested rings instead of ring-referential elements, but this then severly limits which rings can be returned from functions.
For example we might want a function to produce Fq with a mempool-based big integer implementation like
```rust,ignore
fn galois_field(q: i64, exponent: usize) -> RingExtension<ZnBarett<MempoolBigIntRing>> {
    ...
}
```
This would be impossible, as well as many other use cases.
On the other hand, it is simpler to perform runtime checks in case of static lifetime analysis if we want to have ring-referential elements.
This is maybe slightly unnatural, but very usable.
And really, if elements need a reference to the ring, they won't be small and the arithmetic cost will greatly exceed the runtime management cost.