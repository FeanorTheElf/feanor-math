
# feanor-math

This is a library for number theory, written completely in Rust. 
The idea is to provide a more modern alternative to projects like NTL or FLINT, however due to the large scope of those projects, the current implementation is still far away from that.
In particular, we use modern language features - in particular the trait system - to provide a generic framework for rings, that makes it easy to nest them, or create custom rings, while still achieving high performance.
This is impossible in NTL, and while FLINT provides a framework for this, it is quite complicated and not extensible.
From a user's point of view, we thus envision this library to be somewhat closer to high-level computer algebra systems like sage, but also have a strong static type system and native performance.

The current state is far away from this vision, as only a small set of most important algorithms have been implemented.
Furthermore, the provided algorithms are not yet as optimized as their counterparts in other systems.
Nevertheless, I think this library can already be useful, and I regularly use it for various applications, including cryptography.

## A short introduction

The two fundamental traits in this crate are `RingBase` and `RingStore`.
The trait `RingBase` is designed for implementors, i.e. to define a ring structure as simply as possible.
The trait `RingStore` on the other hand is designed for using a ring, and is implemented by objects that provide access to an underlying `RingBase` object.
The reasons for this separation are explained further down this page.

## Using rings

As simple example of how to use the library, we implement Fermat primality test here
```rust
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::rings::zn::zn_dyn::*;
use feanor_math::rings::zn::*;
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
        let a_n = Zn.pow(&a, n as usize);
        if !Zn.eq(&a, &a_n) {
            return false;
        }
    }
    return true;
}

// the miller-rabin primality test is implemented in feanor_math::algorithms, so we can
// check our implementation
assert!(algorithms::miller_rabin::is_prime(StaticRing::<i64>::RING, &91, 6) == fermat_is_prime(91));
```
If we want to support arbitrary rings of integers - e.g. `DefaultBigIntRing::RING`, which is a simple
implementation of arbitrary-precision integers - we could make the function generic as

```rust
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::rings::bigint::*;
use feanor_math::rings::zn::zn_dyn::*;
use feanor_math::rings::zn::*;
use feanor_math::algorithms;

use oorandom;

fn fermat_is_prime<R: IntegerRingStore>(ZZ: R, n: El<R>) -> bool {
    // the Fermat primality test is based on the observation that a^n == a mod n if n
    // is a prime; On the other hand, if n is not prime, we hope that there are many
    // a such that this is not the case. 
    // Note that this is not always the case, and so more advanced primality tests should 
    // be used in practice. This is just a proof of concept.

    // ZZ is not guaranteed to be Copy anymore, so use reference instead
    let Zn = Zn::new(&ZZ, n.clone());

    // check for 6 random a whether a^n == a mod n
    let mut rng = oorandom::Rand64::new(0);
    for _ in 0..6 {
        let a = Zn.random_element(|| rng.rand_u64());
        // use a generic square-and-multiply powering function that works with any implementation
        // of integers
        let a_n = Zn.pow_gen(&a, &n, &ZZ);
        if !Zn.eq(&a, &a_n) {
            return false;
        }
    }
    return true;
}

// the miller-rabin primality test is implemented in feanor_math::algorithms, so we can
// check our implementation
let n = DefaultBigIntRing::RING.from_z(91);
assert!(algorithms::miller_rabin::is_prime(DefaultBigIntRing::RING, &n, 6) == fermat_is_prime(DefaultBigIntRing::RING, n));
```
This function now works with any ring that implements `IntegerRing`, a subtrait of `RingBase`.

## Implementing rings

To implement a custom ring, just create a struct and add an `impl RingBase` and an `impl CanonicalIso<Self>` - that's it!
Assuming we want to provide our own implementation of the finite binary field F2, we could do it as follows.
```rust
use feanor_math::ring::*;

struct F2Base;

impl RingBase for F2Base {
   
    type Element = u8;

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = (*lhs + rhs) % 2;
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        *lhs = (2 - *lhs) % 2;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = (*lhs * rhs) % 2;
    }
    
    fn from_z(&self, value: i32) -> Self::Element {
        // make sure that we handle negative numbers correctly
        (((value % 2) + 2) % 2) as u8
    }

    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
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

assert!(F2.eq(&F2.from_z(1), &F2.add(F2.one(), F2.zero())));
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