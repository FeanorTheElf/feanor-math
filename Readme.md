# feanor-math

This is a library for number theory, written completely in Rust. 
The idea is to provide a more modern alternative to projects like NTL or FLINT, however due to the large scope of those projects, the current implementation is still far away from that.
More concretely, we use modern language features - in particular the trait system - to provide a generic framework for rings, that makes it easy to nest them, or create custom rings, while still achieving high performance.
This is impossible in NTL, and while FLINT provides a framework for this, it is quite complicated and (since FLINT is written in C) relies on runtime polymorphism - which prevents thorough type checking.
From a user's point of view, we thus envision this library to be somewhat closer to high-level computer algebra systems like sagemath, but also have a strong static type system and very high performance.

## Current State

The current state is far away from this vision, as only some of most important algorithms have been implemented.
Furthermore, there might be bugs, and many implementations are not particularly optimized.
Nevertheless, I think this library can already be useful, and I regularly use it for various applications, including cryptography.
Note that I will try to keep the interfaces stable, unless they are marked with `#[stability::unstable]`.

This library uses nightly Rust, ~~and even unstable features like const-generics and specialization~~ - this caused too much of a headache, so I removed those uses again.

## A short introduction

The project started with the idea that algorithmic number theory usually focuses on the rings it works with.
Hence, it makes sense to start with a trait `Ring`, and then create a hierarch of subtraits for additional properties (say `Domain`, `Field` or `EuclideanRing`).
Algorithms working on rings can then state clearly what properties of the underlying ring they require, and be generic in the rings they work with.
Furthermore, rings that build on other rings (e.g. polynomial rings or algebraic extensions) can declare their own properties (i.e. implement the traits) depending on the properties of the base ring.
In practice, once we heavily follow this approach, we soon run into limitations of the type system (borrowing vs owning base rings, conflicting blanket impls, lack of specialization, ...).
To mitigate this, instead of a single trait `Ring` I have introduced two traits `RingBase` and `RingStore`, as explained below.
Furthermore, blanket implementations are used sparingly, only when they can actually cover a very large class of rings.
In the end, though not perfect, this turns out to work quite well.

## Features

The following rings are provided
 - The integer ring `Z`, as a trait [`crate::integer::IntegerRing`] with implementations for all primitive ints (`i8` to `i128`) given by [`crate::primitive_int::StaticRing`], an arbitrary-precision implementation [`crate::rings::rust_bigint::RustBigintRing`], and an optional implementation using bindings to the heavily optimized library [mpir](https://github.com/wbhart/mpir) (enable with `features=mpir`) given by [`crate::rings::mpir::MPZ`].
 - The quotient ring `Z/nZ`, as a trait [`crate::rings::zn::ZnRing`] with four implementations. One where the modulus is small and known at compile-time [`crate::rings::zn::zn_static::Zn`], an optimized implementation of Barett-reductions for moduli somewhat smaller than 64 bits [`crate::rings::zn::zn_64::Zn`], a generic implementation of Barett-reductions for any modulus and any integer ring (optimized for arbitrary-precision ones) [`crate::rings::zn::zn_big::Zn`] and a residue-number-system implementation for highly composite moduli [`crate::rings::zn::zn_rns::Zn`].
 - The polynomial ring `R[X]` over any base ring, as a trait [`crate::rings::poly::PolyRing`] with two implementations, one for densely filled polynomials [`crate::rings::poly::dense_poly::DensePolyRing`] and one for sparsely filled polynomials [`crate::rings::poly::sparse_poly::SparsePolyRing`].
 - Finite-rank simple and free ring extensions, as a trait [`crate::rings::extension::FreeAlgebra`], with an implementation based on polynomial division [`crate::rings::extension::extension_impl::FreeAlgebraImpl`]. In particular, this includes finite/galois fields and number fields.
 - Multivariate polynomial rings `R[X1, ..., XN]` over any base ring, as the trait [`crate::rings::multivariate::MultivariatePolyRing`] and one implementation [`crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl`] based on a sparse representation using ordered vectors.
 - Combining the above, you can get Galois fields (easily available using [`crate::rings::extension::galois_field::GaloisField`]) or arbitrary number fields (they also have the wrapper type [`crate::rings::extension::number_field::NumberField`] for a more convenient interface).

The following algorithms are implemented
 - Fast Fourier transforms, including an optimized implementation of the Cooley-Tukey algorithm for the power-of-two and power-of-three cases, an implementation of the Bluestein algorithm for arbitrary lengths, and a factor FFT implementation (also based on the Cooley-Tukey algorithm). The Fourier transforms work on all rings that have suitable roots of unity, in particular the complex numbers `C` and suitable finite rings `Fq`.
 - Various algorithms for computing convolutions, including Karatsuba's algorithm and FFT-based methods.
 - An implementation of the Cantor-Zassenhaus algorithm to factor polynomials over finite fields.
 - Factoring polynomials over the rationals/integers (using Hensel lifting) and over number fields.
 - Lenstra's Elliptic Curve algorithm to factor integers.
 - LLL algorithm for lattice reduction.
 - The Finke-Pohst lattice point enumeration algorithm.
 - Basic linear algebra over various rings, including finite integral extensions of principal ideal rings.
 - Miller-Rabin test to check primality of integers.
 - A baby-step-giant-step and factorization-based algorithm to compute arbitrary discrete logarithms.
 - Buchberger's algorithm (F4-style) to compute Gröbner basis.

Unfortunately, operations with polynomials over infinite rings (integers, rationals, number fields) are currently very slow, since efficient implementation require a lot of care to prevent coefficient blowup, which I did not have time or need to invest.

### Most important missing features

 - Optimization and better handling of coefficient blowup during algorithms over infinite rings (in particular `Q`, `Z` and number fields). 
 - More linear algebra operations, apart from determinants and solving linear systems.
 - Higher-level number theory algorithms, e.g. computing maximal orders.
 - An implementation of general fraction fields (implementations of `Q` and the ring of rational functions already exist).

## SemVer

In version `1.x.x` the library used an inconsistent version scheme.
This is now fixed, and all versions from `2.x.x` onwards use semantic versioning, as described in the [Cargo book](https://doc.rust-lang.org/cargo/reference/resolver.html#semver-compatibility).
Note that all items marked with the annotation `#[stability::unstable]` from the rust library [`stability`](https://docs.rs/stability/latest/stability/index.html) are exempt from semantic version.
In other words, breaking changes in the interface of these structs/traits/functions will only increment the minor version number.
Note that these are not visible to other crates at all, unless the feature `unstable-enable` is active.

# Similar Projects

I have recently been notified of [Symbolica](https://symbolica.io/), which takes a similar approach to computer algebra in Rust.
As opposed to `feanor-math` which is mainly built for number theory, its main focus are computations with multivariate polynomials (including floating point number), and in this area, Symbolica does perform better than `feanor-math`.
If this suits your use case better, check it out!
Personally, I think it is an amazing project as well.

# Examples

## Linear algebra on finite fields

The simplest way of using `feanor-math` is by running a provided algorithm on a provided ring.
For example, we can solve a linear system over a finite field as follows:
```rust
use feanor_math::homomorphism::*;
use feanor_math::rings::extension::galois_field::*;
use feanor_math::assert_el_eq;
use feanor_math::rings::extension::*;
use feanor_math::matrix::*;
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::algorithms::linsolve::*;

// create the field F25 and the embedding ZZ -> F25
let F25 = GaloisField::new(5, 2);
let ZZ_to_F25 = F25.int_hom();

// we create matrices [ 1  1 ] and [ 0 ]
//                    [ 1  θ ]     [ 1 ]
// θ is chosen as the "canonical generator" of F25
let mut lhs = OwnedMatrix::from_fn(2, 2, |i, j| F25.pow(F25.canonical_gen(), i * j));
let mut rhs = OwnedMatrix::from_fn(2, 1, |i, _j| ZZ_to_F25.map(i as i32));

// solve the linear system
let mut result = OwnedMatrix::zero(2, 1, &F25);
F25.solve_right(lhs.data_mut(), rhs.data_mut(), result.data_mut()).assert_solved();

// verify the first row of the equation; note that we cannot use arithmetic operators + or *,
// since addition resp. multiplication are always relative to a ring, and a ring element does
// not generally know which ring it belongs to (if you want to store ring elements together with
// a reference to the ring, see `RingElementWrapper`)
println!("Solution is [{}, {}]", F25.format(result.at(0, 0)), F25.format(result.at(1, 0)));
assert_el_eq!(&F25, F25.zero(), F25.add_ref(result.at(0, 0), result.at(1, 0)));
```

## Solving polynomial equations over Zn

Another example of the above type is solving of equations over supported fields.
Note that this code currently requires unstable features, i.e. activating the feature `unstable-enable`.
```rust
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::field::*;
use feanor_math::primitive_int::*;
use feanor_math::algorithms::buchberger::*;
use feanor_math::rings::multivariate::*;
use feanor_math::rings::multivariate::multivariate_impl::*;
use feanor_math::algorithms::poly_factor::*;
use feanor_math::rings::poly::dense_poly::*;
use feanor_math::seq::*;
use feanor_math::divisibility::*;
use feanor_math::pid::*;
use feanor_math::rings::poly::*;

// first, we create multiple rings: the prime field F7, the bivariate polynomial ring F7[X, Y] and the univariate 
//                                  polynomial ring F7[T]
let F7 = Zn::new(7).as_field().ok().unwrap();
let F7XY = MultivariatePolyRingImpl::new(&F7, 2);
let F7T = DensePolyRing::new(&F7, "T");

// we create elements of the ring; the easiest way is to use `with_wrapped_indeterminates()`, which temporarily
// wraps the generators with a `RingElementWrapper`, thus allowing us to use + and * for arithmetic operations.
let [f1, f2] = F7XY.with_wrapped_indeterminates(|[X, Y]| [X * X * Y - 1, X * Y - 2]);

// now compute a groebner basis
let groebner_basis_degrevlex = buchberger_simple(&F7XY, vec![F7XY.clone_el(&f1), F7XY.clone_el(&f2)], DegRevLex);
// if the groebner basis contains a unit, the system is unsolvable
assert!(groebner_basis_degrevlex.iter().all(|f| F7XY.appearing_indeterminates(f).len() > 0), "system has no solution");
// now compute a lex-groebner basis; note that it still makes sense to do this on top of the degrevlex groebner
// basis, as it will drastically speed up the computation
let mut groebner_basis_lex = buchberger_simple(&F7XY, groebner_basis_degrevlex, Lex);

// sort descending by leading terms, which means the system will have the shape [f1(X, Y), ..., fr(X, Y), g(Y)]
groebner_basis_lex.sort_unstable_by(|f, g| Lex.compare(&F7XY, F7XY.LT(f, Lex).unwrap().1, F7XY.LT(g, Lex).unwrap().1).reverse());

// we can now solve by choosing `y` as a root of `g` and `x` as a joint root of `fi(X, y)`, i.e. the `fi` with `y` "plugged in"
let poly_in_y = F7XY.evaluate(&groebner_basis_lex.pop().unwrap(), [F7T.zero(), F7T.indeterminate()].clone_ring_els(&F7T), F7T.inclusion());
let (poly_in_y_factorization, _) = <_ as FactorPolyField>::factor_poly(&F7T, &poly_in_y);
let y = poly_in_y_factorization.into_iter().filter_map(|(f, _)| if F7T.degree(&f).unwrap() != 1 { None } else { Some(F7.negate(F7.div(F7T.coefficient_at(&f, 0), F7T.coefficient_at(&f, 1)))) }).next().unwrap();

// we found `y`! now plug in `y` into the `fi(X, Y)`, take their gcd (as univariate polynomials in `X`), and find a root `x`
let poly_in_x = groebner_basis_lex.into_iter().fold(F7T.zero(), |f, g| F7T.ideal_gen(&f, &F7XY.evaluate(&g, [F7T.indeterminate(), F7T.inclusion().map(y)].clone_ring_els(&F7T), F7T.inclusion())));
let (poly_in_x_factorization, _) = <_ as FactorPolyField>::factor_poly(&F7T, &poly_in_x);
let x = poly_in_x_factorization.into_iter().filter_map(|(f, _)| if F7T.degree(&f).unwrap() != 1 { None } else { Some(F7.negate(F7.div(F7T.coefficient_at(&f, 0), F7T.coefficient_at(&f, 1)))) }).next().unwrap();

// check the solution
assert_el_eq!(F7, F7.zero(), F7XY.evaluate(&f1, [x, y].clone_ring_els(F7), F7.identity()));
assert_el_eq!(F7, F7.zero(), F7XY.evaluate(&f2, [x, y].clone_ring_els(F7), F7.identity()));
```
A similar approach also works over the rationals.

## Using rings

This library is not "just" a collection of algorithms on certain rings, but is supposed to be seamlessly extensible.
In particular, a user should be able to write high-performance algorithms that directly operate on provided rings, or rings that work seamlessly with provided algorithms.
First, we show demonstrate how one could implement the Fermat primality test.
```rust
use feanor_math::homomorphism::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::rings::zn::zn_big::*;
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
    let mut rng = oorandom::Rand64::new(1);
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
If we want to support arbitrary rings of integers - e.g. `BigIntRing::RING`, which is a simple
implementation of arbitrary-precision integers - we could make the function generic as

```rust
use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::zn_big::*;
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
    let mut rng = oorandom::Rand64::new(1);
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

To implement a custom ring, just create a struct and add an `impl PartialEq` and an `impl RingBase` - that's it!
Assuming we want to provide our own implementation of the finite binary field F2, we could do it as follows.
```rust
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;

#[derive(PartialEq)]
struct F2Base;

// this is a minimal implementation of `RingBase`. Note that `RingBase` contains many more
// functions with default implementations, that can be overwritten to provide better performance
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

    fn characteristic<I>(&self, ZZ: I) -> Option<El<I>>
        where I: RingStore + Copy, I::Type: IntegerRing
    {
        Some(ZZ.int_hom().map(2))
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _: EnvBindingStrength) -> std::fmt::Result {
        write!(out, "{}", *value)
    }
}

// in a real scenario, we might want to implement more traits like `ZnRing`, `DivisibilityRing`
// or `Field`; Also it might be useful to provide canonical homomorphisms by implementing `CanHomFrom`,
// in particular the self-isomorphisms `F2Base: CanHomFrom<F2Base>` and `F2Base: CanIsoFromTo<F2Base>` 
// might be useful

pub const F2: RingValue<F2Base> = RingValue::from(F2Base);

assert_el_eq!(F2, F2.int_hom().map(1), F2.add(F2.one(), F2.zero()));
```

## Both together

One of the main goals of this trait was to make it easy to nest rings, so implement a functor on the category of rings.
Classical examples are polynomial rings `R[X]` that exist for any ring `R`.
Since in that case we are both using and implementing rings, we should use both sides of the framework.
For example, a simple polynomial ring implementation could look like this.
```rust
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::integer::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use std::cmp::{min, max};

pub struct MyPolyRingBase<R: RingStore> {
    base_ring: R
}

// We give the actual implementation some name postfixed by `Base`, and
// then define a type alias to the simplest `RingStore` wrapping the implementation.
// This is a pattern that is often used throughout the library, since it means that
// users don't always have to mention the `RingValue` when they use the ring.
pub type MyPolyRing<R> = RingValue<MyPolyRingBase<R>>;

impl<R: RingStore> PartialEq for MyPolyRingBase<R> {

    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<R: RingStore> RingBase for MyPolyRingBase<R> {

    // in a real implementation, we might want to wrap this in a newtype, to avoid
    // exposing the vector interface (although exposing that interface might be intended - 
    // the crate does not judge whether this is a good idea)
    type Element = Vec<El<R>>;

    fn clone_el(&self, el: &Self::Element) -> Self::Element {
        el.iter().map(|x| self.base_ring.clone_el(x)).collect()
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        for i in 0..min(lhs.len(), rhs.len()) {
            self.base_ring.add_assign_ref(&mut lhs[i], &rhs[i]);
        }
        for i in lhs.len()..rhs.len() {
            lhs.push(self.base_ring.clone_el(&rhs[i]))
        }
    }

    fn negate_inplace(&self, val: &mut Self::Element) {
        for x in val {
            self.base_ring.negate_inplace(x);
        }
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        // this is just for demonstration purposes - note that the length of the vectors would slowly increase,
        // even if the degree of the polynomials doesn't
        let mut result = (0..(lhs.len() + rhs.len())).map(|_| self.base_ring.zero()).collect::<Vec<_>>();
        for i in 0..lhs.len() {
            for j in 0..rhs.len() {
                self.base_ring.add_assign(&mut result[i + j], self.base_ring.mul_ref(&lhs[i], &rhs[j]));
            }
        }
        return result;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.mul_ref(lhs, &rhs);
    }

    fn from_int(&self, x: i32) -> Self::Element {
        vec![self.base_ring.int_hom().map(x)]
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        let zero = self.base_ring.zero();
        for i in 0..max(lhs.len(), rhs.len()) {
            if !self.base_ring.eq_el(lhs.get(i).unwrap_or(&zero), rhs.get(i).unwrap_or(&zero)) {
                return false;
            }
        }
        return true;
    }

    fn is_commutative(&self) -> bool {
        self.base_ring.is_commutative()
    }

    fn is_noetherian(&self) -> bool {
        // by Hilbert's basis theorem
        self.base_ring.is_noetherian()
    }

    fn is_approximate(&self) -> bool {
        self.base_ring.get_ring().is_approximate()
    }

    fn dbg_within(&self, val: &Self::Element, f: &mut std::fmt::Formatter, env: EnvBindingStrength) -> Result<(), std::fmt::Error> {
        // this is just for demonstration purposes - note that this prints zero coefficients, and
        // does not print parenthesis even when `env > EnvBindingStrength::Sum`
        for i in 0..(val.len() - 1) {
            write!(f, "{} * X^{} + ", self.base_ring.format(&val[i]), i)?;
        }
        write!(f, "{} * X^{}", self.base_ring.format(val.last().unwrap()), val.len() - 1)
    }

    fn characteristic<I>(&self, ZZ: I) -> Option<El<I>>
        where I: RingStore + Copy, I::Type: IntegerRing
    {
        self.base_ring.get_ring().characteristic(ZZ)
    }
}

// in a real implementation, we definitely should implement also `feanor_math::rings::poly::PolyRing`, and
// possibly other traits (`CanHomFrom<other polynomial rings>`, `RingExtension`, `DivisibilityRing`, `EuclideanRing`)

let base = Zn::new(17);
// we do not use the `RingBase`-implementor directly, but wrap it in a `RingStore`;
// while it is possible to use "raw" `RingBase`s, it is usually recommended to use them
// through `RingStore`s instead, since `RingStore` provides a slightly higher-level interface
let ring = MyPolyRing::from(MyPolyRingBase { base_ring: base });
let x = vec![base.zero(), base.one()];
let f = ring.add_ref(&x, &ring.int_hom().map(8));
let g = ring.add_ref(&x, &ring.int_hom().map(7));
let h = ring.add(ring.mul_ref(&x, &x), ring.add_ref(&ring.mul_ref(&x, &ring.int_hom().map(-2)), &ring.int_hom().map(5)));
assert_el_eq!(ring, h, ring.mul(f, g));
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

## Conventions and best practices

 - Functions that take a ring as parameter should usually be generic and take `R` where `R: RingStore`.
   In cases where we would usually take the ring by reference, prefer instead to take `R` by value with `R: RingStore + Copy`.
   Since for `R: RingStore` also `&R: RingStore`, this just makes the interface more general.
   The main exception for this rule is if you want to use dynamic dispatch (or for some other reason avoid generic parameters), in which case you should take `&R` where `R` is some fixed type with `R: RingBase`.
 - Rings that wrap a base ring (like `MyRing<BaseRing: RingStore>`) should not implement `Copy`, unless both `BaseRing: Copy` and `El<BaseRing>: Copy`.
   There are some cases where I had previously added `#[derive(Copy)]`, which then made adding struct members of base ring elements to the ring a breaking change.
 - Equality (via `PartialEq`) of rings implies that they are "the same" and their elements can be used interchangeably without conversion.
   Being (canonically) isomorphic (via [`crate::homomorphism::CanIsoFromTo`]) implies that two rings are "the same", but their elements might use different internal
   format. Using the functions [`crate::homomorphism::CanIsoFromTo`], they can be converted between both rings. For more info, see also [`crate::ring::RingBase`].
 - Algorithms are exposed in one of three ways: As global function, as subtrait of [`crate::ring::RingBase`] or as their own trait, generic in some ring type `R: RingBase`.
   The first should be seen as the default, while the second way can be used if the implementation of the algorithm depends heavily on the ring in question (e.g. factoring polynomials, see [`crate::algorithms::poly_factor::FactorPolyField`]).
   Furthermore, in some situations, one might want algorithms to store data between multiple executions, and/or make higher-level algorithms (or rings) configurable with a concrete implementation of a used sub-algorithm (strategy pattern).
   In these cases, it makes sense to define a new trait for objects representing an implementation of the algorithm (e.g. computing convolutions, see [`crate::algorithms::convolution::ConvolutionAlgorithm`]).

# Performance

Generally speaking, I want performance to be a high priority in this crate.
However, I did not have the time so far to thoroughly optimize many of the algorithms.

## Tipps for achieving optimal performance

 - Use `lto = "fat"` in the `Cargo.toml` of your project. This is absolutely vital to enable inlining across crate boundaries, and can have a huge impact if you extensively use rings that have "simple" basic arithmetic - like [`crate::rings::zn::zn_64::Zn`] or [`crate::primitive_int::StaticRing`].
 - Different parts of this library are at different stages of optimization. While I have spent some time on finite fields and the FFT algorithms, for example working over rationals is currently somewhat slow.
 - If you extensively use rings whose elements require dynamic memory allocation, be careful to use a custom allocator, e.g. one from [`feanor-mempool`](https://github.com/FeanorTheElf/feanor-mempool).
 - The default arbitrary-precision integer arithmetic is somewhat slow. Use the feature "mpir" together with an installation of the [mpir](https://github.com/wbhart/mpir) library if you heavily use arbitrary-precision integers. 
 - Write your code so that it is easy to replace ring types and other generic parameters! `feanor-math` often provides different implementations of the same thing, but with different performance characteristics (e.g. [`crate::rings::poly::sparse_poly::SparsePolyRing`] vs [`crate::rings::poly::dense_poly::DensePolyRing`], [`crate::algorithms::convolution::KaratsubaAlgorithm`] vs [`crate::algorithms::convolution::ntt::NTTConvolution`] and so on). If your code makes it easy to replace one with the other, you can just experiment which version gives the best performance. `feanor-math` supports that by exposing basically all interfaces through traits.