# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`feanor-math` is a Rust library for number theory and computer algebra (arithmetic in rings, polynomial factorization, FFT, lattice reduction, Gröbner bases, etc.). It targets the same problem space as NTL/FLINT but uses Rust's trait system to give a SageMath-like generic API with static type checking and high performance. Currently version 4.0.0; the active development branch is `v4.0.0_2`.

## Toolchain

Pinned nightly: `nightly-2026-03-01` via `rust-toolchain.toml`. Roughly two dozen `#![feature(...)]` flags are enabled in `src/lib.rs` (specialization, allocator_api, etc.) — pinning matters.

## Common commands

```bash
cargo test                                          # standard testing command
cargo test --release -- --nocapture --ignored       # run expensive, ignored tests (which often serve a dual purpose as benchmarks)
cargo test --doc                                    # run doctests, usually this should be with feature `unstable-enable` since many doctests won't compile without unstable features
cargo fm t                                           # auto-formatting
```

Feature flags worth knowing:
  mpir                        — link to mpir for fast bignum (requires RUSTFLAGS="-L ...")
  ndarray                     — minimal ndarray matrix interop
  generic_tests               — exposes test_*_axioms drivers for downstream crates
  unstable-enable             — exposes #[stability::unstable] items to other crates
  unroll_recursive_algorithms — heavier codegen for Strassen/Karatsuba

## Architecture

The library is built around a deliberate two-trait split — read `Readme.md` (long, accurate, embedded as the crate-level rustdoc via `#![doc = include_str!("../Readme.md")]`) before making structural changes. Key points:

### `RingBase` vs `RingStore`

- **`RingBase`** (in `src/ring.rs`) is what ring authors implement. It defines `Element`, `add_assign`, `mul_assign`, `from_int`, `eq_el`, `fmt_el_within`, etc. Many methods have default implementations; override them for performance.
- **`RingStore`** is a wrapper that *holds* a `RingBase` (by value, ref, `Box`, `Arc`, ...) and provides the user-facing API. `RingValue<R>` is the canonical `RingStore`. Custom `RingStore` impls are rare — almost no one should write one.
- The split exists so that nested rings like `PolyRing<R>`, `PolyRing<&R>`, `PolyRing<Box<R>>` interoperate without bleeding `Clone` requirements through every generic bound. The pattern across the crate is: define `FooBase` (implements `RingBase`) and a type alias `Foo = RingValue<FooBase>`.

### Trait hierarchy

`RingBase` → `DivisibilityRing` → `PrincipalIdealRing` → `EuclideanRing` / `Field`, with orthogonal traits `IntegerRing`, `ZnRing`, `FreeAlgebra`, `PolyRing`, `MultivariatePolyRing`, `FiniteRing`, `OrderedRing`, `ApproxRealField`. Algorithms declare exactly the algebraic structure they need.

### Conventions (from `Readme.md` "Conventions and best practices")

- Functions taking a ring are usually generic `R: RingStore`. When you'd otherwise take by reference, prefer `R: RingStore + Copy` taken by value — `&R: RingStore` holds for any `R: RingStore`, so this is strictly more general. Use `&dyn RingTrait<Element = _>` only when you specifically want dynamic dispatch.
- Wrapper rings `MyRing<BaseRing: RingStore>` should *not* derive `Copy` unless `BaseRing: Copy` and `El<BaseRing>: Copy` — otherwise adding any owned base-ring element later is a silent breaking change.
- `PartialEq` on rings means "same ring, elements interchangeable without conversion". `CanIsoFromTo` means "canonically isomorphic, but elements may need conversion". Distinct concepts.
- Tracing: emit at `TRACE` level only (call sites use `span!`/`event!`). Place tracing on the function that *implements* the algorithm, not on trait-method delegations.
- For multi-step constructors: `new()` makes default choices, `new_with_xyz()` configures specifics, `create()` is full customization and always `#[stability::unstable]`.
- "Self-contained" objects store a `RingStore` and are generic in `R: RingStore`. "Ring-dependent" objects store only elements and take the ring as a parameter to each method, generic in `T` with method-level bounds `R: RingStore, R::Type: RingBase<Element = T>`. Pick deliberately; the codebase currently leans toward self-contained for higher-level objects.

### Homomorphisms

Homomorphisms come in both "ring-dependent" and "self-contained" format.
 - `src/homomorphism.rs` defines traits `CanHomFrom`, `CanIsoFromTo` to implement for rings that have canonical homomorphisms/isomorphisms to other rings; these are "ring-dependent", and using a homomorphism through these always requires passing both domain and codomain rings as parameters for both functions
 - `src/homomorphism.rs` also defines the trait `Homomorphism`, with implementations `Identity` (get via `.identity()`), `Inclusion` (get via `.inclusion()`), `IntHom` (get via `.int_hom()`), `CanHom` (get via `.can_hom()` if rings implement `CanHomFrom`). These are "self-contained", and store both domain and codomain, accessible with `.domain()` and `.codomain()`.
 - `CanHomFrom` is *not* universally implementable due to coherence - the conrete implementations (in particular blanket implementations) have to be carefully chosen for each ring on a case-by-case basis

### Module map

- `src/ring.rs`, `src/homomorphism.rs`, `src/divisibility.rs`, `src/field.rs`, `src/pid.rs`, `src/integer.rs`, `src/ordered.rs`, `src/finite.rs` — core traits.
- `src/primitive_int.rs` — `StaticRing<i8..i128>`.
- `src/rings/` — concrete rings: `zn` (four `Z/nZ` impls), `poly` (dense/sparse univariate), `multivariate`, `extension` (free algebras, `GaloisField`, `NumberField`), `rust_bigint`, `mpir`, `rational`, `fraction`, `local`, `float_complex`, `approx_real`, `direct_power`.
- `src/algorithms/` — concrete algorithms: `fft`, `convolution` (Karatsuba, NTT), `poly_factor` (Cantor–Zassenhaus, finite/rational/number-field), `poly_gcd`, `hensel`, `ec_factor`, `miller_rabin`, `lll`, `fincke_pohst`, `buchberger`, `discrete_log`, `linsolve`, `qr`, `matmul`, `eea`, `eratosthenes`, `int_factor`, `resultant`, `splitting_field`, `interpolate`, `cyclotomic`, `galois`.
- `src/reduce_lift/` — traits `LiftPolyEvalRing`, `PolyLiftFactorsDomain` formalizing reduce-mod-prime / reconstruct workflows.
- `src/matrix/` — `Submatrix`/`SubmatrixMut` (raw-pointer, non-owning); concentrated `unsafe`.
- `src/seq/` — `VectorView`, `VectorFn` sequence traits.
- `src/delegate.rs` — newtype-pattern macro/trait for wrapper rings.
- `src/specialization.rs` — compile-time specialization workaround (no nightly `specialization` feature used here).
- `src/wrapper.rs` — `RingElementWrapper` bundles an element with its ring (needed for `HashSet`, operator overloads).

### Stability annotations

`#[stability::unstable(feature = "enable")]` is used heavily. Such items are invisible to downstream crates unless they enable the `unstable-enable` feature. Anything *not* marked is part of the SemVer-stable surface — breaking it requires a major version bump. Constructors named `create()` are typically unstable; `new()` and `new_with_xyz()` are stable.

### Testing conventions

- All tests are inline `#[cfg(any(test, feature = "generic_tests"))]`. There is no `tests/` directory.
- The `generic_tests` feature is a real architectural feature, not just an internal helper: modules expose `test_*_axioms` functions (e.g. `ring::generic_tests::test_ring_axioms`) that downstream crates can call to verify their own `RingBase` implementations satisfy the trait contract.
- `RANDOM_TEST_INSTANCE_COUNT = 10` in `lib.rs` controls randomized test breadth.
- `MAX_PROBABILISTIC_REPETITIONS` / `DEFAULT_PROBABILISTIC_REPETITIONS = 30` for probabilistic algorithms (Miller–Rabin, Cantor–Zassenhaus, etc.).
- Some tests are `#[ignore]`d for runtime (large EC factorizations, cyclic-7/8 Gröbner bases) — that's intentional.

## Performance notes

- For optimal performance in *consumer* projects: use `lto = "fat"`, since cross-crate inlining is critical for the small-element rings (`Zn64B`, `StaticRing`).
- The default arbitrary-precision integer (`RustBigintRing`) is slow on purpose (pure Rust, low optimization). Real workloads should enable the `mpir` feature.
- Many APIs come in multiple implementations with different perf profiles (`SparsePolyRing` vs `DensePolyRing`, `KaratsubaAlgorithm` vs `NTTConvolution`); consumer code should be written generically so swapping is trivial.
