# Buchberger/F4: task list and notes

This covers the follow-up work from a review of [src/algorithms/buchberger.rs](src/algorithms/buchberger.rs):

- the task list, by priority;
- a design sketch for a strategy trait (task T5);
- an explanation of sugar degree;
- whether the algorithm is correct over local rings in which every ideal is principal.

The reproducers below were run against commit `198a100`.

## Task list

Priority: **P0** means the results are wrong or the code crashes; **P1** is a large performance or design gain; **P2** is a local improvement; **P3** is a larger project.

### T1 (P0): Fix the infinite restart recursion

(potentially subsumed by T4)

- The restart condition at [L632](src/algorithms/buchberger.rs#L632) counts A-polys against an `n² + 1` budget. When a restart can't reduce the pair count, it recurses forever and overflows the stack.
  - Reproducer: `Z/16[x]`, `{8x³ + 4x² + 2x + 1}` gives three A-polys for one input, and 3 > 2.
  - After a restart, the first iteration uses `current_deg = 0`, reduces nothing and then hits the same condition with the same state.
- Only restart if a fresh start would produce strictly fewer pairs than are currently open plus filtered. Otherwise, count only standard pairs.
- Replace the recursion with a loop.
- **Done when:** the Z/16 reproducer is a regression test and returns `{1}`.

### T2 (P0): Reject base rings that are not local

- The code builds only lcm-based S-polys and annihilator polys (A-polys). `find_reducer` ([L182](src/algorithms/buchberger.rs#L182)) does strong reduction, which only gives normal forms for a strong Gröbner basis. Over a ring that is not local, a strong Gröbner basis also needs G-polys, which are not implemented.
  - Reproducer: `{2x, 3x}` over Z/6 or over Z returns `{2x, 3x}`, and `multivariate_division(x, gb)` returns `x`, although `x = 3x − 2x` is in the ideal.
- Over local rings (fields, DVRs, and special principal ideal rings such as Z/p^e), ideals are totally ordered, so G-polys are never needed and the current algorithm is correct. See the last section of this document.
- At the start of `buchberger_with_strategy`, check `ring.base_ring().get_ring().is_local()` (the new method on `PrincipalIdealRing`) and panic with a clear message if it is `false`. `buchberger()` requires `Field`, so it always passes the check.
- Document the precondition: the base ring must be local. For Z/n with composite n, callers should decompose into the Z/p^e components by CRT themselves.
- **Done when:** `{2x, 3x}` over Z/6 and over Z panics upfront with the documented message, and the existing tests over fields and Z/16 still pass.

### T3 (P0): Regression and property tests, plus docs

- **Tests:**
  - Randomized small systems over F_p and Z/p^k.
  - Check that each input reduces to 0, and that every S-pair and A-pair of the output reduces to 0.
  - Over F_p, also compare the leading-term set against an independent run using a different sort order.
- **Docs:**
  - State that the output is minimal but not reduced: tails are not reduced and coefficients are not normalised.
  - State that `multivariate_division` is a normal form only when the basis is a strong Gröbner basis.
  - Remove the claim "as efficient as standard F4".
- **Comment at [L231](src/algorithms/buchberger.rs#L231):** replace it with a note that "divides" without the index tie-break is unsound. With three pairs sharing one lcm, S(i,k) and S(j,k) would each be deleted by relying on the other.

### T4 (P1): Replace restarts with a Gebauer–Möller-style update

- Keep one append-only, indexed basis with an "is redundant" flag, and never form new pairs with redundant elements.
- Do tail reduction in place. It doesn't change leading terms, so existing pairs stay valid.
- Use Gebauer–Möller pair pruning (O(n) per new element) instead of the current O(k)-per-pair chain check, which is O(n³) overall.
- This removes both the restart recomputation and the final verification pass at [L584](src/algorithms/buchberger.rs#L584). T5 depends on it, because provenance can't survive a restart that throws the history away.
- **Done when:** there are no restarts left, and the cyclic-6 and cyclic-7 timings don't get worse.

### T5 (P1): Design a `GBStrategy` trait

- It should subsume `sort_spolys` and `abort_early_if`, and let users track how each basis polynomial is computed from the inputs.
- See the design sketch below.

### T6 (P1): Cache leading terms

- Store `(lc, lm, expanded lm)` next to every basis element and reducer.
- Remove the repeated `LT` calls in `find_reducer`, `filter_spoly`, `cancelled_term` and `reduce_poly`. For any order other than DegRevLex, `MultivariatePolyRingImpl::LT` scans every term ([multivariate_impl.rs:884](src/ring_impls/multivariate/multivariate_impl.rs#L884)).
- **Done when:** a Lex benchmark, such as a scaled-up `test_gb_lex`, shows the speedup.

### T7 (P2): Sugar-degree selection

- Implement it as the default strategy under T5 (see the section on sugar degree below).
- Also sort A-polys by their leading monomial, instead of the lowest annihilated monomial.
- After progress, set `current_deg` to the minimum open degree instead of 0 ([L604](src/algorithms/buchberger.rs#L604)).

### T8 (P2): Prune pairs that can't produce anything

- Generate only the first A-poly, `ann(LC(f))·f`, per basis element. Each later one is `r·(a·f)` for the first annihilator `a`, so it is redundant.
- Skip standard pairs where `lcm(c_i, c_k) = 0`. Their syzygy is a combination of annihilator syzygies.
- Relax the product criterion at [L219](src/algorithms/buchberger.rs#L219) to comaximal leading coefficients: `is_unit(gcd(c_i, c_k))` together with coprime leading monomials. It currently requires both leading coefficients to be units.

### T9 (P2): Faster reduction inner loop

- Use short divisibility masks and a per-monomial divisor cache in `find_reducer`. It currently scans linearly and allocates an `expand_monomial` vector on every call ([L175](src/algorithms/buchberger.rs#L175)).
- Avoid cloning and scaling the whole reducer on every step ([L366-376](src/algorithms/buchberger.rs#L366-L376)); use geobuckets or a heap-based accumulator.

### T10 (P3): Real F4 linear algebra

- Symbolic preprocessing plus sparse elimination over F_p, and Howell form for Z/p^k.
- Reduce each batch against a fixed reducer set, which makes the parallel step deterministic.
- This builds on T4 to T6.

## T5 in detail: the strategy trait

### Requirements

1. **Pair selection:** it replaces `sort_spolys` together with the hard-coded rule "reduce the batch of minimal degree".
2. **Early abort:** it replaces `abort_early_if`, and gets a proper view of the basis instead of `&[(El<P>, ExpandedMonomial)]`.
3. **Provenance:** users can recover, for each output `g`, how it was computed from the inputs. That could be the full cofactors `g = Σ h_i·f_i`, or something cheaper such as the set of inputs involved.
4. **Zero cost when unused:** no allocation or bookkeeping in the default case.
5. **Engine-agnostic:** it should work with the current reduce-one-polynomial-at-a-time loop and with a future matrix-based F4.

### Core idea

Each polynomial in the algorithm carries an *annotation*, and the strategy says how annotations combine. The same mechanism covers three uses:

- Tracking cofactors: the annotation is `Vec<El<P>>`.
- Tracking which inputs contributed: the annotation is a bitset.
- Sugar degree: the annotation is a `usize`.

It could also carry signatures later, if F5 or GVW are added.

```rust
pub trait GBStrategy<P: RingStore> where P::Ring: MultivariatePolyRing, ... {
    /// Data carried alongside every polynomial in the computation; `()` for no tracking.
    type Annotation: Clone + Send + Sync;

    fn annotate_input(&self, input_index: usize, f: &El<P>) -> Self::Annotation;

    /// Annotation of `Σ c_k * m_k * b_k`, where the `b_k` carry the given annotations.
    /// S-polys and A-polys each call it once; each reduction step calls it with 2 terms;
    /// a matrix-based F4 would call it once per output row.
    fn annotate_combination<'a>(
        &self,
        terms: impl Iterator<Item = (&'a Self::Annotation, &'a PolyCoeff<P>, &'a PolyMonomial<P>)>,
    ) -> Self::Annotation where Self::Annotation: 'a;

    /// Removes the next batch of pairs from `open` for reduction. Replaces sort_spolys + the degree rule.
    fn select_batch(&mut self, open: &mut Vec<CriticalPair<P>>, basis: &BasisView<P, Self::Annotation>)
        -> Vec<CriticalPair<P>>;

    /// Called when a pair reduces to zero. The annotation is then a syzygy of the inputs,
    /// which lets users compute syzygy modules for free.
    fn on_zero_reduction(&mut self, _pair: &CriticalPair<P>, _ann: Self::Annotation) {}

    /// Called after every basis update; returning `true` aborts with the current basis.
    fn should_abort(&mut self, _basis: &BasisView<P, Self::Annotation>) -> bool { false }
}
```

### Design points

- **Composition:** split the trait into three small traits, `Selection`, `Tracking` and `Abort`, and make `GBStrategy` a blanket implementation over them. Implementing them for tuples then allows combinations such as `(SugarSelection, CofactorTracking, DegreeBound(d))`, and keeps each piece testable on its own.
- **Default implementation:** `NormalStrategy` with `Annotation = ()`. Keep an adapter so the old closures still work, e.g. `ClosureStrategy(sort_fn, abort_fn)`, so that `buchberger_with_strategy` can keep its signature for now.
- **Cost:** cofactor tracking multiplies the work per reduction step, which can't be avoided. With `()`, everything monomorphizes away.
- **Inter-reduction and tail reduction must also call `annotate_combination`.** This is a hard dependency on T4, since the current restarts rebuild the basis from `reducers` and lose the history.
- **Parallel reduction:** annotations are computed on the worker threads, which is why they need `Send + Sync`.
- **Open question:** whether annotations should apply only to basis elements or also to pending pairs, computed lazily when the pair is reduced. Lazy is cheaper, since most pairs reduce to zero or get filtered.

## Sugar degree

With homogeneous input, the normal strategy (process pairs by increasing lcm degree) behaves very well. Everything produced while processing degree d is homogeneous of degree d, so the computation proceeds cleanly degree by degree.

With inhomogeneous input, the lcm degree says little. Cancellation lowers degrees, and reductions mix polynomials of different degrees. As a result, pairs get processed in a poor order and intermediate results swell.

The sugar of a polynomial is the degree it *would* have if the whole input had been homogenised with an extra variable `h` and the homogeneous computation had been run. The idea is to track that value without paying the cost of homogenising. It follows three rules:

- `sugar(f) = deg(f)` for each input `f` (total degree).
- `sugar(m·f) = deg(m) + sugar(f)` for a monomial `m`.
- `sugar(f ± g) = max(sugar(f), sugar(g))`.

So the pair `(f_i, f_j)`, with `m_i = lcm/LM(f_i)`, has sugar `max(sugar(f_i) + deg(m_i), sugar(f_j) + deg(m_j))`.

Each reduction step `f ← f − c·m·g` updates the sugar to `max(sugar(f), deg(m) + sugar(g))`.

The selection rule is: process the pairs of smallest sugar first, breaking ties by lcm degree. For homogeneous input, sugar equals the ordinary degree, so this is exactly the normal strategy. For inhomogeneous input, it mimics the well-behaved homogeneous computation.

Sugar was introduced by Giovini, Mora, Niesi, Robbiano and Traverso in "One sugar cube, please" (1991). It is the standard default in Singular, CoCoA and most F4 implementations, where each round takes all pairs of minimal sugar.

In the T5 design, sugar is simply `Annotation = usize` with `annotate_combination = max(deg(m_k) + a_k)`.

## Correctness over local rings in which every ideal is principal

On terminology: a discrete valuation ring (DVR) is a domain by definition, e.g. Z_(p), Z_p or k[[t]]. Non-domain local rings in which every ideal is principal are called *special principal ideal rings* or *finite chain rings*, e.g. Z/p^k, F_q[t]/(t^k) or Galois rings. Both cases are fine in theory, but only the domain case is fine in the current code:

| | G-polys needed? | A-polys | Termination | Current code |
|---|---|---|---|---|
| **DVR** (domain) | No | None, since `ann(c) = 0` | Yes: valuations are well-ordered | Correct |
| **Special principal ideal ring** (non-domain, e.g. Z/p^k) | No | Needed; the first one per element is generated | Yes: the coefficient ring has finite length | Correct in theory; **crashes in practice** until T1 is fixed |

### Why no G-polys are needed

In a local ring whose ideals are all principal, the ideals form a chain. So `gcd(c_i, c_j)` always equals `c_i` or `c_j` up to a unit, and strong divisibility is just comparing valuations.

### Why the rest holds

- The S-polys plus the A-poly `ann(LC(f))·f` generate all syzygies of the leading terms.
- The chain criterion is sound with the `j < i` tie-break. The argument is induction on (lcm under divisibility, larger index, smaller index). Its error terms lie in the annihilator syzygies, which are never filtered.
- The product criterion is sound.
- Different choices of quotient in `checked_div` differ only by annihilator multiples, which the A-polys cover.

### What isn't correct today

In the non-domain case, the T1 bug is a real failure: its reproducer, Z/16, is exactly such a ring. The domain case can't hit it, because without A-polys the pair count `n(n−1)/2` never exceeds `n² + 1`.

### Practical caveat for DVRs

A DVR like Z_p or k[[t]] can only be implemented with finite precision. At that point the computation is over Z/p^k, which is the non-domain case. The Gröbner basis of the truncated ideal is generally not the truncation of the true Gröbner basis, because division by powers of p loses precision. So results there need a precision analysis, not just a correctness proof.
