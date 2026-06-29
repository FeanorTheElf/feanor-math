# Interface changes

 - FFTAlgorithm should take slices, not VectorViews
 - Improve Montgomery implementation; there seem to be problems and missing features
 - Remove canonical homomorphisms in favour of more ring-specific maps. The following homomorphisms should exist, and be creatable via `.something_hom()` on the codomain ring: `IntHom`, `Identity`, `Inclusion`, `CoefficientHom` for rings where this makes sense, `EvaluationHom` for polynomial rings, `MonogenicExtensionHom` for extensions, `ComplexEmbedding` for number fields
 - Make a subtrait `PreimageAwareHomomorphism` which has a function `any_preimage(&self, x: El<Codomain>) -> Option<El<Domain>>`
 - Make it easier to get homomorphisms between field extensions, in particular finite fields
 - Either make Buchberger compatible with TransformTarget, or introduce a BuchbergerStrategy trait
 - expose `minpoly()` and others through RingStore
 - a better fast xgcd algorithm (probably going to be textbook half-gcd), or rather one that I can prove correct