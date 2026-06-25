# Interface changes

 - FFTAlgorithm should take slices, not VectorViews
 - Rework TransformTarget; at least remove reference to ring in basic transform targets; potentially rework it more broadly
 - Improve Montgomery implementation; there seem to be problems and missing features
 - Remove canonical homomorphisms in favour of more ring-specific maps. The following homomorphisms should exist, and be creatable via `.something_hom()` on the codomain ring: `IntHom`, `Identity`, `Inclusion`, `CoefficientHom` for rings where this makes sense, `EvaluationHom` for polynomial rings, `MonogenicExtensionHom` for extensions, `ComplexEmbedding` for number fields
 - Either make Buchberger compatible with TransformTarget, or introduce a BuchbergerStrategy trait