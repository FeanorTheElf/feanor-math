use std::alloc::Global;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use feanor_serde::newtype_struct::*;
use serde::de::DeserializeSeed;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::extension_impl::FreeAlgebraImpl;
use super::{Field, FreeAlgebra};
use crate::algorithms::convolution::*;
use crate::algorithms::newton;
use crate::algorithms::poly_gcd::*;
use crate::delegate::*;
use crate::prelude::*;
use crate::ring_impls::as_field::AsField;
use crate::ring_impls::extension::complex_embedding::ComplexEmbedding;
use crate::ring_impls::extension::extension_impl::*;
use crate::ring_impls::extension::number_field::newton::find_approximate_complex_root;
use crate::ring_impls::extension::*;
use crate::ring_impls::rational::*;
use crate::ring_properties::field::PerfectField;
use crate::ring_properties::serialization::*;
use crate::ring_properties::specialization::*;

/// An algebraic number field, i.e. a finite rank field extension of the rationals.
///
/// This type only wraps an underlying implementation of the ring arithmetic, and adds
/// some number-field specific functionality. However, the implementation type defaults to
/// [`DefaultNumberFieldImpl`], which should be sufficient for almost all purposes.
/// Note that the only way to create a number field that does not use the default
/// implementation is via [`NumberFieldBase::create()`].
///
/// # Example
///
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::extension::number_field::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::integer::*;
/// let ZZ = ZZbig;
/// let ZZX = DensePolyRing::new(ZZ, "X");
/// let [gen_poly] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
/// // the Gaussian numbers `QQ[i]`
/// let QQi = NumberField::new(&ZZX, &gen_poly);
/// let i = QQi.canonical_gen();
/// assert_el_eq!(&QQi, QQi.neg_one(), QQi.pow(i, 2));
/// ```
/// So far, we could have done the same with just [`FreeAlgebraImpl`], which indeed
/// is used as the default implementation of the arithmetic. However, [`NumberField`]
/// provides additional functionality, that is not available for general extensions.
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::extension::number_field::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::algorithms::poly_factor::*;
/// # use feanor_math::integer::*;
/// # let ZZ = ZZbig;
/// # let ZZX = DensePolyRing::new(ZZ, "X");
/// # let [gen_poly] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
/// # let QQi = NumberField::new(&ZZX, &gen_poly);
/// # let i = QQi.canonical_gen();
/// let QQiX = DensePolyRing::new(&QQi, "X");
/// let [f] = QQiX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 4]);
/// let (factorization, _) = <_ as FactorPolyField>::factor_poly(&QQiX, &f);
/// assert_eq!(2, factorization.len());
/// ```
/// The internal generating polynomial of a number field is currently always
/// integral, but you can create a number field also from a rational polynomial
/// using [`NumberField::adjoin_root()`].
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::divisibility::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::extension::number_field::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::rings::rational::*;
/// # use feanor_math::integer::*;
/// let ZZ = ZZbig;
/// let QQ = RationalField::new(ZZ);
/// let QQX = DensePolyRing::new(&QQ, "X");
/// // take `gen_poly = X^2 + 1/4`
/// let gen_poly = QQX.add(
///     QQX.pow(QQX.indeterminate(), 2),
///     QQX.inclusion()
///         .map(QQ.invert(&QQ.int_hom().map(4)).unwrap()),
/// );
/// // this still gives the Gaussian numbers `QQ[i]`
/// let (QQi, i_half) = NumberField::adjoin_root(&QQX, &gen_poly);
/// assert_el_eq!(
///     &QQi,
///     QQi.neg_one(),
///     QQi.pow(QQi.int_hom().mul_ref_map(&i_half, &2), 2)
/// );
/// // however the canonical generator might not be `i/2`
/// assert!(!QQi.eq_el(&QQi.canonical_gen(), &i_half));
/// ```
///
/// # Why not relative number fields?
///
/// Same as [`crate::rings::extension::galois_field::GaloisFieldBase`], this type represents
/// number fields globally, i.e. always in the form `Q[X]/(f(X))`. By the primitive element
/// theorem, each number field can be written in this form. However, it might be more natural
/// in some applications to write it as an extension of a smaller number field, say `L =
/// K[X]/(f(X))`.
///
/// I tried this before, and it turned out to be a constant fight with the type system.
/// The final code worked more or less (see git commit b1ef445cf14733f63d035b39314c2dd66fd7fcb5),
/// but it looks terrible, since we need quite a few "helper" traits to be able to provide all the
/// expected functionality. Basically, every functionality must now be represented by one (or many)
/// traits that are implemented by `QQ` and by any extension `K[X]/(f(X))` for which `K` implements
/// it. In some cases (like polynomial factorization), we want to have "functorial" functions that
/// map a number field to something else (e.g. one of its orders), and each of those now requires
/// a complete parallel hierarchy of traits. If you are not yet frightened, checkout the above
/// commit and see if you can make sense of the corresponding code.
///
/// To summarize, all number fields are represented absolutely, i.e. as extensions of `QQ`.
///
/// # Factoring out denominators
///
/// TODO: At next breaking release, investigate whether it is sensible to have `Impl` be an
/// algebraic extension of `Z` instead of `Q`, and store the joint denominator once for every
/// element.
///
/// # Choice of blanket implementations of [`CanHomFrom`]
///
/// This is done analogously to [`crate::rings::extension::galois_field::GaloisFieldBase`], see
/// the description there.
pub struct NumberFieldBase<Impl = DefaultNumberFieldImpl, I = BigIntRing>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    base: Impl,
}

#[stability::unstable(feature = "enable")]
pub type DefaultNumberFieldImpl = AsField<
    FreeAlgebraImpl<
        RationalField<BigIntRing>,
        Vec<El<RationalField<BigIntRing>>>,
        DynConvolution<'static, RationalFieldBase<BigIntRing>>,
        Global,
    >,
>;

#[stability::unstable(feature = "enable")]
pub type NumberField<Impl = DefaultNumberFieldImpl, I = BigIntRing> = RingValue<NumberFieldBase<Impl, I>>;

impl NumberField {
    /// Given a monic, integral and irreducible polynomial, returns the number field
    /// generated by it. Its canonical generator is a root of the given polynomial.
    ///
    /// For more details, see [`NumberFieldBase::new()`].
    pub fn new<P>(poly_ring: P, generating_poly: &El<P>) -> Self
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = BigIntRingBase>,
    {
        RingValue::from(NumberFieldBase::new(poly_ring, generating_poly))
    }

    /// Computes the number field generated by a root of the given irreducible polynomial, and
    /// returns it together with the root (which is not necessarily
    /// the canonical generator of the number field).
    ///
    /// For more details, see [`NumberFieldBase::adjoin_root()`].
    pub fn adjoin_root<P>(poly_ring: P, generating_poly: &El<P>) -> (Self, El<Self>)
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = RationalFieldBase<BigIntRing>>,
    {
        let (res, root) = NumberFieldBase::adjoin_root(poly_ring, generating_poly);
        (RingValue::from(res), root)
    }
}

impl NumberFieldBase {
    /// If the given polynomial is irreducible, returns the number field generated
    /// by it (with a root of the polynomial as canonical generator). Otherwise,
    /// `None` is returned.
    ///
    /// If the given polynomial is not integral or not monic, consider using
    /// [`NumberField::try_adjoin_root()`] instead.
    pub fn try_new<P>(poly_ring: P, generating_poly: &El<P>) -> Option<Self>
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = BigIntRingBase>,
    {
        assert!(poly_ring.base_ring().is_one(poly_ring.lc(generating_poly).unwrap()));
        let QQ = RationalField::new(ZZbig);
        let rank = poly_ring.degree(generating_poly).unwrap();
        let modulus = (0..rank)
            .map(|i| QQ.negate(QQ.inclusion().map_ref(poly_ring.coefficient_at(generating_poly, i))))
            .collect::<Vec<_>>();
        let log2_padded_len = ZZi64.abs_log2_ceil(&rank.try_into().unwrap()).unwrap();
        let convolution = RationalFieldBase::create_default_convolution(QQ.clone(), Some(2 << log2_padded_len));
        return RingValue::from(FreeAlgebraImplBase::new_with_convolution(
            QQ,
            rank,
            modulus,
            "θ",
            Global,
            convolution,
        ))
        .as_field()
        .ok()
        .map(Self::create);
    }

    /// Given a monic, integral and irreducible polynomial, returns the number field
    /// generated by it (with a root of the polynomial as canonical generator).
    ///
    /// Panics if the polynomial is not irreducible.
    ///
    /// If the given polynomial is not integral or not monic, consider using
    /// [`NumberField::adjoin_root()`] instead.
    pub fn new<P>(poly_ring: P, generating_poly: &El<P>) -> Self
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = BigIntRingBase>,
    {
        Self::try_new(poly_ring, generating_poly).unwrap()
    }

    /// If the given polynopmial is irreducible, computes the number field generated
    /// by one of its roots, and returns it together with the root (which is not necessarily
    /// the canonical generator of the number field). Otherwise, `None` is returned.
    pub fn try_adjoin_root<P>(poly_ring: P, generating_poly: &El<P>) -> Option<(Self, <Self as RingBase>::Element)>
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = RationalFieldBase<BigIntRing>>,
    {
        let QQ = poly_ring.base_ring();
        let ZZ = QQ.base_ring();
        let denominator = poly_ring
            .terms(generating_poly)
            .map(|(c, _)| QQ.get_ring().den(c))
            .fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, b));
        let rank = poly_ring.degree(generating_poly).unwrap();
        let scaled_lc = ZZ
            .checked_div(
                &ZZ.mul_ref(&denominator, QQ.get_ring().num(poly_ring.lc(generating_poly).unwrap())),
                QQ.get_ring().den(poly_ring.lc(generating_poly).unwrap()),
            )
            .unwrap();
        let ZZX = DensePolyRing::new(ZZ, "X");
        let new_generating_poly = ZZX.from_terms(poly_ring.terms(generating_poly).map(|(c, i)| {
            if i == rank {
                (ZZ.one(), rank)
            } else {
                (
                    ZZ.checked_div(
                        &ZZ.mul_ref_fst(
                            &denominator,
                            ZZ.mul_ref_fst(QQ.get_ring().num(c), ZZ.pow(scaled_lc.clone(), rank - i - 1)),
                        ),
                        QQ.get_ring().den(c),
                    )
                    .unwrap(),
                    i,
                )
            }
        }));
        return Self::try_new(ZZX, &new_generating_poly).map(|res| {
            let root = RingRef::from(&res)
                .inclusion()
                .mul_map(res.canonical_gen(), QQ.invert(&QQ.inclusion().map(scaled_lc)).unwrap());
            return (res, root);
        });
    }

    /// Computes the number field generated by a root of the given irreducible polynomial, and
    /// returns it together with the root (which is not necessarily
    /// the canonical generator of the number field).
    pub fn adjoin_root<P>(poly_ring: P, generating_poly: &El<P>) -> (Self, <Self as RingBase>::Element)
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = RationalFieldBase<BigIntRing>>,
    {
        Self::try_adjoin_root(poly_ring, generating_poly).unwrap()
    }
}
impl<Impl, I> NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    /// Creates a new number field with the given underlying implementation.
    ///
    /// Requires that all coefficients of the generating polynomial are integral.
    #[stability::unstable(feature = "enable")]
    pub fn create(implementation: Impl) -> Self {
        let poly_ring = DensePolyRing::new(implementation.base_ring(), "X");
        let gen_poly = implementation.generating_poly(&poly_ring, poly_ring.base_ring().identity());
        assert!(poly_ring.terms(&gen_poly).all(|(c, _)| {
            poly_ring
                .base_ring()
                .base_ring()
                .is_one(poly_ring.base_ring().get_ring().den(c))
        }));
        drop(poly_ring);
        NumberFieldBase { base: implementation }
    }

    fn generating_poly_as_int<P>(&self, poly_ring: P) -> El<P>
    where
        P: RingStore,
        P::Ring: PolyRing,
        BaseRingStore<P>: RingStore<Ring = I::Ring>,
    {
        assert!(poly_ring.base_ring().get_ring() == self.base_ring().base_ring().get_ring());
        RingRef::from(self).generating_poly(
            &poly_ring,
            LambdaHom::new(self.base_ring(), poly_ring.base_ring(), |QQ, ZZ, x| {
                ZZ.checked_div(QQ.get_ring().num(x), QQ.get_ring().den(x)).unwrap()
            }),
        )
    }

    #[stability::unstable(feature = "enable")]
    pub fn into_choose_complex_embedding<S: RingStore<Ring = Self>>(self_: S) -> ComplexEmbedding<S, I> {
        let ZZ = self_.base_ring().base_ring();
        let poly_ring = DensePolyRing::new(ZZ, "X");
        let poly = self_.get_ring().generating_poly_as_int(&poly_ring);
        let (root, error) = find_approximate_complex_root(&poly_ring, &poly).unwrap();
        drop(poly);
        drop(poly_ring);
        return ComplexEmbedding::create(self_, root, error);
    }

    #[stability::unstable(feature = "enable")]
    pub fn choose_complex_embedding<'a>(&'a self) -> ComplexEmbedding<RingRef<'a, Self>, I> {
        Self::into_choose_complex_embedding(RingRef::from(self))
    }
}

impl<Impl, I> Clone for NumberFieldBase<Impl, I>
where
    Impl: RingStore + Clone,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
        }
    }
}

impl<Impl, I> Copy for NumberFieldBase<Impl, I>
where
    Impl: RingStore + Copy,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
    El<Impl>: Copy,
    El<I>: Copy,
{
}

impl<Impl, I> PartialEq for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn eq(&self, other: &Self) -> bool { self.base.get_ring() == other.base.get_ring() }
}

impl<Impl, I> DelegateRing for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    type Base = Impl::Ring;
    type Element = El<Impl>;

    fn get_delegate(&self) -> &Self::Base { self.base.get_ring() }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<Impl, I> DelegateRingImplEuclideanRing for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<Impl, I> Debug for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "NumberField({:?})", self.base.get_ring()) }
}

impl<Impl, I> FiniteRingSpecializable for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output { op.fallback() }
}

impl<Impl, I> Field for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<Impl, I> PerfectField for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<Impl, I> Domain for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
}

impl<Impl, I> PolyTFracGCDRing for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn gcd<P>(_poly_ring: P, _lhs: &El<P>, _rhs: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn power_decomposition<P>(_poly_ring: P, _poly: &El<P>) -> Vec<(El<P>, usize)>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn is_squarefree<P>(_poly_ring: P, _poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn squarefree_part<P>(_poly_ring: P, _poly: &El<P>) -> El<P>
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + DivisibilityRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }
}

impl<Impl, I> FactorPolyField for NumberFieldBase<Impl, I>
where
    Impl: RingStore,
    Impl::Ring: Field + FreeAlgebra,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn factor_poly<P>(_poly_ring: P, _poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }

    fn is_irred<P>(_poly_ring: P, _poly: &El<P>) -> bool
    where
        P: RingStore + Copy,
        P::Ring: PolyRing + EuclideanRing,
        BaseRingStore<P>: RingStore<Ring = Self>,
    {
        unimplemented!()
    }
}

impl<Impl, I> Serialize for NumberFieldBase<Impl, I>
where
    Impl: RingStore + Serialize,
    Impl::Ring: Field + FreeAlgebra + SerializableElementRing,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerializableNewtypeStruct::new("NumberField", &self.base).serialize(serializer)
    }
}

impl<'de, Impl, I> Deserialize<'de> for NumberFieldBase<Impl, I>
where
    Impl: RingStore + Deserialize<'de>,
    Impl::Ring: Field + FreeAlgebra + SerializableElementRing,
    BaseRingStore<Impl>: RingStore<Ring = RationalFieldBase<I>>,
    I: RingStore,
    I::Ring: IntegerRing,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        DeserializeSeedNewtypeStruct::new("NumberField", PhantomData::<Impl>)
            .deserialize(deserializer)
            .map(|res| NumberFieldBase::create(res).into())
    }
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::iters::multi_cartesian_product;

#[test]
fn test_principal_ideal_ring_axioms() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&ZZX, &f);

    let elements = multi_cartesian_product(
        [(-4..4), (-2..2)].into_iter(),
        |slice| K.from_canonical_basis(slice.iter().map(|x| K.base_ring().int_hom().map(*x))),
        |_, x| *x,
    )
    .collect::<Vec<_>>();

    crate::ring_properties::pid::generic_tests::test_principal_ideal_ring_axioms(
        &K,
        elements.iter().map(|x| x.clone()),
    );
}

#[test]
fn test_adjoin_root() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let QQ = RationalField::new(ZZ);
    let QQX = DensePolyRing::new(QQ, "X");
    let [f] = QQX.with_wrapped_indeterminate(|X| [2 * X.pow_ref(3) - 1]);
    let (K, a) = NumberField::adjoin_root(&QQX, &f);
    assert_el_eq!(&K, K.zero(), K.sub(K.mul(K.int_hom().map(2), K.pow(a, 3)), K.one()));
}

#[test]
fn test_poly_gcd_number_field() {
    feanor_tracing::DelayedLogger::init_test();
    let ZZ = ZZbig;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&ZZX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let i = RingElementWrapper::new(&KY, KY.inclusion().map(K.canonical_gen()));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| {
        [
            (Y.pow_ref(3) + 1) * (Y - &i),
            (Y.pow_ref(4) + 2) * (Y.pow_ref(2) + 1),
            Y - i,
        ]
    });
    assert_el_eq!(&KY, &expected, <_ as PolyTFracGCDRing>::gcd(&KY, &g, &h));

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 20 * X.pow_ref(2) + 16]);
    let K = NumberField::new(&ZZX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let [sqrt3, sqrt7] = K.with_wrapped_generator(|a| [a.pow_ref(3) / 8 - 2 * a, a.pow_ref(3) / 8 - 3 * a]);
    assert_el_eq!(&K, K.int_hom().map(3), K.pow(sqrt3.clone(), 2));
    assert_el_eq!(&K, K.int_hom().map(7), K.pow(sqrt7.clone(), 2));

    let half = RingElementWrapper::new(&KY, KY.inclusion().map(K.invert(&K.int_hom().map(2)).unwrap()));
    let sqrt3 = RingElementWrapper::new(&KY, KY.inclusion().map(sqrt3));
    let sqrt7 = RingElementWrapper::new(&KY, KY.inclusion().map(sqrt7));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| {
        [
            Y.pow_ref(2) - &sqrt3 * Y - 1,
            Y.pow_ref(2) + &sqrt7 * Y + 1,
            Y - (sqrt3 - sqrt7) * half,
        ]
    });
    let actual = <_ as PolyTFracGCDRing>::gcd(&KY, &g, &h);
    assert_el_eq!(&KY, &expected, &actual);
}

#[test]
fn random_test_poly_gcd_number_field() {
    feanor_tracing::DelayedLogger::init_test();

    // use tracing_subscriber::Layer;
    // use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
    // use tracing_subscriber::util::SubscriberInitExt;
    // let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    // let filtered_chrome_layer =
    // chrome_layer.with_filter(tracing_subscriber::filter::filter_fn(|metadata|
    //     !["feanor_math::algorithms::bigint_ops", "feanor_math::algorithms::eea",
    // "feanor_math::algorithms::sqr_mul"].contains(&metadata.target()) ));
    // tracing_subscriber::registry().with(filtered_chrome_layer).init();

    let ZZ = ZZbig;
    let QQ = RationalField::new(ZZ);
    let ZZX = DensePolyRing::new(ZZ, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = QQ.base_ring().int_hom().map(1000);
    let rank = 6;

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let genpoly = ZZX.from_terms(
            (0..rank)
                .map(|i| (ZZ.get_uniformly_random(&bound, || rng.rand_u64()), i))
                .chain([(ZZ.one(), rank)].into_iter()),
        );

        let K = NumberField::new(&ZZX, &genpoly);
        let KY = DensePolyRing::new(&K, "Y");

        let mut random_element_K = || {
            K.from_canonical_basis((0..K.rank()).map(|_| {
                QQ.inclusion()
                    .map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64()))
            }))
        };
        let f = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let g = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let h = KY.from_terms((0..=4).map(|i| (random_element_K(), i)));
        let lhs = KY.mul_ref(&f, &h);
        let rhs = KY.mul_ref(&g, &h);

        let gcd = <_ as PolyTFracGCDRing>::gcd(&KY, &lhs, &rhs);

        assert!(KY.divides(&lhs, &gcd));
        assert!(KY.divides(&rhs, &gcd));
        assert!(KY.divides(&gcd, &h));
    }
}
