use std::fmt::{Debug, Formatter};
use std::alloc::Global;
use std::marker::PhantomData;

use crate::algorithms::interpolate::product_except_one;
use crate::algorithms::newton::{self, absolute_error_of_poly_eval};
use crate::algorithms::poly_factor::extension::poly_factor_extension;
use crate::algorithms::poly_factor::factor_locally::{factor_and_lift_mod_pe, FactorAndLiftModpeResult};
use crate::algorithms::poly_gcd::squarefree_part::poly_power_decomposition_local;
use crate::algorithms::poly_gcd::gcd::poly_gcd_local;
use crate::algorithms::resultant::ComputeResultantRing;
use crate::reduce_lift::lift_poly_factors::*;
use crate::rings::extension::number_field::newton::find_approximate_complex_root;
use crate::algorithms::rational_reconstruction::balanced_rational_reconstruction;
use crate::computation::*;
use crate::delegate::*;
use crate::specialization::*;
use crate::algorithms::convolution::*;
use crate::algorithms::poly_gcd::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::field::AsField;
use crate::rings::poly::*;
use crate::rings::zn::ZnRingStore;
use crate::rings::rational::*;
use crate::divisibility::*;
use crate::rings::extension::*;
use crate::rings::extension::extension_impl::*;
use crate::rings::float_complex::{Complex64Base, Complex64};
use crate::serialization::*;
use crate::rings::extension::sparse::SparseMapVector;

use feanor_serde::newtype_struct::*;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::DeserializeSeed;

use super::extension_impl::FreeAlgebraImpl;
use super::Field;
use super::FreeAlgebra;

const TRY_FIND_INERT_PRIME_ATTEMPTS: usize = 10;
const TRY_FACTOR_DIRECTLY_ATTEMPTS: usize = 0;

///
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
/// let ZZ = BigIntRing::RING;
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
/// # let ZZ = BigIntRing::RING;
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
/// let ZZ = BigIntRing::RING;
/// let QQ = RationalField::new(ZZ);
/// let QQX = DensePolyRing::new(&QQ, "X");
/// // take `gen_poly = X^2 + 1/4`
/// let gen_poly = QQX.add(QQX.pow(QQX.indeterminate(), 2), QQX.inclusion().map(QQ.invert(&QQ.int_hom().map(4)).unwrap()));
/// // this still gives the Gaussian numbers `QQ[i]`
/// let (QQi, i_half) = NumberField::adjoin_root(&QQX, &gen_poly);
/// assert_el_eq!(&QQi, QQi.neg_one(), QQi.pow(QQi.int_hom().mul_ref_map(&i_half, &2), 2));
/// // however the canonical generator might not be `i/2`
/// assert!(!QQi.eq_el(&QQi.canonical_gen(), &i_half));
/// ```
/// 
/// # Why not relative number fields?
/// 
/// Same as [`crate::rings::extension::galois_field::GaloisFieldBase`], this type represents
/// number fields globally, i.e. always in the form `Q[X]/(f(X))`. By the primitive element
/// theorem, each number field can be written in this form. However, it might be more natural
/// in some applications to write it as an extension of a smaller number field, say `L = K[X]/(f(X))`.
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
/// 
#[stability::unstable(feature = "enable")]
pub struct NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    base: Impl
}

///
/// An embedding of a number field `K` into the complex numbers `CC`, represented
/// approximately via floating point numbers.
/// 
#[stability::unstable(feature = "enable")]
pub struct ComplexEmbedding<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    from: K,
    image_of_generator: El<Complex64>,
    absolute_error_image_of_generator: f64
}

impl<K, Impl, I> ComplexEmbedding<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// Returns `epsilon > 0` such that when evaluating this homomorphism
    /// at point `x`, the given result is at most `epsilon` from the actual
    /// result (i.e. the result when computed with infinite precision).
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn absolute_error_bound_at(&self, x: &<NumberFieldBase<Impl, I> as RingBase>::Element) -> f64 {
        let CC = Complex64::RING;
        let CCX = DensePolyRing::new(CC, "X");
        let f = self.from.poly_repr(&CCX, x, CC.can_hom(self.from.base_ring()).unwrap());
        return absolute_error_of_poly_eval(&CCX, &f, self.from.rank(), self.image_of_generator, self.absolute_error_image_of_generator / CC.abs(self.image_of_generator));
    }
}

impl<K, Impl, I> Homomorphism<NumberFieldBase<Impl, I>, Complex64Base> for ComplexEmbedding<K, Impl, I>
    where K: RingStore<Type = NumberFieldBase<Impl, I>>,
        Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    type DomainStore = K;
    type CodomainStore = Complex64;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &Complex64::RING
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.from
    }

    fn map_ref(&self, x: &<NumberFieldBase<Impl, I> as RingBase>::Element) -> <Complex64Base as RingBase>::Element {
        let poly_ring = DensePolyRing::new(*self.codomain(), "X");
        let hom = self.codomain().can_hom(self.from.base_ring()).unwrap();
        return poly_ring.evaluate(&self.from.poly_repr(&poly_ring, &x, &hom), &self.image_of_generator, self.codomain().identity());
    }

    fn map(&self, x: <NumberFieldBase<Impl, I> as RingBase>::Element) -> <Complex64Base as RingBase>::Element {
        self.map_ref(&x)
    }
}

#[stability::unstable(feature = "enable")]
pub type DefaultNumberFieldImpl = AsField<FreeAlgebraImpl<RationalField<BigIntRing>, Vec<El<RationalField<BigIntRing>>>, Global, KaratsubaAlgorithm>>;
#[stability::unstable(feature = "enable")]
pub type NumberField<Impl = DefaultNumberFieldImpl, I = BigIntRing> = RingValue<NumberFieldBase<Impl, I>>;

impl NumberField {

    ///
    /// If the given polynomial is irreducible, returns the number field generated
    /// by it (with a root of the polynomial as canonical generator). Otherwise,
    /// `None` is returned.
    /// 
    /// If the given polynomial is not integral or not monic, consider using
    /// [`NumberField::try_adjoin_root()`] instead.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn try_new<P>(poly_ring: P, generating_poly: &El<P>) -> Option<Self>
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = BigIntRingBase>
    {
        assert!(poly_ring.base_ring().is_one(poly_ring.lc(generating_poly).unwrap()));
        let QQ = RationalField::new(BigIntRing::RING);
        let rank = poly_ring.degree(generating_poly).unwrap();
        let modulus = (0..rank).map(|i| QQ.negate(QQ.inclusion().map_ref(poly_ring.coefficient_at(generating_poly, i)))).collect::<Vec<_>>();
        return FreeAlgebraImpl::new_with_convolution(QQ, rank, modulus, "θ", Global, STANDARD_CONVOLUTION).as_field().ok().map(Self::create);
    }
    
    ///
    /// Given a monic, integral and irreducible polynomial, returns the number field 
    /// generated by it (with a root of the polynomial as canonical generator).
    /// 
    /// Panics if the polynomial is not irreducible.
    /// 
    /// If the given polynomial is not integral or not monic, consider using
    /// [`NumberField::adjoin_root()`] instead.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new<P>(poly_ring: P, generating_poly: &El<P>) -> Self
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = BigIntRingBase>
    {
        Self::try_new(poly_ring, generating_poly).unwrap()
    }

    ///
    /// If the given polynopmial is irreducible, computes the number field generated
    /// by one of its roots, and returns it together with the root (which is not necessarily
    /// the canonical generator of the number field). Otherwise, `None` is returned.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn try_adjoin_root<P>(poly_ring: P, generating_poly: &El<P>) -> Option<(Self, El<Self>)>
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<BigIntRing>>
    {
        let QQ = poly_ring.base_ring();
        let ZZ = QQ.base_ring();
        let denominator = poly_ring.terms(generating_poly).map(|(c, _)| QQ.get_ring().den(c)).fold(
            ZZ.one(), 
            |a, b| ZZ.ideal_intersect(&a, b)
        );
        let rank = poly_ring.degree(generating_poly).unwrap();
        let scaled_lc = ZZ.checked_div(&ZZ.mul_ref(&denominator, QQ.get_ring().num(poly_ring.lc(generating_poly).unwrap())), QQ.get_ring().den(poly_ring.lc(generating_poly).unwrap())).unwrap();
        let ZZX = DensePolyRing::new(ZZ, "X");
        let new_generating_poly = ZZX.from_terms(poly_ring.terms(generating_poly).map(|(c, i)| if i == rank {
            (ZZ.one(), rank)
        } else {
            (ZZ.checked_div(&ZZ.mul_ref_fst(&denominator, ZZ.mul_ref_fst(QQ.get_ring().num(c), ZZ.pow(ZZ.clone_el(&scaled_lc), rank - i - 1))), QQ.get_ring().den(c)).unwrap(), i)
        }));
        return Self::try_new(ZZX, &new_generating_poly).map(|res| {
            let root = res.inclusion().mul_map(res.canonical_gen(), QQ.invert(&QQ.inclusion().map(scaled_lc)).unwrap());
            return (res, root);
        });
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn adjoin_root<P>(poly_ring: P, generating_poly: &El<P>) -> (Self, El<Self>)
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<BigIntRing>>
    {
        Self::try_adjoin_root(poly_ring, generating_poly).unwrap()
    }
}

impl<Impl, I> NumberField<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// Creates a new number field with the given underlying implementation.
    /// 
    /// Requires that all coefficients of the generating polynomial are integral.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create(implementation: Impl) -> Self {
        let poly_ring = DensePolyRing::new(implementation.base_ring(), "X");
        let gen_poly = implementation.generating_poly(&poly_ring, poly_ring.base_ring().identity());
        assert!(poly_ring.terms(&gen_poly).all(|(c, _)| poly_ring.base_ring().base_ring().is_one(poly_ring.base_ring().get_ring().den(c))));
        RingValue::from(NumberFieldBase {
            base: implementation,
        })
    }

    #[stability::unstable(feature = "enable")]
    pub fn into_choose_complex_embedding(self) -> ComplexEmbedding<Self, Impl, I> {
        let ZZ = self.base_ring().base_ring();
        let poly_ring = DensePolyRing::new(ZZ, "X");
        let poly = self.get_ring().generating_poly_as_int(&poly_ring);
        let (root, error) = find_approximate_complex_root(&poly_ring, &poly).unwrap();
        return ComplexEmbedding {
            from: self,
            image_of_generator: root,
            absolute_error_image_of_generator: error
        };
    }

    #[stability::unstable(feature = "enable")]
    pub fn choose_complex_embedding<'a>(&'a self) -> ComplexEmbedding<&'a Self, Impl, I> {
        let ZZ = self.base_ring().base_ring();
        let poly_ring = DensePolyRing::new(ZZ, "X");
        let poly = self.get_ring().generating_poly_as_int(&poly_ring);
        let (root, error) = find_approximate_complex_root(&poly_ring, &poly).unwrap();
        return ComplexEmbedding {
            from: self,
            image_of_generator: root,
            absolute_error_image_of_generator: error
        };
    }
}

impl<Impl, I> NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn generating_poly_as_int<'a>(&self, ZZX: &DensePolyRing<&'a I>) -> El<DensePolyRing<&'a I>> {
        let ZZ = *ZZX.base_ring();
        let assume_in_ZZ = LambdaHom::new(self.base_ring(), ZZ, |from, to, x| to.checked_div(from.get_ring().num(x), from.get_ring().den(x)).unwrap());
        return self.base.generating_poly(ZZX, &assume_in_ZZ);
    }
}

impl<Impl, I> Clone for NumberFieldBase<Impl, I>
    where Impl: RingStore + Clone,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
        }
    }
}

impl<Impl, I> Copy for NumberFieldBase<Impl, I>
    where Impl: RingStore + Copy,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing,
        El<Impl>: Copy,
        El<I>: Copy
{}

impl<Impl, I> PartialEq for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<Impl, I> DelegateRing for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    type Base = Impl::Type;
    type Element = El<Impl>;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<Impl, I> DelegateRingImplEuclideanRing for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

impl<Impl, I> Debug for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NumberField({:?})", self.base.get_ring())
    }
}

impl<Impl, I> FiniteRingSpecializable for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<Impl, I> Field for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

impl<Impl, I> PerfectField for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

impl<Impl, I> Domain for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

impl<Impl, I> PolyTFracGCDRing for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn gcd<P>(poly_ring: P, lhs: &El<P>, rhs: &El<P>) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        Self::gcd_with_controller(poly_ring, lhs, rhs, DontObserve)
    }

    fn gcd_with_controller<P, Controller>(poly_ring: P, lhs: &El<P>, rhs: &El<P>, controller: Controller) -> El<P>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        let self_ = NumberFieldByOrder { base: RingRef::new(poly_ring.base_ring().get_ring()) };

        let order_poly_ring = DensePolyRing::new(RingRef::new(&self_), "X");
        let lhs_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, lhs);
        let rhs_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, rhs);

        let result = poly_gcd_local(&order_poly_ring, order_poly_ring.clone_el(&lhs_order), order_poly_ring.clone_el(&rhs_order), controller);

        return self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &result);
    }

    fn power_decomposition<P>(poly_ring: P, poly: &El<P>) -> Vec<(El<P>, usize)>
        where P: RingStore + Copy,
            P::Type: PolyRing + DivisibilityRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self> 
    {
        let self_ = NumberFieldByOrder { base: RingRef::new(poly_ring.base_ring().get_ring()) };
        let order_poly_ring = DensePolyRing::new(RingRef::new(&self_), "X");
        let poly_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, poly);

        let result = poly_power_decomposition_local(&order_poly_ring, poly_order, DontObserve);

        return result.into_iter().map(|(f, k)| (self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &f), k)).collect();
    }
}

enum HeuristicFactorPolyInOrderResult<P>
    where P: RingStore,
        P::Type: PolyRing
{
    PartialFactorization(Vec<(El<P>, usize)>),
    Irreducible,
    Unknown
}

///
/// Tries to factor the polynomial directly, by first finding an inert prime `p`, so that
/// the number ring modulo `p` becomes a finite field. Then we factor the polynomial over
/// the finite field, and hensel-lift it to a factorization in the order. This can fail
/// if we don't find an inert prime - note that they don't have to exist. Note also that the
/// returned factorization may be only a partial factorization.
/// 
/// # Inert primes don't have to exist
/// 
/// E.g. `X^4 - 10 X^2 + 1` is reducible modulo every prime. In fact, it is a theorem that
/// there exists inert primes if and only if the Galois group of the extension is cyclic.
/// 
fn heuristic_factor_poly_directly_in_order<'a, P, Impl, I, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> HeuristicFactorPolyInOrderResult<P>
    where Impl: 'a + RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: 'a + RingStore,
        I::Type: IntegerRing,
        P: RingStore + Copy,
        P::Type: PolyRing + DivisibilityRing,
        <P::Type as RingExtension>::BaseRing: RingStore<Type = NumberFieldByOrder<'a, Impl, I>>,
        Controller: ComputationController
{
    controller.run_computation(format_args!("factor_direct(deg={}, extdeg={})", poly_ring.degree(poly).unwrap(), poly_ring.base_ring().rank()), |controller| {
        let mut rng = oorandom::Rand64::new(1);
        let self_ = poly_ring.base_ring();

        // first, we try to find an inert prime `p` and lift a factorization modulo `p` to the ring
        'try_factor_directly: for attempt in 0..TRY_FACTOR_DIRECTLY_ATTEMPTS {
            let mut inert_prime = None;
            for _ in 0..(TRY_FIND_INERT_PRIME_ATTEMPTS * self_.rank()) {
                let p = self_.get_ring().random_suitable_ideal(|| rng.rand_u64(), attempt);
                if p.minpoly_factors_mod_p.len() == 1 {
                    inert_prime = Some(p);
                    break;
                }
            }
            if let Some(p) = inert_prime {
                log_progress!(controller, "(inert_prime={})", IdealDisplayWrapper::new(self_.base_ring().base_ring().get_ring(), &p.prime));
                let lc_poly = self_.clone_el(poly_ring.lc(poly).unwrap());
                let monic_poly = evaluate_aX(poly_ring, poly, &lc_poly);
                let e = 2 * self_.get_ring().heuristic_exponent(&p, poly_ring.degree(&monic_poly).unwrap(), poly_ring.terms(&monic_poly).map(|(c, _)| c));
                match factor_and_lift_mod_pe(poly_ring, &p, e, &monic_poly, controller.clone()) {
                    FactorAndLiftModpeResult::PartialFactorization(factorization) => {
                        log_progress!(controller, "(partial_success)");
                        debug_assert!(poly_ring.eq_el(&monic_poly, &poly_ring.normalize(poly_ring.prod(factorization.iter().map(|f| poly_ring.clone_el(f))))));
                        let result: Vec<_> = factorization.into_iter().map(|f| (unevaluate_aX(poly_ring, &f, &lc_poly), 1)).collect();
                        debug_assert!(poly_ring.eq_el(&poly_ring.normalize(poly_ring.clone_el(poly)), &poly_ring.normalize(poly_ring.prod(result.iter().map(|(f, e)| poly_ring.pow(poly_ring.clone_el(f), *e))))));
                        return HeuristicFactorPolyInOrderResult::PartialFactorization(result);
                    },
                    FactorAndLiftModpeResult::Irreducible => {
                        return HeuristicFactorPolyInOrderResult::Irreducible;
                    },
                    FactorAndLiftModpeResult::NotSquarefreeModpe => {
                        // probably not square-free
                        let power_decomposition = poly_power_decomposition_local(poly_ring, poly_ring.clone_el(poly), controller.clone());
                        if power_decomposition.len() > 1 {
                            log_progress!(controller, "(partial_success)");
                            return HeuristicFactorPolyInOrderResult::PartialFactorization(power_decomposition);
                        }
                    },
                    FactorAndLiftModpeResult::Unknown => {}
                }
            } else {
                break 'try_factor_directly;
            }
        }
        log_progress!(controller, "(fail)");
        return HeuristicFactorPolyInOrderResult::Unknown;
    })
}

impl<Impl, I> FactorPolyField for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing + ComputeResultantRing
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        Self::factor_poly_with_controller(poly_ring, poly, DontObserve)
    }

    fn factor_poly_with_controller<P, Controller>(poly_ring: P, poly: &El<P>, controller: Controller) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: RingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Controller: ComputationController
    {
        let self_ = NumberFieldByOrder { base: RingRef::new(poly_ring.base_ring().get_ring()) };
        let order_poly_ring = DensePolyRing::new(RingRef::new(&self_), "X");

        let mut to_factor = vec![(poly_ring.clone_el(poly), 1)];
        let mut result = Vec::new();
        while let Some((current, e_base)) = to_factor.pop() {
            if poly_ring.degree(&current).unwrap() == 1 {
                result.push((poly_ring.normalize(current), 1));
            } else {
                let poly_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, &current);
                // try the direct factorization
                match heuristic_factor_poly_directly_in_order(&order_poly_ring, &poly_order, controller.clone()) {
                    HeuristicFactorPolyInOrderResult::PartialFactorization(partial_factorization) => to_factor.extend(
                        partial_factorization.into_iter().map(|(f, e)| (self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &f), e * e_base))
                    ),
                    HeuristicFactorPolyInOrderResult::Irreducible => result.push((current, e_base)),
                    HeuristicFactorPolyInOrderResult::Unknown => result.extend(
                        poly_factor_extension(&poly_ring, &current, controller.clone()).0.into_iter().map(|(f, e)| (f, e * e_base))
                    )
                }
            }
        }
        return (result, poly_ring.base_ring().clone_el(poly_ring.lc(poly).unwrap()));
    }
}

///
/// Implements [`PolyLiftFactorsDomain`] for [`NumberField`].
/// 
/// We don't want to expose the interface of [`PolyLiftFactorsDomain`] for number
/// fields generally, thus use a private newtype.
/// 
/// Note that this does not actually represent the order, since during
/// `reconstruct_ring_el()` we might reconstruct an element outside of the
/// order. Hence, it should remain private.
/// 
struct NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    base: RingRef<'a, NumberFieldBase<Impl, I>>
}

impl<'a, Impl, I> NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    ///
    /// Multiplies the given polynomial with the lcm of the denominators of all coefficients,
    /// and returns the polynomial as element of the current order.
    /// 
    fn scale_poly_to_order<'ring, P1, P2>(&self, from: P1, to: P2, poly: &El<P1>) -> El<P2>
        where P1: RingStore,
            P1::Type: PolyRing,
            <P1::Type as RingExtension>::BaseRing: RingStore<Type = NumberFieldBase<Impl, I>>,
            P2: RingStore,
            P2::Type: PolyRing,
            <P2::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            Self: 'ring
    {
        debug_assert!(self.base.get_ring() == from.base_ring().get_ring());
        debug_assert!(self.base.get_ring() == to.base_ring().get_ring().base.get_ring());
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let denominator = QQ.inclusion().map(from.terms(poly).map(|(c, _)| 
            self.base.wrt_canonical_basis(c).iter().map(|c| ZZ.clone_el(QQ.get_ring().den(&c))).fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, &b))
        ).fold(ZZ.one(), |a, b| ZZ.ideal_intersect(&a, &b)));
        debug_assert!(!QQ.is_zero(&denominator));
        return to.from_terms(from.terms(poly).map(|(c, i)| (self.base.inclusion().mul_ref_map(c, &denominator), i)));
    }

    fn normalize_map_back_from_order<'ring, P1, P2>(&self, from: P1, to: P2, poly: &El<P1>) -> El<P2>
        where P1: RingStore,
            P1::Type: PolyRing,
            <P1::Type as RingExtension>::BaseRing: RingStore<Type = Self>,
            P2: RingStore,
            P2::Type: PolyRing,
            <P2::Type as RingExtension>::BaseRing: RingStore<Type = NumberFieldBase<Impl, I>>,
            Self: 'ring
    {
        debug_assert!(self.base.get_ring() == to.base_ring().get_ring());
        debug_assert!(self.base.get_ring() == from.base_ring().get_ring().base.get_ring());
        let result = to.from_terms(from.terms(poly).map(|(c, i)| (self.clone_el(c), i)));
        return to.normalize(result);
    }
}

impl<'a, Impl, I> PartialEq for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<'a, Impl, I> Debug for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NumberFieldByOrder({:?})", self.base.get_ring())
    }
}

impl<'a, Impl, I> FiniteRingSpecializable for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output {
        op.fallback()
    }
}

impl<'a, Impl, I> DelegateRing for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    type Base = Impl::Type;
    type Element = El<Impl>;

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring().base.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'b>(&self, el: &'b mut Self::Element) -> &'b mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'b>(&self, el: &'b Self::Element) -> &'b <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<'a, Impl, I> Domain for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{}

type LocalRing<'ring, I> = <<I as RingStore>::Type as PolyLiftFactorsDomain>::LocalRing<'ring>;

type ImplementationRing<'ring, I> = AsFieldBase<FreeAlgebraImpl<
    AsField<<<I as RingStore>::Type as IntegerPolyLiftFactorsDomain>::LocalRingAsZn<'ring>>, 
    Vec<El<AsField<<<I as RingStore>::Type as IntegerPolyLiftFactorsDomain>::LocalRingAsZn<'ring>>>>>
>;

struct NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    implementation: RingValue<ImplementationRing<'ring, I>>
}

impl<'ring, I> Clone for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    fn clone(&self) -> Self {
        Self { implementation: RingValue::from(AsFieldBase::promise_is_field(RingValue::from(FreeAlgebraImplBase::create(
            self.implementation.base_ring().clone(),
            self.implementation.rank(),
            self.implementation.get_ring().get_delegate().x_pow_rank().iter().map(|x| self.implementation.base_ring().clone_el(x)).collect(),
            self.implementation.get_ring().get_delegate().gen_name(),
            self.implementation.get_ring().get_delegate().allocator().clone(),
            self.implementation.get_ring().get_delegate().convolution().clone()
        ))).unwrap()) }
    }
}

impl<'ring, I> PartialEq for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    fn eq(&self, other: &Self) -> bool {
        self.implementation.get_ring() == other.implementation.get_ring()
    }
}

impl<'ring, I> Debug for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NumberFieldOrderQuotient({:?})", self.implementation.get_ring())
    }
}

impl<'ring, I> DelegateRing for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    type Base = ImplementationRing<'ring, I>;
    type Element = <ImplementationRing<'ring, I> as RingBase>::Element;

    fn get_delegate(&self) -> &Self::Base {
        self.implementation.get_ring()
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'b>(&self, el: &'b mut Self::Element) -> &'b mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'b>(&self, el: &'b Self::Element) -> &'b <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<'ring, I> Domain for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{}

impl<'ring, I> DelegateRingImplEuclideanRing for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{}

impl<'ring, I> Field for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{}

impl<'ring, I> DelegateRingImplFiniteRing for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{}

impl<'ring, I> CanHomFrom<Self> for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &Self) -> Option<Self::Homomorphism> {
        if self == from {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _from: &Self, el: <Self as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        el
    }
}

impl<'ring, I> CanIsoFromTo<Self> for NumberFieldOrderQuotient<'ring, I>
    where I: 'ring + RingStore,
        I::Type: IntegerRing
{
    type Isomorphism = <Self as CanHomFrom<Self>>::Homomorphism;

    fn has_canonical_iso(&self, from: &Self) -> Option<Self::Isomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &Self, el: Self::Element, iso: &Self::Isomorphism) -> <Self as RingBase>::Element {
        from.map_in(self, el, iso)
    }
}

///
/// A prime ideal of a [`NumberField`].
/// 
/// Used for various implementations that work on the ring modulus prime ideals,
/// and lift the result back to the ring.
/// 
pub struct NumberRingIdeal<'ring, I>
    where I: RingStore,
        I::Type: IntegerRing,
        I: 'ring
{
    prime: <I::Type as PolyLiftFactorsDomain>::SuitableIdeal<'ring>,
    ZZX: DensePolyRing<&'ring I>,
    number_field_poly: El<DensePolyRing<&'ring I>>,
    FpX: DensePolyRing<<I::Type as PolyLiftFactorsDomain>::LocalField<'ring>>,
    Fp_as_ring: <I::Type as PolyLiftFactorsDomain>::LocalRing<'ring>,
    Fp_as_zn: AsField<<I::Type as IntegerPolyLiftFactorsDomain>::LocalRingAsZn<'ring>>,
    minpoly_factors_mod_p: Vec<El<DensePolyRing<<I::Type as PolyLiftFactorsDomain>::LocalField<'ring>>>>
}

impl<'ring, I> NumberRingIdeal<'ring, I>
    where I: RingStore,
        I::Type: IntegerRing,
        I: 'ring
{
    fn lifted_factorization<'a>(&'a self, e: usize) -> (DensePolyRing<<I::Type as PolyLiftFactorsDomain>::LocalRing<'ring>>, Vec<El<DensePolyRing<<I::Type as PolyLiftFactorsDomain>::LocalRing<'ring>>>>) {
        let ZZX = &self.ZZX;
        let ZZ = ZZX.base_ring();
        let ZpeX = DensePolyRing::new(ZZ.get_ring().quotient_ring_at(&self.prime, e, 0), "X");
        let Zpe = ZpeX.base_ring();
        let FpX = &self.FpX;
        let Zpe_to_Fp = PolyLiftFactorsDomainIntermediateReductionMap::new(ZZ.get_ring(), &self.prime, Zpe, e, &self.Fp_as_ring, 1, 0);
        let ZZ_to_Zpe = PolyLiftFactorsDomainReductionMap::new(ZZ.get_ring(), &self.prime, &Zpe, e, 0);

        let factors = hensel::hensel_lift_factorization(
            &Zpe_to_Fp,
            &ZpeX,
            FpX,
            &ZpeX.lifted_hom(ZZX, ZZ_to_Zpe).map_ref(&self.number_field_poly),
            &self.minpoly_factors_mod_p[..],
            DontObserve
        );
        
        return (ZpeX, factors);
    }
}

impl<'ring, I> Debug for NumberRingIdeal<'ring, I>
    where I: RingStore,
        I::Type: IntegerRing,
        I: 'ring,
        <I::Type as PolyLiftFactorsDomain>::SuitableIdeal<'ring>: Debug
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumberRingIdeal")
            .field("prime", &self.prime)
            .finish()
    }
}

impl<'a, Impl, I> PolyLiftFactorsDomain for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    type LocalRingBase<'ring> = FreeAlgebraImplBase<
        LocalRing<'ring, I>, 
        SparseMapVector<LocalRing<'ring, I>>
    >
        where Self: 'ring;

    type LocalRing<'ring> = RingValue<Self::LocalRingBase<'ring>>
        where Self: 'ring;

    type LocalFieldBase<'ring> = NumberFieldOrderQuotient<'ring, I>
        where Self: 'ring;

    type LocalField<'ring> = RingValue<Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type SuitableIdeal<'ring> = NumberRingIdeal<'ring, I>
        where Self: 'ring;

    fn maximal_ideal_factor_count<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>) -> usize
        where Self: 'ring
    {
        ideal.minpoly_factors_mod_p.len()
    }

    fn heuristic_exponent<'ring, 'b, J>(&self, ideal: &Self::SuitableIdeal<'ring>, poly_deg: usize, coefficients: J) -> usize
        where J: Iterator<Item = &'b Self::Element>,
            Self: 'b,
            Self: 'ring
    {
        const HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR: f64 = 0.25;

        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        // to give any mathematically justifiable value, we would probably have to consider the canonical norm;
        // I don't want to deal with this here, so let's just use the coefficient norm instead...
        let log2_max_coeff = coefficients.map(|c| self.base.wrt_canonical_basis(c).iter().map(|c| ZZ.abs_log2_ceil(QQ.get_ring().num(&c)).unwrap_or(0)).max().unwrap()).max().unwrap_or(0);
        let log2_p = BigIntRing::RING.to_float_approx(&ZZ.get_ring().principal_ideal_generator(&ideal.prime)).log2();
        return ((log2_max_coeff as f64 + poly_deg as f64 + (self.rank() as f64).log2()) / log2_p * HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR).ceil() as usize + 1;
    }

    fn random_suitable_ideal<'ring, F>(&'ring self, mut rng: F, attempt: usize) -> Self::SuitableIdeal<'ring>
        where F: FnMut() -> u64
    {
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let ZZX = DensePolyRing::new(ZZ, "X");
        let gen_poly = self.base.generating_poly(&ZZX, LambdaHom::new(QQ, ZZ, |QQ, ZZ, x| {
            assert!(ZZ.is_one(QQ.get_ring().den(&x)));
            ZZ.clone_el(QQ.get_ring().num(&x))
        }));

        // search for a prime `p` such that the minimal polynomial is unramified modulo `p`
        for _ in 0..MAX_PROBABILISTIC_REPETITIONS {
            let p = ZZ.get_ring().random_suitable_ideal(&mut rng, attempt);
            assert_eq!(1, ZZ.get_ring().maximal_ideal_factor_count(&p));

            let Fp_as_ring = ZZ.get_ring().quotient_ring_at(&p, 1, 0);
            let FpX = DensePolyRing::new(ZZ.get_ring().quotient_field_at(&p, 0), "X");
            let Fp = FpX.base_ring();
            let ZZ_to_Fp = LambdaHom::new(ZZ, Fp, |ZZ, Fp, x| ZZ.get_ring().base_ring_to_field(&p, Fp_as_ring.get_ring(), Fp.get_ring(), 0, 
                ZZ.get_ring().reduce_ring_el(&p, (Fp_as_ring.get_ring(), 1), 0, ZZ.clone_el(x))));

            let gen_poly_mod_p = FpX.from_terms(ZZX.terms(&gen_poly).map(|(c, i)| (ZZ_to_Fp.map_ref(c), i)));
            let (factorization, _) = <_ as FactorPolyField>::factor_poly(&FpX, &gen_poly_mod_p);
            if factorization.iter().all(|(_, e)| *e == 1) {
                return NumberRingIdeal {
                    minpoly_factors_mod_p: factorization.into_iter().map(|(f, _)| f).collect(),
                    number_field_poly: gen_poly,
                    FpX: FpX,
                    ZZX: ZZX,
                    Fp_as_zn: ZZ.get_ring().local_ring_into_zn(Fp_as_ring.clone()).as_field().ok().unwrap(),
                    Fp_as_ring: Fp_as_ring,
                    prime: p
                };
            }
        }
        unreachable!()
    }

    fn quotient_field_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, idx: usize) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        assert_eq!(1, ZZ.get_ring().maximal_ideal_factor_count(&ideal.prime));
        let FpX = &ideal.FpX;
        let Fp_to_Fp = WrapHom::new(ideal.Fp_as_zn.get_ring())
            .compose(RingRef::new(ideal.Fp_as_zn.get_ring().get_delegate()).into_can_hom(&ideal.Fp_as_ring).ok().unwrap())
            .compose(PolyLiftFactorsDomainBaseRingToFieldIso::new(ZZ.get_ring(), &ideal.prime, ideal.Fp_as_ring.get_ring(), FpX.base_ring().get_ring(), 0).inv());

        let irred_poly = &ideal.minpoly_factors_mod_p[idx];
        let mut x_pow_rank = (0..FpX.degree(irred_poly).unwrap()).map(|_| Fp_to_Fp.codomain().zero()).collect::<Vec<_>>();
        for (c, i) in FpX.terms(irred_poly) {
            if i < x_pow_rank.len() {
                *x_pow_rank.at_mut(i) = Fp_to_Fp.codomain().negate(Fp_to_Fp.map_ref(c));
            }
        }
        let trailing_zeros = x_pow_rank.iter().rev().take_while(|x| Fp_to_Fp.codomain().is_zero(x)).count();
        x_pow_rank.truncate(x_pow_rank.len() - trailing_zeros);
        return RingValue::from(NumberFieldOrderQuotient {
            implementation: AsField::from(AsFieldBase::promise_is_perfect_field(FreeAlgebraImpl::new(ideal.Fp_as_zn.clone(), FpX.degree(irred_poly).unwrap(), x_pow_rank))),
        });
    }

    fn quotient_ring_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, e: usize, idx: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let (ZpeX, factors) = ideal.lifted_factorization(e);
        let Zpe = ZZ.get_ring().quotient_ring_at(&ideal.prime, e, 0);
        assert!(Zpe.get_ring() == ZpeX.base_ring().get_ring());

        let irred_poly = &factors[idx];
        let degree = ZpeX.degree(irred_poly).unwrap();
        let mut x_pow_rank = SparseMapVector::new(degree, Zpe.clone());
        for (c, i) in ZpeX.terms(irred_poly) {
            if i < x_pow_rank.len() {
                *x_pow_rank.at_mut(i) = Zpe.negate(Zpe.clone_el(c));
            }
        }
        _ = x_pow_rank.at_mut(0);
        return FreeAlgebraImpl::new(Zpe, degree, x_pow_rank);
    }

    fn reduce_ring_el<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRingBase<'ring>, usize), idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let ZZX = &ideal.ZZX;
        let partial_QQ_to_ZZ = LambdaHom::new(QQ, ZZ, |QQ, ZZ, x| {
            assert!(ZZ.is_one(QQ.get_ring().den(&x)));
            ZZ.clone_el(QQ.get_ring().num(&x))
        });
        let ZZ_to_Zpe = PolyLiftFactorsDomainReductionMap::new(ZZ.get_ring(), &ideal.prime, to.0.base_ring(), to.1, 0);

        ZZX.evaluate(
            &self.base.poly_repr(ZZX, &x, partial_QQ_to_ZZ), 
            &to.0.canonical_gen(), 
            RingRef::new(to.0).into_inclusion().compose(ZZ_to_Zpe)
        )
    }

    fn base_ring_to_field<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: &Self::LocalRingBase<'ring>, to: &Self::LocalFieldBase<'ring>, idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalField<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let hom = WrapHom::new(to.base_ring().get_ring()).compose(RingRef::new(to.base_ring().get_ring().get_delegate()).into_can_hom(from.base_ring()).ok().unwrap());
        to.from_canonical_basis(from.wrt_canonical_basis(&x).iter().map(|c| hom.map(c)))
    }

    fn field_to_base_ring<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: &Self::LocalFieldBase<'ring>, to: &Self::LocalRingBase<'ring>, idx: usize, x: El<Self::LocalField<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let hom = RingRef::new(from.base_ring().get_ring().get_delegate()).into_can_iso(to.base_ring()).ok().unwrap().compose(UnwrapHom::new(from.base_ring().get_ring()));
        to.from_canonical_basis(from.wrt_canonical_basis(&x).iter().map(|c| hom.map(c)))
    }

    fn reduce_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRingBase<'ring>, usize), to: (&Self::LocalRingBase<'ring>, usize), idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| ZZ.get_ring().reduce_partial(&ideal.prime, (from.0.base_ring().get_ring(), from.1), (to.0.base_ring().get_ring(), to.1), 0, c)))
    }

    fn lift_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRingBase<'ring>, usize), to: (&Self::LocalRingBase<'ring>, usize), idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| ZZ.get_ring().lift_partial(&ideal.prime, (from.0.base_ring().get_ring(), from.1), (to.0.base_ring().get_ring(), to.1), 0, c)))
    }

    fn reconstruct_ring_el<'local, 'element, 'ring, V1, V2>(&self, ideal: &Self::SuitableIdeal<'ring>, from: V1, e: usize, x: V2) -> Self::Element
        where Self: 'ring,
            V1: VectorFn<&'local Self::LocalRing<'ring>>,
            V2: VectorFn<&'element El<Self::LocalRing<'ring>>>,
            Self::LocalRing<'ring>: 'local,
            El<Self::LocalRing<'ring>>: 'element,
            'ring: 'local + 'element
    {
        assert_eq!(self.maximal_ideal_factor_count(ideal), from.len());
        assert_eq!(self.maximal_ideal_factor_count(ideal), x.len());
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let Zpe = from.at(0).base_ring();
        assert!(from.iter().all(|ring| ring.base_ring().get_ring() == Zpe.get_ring()));
        let ZpeX = DensePolyRing::new(Zpe, "X");
        let ZZ_to_Zpe = PolyLiftFactorsDomainReductionMap::new(ZZ.get_ring(), &ideal.prime, Zpe, e, 0);

        // compute data necessary for inverse CRT
        let mut unit_vectors = (0..self.maximal_ideal_factor_count(ideal)).map(|_| ZpeX.zero()).collect::<Vec<_>>();
        product_except_one(&ZpeX, (&from).map_fn(|galois_ring| galois_ring.generating_poly(&ZpeX, Zpe.identity())), &mut unit_vectors);
        let complete_product = ZpeX.mul_ref_fst(&unit_vectors[0], from.at(0).generating_poly(&ZpeX, Zpe.identity()));
        assert_el_eq!(&ZpeX, &complete_product, &self.base.generating_poly(&ZpeX, ZZ_to_Zpe.compose(LambdaHom::new(QQ, ZZ, |QQ, ZZ, x| {
            assert!(ZZ.is_one(QQ.get_ring().den(&x)));
            ZZ.clone_el(QQ.get_ring().num(&x))
        }))));

        for i in 0..self.maximal_ideal_factor_count(ideal) {
            let galois_ring = from.at(i);
            let inv_normalization_factor = ZpeX.evaluate(unit_vectors.at(i), &galois_ring.canonical_gen(), galois_ring.inclusion());
            let normalization_factor = galois_ring.invert(&inv_normalization_factor).unwrap();
            let lifted_normalization_factor = galois_ring.poly_repr(&ZpeX, &normalization_factor, Zpe.identity());
            let unreduced_new_unit_vector = ZpeX.mul(std::mem::replace(&mut unit_vectors[i], ZpeX.zero()), lifted_normalization_factor);
            unit_vectors[i] = ZpeX.div_rem_monic(unreduced_new_unit_vector, &complete_product).1;
        }

        // now apply inverse CRT to get the value over ZpeX
        let combined = <_ as RingStore>::sum(&ZpeX, (0..self.maximal_ideal_factor_count(ideal)).map(|i| {
            let galois_ring = from.at(i);
            let unreduced_result = ZpeX.mul_ref_snd(galois_ring.poly_repr(&ZpeX, x.at(i), Zpe.identity()), &unit_vectors[i]);
            ZpeX.div_rem_monic(unreduced_result, &complete_product).1
        }));

        for i in 0..self.maximal_ideal_factor_count(ideal) {
            let galois_ring = from.at(i);
            debug_assert!(galois_ring.eq_el(x.at(i), &ZpeX.evaluate(&combined, &galois_ring.canonical_gen(), galois_ring.inclusion())));
        }

        // now lift the polynomial modulo `p^e` to the rationals
        let Zpe_as_zn = ZZ.get_ring().local_ring_as_zn(&Zpe);
        let Zpe_to_as_zn = Zpe_as_zn.can_hom(Zpe).unwrap();
        let result = self.from_canonical_basis((0..self.rank()).map(|i| {
            let (num, den) = balanced_rational_reconstruction(Zpe_as_zn, Zpe_to_as_zn.map_ref(ZpeX.coefficient_at(&combined, i)));
            return QQ.div(&QQ.inclusion().map(int_cast(num, ZZ, Zpe_as_zn.integer_ring())), &QQ.inclusion().map(int_cast(den, ZZ, Zpe_as_zn.integer_ring())));
        }));
        return result;
    }

    fn dbg_ideal<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        QQ.base_ring().get_ring().dbg_ideal(&ideal.prime, out)
    }
}

impl<Impl, I> Serialize for NumberFieldBase<Impl, I>
    where Impl: RingStore + Serialize,
        Impl::Type: Field + FreeAlgebra + SerializableElementRing,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        SerializableNewtypeStruct::new("NumberField", &self.base).serialize(serializer)
    }
}

impl<'de, Impl, I> Deserialize<'de> for NumberFieldBase<Impl, I>
    where Impl: RingStore + Deserialize<'de>,
        Impl::Type: Field + FreeAlgebra + SerializableElementRing,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        DeserializeSeedNewtypeStruct::new("NumberField", PhantomData::<Impl>).deserialize(deserializer).map(|res| NumberField::create(res).into())
    }
}

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;
#[cfg(test)]
use crate::iters::multi_cartesian_product;

#[test]
fn test_principal_ideal_ring_axioms() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&ZZX, &f);

    let elements = multi_cartesian_product([(-4..4), (-2..2)].into_iter(), |slice| K.from_canonical_basis(slice.iter().map(|x| K.base_ring().int_hom().map(*x))), |_, x| *x).collect::<Vec<_>>();

    crate::pid::generic_tests::test_principal_ideal_ring_axioms(&K, elements.iter().map(|x| K.clone_el(x)));
}

#[test]
fn test_adjoin_root() {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);
    let QQX = DensePolyRing::new(QQ, "X");
    let [f] = QQX.with_wrapped_indeterminate(|X| [2 * X.pow_ref(3) - 1]);
    let (K, a) = NumberField::adjoin_root(&QQX, &f);
    assert_el_eq!(&K, K.zero(), K.sub(K.mul(K.int_hom().map(2), K.pow(a, 3)), K.one()));
}

#[test]
fn test_poly_gcd_number_field() {
    let ZZ = BigIntRing::RING;
    let ZZX = DensePolyRing::new(ZZ, "X");

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&ZZX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let i = RingElementWrapper::new(&KY, KY.inclusion().map(K.canonical_gen()));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| [
        (Y.pow_ref(3) + 1) * (Y - &i),
        (Y.pow_ref(4) + 2) * (Y.pow_ref(2) + 1),
        Y - i
    ]);
    assert_el_eq!(&KY, &expected, <_ as PolyTFracGCDRing>::gcd(&KY, &g, &h));

    let [f] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 20 * X.pow_ref(2) + 16]);
    let K = NumberField::new(&ZZX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let [sqrt3, sqrt7] = K.with_wrapped_generator(|a| [a.pow_ref(3) / 8 - 2 * a, a.pow_ref(3) / 8 - 3 * a]);
    assert_el_eq!(&K, K.int_hom().map(3), K.pow(K.clone_el(&sqrt3), 2));
    assert_el_eq!(&K, K.int_hom().map(7), K.pow(K.clone_el(&sqrt7), 2));

    let half = RingElementWrapper::new(&KY, KY.inclusion().map(K.invert(&K.int_hom().map(2)).unwrap()));
    let sqrt3 = RingElementWrapper::new(&KY, KY.inclusion().map(sqrt3));
    let sqrt7 = RingElementWrapper::new(&KY, KY.inclusion().map(sqrt7));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| [
        Y.pow_ref(2) - &sqrt3 * Y - 1,
        Y.pow_ref(2) + &sqrt7 * Y + 1,
        Y - (sqrt3 - sqrt7) * half
    ]);
    let actual = <_ as PolyTFracGCDRing>::gcd(&KY, &g, &h);
    assert_el_eq!(&KY, &expected, &actual);
}

#[test]
#[ignore]
fn random_test_poly_gcd_number_field() {
    let ZZ = BigIntRing::RING;
    let QQ = RationalField::new(ZZ);
    let ZZX = DensePolyRing::new(ZZ, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = QQ.base_ring().int_hom().map(1000);
    let rank = 6;

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let genpoly = ZZX.from_terms((0..rank).map(|i| (ZZ.get_uniformly_random(&bound, || rng.rand_u64()), i)).chain([(ZZ.one(), rank)].into_iter()));

        let K = NumberField::new(&ZZX, &genpoly);
        let KY = DensePolyRing::new(&K, "Y");

        let mut random_element_K = || K.from_canonical_basis((0..6).map(|_| QQ.inclusion().map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64()))));
        let f = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let g = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let h = KY.from_terms((0..=4).map(|i| (random_element_K(), i)));
        // println!("Testing gcd on ({}) * ({}) and ({}) * ({})", poly_ring.formatted_el(&f), poly_ring.formatted_el(&h), poly_ring.formatted_el(&g), poly_ring.formatted_el(&h));
        let lhs = KY.mul_ref(&f, &h);
        let rhs = KY.mul_ref(&g, &h);

        let gcd = <_ as PolyTFracGCDRing>::gcd(&KY, &lhs, &rhs);
        // println!("Result {}", poly_ring.formatted_el(&gcd));

        assert!(KY.divides(&lhs, &gcd));
        assert!(KY.divides(&rhs, &gcd));
        assert!(KY.divides(&gcd, &h));
    }
}