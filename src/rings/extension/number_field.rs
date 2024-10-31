
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use factor::heuristic_factor_poly_local;
use gcd::poly_gcd_local;
use gcd_locally::*;
use sparse::SparseMapVector;
use squarefree_part::poly_power_decomposition_local;

use crate::algorithms::rational_reconstruction::rational_reconstruction;
use crate::computation::LogProgress;
use crate::delegate::DelegateRingImplEuclideanRing;
use crate::specialization::*;
use crate::algorithms::convolution::*;
use crate::algorithms::eea::signed_lcm;
use crate::algorithms::poly_factor::extension::poly_factor_extension;
use crate::algorithms::poly_gcd::*;
use crate::MAX_PROBABILISTIC_REPETITIONS;
use crate::delegate::DelegateRing;
use crate::integer::*;
use crate::pid::*;
use crate::ring::*;
use crate::rings::field::AsField;
use crate::rings::poly::*;
use crate::rings::zn::ZnRingStore;
use crate::rings::rational::*;
use crate::divisibility::*;
use crate::rings::extension::*;

use super::extension_impl::FreeAlgebraImpl;
use super::Field;
use super::FreeAlgebra;

///
/// An algebraic number field, i.e. a finite rank field extension of the rationals.
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

#[stability::unstable(feature = "enable")]
pub type DefaultNumberFieldImpl = AsField<FreeAlgebraImpl<RationalField<BigIntRing>, Vec<El<RationalField<BigIntRing>>>, Global, KaratsubaAlgorithm>>;
#[stability::unstable(feature = "enable")]
pub type NumberField<Impl = DefaultNumberFieldImpl, I = BigIntRing> = RingValue<NumberFieldBase<Impl, I>>;

impl NumberField {

    #[stability::unstable(feature = "enable")]
    pub fn new<P>(poly_ring: P, generating_poly: &El<P>) -> Self
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<BigIntRing>>
    {
        assert!(poly_ring.base_ring().is_one(poly_ring.lc(generating_poly).unwrap()));
        let rank = poly_ring.degree(generating_poly).unwrap();
        let modulus = (0..rank).map(|i| poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(poly_ring.coefficient_at(generating_poly, i)))).collect::<Vec<_>>();
        return Self::create(FreeAlgebraImpl::new_with(RingValue::from(poly_ring.base_ring().get_ring().clone()), rank, modulus, "θ", Global, STANDARD_CONVOLUTION).as_field().ok().unwrap());
    }

    #[stability::unstable(feature = "enable")]
    pub fn new_from_rational<P>(poly_ring: P, generating_poly: &El<P>) -> Self
        where P: RingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<BigIntRing>>
    {
        let QQ = poly_ring.base_ring();
        let ZZ = QQ.base_ring();
        let denominator = poly_ring.terms(generating_poly).map(|(c, _)| QQ.get_ring().den(c)).fold(
            ZZ.one(), 
            |a, b| signed_lcm(a, ZZ.clone_el(b), ZZ)
        );
        let rank = poly_ring.degree(generating_poly).unwrap();
        let new_lc = ZZ.checked_div(&ZZ.mul_ref(&denominator, QQ.get_ring().num(poly_ring.lc(generating_poly).unwrap())), QQ.get_ring().den(poly_ring.lc(generating_poly).unwrap())).unwrap();
        let new_generating_poly = poly_ring.from_terms(poly_ring.terms(generating_poly).map(|(c, i)| if i == rank {
            (ZZ.one(), rank)
        } else {
            (ZZ.checked_div(&ZZ.mul_ref_fst(&denominator, ZZ.mul_ref_fst(QQ.get_ring().num(c), ZZ.pow(ZZ.clone_el(&new_lc), rank - i - 1))), QQ.get_ring().den(c)).unwrap(), i)
        }).map(|(c, i)| (QQ.inclusion().map(c), i)));
        return Self::new(poly_ring, &new_generating_poly);
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

impl<Impl, I> FiniteRingSpecializable for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
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

impl<Impl, I> PolyGCDRing for NumberFieldBase<Impl, I>
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
        let self_ = NumberFieldByOrder { base: RingRef::new(poly_ring.base_ring().get_ring()) };

        let order_poly_ring = DensePolyRing::new(RingRef::new(&self_), "X");
        let lhs_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, lhs);
        let rhs_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, rhs);

        let result = poly_gcd_local(&order_poly_ring, lhs_order, rhs_order, LogProgress);

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

        let result = poly_power_decomposition_local(&order_poly_ring, poly_order, LogProgress);

        return result.into_iter().map(|(f, k)| (self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &f), k)).collect();
    }
}

impl<Impl, I> FactorPolyField for NumberFieldBase<Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn factor_poly<P>(poly_ring: P, poly: &El<P>) -> (Vec<(El<P>, usize)>, Self::Element)
        where P: PolyRingStore + Copy,
            P::Type: PolyRing + EuclideanRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = Self>
    {
        let self_ = NumberFieldByOrder { base: RingRef::new(poly_ring.base_ring().get_ring()) };
        let order_poly_ring = DensePolyRing::new(RingRef::new(&self_), "X");
        let poly_order = self_.scale_poly_to_order(poly_ring, &order_poly_ring, poly);

        let mut result = Vec::new();
        for (factor, e1) in heuristic_factor_poly_local(&order_poly_ring, poly_order, 1., LogProgress) {
            for (irred_factor, e2) in poly_factor_extension(&poly_ring, &self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &factor)).0 {
                result.push((irred_factor, e1 * e2));
            }
        }
        return (result, self_.clone_el(poly_ring.lc(poly).unwrap()));
    }
}

///
/// Implements [`PolyGCDLocallyDomain`] for [`NumberField`].
/// 
/// We don't want to expose the interface of [`PolyGCDLocallyDomain`] for number
/// fields generally, thus use a private newtype.
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
            self.base.wrt_canonical_basis(c).iter().map(|c| ZZ.clone_el(QQ.get_ring().den(&c))).fold(ZZ.one(), |a, b| signed_lcm(a, b, ZZ))
        ).fold(ZZ.one(), |a, b| signed_lcm(a, b, ZZ)));
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

impl<'a, Impl, I> FiniteRingSpecializable for NumberFieldByOrder<'a, Impl, I>
    where Impl: RingStore,
        Impl::Type: Field + FreeAlgebra,
        <Impl::Type as RingExtension>::BaseRing: RingStore<Type = RationalFieldBase<I>>,
        I: RingStore,
        I::Type: IntegerRing
{
    fn specialize<O: FiniteRingOperation<Self>>(_op: O) -> Result<O::Output, ()> {
        Err(())
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

type LocalRing<'ring, I> = <<I as RingStore>::Type as PolyGCDLocallyDomain>::LocalRing<'ring>;
type LocalField<'ring, I> = <<I as RingStore>::Type as PolyGCDLocallyDomain>::LocalField<'ring>;

impl<'a, Impl, I> PolyGCDLocallyDomain for NumberFieldByOrder<'a, Impl, I>
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

    type LocalFieldBase<'ring> = AsFieldBase<FreeAlgebraImpl<
        LocalField<'ring, I>, 
        SparseMapVector<LocalField<'ring, I>>
    >>
        where Self: 'ring;

    type LocalField<'ring> = RingValue<Self::LocalFieldBase<'ring>>
        where Self: 'ring;

    type MaximalIdeal<'ring> = <<I as RingStore>::Type as PolyGCDLocallyDomain>::MaximalIdeal<'ring>
        where Self: 'ring;

    fn heuristic_exponent<'ring, 'b, J>(&self, p: &Self::MaximalIdeal<'ring>, poly_deg: usize, coefficients: J) -> usize
        where J: Iterator<Item = &'b Self::Element>,
            Self: 'b,
            Self: 'ring
    {
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        // to give any mathematically justifiable value, we would probably have to consider the canonical norm;
        // I don't want to deal with this here, so let's just use the coefficient norm instead...
        let log2_max_coeff = coefficients.map(|c| self.base.wrt_canonical_basis(c).iter().map(|c| ZZ.abs_log2_ceil(QQ.get_ring().num(&c)).unwrap_or(0)).max().unwrap()).max().unwrap_or(0);
        return ((log2_max_coeff as f64 + poly_deg as f64 + (self.rank() as f64).log2()) / (ZZ.get_ring().maximal_ideal_gen(p) as f64).log2()).ceil() as usize + 1;
    }

    fn random_maximal_ideal<'ring, F>(&'ring self, mut rng: F) -> Self::MaximalIdeal<'ring>
        where F: FnMut() -> u64
    {
        let QQ = self.base.base_ring();
        let poly_ring = DensePolyRing::new(QQ, "X");
        let gen_poly = self.base.generating_poly(&poly_ring, QQ.identity());
        debug_assert!(poly_ring.terms(&gen_poly).all(|(c, _)| QQ.base_ring().is_one(QQ.get_ring().den(c))));
        for _ in 0..(MAX_PROBABILISTIC_REPETITIONS * self.base.rank()) {
            let p = QQ.base_ring().get_ring().random_maximal_ideal(&mut rng);
            let local_field = QQ.base_ring().get_ring().local_field_at(&p);
            let local_ring = QQ.base_ring().get_ring().local_ring_at(&p, 1);
            let local_poly_ring = DensePolyRing::new(&local_field, "X");
            let red_map = local_field.can_hom(&local_ring).unwrap().compose(ReductionMap::new(QQ.base_ring().get_ring(), &p, 1));
            let gen_poly_mod = local_poly_ring.from_terms(poly_ring.terms(&gen_poly).map(|(c, i)| (red_map.map_ref(QQ.get_ring().num(c)), i)));
            if <_ as FactorPolyField>::is_irred(&local_poly_ring, &gen_poly_mod) {
                return p;
            }
        }
        unreachable!()
    }

    fn local_field_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        let poly_ring = DensePolyRing::new(QQ, "X");
        let gen_poly = self.base.generating_poly(&poly_ring, QQ.identity());
        debug_assert!(poly_ring.terms(&gen_poly).all(|(c, _)| QQ.base_ring().is_one(QQ.get_ring().den(c))));

        let local_field = QQ.base_ring().get_ring().local_field_at(&p);
        let red_map_base = ReductionMap::new(QQ.base_ring().get_ring(), &p, 1);
        let red_map = local_field.can_hom(red_map_base.codomain()).unwrap().compose(&red_map_base);
        let mut x_pow_rank = SparseMapVector::new(self.base.rank(), QQ.base_ring().get_ring().local_field_at(&p));
        for (c, i) in poly_ring.terms(&gen_poly) {
            if i < self.rank() {
                *x_pow_rank.at_mut(i) = red_map.codomain().negate(red_map.map_ref(QQ.get_ring().num(c)));
            }
        }
        x_pow_rank.at_mut(0);
        return AsField::from(AsFieldBase::promise_is_perfect_field(FreeAlgebraImpl::new(local_field, self.rank(), x_pow_rank)));
    }

    fn local_ring_at<'ring>(&self, p: &Self::MaximalIdeal<'ring>, e: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        let poly_ring = DensePolyRing::new(QQ, "X");
        let gen_poly = self.base.generating_poly(&poly_ring, QQ.identity());
        debug_assert!(poly_ring.terms(&gen_poly).all(|(c, _)| QQ.base_ring().is_one(QQ.get_ring().den(c))));

        let local_ring = QQ.base_ring().get_ring().local_ring_at(&p, e);
        let red_map = ReductionMap::new(QQ.base_ring().get_ring(), &p, e);
        let mut x_pow_rank = SparseMapVector::new(self.base.rank(), QQ.base_ring().get_ring().local_ring_at(&p, e));
        for (c, i) in poly_ring.terms(&gen_poly) {
            if i < self.rank() {
                *x_pow_rank.at_mut(i) = red_map.codomain().negate(red_map.map_ref(QQ.get_ring().num(c)));
            }
        }
        x_pow_rank.at_mut(0);
        return FreeAlgebraImpl::new(local_ring, self.rank(), x_pow_rank);
    }

    fn reduce_ring_el<'ring>(&self, p: &Self::MaximalIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), x: Self::Element) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        to.0.from_canonical_basis(self.base.wrt_canonical_basis(&x).iter().map(|c| {
            debug_assert!(QQ.base_ring().is_one(QQ.get_ring().den(&c)));
            QQ.base_ring().get_ring().reduce_ring_el(p, (to.0.base_ring(), to.1), QQ.base_ring().clone_el(QQ.get_ring().num(&c)))
        }))
    }

    fn reduce_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| QQ.base_ring().get_ring().reduce_partial(p, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), c)))
    }

    fn lift_partial<'ring>(&self, p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| QQ.base_ring().get_ring().lift_partial(p, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), c)))
    }

    fn reconstruct_ring_el<'ring>(&self, _p: &Self::MaximalIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), x: El<Self::LocalRing<'ring>>) -> Self::Element
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let local_ring_as_zn = ZZ.get_ring().local_ring_as_zn(from.0.base_ring());
        let hom = local_ring_as_zn.can_hom(from.0.base_ring()).unwrap();
        self.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| {
            let (n, d) = rational_reconstruction(&local_ring_as_zn, hom.map(c));
            QQ.div(&QQ.inclusion().map(int_cast(n, ZZ, local_ring_as_zn.integer_ring())), &QQ.inclusion().map(int_cast(d, ZZ, local_ring_as_zn.integer_ring())))
        }))
    }

    fn dbg_maximal_ideal<'ring>(&self, p: &Self::MaximalIdeal<'ring>, out: &mut std::fmt::Formatter) -> std::fmt::Result
        where Self: 'ring
    {
        let QQ = self.base.base_ring();
        QQ.base_ring().get_ring().dbg_maximal_ideal(p, out)
    }
}

#[test]
fn test_poly_gcd_number_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let QQX = DensePolyRing::new(QQ, "X");

    let [f] = QQX.with_wrapped_indeterminate(|X| [X.pow_ref(2) + 1]);
    let K = NumberField::new(&QQX, &f);
    let KY = DensePolyRing::new(&K, "Y");

    let i = RingElementWrapper::new(&KY, KY.inclusion().map(K.canonical_gen()));
    let [g, h, expected] = KY.with_wrapped_indeterminate(|Y| [
        (Y.pow_ref(3) + 1) * (Y - &i),
        (Y.pow_ref(4) + 2) * (Y.pow_ref(2) + 1),
        Y - i
    ]);
    assert_el_eq!(&KY, &expected, <_ as PolyGCDRing>::gcd(&KY, &g, &h));

    let [f] = QQX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 20 * X.pow_ref(2) + 16]);
    let K = NumberField::new(&QQX, &f);
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
        Y - (sqrt3 + sqrt7) * half
    ]);
    assert_el_eq!(&KY, &expected, <_ as PolyGCDRing>::gcd(&KY, &g, &h));
}