
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use gcd::poly_gcd_local;
use gcd_locally::*;
use hensel::hensel_lift_factorization;
use sparse::SparseMapVector;
use squarefree_part::poly_power_decomposition_local;

use crate::algorithms::interpolate::product_except_one;
use crate::algorithms::poly_factor::finite::poly_factor_finite_field;
use crate::algorithms::rational_reconstruction::rational_reconstruction;
use crate::computation::*;
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

        let result = poly_gcd_local(&order_poly_ring, order_poly_ring.clone_el(&lhs_order), order_poly_ring.clone_el(&rhs_order), LogProgress);

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
        for (irred_factor, e) in poly_factor_extension(&poly_ring, &self_.normalize_map_back_from_order(&order_poly_ring, poly_ring, &poly_order)).0 {
            result.push((irred_factor, e));
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

pub struct NumberRingIdeal<'ring, I>
    where I: RingStore,
        I::Type: IntegerRing,
        I: 'ring
{
    prime: <I::Type as PolyGCDLocallyDomain>::SuitableIdeal<'ring>,
    ZZX: DensePolyRing<&'ring I>,
    number_field_poly: El<DensePolyRing<&'ring I>>,
    FpX: DensePolyRing<<I::Type as PolyGCDLocallyDomain>::LocalField<'ring>>,
    Fp_as_ring: <I::Type as PolyGCDLocallyDomain>::LocalRing<'ring>,
    minpoly_factors_mod_p: Vec<El<DensePolyRing<<I::Type as PolyGCDLocallyDomain>::LocalField<'ring>>>>
}

impl<'ring, I> NumberRingIdeal<'ring, I>
    where I: RingStore,
        I::Type: IntegerRing,
        I: 'ring
{
    fn lifted_factorization<'a>(&'a self, e: usize) -> (DensePolyRing<<I::Type as PolyGCDLocallyDomain>::LocalRing<'ring>>, Vec<El<DensePolyRing<<I::Type as PolyGCDLocallyDomain>::LocalRing<'ring>>>>) {
        let ZZX = &self.ZZX;
        let ZZ = ZZX.base_ring();
        let ZpeX = DensePolyRing::new(ZZ.get_ring().local_ring_at(&self.prime, e, 0), "X");
        let Zpe = ZpeX.base_ring();
        let FpX = &self.FpX;
        let Zpe_to_Fp = IntermediateReductionMap::new(ZZ.get_ring(), &self.prime, Zpe, e, &self.Fp_as_ring, 1, 0);
        let ZZ_to_Zpe = ReductionMap::new(ZZ.get_ring(), &self.prime, &Zpe, e, 0);

        let factors = hensel_lift_factorization(
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
        let log2_p = (ZZ.get_ring().principal_ideal_generator(&ideal.prime) as f64).log2();
        return ((log2_max_coeff as f64 + poly_deg as f64 + (self.rank() as f64).log2()) / log2_p * HEURISTIC_FACTOR_SIZE_OVER_POLY_SIZE_FACTOR).ceil() as usize + 1;
    }

    fn random_suitable_ideal<'ring, F>(&'ring self, mut rng: F) -> Self::SuitableIdeal<'ring>
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
            let p = ZZ.get_ring().random_suitable_ideal(&mut rng);
            assert_eq!(1, ZZ.get_ring().maximal_ideal_factor_count(&p));

            let Fp_as_ring = ZZ.get_ring().local_ring_at(&p, 1, 0);
            let FpX = DensePolyRing::new(ZZ.get_ring().local_field_at(&p, 0), "X");
            let Fp = FpX.base_ring();
            let ZZ_to_Fp = Fp.can_hom(&Fp_as_ring).unwrap().compose(ReductionMap::new(ZZ.get_ring(), &p, &Fp_as_ring, 1, 0));

            let gen_poly_mod_p = FpX.from_terms(ZZX.terms(&gen_poly).map(|(c, i)| (ZZ_to_Fp.map_ref(c), i)));
            let (factorization, _) = poly_factor_finite_field(&FpX, &gen_poly_mod_p);
            if factorization.iter().all(|(_, e)| *e == 1) {

                return NumberRingIdeal {
                    minpoly_factors_mod_p: factorization.into_iter().map(|(f, _)| f).collect(),
                    number_field_poly: gen_poly,
                    FpX: FpX,
                    ZZX: ZZX,
                    Fp_as_ring: Fp_as_ring,
                    prime: p
                };
            }
        }
        unreachable!()
    }

    fn local_field_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, idx: usize) -> Self::LocalField<'ring>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let Fp = ZZ.get_ring().local_field_at(&ideal.prime, 0);
        let FpX = &ideal.FpX;
        assert!(Fp.get_ring() == FpX.base_ring().get_ring());

        let irred_poly = &ideal.minpoly_factors_mod_p[idx];
        let mut x_pow_rank = SparseMapVector::new(FpX.degree(irred_poly).unwrap(), ZZ.get_ring().local_field_at(&ideal.prime, 0));
        for (c, i) in FpX.terms(irred_poly) {
            if i < x_pow_rank.len() {
                *x_pow_rank.at_mut(i) = Fp.negate(Fp.clone_el(c));
            }
        }
        x_pow_rank.at_mut(0);
        return AsField::from(AsFieldBase::promise_is_perfect_field(FreeAlgebraImpl::new(Fp, FpX.degree(irred_poly).unwrap(), x_pow_rank)));
    }

    fn local_ring_at<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, e: usize, idx: usize) -> Self::LocalRing<'ring>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        let (ZpeX, factors) = ideal.lifted_factorization(e);
        let Zpe = ZpeX.base_ring();

        let irred_poly = &factors[idx];
        let degree = ZpeX.degree(irred_poly).unwrap();
        let mut x_pow_rank = SparseMapVector::new(degree, ZZ.get_ring().local_ring_at(&ideal.prime, e, 0));
        for (c, i) in ZpeX.terms(irred_poly) {
            if i < x_pow_rank.len() {
                *x_pow_rank.at_mut(i) = Zpe.negate(Zpe.clone_el(c));
            }
        }
        x_pow_rank.at_mut(0);
        return FreeAlgebraImpl::new(ZZ.get_ring().local_ring_at(&ideal.prime, e, 0), degree, x_pow_rank);
    }

    fn reduce_ring_el<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, to: (&Self::LocalRing<'ring>, usize), idx: usize, x: Self::Element) -> El<Self::LocalRing<'ring>>
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
        let ZZ_to_Zpe = ReductionMap::new(ZZ.get_ring(), &ideal.prime, to.0.base_ring(), to.1, 0);

        ZZX.evaluate(
            &self.base.poly_repr(ZZX, &x, partial_QQ_to_ZZ), 
            &to.0.canonical_gen(), 
            to.0.inclusion().compose(ZZ_to_Zpe)
        )
    }

    fn reduce_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| ZZ.get_ring().reduce_partial(&ideal.prime, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), 0, c)))
    }

    fn lift_partial<'ring>(&self, ideal: &Self::SuitableIdeal<'ring>, from: (&Self::LocalRing<'ring>, usize), to: (&Self::LocalRing<'ring>, usize), idx: usize, x: El<Self::LocalRing<'ring>>) -> El<Self::LocalRing<'ring>>
        where Self: 'ring
    {
        assert!(idx < self.maximal_ideal_factor_count(ideal));
        let QQ = self.base.base_ring();
        let ZZ = QQ.base_ring();
        to.0.from_canonical_basis(from.0.wrt_canonical_basis(&x).iter().map(|c| ZZ.get_ring().lift_partial(&ideal.prime, (from.0.base_ring(), from.1), (to.0.base_ring(), to.1), 0, c)))
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
        let ZZ_to_Zpe = ReductionMap::new(ZZ.get_ring(), &ideal.prime, Zpe, e, 0);

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
            let (num, den) = rational_reconstruction(Zpe_as_zn, Zpe_to_as_zn.map_ref(ZpeX.coefficient_at(&combined, i)));
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

#[cfg(test)]
use crate::RANDOM_TEST_INSTANCE_COUNT;

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
        Y - (sqrt3 - sqrt7) * half
    ]);
    let actual = <_ as PolyGCDRing>::gcd(&KY, &g, &h);
    assert_el_eq!(&KY, &expected, &actual);
}

#[test]
fn random_test_poly_gcd_number_field() {
    let QQ = RationalField::new(BigIntRing::RING);
    let QQX = DensePolyRing::new(&QQ, "X");
    let mut rng = oorandom::Rand64::new(1);
    let bound = QQ.base_ring().int_hom().map(1000);
    let rank = 6;

    for _ in 0..RANDOM_TEST_INSTANCE_COUNT {
        let genpoly = QQX.from_terms((0..rank).map(|i| (QQ.inclusion().map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64())), i)).chain([(QQ.one(), rank)].into_iter()));
        let K = NumberField::new(&QQX, &genpoly);
        let KY = DensePolyRing::new(&K, "Y");

        let mut random_element_K = || K.from_canonical_basis((0..6).map(|_| QQ.inclusion().map(QQ.base_ring().get_uniformly_random(&bound, || rng.rand_u64()))));
        let f = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let g = KY.from_terms((0..=5).map(|i| (random_element_K(), i)));
        let h = KY.from_terms((0..=4).map(|i| (random_element_K(), i)));
        // println!("Testing gcd on ({}) * ({}) and ({}) * ({})", poly_ring.format(&f), poly_ring.format(&h), poly_ring.format(&g), poly_ring.format(&h));
        let lhs = KY.mul_ref(&f, &h);
        let rhs = KY.mul_ref(&g, &h);
        let gcd = <_ as PolyGCDRing>::gcd(&KY, &lhs, &rhs);
        // println!("Result {}", poly_ring.format(&gcd));

        assert!(KY.divides(&lhs, &gcd));
        assert!(KY.divides(&rhs, &gcd));
        assert!(KY.divides(&gcd, &h));
    }
}