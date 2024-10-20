use std::alloc::Allocator;
use std::alloc::Global;

use extension_impl::FreeAlgebraImplBase;
use sparse::SparseMapVector;
use zn_64::Zn;

use crate::algorithms::convolution::ConvolutionAlgorithm;
use crate::algorithms::convolution::KaratsubaAlgorithm;
use crate::algorithms::convolution::KaratsubaHint;
use crate::algorithms::convolution::STANDARD_CONVOLUTION;
use crate::algorithms::eea::signed_gcd;
use crate::algorithms::int_factor::factor;
use crate::algorithms::int_factor::is_prime_power;
use crate::algorithms::matmul::ComputeInnerProduct;
use crate::algorithms::matmul::StrassenHint;
use crate::algorithms::poly_factor::finite_field::poly_squarefree_part_finite_field;
use crate::algorithms::unity_root::*;
use crate::delegate::DelegateRing;
use crate::divisibility::DivisibilityRingStore;
use crate::divisibility::Domain;
use crate::field::*;
use crate::pid::PrincipalIdealRing;
use crate::rings::extension::extension_impl::FreeAlgebraImpl;
use crate::rings::finite::*;
use crate::algorithms::convolution::fft::FFTBasedConvolution;
use crate::algorithms::convolution::fft::FFTBasedConvolutionZn;
use crate::algorithms::poly_factor::cantor_zassenhaus;
use crate::pid::EuclideanRing;
use crate::primitive_int::StaticRing;
use crate::primitive_int::StaticRingBase;
use crate::rings::field::AsField;
use crate::rings::field::AsFieldBase;
use crate::rings::local::AsLocalPIR;
use crate::rings::local::AsLocalPIRBase;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::rings::poly::PolyRing;
use crate::rings::zn::*;
use crate::ring::*;
use crate::rings::extension::*;
use crate::integer::*;

fn filter_irreducible<R, P>(poly_ring: P, mod_f_ring: R, degree: usize) -> Option<El<P>>
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + FiniteRing + Field,
        R: RingStore,
        R::Type: FreeAlgebra,
        <R::Type as RingExtension>::BaseRing: RingStore<Type = <<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let f = mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity());
    let squarefree_part = poly_squarefree_part_finite_field(&poly_ring, &f);
    if poly_ring.degree(&squarefree_part) != Some(degree) {
        return None;
    }
    let distinct_degree_factorization = cantor_zassenhaus::distinct_degree_factorization_base(&poly_ring, &mod_f_ring);
    if distinct_degree_factorization.len() <= degree || poly_ring.degree(&distinct_degree_factorization[degree]) != Some(degree) {
        return None;
    }
    return Some(mod_f_ring.generating_poly(&poly_ring, &poly_ring.base_ring().identity()));
}

///
/// Tries to find a "small" irreducible polynomial of given degree.
/// 
/// "small" is not well-defined, but can mean sparse, or having a small `deg(f - lt(f))`.
/// Both will lead to more efficient arithmetic in the ring `k[X]/(f)`. However, in general
/// finding a very small `f` is hard, thus we use heuristics.
/// 
fn find_small_irreducible_poly_base<P, C>(poly_ring: P, degree: usize, convolution: C, rng: &mut oorandom::Rand64) -> El<P>
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: Copy,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + FiniteRing + Field + CanHomFrom<StaticRingBase<i64>>,
        C: ConvolutionAlgorithm<<<P::Type as RingExtension>::BaseRing as RingStore>::Type>
{
    let Fp = *poly_ring.base_ring();
    let create_mod_f_ring = |f: &El<P>| {
        let mut f_body = SparseMapVector::new(degree, poly_ring.base_ring());
        for (c, i) in poly_ring.terms(f) {
            if i != degree {
                *f_body.at_mut(i) = poly_ring.base_ring().negate(poly_ring.base_ring().clone_el(c));
            }
        }
        return FreeAlgebraImpl::new_with(Fp, degree, f_body, "θ", Global, &convolution);
    };

    if degree > 3 {

        // first try trinomials, they seem to have a good chance of being irreducible
        for _ in 0..16 {
            let i = (StaticRing::<i64>::RING.get_uniformly_random(&(degree as i64 - 1), || rng.rand_u64()) + 1) as usize;
            let a = Fp.random_element(|| rng.rand_u64());
            let b = Fp.random_element(|| rng.rand_u64());
            let f = poly_ring.from_terms([(a, 0), (b, i), (Fp.one(), degree)].into_iter());
            if let Some(result)  = filter_irreducible(&poly_ring, create_mod_f_ring(&f), degree) {
                return result;
            }
        }
    
        // next, check for cases where we can take `small_poly(X^high_power)`; these cases are characterized by the
        // fact that we have `degree = small_d * large_d` with `large_d | #F_(q^small_d)*`
        let ZZbig = BigIntRing::RING;
        let ZZ = StaticRing::<i64>::RING;
    
        let p = Fp.size(&ZZbig).unwrap();
        let mut small_d = 1;
        let Fq_star_order = ZZbig.sub(ZZbig.pow(ZZbig.clone_el(&p), small_d as usize), ZZbig.one());
        let mut large_d = int_cast(signed_gcd(Fq_star_order, int_cast(degree as i64 / small_d, ZZbig, StaticRing::<i64>::RING), ZZbig), ZZ, ZZbig);
        let mut increased_small_d = true;
        while increased_small_d {
            increased_small_d = false;
            // use a greedy approach, just add a factor to small_d if it makes large_d larger
            for (k, _) in factor(&ZZ, degree as i64 / small_d) {
                let new_small_d = small_d * k;
                let Fq_star_order = ZZbig.sub(ZZbig.pow(ZZbig.clone_el(&p), new_small_d as usize), ZZbig.one());
                let new_large_d = int_cast(signed_gcd(Fq_star_order, int_cast(degree as i64 / new_small_d, ZZbig, StaticRing::<i64>::RING), ZZbig), ZZ, ZZbig);
                if new_large_d > large_d {
                    small_d = new_small_d;
                    large_d = new_large_d;
                    increased_small_d = true;
                    break;
                }
            }
        }
        small_d = degree as i64 / large_d;
        if large_d != 1 {
            let Fq_star_order = ZZbig.sub(ZZbig.pow(ZZbig.clone_el(&p), small_d as usize), ZZbig.one());
            // careful here to not get an infinite generic argument recursion
            let Fq = GaloisField::new_internal(Fp, small_d as usize, Global, &convolution);
            // the primitive element of `Fq` clearly has no `k`-th root, for every `k | large_d` since `large_d | #Fq*`;
            // hence we can use `MinPoly(primitive_element)(X^large_d)`
            let primitive_element = if is_prim_root_of_unity_gen(&Fq, &Fq.canonical_gen(), &Fq_star_order, ZZbig) {
                Fq.canonical_gen()
            } else {
                get_prim_root_of_unity_gen(&Fq, &Fq_star_order, ZZbig).unwrap()
            };
            // I thought for a while that it would be enough to have a primitive `lcm(Fq_star_order, large_d^inf)`-th root of unity,
            // however it is not guaranteed that this would indeed generate the field
            let FqX = DensePolyRing::new(&Fq, "X");
            let minpoly = FqX.prod((0..small_d).map(|i| FqX.from_terms([(Fq.negate(Fq.pow_gen(Fq.clone_el(&primitive_element), &ZZbig.pow(ZZbig.clone_el(&p), i as usize), ZZbig)), 0), (Fq.one(), 1)].into_iter())));
            let minpoly_Fp = poly_ring.from_terms(FqX.terms(&minpoly).map(|(c, i)| {
                let c_wrt_basis = Fq.wrt_canonical_basis(c);
                assert!(c_wrt_basis.iter().skip(1).all(|x| Fp.is_zero(&x)));
                return (c_wrt_basis.at(0), i);
            }));
            let f = poly_ring.evaluate(&minpoly_Fp, &poly_ring.from_terms([(Fp.one(), large_d as usize)].into_iter()), &poly_ring.inclusion());
            return f;
        }
    }

    // fallback, just generate a random irreducible polynomial
    loop {
        let f = poly_ring.from_terms((0..degree).map(|i| (Fp.random_element(|| rng.rand_u64()), i)).chain([(Fp.one(), degree)].into_iter()));
        if let Some(result) = filter_irreducible(&poly_ring, create_mod_f_ring(&f), degree) {
            return result;
        }
    }
}

fn find_small_irreducible_poly<P>(poly_ring: P, degree: usize, rng: &mut oorandom::Rand64) -> El<P>
    where P: RingStore,
        P::Type: PolyRing + EuclideanRing,
        <P::Type as RingExtension>::BaseRing: Copy,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + FiniteRing + Field + CanHomFrom<StaticRingBase<i64>>
{
    let log2_modulus = poly_ring.base_ring().integer_ring().abs_log2_ceil(poly_ring.base_ring().modulus()).unwrap();
    let fft_convolution = FFTBasedConvolution::new_with(Global);
    if fft_convolution.can_compute(StaticRing::<i64>::RING.abs_log2_ceil(&(degree as i64)).unwrap() + 1, log2_modulus) {
        find_small_irreducible_poly_base(
            &poly_ring,
            degree,
            FFTBasedConvolutionZn::from(fft_convolution),
            rng
        )
    } else {
        find_small_irreducible_poly_base(
            &poly_ring,
            degree,
            STANDARD_CONVOLUTION,
            rng
        )
    }
}

///
/// Implementation of a galois field `GF(p^e)`; Also known as galois field, and sometimes denoted by `Fq` where `q = p^e`. 
/// 
/// There exists a finite field with `q` elements if and only if `q = p^e` is a prime power. In these cases,
/// this struct provides an implementation of arithmetic in these fields. Note that since those fields are always finite-degree
/// extensions of `Z/pZ`, they can also be used by creating a suitable instance of [`super::extension_impl::FreeAlgebraImpl`]. In fact,
/// this is the way the old implementation of galois fields was designed. However, providing a new type can provide both 
/// ergonomic and performance benefits, and also gives better encapsulation.
/// 
/// # Example
/// 
/// The easiest way to create a Galois field is by using `new()`:
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let F25: GaloisField = GaloisField::new(5, 2);
/// assert_eq!(5, F25.characteristic(&StaticRing::<i64>::RING).unwrap());
/// assert_eq!(2, F25.rank());
/// ```
/// More configurations are possible using [`GaloisField::new_with()`] or [`GaloisField::create()`].
/// 
/// We also support conversion to and from a plain [`super::extension_impl::FreeAlgebraImpl`] representation.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::extension::extension_impl::*;
/// # use feanor_math::rings::field::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let F25: GaloisField = GaloisField::new(5, 2);
/// let raw_F25: AsField<FreeAlgebraImpl<_, _>> = F25.clone().into().unwrap_self();
/// assert!(F25.can_iso(&raw_F25).is_some());
/// ```
/// The other way is slightly more dangerous, since at some point we either have to check, or assume
/// that the extension ring is indeed a field.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::extension::*;
/// # use feanor_math::rings::extension::extension_impl::*;
/// # use feanor_math::rings::field::*;
/// # use feanor_math::rings::finite::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::rings::extension::galois_field::*;
/// let base_ring = Zn::new(5).as_field().ok().unwrap();
/// let raw_F25: FreeAlgebraImpl<_, _> = FreeAlgebraImpl::new(base_ring, 2, [base_ring.int_hom().map(2)]);
/// let asfield_F25 = raw_F25.clone().as_field().ok().unwrap();
/// // alternatively, you can ensure yourself that the ring is a field and use `promise_is_field` to avoid the check at runtime; be careful when doing this!
/// let asfield_F25 = AsField::from(AsFieldBase::promise_is_field(raw_F25).ok().unwrap());
/// let F25 = GaloisField::create(asfield_F25);
/// assert!(F25.can_iso(&raw_F25).is_some());
/// ```
/// 
#[repr(transparent)]
pub struct GaloisFieldBase<R = AsField<Zn>, V = SparseMapVector<R>, A = Global, C = KaratsubaAlgorithm>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    base: FreeAlgebraImpl<R, V, A, C>
}

///
/// Implementation of finite/galois fields.
/// 
/// This is the [`RingStore`] corresponding to [`GaloisFieldBase`]. For more details, see [`GaloisFieldBase`].
/// 
pub type GaloisField<R = AsField<Zn>, V = SparseMapVector<R>, A = Global, C = KaratsubaAlgorithm> = RingValue<GaloisFieldBase<R, V, A, C>>;

impl GaloisField {

    ///
    /// Creates a new instance of the finite/galois field `GF(p^degree)`.
    /// 
    /// If you need more options for configuration, consider using [`GaloisField::new_with()`] or
    /// [`GaloisField::create()`].
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::extension::*;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::extension::galois_field::*;
    /// let F25 = GaloisField::new(5, 2);
    /// let generator = F25.canonical_gen();
    /// let norm = F25.mul_ref_fst(&generator, F25.pow(F25.clone_el(&generator), 5));
    /// let inclusion = F25.inclusion();
    /// // the norm must be an element of the prime field
    /// assert!(F25.base_ring().elements().any(|x| {
    ///     F25.eq_el(&norm, &inclusion.map(x))
    /// }));
    /// ```
    /// 
    pub fn new(p: i64, degree: usize) -> Self {
        Self::new_with(Zn::new(p as u64).as_field().ok().unwrap(), degree, Global, STANDARD_CONVOLUTION)
    }
}

impl<R, A, C> GaloisField<R, SparseMapVector<R>, A, C>
    where R: RingStore + Clone,
        R::Type: ZnRing + FiniteRing + Field + CanHomFrom<StaticRingBase<i64>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    ///
    /// Creates a new instance of a finite/galois field, as given-degree extension of the
    /// given base ring. The base ring must have prime characteristic.
    /// 
    /// If you need to specify the minimal polynomial used, see also [`GaloisField::create()`].
    /// 
    /// # Example
    /// 
    /// Sometimes it is useful to have the base ring also be a field. This can e.g. be achieved
    /// by
    /// ```
    /// #![feature(allocator_api)]
    /// # use std::alloc::Global;
    /// # use feanor_math::ring::*;
    /// # use feanor_math::rings::extension::*;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::algorithms::convolution::*;
    /// # use feanor_math::rings::extension::galois_field::*;
    /// let F25 = GaloisField::new_with(Zn::new(5).as_field().ok().unwrap(), 2, Global, STANDARD_CONVOLUTION);
    /// let generator = F25.canonical_gen();
    /// let norm = F25.mul_ref_fst(&generator, F25.pow(F25.clone_el(&generator), 5));
    /// let inclusion = F25.inclusion();
    /// // the norm must be an element of the prime field
    /// assert!(F25.base_ring().elements().any(|x| {
    ///     F25.eq_el(&norm, &inclusion.map(x))
    /// }));
    /// ```
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn new_with(base_field: R, degree: usize, allocator: A, convolution_algorithm: C) -> Self {
        assert!(degree >= 1);
        let poly_ring = DensePolyRing::new(&base_field, "X");
        let mut rng = oorandom::Rand64::new(poly_ring.base_ring().integer_ring().default_hash(poly_ring.base_ring().modulus()) as u128);
        let modulus = find_small_irreducible_poly(&poly_ring, degree, &mut rng);
        let mut modulus_vec = SparseMapVector::new(degree, base_field.clone());
        for (c, i) in poly_ring.terms(&modulus) {
            if i != degree {
                *modulus_vec.at_mut(i) = base_field.negate(base_field.clone_el(c));
            }
        }
        return RingValue::from(GaloisFieldBase { 
            base: FreeAlgebraImpl::new_with(base_field, degree, modulus_vec, "θ", allocator, convolution_algorithm)
        });
    }

    ///
    /// Instantiates the galois field by calling `find_small_irreducible_poly()` with a poly ring whose base ring
    /// is exactly `R`; this prevents infinite generic argument recursion, which might otherwise happen if each
    /// recursive call adds a reference `&` to `R`.
    /// 
    fn new_internal(base_ring: R, degree: usize, allocator: A, convolution_algorithm: C) -> Self
        where R: Copy, R::Type: Field
    {
        assert!(degree >= 1);
        let poly_ring = DensePolyRing::new(base_ring.clone(), "X");
        let mut rng = oorandom::Rand64::new(poly_ring.base_ring().integer_ring().default_hash(poly_ring.base_ring().modulus()) as u128);
        let modulus = find_small_irreducible_poly(&poly_ring, degree, &mut rng);
        let mut modulus_vec = SparseMapVector::new(degree, base_ring.clone());
        for (c, i) in poly_ring.terms(&modulus) {
            if i != degree {
                *modulus_vec.at_mut(i) = base_ring.negate(base_ring.clone_el(c));
            }
        }
        return RingValue::from(GaloisFieldBase { 
            base: FreeAlgebraImpl::new_with(base_ring, degree, modulus_vec, "θ", allocator, convolution_algorithm)
        });
    }
}

impl<V, A> GaloisFieldBase<AsField<Zn>, V, A, KaratsubaAlgorithm>
    where V: VectorView<El<AsField<Zn>>>,
        A: Allocator + Clone
{
    ///
    /// Creates the galois ring `GR(p, e, degree)`, mirroring the structure of this galois field.
    /// 
    /// A galois ring is similar to a galois field, but not a field anymore since it has prime power characteristic
    /// instead of prime characteristic. It can be constructed as `(Z/p^eZ)[X]/(f)` where `f` is a monic polynomial
    /// of degree `degree` such that `f mod p` is irreducible. When using this function, the generating polynomial of
    /// the resulting galois ring will be the coefficient-wise shortest lift of the generating polynomial of this
    /// galois field.
    /// 
    /// For more configuration options, use [`GaloisFieldBase::galois_ring_with()`].
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn galois_ring(&self, e: usize) -> AsLocalPIR<FreeAlgebraImpl<Zn, SparseMapVector<Zn>, A, KaratsubaAlgorithm>> {
        self.galois_ring_with(Zn::new(StaticRing::<i64>::RING.pow(*self.base_ring().modulus(), e) as u64), self.base.get_ring().allocator().clone(), STANDARD_CONVOLUTION)
    }
}

impl<R, V, A, C> GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    ///
    /// 
    /// Creates the galois ring `GR(p, e, degree)`, mirroring the structure of this galois field.
    /// 
    /// A galois ring is similar to a galois field, but not a field anymore since it has prime power characteristic
    /// instead of prime characteristic. It can be constructed as `(Z/p^eZ)[X]/(f)` where `f` is a monic polynomial
    /// of degree `degree` such that `f mod p` is irreducible. When using this function, the generating polynomial of
    /// the resulting galois ring will be the coefficient-wise shortest lift of the generating polynomial of this
    /// galois field.
    /// 
    /// See also [`GaloisFieldBase::galois_ring()`] for a simpler version of this function.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn galois_ring_with<S, A2, C2>(&self, new_base_ring: S, allocator: A2, convolution_algorithm: C2) -> AsLocalPIR<FreeAlgebraImpl<S, SparseMapVector<S>, A2, C2>>
        where S: RingStore + Clone,
            S::Type: ZnRing + CanHomFrom<<R::Type as ZnRing>::IntegerRingBase>,
            C2: ConvolutionAlgorithm<S::Type>,
            A2: Allocator + Clone
    {
        let (p, e) = is_prime_power(&BigIntRing::RING, &new_base_ring.size(&BigIntRing::RING).unwrap()).unwrap();
        assert!(BigIntRing::RING.eq_el(&p, &self.base_ring().size(&BigIntRing::RING).unwrap()));
        let mut modulus_vec = SparseMapVector::new(self.rank(), new_base_ring.clone());
        let x_pow_deg = RingRef::new(self).pow(self.canonical_gen(), self.rank());
        let x_pow_deg = self.wrt_canonical_basis(&x_pow_deg);
        let hom = new_base_ring.can_hom(self.base_ring().integer_ring()).unwrap();
        for i in 0..self.rank() {
            if !self.base_ring().is_zero(&x_pow_deg.at(i)) {
                *modulus_vec.at_mut(i) = hom.map(self.base_ring().smallest_lift(x_pow_deg.at(i)));
            }
        }
        let result = FreeAlgebraImpl::new_with(
            new_base_ring,
            self.rank(),
            modulus_vec,
            "θ",
            allocator,
            convolution_algorithm
        );
        let hom = result.base_ring().can_hom(self.base_ring().integer_ring()).unwrap();
        let max_ideal_gen = result.inclusion().map(hom.map_ref(self.base_ring().modulus()));
        return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(result, max_ideal_gen, Some(e)));
    }

    #[stability::unstable(feature = "enable")]
    pub fn unwrap_self(self) -> AsField<FreeAlgebraImpl<R, V, A, C>> {
        AsField::from(AsFieldBase::promise_is_perfect_field(self.base))
    }
    
    #[stability::unstable(feature = "enable")]
    pub fn allocator(&self) -> &A {
        self.base.get_ring().allocator()
    }
}

impl<R, V, A, C> GaloisField<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    ///
    /// Most generic function to create a finite/galois field.
    /// 
    /// It allows specifying all associated data. Note also that the passed implementation
    /// must indeed be a field, and this is not checked at runtime. Passing a non-field
    /// is impossible, unless [`AsFieldBase::promise_is_field()`] is used.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn create(base: AsField<FreeAlgebraImpl<R, V, A, C>>) -> Self {
        RingValue::from(GaloisFieldBase {
            base: base.into().unwrap_self()
        })
    }
}

impl<R, V, A, C> Clone for GaloisFieldBase<R, V, A, C>
    where R: Clone + RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: Clone + VectorView<El<R>>,
        C: Clone + ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone()
        }
    }
}

impl<R, V, A, C> Copy for GaloisFieldBase<R, V, A, C>
    where R: Copy + RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: Copy + VectorView<El<R>>,
        C: Copy + ConvolutionAlgorithm<R::Type>,
        A: Copy + Allocator,
        El<R>: Copy
{}

impl<R, V, A, C> PartialEq for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring()
    }
}

impl<R, V, A, C> DelegateRing for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    type Base = FreeAlgebraImplBase<R, V, A, C>;
    type Element = El<FreeAlgebraImpl<R, V, A, C>>;

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }

    fn get_delegate(&self) -> &Self::Base {
        self.base.get_ring()
    }
}

impl<R, V, A, C> Domain for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{}

impl<R, V, A, C> Field for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{}

impl<R, V, A, C> PerfectField for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{}

impl<R, V, A, C> EuclideanRing for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn euclidean_div_rem(&self, lhs: Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element) {
        assert!(!self.is_zero(rhs));
        (self.base.checked_div(&lhs, rhs).unwrap(), self.zero())
    }

    fn euclidean_deg(&self, val: &Self::Element) -> Option<usize> {
        if self.is_zero(val) {
            Some(0)
        } else {
            Some(1)
        }
    }

    fn euclidean_rem(&self, _: Self::Element, rhs: &Self::Element) -> Self::Element {
        assert!(!self.is_zero(rhs));
        self.zero()
    }
}

impl<R, V, A, C> PrincipalIdealRing for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.checked_left_div(lhs, rhs)
    }
    
    fn extended_ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        if self.is_zero(lhs) {
            (self.zero(), self.one(), self.clone_el(rhs))
        } else {
            (self.one(), self.zero(), self.clone_el(lhs))
        }
    }
}

impl<R, V, A, C> KaratsubaHint for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn karatsuba_threshold(&self) -> usize {
        self.get_delegate().karatsuba_threshold()
    }
}

impl<R, V, A, C> ComputeInnerProduct for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn inner_product<I: Iterator<Item = (Self::Element, Self::Element)>>(&self, els: I) -> Self::Element {
        self.rev_delegate(self.get_delegate().inner_product(els.map(|(a, b)| (self.delegate(a), self.delegate(b)))))
    }

    fn inner_product_ref<'a, I: Iterator<Item = (&'a Self::Element, &'a Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.rev_delegate(self.get_delegate().inner_product_ref(els.map(|(a, b)| (self.delegate_ref(a), self.delegate_ref(b)))))
    }

    fn inner_product_ref_fst<'a, I: Iterator<Item = (&'a Self::Element, Self::Element)>>(&self, els: I) -> Self::Element
        where Self::Element: 'a,
            Self: 'a
    {
        self.rev_delegate(self.get_delegate().inner_product_ref_fst(els.map(|(a, b)| (self.delegate_ref(a), self.delegate(b)))))
    }
}

impl<R, V, A, C> StrassenHint for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn strassen_threshold(&self) -> usize {
        self.get_delegate().strassen_threshold()
    }
}

impl<R, V, A, C, S> CanHomFrom<S> for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        S: ?Sized + RingBase,
        FreeAlgebraImplBase<R, V, A, C>: CanHomFrom<S>
{
    type Homomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanHomFrom<S>>::Homomorphism;

    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &S, el: <S as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.base.get_ring().map_in(from, el, hom)
    }
}


impl<R, V, A, C, R2, V2, A2, C2> CanHomFrom<GaloisFieldBase<R2, V2, A2, C2>> for FreeAlgebraImplBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<R2::Type>,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        R2: RingStore,
        R2::Type: ZnRing + FiniteRing + Field,
        V2: VectorView<El<R2>>,
        C2: ConvolutionAlgorithm<R2::Type>,
        A2: Allocator + Clone
{
    type Homomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanHomFrom<FreeAlgebraImplBase<R2, V2, A2, C2>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &GaloisFieldBase<R2, V2, A2, C2>) -> Option<Self::Homomorphism> {
        self.has_canonical_hom(from.base.get_ring())
    }

    fn map_in(&self, from: &GaloisFieldBase<R2, V2, A2, C2>, el: <GaloisFieldBase<R2, V2, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(from.base.get_ring(), el, hom)
    }
}

impl<R, V, A, C, S> CanIsoFromTo<S> for GaloisFieldBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + FiniteRing + Field,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        S: ?Sized + RingBase,
        FreeAlgebraImplBase<R, V, A, C>: CanIsoFromTo<S>
{
    type Isomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanIsoFromTo<S>>::Isomorphism;

    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism> {
        self.base.get_ring().has_canonical_iso(from)
    }

    fn map_out(&self, from: &S, el: Self::Element, iso: &Self::Isomorphism) -> <S as RingBase>::Element {
        self.base.get_ring().map_out(from, el, iso)
    }
}


impl<R, V, A, C, R2, V2, A2, C2> CanIsoFromTo<GaloisFieldBase<R2, V2, A2, C2>> for FreeAlgebraImplBase<R, V, A, C>
    where R: RingStore,
        R::Type: ZnRing + CanIsoFromTo<R2::Type>,
        V: VectorView<El<R>>,
        C: ConvolutionAlgorithm<R::Type>,
        A: Allocator + Clone,
        R2: RingStore,
        R2::Type: ZnRing + FiniteRing + Field,
        V2: VectorView<El<R2>>,
        C2: ConvolutionAlgorithm<R2::Type>,
        A2: Allocator + Clone
{
    type Isomorphism = <FreeAlgebraImplBase<R, V, A, C> as CanIsoFromTo<FreeAlgebraImplBase<R2, V2, A2, C2>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &GaloisFieldBase<R2, V2, A2, C2>) -> Option<Self::Isomorphism> {
        self.has_canonical_iso(from.base.get_ring())
    }
    
    fn map_out(&self, from: &GaloisFieldBase<R2, V2, A2, C2>, el: Self::Element, iso: &Self::Isomorphism) -> <GaloisFieldBase<R2, V2, A2, C2> as RingBase>::Element {
        self.map_out(from.base.get_ring(), el, iso)
    }
}

#[cfg(test)]
use test::Bencher;

#[test]
fn test_galois_field() {
    let field = GaloisField::new(3, 2);
    assert_eq!(9, field.elements().count());
    crate::ring::generic_tests::test_ring_axioms(&field, field.elements());
    crate::ring::generic_tests::test_self_iso(&field, field.elements());
    crate::field::generic_tests::test_field_axioms(&field, field.elements());
}

#[test]
fn test_galois_field_even() {
    for degree in 1..=9 {
        let field = GaloisField::new_with(Zn::new(2).as_field().ok().unwrap(), degree, Global, STANDARD_CONVOLUTION);
        assert_eq!(degree, field.rank());
        assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());
    }
}

#[test]
fn test_galois_field_odd() {
    for degree in 1..=9 {
        let field = GaloisField::new_with(Zn::new(3).as_field().ok().unwrap(), degree, Global, STANDARD_CONVOLUTION);
        assert_eq!(degree, field.rank());
        assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());
    }

    for degree in 1..=9 {
        let field = GaloisField::new_with(Zn::new(5).as_field().ok().unwrap(), degree, Global, STANDARD_CONVOLUTION);
        assert_eq!(degree, field.rank());
        assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());
    }
}

#[test]
fn test_galois_field_no_trinomial() {
    let field = GaloisField::new_with(Zn::new(2).as_field().ok().unwrap(), 24, Global, STANDARD_CONVOLUTION);
    assert_eq!(24, field.rank());
    let poly_ring = DensePolyRing::new(field.base_ring(), "X");
    poly_ring.println(&field.generating_poly(&poly_ring, &poly_ring.base_ring().identity()));
    assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());

    let field = GaloisField::new_with(Zn::new(3).as_field().ok().unwrap(), 30, Global, STANDARD_CONVOLUTION);
    assert_eq!(30, field.rank());
    let poly_ring = DensePolyRing::new(field.base_ring(), "X");
    poly_ring.println(&field.generating_poly(&poly_ring, &poly_ring.base_ring().identity()));
    assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());

    let field = GaloisField::new_with(Zn::new(17).as_field().ok().unwrap(), 32, Global, STANDARD_CONVOLUTION);
    assert_eq!(32, field.rank());
    let poly_ring = DensePolyRing::new(field.base_ring(), "X");
    poly_ring.println(&field.generating_poly(&poly_ring, &poly_ring.base_ring().identity()));
    assert!(field.into().unwrap_self().into().unwrap_self().as_field().is_ok());
}

#[bench]
fn bench_create_galois_ring_2_14_96(bencher: &mut Bencher) {
    bencher.iter(|| {
        let field = GaloisField::new(2, 96);
        let ring = field.get_ring().galois_ring(14);
        assert_eq!(96, ring.rank());
    });
}