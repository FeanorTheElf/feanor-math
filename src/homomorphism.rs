use std::fmt::Debug;
use std::marker::PhantomData;

use crate::ring::*;
use crate::primitive_int::{StaticRingBase, StaticRing};

///
/// The user-facing trait for ring homomorphisms, i.e. maps `R -> S`
/// between rings that respect the ring structure. Since all considered
/// rings are unital, ring homomorphisms also must be unital.
/// 
/// Objects are expected to know their domain and codomain rings and
/// can thus make sense without an implicit ambient ring (unlike e.g.
/// ring elements).
/// 
/// Ring homomorphisms are usually obtained by a corresponding method
/// on [`RingStore`], and their functionality is provided by underlying
/// functions of [`RingBase`]. Main examples include
///  - Every ring `R` has a homomorphism `Z -> R`. The corresponding
///    [`Homomorphism`]-object is obtained with [`RingStore::int_hom()`],
///    and the functionality provided by [`RingBase::from_int()`].
///  - [`RingExtension`]s have give a (injective) homomorphism `R -> S`
///    which can be obtained by [`RingExtensionStore::inclusion()`].
///    The functionality is provided by functions on [`RingExtension`],
///    like [`RingExtension::from()`].
///  - Other "natural" homomorphisms can be obtained via [`RingStore::can_hom()`].
///    This requires the underlying [`RingBase`]s to implement [`CanHomFrom`],
///    and the functions of that trait define the homomorphism.
///  
pub trait Homomorphism<Domain: ?Sized, Codomain: ?Sized> 
    where Domain: RingBase, Codomain: RingBase
{
    ///
    /// The type of the [`RingStore`] used by this object to store the domain ring.
    /// 
    type DomainStore: RingStore<Type = Domain>;
    ///
    /// The type of the [`RingStore`] used by this object to store the codomain ring.
    /// 
    type CodomainStore: RingStore<Type = Codomain>;

    ///
    /// Returns a reference to the domain ring.
    /// 
    fn domain<'a>(&'a self) -> &'a Self::DomainStore;

    ///
    /// Returns a reference to the codomain ring.
    /// 
    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore;

    ///
    /// Applies this homomorphism to the given element from the domain ring,
    /// resulting in an element in the codomain ring.
    /// 
    fn map(&self, x: Domain::Element) -> Codomain::Element;

    ///
    /// Applies this homomorphism to the given element from the domain ring,
    /// resulting in an element in the codomain ring.
    /// 
    fn map_ref(&self, x: &Domain::Element) -> Codomain::Element {
        self.map(self.domain().clone_el(x))
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_assign_map(&self, lhs: &mut Codomain::Element, rhs: Domain::Element) {
        self.codomain().mul_assign(lhs, self.map(rhs))
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_assign_ref_map(&self, lhs: &mut Codomain::Element, rhs: &Domain::Element) {
        self.codomain().mul_assign(lhs, self.map_ref(rhs))
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_map(&self, mut lhs: Codomain::Element, rhs: Domain::Element) -> Codomain::Element {
        self.mul_assign_map(&mut lhs, rhs);
        lhs
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_ref_fst_map(&self, lhs: &Codomain::Element, rhs: Domain::Element) -> Codomain::Element {
        self.mul_map(self.codomain().clone_el(lhs), rhs)
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_ref_snd_map(&self, mut lhs: Codomain::Element, rhs: &Domain::Element) -> Codomain::Element {
        self.mul_assign_ref_map(&mut lhs, rhs);
        lhs
    }

    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this homomorphism to a given element from the domain ring.
    /// 
    /// This is equivalent to, but may be faster than, first mapping the domain
    /// ring element via this homomorphism, and then performing ring multiplication.
    /// 
    fn mul_ref_map(&self, lhs: &Codomain::Element, rhs: &Domain::Element) -> Codomain::Element {
        self.mul_ref_snd_map(self.codomain().clone_el(lhs), rhs)
    }

    ///
    /// Constructs the homomorphism `x -> self.map(prev.map(x))`.
    /// 
    fn compose<F, PrevDomain: ?Sized + RingBase>(self, prev: F) -> ComposedHom<PrevDomain, Domain, Codomain, F, Self>
        where Self: Sized, F: Homomorphism<PrevDomain, Domain>
    {
        assert!(prev.codomain().get_ring() == self.domain().get_ring());
        ComposedHom { f: prev, g: self, domain: PhantomData, intermediate: PhantomData, codomain: PhantomData }
    }
    
    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this and another homomorphism to a given element from another ring.
    /// 
    /// The equivalent of [`Homomorphism::mul_assign_map()`] for a chain of two 
    /// homomorphisms. While providing specialized implementations for longer and longer
    /// chains soon becomes ridiculous, there is one important use case why we would want
    /// at least length-2 chains:
    /// 
    /// In particular, many [`RingExtension`]s have elements that consist of multiple
    /// elements of the base ring, with base-ring-multiplication being scalar-vector
    /// multiplication. Hence, if the base ring allows a fast-multiplication through
    /// a single homomorphism, it makes sense to extend that along an [`Inclusion`].
    /// Hence, we also have [`RingExtension::mul_assign_base_through_hom()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn mul_assign_ref_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, Domain>>(&self, lhs: &mut Codomain::Element, rhs: &First::Element, hom: H) {
        self.mul_assign_map(lhs, hom.map_ref(rhs));
    }
    
    ///
    /// Multiplies the given element in the codomain ring with an element obtained
    /// by applying this and another homomorphism to a given element from another ring.
    /// 
    /// For details, see [`Homomorphism::mul_assign_ref_map_through_hom()`].
    /// 
    #[stability::unstable(feature = "enable")]
    fn mul_assign_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, Domain>>(&self, lhs: &mut Codomain::Element, rhs: First::Element, hom: H) {
        self.mul_assign_map(lhs, hom.map(rhs));
    }
}

///
/// Trait for rings R that have a canonical homomorphism `S -> R`.
/// A ring homomorphism is expected to be unital. 
/// 
/// This trait is considered implementor-facing, so it is designed to easily 
/// implement natural maps between rings. When using homomorphisms, consider
/// using instead [`CanHom`], as it does not require weird syntax like
/// `R.get_ring().map_in(S.get_ring(), x, &hom)`.
/// 
/// **Warning** Because of type-system limitations (see below), this trait
/// is not implemented in all cases where it makes sense, in particular when
/// type parameters are involved. Thus, you should always consider this trait
/// to be for convenience only. A truly generic algorithm should, if possible,
/// not constrain input ring types using `CanHomFrom`, but instead take an 
/// additional object of generic type bounded by [`Homomorphism`] that provides
/// the required homomorphism.
/// 
/// # Exact requirements
/// 
/// Which homomorphisms are considered canonical is up to implementors,
/// as long as any diagram of canonical homomorphisms commutes. In
/// other words, if there are rings `R, S` and "intermediate rings"
/// `R1, ..., Rn` resp. `R1', ..., Rm'` such that there are canonical
/// homomorphisms
/// ```text
///   S -> R1 -> R2 -> ... -> Rn -> R
/// ```
/// and
/// ```text
///   S -> R1' -> R2' -> ... -> Rm' -> R
/// ```
/// then both homomorphism chains should yield same results on same
/// inputs.
/// 
/// If the canonical homomorphism might be an isomorphism, consider also
/// implementing [`CanIsoFromTo`].
/// 
/// # Example
/// 
/// Most integer rings support canonical homomorphisms between them.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// let R = StaticRing::<i64>::RING;
/// let S = BigIntRing::RING;
/// let eight = S.int_hom().map(8);
/// // on RingBase level
/// let hom = R.get_ring().has_canonical_hom(S.get_ring()).unwrap();
/// assert_eq!(8, R.get_ring().map_in(S.get_ring(), S.clone_el(&eight), &hom));
/// // on RingStore level
/// assert_eq!(8, R.coerce(&S, S.clone_el(&eight)));
/// // or
/// let hom = R.can_hom(&S).unwrap();
/// assert_eq!(8, hom.map_ref(&eight));
/// ```
/// 
/// # Limitations
/// 
/// The rust constraints regarding conflicting impl make it, in some cases,
/// impossible to implement all the canonical homomorphisms that we would like.
/// This is true in particular, if the rings are highly generic, and build
/// on base rings. In this case, it should always be preferred to implement
/// `CanIsoFromTo` for rings that are "the same", and on the other hand not
/// to implement classical homomorphisms, like `ZZ -> R` which exists for any
/// ring R. In applicable cases, consider also implementing [`RingExtension`].
/// 
/// Because of this reason, implementing [`RingExtension`] also does not require
/// an implementation of `CanHomFrom<Self::BaseRing>`. Hence, if you as a user
/// miss a certain implementation of `CanHomFrom`, check whether there maybe
/// is a corresponding implementation of [`RingExtension`], or a member function.
/// 
/// # More examples
/// 
/// ## Integer rings
/// 
/// All given integer rings have canonical isomorphisms between each other.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// let Z_i8 = StaticRing::<i8>::RING;
/// let Z_i32 = StaticRing::<i32>::RING;
/// let Z_i128 = StaticRing::<i128>::RING;
/// let Z_big = BigIntRing::RING;
/// 
/// assert!(Z_i8.can_iso(&Z_i8).is_some());
/// assert!(Z_i8.can_iso(&Z_i32).is_some());
/// assert!(Z_i8.can_iso(&Z_i128).is_some());
/// assert!(Z_i8.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_i32.can_iso(&Z_i8).is_some());
/// assert!(Z_i32.can_iso(&Z_i32).is_some());
/// assert!(Z_i32.can_iso(&Z_i128).is_some());
/// assert!(Z_i32.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_i128.can_iso(&Z_i8).is_some());
/// assert!(Z_i128.can_iso(&Z_i32).is_some());
/// assert!(Z_i128.can_iso(&Z_i128).is_some());
/// assert!(Z_i128.can_iso(&Z_big).is_some());
/// 
/// assert!(Z_big.can_iso(&Z_i8).is_some());
/// assert!(Z_big.can_iso(&Z_i32).is_some());
/// assert!(Z_big.can_iso(&Z_i128).is_some());
/// assert!(Z_big.can_iso(&Z_big).is_some());
/// ```
/// 
/// ## Integer quotient rings `Z/nZ`
/// 
/// Due to conflicting implementations, only the most useful conversions
/// are implemented for `Z/nZ`.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_big;
/// # use feanor_math::rings::zn::zn_rns;
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = BigIntRing::RING;
/// 
/// let zn_big_i128 = zn_big::Zn::new(ZZ, 17 * 257);
/// let zn_big_big = zn_big::Zn::new(ZZ_big, ZZ_big.int_hom().map(17 * 257));
/// let Zn_std = zn_64::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::create_from_primes(vec![17, 257], ZZ_big);
/// 
/// assert!(zn_big_i128.can_iso(&zn_big_i128).is_some());
/// assert!(zn_big_i128.can_iso(&zn_big_big).is_some());
/// 
/// assert!(zn_big_big.can_iso(&zn_big_i128).is_some());
/// assert!(zn_big_big.can_iso(&zn_big_big).is_some());
/// 
/// assert!(Zn_std.can_iso(&zn_big_i128).is_some());
/// assert!(Zn_std.can_iso(&Zn_std).is_some());
/// 
/// assert!(Zn_rns.can_iso(&zn_big_i128).is_some());
/// assert!(Zn_rns.can_iso(&zn_big_big).is_some());
/// assert!(Zn_rns.can_iso(&Zn_rns).is_some());
/// ```
/// Most notably, reduction homomorphisms are currently not available.
/// You can use [`crate::rings::zn::ZnReductionMap`] instead.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::assert_el_eq;
/// let Z9 = zn_64::Zn::new(9);
/// let Z3 = zn_64::Zn::new(3);
/// assert!(Z3.can_hom(&Z9).is_none());
/// let mod_3 = ZnReductionMap::new(&Z9, &Z3).unwrap();
/// assert_el_eq!(Z3, Z3.one(), mod_3.map(Z9.int_hom().map(4)));
/// ```
/// Additionally, there are the projections `Z -> Z/nZ`.
/// They are all implemented, even though [`crate::rings::zn::ZnRing`] currently
/// only requires the projection from the "associated" integer ring.
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::zn::*;
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = BigIntRing::RING;
/// 
/// let zn_big_i128 = zn_big::Zn::new(ZZ, 17 * 257);
/// let zn_big_big = zn_big::Zn::new(ZZ_big, ZZ_big.int_hom().map(17 * 257));
/// let Zn_std = zn_64::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::create_from_primes(vec![17, 257], ZZ_big);
/// 
/// assert!(zn_big_i128.can_hom(&ZZ).is_some());
/// assert!(zn_big_i128.can_hom(&ZZ_big).is_some());
/// 
/// assert!(zn_big_big.can_hom(&ZZ).is_some());
/// assert!(zn_big_big.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_std.can_hom(&ZZ).is_some());
/// assert!(Zn_std.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_rns.can_hom(&ZZ).is_some());
/// assert!(Zn_rns.can_hom(&ZZ_big).is_some());
/// ```
/// 
/// ## Polynomial Rings
/// 
/// For the two provided univariate polynomial ring implementations, we have the isomorphisms
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::integer::*;
/// # use feanor_math::rings::poly::*;
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let P_dense = dense_poly::DensePolyRing::new(ZZ, "X");
/// let P_sparse = sparse_poly::SparsePolyRing::new(ZZ, "X");
/// 
/// assert!(P_dense.can_iso(&P_dense).is_some());
/// assert!(P_dense.can_iso(&P_sparse).is_some());
/// assert!(P_sparse.can_iso(&P_dense).is_some());
/// assert!(P_sparse.can_iso(&P_sparse).is_some());
/// ```
/// Unfortunately, the inclusions `R -> R[X]` are not implemented as canonical homomorphisms,
/// however provided by the functions of [`RingExtension`].
/// 
pub trait CanHomFrom<S>: RingBase
    where S: RingBase + ?Sized
{
    ///
    /// Data required to compute the action of the canonical homomorphism on ring elements.
    /// 
    type Homomorphism;

    ///
    /// If there is a canonical homomorphism `from -> self`, returns `Some(data)`, where 
    /// `data` is additional data that can be used to compute the action of the homomorphism
    /// on ring elements. Otherwise, `None` is returned.
    /// 
    fn has_canonical_hom(&self, from: &S) -> Option<Self::Homomorphism>;
    fn map_in(&self, from: &S, el: S::Element, hom: &Self::Homomorphism) -> Self::Element;

    fn map_in_ref(&self, from: &S, el: &S::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in(from, from.clone_el(el), hom)
    }

    fn mul_assign_map_in(&self, from: &S, lhs: &mut Self::Element, rhs: S::Element, hom: &Self::Homomorphism) {
        self.mul_assign(lhs, self.map_in(from, rhs, hom));
    }

    fn mul_assign_map_in_ref(&self, from: &S, lhs: &mut Self::Element, rhs: &S::Element, hom: &Self::Homomorphism) {
        self.mul_assign(lhs, self.map_in_ref(from, rhs, hom));
    }
}

///
/// Trait for rings R that have a canonical isomorphism `S -> R`.
/// A ring homomorphism is expected to be unital.
/// 
/// **Warning** Because of type-system limitations (see [`CanHomFrom`]), this trait
/// is not implemented in all cases where it makes sense, in particular when
/// type parameters are involved. Thus, you should always consider this trait
/// to be for convenience only. A truly generic algorithm should, if possible,
/// not constrain input ring types using `CanHomFrom`, but instead take an 
/// additional object of generic type bounded by [`Homomorphism`] that provides
/// the required homomorphism.
/// 
/// # Exact requirements
/// 
/// Same as for [`CanHomFrom`], it is up to implementors to decide which
/// isomorphisms are canonical, as long as each diagram that contains
/// only canonical homomorphisms, canonical isomorphisms and their inverses
/// commutes.
/// In other words, if there are rings `R, S` and "intermediate rings"
/// `R1, ..., Rn` resp. `R1', ..., Rm'` such that there are canonical
/// homomorphisms `->` or isomorphisms `<~>` connecting them - e.g. like
/// ```text
///   S -> R1 -> R2 <~> R3 <~> R4 -> ... -> Rn -> R
/// ```
/// and
/// ```text
///   S <~> R1' -> R2' -> ... -> Rm' -> R
/// ```
/// then both chains should yield same results on same inputs.
/// 
/// Hence, it would be natural if the trait were symmetrical, i.e.
/// for any implementation `R: CanIsoFromTo<S>` there is also an
/// implementation `S: CanIsoFromTo<R>`. However, because of the trait
/// impl constraints of Rust, this is unpracticable and so we only
/// require the implementation `R: CanHomFrom<S>`.
/// 
pub trait CanIsoFromTo<S>: CanHomFrom<S>
    where S: RingBase + ?Sized
{
    ///
    /// Data required to compute a preimage under the canonical homomorphism.
    /// 
    type Isomorphism;

    ///
    /// If there is a canonical homomorphism `from -> self`, and this homomorphism
    /// is an isomorphism, returns `Some(data)`, where `data` is additional data that
    /// can be used to compute preimages under the homomorphism. Otherwise, `None` is
    /// returned.
    /// 
    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism>;

    ///
    /// Computes the preimage of `el` under the canonical homomorphism `from -> self`.
    /// 
    fn map_out(&self, from: &S, el: Self::Element, iso: &Self::Isomorphism) -> S::Element;
}

///
/// Basically an alias for `CanIsoFromTo<Self>`, but implemented as new
/// trait since trait aliases are not available.
/// 
pub trait SelfIso: CanIsoFromTo<Self> {}

impl<R: ?Sized + CanIsoFromTo<R>> SelfIso for R {}

///
/// A high-level wrapper of [`CanHomFrom::Homomorphism`] that references the
/// domain and codomain rings, and is much easier to use.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// let from = StaticRing::<i32>::RING;
/// let to = StaticRing::<i64>::RING;
/// let hom = to.can_hom(&from).unwrap();
/// assert_eq!(7, hom.map(7));
/// // instead of
/// let hom = to.get_ring().has_canonical_hom(from.get_ring()).unwrap();
/// assert_eq!(7, to.get_ring().map_in(from.get_ring(), 7, &hom));
/// ```
/// 
/// # See also
/// The "bi-directional" variant [`CanHom`], the basic interfaces [`CanHomFrom`] and
/// [`CanIsoFromTo`] and the very simplified function [`RingStore::coerce`].
/// 
pub struct CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    from: R,
    to: S,
    data: <S::Type as CanHomFrom<R::Type>>::Homomorphism
}

impl<R, S> Debug for CanHom<R, S>
    where R: RingStore + Debug, S: RingStore + Debug, S::Type: CanHomFrom<R::Type>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CanHom({:?}, {:?})", self.from, self.to)
    }
}

impl<R, S> CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    pub fn new(from: R, to: S) -> Result<Self, (R, S)> {
        match to.get_ring().has_canonical_hom(from.get_ring()) {
            Some(data) => Ok(Self::from_raw_parts(from, to, data)),
            _ => Err((from, to))
        }
    }

    pub fn raw_hom(&self) -> &<S::Type as CanHomFrom<R::Type>>::Homomorphism {
        &self.data
    }

    pub fn into_raw_hom(self) -> <S::Type as CanHomFrom<R::Type>>::Homomorphism {
        self.data
    }

    #[stability::unstable(feature = "enable")]
    pub fn from_raw_parts(from: R, to: S, data: <S::Type as CanHomFrom<R::Type>>::Homomorphism) -> Self {
        Self { from, to, data }
    }
}

impl<R, S> Clone for CanHom<R, S>
    where R: RingStore + Clone, S: RingStore + Clone, S::Type: CanHomFrom<R::Type>
{
    fn clone(&self) -> Self {
        Self::new(self.from.clone(), self.to.clone()).ok().unwrap()
    }
}

impl<R, S> Homomorphism<R::Type, S::Type> for CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    type CodomainStore = S;
    type DomainStore = R;

    fn map(&self, el: El<R>) -> El<S> {
        self.to.get_ring().map_in(self.from.get_ring(), el, &self.data)
    }

    fn map_ref(&self, el: &El<R>) -> El<S> {
        self.to.get_ring().map_in_ref(self.from.get_ring(), el, &self.data)
    }
    
    fn domain(&self) -> &R {
        &self.from
    }

    fn codomain(&self) -> &S {
        &self.to
    }

    fn mul_assign_map(&self, lhs: &mut <S::Type as RingBase>::Element, rhs: <R::Type as RingBase>::Element) {
        self.to.get_ring().mul_assign_map_in(self.from.get_ring(), lhs, rhs, &self.data);
    }

    fn mul_assign_ref_map(&self, lhs: &mut <S::Type as RingBase>::Element, rhs: &<R::Type as RingBase>::Element) {
        self.to.get_ring().mul_assign_map_in_ref(self.from.get_ring(), lhs, rhs, &self.data);
    }
}

///
/// A wrapper of [`CanHomFrom::Homomorphism`] that does not own the data associated
/// with the homomorphism. Use cases are rare, prefer to use [`CanHom`] whenever possible.
/// 
/// More concretely, this should only be used when you only have a reference to `<R as CanHomFrom<S>>::Homomorphism`,
/// but cannot refactor code to wrap that object in a [`CanHom`] instead. The main situation
/// where this occurs is when implementing [`CanHomFrom`], since a the lifetime of [`CanHom`] is
/// bound by the lifetime of the domain and codomain rings, but `CanHomFrom::Type` does not allow
/// this.
/// 
#[stability::unstable(feature = "enable")]
pub struct CanHomRef<'a, R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    from: R,
    to: S,
    data: &'a <S::Type as CanHomFrom<R::Type>>::Homomorphism
}

impl<'a, R, S> CanHomRef<'a, R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    #[stability::unstable(feature = "enable")]
    pub fn raw_hom(&self) -> &<S::Type as CanHomFrom<R::Type>>::Homomorphism {
        &self.data
    }

    #[stability::unstable(feature = "enable")]
    pub fn from_raw_parts(from: R, to: S, data: &'a <S::Type as CanHomFrom<R::Type>>::Homomorphism) -> Self {
        Self { from, to, data }
    }
}

impl<'a, R, S> Clone for CanHomRef<'a, R, S>
    where R: RingStore + Clone, S: RingStore + Clone, S::Type: CanHomFrom<R::Type>
{
    fn clone(&self) -> Self {
        Self::from_raw_parts(self.from.clone(), self.to.clone(), self.data)
    }
}

impl<'a, R, S> Copy for CanHomRef<'a, R, S>
    where R: RingStore + Copy, 
        S: RingStore + Copy, 
        S::Type: CanHomFrom<R::Type>,
        El<R>: Copy,
        El<S>: Copy
{}

impl<'a, R, S> Homomorphism<R::Type, S::Type> for CanHomRef<'a, R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    type CodomainStore = S;
    type DomainStore = R;

    fn map(&self, el: El<R>) -> El<S> {
        self.to.get_ring().map_in(self.from.get_ring(), el, &self.data)
    }

    fn map_ref(&self, el: &El<R>) -> El<S> {
        self.to.get_ring().map_in_ref(self.from.get_ring(), el, &self.data)
    }
    
    fn domain(&self) -> &R {
        &self.from
    }

    fn codomain(&self) -> &S {
        &self.to
    }
}

///
/// A wrapper around [`CanIsoFromTo::Isomorphism`] that references the domain and
/// codomain rings, making it much easier to use. This also contains the homomorphism
/// in the other direction, i.e. allows mapping in both directions.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// 
/// let from = StaticRing::<i32>::RING;
/// let to = StaticRing::<i64>::RING;
/// let hom = to.can_iso(&from).unwrap();
/// assert_eq!(7, hom.map(7));
/// // instead of
/// let hom = to.get_ring().has_canonical_iso(from.get_ring()).unwrap();
/// assert_eq!(7, from.get_ring().map_out(to.get_ring(), 7, &hom));
/// ```
/// 
/// # See also
/// The "one-directional" variant [`CanHom`], the basic interfaces [`CanHomFrom`] and
/// [`CanIsoFromTo`] and the very simplified function [`RingStore::coerce()`].
/// 
pub struct CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanIsoFromTo<R::Type>
{
    from: R,
    to: S,
    data: <S::Type as CanIsoFromTo<R::Type>>::Isomorphism
}

impl<R, S> Debug for CanIso<R, S>
    where R: RingStore + Debug, S: RingStore + Debug, S::Type: CanIsoFromTo<R::Type>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CanIso({:?}, {:?})", self.from, self.to)
    }
}

impl<R, S> Clone for CanIso<R, S>
    where R: RingStore + Clone, S: RingStore + Clone, S::Type: CanIsoFromTo<R::Type>
{
    fn clone(&self) -> Self {
        Self::new(self.from.clone(), self.to.clone()).ok().unwrap()
    }
}

impl<R, S> CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanIsoFromTo<R::Type>
{
    pub fn new(from: R, to: S) -> Result<Self, (R, S)> {
        match to.get_ring().has_canonical_iso(from.get_ring()) {
            Some(data) => {
                assert!(to.get_ring().has_canonical_hom(from.get_ring()).is_some());
                Ok(Self { from, to, data })
            },
            _ => Err((from, to))
        }
    }

    pub fn into_inv(self) -> CanHom<R, S> {
        CanHom::new(self.from, self.to).unwrap_or_else(|_| unreachable!())
    }

    pub fn inv<'a>(&'a self) -> CanHom<&'a R, &'a S> {
        CanHom::new(&self.from, &self.to).unwrap_or_else(|_| unreachable!())
    }

    pub fn raw_iso(&self) -> &<S::Type as CanIsoFromTo<R::Type>>::Isomorphism {
        &self.data
    }

    pub fn into_raw_iso(self) -> <S::Type as CanIsoFromTo<R::Type>>::Isomorphism {
        self.data
    }
}

impl<R, S> Homomorphism<S::Type,R::Type> for CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanIsoFromTo<R::Type>
{
    type DomainStore = S;
    type CodomainStore = R;
    
    fn map(&self, x: El<S>) -> El<R> {
        self.to.get_ring().map_out(self.from.get_ring(), x, &self.data)
    }

    fn domain(&self) -> &S {
        &self.to
    }

    fn codomain(&self) -> &R {
        &self.from
    }
}

///
/// The ring homomorphism induced by a [`RingExtension`].
/// 
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// let base = StaticRing::<i32>::RING;
/// let extension = dense_poly::DensePolyRing::new(base, "X");
/// let hom = extension.inclusion();
/// let f = extension.add(hom.map(8), extension.indeterminate());
/// assert_el_eq!(extension, extension.from_terms([(8, 0), (1, 1)].into_iter()), &f);
/// ```
/// 
#[derive(Clone, Debug)]
pub struct Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
    ring: R
}

impl<R: RingStore> Copy for Inclusion<R>
    where R: Copy, El<R>: Copy, R::Type: RingExtension
{}

impl<R> Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
    ///
    /// Returns the [`Inclusion`] from the base ring of the given ring to itself.
    /// 
    pub fn new(ring: R) -> Self {
        Inclusion { ring }
    }
}
    
impl<R> Homomorphism<<<R::Type as RingExtension>::BaseRing as RingStore>::Type, R::Type> for Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
    type CodomainStore = R;
    type DomainStore = <R::Type as RingExtension>::BaseRing;

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        self.ring.base_ring()
    }

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.ring
    }

    fn map(&self, x: <<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) -> <R::Type as RingBase>::Element {
        self.ring.get_ring().from(x)
    }

    fn map_ref(&self, x: &<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) -> <R::Type as RingBase>::Element {
        self.ring.get_ring().from_ref(x)
    }

    fn mul_assign_ref_map(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: &<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) {
        self.ring.get_ring().mul_assign_base(lhs, rhs)
    }

    fn mul_assign_map(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: <<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) {
        self.mul_assign_ref_map(lhs, &rhs)
    }

    fn mul_assign_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, <<R::Type as RingExtension>::BaseRing as RingStore>::Type>>(&self, lhs: &mut El<R>, rhs: First::Element, hom: H) {
        self.ring.get_ring().mul_assign_base_through_hom(lhs, &rhs, hom)
    }

    fn mul_assign_ref_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, <<R::Type as RingExtension>::BaseRing as RingStore>::Type>>(&self, lhs: &mut El<R>, rhs: &First::Element, hom: H) {
        self.ring.get_ring().mul_assign_base_through_hom(lhs, rhs, hom)
    }
}

///
/// The ring homomorphism `Z -> R` that exists for any ring `R`.
/// 
/// # Example
/// ```rust
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// let ring = zn_static::F17;
/// let hom = ring.int_hom();
/// assert_el_eq!(ring, hom.map(1), hom.map(18));
/// ```
/// 
#[derive(Clone)]
pub struct IntHom<R>
    where R: RingStore
{
    ring: R
}

impl<R: RingStore> Copy for IntHom<R>
    where R: Copy, El<R>: Copy
{}

impl<R> Homomorphism<StaticRingBase<i32>, R::Type> for IntHom<R>
    where R: RingStore
{
    type CodomainStore = R;
    type DomainStore = StaticRing<i32>;

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &StaticRing::<i32>::RING
    }

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.ring
    }

    fn map(&self, x: i32) -> <R::Type as RingBase>::Element {
        self.ring.get_ring().from_int(x)
    }

    fn mul_assign_map(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: i32) {
        self.ring.get_ring().mul_assign_int(lhs, rhs)
    }

    fn mul_assign_ref_map(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: &<StaticRingBase<i32> as RingBase>::Element) {
        self.mul_assign_map(lhs, *rhs)
    }
}

impl<R> IntHom<R>
    where R: RingStore
{
    ///
    /// Creates the [`IntHom`] homomorphism
    /// ```text
    ///   Z -> R, n -> 1 + ... + 1 [n times]
    /// ```
    /// for the given ring `R`.
    /// 
    pub fn new(ring: R) -> Self {
        Self { ring }
    }
}

///
/// The identity homomorphism `R -> R, x -> x` on the given ring `R`.
/// 
#[derive(Clone)]
pub struct Identity<R: RingStore> {
    ring: R
}

impl<R: RingStore> Copy for Identity<R>
    where R: Copy, El<R>: Copy
{}

impl<R: RingStore> Identity<R> {

    ///
    /// Creates the [`Identity`] homomorphism `R -> R, x -> x` on the given ring `R`.
    /// 
    pub fn new(ring: R) -> Self {
        Identity { ring }
    }
}

impl<R: RingStore> Homomorphism<R::Type, R::Type> for Identity<R> {

    type CodomainStore = R;
    type DomainStore = R;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.ring
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.ring
    }

    fn map(&self, x: <R::Type as RingBase>::Element) -> <R::Type as RingBase>::Element {
        x
    }

    fn mul_assign_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, R::Type>>(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: First::Element, hom: H) {
        hom.mul_assign_map(lhs, rhs);
    }

    fn mul_assign_ref_map_through_hom<First: ?Sized + RingBase, H: Homomorphism<First, R::Type>>(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: &First::Element, hom: H) {
        hom.mul_assign_ref_map(lhs, rhs);
    }
}

impl<'a, S, R, H> Homomorphism<S, R> for &'a H 
    where S: ?Sized + RingBase, R: ?Sized + RingBase, H: Homomorphism<S, R>
{
    type CodomainStore = H::CodomainStore;
    type DomainStore = H::DomainStore;

    fn codomain<'b>(&'b self) -> &'b Self::CodomainStore {
        (*self).codomain()
    }

    fn domain<'b>(&'b self) -> &'b Self::DomainStore {
        (*self).domain()
    }

    fn map(&self, x: <S as RingBase>::Element) -> <R as RingBase>::Element {
        (*self).map(x)
    }

    fn map_ref(&self, x: &<S as RingBase>::Element) -> <R as RingBase>::Element {
        (*self).map_ref(x)
    }

    fn mul_assign_map(&self, lhs: &mut <R as RingBase>::Element, rhs: <S as RingBase>::Element) {
        (*self).mul_assign_map(lhs, rhs)
    }

    fn mul_assign_ref_map(&self, lhs: &mut <R as RingBase>::Element, rhs: &<S as RingBase>::Element) {
        (*self).mul_assign_ref_map(lhs, rhs)
    }
}

///
/// A homomorphism between rings `R -> S`, with its action on elements
/// defined by a user-provided closure.
/// 
/// It is up to the user to ensure that the given closure indeed
/// satisfies the ring homomorphism axioms:
///  - For `x, y in R`, it should satisfy `f(x) + f(y) = f(x + y)`
///  - For `x, y in R`, it should satisfy `f(x) * f(y) = f(x * y)`
///  - It should map `f(0) = 0` and `f(1) = 1`
/// 
/// Hence, a [`LambdaHom`] should only be used if none of the builtin
/// homomorphisms can achieve the same result, since a function that does
/// not follow the above axioms will make algorithms misbehave, and can
/// lead to hard-to-debug errors.
/// 
#[derive(Clone)]
pub struct LambdaHom<R: RingStore, S: RingStore, F>
    where F: Fn(&R, &S, &El<R>) -> El<S>
{
    from: R,
    to: S,
    f: F
}

impl<R: RingStore, S: RingStore, F> Copy for LambdaHom<R, S, F>
    where F: Copy + Fn(&R, &S, &El<R>) -> El<S>,
        R: Copy, El<R>: Copy,
        S: Copy, El<S>: Copy
{}

impl<R: RingStore, S: RingStore, F> LambdaHom<R, S, F>
    where F: Fn(&R, &S, &El<R>) -> El<S>
{
    ///
    /// Creates a new [`LambdaHom`] from `from` to `to`, mapping elements as
    /// specified by the given function.
    /// 
    /// It is up to the user to ensure that the given closure indeed
    /// satisfies the ring homomorphism axioms:
    ///  - For `x, y in R`, it should satisfy `f(x) + f(y) = f(x + y)`
    ///  - For `x, y in R`, it should satisfy `f(x) * f(y) = f(x * y)`
    ///  - It should map `f(0) = 0` and `f(1) = 1`
    /// 
    /// Hence, a [`LambdaHom`] should only be used if none of the builtin
    /// homomorphisms can achieve the same result, since a function that does
    /// not follow the above axioms will make algorithms misbehave, and can
    /// lead to hard-to-debug errors.
    /// 
    pub fn new(from: R, to: S, f: F) -> Self {
        Self { from, to, f }
    }

    ///
    /// Returns the stored domain and codomain rings, consuming this object.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn into_domain_codomain(self) -> (R, S) {
        (self.from, self.to)
    }
}

impl<R: RingStore, S: RingStore, F> Homomorphism<R::Type, S::Type> for LambdaHom<R, S, F>
    where F: Fn(&R, &S, &El<R>) -> El<S>
{
    type CodomainStore = S;
    type DomainStore = R;

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        &self.to
    }

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        &self.from
    }

    fn map(&self, x: <R::Type as RingBase>::Element) -> <S::Type as RingBase>::Element {
        (self.f)(self.domain(), self.codomain(), &x)
    }

    fn map_ref(&self, x: &<R::Type as RingBase>::Element) -> <S::Type as RingBase>::Element {
        (self.f)(self.domain(), self.codomain(), x)
    }
}

///
/// The function composition of two homomorphisms `f: R -> S` and `g: S -> T`.
/// 
/// More concretely, this is the homomorphism `R -> T` that maps `x` to `g(f(x))`.
/// The best way to create a [`ComposedHom`] is through [`Homomorphism::compose()`].
/// 
pub struct ComposedHom<R, S, T, F, G>
    where F: Homomorphism<R, S>,
        G: Homomorphism<S, T>,
        R: ?Sized + RingBase,
        S: ?Sized + RingBase,
        T: ?Sized + RingBase
{
    f: F,
    g: G,
    domain: PhantomData<R>,
    intermediate: PhantomData<S>,
    codomain: PhantomData<T>
}

impl<R, S, T, F, G> ComposedHom<R, S, T, F, G>
    where F: Clone + Homomorphism<R, S>,
        G: Clone + Homomorphism<S, T>,
        R: ?Sized + RingBase,
        S: ?Sized + RingBase,
        T: ?Sized + RingBase
{
    ///
    /// Returns a reference to `f`, the homomorphism that is applied first
    /// to input elements `x`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn first(&self) -> &F {
        &self.f
    }

    ///
    /// Returns a reference to `g`, the homomorphism that is applied second,
    /// so to `f(x)` when mapping an input element `x`.
    /// 
    #[stability::unstable(feature = "enable")]
    pub fn second(&self) -> &G {
        &self.g
    }
}

impl<R, S, T, F, G> Clone for ComposedHom<R, S, T, F, G>
    where F: Clone + Homomorphism<R, S>,
        G: Clone + Homomorphism<S, T>,
        R: ?Sized + RingBase,
        S: ?Sized + RingBase,
        T: ?Sized + RingBase
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            g: self.g.clone(),
            domain: PhantomData,
            codomain: PhantomData,
            intermediate: PhantomData
        }
    }
}

impl<R, S, T, F, G> Copy for ComposedHom<R, S, T, F, G>
    where F: Copy + Homomorphism<R, S>,
        G: Copy + Homomorphism<S, T>,
        R: ?Sized + RingBase,
        S: ?Sized + RingBase,
        T: ?Sized + RingBase
{}

impl<R, S, T, F, G> Homomorphism<R, T> for ComposedHom<R, S, T, F, G>
    where F: Homomorphism<R, S>,
        G: Homomorphism<S, T>,
        R: ?Sized + RingBase,
        S: ?Sized + RingBase,
        T: ?Sized + RingBase
{
    type DomainStore = <F as Homomorphism<R, S>>::DomainStore;
    type CodomainStore = <G as Homomorphism<S, T>>::CodomainStore;

    fn domain<'a>(&'a self) -> &'a Self::DomainStore {
        self.f.domain()
    }

    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore {
        self.g.codomain()
    }

    fn map(&self, x: <R as RingBase>::Element) -> <T as RingBase>::Element {
        self.g.map(self.f.map(x))
    }

    fn map_ref(&self, x: &<R as RingBase>::Element) -> <T as RingBase>::Element {
        self.g.map(self.f.map_ref(x))
    }

    fn mul_assign_map(&self, lhs: &mut <T as RingBase>::Element, rhs: <R as RingBase>::Element) {
        self.g.mul_assign_map_through_hom(lhs, rhs, &self.f)
    }

    fn mul_assign_ref_map(&self, lhs: &mut <T as RingBase>::Element, rhs: &<R as RingBase>::Element) {
        self.g.mul_assign_ref_map_through_hom(lhs, rhs, &self.f)
    }
}

///
/// Implements the trivial canonical isomorphism `Self: CanIsoFromTo<Self>` for the
/// given type. 
/// 
/// Note that this does not support generic types, as for those, it is
/// usually better to implement
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::delegate::*;
/// // define `RingConstructor<R: RingStore>`
/// # struct RingConstructor<R: RingStore>(R);
/// # impl<R: RingStore> DelegateRing for RingConstructor<R> {
/// #     type Element = El<R>;
/// #     type Base = R::Type;
/// #     fn get_delegate(&self) -> &Self::Base { self.0.get_ring() }
/// #     fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }
/// #     fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }
/// #     fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }
/// #     fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
/// # }
/// # impl<R: RingStore> PartialEq for RingConstructor<R> {
/// #     fn eq(&self, other: &Self) -> bool {
/// #         self.0.get_ring() == other.0.get_ring()
/// #     }
/// # }
/// impl<R, S> CanHomFrom<RingConstructor<S>> for RingConstructor<R>
///     where R: RingStore, S: RingStore, R::Type: CanHomFrom<S::Type>
/// {
///     type Homomorphism = <R::Type as CanHomFrom<S::Type>>::Homomorphism;
/// 
///     fn has_canonical_hom(&self, from: &RingConstructor<S>) -> Option<<R::Type as CanHomFrom<S::Type>>::Homomorphism> {
///         // delegate to base ring of type `R::Type`
/// #       self.get_delegate().has_canonical_hom(from.get_delegate())
///     }
/// 
///     fn map_in(&self, from: &RingConstructor<S>, el: <RingConstructor<S> as RingBase>::Element, hom: &Self::Homomorphism) -> <Self as RingBase>::Element {
///         // delegate to base ring of type `R::Type`
/// #       self.get_delegate().map_in(from.get_delegate(), el, hom)
///     }
/// }
/// 
/// // and same for CanIsoFromTo
/// ```
/// or something similar.
/// 
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// 
/// #[derive(PartialEq, Clone)]
/// struct MyI32Ring;
/// 
/// impl DelegateRing for MyI32Ring {
/// 
///     type Base = StaticRingBase<i32>;
///     type Element = i32;
/// 
///     fn get_delegate(&self) -> &Self::Base {
///         StaticRing::<i32>::RING.get_ring()
///     }
/// 
///     fn delegate_ref<'a>(&self, el: &'a i32) -> &'a i32 {
///         el
///     }
/// 
///     fn delegate_mut<'a>(&self, el: &'a mut i32) -> &'a mut i32 {
///         el
///     }
/// 
///     fn delegate(&self, el: i32) -> i32 {
///         el
///     }
/// 
///     fn postprocess_delegate_mut(&self, _: &mut i32) {
///         // sometimes it might be necessary to fix some data of `Self::Element`
///         // if the underlying `Self::Base::Element` was modified via `delegate_mut()`;
///         // this is not the case here, so leave empty
///     }
/// 
///     fn rev_delegate(&self, el: i32) -> i32 {
///         el
///     }
/// }
/// 
/// // since we provide `PartialEq`, the trait `CanIsoFromTo<Self>` is trivial
/// // to implement
/// impl_eq_based_self_iso!{ MyI32Ring }
/// 
/// let ring = RingValue::from(MyI32Ring);
/// assert_el_eq!(ring, ring.int_hom().map(1), ring.one());
/// ```
/// 
#[macro_export]
macro_rules! impl_eq_based_self_iso {
    ($type:ty) => {
        impl $crate::homomorphism::CanHomFrom<Self> for $type {

            type Homomorphism = ();

            fn has_canonical_hom(&self, from: &Self) -> Option<()> {
                if self == from {
                    Some(())
                } else {
                    None
                }
            }

            fn map_in(&self, _from: &Self, el: <Self as $crate::ring::RingBase>::Element, _: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                el
            }
        }
        
        impl $crate::homomorphism::CanIsoFromTo<Self> for $type {

            type Isomorphism = ();

            fn has_canonical_iso(&self, from: &Self) -> Option<()> {
                if self == from {
                    Some(())
                } else {
                    None
                }
            }

            fn map_out(&self, _from: &Self, el: <Self as $crate::ring::RingBase>::Element, _: &Self::Homomorphism) -> <Self as $crate::ring::RingBase>::Element {
                el
            }
        }
    };
}

#[cfg(any(test, feature = "generic_tests"))]
pub mod generic_tests {

    use super::*;

    pub fn test_homomorphism_axioms<R: ?Sized + RingBase, S: ?Sized + RingBase, H, I: Iterator<Item = R::Element>>(hom: H, edge_case_elements: I)
        where H: Homomorphism<R, S>
    {
        let from = hom.domain();
        let to = hom.codomain();
        let elements = edge_case_elements.collect::<Vec<_>>();

        assert!(to.is_zero(&hom.map(from.zero())));
        assert!(to.is_one(&hom.map(from.one())));

        for a in &elements {
            for b in &elements {
                {
                    let map_a = hom.map_ref(a);
                    let map_b = hom.map_ref(b);
                    let map_sum = to.add_ref(&map_a, &map_b);
                    let sum_map = hom.map(from.add_ref(a, b));
                    assert!(to.eq_el(&map_sum, &sum_map), "Additive homomorphic property failed: hom({} + {}) = {} != {} = {} + {}", from.format(a), from.format(b), to.format(&sum_map), to.format(&map_sum), to.format(&map_a), to.format(&map_b));
                }
                {
                    let map_a = hom.map_ref(a);
                    let map_b = hom.map_ref(b);
                    let map_prod = to.mul_ref(&map_a, &map_b);
                    let prod_map = hom.map(from.mul_ref(a, b));
                    assert!(to.eq_el(&map_prod, &prod_map), "Multiplicative homomorphic property failed: hom({} * {}) = {} != {} = {} * {}", from.format(a), from.format(b), to.format(&prod_map), to.format(&map_prod), to.format(&map_a), to.format(&map_b));
                }
                {
                    let map_a = hom.map_ref(a);
                    let prod_map = hom.map(from.mul_ref(a, b));
                    let mut mul_assign = to.clone_el(&map_a);
                    hom.mul_assign_ref_map( &mut mul_assign, b);
                    assert!(to.eq_el(&mul_assign, &prod_map), "mul_assign_ref_map() failed: hom({} * {}) = {} != {} = mul_map_in(hom({}), {})", from.format(a), from.format(b), to.format(&prod_map), to.format(&mul_assign), to.format(&map_a), from.format(b));
                }
            }
        }
    }
}