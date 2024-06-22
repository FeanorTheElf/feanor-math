use std::marker::PhantomData;

use crate::ring::*;
use crate::primitive_int::{StaticRingBase, StaticRing};

///
/// The user-facing trait for ring homomorphisms, i.e. maps `R -> S`
/// between rings that respect the ring structure.
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
    type DomainStore: RingStore<Type = Domain>;
    type CodomainStore: RingStore<Type = Codomain>;

    fn domain<'a>(&'a self) -> &'a Self::DomainStore;
    fn codomain<'a>(&'a self) -> &'a Self::CodomainStore;

    fn map(&self, x: Domain::Element) -> Codomain::Element;

    fn map_ref(&self, x: &Domain::Element) -> Codomain::Element {
        self.map(self.domain().clone_el(x))
    }

    fn mul_assign_map(&self, lhs: &mut Codomain::Element, rhs: Domain::Element) {
        self.codomain().mul_assign(lhs, self.map(rhs))
    }

    fn mul_assign_map_ref(&self, lhs: &mut Codomain::Element, rhs: &Domain::Element) {
        self.codomain().mul_assign(lhs, self.map_ref(rhs))
    }

    fn mul_map(&self, mut lhs: Codomain::Element, rhs: Domain::Element) -> Codomain::Element {
        self.mul_assign_map(&mut lhs, rhs);
        lhs
    }

    fn mul_ref_fst_map(&self, lhs: &Codomain::Element, rhs: Domain::Element) -> Codomain::Element {
        self.mul_map(self.codomain().clone_el(lhs), rhs)
    }

    fn mul_ref_snd_map(&self, mut lhs: Codomain::Element, rhs: &Domain::Element) -> Codomain::Element {
        self.mul_assign_map_ref(&mut lhs, rhs);
        lhs
    }

    fn mul_ref_map(&self, lhs: &Codomain::Element, rhs: &Domain::Element) -> Codomain::Element {
        self.mul_ref_snd_map(self.codomain().clone_el(lhs), rhs)
    }

    fn compose<F, PrevDomain: ?Sized + RingBase>(self, prev: F) -> ComposedHom<PrevDomain, Domain, Codomain, F, Self>
        where Self: Sized, F: Homomorphism<PrevDomain, Domain>
    {
        assert!(prev.codomain().get_ring() == self.domain().get_ring());
        ComposedHom { f: prev, g: self, domain: PhantomData, intermediate: PhantomData, codomain: PhantomData }
    }
}

///
/// Trait for rings R that have a canonical homomorphism `S -> R`.
/// A ring homomorphism is expected to be unital. 
/// 
/// This trait is
/// considered implementor-facing, so it is designed to easily implement
/// natural maps between rings. When using homomorphisms, consider
/// using instead [`CanHom`], as it does not require weird syntax like
/// `R.get_ring().map_in(S.get_ring(), x, &hom)`.
/// 
/// # Exact requirements
/// 
/// Which homomorphisms are considered canonical is up to implementors,
/// as long as any diagram of canonical homomorphisms commutes. In
/// other words, if there are rings `R, S` and "intermediate rings"
/// `R1, ..., Rn` resp. `R1', ..., Rm'` such that there are canonical
/// homomorphisms
/// ```text
/// S -> R1 -> R2 -> ... -> Rn -> R
/// ```
/// and
/// ```text
/// S -> R1' -> R2' -> ... -> Rm' -> R
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
/// ```
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
/// ```
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
/// ```
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
/// let Zn_rns = zn_rns::Zn::create_from_primes(ZZ_big, vec![17, 257]);
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
/// Additionally, there are the projections `Z -> Z/nZ`.
/// They are all implemented, even though [`crate::rings::zn::ZnRing`] currently
/// only requires the projection from the "associated" integer ring.
/// ```
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
/// let Zn_rns = zn_rns::Zn::create_from_primes(ZZ_big, vec![17, 257]);
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
/// ```
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
    type Homomorphism;

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
/// I am currently thinking about removing this trait entirely.
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
/// S -> R1 -> R2 <~> R3 <~> R4 -> ... -> Rn -> R
/// ```
/// and
/// ```text
/// S <~> R1' -> R2' -> ... -> Rm' -> R
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
    type Isomorphism;

    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism>;
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
/// ```
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

impl<R, S> CanHom<R, S>
    where R: RingStore, S: RingStore, S::Type: CanHomFrom<R::Type>
{
    pub fn new(from: R, to: S) -> Result<Self, (R, S)> {
        match to.get_ring().has_canonical_hom(from.get_ring()) {
            Some(data) => Ok(Self { from, to, data }),
            _ => Err((from, to))
        }
    }

    pub fn raw_hom(&self) -> &<S::Type as CanHomFrom<R::Type>>::Homomorphism {
        &self.data
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
}

///
/// A wrapper around [`CanIsoFromTo::Isomorphism`] that references the domain and
/// codomain rings, making it much easier to use. This also contains the homomorphism
/// in the other direction, i.e. allows mapping in both directions.
/// 
/// # Example
/// ```
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

    pub fn inv<'a>(&'a self) -> CanHom<&'a R, &'a S> {
        CanHom::new(&self.from, &self.to).unwrap_or_else(|_| unreachable!())
    }

    pub fn raw_iso(&self) -> &<S::Type as CanIsoFromTo<R::Type>>::Isomorphism {
        &self.data
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
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::poly::*;
/// let base = StaticRing::<i32>::RING;
/// let extension = dense_poly::DensePolyRing::new(base, "X");
/// let hom = extension.inclusion();
/// let f = extension.add(hom.map(8), extension.indeterminate());
/// assert_el_eq!(&extension, &extension.from_terms([(8, 0), (1, 1)].into_iter()), &f);
/// ```
/// 
#[derive(Copy, Clone)]
pub struct Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
    ring: R
}

impl<R> Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
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

    fn mul_assign_map_ref(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: &<<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) {
        self.ring.get_ring().mul_assign_base(lhs, rhs)
    }

    fn mul_assign_map(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: <<<R::Type as RingExtension>::BaseRing as RingStore>::Type as RingBase>::Element) {
        self.mul_assign_map_ref(lhs, &rhs)
    }
}

///
/// The ring homomorphism `Z -> R` that exists for any ring `R`.
/// 
/// # Example
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// let ring = zn_static::F17;
/// let hom = ring.int_hom();
/// assert_el_eq!(&ring, &hom.map(1), &hom.map(18));
/// ```
/// 
#[derive(Clone, Copy)]
pub struct IntHom<R>
    where R: RingStore
{
    ring: R
}

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

    fn mul_assign_map_ref(&self, lhs: &mut <R::Type as RingBase>::Element, rhs: &<StaticRingBase<i32> as RingBase>::Element) {
        self.mul_assign_map(lhs, *rhs)
    }
}

impl<R> IntHom<R>
    where R: RingStore
{
    pub fn new(ring: R) -> Self {
        Self { ring }
    }
}

#[derive(Copy, Clone)]
pub struct Identity<R: RingStore> {
    ring: R
}

impl<R: RingStore> Identity<R> {

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

    fn mul_assign_map_ref(&self, lhs: &mut <R as RingBase>::Element, rhs: &<S as RingBase>::Element) {
        (*self).mul_assign_map_ref(lhs, rhs)
    }
}

#[derive(Clone, Copy)]
pub struct LambdaHom<R: RingStore, S: RingStore, F>
    where F: Fn(&R, &S, &El<R>) -> El<S>
{
    from: R,
    to: S,
    f: F
}

impl<R: RingStore, S: RingStore, F> LambdaHom<R, S, F>
    where F: Fn(&R, &S, &El<R>) -> El<S>
{
    pub fn new(from: R, to: S, f: F) -> Self {
        Self { from, to, f }
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
        self.g.mul_assign_map(lhs, self.f.map(rhs))
    }

    fn mul_assign_map_ref(&self, lhs: &mut <T as RingBase>::Element, rhs: &<R as RingBase>::Element) {
        self.g.mul_assign_map(lhs, self.f.map_ref(rhs))
    }
}

///
/// Implements the trivial canonical isomorphism `Self: CanIsoFromTo<Self>` for the
/// given type. 
/// 
/// Note that this does not support generic types, as for those, it is
/// usually better to implement
/// ```rust,ignore
/// RingConstructor<R>: CanIsoFromTo<RingConstructor<S>>
///     where R: CanIsoFromTo<S>
/// ```
/// or something similar.
/// 
/// # Example
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::primitive_int::*;
/// # use feanor_math::delegate::*;
/// # use feanor_math::{assert_el_eq, impl_eq_based_self_iso};
/// 
/// #[derive(PartialEq, Clone, Copy)]
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
/// assert_el_eq!(&ring, &ring.int_hom().map(1), &ring.one());
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
                    hom.mul_assign_map_ref( &mut mul_assign, b);
                    assert!(to.eq_el(&mul_assign, &prod_map), "mul_assign_map_ref() failed: hom({} * {}) = {} != {} = mul_map_in(hom({}), {})", from.format(a), from.format(b), to.format(&prod_map), to.format(&mul_assign), to.format(&map_a), from.format(b));
                }
            }
        }
    }
}