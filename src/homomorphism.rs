use crate::{ring::*, primitive_int::{StaticRingBase, StaticRing}};

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
}

///
/// Trait for rings R that have a canonical homomorphism `S -> R`.
/// A ring homomorphism is expected to be unital.
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
/// implementing [`CanonicalIso`].
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
/// // on RingBase level
/// let hom = R.get_ring().has_canonical_hom(S.get_ring()).unwrap();
/// assert_eq!(8, R.get_ring().map_in(S.get_ring(), S.int_hom().map(8), &hom));
/// // on RingStore level
/// assert_eq!(8, R.coerce(&S, S.int_hom().map(8)));
/// ```
/// 
/// # Limitations
/// 
/// The rust constraints regarding conflicting impl make it, in some cases,
/// impossible to implement all the canonical homomorphisms that we would like.
/// This is true in particular, if the rings are highly generic, and build
/// on base rings. In this case, it should always be preferred to implement
/// `CanonicalIso` for rings that are "the same", and on the other hand not
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
/// Basically, all given integer rings have canonical isomorphisms between each other.
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
/// 
/// // there are also some blanket implementations/trait bounds
/// fn from_i32<I: IntegerRingStore>(to: &I) where I::Type: IntegerRing {
///     to.can_hom(&StaticRing::<i32>::RING);
/// }
/// 
/// fn to_i32<I: IntegerRingStore>(from: &I) where I::Type: IntegerRing {
///     StaticRing::<i32>::RING.can_hom(from);
/// }
/// ```
/// Notably, the only blanket implementations are currently
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
/// # use feanor_math::rings::zn::zn_42;
/// # use feanor_math::rings::zn::zn_barett;
/// # use feanor_math::rings::zn::zn_rns;
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = BigIntRing::RING;
/// 
/// let Zn_barett_i128 = zn_barett::Zn::new(ZZ, 17 * 257);
/// let Zn_barett_big = zn_barett::Zn::new(ZZ_big, ZZ_big.int_hom().map(17 * 257));
/// let Zn_std = zn_42::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::from_primes(ZZ_big, vec![17, 257]);
/// 
/// assert!(Zn_barett_i128.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_barett_i128.can_iso(&Zn_barett_big).is_some());
/// 
/// assert!(Zn_barett_big.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_barett_big.can_iso(&Zn_barett_big).is_some());
/// 
/// assert!(Zn_std.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_std.can_iso(&Zn_std).is_some());
/// 
/// assert!(Zn_rns.can_iso(&Zn_barett_i128).is_some());
/// assert!(Zn_rns.can_iso(&Zn_barett_big).is_some());
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
/// 
/// let ZZ = StaticRing::<i128>::RING;
/// let ZZ_big = BigIntRing::RING;
/// 
/// let Zn_barett_i128 = zn_barett::Zn::new(ZZ, 17 * 257);
/// let Zn_barett_big = zn_barett::Zn::new(ZZ_big, ZZ_big.int_hom().map(17 * 257));
/// let Zn_std = zn_42::Zn::new(17 * 257);
/// let Zn_rns = zn_rns::Zn::from_primes(ZZ_big, vec![17, 257]);
/// 
/// assert!(Zn_barett_i128.can_hom(&ZZ).is_some());
/// assert!(Zn_barett_i128.can_hom(&ZZ_big).is_some());
/// 
/// assert!(Zn_barett_big.can_hom(&ZZ).is_some());
/// assert!(Zn_barett_big.can_hom(&ZZ_big).is_some());
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
///  for any implementation `R: CanonicalIso<S>` there is also an
/// implementation `S: CanonicalIso<R>`. However, because of the trait
/// impl constraints of Rust, this is unpracticable and so we only
/// require the implementation `R: CanHomFrom<S>`.
/// 
pub trait CanonicalIso<S>: CanHomFrom<S>
    where S: RingBase + ?Sized
{
    type Isomorphism;

    fn has_canonical_iso(&self, from: &S) -> Option<Self::Isomorphism>;
    fn map_out(&self, from: &S, el: Self::Element, iso: &Self::Isomorphism) -> S::Element;
}

///
/// Basically an alias for `CanonicalIso<Self>`, but implemented as new
/// trait since trait aliases are not available.
/// 
pub trait SelfIso: CanonicalIso<Self> {}

impl<R: ?Sized + CanonicalIso<R>> SelfIso for R {}

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
/// 
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
/// [`CanonicalIso`] and the very simplified functions [`RingStore::coerce`], [`RingStore::coerce_ref`]
/// and [`RingStore::cast`].
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
/// A wrapper around [`CanonicalIso::Isomorphism`] that references the domain and
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
/// [`CanonicalIso`] and the very simplified functions [`RingStore::coerce`], [`RingStore::coerce_ref`]
/// and [`RingStore::cast`].
/// 
pub struct CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalIso<R::Type>
{
    from: R,
    to: S,
    data: <S::Type as CanonicalIso<R::Type>>::Isomorphism
}

impl<R, S> CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalIso<R::Type>
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

    pub fn raw_iso(&self) -> &<S::Type as CanonicalIso<R::Type>>::Isomorphism {
        &self.data
    }
}

impl<R, S> Homomorphism<S::Type,R::Type> for CanIso<R, S>
    where R: RingStore, S: RingStore, S::Type: CanonicalIso<R::Type>
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

pub struct Inclusion<R>
    where R: RingStore, R::Type: RingExtension
{
    ring: R
}

impl<R>  Inclusion<R>
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
}

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
}

impl<R> IntHom<R>
    where R: RingStore
{
    pub fn new(ring: R) -> Self {
        Self { ring }
    }
}

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

    
}