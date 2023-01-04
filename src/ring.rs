use std::rc::Rc;

///
/// Basic trait for objects that have a ring structure.
/// 
/// Implementors of this trait should provide the basic ring operations,
/// and additionally operators for displaying and equality testing. If
/// a performance advantage can be achieved by accepting some arguments by
/// reference instead of by value, the default-implemented functions for
/// ring operations on references should be overwritten.
/// 
/// Note that usually, this trait will not be used directly, but always
/// through a [`crate::ring::RingWrapper`]. In more detail, while this trait
/// defines the functionality, [`crate::ring::RingWrapper`] allows abstracting
/// the storage - everything that allows access to a ring then is a 
/// [`crate::ring::RingWrapper`]. For example, references or shared pointers
/// to rings. If you want to use rings directly by value, some technical
/// details make it necessary to use the no-op container [`crate::ring::RingValue`].
/// 
pub trait RingBase {

    type Element: Clone;

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.add_assign(lhs, rhs.clone()) }
    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element);
    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.sub_assign(lhs, rhs.clone()) }
    fn negate_inplace(&self, lhs: &mut Self::Element);
    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element);
    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.mul_assign(lhs, rhs.clone()) }
    fn zero(&self) -> Self::Element { self.from_z(0) }
    fn one(&self) -> Self::Element { self.from_z(1) }
    fn neg_one(&self) -> Self::Element { self.from_z(-1) }
    fn from_z(&self, value: i32) -> Self::Element;
    fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool;
    fn is_zero(&self, value: &Self::Element) -> bool { self.eq(value, &self.zero()) }
    fn is_one(&self, value: &Self::Element) -> bool { self.eq(value, &self.one()) }
    fn is_neg_one(&self, value: &Self::Element) -> bool { self.eq(value, &self.neg_one()) }
    fn is_commutative(&self) -> bool;
    fn is_noetherian(&self) -> bool;

    fn negate(&self, mut value: Self::Element) -> Self::Element {
        self.negate_inplace(&mut value);
        return value;
    }
    
    fn sub_assign(&self, lhs: &mut Self::Element, mut rhs: Self::Element) {
        self.negate_inplace(&mut rhs);
        self.add_assign(lhs, rhs);
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = lhs.clone();
        self.add_assign_ref(&mut result, rhs);
        return result;
    }

    fn add_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        self.add_assign_ref(&mut rhs, lhs);
        return rhs;
    }

    fn add_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.add_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn add(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.add_assign(&mut lhs, rhs);
        return lhs;
    }

    fn sub_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = lhs.clone();
        self.sub_assign_ref(&mut result, rhs);
        return result;
    }

    fn sub_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        self.sub_assign_ref(&mut rhs, lhs);
        self.negate_inplace(&mut rhs);
        return rhs;
    }

    fn sub_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.sub_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn sub(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.sub_assign(&mut lhs, rhs);
        return lhs;
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = lhs.clone();
        self.mul_assign_ref(&mut result, rhs);
        return result;
    }

    fn mul_ref_fst(&self, lhs: &Self::Element, mut rhs: Self::Element) -> Self::Element {
        if self.is_commutative() {
            self.mul_assign_ref(&mut rhs, lhs);
            return rhs;
        } else {
            let mut result = lhs.clone();
            self.mul_assign(&mut result, rhs);
            return result;
        }
    }

    fn mul_ref_snd(&self, mut lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.mul_assign_ref(&mut lhs, rhs);
        return lhs;
    }

    fn mul(&self, mut lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.mul_assign(&mut lhs, rhs);
        return lhs;
    }
}

macro_rules! delegate {
    (fn $name:ident (&self, $($pname:ident: $ptype:ty),*) -> $rtype:ty) => {
        fn $name (&self, $($pname: $ptype),*) -> $rtype {
            self.get_ring().$name($($pname),*)
        }
    };
    (fn $name:ident (&self) -> $rtype:ty) => {
        fn $name (&self) -> $rtype {
            self.get_ring().$name()
        }
    };
}

///
/// Basic trait for objects that store (in some sense) a ring. This can
/// be a ring-by-value, a reference to a ring, or a box to a ring. Note
/// that this trait is also designed to allow chaining, with the exception
/// of [`crate::ring::RingValue`].
/// 
/// As opposed to [`crate::ring::RingBase`], which is responsible for the
/// functionality and ring operations, this trait is solely responsible for
/// the storage. Note however, that storage can be quite difficult once we
/// build rings onto other rings and so on.
/// 
pub trait RingWrapper {
    
    type Type: RingBase;

    fn get_ring<'a>(&'a self) -> &'a Self::Type;
    
    fn map_in<S>(&self, from: &S, el: El<S>) -> El<Self>
        where S: RingWrapper, Self::Type: CanonicalHom<S::Type> 
    {
        self.get_ring().map_in(from.get_ring(), el)
    }

    fn map_out<S>(&self, to: &S, el: El<Self>) -> El<S>
        where S: RingWrapper, Self::Type: CanonicalIso<S::Type> 
    {
        self.get_ring().map_out(to.get_ring(), el)
    }

    fn sum<I>(&self, els: I) -> El<Self> 
        where I: Iterator<Item = El<Self>>
    {
        els.fold(self.zero(), |a, b| self.add(a, b))
    }

    delegate!{ fn add_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn add_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn sub_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn negate_inplace(&self, lhs: &mut El<Self>) -> () }
    delegate!{ fn mul_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn mul_assign_ref(&self, lhs: &mut El<Self>, rhs: &El<Self>) -> () }
    delegate!{ fn zero(&self) -> El<Self> }
    delegate!{ fn one(&self) -> El<Self> }
    delegate!{ fn neg_one(&self) -> El<Self> }
    delegate!{ fn from_z(&self, value: i32) -> El<Self> }
    delegate!{ fn eq(&self, lhs: &El<Self>, rhs: &El<Self>) -> bool }
    delegate!{ fn is_zero(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_one(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_neg_one(&self, value: &El<Self>) -> bool }
    delegate!{ fn is_commutative(&self) -> bool }
    delegate!{ fn is_noetherian(&self) -> bool }
    delegate!{ fn negate(&self, value: El<Self>) -> El<Self> }
    delegate!{ fn sub_assign(&self, lhs: &mut El<Self>, rhs: El<Self>) -> () }
    delegate!{ fn add_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn add_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn add_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn add(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn sub_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn sub_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn sub_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn sub(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn mul_ref(&self, lhs: &El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn mul_ref_fst(&self, lhs: &El<Self>, rhs: El<Self>) -> El<Self> }
    delegate!{ fn mul_ref_snd(&self, lhs: El<Self>, rhs: &El<Self>) -> El<Self> }
    delegate!{ fn mul(&self, lhs: El<Self>, rhs: El<Self>) -> El<Self> }

    fn base_ring<'a>(&'a self) -> &'a <Self::Type as RingExtension>::BaseRing
        where Self::Type: RingExtension
    {
        self.get_ring().base_ring()
    }

    fn from(&self, x: El<<Self::Type as RingExtension>::BaseRing>) -> El<Self>
        where Self::Type: RingExtension
    {
        self.get_ring().from(x)
    }

    fn from_ref(&self, x: &El<<Self::Type as RingExtension>::BaseRing>) -> El<Self>
        where Self::Type: RingExtension
    {
        self.get_ring().from_ref(x)
    }
}

pub trait CanonicalHom<S> : RingBase
    where S: RingBase
{
    fn has_canonical_hom(&self, from: &S) -> bool;
    fn map_in(&self, from: &S, el: S::Element) -> Self::Element;
}

pub trait CanonicalIso<S> : CanonicalHom<S>
    where S: RingBase
{
    fn has_canonical_iso(&self, from: &S) -> bool;
    fn map_out(&self, from: &S, el: Self::Element) -> S::Element;
}

pub trait RingExtension: RingBase {
    type BaseRing: RingWrapper;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing;
    fn from(&self, x: El<Self::BaseRing>) -> Self::Element;
    
    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        self.from(x.clone())
    }
}

pub type El<R> = <<R as RingWrapper>::Type as RingBase>::Element;

///
/// The most fundamental [`crate::ring::RingWrapper`]. It is basically
/// a no-op container, i.e. stores a [`crate::ring::RingBase`] object
/// by value, and allows accessing it.
/// 
/// # Why is this necessary?
/// 
/// In fact, that we need this trait is just the result of a technical
/// detail. We cannot implement
/// ```ignore
/// impl<R: RingBase> RingWrapper for R {}
/// impl<'a, R: RingWrapper> RingWrapper for &;a R {}
/// ```
/// since this might cause conflicting implementations.
/// Instead, we implement
/// ```ignore
/// impl<R: RingBase> RingWrapper for RingValue<R> {}
/// impl<'a, R: RingWrapper> RingWrapper for &;a R {}
/// ```
/// This causes some inconvenience, as now we cannot chain
/// [`crate::ring::RingWrapper`] in the case of [`crate::ring::RingValue`].
/// Furthermore, this trait will be necessary everywhere - 
/// to define a reference to a ring of type `A`, we now have to
/// write `&RingValue<A>`.
/// 
/// To simplify this, we propose to use the following simple pattern:
/// Create your ring type as
/// ```ignore
/// struct ABase { ... }
/// impl RingBase for ABase { ... } 
/// ```
/// and then provide a type alias
/// ```ignore
/// type A = RingValue<ABase>;
/// ```
/// 
#[derive(Copy, Clone)]
pub struct RingValue<R: RingBase> {
    ring: R
}

impl<R: RingBase> RingValue<R> {
    pub const fn new(value: R) -> Self {
        RingValue { ring: value }
    }
}

impl<R: RingBase> RingWrapper for RingValue<R> {

    type Type = R;
    
    fn get_ring(&self) -> &R {
        &self.ring
    }
}

impl<'a, R: RingWrapper> RingWrapper for &'a R {
    
    type Type = <R as RingWrapper>::Type;
    
    fn get_ring(&self) -> &Self::Type {
        (**self).get_ring()
    }
}

impl<'a, R: RingWrapper> RingWrapper for &'a mut R {
    
    type Type = <R as RingWrapper>::Type;
    
    fn get_ring(&self) -> &Self::Type {
        (**self).get_ring()
    }
}

impl<'a, R: RingWrapper> RingWrapper for Box<R> {
    
    type Type = <R as RingWrapper>::Type;
    
    fn get_ring(&self) -> &Self::Type {
        (**self).get_ring()
    }
}

impl<'a, R: RingWrapper> RingWrapper for Rc<R> {
    
    type Type = <R as RingWrapper>::Type;
    
    fn get_ring(&self) -> &Self::Type {
        (**self).get_ring()
    }
}

impl<'a, R: RingWrapper> RingWrapper for std::sync::Arc<R> {
    
    type Type = <R as RingWrapper>::Type;
    
    fn get_ring(&self) -> &Self::Type {
        (**self).get_ring()
    }
}

#[test]
fn test_internal_wrappings_dont_matter() {
    
    #[derive(Copy, Clone)]
    pub struct ABase;

    #[allow(unused)]
    #[derive(Copy, Clone)]
    pub struct BBase<R: RingWrapper> {
        base: R
    }

    impl RingBase for ABase {
        type Element = i32;

        fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs += rhs;
        }

        fn negate_inplace(&self, lhs: &mut Self::Element) {
            *lhs = -*lhs;
        }

        fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
            *lhs == *rhs
        }

        fn is_commutative(&self) -> bool {
            true
        }

        fn is_noetherian(&self) -> bool {
            true
        }

        fn from_z(&self, value: i32) -> Self::Element {
            value
        }

        fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs *= rhs;
        }
    }

    impl<R: RingWrapper> RingBase for BBase<R> {
        type Element = i32;

        fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs += rhs;
        }
        fn negate_inplace(&self, lhs: &mut Self::Element) {
            *lhs = -*lhs;
        }

        fn eq(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
            *lhs == *rhs
        }

        fn is_commutative(&self) -> bool {
            true
        }

        fn is_noetherian(&self) -> bool {
            true
        }

        fn from_z(&self, value: i32) -> Self::Element {
            value
        }

        fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
            *lhs *= rhs;
        }
    }

    impl<R: RingWrapper> CanonicalHom<ABase> for BBase<R> {

        fn has_canonical_hom(&self, _: &ABase) -> bool {
            true
        }

        fn map_in(&self, _: &ABase, el: <ABase as RingBase>::Element) -> Self::Element {
            el
        }
    }

    impl<R: RingWrapper, S: RingWrapper> CanonicalHom<BBase<S>> for BBase<R> 
        where R::Type: CanonicalHom<S::Type>
    {
        fn has_canonical_hom(&self, _: &BBase<S>) -> bool {
            true
        }

        fn map_in(&self, _: &BBase<S>, el: <BBase<S> as RingBase>::Element) -> Self::Element {
            el
        }
    }

    type A = RingValue<ABase>;
    type B<R> = RingValue<BBase<R>>;

    let a: A = RingValue { ring: ABase };
    let b1: B<A> = RingValue { ring: BBase { base: a } };
    let b2: B<&B<A>> = RingValue { ring: BBase { base: &b1 } };
    let b3: B<&A> = RingValue { ring: BBase { base: &a } };
    b1.map_in(&a, 0);
    b2.map_in(&a, 0);
    b2.map_in(&b1, 0);
    b2.map_in(&b3, 0);
    (&b2).map_in(&b3, 0);
    (&b2).map_in(&&&b3, 0);
}
