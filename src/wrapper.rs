use std::{ops::*, hash::Hash, fmt::{Display, Debug}};

use crate::homomorphism::*;
use crate::ring::*;

///
/// Stores a ring element together with its ring, so that ring operations do
/// not require explicit mention of the ring object. This can be used both for
/// convenience of notation (i.e. use `a + b` instead of `ring.add(a, b)`) and
/// might also be necessary when e.g. storing elements in a set.
/// 
/// # Examples
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::primitive_int::*;
/// let ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let x = RingElementWrapper::new(&ring, ring.indeterminate());
/// println!("The result is: {}", x.clone() + x.clone() * x);
/// // instead of
/// let x = ring.indeterminate();
/// println!("The result is: {}", ring.format(&ring.add(ring.mul(ring.clone_el(&x), ring.clone_el(&x)), ring.clone_el(&x))));
/// ```
/// You can also retrieve the wrapped element
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::poly::*;
/// # use feanor_math::rings::poly::dense_poly::*;
/// # use feanor_math::wrapper::*;
/// # use feanor_math::primitive_int::*;
/// let ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
/// let x = RingElementWrapper::new(&ring, ring.indeterminate());
/// assert_el_eq!(&ring, ring.add(ring.mul(ring.clone_el(&x), ring.clone_el(&x)), ring.clone_el(&x)), (x.clone() + x.clone() * x).unwrap());
/// ```
/// 
pub struct RingElementWrapper<R>
    where R: RingStore
{
    ring: R,
    element: El<R>
}

impl<R: RingStore> RingElementWrapper<R> {

    pub const fn new(ring: R, element: El<R>) -> Self {
        Self { ring, element }
    }

    pub fn pow(self, power: usize) -> Self {
        Self {
            element: self.ring.pow(self.element, power),
            ring: self.ring
        }
    }

    pub fn unwrap(self) -> El<R> {
        self.element
    }
}

macro_rules! impl_xassign_trait {
    ($trait_name:ident, $fn_name:ident, $fn_ref_name:ident) => {
        
        impl<R: RingStore> $trait_name for RingElementWrapper<R> {

            fn $fn_name(&mut self, rhs: Self) {
                debug_assert!(self.ring.get_ring() == rhs.ring.get_ring());
                self.ring.$fn_name(&mut self.element, rhs.element);
            }
        }

        impl<'a, R: RingStore> $trait_name<&'a Self> for RingElementWrapper<R> {

            fn $fn_name(&mut self, rhs: &'a Self) {
                debug_assert!(self.ring.get_ring() == rhs.ring.get_ring());
                self.ring.$fn_ref_name(&mut self.element, &rhs.element);
            }
        }
    };
}

macro_rules! impl_trait {
    ($trait_name:ident, $fn_name:ident) => {
        
        impl<R: RingStore> $trait_name for RingElementWrapper<R> {
            type Output = Self;

            fn $fn_name(self, rhs: Self) -> Self::Output {
                debug_assert!(self.ring.get_ring() == rhs.ring.get_ring());
                Self { ring: self.ring, element: rhs.ring.$fn_name(self.element, rhs.element) }
            }
        }
    };
}

impl_xassign_trait!{ AddAssign, add_assign, add_assign_ref }
impl_xassign_trait!{ MulAssign, mul_assign, mul_assign_ref }
impl_xassign_trait!{ SubAssign, sub_assign, sub_assign_ref }
impl_trait!{ Add, add }
impl_trait!{ Mul, mul }
impl_trait!{ Sub, sub }

macro_rules! impl_xassign_trait_int {
    ($trait_name:ident, $fn_name:ident) => {
        
        impl<R: RingStore> $trait_name<i32> for RingElementWrapper<R> {

            fn $fn_name(&mut self, rhs: i32) {
                self.ring.$fn_name(&mut self.element, self.ring.int_hom().map(rhs));
            }
        }
    };
}

macro_rules! impl_trait_int {
    ($trait_name:ident, $fn_name:ident) => {
        
        impl<R: RingStore> $trait_name<i32> for RingElementWrapper<R> {
            type Output = Self;

            fn $fn_name(self, rhs: i32) -> Self::Output {
                RingElementWrapper { element: self.ring.$fn_name(self.element, self.ring.int_hom().map(rhs)), ring: self.ring }
            }
        }

        impl<R: RingStore> $trait_name<RingElementWrapper<R>> for i32 {
            type Output = RingElementWrapper<R>;

            fn $fn_name(self, rhs: RingElementWrapper<R>) -> Self::Output {
                RingElementWrapper { element: rhs.ring.$fn_name(rhs.element, rhs.ring.int_hom().map(self)), ring: rhs.ring }
            }
        }
    };
}

impl_xassign_trait_int!{ AddAssign, add_assign }
impl_xassign_trait_int!{ MulAssign, mul_assign }
impl_xassign_trait_int!{ SubAssign, sub_assign }
impl_trait_int!{ Add, add }
impl_trait_int!{ Mul, mul }
impl_trait_int!{ Sub, sub }

#[cfg(feature = "elementwrapper-caret-pow")]
impl<R: RingStore> BitXor<usize> for RingElementWrapper<R> {
    type Output = RingElementWrapper<R>;

    fn bitxor(self, rhs: usize) -> Self::Output {
        self.pow(rhs)
    }
}

#[cfg(feature = "elementwrapper-caret-pow")]
impl<'a, R: RingStore + Clone> BitXor<usize> for &'a RingElementWrapper<R> {
    type Output = RingElementWrapper<R>;

    fn bitxor(self, rhs: usize) -> Self::Output {
        <RingElementWrapper<_> as Clone>::clone(self).pow(rhs)
    }
}

impl<R: RingStore + Copy> Copy for RingElementWrapper<R> 
    where El<R>: Copy
{}

impl<R: RingStore + Clone> Clone for RingElementWrapper<R> {

    fn clone(&self) -> Self {
        Self { ring: self.ring.clone(), element: self.ring.clone_el(&self.element) }
    }
}

impl<R: RingStore> PartialEq for RingElementWrapper<R> {

    fn eq(&self, other: &Self) -> bool {
        debug_assert!(self.ring.get_ring() == other.ring.get_ring());
        self.ring.eq_el(&self.element, &other.element)
    }
}

impl<R: RingStore> Eq for RingElementWrapper<R> {}

impl<R: RingStore + HashableElRingStore> Hash for RingElementWrapper<R> 
    where R::Type: HashableElRing
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ring.hash(&self.element, state)
    }
}

impl<R: RingStore> Display for RingElementWrapper<R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.get_ring().dbg(&self.element, f)
    }
}

impl<R: RingStore> Debug for RingElementWrapper<R> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ring.get_ring().dbg(&self.element, f)
    }
}

impl<R: RingStore> Deref for RingElementWrapper<R> {
    type Target = El<R>;

    fn deref(&self) -> &Self::Target {
        &self.element
    }
}

#[cfg(feature = "elementwrapper-caret-pow")]
#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::finite::FiniteRingStore;
#[cfg(test)]
use crate::rings::zn::zn_64;

#[cfg(feature = "elementwrapper-caret-pow")]
#[test]
fn test_pow_caret() {
    let ring = StaticRing::<i64>::RING;
    let a = RingElementWrapper::new(&ring, 3);
    assert_eq!(a * a * a * a, a^4);
}

#[test]
fn test_arithmetic_expression() {
    let ring = zn_64::Zn::new(17);

    for x in ring.elements() {
        for y in ring.elements() {
            for z in ring.elements() {
                let expected = ring.add(ring.mul(x, y), ring.mul(ring.add(x, z), ring.sub(y, z)));
                let x = RingElementWrapper::new(&ring, x);
                let y = RingElementWrapper::new(&ring, y);
                let z = RingElementWrapper::new(&ring, z);
                assert_el_eq!(ring, expected, (x * y + (x + z) * (y - z)).unwrap());
            }
        }
    }
}

#[test]
fn test_arithmetic_expression_int() {
    let ring = zn_64::Zn::new(17);

    for x in ring.elements() {
        for y in ring.elements() {
            for z in ring.elements() {
                let expected = ring.add(ring.add(ring.int_hom().mul_map(ring.mul(x, y), 8), ring.mul(ring.add(ring.add(ring.one(), x), ring.int_hom().mul_map(z, 2)), ring.sub(y, ring.int_hom().mul_map(z, 2)))), ring.int_hom().map(5));
                let x = RingElementWrapper::new(&ring, x);
                let y = RingElementWrapper::new(&ring, y);
                let z = RingElementWrapper::new(&ring, z);
                assert_el_eq!(ring, expected, (x * 8 * y + (1 + x + 2 * z) * (y - z * 2) + 5).unwrap());
            }
        }
    }
}