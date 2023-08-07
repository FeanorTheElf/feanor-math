use std::{ops::*, hash::Hash, fmt::{Display, Debug}};

use crate::ring::*;

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