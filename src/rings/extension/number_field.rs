
use crate::integer::*;
use crate::ring::*;
use crate::rings::rational::RationalFieldBase;
use super::FreeAlgebra;

pub trait FiniteExtensionOfQQ: RingBase {

    type IntegerRingBase: ?Sized + IntegerRing;
    type Integers: RingStore<Type = Self::IntegerRingBase>;

    fn integer_ring(&self) -> &Self::Integers;
    fn from_integer(&self, x: El<Self::Integers>) -> Self::Element;
}

impl<I> FiniteExtensionOfQQ for RationalFieldBase<I>
    where I: RingStore,
        I::Type: IntegerRing
{
    type IntegerRingBase = I::Type;
    type Integers = I;

    fn integer_ring(&self) -> &Self::Integers {
        self.base_ring()
    }
    
    fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
        self.from(x)
    }
}

impl<R> FiniteExtensionOfQQ for R
    where R: FreeAlgebra,
        <R::BaseRing as RingStore>::Type: FiniteExtensionOfQQ
{
    type IntegerRingBase = <<R::BaseRing as RingStore>::Type as FiniteExtensionOfQQ>::IntegerRingBase;
    type Integers = <<R::BaseRing as RingStore>::Type as FiniteExtensionOfQQ>::Integers;

    fn integer_ring(&self) -> &Self::Integers {
        self.base_ring().get_ring().integer_ring()
    }
    
    fn from_integer(&self, x: El<Self::Integers>) -> Self::Element {
        self.from(self.base_ring().get_ring().from_integer(x))
    }
}

pub struct NumberField<Impl>
    where Impl: RingStore,
        Impl::Type: FiniteExtensionOfQQ
{
    base: Impl
}

